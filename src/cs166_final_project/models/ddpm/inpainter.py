import torch
import torch.nn.functional as F

from cs166_final_project.models.base import BaseInpaintingModel
from cs166_final_project.models.ddpm.ddpm import DDPMUnet


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return values.gather(0, timesteps).view(-1, 1, 1, 1).to(device=x.device, dtype=x.dtype)


class DDPMInpainter(BaseInpaintingModel):
    def __init__(
        self,
        image_channels: int,
        hidden_dim: int = 64,
        num_timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()

        self.image_channels = image_channels
        self.num_timesteps = num_timesteps
        self.net = DDPMUnet(
            in_channels=image_channels * 2 + 1,
            out_channels=image_channels,
            hidden_dim=hidden_dim,
        )

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - alpha_bars_prev)
        posterior_variance = posterior_variance / (1.0 - alpha_bars).clamp_min(1e-8)
        posterior_variance[0] = 0.0

        self.register_buffer("posterior_variance", posterior_variance)

    def _model_input(
        self,
        noisy_images: torch.Tensor,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        known_images = images * masks
        return torch.cat([noisy_images, known_images, masks], dim=1)

    def _predict_noise(
        self,
        noisy_images: torch.Tensor,
        images: torch.Tensor,
        masks: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(self._model_input(noisy_images, images, masks), timesteps)

    def compute_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=images.device,
        )
        noise = torch.randn_like(images)

        sqrt_alpha_bar = _extract(self.sqrt_alpha_bars, timesteps, images)
        sqrt_one_minus_alpha_bar = _extract(
            self.sqrt_one_minus_alpha_bars, timesteps, images
        )

        # diffusion training objective
        #          sqrt(bar{alpha}) * x + sqrt(1-bar{alpha}) * epsilon
        noisy_images = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * noise
        # for inptainting
        noisy_images = noisy_images * (1.0 - masks) + images * masks
        pred_noise = self._predict_noise(noisy_images, images, masks, timesteps)

        missing = 1.0 - masks

        #### only missing regions contributre to loss
        loss = F.mse_loss(pred_noise * missing, noise * missing, reduction="sum")

        return loss / (missing.sum() * images.shape[1]).clamp_min(1.0)

    @torch.no_grad()
    def inpaint(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = torch.randn_like(images)
        x = x * (1.0 - masks) + images * masks

        for timestep in reversed(range(self.num_timesteps)):
            timesteps = torch.full(
                (images.shape[0],),
                timestep,
                device=images.device,
                dtype=torch.long,
            )
            pred_noise = self._predict_noise(x, images, masks, timesteps)

            beta_t = _extract(self.betas, timesteps, x)
            sqrt_one_minus_alpha_bar_t = _extract(
                self.sqrt_one_minus_alpha_bars, timesteps, x
            )
            sqrt_recip_alpha_t = _extract(self.sqrt_recip_alphas, timesteps, x)

            model_mean = sqrt_recip_alpha_t * (
                x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t.clamp_min(1e-8)
            )

            if timestep > 0:
                # use posterior variance
                variance_t = _extract(self.posterior_variance, timesteps, x)
                x = model_mean + torch.sqrt(variance_t.clamp_min(1e-20)) * torch.randn_like(x)
            else:
                x = model_mean

            x = x * (1.0 - masks) + images * masks

        return x.clamp(-1.0, 1.0)

