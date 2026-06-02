import torch
import torch.nn.functional as F

from cs166_final_project.models.base import BaseInpaintingModel
from cs166_final_project.models.flow.flow import FlowUnet


class FlowInpainter(BaseInpaintingModel):
    def __init__(
        self,
        image_channels: int,
        hidden_dim: int = 64,
        num_steps: int = 25,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.num_steps = num_steps
        self.net = FlowUnet(
            in_channels=image_channels * 2 + 1,
            out_channels=image_channels,
            hidden_dim=hidden_dim,
        )

    def _model_input(
        self,
        x_t: torch.Tensor,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        known_images = images * masks
        return torch.cat([x_t, known_images, masks], dim=1)

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        images: torch.Tensor,
        masks: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(self._model_input(x_t, images, masks), times)

    def compute_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        noise = torch.randn_like(images)
        times = torch.rand(batch_size, device=images.device)

        t = times[:, None, None, None]
        x_t = (1.0 - t) * noise + t * images       # forward: the linear inteporlation
        x_t = x_t * (1.0 - masks) + images * masks # inpainting
        target_velocity = images - noise

        pred_velocity = self._predict_velocity(x_t, images, masks, times)

        ### only compute loss on onccluded region
        missing = 1.0 - masks
        loss = F.mse_loss(
            pred_velocity * missing,
            target_velocity * missing,
            reduction="sum",
        )

        return loss / (missing.sum() * images.shape[1]).clamp_min(1.0)

    @torch.no_grad()
    def inpaint(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = torch.randn_like(images)
        x = x * (1.0 - masks) + images * masks
        dt = 1.0 / self.num_steps

        for step in range(self.num_steps):
            times = torch.full(
                (images.shape[0],),
                step / self.num_steps,
                device=images.device,
            )

            velocity = self._predict_velocity(x, images, masks, times)

            # simple numerical integration
            # euler
            x = x + dt * velocity
            x = x * (1.0 - masks) + images * masks

        return x.clamp(-1.0, 1.0)

