import torch
import torch.nn.functional as F

from cs166_final_project.models.base import BaseInpaintingModel
from cs166_final_project.models.unet.unet import Unet


class UnetInpainter(BaseInpaintingModel):
    def __init__(self, image_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.net = Unet(
            in_channels=image_channels + 1, # [x*mask, mask]
            out_channels=image_channels,    # [pred]
            hidden_dim=hidden_dim,
        )

    def _predict(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masked_images = images * masks
        return self.net(torch.cat([masked_images, masks], dim=1))

    def compute_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        pred = self._predict(images, masks)
        missing = 1.0 - masks
        loss = F.mse_loss(pred * missing, images * missing, reduction="sum")
        return loss / (missing.sum() * images.shape[1]).clamp_min(1.0)

    @torch.no_grad()
    def inpaint(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        pred = self._predict(images, masks)
        return (images * masks + pred * (1.0 - masks)).clamp(-1.0, 1.0)

