
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class BaseInpaintingModel(nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        args:
            images: tensor (B, C, H, W)
            masks: tensor (B, 1, H, W)

        return:
            loss: float

        """
        pass

    @abstractmethod
    @torch.no_grad()
    def inpaint(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        args:
            images: tensor (B, C, H, W)
            masks: tensor (B, 1, H, W)

        return:
            inpainted images: tensor (B, C, H, W)

        """
        pass

