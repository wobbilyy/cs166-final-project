import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        B
        -> B dim
        """
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class TimeConditionedConvBlock(nn.Module):
    """
    same as unet conv block; but ADD (not concat) time embedding
    """
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x: B in H W
        time embedding: B time dimension

        -> B out H W
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)

        # broadcast:
        #  B out 1 1
        #  added to
        #  B out H W
        time_bias = self.time_proj(time_emb)[:, :, None, None]
        x = x + time_bias

        x = self.conv2(x)
        x = self.norm2(x)

        return F.silu(x)


class DDPMUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 64):
        super().__init__()
        time_dim = hidden_dim * 4

        """
        B -> B time dimension
        """
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down1 = TimeConditionedConvBlock(in_channels, hidden_dim, time_dim)
        self.down2 = TimeConditionedConvBlock(hidden_dim, hidden_dim * 2, time_dim)

        self.bottleneck = TimeConditionedConvBlock(hidden_dim * 2, hidden_dim * 4, time_dim)

        self.up2 = TimeConditionedConvBlock(
            hidden_dim * 4 + hidden_dim * 2, hidden_dim * 2, time_dim
        )
        self.up1 = TimeConditionedConvBlock(
            hidden_dim * 2 + hidden_dim, hidden_dim, time_dim
        )
        self.out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        x:              B in H W
        timesteps:      B
        time embedding: B time_dim (added as bias at every conv block)

        1. down1:   B hidden   H    W    <- save as skip1
           avgpool: B hidden   H//2 W//2
        2. down2:   B hidden*2 H//2 W//2 <- save as skip2
           avgpool: B hidden*2 H//4 W//4

        bottleneck: B hidden*2 H//4 W//4 -> B hidden*4 H//4 W//4

        3. up2: B hidden*4+hidden*2 H//2 W//2 -> B hidden*2 H//2 W//2 <- concat skip2 to bottleneck
        4. up1: B hidden*2+hidden   H    W    -> B hidden   H    W    <- concat skip1
        5. out: B hidden            H    W    -> B out      H    W
        """
        time_emb = self.time_mlp(timesteps)

        skip1 = self.down1(x, time_emb)
        skip2 = self.down2(F.avg_pool2d(skip1, kernel_size=2), time_emb)

        x = self.bottleneck(F.avg_pool2d(skip2, kernel_size=2), time_emb)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, skip2], dim=1), time_emb)

        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, skip1], dim=1), time_emb)

        return self.out(x)

