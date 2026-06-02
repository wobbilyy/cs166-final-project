import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

            # layer norm?
            nn.GroupNorm(_num_groups(out_channels), out_channels),

            # test out other activation functions?
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        B in H W -> B out H W

        after avg_pool(..., 2) -> B out H//2 W//2
        after interpolate up -> B out H*2 W*2
        """
        return self.net(x)


class Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 64):
        super().__init__()

        self.down1 = ConvBlock(in_channels, hidden_dim)
        self.down2 = ConvBlock(hidden_dim, hidden_dim * 2)

        self.bottleneck = ConvBlock(hidden_dim * 2, hidden_dim * 4)

        self.up2 = ConvBlock(hidden_dim * 4 + hidden_dim * 2, hidden_dim * 2)
        self.up1 = ConvBlock(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B in H W

        1. down1:   B hidden   H    W    <- save as skip
           avgpool: B hidden   H//2 W//2
        2. down2:   B hidden*2 H//2 W//2 <- save as skip
           avgpool: B hidden*2 H//4 W//4

        bottleneck: B hidden*2 H//4 W//4 -> B hidden*4 H//4 W//4

        3. up2: B hidden*4+hidden*2 H//4 W//4 -> B hidden * 2 H//2 W//2 <- concat second skip to bottleneck
        4. up1: B hidden*2+hidden   H//2 W//2 -> B hidden     H    W    <- concat first skip
        5. out: B hidden            H    W    -> B out        H    W
        """
        skip1 = self.down1(x)
        x = F.avg_pool2d(skip1, kernel_size=2)
        skip2 = self.down2(x)
        x = F.avg_pool2d(skip2, kernel_size=2)

        x = self.bottleneck(x)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, skip2], dim=1))

        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, skip1], dim=1))

        return self.out(x)

