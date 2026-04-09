from __future__ import annotations

import torch
import torch.nn as nn


class LFSA(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction_ratio, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))
