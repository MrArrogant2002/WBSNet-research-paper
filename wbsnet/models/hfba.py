from __future__ import annotations

import torch
import torch.nn as nn


class HFBA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        merged_channels = channels * 3
        self.depthwise = nn.Sequential(
            nn.Conv2d(merged_channels, merged_channels, kernel_size=3, padding=1, groups=merged_channels, bias=False),
            nn.BatchNorm2d(merged_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(merged_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        merged = torch.cat([lh, hl, hh], dim=1)
        reduced = self.pointwise(self.depthwise(merged))
        boundary_logits = self.boundary_head(reduced)
        gate = torch.sigmoid(boundary_logits)
        attended = reduced * gate
        return attended, gate, boundary_logits
