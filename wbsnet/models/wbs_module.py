from __future__ import annotations

import torch
import torch.nn as nn

from .wavelet import InverseWaveletTransform2d, WaveletTransform2d


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

    def forward(self, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        merged = torch.cat([lh, hl, hh], dim=1)
        reduced = self.pointwise(self.depthwise(merged))
        boundary_logits = self.boundary_head(reduced)
        attended = reduced * torch.sigmoid(boundary_logits)
        return attended, boundary_logits


class RawAttentionSkip(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, use_lfsa: bool = True, use_hfba: bool = True) -> None:
        super().__init__()
        self.use_lfsa = use_lfsa
        self.use_hfba = use_hfba
        self.channel_attention = LFSA(channels, reduction_ratio) if use_lfsa else nn.Identity()
        if use_hfba:
            self.spatial = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, kernel_size=1),
            )
        else:
            self.spatial = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.channel_attention(x)
        if self.spatial is None:
            return out, None
        boundary_logits = self.spatial(out)
        out = out * torch.sigmoid(boundary_logits)
        return out, boundary_logits


class WBSModule(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        use_wavelet: bool = True,
        use_lfsa: bool = True,
        use_hfba: bool = True,
        boundary_supervision: bool = True,
        wavelet_type: str = "haar",
    ) -> None:
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_lfsa = use_lfsa
        self.use_hfba = use_hfba
        self.boundary_supervision = boundary_supervision

        self.dwt = WaveletTransform2d(wavelet_type=wavelet_type)
        self.idwt = InverseWaveletTransform2d(wavelet_type=wavelet_type)
        self.lfsa = LFSA(channels, reduction_ratio) if use_lfsa else nn.Identity()
        self.hfba = HFBA(channels) if use_hfba else None
        self.raw_attention = RawAttentionSkip(channels, reduction_ratio, use_lfsa, use_hfba)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_wavelet:
            refined, boundary_logits = self.raw_attention(x)
            if not self.boundary_supervision:
                boundary_logits = None
            return refined, boundary_logits

        ll, lh, hl, hh = self.dwt(x)
        if self.use_lfsa:
            ll = self.lfsa(ll)

        if self.use_hfba and self.hfba is not None:
            hf, boundary_logits = self.hfba(lh, hl, hh)
            lh = hf
            hl = hf
            hh = hf
        else:
            boundary_logits = None

        refined = self.idwt(ll, lh, hl, hh)
        if not self.boundary_supervision:
            boundary_logits = None
        return refined, boundary_logits
