from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


def _haar_filters() -> tuple[list[float], list[float], list[float], list[float]]:
    scale = 2**-0.5
    dec_lo = [scale, scale]
    dec_hi = [-scale, scale]
    rec_lo = [scale, scale]
    rec_hi = [scale, -scale]
    return dec_lo, dec_hi, rec_lo, rec_hi


@lru_cache(maxsize=8)
def _wavelet_filters(wavelet_type: str) -> tuple[list[float], list[float], list[float], list[float]]:
    if wavelet_type == "haar":
        return _haar_filters()
    try:
        import pywt

        wavelet = pywt.Wavelet(wavelet_type)
        return (
            list(wavelet.dec_lo),
            list(wavelet.dec_hi),
            list(wavelet.rec_lo),
            list(wavelet.rec_hi),
        )
    except Exception as exc:
        raise NotImplementedError(
            f"Wavelet '{wavelet_type}' requires pywavelets or an explicit implementation."
        ) from exc


def _stack_filters(lo: list[float], hi: list[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lo_tensor = torch.tensor(lo, device=device, dtype=dtype)
    hi_tensor = torch.tensor(hi, device=device, dtype=dtype)
    ll = torch.outer(lo_tensor, lo_tensor)
    lh = torch.outer(lo_tensor, hi_tensor)
    hl = torch.outer(hi_tensor, lo_tensor)
    hh = torch.outer(hi_tensor, hi_tensor)
    return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)


class WaveletTransform2d(torch.nn.Module):
    def __init__(self, wavelet_type: str = "haar") -> None:
        super().__init__()
        self.wavelet_type = wavelet_type
        dec_lo, dec_hi, _, _ = _wavelet_filters(wavelet_type)
        # Build as float32 once and register as buffer — autocast handles dtype promotion at forward time
        filters = _stack_filters(dec_lo, dec_hi, torch.device("cpu"), torch.float32)
        self.register_buffer("dec_filters", filters)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, height, width = x.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Wavelet input must be even-sized, got {x.shape}")

        padding = max(self.dec_filters.shape[-1] // 2 - 1, 0)
        weight = self.dec_filters.to(dtype=x.dtype).repeat(channels, 1, 1, 1)
        transformed = F.conv2d(x, weight, stride=2, padding=padding, groups=channels)
        transformed = transformed.view(bsz, channels, 4, transformed.shape[-2], transformed.shape[-1])
        return transformed[:, :, 0], transformed[:, :, 1], transformed[:, :, 2], transformed[:, :, 3]


class InverseWaveletTransform2d(torch.nn.Module):
    def __init__(self, wavelet_type: str = "haar") -> None:
        super().__init__()
        self.wavelet_type = wavelet_type
        _, _, rec_lo, rec_hi = _wavelet_filters(wavelet_type)
        # Build as float32 once and register as buffer
        filters = _stack_filters(rec_lo, rec_hi, torch.device("cpu"), torch.float32)
        self.register_buffer("rec_filters", filters)

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        _, channels, _, _ = ll.shape
        # Match the analysis-side padding so longer filters such as db2
        # reconstruct to the original spatial resolution.
        padding = max(self.rec_filters.shape[-1] // 2 - 1, 0)
        packed = torch.stack([ll, lh, hl, hh], dim=2).reshape(ll.shape[0], channels * 4, ll.shape[-2], ll.shape[-1])
        weight = self.rec_filters.to(dtype=ll.dtype).repeat(channels, 1, 1, 1)
        return F.conv_transpose2d(packed, weight, stride=2, padding=padding, groups=channels)
