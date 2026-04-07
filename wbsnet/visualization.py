from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .utils.io import ensure_dir


def _denormalize(image: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)
    image_np = (image_np * std_np) + mean_np
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return image_np


def _mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    return (mask_np > 0.5).astype(np.uint8) * 255


def save_prediction_triplet(
    save_dir: str | Path,
    sample_id: str,
    image: torch.Tensor,
    target_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> dict[str, str]:
    save_path = ensure_dir(save_dir)
    image_np = _denormalize(image, mean, std)
    target_np = _mask_to_uint8(target_mask)
    pred_np = _mask_to_uint8(pred_mask)

    overlay = image_np.copy()
    overlay[..., 0] = np.where(pred_np > 0, 255, overlay[..., 0])
    overlay[..., 1] = np.where(target_np > 0, 255, overlay[..., 1])

    image_path = save_path / f"{sample_id}_image.png"
    target_path = save_path / f"{sample_id}_target.png"
    pred_path = save_path / f"{sample_id}_prediction.png"
    overlay_path = save_path / f"{sample_id}_overlay.png"

    Image.fromarray(image_np).save(image_path)
    Image.fromarray(target_np).save(target_path)
    Image.fromarray(pred_np).save(pred_path)
    Image.fromarray(overlay).save(overlay_path)

    return {
        "image": str(image_path),
        "target": str(target_path),
        "prediction": str(pred_path),
        "overlay": str(overlay_path),
    }
