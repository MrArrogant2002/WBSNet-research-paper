from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .utils.boundary_gt import boundary_targets_from_masks


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, targets.ndim))
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


def boundary_loss(boundary_logits: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    if not boundary_logits:
        return targets.new_tensor(0.0)

    boundary_gt = boundary_targets_from_masks(targets)
    losses = []
    for logits in boundary_logits:
        upsampled_logits = F.interpolate(logits, size=boundary_gt.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.binary_cross_entropy_with_logits(upsampled_logits, boundary_gt))
    return torch.stack(losses).mean()


def total_loss(model_output: dict[str, Any], targets: torch.Tensor, boundary_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    seg = segmentation_loss(model_output["logits"], targets)
    bnd = boundary_loss(model_output.get("boundary_logits", []), targets)
    total = seg + boundary_weight * bnd
    return total, {
        "segmentation_loss": float(seg.detach().item()),
        "boundary_loss": float(bnd.detach().item()),
        "total_loss": float(total.detach().item()),
    }
