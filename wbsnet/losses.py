from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


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


def boundary_targets_from_masks(masks: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        targets = masks.detach().float()
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=targets.device,
            dtype=targets.dtype,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=targets.device,
            dtype=targets.dtype,
        ).unsqueeze(0)
        grad_x = F.conv2d(targets, sobel_x, padding=1)
        grad_y = F.conv2d(targets, sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
        edges = (magnitude > 0).float()
        edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)
    return edges


def boundary_loss(boundary_logits: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    if not boundary_logits:
        return targets.new_tensor(0.0)

    boundary_gt = boundary_targets_from_masks(targets)
    losses = []
    for logits in boundary_logits:
        resized = F.interpolate(boundary_gt, size=logits.shape[-2:], mode="nearest")
        losses.append(F.binary_cross_entropy_with_logits(logits, resized))
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
