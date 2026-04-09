from __future__ import annotations

import torch
import torch.nn.functional as F


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
        return F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)
