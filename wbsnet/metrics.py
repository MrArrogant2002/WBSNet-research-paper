from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .utils.distributed import DistributedState, gather_objects, reduce_counts


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    neighbors = [
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
    ]
    same = np.logical_and.reduce([mask == neighbor for neighbor in neighbors])
    return np.logical_and(mask.astype(bool), np.logical_not(same))


def _pairwise_min_distances(points_a: np.ndarray, points_b: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.array([], dtype=np.float32)
    mins: list[np.ndarray] = []
    for start in range(0, len(points_a), chunk_size):
        chunk = points_a[start : start + chunk_size]
        distances = np.sqrt(((chunk[:, None, :] - points_b[None, :, :]) ** 2).sum(axis=2))
        mins.append(distances.min(axis=1))
    return np.concatenate(mins, axis=0)


def hd95_score(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    if pred_mask.sum() == 0 and target_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or target_mask.sum() == 0:
        return float(np.hypot(*pred_mask.shape))

    pred_boundary = np.argwhere(_mask_boundary(pred_mask))
    target_boundary = np.argwhere(_mask_boundary(target_mask))
    if len(pred_boundary) == 0 or len(target_boundary) == 0:
        return float(np.hypot(*pred_mask.shape))

    pred_to_target = _pairwise_min_distances(pred_boundary, target_boundary)
    target_to_pred = _pairwise_min_distances(target_boundary, pred_boundary)
    distances = np.concatenate([pred_to_target, target_to_pred], axis=0)
    return float(np.percentile(distances, 95))


@dataclass
class BinarySegmentationMeter:
    threshold: float = 0.5
    compute_hd95: bool = True
    counts: dict[str, float] = field(
        default_factory=lambda: {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
            "loss": 0.0,
            "num_batches": 0.0,
            "num_samples": 0.0,
        }
    )
    hd95_values: list[float] = field(default_factory=list)

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()

        self.counts["tp"] += float(((preds == 1) & (targets == 1)).sum().item())
        self.counts["fp"] += float(((preds == 1) & (targets == 0)).sum().item())
        self.counts["fn"] += float(((preds == 0) & (targets == 1)).sum().item())
        self.counts["tn"] += float(((preds == 0) & (targets == 0)).sum().item())
        self.counts["loss"] += float(loss)
        self.counts["num_batches"] += 1.0
        self.counts["num_samples"] += float(targets.shape[0])

        if self.compute_hd95:
            pred_np = preds.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            for pred_sample, target_sample in zip(pred_np, target_np):
                self.hd95_values.append(hd95_score(pred_sample[0], target_sample[0]))

    def compute(self, distributed_state: DistributedState | None = None) -> dict[str, float]:
        counts = self.counts
        hd95_values = self.hd95_values
        if distributed_state is not None:
            counts = reduce_counts(counts, distributed_state)
            hd95_values = gather_objects(hd95_values, distributed_state)

        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        dice = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
        iou = _safe_divide(tp, tp + fp + fn)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
        specificity = _safe_divide(tn, tn + fp)
        mean_loss = _safe_divide(counts["loss"], counts["num_batches"])

        metrics = {
            "loss": mean_loss,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "specificity": specificity,
        }
        if hd95_values:
            metrics["hd95"] = float(np.mean(hd95_values))
        return metrics
