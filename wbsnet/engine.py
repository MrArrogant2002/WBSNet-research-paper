from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .losses import total_loss
from .metrics import BinarySegmentationMeter
from .utils.distributed import DistributedState, is_main_process
from .utils.io import ensure_dir, save_json
from .visualization import save_prediction_triplet


def select_device(config: dict[str, Any], distributed_state: DistributedState) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{distributed_state.local_rank}" if distributed_state.enabled else "cuda")
    return torch.device("cpu")


def configure_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", True))
    if runtime.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    train_cfg = config["train"]
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(param) for param in encoder_params}
    decoder_params = [param for param in model.parameters() if id(param) not in encoder_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(train_cfg.get("encoder_lr", 1e-4))},
            {"params": decoder_params, "lr": float(train_cfg.get("decoder_lr", 1e-3))},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    return optimizer, scheduler


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    module = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "state_dict": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _progress_bar(loader: Any, desc: str, enabled: bool) -> Any:
    return tqdm(loader, desc=desc, leave=False, disable=not enabled)


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: torch.device,
    config: dict[str, Any],
    distributed_state: DistributedState,
    training: bool,
    epoch: int,
) -> dict[str, float]:
    amp_enabled = bool(config["train"].get("amp", True) and device.type == "cuda")
    grad_accum_steps = int(config["train"].get("grad_accum_steps", 1))
    boundary_weight = float(config["train"].get("boundary_loss_weight", 0.5))
    clip_grad_norm = float(config["train"].get("clip_grad_norm", 0.0))
    meter = BinarySegmentationMeter(
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        compute_hd95=bool(config["evaluation"].get("compute_hd95", True)),
    )

    model.train(training)
    if training and hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    iterator = _progress_bar(loader, "train" if training else "eval", is_main_process(distributed_state))
    for step, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
                loss, _ = total_loss(output, masks, boundary_weight)
                scaled_loss = loss / grad_accum_steps

            if training and optimizer is not None:
                scaler.scale(scaled_loss).backward()
                if step % grad_accum_steps == 0:
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        meter.update(output["logits"].detach(), masks.detach(), float(loss.detach().item()))

    if training and optimizer is not None and len(loader) % grad_accum_steps != 0:
        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return meter.compute(distributed_state)


@torch.no_grad()
def evaluate_and_save_predictions(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    config: dict[str, Any],
    distributed_state: DistributedState,
    save_dir: str | Path | None = None,
) -> dict[str, float]:
    model.eval()
    meter = BinarySegmentationMeter(
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        compute_hd95=bool(config["evaluation"].get("compute_hd95", True)),
    )

    saved = 0
    max_visualizations = int(config["evaluation"].get("max_visualizations", 24))
    mean = config["dataset"].get("normalize_mean", [0.485, 0.456, 0.406])
    std = config["dataset"].get("normalize_std", [0.229, 0.224, 0.225])

    for batch in _progress_bar(loader, "predict", is_main_process(distributed_state)):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        output = model(images)
        loss, _ = total_loss(output, masks, float(config["train"].get("boundary_loss_weight", 0.5)))
        meter.update(output["logits"], masks, float(loss.item()))

        if save_dir is not None and is_main_process(distributed_state):
            probs = (torch.sigmoid(output["logits"]) >= float(config["evaluation"].get("threshold", 0.5))).float()
            for idx in range(images.shape[0]):
                if saved >= max_visualizations:
                    break
                save_prediction_triplet(
                    save_dir=save_dir,
                    sample_id=batch["sample_id"][idx],
                    image=batch["image"][idx],
                    target_mask=batch["mask"][idx],
                    pred_mask=probs[idx],
                    mean=mean,
                    std=std,
                )
                saved += 1

    return meter.compute(distributed_state)


def persist_run_summary(
    output_dir: str | Path,
    config: dict[str, Any],
    best_metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    summary = {
        "experiment_name": config["experiment"]["name"],
        "run_name": config["experiment"]["run_name"],
        "dataset_name": config["dataset"]["name"],
        "seed": config["experiment"]["seed"],
        "best_metrics": best_metrics,
    }
    if extra:
        summary.update(extra)
    save_json(Path(output_dir) / "run_summary.json", summary)


def persist_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    save_json(path, metrics)
