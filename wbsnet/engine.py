from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch.amp import GradScaler
from tqdm import tqdm

from .losses import total_loss
from .metrics import BinarySegmentationMeter
from .utils.distributed import DistributedState, is_main_process
from .utils.io import ensure_dir, save_json
from .visualization import (
    create_prediction_visuals,
    save_contact_sheet,
    save_prediction_triplet,
)

if TYPE_CHECKING:
    from .utils.logger import ExperimentLogger


def select_device(
    config: dict[str, Any], distributed_state: DistributedState
) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device(
            f"cuda:{distributed_state.local_rank}"
            if distributed_state.enabled
            else "cuda"
        )
    return torch.device("cpu")


def configure_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", True))
    allow_tf32 = bool(runtime.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
        torch.set_float32_matmul_precision("high")
    if runtime.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(
    model: torch.nn.Module, config: dict[str, Any]
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    train_cfg = config["train"]
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(param) for param in encoder_params}
    decoder_params = [
        param for param in model.parameters() if id(param) not in encoder_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(train_cfg.get("encoder_lr", 1e-4))},
            {"params": decoder_params, "lr": float(train_cfg.get("decoder_lr", 1e-3))},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["epochs"])
    )
    return optimizer, scheduler


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    module = model.module if hasattr(model, "module") else model
    return module._orig_mod if hasattr(module, "_orig_mod") else module


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    include_training_state: bool = True,
) -> None:
    module = _unwrap_model(model)
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "state_dict": module.state_dict(),
    }
    if include_training_state:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
        payload["scaler"] = scaler.state_dict()
    target = Path(path)
    ensure_dir(target.parent)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp_target)
    tmp_target.replace(target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    # weights_only=False is explicit: checkpoints carry optimizer/scheduler/scaler
    # state plus the run config, which require pickle deserialization. Only load
    # checkpoints produced by this codebase or other trusted sources.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    module = _unwrap_model(model)
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


def _mean_scalar_dict(totals: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        return {key: 0.0 for key in totals}
    return {key: float(value / count) for key, value in totals.items()}


def _raise_if_nonfinite(name: str, value: torch.Tensor, *, epoch: int, step: int) -> None:
    if not torch.isfinite(value.detach()).all():
        raise FloatingPointError(f"Non-finite {name} at epoch={epoch + 1}, step={step}.")


def _finish_optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    clip_grad_norm: float,
    nonfinite_grad_action: str,
) -> bool:
    scaler.unscale_(optimizer)
    should_skip = False
    if clip_grad_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_grad_norm,
            error_if_nonfinite=False,
        )
        should_skip = not torch.isfinite(total_norm).all()
    else:
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                should_skip = True
                break

    if should_skip:
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
        if nonfinite_grad_action == "skip":
            return False
        raise FloatingPointError("Non-finite gradient detected before optimizer step.")

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return True


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
    logger: "ExperimentLogger | None" = None,
    split_name: str | None = None,
) -> dict[str, float]:
    amp_enabled = bool(config["train"].get("amp", True) and device.type == "cuda")
    grad_accum_steps = int(config["train"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError(f"train.grad_accum_steps must be >= 1, got {grad_accum_steps}.")
    boundary_weight = float(config["train"].get("boundary_loss_weight", 0.5))
    clip_grad_norm = float(config["train"].get("clip_grad_norm", 0.0))
    nonfinite_grad_action = str(config["train"].get("nonfinite_grad_action", "error")).lower()
    if nonfinite_grad_action not in {"error", "skip"}:
        raise ValueError(
            "train.nonfinite_grad_action must be either 'error' or 'skip', "
            f"got {nonfinite_grad_action!r}."
        )
    max_nonfinite_grad_steps = int(config["train"].get("max_nonfinite_grad_steps", 0))
    threshold = float(config["evaluation"].get("threshold", 0.5))
    meter = BinarySegmentationMeter(
        threshold=threshold,
        compute_hd95=(
            False if training else bool(config["evaluation"].get("compute_hd95", True))
        ),
    )
    loss_totals = {"segmentation_loss": 0.0, "boundary_loss": 0.0, "total_loss": 0.0}
    loss_steps = 0
    skipped_optimizer_steps = 0
    captured_images: list[dict[str, Any]] = []
    should_log_images = (
        (not training)
        and logger is not None
        and is_main_process(distributed_state)
        and bool(
            config.get("runtime", {}).get("wandb", {}).get("upload_val_examples", True)
        )
        and (
            (epoch + 1)
            % int(config.get("runtime", {}).get("wandb", {}).get("log_images_every", 1))
            == 0
        )
    )
    max_wandb_images = int(
        config.get("runtime", {}).get("wandb", {}).get("max_images", 4)
    )
    mean = config["dataset"].get("normalize_mean", [0.485, 0.456, 0.406])
    std = config["dataset"].get("normalize_std", [0.229, 0.224, 0.225])

    model.train(training)
    if training and hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    iterator = _progress_bar(
        loader, "train" if training else "eval", is_main_process(distributed_state)
    )
    for step, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
                loss, loss_parts = total_loss(output, masks, boundary_weight)
                _raise_if_nonfinite("loss", loss, epoch=epoch, step=step)
                scaled_loss = loss / grad_accum_steps

            if training and optimizer is not None:
                scaler.scale(scaled_loss).backward()
                if step % grad_accum_steps == 0:
                    optimizer_stepped = _finish_optimizer_step(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        clip_grad_norm=clip_grad_norm,
                        nonfinite_grad_action=nonfinite_grad_action,
                    )
                    if not optimizer_stepped:
                        skipped_optimizer_steps += 1
                        if (
                            max_nonfinite_grad_steps > 0
                            and skipped_optimizer_steps > max_nonfinite_grad_steps
                        ):
                            raise FloatingPointError(
                                "Exceeded train.max_nonfinite_grad_steps="
                                f"{max_nonfinite_grad_steps} at epoch={epoch + 1}."
                            )

        meter.update(
            output["logits"].detach(), masks.detach(), float(loss.detach().item())
        )
        for key in loss_totals:
            loss_totals[key] += float(loss_parts[key])
        loss_steps += 1

        if should_log_images and len(captured_images) < max_wandb_images:
            probs = (torch.sigmoid(output["logits"].detach()) >= threshold).float()
            for idx in range(images.shape[0]):
                if len(captured_images) >= max_wandb_images:
                    break
                visuals = create_prediction_visuals(
                    image=batch["image"][idx],
                    target_mask=batch["mask"][idx],
                    pred_mask=probs[idx].cpu(),
                    mean=mean,
                    std=std,
                )
                captured_images.append(
                    {
                        "image": visuals["paper_panel"],
                        "caption": str(batch["sample_id"][idx]),
                    }
                )

    if training and optimizer is not None and len(loader) % grad_accum_steps != 0:
        optimizer_stepped = _finish_optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad_norm=clip_grad_norm,
            nonfinite_grad_action=nonfinite_grad_action,
        )
        if not optimizer_stepped:
            skipped_optimizer_steps += 1
            if max_nonfinite_grad_steps > 0 and skipped_optimizer_steps > max_nonfinite_grad_steps:
                raise FloatingPointError(
                    "Exceeded train.max_nonfinite_grad_steps="
                    f"{max_nonfinite_grad_steps} at epoch={epoch + 1}."
                )

    metrics = meter.compute(distributed_state)
    metrics.update(_mean_scalar_dict(loss_totals, loss_steps))
    if training:
        metrics["skipped_optimizer_steps"] = float(skipped_optimizer_steps)
    if should_log_images and captured_images:
        logger.log_panel_images(
            f"{split_name or 'val'}/paper_panels", captured_images, step=epoch
        )
    return metrics


@torch.no_grad()
def evaluate_and_save_predictions(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    config: dict[str, Any],
    distributed_state: DistributedState,
    save_dir: str | Path | None = None,
    logger: "ExperimentLogger | None" = None,
    step: int = 0,
    split_name: str = "evaluation",
) -> dict[str, float]:
    model.eval()
    amp_enabled = bool(config["train"].get("amp", True) and device.type == "cuda")
    threshold = float(config["evaluation"].get("threshold", 0.5))
    meter = BinarySegmentationMeter(
        threshold=threshold,
        compute_hd95=bool(
            config["evaluation"].get("compute_hd95", True)
        ),  # always use config for eval
    )
    loss_totals = {"segmentation_loss": 0.0, "boundary_loss": 0.0, "total_loss": 0.0}
    loss_steps = 0

    saved = 0
    max_visualizations = int(config["evaluation"].get("max_visualizations", 24))
    save_paper_panels = bool(config["evaluation"].get("save_paper_panels", True))
    save_contact_sheet_enabled = bool(
        config["evaluation"].get("save_contact_sheet", True)
    )
    contact_sheet_columns = int(config["evaluation"].get("contact_sheet_columns", 2))
    mean = config["dataset"].get("normalize_mean", [0.485, 0.456, 0.406])
    std = config["dataset"].get("normalize_std", [0.229, 0.224, 0.225])
    logged_images: list[dict[str, Any]] = []
    max_wandb_images = int(
        config.get("runtime", {}).get("wandb", {}).get("max_images", 4)
    )
    upload_eval_examples = bool(
        config.get("runtime", {}).get("wandb", {}).get("upload_eval_examples", True)
    )
    paper_panel_paths: list[str] = []

    for batch in _progress_bar(loader, "predict", is_main_process(distributed_state)):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(images)
            loss, loss_parts = total_loss(
                output, masks, float(config["train"].get("boundary_loss_weight", 0.5))
            )
        _raise_if_nonfinite("evaluation loss", loss, epoch=step, step=loss_steps + 1)
        meter.update(output["logits"], masks, float(loss.item()))
        for key in loss_totals:
            loss_totals[key] += float(loss_parts[key])
        loss_steps += 1

        if save_dir is not None and is_main_process(distributed_state):
            probs = (torch.sigmoid(output["logits"]) >= threshold).float()
            for idx in range(images.shape[0]):
                if saved >= max_visualizations:
                    break
                saved_paths = save_prediction_triplet(
                    save_dir=save_dir,
                    sample_id=batch["sample_id"][idx],
                    image=batch["image"][idx],
                    target_mask=batch["mask"][idx],
                    pred_mask=probs[idx],
                    mean=mean,
                    std=std,
                )
                if save_paper_panels:
                    paper_panel_paths.append(saved_paths["paper_panel"])
                if (
                    logger is not None
                    and upload_eval_examples
                    and len(logged_images) < max_wandb_images
                ):
                    visuals = create_prediction_visuals(
                        image=batch["image"][idx],
                        target_mask=batch["mask"][idx],
                        pred_mask=probs[idx].cpu(),
                        mean=mean,
                        std=std,
                    )
                    logged_images.append(
                        {
                            "image": visuals["paper_panel"],
                            "caption": str(batch["sample_id"][idx]),
                        }
                    )
                saved += 1

    if (
        save_dir is not None
        and is_main_process(distributed_state)
        and save_contact_sheet_enabled
        and paper_panel_paths
    ):
        save_contact_sheet(
            panel_paths=paper_panel_paths,
            output_path=Path(save_dir) / "paper_contact_sheet.png",
            columns=contact_sheet_columns,
        )
    if logger is not None and logged_images:
        logger.log_panel_images(f"{split_name}/paper_panels", logged_images, step=step)

    metrics = meter.compute(distributed_state)
    metrics.update(_mean_scalar_dict(loss_totals, loss_steps))
    return metrics


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
