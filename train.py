from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from wbsnet.config import load_config
from wbsnet.data import build_dataloaders
from wbsnet.engine import (
    build_optimizer,
    configure_runtime,
    persist_metrics,
    persist_run_summary,
    run_epoch,
    save_checkpoint,
    select_device,
)
from wbsnet.models import build_model, variant_name_from_config
from wbsnet.utils import ensure_dir, load_env_file, seed_everything
from wbsnet.utils.io import load_json
from wbsnet.utils.distributed import cleanup_distributed, init_distributed, is_main_process
from wbsnet.utils.io import timestamp
from wbsnet.utils.logger import ExperimentLogger
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WBSNet on a medical segmentation dataset.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides in key=value form.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0], "torch": torch.__version__}
    for package in [
        "torchvision",
        "numpy",
        "Pillow",
        "PyYAML",
        "wandb",
        "pandas",
        "matplotlib",
        "pywavelets",
        "albumentations",
        "opencv-python",
        "scipy",
        "segmentation-models-pytorch",
        "pytorch-wavelets",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _runtime_report(device: torch.device, loaders: dict[str, object]) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "git_commit": _git_commit(),
        "packages": _package_versions(),
        "torch": {
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if cuda_available and device.type == "cuda" else "cpu",
            "device_count": torch.cuda.device_count(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "dataloaders": {
            name: {
                "num_batches": len(loader),  # type: ignore[arg-type]
                "num_samples": len(loader.dataset),  # type: ignore[attr-defined]
            }
            for name, loader in loaders.items()
        },
    }


def main() -> None:
    args = parse_args()
    load_env_file(".env")
    config = load_config(args.config, args.override)
    if config["experiment"].get("run_name") == config["experiment"]["name"]:
        config["experiment"]["run_name"] = (
            f"{config['experiment']['name']}_seed{config['experiment']['seed']}_{timestamp()}"
        )

    distributed_state = init_distributed(config["runtime"]["distributed"].get("backend", "nccl"))
    device = select_device(config, distributed_state)
    configure_runtime(config)
    seed_everything(int(config["experiment"]["seed"]), bool(config["runtime"].get("deterministic", False)))

    output_dir = (
        Path(config["experiment"]["output_root"])
        / config["experiment"]["name"]
        / config["experiment"]["run_name"]
    )
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    resume_path = Path(args.resume) if args.resume else checkpoint_dir / "last.pt"
    will_resume = resume_path.exists()
    logger = ExperimentLogger(
        output_dir=output_dir,
        config=config,
        enabled=bool(config["runtime"]["wandb"].get("enabled", True)),
        rank=distributed_state.rank,
        append_csv=will_resume,
    )

    model = build_model(config).to(device)
    if config["runtime"]["distributed"].get("sync_batchnorm", False) and distributed_state.enabled:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer, scheduler = build_optimizer(model, config)
    if bool(config["train"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    scaler = GradScaler(enabled=bool(config["train"].get("amp", True) and device.type == "cuda"))

    start_epoch = 0
    best_metric = float("-inf") if config["train"].get("monitor_mode", "max") == "max" else float("inf")
    if distributed_state.enabled:
        model = DDP(model, device_ids=[distributed_state.local_rank] if device.type == "cuda" else None)

    if args.resume:
        from wbsnet.engine import load_checkpoint

        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
    elif will_resume:
        from wbsnet.engine import load_checkpoint

        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        print(f"Resuming {config['experiment']['run_name']} from epoch {start_epoch + 1}.")

    loaders = build_dataloaders(config, distributed_state)
    if is_main_process(distributed_state):
        ensure_dir(output_dir)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "run_environment.json").write_text(
            json.dumps(_runtime_report(device, loaders), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    monitor_key = config["train"].get("monitor", "dice")
    maximize = config["train"].get("monitor_mode", "max") == "max"
    variant_name = variant_name_from_config(config)
    best_metrics_path = output_dir / "best_metrics.json"
    best_metrics_payload = load_json(best_metrics_path) if best_metrics_path.exists() else {"monitor": best_metric}
    patience = config["train"].get("early_stopping_patience")
    patience = None if patience is None else int(patience)
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, int(config["train"]["epochs"])):
            last_epoch = epoch
            train_metrics = run_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                config=config,
                distributed_state=distributed_state,
                training=True,
                epoch=epoch,
                logger=logger,
                split_name="train",
            )
            val_metrics = run_epoch(
                model=model,
                loader=loaders["val"],
                optimizer=None,
                scaler=scaler,
                device=device,
                config=config,
                distributed_state=distributed_state,
                training=False,
                epoch=epoch,
                logger=logger,
                split_name="val",
            )
            scheduler.step()

            payload = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": {
                    "encoder": optimizer.param_groups[0]["lr"],
                    "decoder": optimizer.param_groups[1]["lr"],
                },
            }
            logger.log_metrics(epoch, payload)

            current_metric = float(val_metrics[monitor_key])
            improved = current_metric > best_metric if maximize else current_metric < best_metric
            if improved:
                best_metric = current_metric
                epochs_without_improvement = 0
                best_metrics_payload = dict(val_metrics)
                if is_main_process(distributed_state):
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_metric,
                        config,
                        include_training_state=bool(config["train"].get("save_best_full_state", True)),
                    )
                    persist_metrics(output_dir / "best_metrics.json", val_metrics)
            else:
                epochs_without_improvement += 1

            save_every = int(config["train"].get("save_every", 10))
            if is_main_process(distributed_state) and save_every > 0 and (epoch + 1) % save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_metric,
                    config,
                )
            if patience is not None and epochs_without_improvement >= patience:
                if is_main_process(distributed_state):
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"{monitor_key} did not improve for {patience} validation epochs."
                    )
                break

        if is_main_process(distributed_state):
            if bool(config["train"].get("save_last_checkpoint", True)):
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    last_epoch,
                    best_metric,
                    config,
                )
            persist_run_summary(
                output_dir=output_dir,
                config=config,
                best_metrics=best_metrics_payload,
                extra={"variant_name": variant_name, "last_epoch": last_epoch},
            )
    finally:
        logger.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
