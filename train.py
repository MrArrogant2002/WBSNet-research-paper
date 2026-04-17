from __future__ import annotations

import argparse
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
from wbsnet.utils.distributed import cleanup_distributed, init_distributed, is_main_process
from wbsnet.utils.io import timestamp
from wbsnet.utils.logger import ExperimentLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WBSNet on a medical segmentation dataset.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides in key=value form.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


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
    logger = ExperimentLogger(
        output_dir=output_dir,
        config=config,
        enabled=bool(config["runtime"]["wandb"].get("enabled", True)),
        rank=distributed_state.rank,
    )

    model = build_model(config).to(device)
    if config["runtime"]["distributed"].get("sync_batchnorm", False) and distributed_state.enabled:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if bool(config["train"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer, scheduler = build_optimizer(model, config)
    scaler = GradScaler(enabled=bool(config["train"].get("amp", True) and device.type == "cuda"))

    start_epoch = 0
    best_metric = float("-inf") if config["train"].get("monitor_mode", "max") == "max" else float("inf")
    if distributed_state.enabled:
        model = DDP(model, device_ids=[distributed_state.local_rank] if device.type == "cuda" else None)

    if args.resume:
        from wbsnet.engine import load_checkpoint

        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    loaders = build_dataloaders(config, distributed_state)
    monitor_key = config["train"].get("monitor", "dice")
    maximize = config["train"].get("monitor_mode", "max") == "max"
    variant_name = variant_name_from_config(config)
    best_metrics_payload = {"monitor": best_metric}

    try:
        for epoch in range(start_epoch, int(config["train"]["epochs"])):
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
                    )
                    persist_metrics(output_dir / "best_metrics.json", val_metrics)

            save_every = int(config["train"].get("save_every", 10))
            if is_main_process(distributed_state) and (epoch + 1) % save_every == 0:
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

        if is_main_process(distributed_state):
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                int(config["train"]["epochs"]) - 1,
                best_metric,
                config,
            )
            persist_run_summary(
                output_dir=output_dir,
                config=config,
                best_metrics=best_metrics_payload,
                extra={"variant_name": variant_name},
            )
    finally:
        logger.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
