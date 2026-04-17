from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wbsnet.config import load_config
from wbsnet.data.datasets import build_inference_loader
from wbsnet.engine import evaluate_and_save_predictions, load_checkpoint, persist_metrics, select_device
from wbsnet.models import build_model, variant_name_from_config
from wbsnet.utils import ensure_dir, load_env_file
from wbsnet.utils.distributed import cleanup_distributed, init_distributed, is_main_process
from wbsnet.utils.logger import ExperimentLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained WBSNet checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides in key=value form.")
    return parser.parse_args()


def _dataset_variants(config: dict) -> list[dict]:
    extra = config.get("evaluation", {}).get("datasets", [])
    if extra:
        return extra
    dataset_cfg = dict(config["dataset"])
    dataset_cfg["split"] = "test"
    return [dataset_cfg]


def main() -> None:
    args = parse_args()
    load_env_file(".env")
    config = load_config(args.config, args.override)
    # Final evaluation should report HD95. The shared default config disables it
    # during training for speed; force it on here unless the user explicitly
    # opted out via --override evaluation.compute_hd95=false.
    user_set_hd95 = any(o.startswith("evaluation.compute_hd95=") for o in args.override)
    if not user_set_hd95:
        config.setdefault("evaluation", {})["compute_hd95"] = True
    distributed_state = init_distributed(config["runtime"]["distributed"].get("backend", "nccl"))
    device = select_device(config, distributed_state)

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    run_dir = Path(args.checkpoint).resolve().parents[1]
    evaluation_dir = ensure_dir(run_dir / "evaluation")
    logger = None
    if bool(config["runtime"]["wandb"].get("enabled", True) and config["runtime"]["wandb"].get("upload_eval_examples", True)):
        logger = ExperimentLogger(
            output_dir=evaluation_dir,
            config=config,
            enabled=True,
            rank=distributed_state.rank,
            open_csv=False,
        )

    try:
        for dataset_cfg in _dataset_variants(config):
            merged_dataset = dict(config["dataset"])
            merged_dataset.update(dataset_cfg)
            loader = build_inference_loader(
                dataset_config=merged_dataset,
                split=dataset_cfg.get("split", args.split),
                batch_size=int(config["train"]["batch_size"]),
                distributed_state=distributed_state,
            )
            save_dir = evaluation_dir / merged_dataset["name"] / "predictions" if is_main_process(distributed_state) else None
            metrics = evaluate_and_save_predictions(
                model=model,
                loader=loader,
                device=device,
                config={**config, "dataset": merged_dataset},
                distributed_state=distributed_state,
                save_dir=save_dir,
                logger=logger,
                step=0,
                split_name=f"evaluation/{merged_dataset['name']}",
            )
            if is_main_process(distributed_state):
                payload = {
                    "metrics": metrics,
                    "dataset_name": merged_dataset["name"],
                    "variant_name": variant_name_from_config(config),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "experiment_name": config["experiment"]["name"],
                    "run_name": config["experiment"]["run_name"],
                    "seed": config["experiment"]["seed"],
                }
                persist_metrics(evaluation_dir / f"{merged_dataset['name']}.json", payload)
    finally:
        if logger is not None:
            logger.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
