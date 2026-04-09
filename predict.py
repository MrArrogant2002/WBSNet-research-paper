from __future__ import annotations

import argparse
from pathlib import Path

from wbsnet.config import load_config
from wbsnet.data.datasets import build_inference_loader
from wbsnet.engine import evaluate_and_save_predictions, load_checkpoint, select_device
from wbsnet.models import build_model
from wbsnet.utils import ensure_dir, load_env_file
from wbsnet.utils.distributed import cleanup_distributed, init_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save qualitative predictions from a WBSNet checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--output-dir", default=None, help="Optional prediction output directory.")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides in key=value form.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(".env")
    config = load_config(args.config, args.override)
    distributed_state = init_distributed(config["runtime"]["distributed"].get("backend", "nccl"))
    device = select_device(config, distributed_state)

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    loader = build_inference_loader(
        dataset_config=config["dataset"],
        split=args.split,
        batch_size=int(config["train"]["batch_size"]),
        distributed_state=distributed_state,
    )
    run_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parents[1] / "predictions"
    ensure_dir(run_dir)

    try:
        metrics = evaluate_and_save_predictions(
            model=model,
            loader=loader,
            device=device,
            config=config,
            distributed_state=distributed_state,
            save_dir=run_dir,
        )
        print(metrics)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
