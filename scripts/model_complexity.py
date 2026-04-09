from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbsnet.config import load_config
from wbsnet.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate parameter counts and FLOPs for WBSNet configs.")
    parser.add_argument("--configs", nargs="*", default=None, help="Config files to analyze. Defaults to all configs/*.yaml.")
    parser.add_argument("--output", required=True, help="Output directory for complexity reports.")
    parser.add_argument("--height", type=int, default=None, help="Optional input height override.")
    parser.add_argument("--width", type=int, default=None, help="Optional input width override.")
    return parser.parse_args()


def _count_parameters(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _estimate_flops(model: Any, height: int, width: int, channels: int) -> float:
    import torch
    from torch.profiler import ProfilerActivity, profile

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, channels, height, width, device=device)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
            model(dummy)
    return float(sum(max(0, int(event.flops)) for event in prof.key_averages()))


def main() -> None:
    import torch

    args = parse_args()
    output_dir = ensure_dir(args.output)
    config_paths = [Path(path).resolve() for path in args.configs] if args.configs else sorted(Path("configs").glob("*.yaml"))

    from wbsnet.models import build_model, variant_name_from_config

    records: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        config["model"]["encoder_pretrained"] = False
        config["model"]["encoder_pretrained_checkpoint"] = None
        model = build_model(config)
        total_params, trainable_params = _count_parameters(model)

        image_size = config["dataset"].get("image_size", [352, 352])
        height = int(args.height or image_size[0])
        width = int(args.width or image_size[1])
        channels = int(config["model"].get("in_channels", 3))

        flops = _estimate_flops(model, height, width, channels)
        records.append(
            {
                "config_path": str(config_path),
                "experiment_name": config["experiment"]["name"],
                "variant_name": variant_name_from_config(config),
                "input_shape": f"1x{channels}x{height}x{width}",
                "params_total": int(total_params),
                "params_trainable": int(trainable_params),
                "params_millions": float(total_params / 1e6),
                "flops": float(flops),
                "gflops": float(flops / 1e9),
                "gmacs": float(flops / 2e9),
                "torch_version": torch.__version__,
            }
        )

    frame = pd.DataFrame(records).sort_values(["params_total", "config_path"])
    frame.to_csv(output_dir / "model_complexity.csv", index=False)
    (output_dir / "model_complexity.md").write_text(frame.to_markdown(index=False), encoding="utf-8")
    (output_dir / "model_complexity.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved model complexity report to {output_dir}")


if __name__ == "__main__":
    main()
