from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ABLATIONS = [
    "configs/ablation_identity_unet.yaml",
    "configs/kvasir_wbsnet.yaml",
    "configs/ablation_lfsa_only.yaml",
    "configs/ablation_hfba_only.yaml",
    "configs/ablation_no_boundary_supervision.yaml",
    "configs/ablation_no_wavelet_attention.yaml",
    "configs/ablation_db2_wavelet.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the WBSNet ablation suite across multiple seeds.")
    parser.add_argument("--base-config", default="configs/kvasir_wbsnet.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    for config_path in ABLATIONS:
        for seed in args.seeds:
            command = [
                "python3",
                str(root / "train.py"),
                "--config",
                str(root / config_path),
                "--override",
                f"experiment.seed={seed}",
            ]
            if args.dry_run:
                print(" ".join(command))
                continue
            subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
