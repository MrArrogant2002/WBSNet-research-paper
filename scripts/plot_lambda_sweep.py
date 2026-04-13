"""Plot boundary-loss-weight sensitivity (Fig. 4 in the paper).

Reads either an aggregated CSV produced by ``aggregate_results.py`` or a plain
CSV with columns ``lambda,dice,hd95`` and writes a PNG suitable for
``paper/figures/lambda_sensitivity.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with columns lambda,dice,hd95")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Boundary loss weight sensitivity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"lambda", "dice", "hd95"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"Input CSV must contain columns {required}; found {list(frame.columns)}")

    frame = frame.sort_values("lambda").reset_index(drop=True)
    fig, ax_dice = plt.subplots(figsize=(6, 3.2))
    ax_hd = ax_dice.twinx()

    ax_dice.plot(frame["lambda"], frame["dice"], marker="o", color="tab:blue", label="Dice")
    ax_hd.plot(frame["lambda"], frame["hd95"], marker="s", color="tab:red", label="HD95")

    ax_dice.set_xlabel(r"$\lambda$ (boundary loss weight)")
    ax_dice.set_ylabel("Dice", color="tab:blue")
    ax_hd.set_ylabel("HD95", color="tab:red")
    ax_dice.tick_params(axis="y", labelcolor="tab:blue")
    ax_hd.tick_params(axis="y", labelcolor="tab:red")
    ax_dice.grid(True, alpha=0.3)
    ax_dice.set_title(args.title)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
