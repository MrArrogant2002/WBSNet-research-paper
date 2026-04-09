from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbsnet.visualization import save_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-ready contact sheet from saved WBSNet paper panels.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_paper_panel.png files.")
    parser.add_argument("--output", default=None, help="Optional output path for the contact sheet.")
    parser.add_argument("--pattern", default="*_paper_panel.png", help="Glob pattern for panel files.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum number of panels to include.")
    parser.add_argument("--columns", type=int, default=2, help="Number of columns in the contact sheet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    panel_paths = sorted(input_dir.glob(args.pattern))
    if not panel_paths:
        raise FileNotFoundError(f"No panels matching '{args.pattern}' were found in {input_dir}")

    selected = panel_paths[: max(1, args.limit)]
    output_path = Path(args.output).resolve() if args.output else input_dir / "paper_contact_sheet.png"
    target = save_contact_sheet(selected, output_path=output_path, columns=max(1, args.columns))
    print(f"Saved paper contact sheet to {target}")


if __name__ == "__main__":
    main()
