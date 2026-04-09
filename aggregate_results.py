from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from wbsnet.utils.io import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate WBSNet experiment outputs for paper tables.")
    parser.add_argument("--root", required=True, help="Root output directory to scan.")
    parser.add_argument("--output", required=True, help="Directory for aggregated files.")
    return parser.parse_args()


def _collect_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in root.rglob("run_summary.json"):
        payload = load_json(path)
        best_metrics = payload.get("best_metrics", {})
        record = {
            "source": str(path),
            "record_type": "run_summary",
            "experiment_name": payload.get("experiment_name"),
            "run_name": payload.get("run_name"),
            "dataset_name": payload.get("dataset_name"),
            "variant_name": payload.get("variant_name"),
            "seed": payload.get("seed"),
        }
        for key, value in best_metrics.items():
            record[key] = value
        records.append(record)

    for path in root.rglob("evaluation/*.json"):
        payload = load_json(path)
        metrics = payload.get("metrics", {})
        record = {
            "source": str(path),
            "record_type": "evaluation",
            "experiment_name": payload.get("experiment_name"),
            "run_name": payload.get("run_name"),
            "dataset_name": payload.get("dataset_name"),
            "variant_name": payload.get("variant_name"),
            "checkpoint": payload.get("checkpoint"),
            "seed": payload.get("seed"),
        }
        for key, value in metrics.items():
            record[key] = value
        records.append(record)
    return records


def _aggregate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in frame.columns if column not in {"source", "record_type", "experiment_name", "run_name", "dataset_name", "variant_name", "checkpoint", "seed"}]
    available = [column for column in metric_columns if pd.api.types.is_numeric_dtype(frame[column])]
    if not available:
        return frame

    grouped = frame.groupby(["record_type", "dataset_name", "variant_name"], dropna=False)[available]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=0).fillna(0.0).add_suffix("_std")
    count_df = grouped.size().rename("num_runs")
    return pd.concat([mean_df, std_df, count_df], axis=1).reset_index()


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No experiment records found."
    return frame.to_markdown(index=False)


def _to_latex_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "% No experiment records found."
    return frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = ensure_dir(args.output)

    records = _collect_records(root)
    frame = pd.DataFrame(records)
    summary = _aggregate_frame(frame) if not frame.empty else pd.DataFrame()

    raw_csv = output_dir / "aggregated_results.csv"
    summary_csv = output_dir / "aggregated_summary.csv"
    markdown_path = output_dir / "paper_table.md"
    latex_path = output_dir / "paper_table.tex"
    json_path = output_dir / "aggregated_summary.json"

    frame.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    markdown_path.write_text(_to_markdown_table(summary), encoding="utf-8")
    latex_path.write_text(_to_latex_table(summary), encoding="utf-8")
    save_json(json_path, {"records": records, "summary": summary.to_dict(orient="records")})

    print(f"Saved aggregated outputs to {output_dir}")


if __name__ == "__main__":
    main()
