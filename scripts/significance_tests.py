from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbsnet.utils.io import ensure_dir, load_json, save_json

LOWER_IS_BETTER = {"hd95", "loss", "segmentation_loss", "boundary_loss", "total_loss"}
IDENTITY_COLUMNS = {
    "source",
    "record_type",
    "experiment_name",
    "run_name",
    "dataset_name",
    "split",
    "variant_name",
    "checkpoint",
    "checkpoint_experiment_name",
    "checkpoint_run_name",
    "checkpoint_seed",
    "seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run significance tests across WBSNet experiment outputs.")
    parser.add_argument("--root", required=True, help="Root output directory to scan.")
    parser.add_argument("--output", required=True, help="Directory to save significance outputs.")
    parser.add_argument("--record-type", default="evaluation", choices=["evaluation", "run_summary", "all"])
    parser.add_argument("--metrics", nargs="*", default=["dice", "iou", "precision", "recall", "accuracy", "specificity", "hd95"])
    parser.add_argument("--reference", default=None, help="Optional reference variant. If omitted, all pairwise tests are run.")
    return parser.parse_args()


def _collect_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in root.rglob("run_summary.json"):
        payload = load_json(path)
        record = {
            "source": str(path),
            "record_type": "run_summary",
            "experiment_name": payload.get("experiment_name"),
            "run_name": payload.get("run_name"),
            "dataset_name": payload.get("dataset_name"),
            "variant_name": payload.get("variant_name"),
            "seed": payload.get("seed"),
        }
        record.update(payload.get("best_metrics", {}))
        records.append(record)

    for path in root.rglob("evaluation/*.json"):
        payload = load_json(path)
        record = {
            "source": str(path),
            "record_type": "evaluation",
            "experiment_name": payload.get("experiment_name"),
            "run_name": payload.get("run_name"),
            "dataset_name": payload.get("dataset_name"),
            "split": payload.get("split"),
            "variant_name": payload.get("variant_name"),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_experiment_name": payload.get("checkpoint_experiment_name"),
            "checkpoint_run_name": payload.get("checkpoint_run_name"),
            "checkpoint_seed": payload.get("checkpoint_seed"),
            "seed": payload.get("seed"),
        }
        record.update(payload.get("metrics", {}))
        records.append(record)
    return records


def _metric_by_seed(frame: pd.DataFrame, metric: str) -> pd.Series:
    data = frame[["seed", metric]].dropna(subset=["seed", metric]).copy()
    if data.empty:
        return pd.Series(dtype="float64")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])
    if data.empty:
        return pd.Series(dtype="float64")
    # Some runs can produce multiple evaluation records for the same seed.
    # Collapse those records before paired tests so each seed contributes once.
    return data.groupby("seed", sort=True)[metric].mean()


def _paired_or_independent_test(frame_a: pd.DataFrame, frame_b: pd.DataFrame, metric: str) -> dict[str, Any]:
    data_a = _metric_by_seed(frame_a, metric)
    data_b = _metric_by_seed(frame_b, metric)
    if not data_a.empty and not data_b.empty:
        shared = sorted(set(data_a.index) & set(data_b.index))
        if len(shared) >= 2:
            paired_a = data_a.loc[shared].astype(float)
            paired_b = data_b.loc[shared].astype(float)
            statistic, p_value = stats.ttest_rel(paired_a.to_numpy(), paired_b.to_numpy())
            return {
                "test_type": "paired_t_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "n_compared": int(len(shared)),
            }

    values_a = data_a.dropna().astype(float)
    values_b = data_b.dropna().astype(float)
    if len(values_a) >= 2 and len(values_b) >= 2:
        statistic, p_value = stats.ttest_ind(values_a.to_numpy(), values_b.to_numpy(), equal_var=False)
        return {
            "test_type": "welch_t_test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_compared": int(min(len(values_a), len(values_b))),
        }

    return {
        "test_type": "insufficient_samples",
        "statistic": math.nan,
        "p_value": math.nan,
        "n_compared": int(min(len(values_a), len(values_b))),
    }


def _select_comparisons(variants: list[str], reference: str | None) -> list[tuple[str, str]]:
    ordered = sorted(set(variants))
    if reference is None:
        return list(itertools.combinations(ordered, 2))
    if reference not in ordered:
        return []
    return [(reference, variant) for variant in ordered if variant != reference]


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = ensure_dir(args.output)

    frame = pd.DataFrame(_collect_records(root))
    if frame.empty:
        raise RuntimeError(f"No experiment records were found under {root}")

    if args.record_type != "all":
        frame = frame[frame["record_type"] == args.record_type].copy()

    metrics = [metric for metric in args.metrics if metric in frame.columns]
    if not metrics:
        raise RuntimeError("None of the requested metrics were present in the collected records.")

    results: list[dict[str, Any]] = []
    for (record_type, dataset_name, split), group in frame.groupby(["record_type", "dataset_name", "split"], dropna=False):
        variants = sorted(group["variant_name"].dropna().unique().tolist())
        for variant_a, variant_b in _select_comparisons(variants, args.reference):
            frame_a = group[group["variant_name"] == variant_a]
            frame_b = group[group["variant_name"] == variant_b]
            for metric in metrics:
                test_result = _paired_or_independent_test(frame_a, frame_b, metric)
                mean_a = float(frame_a[metric].dropna().mean()) if metric in frame_a else math.nan
                mean_b = float(frame_b[metric].dropna().mean()) if metric in frame_b else math.nan
                better_variant = variant_a if ((metric in LOWER_IS_BETTER and mean_a < mean_b) or (metric not in LOWER_IS_BETTER and mean_a > mean_b)) else variant_b
                results.append(
                    {
                        "record_type": record_type,
                        "dataset_name": dataset_name,
                        "split": split,
                        "metric": metric,
                        "variant_a": variant_a,
                        "variant_b": variant_b,
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "mean_diff_a_minus_b": float(mean_a - mean_b),
                        "better_variant": better_variant,
                        "significant_0_05": bool(test_result["p_value"] < 0.05) if not math.isnan(test_result["p_value"]) else False,
                        **test_result,
                    }
                )

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_dir / "significance_tests.csv", index=False)
    try:
        markdown = result_frame.to_markdown(index=False)
    except ImportError:
        markdown = result_frame.to_csv(index=False)
    (output_dir / "significance_tests.md").write_text(markdown, encoding="utf-8")
    save_json(output_dir / "significance_tests.json", {"results": result_frame.to_dict(orient="records")})
    print(f"Saved significance analysis to {output_dir}")


if __name__ == "__main__":
    main()
