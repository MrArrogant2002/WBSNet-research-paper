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
IDENTITY_COLUMNS = {"source", "record_type", "experiment_name", "run_name", "dataset_name", "variant_name", "checkpoint", "seed"}


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
            "variant_name": payload.get("variant_name"),
            "checkpoint": payload.get("checkpoint"),
            "seed": payload.get("seed"),
        }
        record.update(payload.get("metrics", {}))
        records.append(record)
    return records


def _paired_or_independent_test(frame_a: pd.DataFrame, frame_b: pd.DataFrame, metric: str) -> dict[str, Any]:
    data_a = frame_a[["seed", metric]].dropna()
    data_b = frame_b[["seed", metric]].dropna()
    if not data_a.empty and not data_b.empty:
        shared = sorted(set(data_a["seed"]) & set(data_b["seed"]))
        if len(shared) >= 2:
            paired_a = data_a.set_index("seed").loc[shared][metric].astype(float)
            paired_b = data_b.set_index("seed").loc[shared][metric].astype(float)
            statistic, p_value = stats.ttest_rel(paired_a.to_numpy(), paired_b.to_numpy())
            return {
                "test_type": "paired_t_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "n_compared": int(len(shared)),
            }

    values_a = frame_a[metric].dropna().astype(float)
    values_b = frame_b[metric].dropna().astype(float)
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
        raise ValueError(f"Reference variant '{reference}' was not found in the available variants: {ordered}")
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
    for (record_type, dataset_name), group in frame.groupby(["record_type", "dataset_name"], dropna=False):
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
    (output_dir / "significance_tests.md").write_text(result_frame.to_markdown(index=False), encoding="utf-8")
    save_json(output_dir / "significance_tests.json", {"results": result_frame.to_dict(orient="records")})
    print(f"Saved significance analysis to {output_dir}")


if __name__ == "__main__":
    main()
