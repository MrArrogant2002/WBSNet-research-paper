"""One-shot driver for the Option-A paper run.

Option A scope:
    * 7 ablation variants (A1-A7) on Kvasir-SEG, multiple seeds.
    * Full WBSNet on CVC-ClinicDB and ISIC2018, multiple seeds.
    * U-Net baselines on Kvasir, CVC-ClinicDB, ISIC2018.
    * Generalization eval: Kvasir checkpoint -> CVC-ColonDB.
    * Aggregation, significance tests, model complexity, figures.

Usage:
    python3 scripts/run_paper_optionA.py --seeds 3407 3408 3409
    python3 scripts/run_paper_optionA.py --seeds 3407 --skip ablations baselines  # eval only
    python3 scripts/run_paper_optionA.py --dry-run

The script is idempotent at the run-name level: completed runs (whose
``best.pt`` exists under ``outputs/<experiment>/<run>``) are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


ABLATION_CONFIGS: tuple[str, ...] = (
    "configs/ablation_identity_unet.yaml",
    "configs/kvasir_wbsnet.yaml",
    "configs/ablation_lfsa_only.yaml",
    "configs/ablation_hfba_only.yaml",
    "configs/ablation_no_boundary_supervision.yaml",
    "configs/ablation_no_wavelet.yaml",
    "configs/ablation_db2_wavelet.yaml",
)

MAIN_RESULTS_CONFIGS: tuple[str, ...] = (
    "configs/clinicdb_wbsnet.yaml",
    "configs/isic2018_wbsnet.yaml",
)

BASELINE_CONFIGS: tuple[str, ...] = (
    "configs/kvasir_unet_baseline.yaml",
    "configs/clinicdb_unet_baseline.yaml",
    "configs/isic2018_unet_baseline.yaml",
)

GENERALIZATION_CONFIG = "configs/kvasir_colondb_generalization.yaml"


@dataclass(frozen=True)
class Args:
    seeds: tuple[int, ...]
    skip: frozenset[str]
    dry_run: bool
    extra_overrides: tuple[str, ...]
    fail_fast: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[3407, 3408, 3409],
        help="Seeds to sweep across (default: 3407 3408 3409).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["ablations", "main_results", "baselines", "generalization", "aggregate", "significance", "complexity"],
        help="Stages to skip.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command plan without executing.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Extra train.py overrides applied to every training run (e.g. train.epochs=150 runtime.wandb.mode=offline).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failure (default: continue and report at the end).",
    )
    namespace = parser.parse_args()
    return Args(
        seeds=tuple(namespace.seeds),
        skip=frozenset(namespace.skip),
        dry_run=namespace.dry_run,
        extra_overrides=tuple(namespace.override),
        fail_fast=namespace.fail_fast,
    )


def _load_experiment_name(config_path: Path) -> str:
    sys.path.insert(0, str(ROOT))
    from wbsnet.config import load_config

    config = load_config(config_path)
    return config["experiment"]["name"]


def _load_run_config(config_path: str, seed: int, extra: tuple[str, ...], run_name: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from wbsnet.config import load_config

    overrides = (f"experiment.seed={seed}", *extra, f"experiment.run_name={run_name}")
    return load_config(ROOT / config_path, overrides)


def _has_best_checkpoint(experiment: str, run_name: str) -> bool:
    return (ROOT / "outputs" / experiment / run_name / "checkpoints" / "best.pt").exists()


def _best_checkpoint(experiment: str, run_name: str) -> Path:
    return ROOT / "outputs" / experiment / run_name / "checkpoints" / "best.pt"


def _preferred_eval_split(config: dict) -> str:
    dataset = config["dataset"]
    if dataset.get("split_strategy") != "pre_split_dirs":
        return "test"

    root = Path(dataset["root"])
    if not root.is_absolute():
        root = ROOT / root
    if (root / "test").exists():
        return "test"
    if (root / "val").exists():
        return "val"
    return "all"


def _evaluation_json_path(experiment: str, run_name: str, dataset_name: str, split: str) -> Path:
    return ROOT / "outputs" / experiment / run_name / "evaluation" / f"{dataset_name}_{split}.json"


def _run(cmd: list[str], dry_run: bool) -> int:
    pretty = " ".join(cmd)
    print(f"\n>>> {pretty}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=ROOT)
    return completed.returncode


def _train_command(config: str, seed: int, extra: tuple[str, ...]) -> list[str]:
    cmd = [
        "python3",
        str(ROOT / "train.py"),
        "--config",
        str(ROOT / config),
        "--override",
        f"experiment.seed={seed}",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def _evaluate_trained_run(
    config: str,
    seed: int,
    extra: tuple[str, ...],
    run_name: str,
    dry_run: bool,
    fail_fast: bool,
    failures: list[str],
) -> None:
    run_config = _load_run_config(config, seed, extra, run_name)
    experiment = run_config["experiment"]["name"]
    checkpoint = _best_checkpoint(experiment, run_name)
    if not checkpoint.exists() and not dry_run:
        failures.append(f"evaluation: missing checkpoint for {run_name}")
        if fail_fast:
            raise SystemExit(failures[-1])
        return

    eval_split = _preferred_eval_split(run_config)
    dataset_name = run_config["dataset"]["name"]
    metrics_path = _evaluation_json_path(experiment, run_name, dataset_name, eval_split)
    if not dry_run and metrics_path.exists():
        print(f"[skip] evaluation {run_name} ({dataset_name}/{eval_split})")
        return

    cmd = [
        "python3",
        str(ROOT / "evaluate.py"),
        "--config",
        str(ROOT / config),
        "--checkpoint",
        str(checkpoint),
        "--split",
        eval_split,
        "--override",
        f"experiment.seed={seed}",
        *extra,
        f"experiment.run_name={run_name}",
    ]
    rc = _run(cmd, dry_run)
    if rc != 0:
        failures.append(f"evaluation: {config} seed={seed} split={eval_split}")
        if fail_fast:
            raise SystemExit(failures[-1])


def _train_runs(
    label: str,
    configs: tuple[str, ...],
    seeds: tuple[int, ...],
    extra: tuple[str, ...],
    dry_run: bool,
    fail_fast: bool,
    failures: list[str],
) -> None:
    print(f"\n========== {label} ({len(configs)} configs x {len(seeds)} seeds = {len(configs) * len(seeds)} runs) ==========")
    for config in configs:
        experiment = _load_experiment_name(Path(config))
        for seed in seeds:
            run_name = f"{experiment}_seed{seed}"
            if not dry_run and _has_best_checkpoint(experiment, run_name):
                print(f"[skip] {run_name} (best.pt already exists)")
            else:
                cmd = _train_command(config, seed, extra + (f"experiment.run_name={run_name}",))
                rc = _run(cmd, dry_run)
                if rc != 0:
                    failures.append(f"{label}: {config} seed={seed}")
                    if fail_fast:
                        raise SystemExit(f"Aborting on first failure: {failures[-1]}")
                    continue
            _evaluate_trained_run(config, seed, extra, run_name, dry_run, fail_fast, failures)


def _generalization_eval(args: Args, failures: list[str]) -> None:
    print("\n========== GENERALIZATION (Kvasir -> CVC-ColonDB, eval-only) ==========")
    kvasir_experiment = _load_experiment_name(Path("configs/kvasir_wbsnet.yaml"))
    for seed in args.seeds:
        run_name = f"{kvasir_experiment}_seed{seed}"
        ckpt = ROOT / "outputs" / kvasir_experiment / run_name / "checkpoints" / "best.pt"
        if not ckpt.exists() and not args.dry_run:
            msg = f"missing checkpoint for {run_name}; skipping ColonDB generalization"
            print(f"[warn] {msg}")
            failures.append(f"generalization: {msg}")
            continue
        metrics_path = ROOT / "outputs" / kvasir_experiment / run_name / "evaluation" / "CVC-ColonDB_all.json"
        if not args.dry_run and metrics_path.exists():
            print(f"[skip] generalization {run_name} (CVC-ColonDB/all)")
            continue
        cmd = [
            "python3",
            str(ROOT / "evaluate.py"),
            "--config",
            str(ROOT / GENERALIZATION_CONFIG),
            "--checkpoint",
            str(ckpt),
            "--split",
            "all",
            "--override",
            *args.extra_overrides,
            f"experiment.seed={seed}",
            f"experiment.run_name={run_name}",
        ]
        rc = _run(cmd, args.dry_run)
        if rc != 0:
            failures.append(f"generalization: seed={seed}")
            if args.fail_fast:
                raise SystemExit(failures[-1])


def _post_processing(args: Args, failures: list[str]) -> None:
    if "aggregate" not in args.skip:
        rc = _run(
            [
                "python3",
                str(ROOT / "aggregate_results.py"),
                "--root",
                "outputs",
                "--output",
                "outputs/aggregated",
            ],
            args.dry_run,
        )
        if rc != 0:
            failures.append("aggregate_results")

    if "significance" not in args.skip:
        rc = _run(
            [
                "python3",
                str(ROOT / "scripts/significance_tests.py"),
                "--root",
                "outputs",
                "--output",
                "outputs/significance",
                "--record-type",
                "evaluation",
                "--reference",
                "A1_identity_unet",
            ],
            args.dry_run,
        )
        if rc != 0:
            failures.append("significance_tests")

    if "complexity" not in args.skip:
        rc = _run(
            ["python3", str(ROOT / "scripts/model_complexity.py"), "--output", "outputs/model_complexity"],
            args.dry_run,
        )
        if rc != 0:
            failures.append("model_complexity")


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    if "ablations" not in args.skip:
        _train_runs("KVASIR ABLATIONS (A1-A7)", ABLATION_CONFIGS, args.seeds, args.extra_overrides, args.dry_run, args.fail_fast, failures)

    if "main_results" not in args.skip:
        _train_runs("MAIN RESULTS (ClinicDB, ISIC2018)", MAIN_RESULTS_CONFIGS, args.seeds, args.extra_overrides, args.dry_run, args.fail_fast, failures)

    if "baselines" not in args.skip:
        _train_runs("U-NET BASELINES", BASELINE_CONFIGS, args.seeds, args.extra_overrides, args.dry_run, args.fail_fast, failures)

    if "generalization" not in args.skip:
        _generalization_eval(args, failures)

    _post_processing(args, failures)

    print("\n========== SUMMARY ==========")
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("All Option-A stages completed successfully.")


if __name__ == "__main__":
    main()
