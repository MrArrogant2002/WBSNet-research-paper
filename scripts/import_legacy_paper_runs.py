"""Import legacy Kaggle paper-run artifacts into the current output layout.

The Kaggle notebook saved seed-3407 runs under:

    wbsnet_paper_runs/paper_suite/<dataset>/<variant>/seed_3407/<run_name>/

The current script pipeline skips completed work only when it finds:

    outputs/<experiment>/<experiment>_seed<seed>/checkpoints/best.pt

This importer copies the usable legacy artifacts into that layout and adapts
the small checkpoint naming difference introduced when the encoder stem was
wrapped in a module. It does not delete or mutate the original legacy folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbsnet.config import load_config
from wbsnet.models import build_model, variant_name_from_config
from wbsnet.utils.io import ensure_dir, save_json


@dataclass(frozen=True)
class ImportSpec:
    legacy_dataset: str
    legacy_variant: str
    config_path: str
    experiment_name: str
    run_name: str
    note: str


SEED_3407_IMPORTS: tuple[ImportSpec, ...] = (
    ImportSpec("kvasir", "A1", "configs/ablation_identity_unet.yaml", "kvasir_a1_identity_unet", "kvasir_a1_identity_unet_seed3407", "Kvasir ablation A1"),
    ImportSpec("kvasir", "A2", "configs/kvasir_wbsnet.yaml", "kvasir_wbsnet", "kvasir_wbsnet_seed3407", "Kvasir full WBSNet"),
    ImportSpec("kvasir", "A3", "configs/ablation_lfsa_only.yaml", "kvasir_a3_lfsa_only", "kvasir_a3_lfsa_only_seed3407", "Kvasir ablation A3"),
    ImportSpec("kvasir", "A4", "configs/ablation_hfba_only.yaml", "kvasir_a4_hfba_only", "kvasir_a4_hfba_only_seed3407", "Kvasir ablation A4"),
    ImportSpec("kvasir", "A5", "configs/ablation_no_boundary_supervision.yaml", "kvasir_a5_no_boundary_supervision", "kvasir_a5_no_boundary_supervision_seed3407", "Kvasir ablation A5"),
    ImportSpec("kvasir", "A6", "configs/ablation_no_wavelet.yaml", "kvasir_a6_no_wavelet", "kvasir_a6_no_wavelet_seed3407", "Kvasir ablation A6"),
    ImportSpec("kvasir", "A7", "configs/ablation_db2_wavelet.yaml", "kvasir_a7_db2_wavelet", "kvasir_a7_db2_wavelet_seed3407", "Kvasir ablation A7"),
    ImportSpec("kvasir", "A1", "configs/kvasir_unet_baseline.yaml", "kvasir_unet_baseline", "kvasir_unet_baseline_seed3407", "Kvasir U-Net baseline alias of A1"),
    ImportSpec("cvc_clinicdb", "A1", "configs/clinicdb_unet_baseline.yaml", "clinicdb_unet_baseline", "clinicdb_unet_baseline_seed3407", "ClinicDB U-Net baseline"),
    ImportSpec("cvc_clinicdb", "A2", "configs/clinicdb_wbsnet.yaml", "clinicdb_wbsnet", "clinicdb_wbsnet_seed3407", "ClinicDB full WBSNet"),
    ImportSpec("isic2018", "A1", "configs/isic2018_unet_baseline.yaml", "isic2018_unet_baseline", "isic2018_unet_baseline_seed3407", "ISIC2018 U-Net baseline"),
)


EVAL_DATASET_NAMES = {
    "kvasir": "Kvasir-SEG",
    "cvc_clinicdb": "CVC-ClinicDB",
    "cvc_colondb": "CVC-ColonDB",
    "isic2018": "ISIC2018",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-root", default="wbsnet_paper_runs", help="Root of the legacy Kaggle artifact folder.")
    parser.add_argument("--output-root", default="outputs", help="Current output root to import into.")
    parser.add_argument("--seed", type=int, default=3407, choices=[3407], help="Legacy seed to import. Only 3407 is supported.")
    parser.add_argument("--overwrite", action="store_true", help="Replace destination files if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the import plan without writing files.")
    parser.add_argument("--verify-forward", action="store_true", help="Run one CPU forward pass after adapting each checkpoint.")
    return parser.parse_args()


def _legacy_run_dir(legacy_root: Path, spec: ImportSpec, seed: int) -> Path:
    legacy_run_name = f"{spec.legacy_dataset}_{spec.legacy_variant}_seed{seed}"
    return legacy_root / "paper_suite" / spec.legacy_dataset / spec.legacy_variant / f"seed_{seed}" / legacy_run_name


def _load_legacy_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _adapt_state_dict(old_state_dict: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    adapted: dict[str, torch.Tensor] = {}
    for key, value in old_state_dict.items():
        new_key = key
        if key == "encoder.conv1.weight":
            new_key = "encoder.stem.conv1.weight"
        elif key.startswith("encoder.bn1."):
            new_key = f"encoder.stem.bn1.{key.removeprefix('encoder.bn1.')}"
        adapted[new_key] = value

    current_state = model.state_dict()
    for key, value in current_state.items():
        if key.endswith(".dwt.dec_filters") or key.endswith(".idwt.rec_filters"):
            adapted.setdefault(key, value)
    return adapted


def _current_config(spec: ImportSpec, seed: int) -> dict[str, Any]:
    return load_config(
        ROOT / spec.config_path,
        overrides=[
            f"experiment.seed={seed}",
            f"experiment.run_name={spec.run_name}",
            "model.encoder_pretrained=false",
        ],
    )


def _write_checkpoint(
    *,
    source_checkpoint: Path,
    destination_checkpoint: Path,
    config: dict[str, Any],
    overwrite: bool,
    verify_forward: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if destination_checkpoint.exists() and not overwrite:
        return {"status": "skipped_existing", "path": str(destination_checkpoint)}

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Legacy checkpoint has no state_dict: {source_checkpoint}")

    model = build_model(config)
    adapted_state_dict = _adapt_state_dict(checkpoint["state_dict"], model)
    model.load_state_dict(adapted_state_dict, strict=True)
    if verify_forward:
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 352, 352))
        if tuple(output["logits"].shape) != (1, 1, 352, 352):
            raise RuntimeError(f"Unexpected forward output shape for {source_checkpoint}: {tuple(output['logits'].shape)}")

    payload = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_metric": float(checkpoint.get("best_metric", float("nan"))),
        "config": config,
        "state_dict": adapted_state_dict,
        "imported_from": str(source_checkpoint),
        "legacy_config": checkpoint.get("config", {}),
    }
    if not dry_run:
        ensure_dir(destination_checkpoint.parent)
        torch.save(payload, destination_checkpoint)
    return {"status": "written", "path": str(destination_checkpoint), "epoch": payload["epoch"], "best_metric": payload["best_metric"]}


def _copy_file(source: Path, destination: Path, *, overwrite: bool, dry_run: bool) -> str:
    if not source.exists():
        return "missing"
    if destination.exists() and not overwrite:
        return "skipped_existing"
    if not dry_run:
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
    return "written"


def _write_run_summary(
    *,
    destination_run_dir: Path,
    spec: ImportSpec,
    config: dict[str, Any],
    legacy_summary: dict[str, Any],
    checkpoint_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> str:
    destination = destination_run_dir / "run_summary.json"
    if destination.exists() and not overwrite:
        return "skipped_existing"
    best_metrics = legacy_summary.get("best_val_metrics") or _load_legacy_json(destination_run_dir / "best_metrics.json")
    summary = {
        "experiment_name": config["experiment"]["name"],
        "run_name": config["experiment"]["run_name"],
        "dataset_name": config["dataset"]["name"],
        "seed": config["experiment"]["seed"],
        "variant_name": variant_name_from_config(config),
        "best_metrics": best_metrics,
        "last_epoch": None,
        "imported_from_legacy": True,
        "legacy_run_name": legacy_summary.get("run_name"),
        "legacy_variant_id": legacy_summary.get("variant_id"),
        "legacy_paper_name": legacy_summary.get("paper_name"),
        "legacy_note": spec.note,
        "checkpoint": str(checkpoint_path),
    }
    if "params_total" in legacy_summary:
        summary["params_total"] = legacy_summary["params_total"]
    if "params_trainable" in legacy_summary:
        summary["params_trainable"] = legacy_summary["params_trainable"]
    if not dry_run:
        save_json(destination, summary)
    return "written"


def _import_evaluations(
    *,
    source_run_dir: Path,
    destination_run_dir: Path,
    spec: ImportSpec,
    config: dict[str, Any],
    checkpoint_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    evaluation_root = source_run_dir / "evaluation"
    if not evaluation_root.exists():
        return records

    for metrics_path in sorted(evaluation_root.glob("*/metrics.json")):
        legacy_payload = _load_legacy_json(metrics_path)
        legacy_eval_dataset = str(legacy_payload.get("eval_dataset", "")).lower()
        dataset_name = EVAL_DATASET_NAMES.get(legacy_eval_dataset, legacy_payload.get("eval_dataset", "unknown"))
        legacy_split = str(legacy_payload.get("split", "all"))
        current_split = "all" if legacy_eval_dataset == "cvc_colondb" else legacy_split
        destination_json = destination_run_dir / "evaluation" / f"{dataset_name}_{current_split}.json"
        status = "skipped_existing" if destination_json.exists() and not overwrite else "written"
        if status == "written" and not dry_run:
            payload = {
                "metrics": legacy_payload.get("metrics", {}),
                "dataset_name": dataset_name,
                "split": current_split,
                "variant_name": variant_name_from_config(config),
                "checkpoint": str(checkpoint_path),
                "experiment_name": config["experiment"]["name"],
                "run_name": config["experiment"]["run_name"],
                "seed": config["experiment"]["seed"],
                "checkpoint_experiment_name": config["experiment"]["name"],
                "checkpoint_run_name": config["experiment"]["run_name"],
                "checkpoint_seed": config["experiment"]["seed"],
                "imported_from_legacy": True,
                "legacy_metrics_path": str(metrics_path),
                "legacy_eval_dataset": legacy_payload.get("eval_dataset"),
                "legacy_split": legacy_split,
                "legacy_run_name": legacy_payload.get("run_name"),
            }
            save_json(destination_json, payload)

        sample_metrics = metrics_path.parent / "sample_metrics.csv"
        sample_destination = destination_run_dir / "evaluation" / f"{dataset_name}_{current_split}" / "sample_metrics.csv"
        sample_status = _copy_file(sample_metrics, sample_destination, overwrite=overwrite, dry_run=dry_run)
        records.append({"metrics": str(destination_json), "status": status, "sample_metrics": sample_status})
    return records


def _import_one(args: argparse.Namespace, spec: ImportSpec) -> dict[str, Any]:
    legacy_root = Path(args.legacy_root).resolve()
    output_root = Path(args.output_root).resolve()
    source_run_dir = _legacy_run_dir(legacy_root, spec, args.seed)
    source_checkpoint = source_run_dir / "checkpoints" / "best.pt"
    destination_run_dir = output_root / spec.experiment_name / spec.run_name
    destination_checkpoint = destination_run_dir / "checkpoints" / "best.pt"

    result: dict[str, Any] = {
        "source": str(source_run_dir),
        "destination": str(destination_run_dir),
        "experiment_name": spec.experiment_name,
        "run_name": spec.run_name,
        "note": spec.note,
    }
    if not source_checkpoint.exists():
        result["status"] = "missing_source_checkpoint"
        return result

    config = _current_config(spec, args.seed)
    checkpoint_result = _write_checkpoint(
        source_checkpoint=source_checkpoint,
        destination_checkpoint=destination_checkpoint,
        config=config,
        overwrite=args.overwrite,
        verify_forward=args.verify_forward,
        dry_run=args.dry_run,
    )
    legacy_summary = _load_legacy_json(source_run_dir / "run_summary.json")
    result["checkpoint"] = checkpoint_result
    result["metrics_csv"] = _copy_file(source_run_dir / "metrics.csv", destination_run_dir / "metrics.csv", overwrite=args.overwrite, dry_run=args.dry_run)
    result["best_metrics_json"] = _copy_file(source_run_dir / "best_metrics.json", destination_run_dir / "best_metrics.json", overwrite=args.overwrite, dry_run=args.dry_run)
    result["legacy_resolved_config"] = _copy_file(source_run_dir / "resolved_config.json", destination_run_dir / "legacy_resolved_config.json", overwrite=args.overwrite, dry_run=args.dry_run)
    result["run_summary"] = _write_run_summary(
        destination_run_dir=destination_run_dir,
        spec=spec,
        config=config,
        legacy_summary=legacy_summary,
        checkpoint_path=destination_checkpoint,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    result["evaluations"] = _import_evaluations(
        source_run_dir=source_run_dir,
        destination_run_dir=destination_run_dir,
        spec=spec,
        config=config,
        checkpoint_path=destination_checkpoint,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    result["status"] = "ok"
    return result


def main() -> None:
    args = parse_args()
    results = [_import_one(args, spec) for spec in SEED_3407_IMPORTS]
    imported = sum(1 for result in results if result.get("status") == "ok")
    missing = [result for result in results if result.get("status") != "ok"]

    manifest = {
        "legacy_root": str(Path(args.legacy_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "imported": imported,
        "missing": missing,
        "results": results,
    }
    if not args.dry_run:
        save_json(Path(args.output_root) / "legacy_seed3407_import_manifest.json", manifest)

    print(json.dumps(manifest, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
