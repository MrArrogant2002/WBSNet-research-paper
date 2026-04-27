# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WBSNet (Wavelet Boundary Skip Network) is a PyTorch research codebase for an IEEE journal paper on medical image segmentation. The paper lives in `paper/paper.tex`. The primary compute platform is Kaggle (GPU notebooks) or a DGX server. All experiments run via `WBSNet_Model.ipynb` on Kaggle, or via CLI scripts on DGX/local.

---

## Common Commands

### Environment setup
```bash
bash scripts/setup_dgx.sh wbsnet   # creates conda env 'wbsnet'
conda activate wbsnet
pip install -e .                    # editable install of the wbsnet package
```

### Verify repo integrity (runs structural + runtime checks)
```bash
python3 scripts/verify_repo.py
python3 scripts/check_system.py    # saves GPU/package report to artifacts/system_report.json
```

### Train
```bash
# Single GPU (local)
python train.py --config configs/kvasir_wbsnet.yaml

# Multi-GPU on DGX
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8

# 1-epoch smoke test (no wandb upload)
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 1 \
  --override train.epochs=1 train.batch_size=2 runtime.wandb.mode=offline

# Config overrides from CLI (dot-notation)
python train.py --config configs/kvasir_wbsnet.yaml \
  --override train.epochs=5 dataset.root=/path/to/data
```

### Evaluate, predict, aggregate
```bash
python evaluate.py --config configs/kvasir_wbsnet.yaml \
  --checkpoint outputs/kvasir_wbsnet/<run_name>/checkpoints/best.pt

python predict.py --config configs/kvasir_wbsnet.yaml \
  --checkpoint outputs/.../checkpoints/best.pt --split test

python aggregate_results.py --root outputs --output outputs/aggregated

python scripts/run_ablation_suite.py --seeds 3407 3408 3409
python scripts/significance_tests.py --root outputs --output outputs/significance \
  --record-type evaluation --reference A1_identity_unet
python scripts/model_complexity.py --output outputs/model_complexity
```

### Slurm (if the host scheduler is available)
```bash
sbatch scripts/slurm_train.sh configs/kvasir_wbsnet.yaml
```

### Paper qualitative figures
```bash
python scripts/make_paper_figures.py --input-dir outputs/.../predictions --limit 8 --columns 2
```

### Build the paper PDF
```bash
cd paper && make
```

### Tests
There is no automated test suite in this repo. `scripts/verify_repo.py` is the closest equivalent — it performs structural checks plus optional runtime smoke checks when `torch` is importable. Do not fabricate `pytest`/`unittest` commands.

---

## Architecture

### Model: WBSNet

**Entry point:** `wbsnet/models/wbsnet.py` — `WBSNet` class; `build_model(config)` is the factory.

**Encoder:** `wbsnet/models/resnet.py` — `ResNetEncoder` wraps ResNet-34 stages. Outputs five tensors: `(stem, layer1, layer2, layer3, bottleneck)`. ImageNet pretrained weights load via `load_imagenet_pretrained()`, which remaps torchvision keys (`conv1.*`, `bn1.*`) to internal `stem.*` keys.

**Skip modules:** `wbsnet/models/wbs_module.py` — `WBSModule` applied at all four skip levels. Each module:
1. Applies `WaveletTransform2d` (DWT) → 4 subbands: LL, LH, HL, HH
2. `LFSA` (Low-Freq Semantic Attention) on LL subband — SE-style channel attention (`wbsnet/models/lfsa.py`)
3. `HFBA` (High-Freq Boundary Attention) fuses LH+HL+HH → produces a boundary logit map and a gate; gate is applied back to each HF subband separately (not to fused output)
4. `InverseWaveletTransform2d` (IDWT) reconstructs the skip feature
5. Returns `(refined_skip, boundary_logit_or_None)`

When `use_wavelet=False`, a lightweight `RawAttentionSkip` is used instead (LFSA channel attention + spatial boundary gate).

**Decoder:** `wbsnet/models/decoder.py` — four `DecoderBlock` stages + final `Conv2d(32→1)` head + 2× bilinear upsample.

**Wavelet implementation:** `wbsnet/models/wavelet.py` — Pure PyTorch grouped-stride-2 convolution (no external wavelet library at runtime). Filters are built once at `__init__` and stored as `register_buffer` (AMP-safe; dtype-cast at forward time).

### Loss

`wbsnet/losses.py` — `total_loss()`:
- `segmentation_loss`: BCE + Dice on main logits
- `boundary_loss`: per-level BCE on boundary logits, with logits **upsampled** (bilinear) to GT resolution (not the other way around — this preserves the 1-pixel-wide GT edge signal)
- Combined: `total = seg + λ * boundary`; λ = `train.boundary_loss_weight` (default 0.5)

### Metrics

`wbsnet/metrics.py` — `BinarySegmentationMeter` accumulates global TP/FP/FN/TN for Dice, IoU, Precision, Recall, Accuracy, Specificity. HD95 is **disabled during training** (`compute_hd95=False if training else ...`) and only computed in final eval runs.

> **Important:** `compute_hd95` is `false` in `configs/default.yaml` on purpose. Enable it only in dataset-specific eval configs (e.g. `configs/kvasir_wbsnet.yaml`).

### Config system

`wbsnet/config.py` — YAML with inheritance via `base_config:` key. All configs inherit from `configs/default.yaml`. Override at CLI with `--override key.nested=value`.

Key model knobs: `model.use_wavelet`, `model.use_lfsa`, `model.use_hfba`, `model.boundary_supervision`, `model.wavelet_type`.

### Ablation variants (A1–A7)

| ID | Config | Description |
|----|--------|-------------|
| A1 | `ablation_identity_unet.yaml` | No wavelet, no LFSA, no HFBA — plain U-Net |
| A2 | `kvasir_wbsnet.yaml` | Full WBSNet (haar) |
| A3 | `ablation_lfsa_only.yaml` | Wavelet + LFSA, no HFBA/boundary |
| A4 | `ablation_hfba_only.yaml` | Wavelet + HFBA, no LFSA |
| A5 | `ablation_no_boundary_supervision.yaml` | Full WBSNet, boundary loss disabled |
| A6 | `ablation_no_wavelet.yaml` | LFSA+HFBA but no wavelet decomposition |
| A7 | `ablation_db2_wavelet.yaml` | Full WBSNet with db2 wavelet |

`variant_name_from_config()` in `wbsnet/models/__init__.py` infers variant ID from config flags.

### Baseline and generalization configs (outside A1–A7)

| Config | Purpose |
|--------|---------|
| `kvasir_unet_baseline.yaml`, `clinicdb_unet_baseline.yaml`, `isic2018_unet_baseline.yaml` | Vanilla ResNet-34 U-Net baselines for main-results tables |
| `clinicdb_wbsnet.yaml`, `isic2018_wbsnet.yaml` | Full WBSNet on CVC-ClinicDB and ISIC2018 |
| `kvasir_colondb_generalization.yaml` | Kvasir-trained checkpoint evaluated on full CVC-ColonDB |
| `kvasir_colondb_generalization_baseline.yaml` | Baseline U-Net counterpart for the same generalization protocol |

For generalization runs, evaluate with `--split all` so every CVC-ColonDB sample is scored (the config does not define an internal train/val/test split for it).

### Data pipeline

`wbsnet/data/datasets.py` — `discover_samples()` pairs images and masks by **stem name**. Supports `split_strategy: ratio` (random) or `split_strategy: predefined` (from text files). Dataset classes: `PolyDataset` (Kvasir/CVC), `ISICDataset` (`wbsnet/data/isic_dataset.py`), `PolyDataset` (`wbsnet/data/polyp_dataset.py`). `build_dataloaders()` in `wbsnet/data/__init__.py` dispatches by `dataset.name`.

**Kaggle dataset path:** Preprocessed data lives at `/kaggle/input/WBSNet_Dataset/` with subdirs `kvasir/`, `cvc_clinicdb/`, `cvc_colondb/`, `isic2018/` each containing `images/`, `masks/`, `boundaries/`. The notebook (`WBSNet_Model.ipynb`) auto-detects the root via `autodetect_processed_root()`.

### Training loop

`wbsnet/engine.py` — `run_epoch()` handles both train and eval. Key facts:
- Optimizer: AdamW with separate encoder/decoder LR groups (`encoder_lr=1e-4`, `decoder_lr=1e-3`)
- Scheduler: `CosineAnnealingLR` stepped once per epoch after both train and val
- AMP: `torch.amp.GradScaler` (not `torch.cuda.amp` — that import is deprecated)
- `GradScaler` is instantiated in `train.py` and passed through; `engine.py` also imports from `torch.amp`

### Distributed training

`wbsnet/utils/distributed.py` — `init_distributed()` / `cleanup_distributed()` wrap `torch.distributed`. Uses `torchrun` via `scripts/train_dgx.sh`. `sync_batchnorm` opt-in via config.

### W&B logging

`wbsnet/utils/logger.py` — `ExperimentLogger`. Accepts both `WANDB_API_KEY` and legacy `WAND_API_KEY` (auto-remapped) and loads them from a repo-root `.env` if present. Disable with `runtime.wandb.mode=offline` or `runtime.wandb.enabled=false`.

### Run outputs

Each run writes to `outputs/<experiment_name>/<run_name>/`:
- `metrics.csv` — per-epoch train/val metrics
- `best_metrics.json` — best val snapshot
- `run_summary.json` — compact run metadata
- `checkpoints/best.pt`, `last.pt`, `epoch_NNN.pt`
- `evaluation/<dataset>.json` — final eval metrics
- `predictions/` — overlays, paper panels, contact sheets

`aggregate_results.py` scans all `run_summary.json` files and emits `aggregated_results.csv/json` with per-variant mean ± std (using `ddof=1`).

---

## Kaggle Notebook (WBSNet_Model.ipynb)

Self-contained single-file experiment runner for Kaggle. Runs all 7 variants × 3 seeds × 4 datasets. Key cells:

- **Cell 3** — `RESEARCH_CONFIG`, `VARIANT_SPECS`, `DATASET_SPECS`, `PAPER_REPORTED_MAIN` constants; auto-detects `PROCESSED_ROOT`
- **Cell 7** — `WBSModule`, `LFSA`, `HFBA`, wavelet classes (inline)
- **Cell 8** — `ResNetEncoder` with stem key remapping for ImageNet pretrain
- **Cell 10** — `BinarySegmentationMeter`, `boundary_loss`, training utilities
- **Cell 14** — Single-run sanity check (gated by `RUN_SINGLE_EXPERIMENT = False`)
- **Cell 16** — `run_paper_suite()` — launches all experiments
- **Cell 18** — Lambda sweep for boundary loss sensitivity analysis
- **Cell 20** — Result aggregation and paper tables
- **Cell 22** — Complexity, training dynamics, subband visualizations

To run on Kaggle: attach `WBSNet_Dataset` as input, enable GPU accelerator, set `RUN_SINGLE_EXPERIMENT = True` first for smoke test, then call `run_paper_suite(...)`.
