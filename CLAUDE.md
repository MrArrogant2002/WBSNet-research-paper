# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WBSNet (Wavelet Boundary Skip Network) is a PyTorch research codebase for an IEEE journal paper on medical image segmentation. The primary compute platform is Google Colab A100 or a DGX/local server. Colab uses `WBSNet_Colab.ipynb` only as a launcher; experiments run through the Python CLI scripts.

**Paper status:** `paper/paper.tex` is a full IEEEtran journal draft (all sections written: Abstract → Introduction → Related Work → Method → Experiments → Discussion → Conclusion). All result tables have `TODO` placeholders to be filled once experiments complete. `paper/paper-outline.md` is the human-readable section plan and citation checklist. `paper/references.bib` holds all 42 cite keys.

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
# Single GPU (Colab/local)
python train.py --config configs/kvasir_wbsnet.yaml

# Colab A100 smoke test using processed split folders on Google Drive
python train.py --config configs/kvasir_wbsnet.yaml \
  --override dataset.root=/content/drive/MyDrive/WBSNet_Dataset/kvasir \
  dataset.split_strategy=pre_split_dirs dataset.num_workers=2 train.epochs=1 train.batch_size=8 \
  runtime.device=cuda runtime.amp=true runtime.wandb.mode=offline

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

### Paper figures
```bash
# Qualitative prediction grid (Fig. 3 in paper)
python scripts/make_paper_figures.py --input-dir outputs/.../predictions --limit 8 --columns 2

# Lambda sensitivity curve (Fig. 4 in paper) — input is aggregated CSV or plain lambda,dice,hd95 CSV
python scripts/plot_lambda_sweep.py --input outputs/lambda_sweep.csv --output paper/figures/lambda_sensitivity.png
```

### Import legacy Kaggle / Colab seed-3407 runs
If seed-3407 artifacts exist from a prior Kaggle notebook run under
`wbsnet_paper_runs/paper_suite/<dataset>/<variant>/seed_3407/`, copy them into
the current output layout (skips re-training for already-completed seeds):
```bash
python scripts/import_legacy_paper_runs.py --legacy-root wbsnet_paper_runs/paper_suite \
  --output-root outputs
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

> **Important:** `compute_hd95` is `false` in `configs/default.yaml` on purpose. Training keeps HD95 disabled for speed; `evaluate.py` enables final HD95 by default unless overridden.

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

**Colab processed dataset path:** Put the processed dataset on Google Drive, for example `/content/drive/MyDrive/WBSNet_Dataset/`, with subdirs `kvasir/`, `cvc_clinicdb/`, `cvc_colondb/`, `isic2018/`. Use `dataset.split_strategy=pre_split_dirs` for split folders such as `train/images`, `train/masks`, `train/boundaries`, `val/...`, and `test/...`.

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
- `evaluation/<dataset>_<split>.json` — final eval metrics
- `predictions/` — overlays, paper panels, contact sheets

`aggregate_results.py` scans all `run_summary.json` files and emits `aggregated_results.csv/json` with per-variant mean ± std (using `ddof=1`).

---

## Colab A100 Python Workflow

`WBSNet_Colab.ipynb` is a thin control notebook for Google Colab. It mounts Drive, installs the editable package, verifies the runtime, and calls the Python scripts:

- `train.py` for baseline and WBSNet training
- `evaluate.py` for split-aware evaluation
- `aggregate_results.py` for comparison tables
- `scripts/run_paper_optionA.py` for the full baseline/proposed/ablation suite (A1–A7 across all datasets)
- `scripts/significance_tests.py` and `scripts/model_complexity.py` for paper evidence artifacts
- `scripts/import_legacy_paper_runs.py` to promote prior Kaggle seed-3407 artifacts into the current output layout before running the suite (avoids re-training completed seeds)
- `scripts/plot_lambda_sweep.py` to generate the lambda sensitivity figure once a sweep CSV exists

Keep the full experimental evidence in `outputs/`, `results/`, `reports/iteration/`, and `experiments/runs/`; do not write paper claims until those artifacts exist.

### Paper TODO checklist
The following must be completed before the paper can be submitted:
1. Run `scripts/run_paper_optionA.py` (or `run_ablation_suite.py`) for all variants and datasets
2. Run `aggregate_results.py` → fill tables in `paper/paper.tex`
3. Run `scripts/significance_tests.py` → confirm all ablation comparisons reach p < 0.05
4. Run `scripts/model_complexity.py` → fill complexity table
5. Run `scripts/make_paper_figures.py` → replace qualitative figure placeholder
6. Run lambda sweep + `scripts/plot_lambda_sweep.py` → replace lambda figure placeholder
7. Create WBS module block diagram (TikZ) for `paper/figures/wbs_module.*`
8. Final `/paper-polish` pass once all numbers are in
