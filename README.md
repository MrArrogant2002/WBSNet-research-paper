# WBSNet

Wavelet Boundary Skip Network for medical image segmentation.

This repository contains three related but intentionally separate deliverables:

- A local and DGX-friendly PyTorch codebase driven by YAML configs and the CLI entry points `train.py`, `evaluate.py`, and `predict.py`.
- A Kaggle-first research notebook, [`WBSNet_Model.ipynb`](WBSNet_Model.ipynb), designed for constrained notebook storage and session limits.
- The paper sources in [`paper/`](paper/) and supporting figures in [`diagrams/`](diagrams/).

## Workflow Split

This repo supports two execution paths. They are related, but they are not meant to be treated as the same runtime.

| Workflow | Primary file(s) | Best for | Config source | Data layout | Output style |
| --- | --- | --- | --- | --- | --- |
| Kaggle notebook | [`WBSNet_Model.ipynb`](WBSNet_Model.ipynb) | Kaggle sessions, paper sweeps, constrained output storage | Notebook config cells | Processed `WBSNet_Dataset/<dataset>/<split>/...` layout | Lean, storage-aware, notebook-managed |
| Local / DGX Python pipeline | [`train.py`](train.py), [`evaluate.py`](evaluate.py), [`predict.py`](predict.py) | Workstations, servers, DGX, repeatable script runs | `configs/*.yaml` | Raw dataset folders or split-driven layouts | Richer artifacts under `outputs/` |

Important:

- The Kaggle notebook is self-contained and does not read the YAML configs in `configs/`.
- The local `.py` pipeline is the main script-based training path and should be treated as the default for workstation or DGX runs.
- The notebook and the `.py` pipeline share the same research model family and metrics, but their runtime policies are intentionally different because Kaggle has tighter storage and session limits.

## What The Project Contains

| Path | Purpose |
| --- | --- |
| [`WBSNet_Model.ipynb`](WBSNet_Model.ipynb) | Full Kaggle notebook workflow for single runs, paper sweeps, evaluation, exports, and paper tables |
| [`data_preprocessing.ipynb`](data_preprocessing.ipynb) | Dataset preparation notebook |
| [`train.py`](train.py) | Local / DGX training entry point |
| [`evaluate.py`](evaluate.py) | Evaluate a trained checkpoint and export metrics / predictions |
| [`predict.py`](predict.py) | Save qualitative predictions from a trained checkpoint |
| [`aggregate_results.py`](aggregate_results.py) | Aggregate run summaries and evaluation outputs |
| [`configs/`](configs/) | YAML configs for the local / DGX Python pipeline |
| [`scripts/`](scripts/) | Helper scripts for verification, DGX launch, ablations, figures, significance tests, and complexity |
| [`wbsnet/`](wbsnet/) | Package code for models, losses, metrics, data loading, logging, and utilities |
| [`paper/`](paper/) | Manuscript sources |
| [`docs/`](docs/) | Project notes such as hardware planning |

## Model Variants

The paper workflow uses seven named variants:

| Variant | Meaning | Primary config |
| --- | --- | --- |
| `A1` | Identity Skip U-Net baseline | [`configs/ablation_identity_unet.yaml`](configs/ablation_identity_unet.yaml) |
| `A2` | Full WBSNet | [`configs/kvasir_wbsnet.yaml`](configs/kvasir_wbsnet.yaml) |
| `A3` | LFSA only | [`configs/ablation_lfsa_only.yaml`](configs/ablation_lfsa_only.yaml) |
| `A4` | HFBA only | [`configs/ablation_hfba_only.yaml`](configs/ablation_hfba_only.yaml) |
| `A5` | No boundary supervision | [`configs/ablation_no_boundary_supervision.yaml`](configs/ablation_no_boundary_supervision.yaml) |
| `A6` | No wavelet attention | [`configs/ablation_no_wavelet_attention.yaml`](configs/ablation_no_wavelet_attention.yaml) |
| `A7` | `db2` wavelet variant | [`configs/ablation_db2_wavelet.yaml`](configs/ablation_db2_wavelet.yaml) |

Dataset-specific full-model and baseline configs are also provided for Kvasir, CVC-ClinicDB, ISIC2018, and Kvasir-to-ColonDB evaluation.

## Outputs

### Local / DGX Python pipeline

The script-based pipeline writes into `outputs/` by default.

Typical run artifacts:

- `metrics.csv`
- `best_metrics.json`
- `run_summary.json`
- `checkpoints/best.pt`
- `checkpoints/epoch_*.pt` when periodic checkpointing is enabled
- `checkpoints/last.pt` when `train.save_last_checkpoint=true`
- `evaluation/<dataset>.json`
- `predictions/` with masks, overlays, paper panels, and contact sheets

### Kaggle notebook

The notebook writes into `/kaggle/working/wbsnet_paper_runs` and is intentionally more conservative about storage:

- lightweight best-checkpoint handling
- no periodic checkpoint flood by default
- reduced qualitative export pressure
- W&B used for keeping metrics remotely

If you are on Kaggle, treat the notebook as the primary interface and let it manage its own outputs.

## Hardware Guidance

Short version:

- Local minimum: `1 x 12 GB GPU`, `16 GB RAM`, and enough free disk for checkpoints and predictions
- Better local experience: `1 x 24 GB GPU`, `32 GB RAM`
- Ideal for the full paper suite: DGX or another multi-GPU server

More detail is in [docs/HARDWARE_REQUIREMENTS.md](docs/HARDWARE_REQUIREMENTS.md).

## Datasets

Recommended download order:

1. `Kvasir-SEG`
2. `CVC-ClinicDB`
3. `CVC-ColonDB`
4. `ISIC2018`

How they are used:

- `Kvasir-SEG`: first smoke test and primary polyp benchmark
- `CVC-ClinicDB`: second in-domain polyp benchmark
- `CVC-ColonDB`: cross-dataset generalization target
- `ISIC2018`: skin lesion benchmark

## Dataset Layouts

### Local / DGX raw layout for `.py` scripts

This is the default script-based layout:

```text
data/
  Kvasir-SEG/
    images/
    masks/
  CVC-ClinicDB/
    images/
    masks/
  CVC-ColonDB/
    images/
    masks/
  ISIC2018/
    images/
    masks/
```

Important:

- image and mask filenames must share the same stem
- you can override `dataset.root`, `dataset.image_dir`, and `dataset.mask_dir`
- the local `.py` path also supports `ratio`, `predefined`, and `pre_split_dirs` split modes

### Local / DGX pre-split layout for `.py` scripts

If you already have `train`, `val`, and `test` directories, use `dataset.split_strategy=pre_split_dirs`:

```text
some_dataset_root/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

### Kaggle processed layout for the notebook

The notebook expects the processed dataset structure under a detected `WBSNet_Dataset` root:

```text
WBSNet_Dataset/
  kvasir/
    train/
      images/
      masks/
      boundaries/
    val/
      images/
      masks/
      boundaries/
    test/
      images/
      masks/
      boundaries/
  cvc_clinicdb/
    ...
  cvc_colondb/
    ...
  isic2018/
    ...
```

The notebook auto-detects `WBSNet_Dataset` from `/kaggle/input` when possible.

## Installation

Choose one of these local setup options.

### Option 1: `pip`

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Activate the virtual environment with the command that matches your shell:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- Linux or macOS: `source .venv/bin/activate`

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate wbsnet
```

Notes:

- `pyproject.toml` is the main package definition.
- `requirements.txt` is a pinned fallback list, not the best source of truth for the package metadata.

## Local / DGX Python Workflow

Use this path when you want normal script-driven training on a workstation, server, or DGX. This is the preferred path outside Kaggle.

### 1. Verify the repo

```bash
python scripts/verify_repo.py
```

### 2. Run a smoke test

```bash
python train.py --config configs/kvasir_wbsnet.yaml --override train.epochs=1 train.batch_size=2 runtime.wandb.mode=offline
```

### 3. Run the full local training job

```bash
python train.py --config configs/kvasir_wbsnet.yaml
```

### 4. Evaluate the best checkpoint

```bash
python evaluate.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/kvasir_wbsnet/<run_name>/checkpoints/best.pt
```

### 5. Export qualitative predictions

```bash
python predict.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/kvasir_wbsnet/<run_name>/checkpoints/best.pt --split test
```

## DGX Workflow

If you are using a DGX or similar multi-GPU server:

### 1. Connect and start a persistent session

```bash
ssh your_username@your_dgx_ip
tmux new -s wbsnet
```

### 2. Copy the repo

```bash
rsync -avhP /path/to/WBSNET-paper/ your_username@your_dgx_ip:~/WBSNET-paper/
```

### 3. Upload the first dataset

```bash
rsync -avhP ~/datasets/Kvasir-SEG/ your_username@your_dgx_ip:~/wbsnet-data/Kvasir-SEG/
```

### 4. Set up the environment

```bash
cd ~/WBSNET-paper
bash scripts/setup_dgx.sh wbsnet
conda activate wbsnet
python scripts/check_system.py
```

### 5. Run a 1-GPU smoke test

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 1 --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG train.epochs=1 train.batch_size=2 runtime.wandb.mode=offline
```

### 6. Run the real training job

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8 --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG
```

### 7. Evaluate

```bash
python evaluate.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/kvasir_wbsnet/<run_name>/checkpoints/best.pt --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG
```

If you are on Slurm:

```bash
sbatch scripts/slurm_train.sh configs/kvasir_wbsnet.yaml
```

## Kaggle Notebook Workflow

Use this path when you want the notebook-managed paper workflow inside Kaggle.

### What is different from the local `.py` pipeline

- the notebook is self-contained
- the notebook does not read `configs/*.yaml`
- the notebook expects processed splits under `WBSNet_Dataset`
- the notebook is more aggressive about controlling disk growth
- the notebook is designed around Kaggle secrets and `/kaggle/working`

### Recommended Kaggle steps

1. Add the processed dataset as a Kaggle input.
2. Confirm the input contains `WBSNet_Dataset`.
3. Add `WANDB_API_KEY` in Kaggle Secrets if you want W&B logging.
4. Open [`WBSNet_Model.ipynb`](WBSNet_Model.ipynb) in Kaggle.
5. Run the notebook top to bottom.
6. Use the notebook's own config cells to choose single-run checks, paper sweeps, or lambda sweeps.
7. Save important outputs or notebook versions before the session ends.

### Kaggle notebook notes

- The notebook is the right choice when Kaggle output limits matter.
- The notebook is intentionally leaner than the local script path.
- If you want full checkpoint histories and larger export folders, prefer the local `.py` pipeline instead of trying to force the notebook to behave like a workstation run.

## Config Guide For The Local `.py` Pipeline

The local and DGX scripts are controlled by YAML in [`configs/`](configs/).

Useful knobs:

- `dataset.root`
- `dataset.image_dir`
- `dataset.mask_dir`
- `dataset.split_strategy`
- `dataset.split_files`
- `dataset.num_workers`
- `dataset.prefetch_factor`
- `train.epochs`
- `train.batch_size`
- `train.save_every`
- `train.save_last_checkpoint`
- `train.save_best_full_state`
- `runtime.wandb.mode`
- `model.use_wavelet`
- `model.use_lfsa`
- `model.use_hfba`
- `model.boundary_supervision`
- `model.wavelet_type`
- `evaluation.compute_hd95`
- `evaluation.save_paper_panels`
- `evaluation.save_contact_sheet`
- `runtime.wandb.log_images_every`
- `runtime.wandb.max_images`

Override values from the CLI like this:

```bash
python train.py --config configs/kvasir_wbsnet.yaml --override train.epochs=5 train.batch_size=4
```

Useful split modes:

- `dataset.split_strategy=ratio`
- `dataset.split_strategy=predefined`
- `dataset.split_strategy=pre_split_dirs`

Example predefined split override:

```bash
python train.py --config configs/kvasir_wbsnet.yaml --override dataset.split_strategy=predefined dataset.split_files.train=splits/kvasir/train.txt dataset.split_files.val=splits/kvasir/val.txt dataset.split_files.test=splits/kvasir/test.txt
```

## Config Files

### Main local / DGX configs

- [`configs/default.yaml`](configs/default.yaml): shared base config for the local `.py` pipeline
- [`configs/kvasir_wbsnet.yaml`](configs/kvasir_wbsnet.yaml): full WBSNet on Kvasir
- [`configs/clinicdb_wbsnet.yaml`](configs/clinicdb_wbsnet.yaml): full WBSNet on CVC-ClinicDB
- [`configs/isic2018_wbsnet.yaml`](configs/isic2018_wbsnet.yaml): full WBSNet on ISIC2018
- [`configs/kvasir_unet_baseline.yaml`](configs/kvasir_unet_baseline.yaml): baseline U-Net on Kvasir
- [`configs/clinicdb_unet_baseline.yaml`](configs/clinicdb_unet_baseline.yaml): baseline U-Net on CVC-ClinicDB
- [`configs/isic2018_unet_baseline.yaml`](configs/isic2018_unet_baseline.yaml): baseline U-Net on ISIC2018
- [`configs/kvasir_colondb_generalization.yaml`](configs/kvasir_colondb_generalization.yaml): Kvasir-trained WBSNet evaluated on ColonDB
- [`configs/kvasir_colondb_generalization_baseline.yaml`](configs/kvasir_colondb_generalization_baseline.yaml): baseline comparison for ColonDB evaluation

### Ablation configs

- [`configs/ablation_identity_unet.yaml`](configs/ablation_identity_unet.yaml)
- [`configs/ablation_lfsa_only.yaml`](configs/ablation_lfsa_only.yaml)
- [`configs/ablation_hfba_only.yaml`](configs/ablation_hfba_only.yaml)
- [`configs/ablation_no_boundary_supervision.yaml`](configs/ablation_no_boundary_supervision.yaml)
- [`configs/ablation_no_wavelet_attention.yaml`](configs/ablation_no_wavelet_attention.yaml)
- [`configs/ablation_db2_wavelet.yaml`](configs/ablation_db2_wavelet.yaml)

### Optional constrained-environment script configs

These are helper YAMLs for script-based runs in tighter environments. They are not used by the Kaggle notebook itself.

- [`configs/kaggle_kvasir_wbsnet.yaml`](configs/kaggle_kvasir_wbsnet.yaml)
- [`configs/kaggle_clinicdb_wbsnet.yaml`](configs/kaggle_clinicdb_wbsnet.yaml)
- [`configs/kaggle_isic2018_wbsnet.yaml`](configs/kaggle_isic2018_wbsnet.yaml)

## W&B

### Local / DGX

- put your API key in `.env`
- supported names are `WANDB_API_KEY` and `WAND_API_KEY`
- use `runtime.wandb.mode=offline` for smoke tests when needed

### Kaggle

- add `WANDB_API_KEY` in Kaggle Secrets
- let the notebook read it from the environment

Example offline local run:

```bash
python train.py --config configs/kvasir_wbsnet.yaml --override runtime.wandb.mode=offline
```

## Analysis And Reporting Commands

Aggregate completed runs:

```bash
python aggregate_results.py --root outputs --output outputs/aggregated
```

Run the scripted ablation sweep:

```bash
python scripts/run_ablation_suite.py --seeds 3407 3408 3409
```

Estimate parameters and FLOPs:

```bash
python scripts/model_complexity.py --output outputs/model_complexity
```

Run statistical comparisons:

```bash
python scripts/significance_tests.py --root outputs --output outputs/significance --record-type evaluation --reference A1_identity_unet
```

Build qualitative figure sheets:

```bash
python scripts/make_paper_figures.py --input-dir outputs/<experiment>/<run_name>/predictions --limit 8 --columns 2
```

## Helper Scripts

- [`scripts/verify_repo.py`](scripts/verify_repo.py): repository verification and smoke checks
- [`scripts/check_system.py`](scripts/check_system.py): write a hardware / package report
- [`scripts/setup_dgx.sh`](scripts/setup_dgx.sh): create or refresh a DGX conda environment
- [`scripts/train_dgx.sh`](scripts/train_dgx.sh): launch `torchrun` jobs on a DGX
- [`scripts/slurm_train.sh`](scripts/slurm_train.sh): submit training through Slurm
- [`scripts/run_ablation_suite.py`](scripts/run_ablation_suite.py): launch ablations across seeds
- [`scripts/model_complexity.py`](scripts/model_complexity.py): estimate parameter counts and FLOPs
- [`scripts/significance_tests.py`](scripts/significance_tests.py): run statistical comparisons across variants
- [`scripts/make_paper_figures.py`](scripts/make_paper_figures.py): build qualitative figure sheets
- [`scripts/plot_lambda_sweep.py`](scripts/plot_lambda_sweep.py): visualize lambda-sweep results

## Metrics

Tracked metrics include:

- `loss`
- `segmentation_loss`
- `boundary_loss`
- `dice`
- `iou`
- `precision`
- `recall`
- `accuracy`
- `specificity`
- `hd95`

## Paper Files

- manuscript source: [`paper/paper.tex`](paper/paper.tex)
- figure assets: [`diagrams/`](diagrams/)
- build command:

```bash
cd paper
make
```

## Repo Verification And First Runs

### First local run

1. Install the environment.
2. Run `python scripts/verify_repo.py`.
3. Run a 1-epoch Kvasir smoke test.
4. Run the full `configs/kvasir_wbsnet.yaml` training job.
5. Add the remaining datasets after Kvasir is stable.

### First Kaggle run

1. Attach the processed dataset input.
2. Add the `WANDB_API_KEY` secret if needed.
3. Run the notebook single-experiment sanity check first.
4. Start the full paper suite only after the single run is healthy.

## Current Repo Layout

```text
configs/
docs/
paper/
scripts/
tests/
wbsnet/
aggregate_results.py
data_preprocessing.ipynb
evaluate.py
predict.py
train.py
WBSNet_Model.ipynb
```
