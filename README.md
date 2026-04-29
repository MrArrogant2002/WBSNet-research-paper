# WBSNet

Wavelet Boundary Skip Network for medical image segmentation.

This repository contains three related deliverables:

- A PyTorch package driven by YAML configs and the CLI entry points `train.py`, `evaluate.py`, and `predict.py`.
- A Google Colab A100 control notebook, [`WBSNet_Colab.ipynb`](WBSNet_Colab.ipynb), that mounts Drive and launches the `.py` pipeline via [`scripts/run_paper_optionA.py`](scripts/run_paper_optionA.py).
- The paper sources in [`paper/`](paper/) and supporting figures in [`diagrams/`](diagrams/).

## Quick start: Option-A paper run on Colab

The fastest way to reproduce the paper results end-to-end:

1. Open [`WBSNet_Colab.ipynb`](WBSNet_Colab.ipynb) in Colab (Pro+ A100 recommended; Pro L4 also works).
2. Place `WBSNet_Dataset/{kvasir,cvc_clinicdb,cvc_colondb,isic2018}` on Google Drive.
3. Run cells top-to-bottom. The headline command in cell 8 is:

   ```bash
   python3 scripts/run_paper_optionA.py \
       --seeds 3407 \
       --override train.epochs=150 train.batch_size=16 \
                  dataset.split_strategy=pre_split_dirs \
                  dataset.num_workers=2 dataset.prefetch_factor=2 \
                  train.save_every=5 runtime.wandb.mode=offline \
                  evaluation.max_visualizations=8
   ```

   This trains 7 ablation variants on Kvasir + full WBSNet on ClinicDB & ISIC2018 + 3 U-Net baselines, then runs the Kvasir → CVC-ColonDB generalization eval, then aggregates everything. Re-running with additional seeds skips already-completed runs (idempotent).

## Execution Model

This repo is prepared for the script-based Colab A100 workflow. Colab is used only as the control surface for cloning the repo, mounting Drive, installing dependencies, and launching commands. Training, evaluation, prediction export, aggregation, significance tests, and complexity checks run through versioned Python files.

| Layer | Primary file(s) | Purpose |
| --- | --- | --- |
| Colab control panel | [`WBSNet_Colab.ipynb`](WBSNet_Colab.ipynb) | Mount Drive, link the processed dataset, run shell commands |
| Training/evaluation code | [`train.py`](train.py), [`evaluate.py`](evaluate.py), [`predict.py`](predict.py) | Reproducible experiment execution from YAML configs |
| Paper pipeline | [`scripts/run_paper_optionA.py`](scripts/run_paper_optionA.py) | Staged ablations, main results, baselines, generalization, aggregation |
| Package code | [`wbsnet/`](wbsnet/) | Models, losses, metrics, data loading, logging, utilities |

Important:

- Do not use a training notebook as the experiment source of truth.
- Keep experiment claims tied to artifacts under `outputs/`.
- Use `dataset.split_strategy=pre_split_dirs` for the processed Google Drive dataset layout.

## What The Project Contains

| Path | Purpose |
| --- | --- |
| [`WBSNet_Colab.ipynb`](WBSNet_Colab.ipynb) | Google Colab driver for the Option-A paper run (drives `scripts/run_paper_optionA.py`) |
| [`data_preprocessing.ipynb`](data_preprocessing.ipynb) | Dataset preparation notebook |
| [`train.py`](train.py) | Python training entry point |
| [`evaluate.py`](evaluate.py) | Evaluate a trained checkpoint and export metrics / predictions |
| [`predict.py`](predict.py) | Save qualitative predictions from a trained checkpoint |
| [`aggregate_results.py`](aggregate_results.py) | Aggregate run summaries and evaluation outputs |
| [`configs/`](configs/) | YAML configs for the Python pipeline |
| [`scripts/`](scripts/) | Helper scripts for verification, DGX launch, ablations, paper-run driver (`run_paper_optionA.py`), figures, significance tests, and complexity |
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
| `A6` | No wavelet (raw-attention skip) | [`configs/ablation_no_wavelet.yaml`](configs/ablation_no_wavelet.yaml) |
| `A7` | `db2` wavelet variant | [`configs/ablation_db2_wavelet.yaml`](configs/ablation_db2_wavelet.yaml) |

Dataset-specific full-model and baseline configs are also provided for Kvasir, CVC-ClinicDB, ISIC2018, and Kvasir-to-ColonDB evaluation.

## Outputs

### Python pipeline

The script-based pipeline writes into `outputs/` by default.

Typical run artifacts:

- `metrics.csv`
- `best_metrics.json`
- `run_summary.json`
- `checkpoints/best.pt`
- `checkpoints/epoch_*.pt` when periodic checkpointing is enabled
- `checkpoints/last.pt` when `train.save_last_checkpoint=true`
- `evaluation/<dataset>_<split>.json`
- `predictions/` with masks, overlays, paper panels, and contact sheets

## Hardware Guidance

Short version:

- Local minimum: `1 x 12 GB GPU`, `16 GB RAM`, and enough free disk for checkpoints and predictions
- Better local experience: `1 x 24 GB GPU`, `32 GB RAM`
- Ideal for the Colab workflow: A100 GPU with background execution for long runs
- Ideal outside Colab: DGX or another multi-GPU server

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

### Python raw layout for `.py` scripts

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

### Python pre-split layout for `.py` scripts

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

### Processed split layout for Colab

The Colab workflow expects this processed dataset structure on Google Drive under `MyDrive/WBSNet_Dataset`:

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

The Colab notebook symlinks these folders into `data/` and the Python scripts consume them with `dataset.split_strategy=pre_split_dirs`.

## Installation

Choose one of these local setup options.

### Option 1: `pip`

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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

- `pyproject.toml` intentionally does not force-install PyTorch, so Colab keeps its CUDA-enabled torch build.
- `requirements.txt` pins the full local/HPC Python stack, including `torch` and `torchvision`.

## Python Workflow

Use this path when you want normal script-driven training on Google Colab A100, a workstation, server, or DGX.

### 1. Verify the repo

```bash
python scripts/verify_repo.py
```

### 2. Run a smoke test

```bash
python train.py --config configs/kvasir_wbsnet.yaml \
  --override dataset.split_strategy=pre_split_dirs train.epochs=1 train.batch_size=2 runtime.wandb.mode=offline
```

### 3. Run the full training job

```bash
python train.py --config configs/kvasir_wbsnet.yaml \
  --override dataset.split_strategy=pre_split_dirs
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

## Config Guide For The Python Pipeline

The Colab/script workflow is controlled by YAML in [`configs/`](configs/), with run-specific values passed through `--override`.

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

### Main Python configs

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
- [`configs/ablation_no_wavelet.yaml`](configs/ablation_no_wavelet.yaml)
- [`configs/ablation_db2_wavelet.yaml`](configs/ablation_db2_wavelet.yaml)

## W&B

### Colab / Local

- put your API key in `.env` locally or set `WANDB_API_KEY` in Colab
- supported names are `WANDB_API_KEY` and `WAND_API_KEY`
- use `runtime.wandb.mode=offline` for smoke tests and long Colab runs unless online logging is required

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

### First Colab A100 run

1. Open `WBSNet_Colab.ipynb` and select an A100 runtime.
2. Mount Drive and confirm `MyDrive/WBSNet_Dataset` contains the processed split folders.
3. Run `python3 scripts/verify_repo.py`.
4. Run the 1-epoch smoke test cell.
5. Run `scripts/run_paper_optionA.py --seeds 3407` first.
6. Add seeds `3408 3409` only after the first seed produces stable artifacts.

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
WBSNet_Colab.ipynb
```
