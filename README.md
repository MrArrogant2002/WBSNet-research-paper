# WBSNet

WBSNet is a DGX-ready research codebase for the paper idea in this repository:
Wavelet Boundary Skip Network for medical image segmentation.

The repo includes:

- A pure PyTorch implementation of `WBSNet` with four WBS skip modules.
- Baseline and ablation support through YAML configs, including `A1-A7`.
- Training, evaluation, prediction export, and paper-statistics aggregation.
- Single-GPU and multi-GPU execution through `torchrun`.
- Weights & Biases logging through `.env`.
- Beginner-friendly DGX helper scripts for setup, hardware checks, and launching runs.

## What This Repo Produces

Each run writes artifacts that are useful for paper writing:

- `metrics.csv`: epoch-by-epoch train and validation metrics
- `best_metrics.json`: best validation snapshot
- `run_summary.json`: compact run metadata for aggregation
- `evaluation/<dataset>.json`: final evaluation metrics
- `predictions/`: masks and overlays for qualitative figures
- `aggregated_results.*`: merged outputs across seeds and variants
- `artifacts/system_report.json`: DGX hardware and package report

## Hardware Requirements

The short version:

- Minimum: `1 x 12 GB GPU`, `16 GB RAM`, `100 GB SSD`
- Recommended: `1 x 24 GB GPU`, `32 GB RAM`, `500 GB NVMe SSD`
- Ideal for the full paper plan: `A100 40/80 GB` or multi-GPU

Detailed planning notes are in [docs/HARDWARE_REQUIREMENTS.md](/home/eswarbalu/Desktop/WBSNET-paper/docs/HARDWARE_REQUIREMENTS.md).

## Datasets To Download

Recommended order:

1. `Kvasir-SEG`
2. `CVC-ClinicDB`
3. `CVC-ColonDB`
4. `ISIC2018`

How they are used:

- `Kvasir-SEG`: first smoke test and main polyp benchmark
- `CVC-ClinicDB`: second in-domain polyp benchmark
- `CVC-ColonDB`: cross-dataset generalization evaluation
- `ISIC2018`: skin lesion benchmark

## Expected Dataset Layout

The loaders are config-driven, but the default structure is:

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

- Image and mask filenames must have the same stem, such as `0001.jpg` and `0001.png`
- If your downloaded dataset uses a different folder structure, either reorganize it or override `dataset.root`, `dataset.image_dir`, and `dataset.mask_dir`

## Simple DGX Workflow

If you are new to DGX, follow this exact order.

### 1. Connect to the server

```bash
ssh your_username@your_dgx_ip
```

### 2. Start a persistent terminal session

```bash
tmux new -s wbsnet
```

Detach without stopping the job:

```bash
Ctrl+b then d
```

Reattach later:

```bash
tmux attach -t wbsnet
```

### 3. Copy the repo to the DGX

Run this from your local machine:

```bash
rsync -avhP /home/eswarbalu/Desktop/WBSNET-paper/ your_username@your_dgx_ip:~/WBSNET-paper/
```

### 4. Upload the first dataset

Start with `Kvasir-SEG` only.

Run this from your local machine:

```bash
rsync -avhP ~/datasets/Kvasir-SEG/ your_username@your_dgx_ip:~/wbsnet-data/Kvasir-SEG/
```

### 5. Set up the environment on the DGX

```bash
cd ~/WBSNET-paper
bash scripts/setup_dgx.sh wbsnet
conda activate wbsnet
python3 scripts/check_system.py
```

### 6. Run a 1-GPU smoke test first

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 1 --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG train.epochs=1 train.batch_size=2 runtime.wandb.mode=offline
```

### 7. Run the real multi-GPU experiment

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8 --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG
```

### 8. Evaluate and aggregate

```bash
python evaluate.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/kvasir_wbsnet/<run_name>/checkpoints/best.pt --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG
python aggregate_results.py --root outputs --output outputs/aggregated
```

## Dataset Upload Options

### Option 1: Upload from your laptop with `rsync`

```bash
rsync -avhP ~/datasets/Kvasir-SEG/ your_username@your_dgx_ip:~/wbsnet-data/Kvasir-SEG/
rsync -avhP ~/datasets/CVC-ClinicDB/ your_username@your_dgx_ip:~/wbsnet-data/CVC-ClinicDB/
rsync -avhP ~/datasets/CVC-ColonDB/ your_username@your_dgx_ip:~/wbsnet-data/CVC-ColonDB/
rsync -avhP ~/datasets/ISIC2018/ your_username@your_dgx_ip:~/wbsnet-data/ISIC2018/
```

### Option 2: Copy archive files and extract on the DGX

```bash
scp Kvasir-SEG.zip your_username@your_dgx_ip:~/wbsnet-data/
ssh your_username@your_dgx_ip
cd ~/wbsnet-data
unzip -q Kvasir-SEG.zip -d Kvasir-SEG
```

### Option 3: Download directly on the DGX

If the DGX has internet access, download and extract the datasets directly into `~/wbsnet-data/`.

## Quick Start On A DGX Server

```bash
cd ~/WBSNET-paper
bash scripts/setup_dgx.sh wbsnet
conda activate wbsnet
python3 scripts/check_system.py
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 1 --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG train.epochs=1 train.batch_size=2 runtime.wandb.mode=offline
```

If the smoke test passes, launch the real run:

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8 --override dataset.root=/home/your_username/wbsnet-data/Kvasir-SEG
```

## Slurm Workflow

If your DGX is attached to a scheduler:

```bash
sbatch scripts/slurm_train.sh configs/kvasir_wbsnet.yaml
```

## Main Commands

Train on the current machine:

```bash
python train.py --config configs/kvasir_wbsnet.yaml
```

Train on DGX:

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8
```

Run a 1-epoch debug pass:

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 1 --override train.epochs=1 train.batch_size=2 runtime.wandb.mode=offline
```

Evaluate:

```bash
python evaluate.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/kvasir_wbsnet/<run_name>/checkpoints/best.pt
```

Export qualitative predictions:

```bash
python predict.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/.../checkpoints/best.pt --split test
```

Aggregate results:

```bash
python aggregate_results.py --root outputs --output outputs/aggregated
```

Run the ablation suite:

```bash
python scripts/run_ablation_suite.py --seeds 3407 3408 3409
```

Generate a hardware report:

```bash
python3 scripts/check_system.py
```

Verify the repo:

```bash
python3 scripts/verify_repo.py
```

## Config Notes

The experiment behavior is controlled through YAML in `configs/`.

Useful knobs:

- `dataset.root`
- `dataset.image_dir`
- `dataset.mask_dir`
- `dataset.split_strategy`
- `dataset.split_files`
- `train.epochs`
- `train.batch_size`
- `runtime.wandb.mode`
- `model.use_wavelet`
- `model.use_lfsa`
- `model.use_hfba`
- `model.boundary_supervision`
- `model.wavelet_type`

You can override config values from the command line:

```bash
python train.py --config configs/kvasir_wbsnet.yaml --override train.epochs=5 train.batch_size=4
```

Useful split modes:

- `dataset.split_strategy=ratio`: random train/val/test split
- `dataset.split_strategy=predefined`: read exact sample ids from `dataset.split_files`
- `--split all`: evaluate every sample in a dataset, which is what you want for `CVC-ColonDB`

Example predefined split override:

```bash
python train.py --config configs/kvasir_wbsnet.yaml --override \
  dataset.split_strategy=predefined \
  dataset.split_files.train=splits/kvasir/train.txt \
  dataset.split_files.val=splits/kvasir/val.txt \
  dataset.split_files.test=splits/kvasir/test.txt
```

Important configs that match the paper plan:

- `configs/kvasir_wbsnet.yaml`: full WBSNet on Kvasir
- `configs/clinicdb_wbsnet.yaml`: full WBSNet on CVC-ClinicDB
- `configs/isic2018_wbsnet.yaml`: full WBSNet on ISIC 2018
- `configs/kvasir_unet_baseline.yaml`: vanilla ResNet-34 U-Net baseline on Kvasir
- `configs/clinicdb_unet_baseline.yaml`: vanilla ResNet-34 U-Net baseline on CVC-ClinicDB
- `configs/isic2018_unet_baseline.yaml`: vanilla ResNet-34 U-Net baseline on ISIC 2018
- `configs/ablation_identity_unet.yaml`: A1
- `configs/kvasir_wbsnet.yaml`: A2
- `configs/ablation_lfsa_only.yaml`: A3
- `configs/ablation_hfba_only.yaml`: A4
- `configs/ablation_no_boundary_supervision.yaml`: A5
- `configs/ablation_no_wavelet_attention.yaml`: A6
- `configs/ablation_db2_wavelet.yaml`: A7
- `configs/kvasir_colondb_generalization.yaml`: Kvasir-trained checkpoint evaluated on the full ColonDB set
- `configs/kvasir_colondb_generalization_baseline.yaml`: baseline U-Net checkpoint evaluated on the full ColonDB set

## W&B Note

This repo accepts either `WANDB_API_KEY` or the existing `WAND_API_KEY` from `.env`.
If only `WAND_API_KEY` exists, it is mapped automatically to `WANDB_API_KEY` at runtime.

For smoke tests, use offline mode:

```bash
--override runtime.wandb.mode=offline
```

## Current Repo Layout

```text
configs/
docs/
scripts/
wbsnet/
  data/
  models/
  utils/
train.py
evaluate.py
test.py
predict.py
aggregate_results.py
requirements.txt
environment.yml
```

## Helper Scripts

- `scripts/setup_dgx.sh`: create or update the DGX conda environment
- `scripts/check_system.py`: save GPU and package information to `artifacts/system_report.json`
- `scripts/train_dgx.sh`: launch single-node or multi-node `torchrun`
- `scripts/slurm_train.sh`: submit through Slurm
- `scripts/run_ablation_suite.py`: launch ablations across seeds
- `scripts/verify_repo.py`: structural verification plus optional runtime smoke checks when `torch` is installed

## Recommended First Run

1. Upload repo and `Kvasir-SEG`
2. Run `bash scripts/setup_dgx.sh wbsnet`
3. Run `python3 scripts/check_system.py`
4. Run the 1-epoch smoke test
5. Run the full `Kvasir-SEG` experiment
6. Add the remaining datasets after the first full run is stable
