# WBSNet

WBSNet is a DGX-ready research codebase for the paper idea in this repository:
Wavelet Boundary Skip Network for medical image segmentation.

The repo includes:

- A pure PyTorch implementation of `WBSNet` with four WBS skip modules.
- Baseline and ablation support through config flags.
- Training, evaluation, prediction export, and paper-statistics aggregation.
- Single-GPU and multi-GPU execution through `torchrun`.
- Weights & Biases logging with `.env` support.

## What This Repo Produces

Each experiment run writes the artifacts needed for paper drafting:

- `metrics.csv`: epoch-by-epoch train/val metrics.
- `best_metrics.json`: best validation snapshot.
- `run_summary.json`: compact run metadata for aggregation.
- `predictions/`: saved masks and overlays for qualitative figures.
- `evaluation/<dataset>.json`: per-dataset test metrics.
- `aggregated_results.*`: merged tables across multiple seeds or variants.

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

If your folders differ, update the matching config in `configs/`.

## Quick Start On A DGX Server

1. Create the environment:

```bash
bash scripts/setup_dgx.sh wbsnet
conda activate wbsnet
```

2. Check the detected GPUs, CUDA stack, and installed packages:

```bash
python3 scripts/check_system.py
```

3. Check that your `.env` contains either `WANDB_API_KEY` or `WAND_API_KEY`.

4. Train the full model:

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8
```

5. Evaluate the best checkpoint:

```bash
python evaluate.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/kvasir_wbsnet/<run_name>/checkpoints/best.pt
```

6. Aggregate repeated runs for the paper:

```bash
python aggregate_results.py --root outputs --output outputs/aggregated
```

If your DGX is attached to a scheduler, submit with:

```bash
sbatch scripts/slurm_train.sh configs/kvasir_wbsnet.yaml
```

## Main Commands

Train:

```bash
python train.py --config configs/kvasir_wbsnet.yaml
```

Multi-GPU train on DGX:

```bash
bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8
```

Generate a hardware report:

```bash
python3 scripts/check_system.py
```

Run the ablation suite:

```bash
python scripts/run_ablation_suite.py --base-config configs/kvasir_wbsnet.yaml --seeds 3407 3408 3409
```

Export qualitative predictions:

```bash
python predict.py --config configs/kvasir_wbsnet.yaml --checkpoint outputs/.../checkpoints/best.pt --split test
```

## Config Notes

The experiment behavior is controlled entirely by YAML:

- `model.use_wavelet`
- `model.use_lfsa`
- `model.use_hfba`
- `model.boundary_supervision`
- `model.wavelet_type`

These switches support the baseline and ablation variants described in the paper plan.

## W&B Note

This repo accepts either `WANDB_API_KEY` or the existing `WAND_API_KEY` from your `.env`. If only `WAND_API_KEY` exists, it is mapped automatically to `WANDB_API_KEY` at runtime.

## Repo Layout

```text
wbsnet/
  data/
  models/
  utils/
configs/
scripts/
train.py
evaluate.py
predict.py
aggregate_results.py
```

## Verification

Use the lightweight verifier after editing:

```bash
python3 scripts/verify_repo.py
```

## Recommended Remote Workflow

1. Copy the repo to the server and place datasets under `data/`.
2. Run `bash scripts/setup_dgx.sh wbsnet`.
3. Run `python3 scripts/check_system.py` and inspect `artifacts/system_report.json`.
4. Start with one seed on one dataset.
5. Move to multi-seed ablations only after the first run completes cleanly.
