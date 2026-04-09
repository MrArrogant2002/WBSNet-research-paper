# WBSNet Hardware Requirements

This file summarizes the hardware target for running the full WBSNet plan on another system.

## Recommended Target

- GPU: `1 x RTX 3090 / 4090 / A5000` with `24 GB` VRAM
- CPU: `16 cores`
- RAM: `32 GB`
- Storage: `500 GB NVMe SSD`
- CUDA: `12.x`
- Python: `3.10+`
- PyTorch: `2.2+`

This is the most practical single-machine setup for:

- vanilla U-Net baselines
- full WBSNet training
- all ablations including `A7 db2`
- prediction export and HD95 evaluation

## Minimum Viable System

- GPU: `1 x RTX 3060` with `12 GB` VRAM
- CPU: `8 cores`
- RAM: `16 GB`
- Storage: `100 GB SSD`

On the minimum setup:

- use AMP / FP16
- reduce batch size from `16` to `4-8`
- if needed, keep the effective batch size with `train.grad_accum_steps`

## Ideal Research Setup

- GPU: `1-4 x A100` with `40-80 GB` VRAM
- RAM: `64+ GB`
- Storage: `1 TB NVMe`

This is best if you want to run the full paper plan fast, especially the multi-seed ablation suite.

## Expected GPU Memory

- WBSNet at `352 x 352`, batch size `16`, ResNet-34 encoder: about `8-10 GB` peak VRAM
- The wavelet path temporarily expands feature maps, so leave extra headroom instead of packing the GPU to 100%

## Expected Training Time

Single run, single GPU:

- `Kvasir-SEG`, 200 epochs: about `4-5 h` on RTX 3060, `2-3 h` on RTX 3090
- `CVC-ClinicDB`, 200 epochs: about `3-4 h` on RTX 3060, `1.5-2 h` on RTX 3090
- `ISIC2018`, 100 epochs: about `6-8 h` on RTX 3060, `3-4 h` on RTX 3090

Full research plan estimate:

- main experiments: `15-20 h` on RTX 3060, `8-10 h` on RTX 3090
- ablations on one dataset: `28-35 h` on RTX 3060, `14-21 h` on RTX 3090
- full 3-seed paper campaign: about `130-165 h` on RTX 3060, `66-93 h` on RTX 3090

## Storage Planning

Reserve space for:

- datasets
- checkpoints
- predictions and overlays
- aggregated tables
- optional raw logs and W&B caches

Recommended storage budget:

- minimum workable: `100 GB`
- comfortable research setup: `500 GB`

## Transfer Checklist

1. Copy this repo to the larger system.
2. Create the conda environment with `bash scripts/setup_dgx.sh wbsnet`.
3. Upload datasets under the configured roots or override `dataset.root`.
4. Run `python3 scripts/check_system.py`.
5. Run `python3 scripts/verify_repo.py`.
6. Start with a 1-epoch smoke test before full training.
