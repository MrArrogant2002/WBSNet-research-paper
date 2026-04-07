#!/usr/bin/env bash
#SBATCH --job-name=wbsnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%j.out

set -euo pipefail

CONFIG_PATH="${1:-configs/kvasir_wbsnet.yaml}"
ENV_NAME="${ENV_NAME:-wbsnet}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
fi

export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

srun torchrun \
  --nnodes="${SLURM_NNODES:-1}" \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-8}" \
  --node_rank="${SLURM_NODEID:-0}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py --config "${CONFIG_PATH}"
