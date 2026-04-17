#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/kvasir_wbsnet.yaml}"
NUM_GPUS="${2:-8}"
EXTRA_ARGS=("${@:3}")
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
