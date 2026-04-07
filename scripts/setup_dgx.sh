#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-wbsnet}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load your Anaconda/Miniconda module first."
  exit 1
fi

echo "[1/4] Creating or updating conda environment: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

echo "[2/4] Activating environment"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[3/4] Verifying Python packages"
python3 -m pip install -e .

echo "[4/4] Capturing hardware report"
python3 scripts/check_system.py

echo
echo "DGX environment is ready."
echo "Next:"
echo "  bash scripts/train_dgx.sh configs/kvasir_wbsnet.yaml 8"
