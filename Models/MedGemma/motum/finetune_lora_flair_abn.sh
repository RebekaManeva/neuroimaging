#!/bin/bash
#SBATCH --job-name=finetune_flair
#SBATCH --partition=x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=x.out
#SBATCH --error=x.err

set -euo pipefail

BASE_DIR="${USER}"
CODE_DIR="${BASE_DIR}/code/MedGemma"
DATA_DIR="${BASE_DIR}/data/medgemma"

export PYTHONPATH="${BASE_DIR}/py_pkgs"
export HF_HOME="${BASE_DIR}/hf_cache"
export HF_TOKEN="your_token_here"
export MPLCONFIGDIR="${BASE_DIR}/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python3 "${CODE_DIR}/finetune_lora.py" \
  --dataset-dir "${DATA_DIR}/finetune_dataset" \
  --output-dir  "${DATA_DIR}/finetuned_models" \
  --model-id    google/medgemma-1.5-4b-it \
  --label       flair_abn \
  --epochs      30 \
  --batch-size  2 \
  --lr          1e-4