#!/bin/bash
#SBATCH --job-name=infer_finetuned
#SBATCH --partition=x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=x.out
#SBATCH --error=x.err

set -euo pipefail

BASE_DIR="${USER}"
CODE_DIR="${BASE_DIR}/code/MedGemma"
DATA_DIR="${BASE_DIR}/data/medgemma"

export PYTHONPATH="${BASE_DIR}/py_pkgs"
export HF_HOME="${BASE_DIR}/hf_cache"
export HF_TOKEN="your_token_here"

python3 "${CODE_DIR}/infer_finetuned.py" \
  --dataset-dir  "${DATA_DIR}/finetune_dataset" \
  --model-dir    "${DATA_DIR}/finetuned_models/best_ce_core" \
  --base-model-id google/medgemma-1.5-4b-it \
  --out-dir      "${DATA_DIR}/infer_out_ce_core" \
  --label        ce_core \
  --split        val