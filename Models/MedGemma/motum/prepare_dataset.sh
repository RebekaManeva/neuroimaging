#!/bin/bash
#SBATCH --job-name=prepare_dataset
#SBATCH --partition=openlab-queue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=x.out
#SBATCH --error=x.err

set -euo pipefail
set -x

BASE_DIR="${USER}"
CODE_DIR="${BASE_DIR}/code/MedGemma"
DATA_DIR="${BASE_DIR}/data/medgemma"
MOTUM_DIR="${BASE_DIR}/data/motum/MOTUM-v.2.2"

mkdir -p "${CODE_DIR}/logs"
export PYTHONPATH="${BASE_DIR}/py_pkgs"

python3 "${CODE_DIR}/prepare_dataset.py" \
  --motum-root "${MOTUM_DIR}" \
  --out-dir    "${DATA_DIR}/finetune_dataset" \
  --png-size   512 \
  --slice-axis 2 \
  --trim-frac  0.0

