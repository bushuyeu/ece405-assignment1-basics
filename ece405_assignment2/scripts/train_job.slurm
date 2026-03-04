#!/bin/bash
#SBATCH --job-name=train-gpt2
#SBATCH --partition=ece405
#SBATCH --account=ece405
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:2

set -euo pipefail

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}" "${RESULTS_DIR}"
export RESULTS_DIR REPO_DIR

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "=== GPT-2 Training Job ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start: $(date)"
echo

# Setup environment
export PATH="${HOME}/.local/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${JOB_DIR}/../.venv"

if ! uv --help >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  echo "Creating virtual environment..."
  uv venv "${UV_PROJECT_ENVIRONMENT}"
  uv sync
else
  echo "Using existing virtual environment"
  uv sync --quiet 2>&1
fi

# Step 1: Create validation split if it doesn't exist
VALID_BIN="/mnt/lustre/koa/scratch/pavelb/ece405_tokenized/valid.bin"
if [[ ! -f "${VALID_BIN}" ]]; then
  echo "Creating validation split..."
  uv run python scripts/split_validation.py \
    --input /mnt/lustre/koa/scratch/pavelb/ece405_tokenized/train.bin
  echo
fi

# Step 2: wandb offline mode (sync later with `wandb sync`)
export WANDB_MODE=offline

# Step 3: Train with DDP (2 GPUs)
echo "=== Starting Training ==="
echo "Train: /mnt/lustre/koa/scratch/pavelb/ece405_tokenized/train.bin"
echo "Valid: ${VALID_BIN}"
echo "Output: /mnt/lustre/koa/scratch/pavelb/ece405_training/your_data"
echo

NUM_GPUS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run from cs336-basics so hydra finds configs/
cd cs336-basics

uv run torchrun \
  --standalone \
  --nproc_per_node=${NUM_GPUS} \
  scripts/train.py \
  --config-name=experiment/your_data

echo
echo "=== Done ==="
echo "Finished: $(date)"
ls -lh /mnt/lustre/koa/scratch/pavelb/ece405_training/your_data/
