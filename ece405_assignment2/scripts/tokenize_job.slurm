#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=ece405
#SBATCH --account=ece405
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1

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

echo "=== Tokenization Job ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"
echo

# Setup environment (same as filter_job.slurm)
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

INPUT_DIR="/mnt/lustre/koa/scratch/pavelb/ece405_filtered"
OUTPUT_DIR="/mnt/lustre/koa/scratch/pavelb/ece405_tokenized"
mkdir -p "${OUTPUT_DIR}"

WORKERS="${SLURM_CPUS_PER_TASK:-8}"

echo "Input: ${INPUT_DIR} ($(ls ${INPUT_DIR}/*.warc.wet.gz 2>/dev/null | wc -l) files)"
echo "Output: ${OUTPUT_DIR}/train.bin"
echo "Workers: ${WORKERS}"
echo

srun uv run python scripts/tokenize_data.py \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}/train.bin" \
  --workers "${WORKERS}"

echo
echo "=== Done ==="
echo "Finished: $(date)"
ls -lh "${OUTPUT_DIR}/train.bin"
