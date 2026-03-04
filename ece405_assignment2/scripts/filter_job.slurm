#!/bin/bash
#SBATCH --job-name=filter-wet
#SBATCH --partition=sandbox
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --nodes=1

set -euo pipefail

# Chunk params (set via env: CHUNK_START, CHUNK_COUNT)
CHUNK_START="${CHUNK_START:-0}"
CHUNK_COUNT="${CHUNK_COUNT:-5000}"

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}" "${RESULTS_DIR}"
export RESULTS_DIR REPO_DIR

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Chunk: start=${CHUNK_START} count=${CHUNK_COUNT}"
echo "Started: $(date)"
echo

# Setup environment (CPU-only, no CUDA needed)
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
fi

# Paths — shared data on class filesystem
WET_DIR="/mnt/lustre/koa/class/ece605_class/wet"
MODELS_DIR="/mnt/lustre/koa/class/ece605_class/models"
OUTPUT_DIR="/mnt/lustre/koa/scratch/pavelb/ece405_filtered"
# Each worker loads ~3 GB of models (lid + dolma + quality classifier)
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${OUTPUT_DIR}"

echo "==== Filter Pipeline ====="
echo "WET dir: ${WET_DIR}"
echo "Models dir: ${MODELS_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo "WET files total: $(ls ${WET_DIR}/*.gz 2>/dev/null | wc -l)"
echo

srun uv run python scripts/filter_data.py \
  --input_dir "${WET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --models_dir "${MODELS_DIR}" \
  --glob "*.warc.wet.gz" \
  --start "${CHUNK_START}" \
  --count "${CHUNK_COUNT}"

echo
echo "==== Done ====="
echo "Finished: $(date)"
echo "Filtered output: ${OUTPUT_DIR}"
du -sh "${OUTPUT_DIR}"
