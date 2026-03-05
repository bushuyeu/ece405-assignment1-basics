#!/bin/bash
# Train GPT-2 small on 8x A100 (Lambda Cloud)
# Expected runtime: ~2-3 hours for 20K steps
set -euo pipefail

# --- Configuration ---
KOA_USER="pavelb"
KOA_HOST="koa"  # assumes SSH config alias
DATA_DIR="/home/ubuntu/data"
REPO_DIR="/home/ubuntu/LLM-from-scratch"
WORK_DIR="${REPO_DIR}/ece405_assignment2/cs336-basics"

echo "=== Step 1: Transfer data from Koa ==="
mkdir -p "$DATA_DIR"
if [[ ! -f "$DATA_DIR/train.bin" ]]; then
    echo "Downloading train.bin (17 GB) from Koa..."
    rsync -avP "${KOA_HOST}:/mnt/lustre/koa/scratch/${KOA_USER}/ece405_tokenized/train.bin" "$DATA_DIR/"
else
    echo "train.bin already exists, skipping"
fi
if [[ ! -f "$DATA_DIR/valid.bin" ]]; then
    echo "Downloading valid.bin (20 MB) from Koa..."
    rsync -avP "${KOA_HOST}:/mnt/lustre/koa/scratch/${KOA_USER}/ece405_tokenized/valid.bin" "$DATA_DIR/"
else
    echo "valid.bin already exists, skipping"
fi

echo ""
echo "=== Step 2: Setup environment ==="
cd "$WORK_DIR"
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

echo ""
echo "=== Step 3: Verify GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

echo ""
echo "=== Step 4: Launch training ==="
echo "Config: experiment/8xa100"
echo "GPUs: ${NUM_GPUS}"
echo "Effective batch: 128 * ${NUM_GPUS} = $((128 * NUM_GPUS))"
echo "Tokens/step: $((128 * NUM_GPUS * 512))"
echo ""

# Set NCCL optimizations for multi-GPU
export NCCL_P2P_LEVEL=NVL  # use NVLink if available
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

uv run torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train.py \
    --config-name=experiment/8xa100

echo ""
echo "=== Training complete ==="
echo "Model saved to: /home/ubuntu/output/8xa100/"
ls -lh /home/ubuntu/output/8xa100/model.pt
