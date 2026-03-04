#!/bin/bash
# Submit remaining filter chunks (1750-4999) as sandbox slots become available.
# Sandbox limit: max 5 submitted, max 2 running at a time.
# Each chunk: 350 files, ~3 CPUs, ~1h runtime.

set -uo pipefail  # removed -e so SSH failures don't kill the script
cd /Users/p/projects/LLM-from-scratch/ece405_assignment2

CHUNK_SIZE=350
TOTAL_FILES=5000
FIRST_CHUNK_START=2450  # chunks 0-2449 already submitted (0-2099 done, 2100-2449 running)

for START in $(seq $FIRST_CHUNK_START $CHUNK_SIZE $((TOTAL_FILES - 1))); do
  COUNT=$CHUNK_SIZE
  # Last chunk may be smaller
  if (( START + COUNT > TOTAL_FILES )); then
    COUNT=$((TOTAL_FILES - START))
  fi

  # Wait until we have room (max 5 submitted jobs on sandbox)
  while true; do
    SANDBOX_JOBS=$(ssh koa "squeue -u pavelb -p sandbox -h | wc -l" 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$SANDBOX_JOBS" ]]; then
      echo "$(date): SSH failed, retrying in 30s..."
      sleep 30
      continue
    fi
    if (( SANDBOX_JOBS < 5 )); then
      break
    fi
    echo "$(date): ${SANDBOX_JOBS}/5 sandbox slots used, waiting 60s..."
    sleep 60
  done

  echo "$(date): Submitting chunk start=${START} count=${COUNT}..."
  koa submit scripts/filter_job.slurm --no-auto-gpu \
    --env "CHUNK_START=${START}" --env "CHUNK_COUNT=${COUNT}" \
    --desc "filter-${START}" 2>&1
  echo
done

echo "$(date): All chunks submitted!"
