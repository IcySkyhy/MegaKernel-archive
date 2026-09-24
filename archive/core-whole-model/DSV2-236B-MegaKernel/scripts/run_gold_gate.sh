#!/usr/bin/env bash
# Numerical gold gate + decode benchmark in one load+compile, with PTX/CUBIN
# dumps and an environment fingerprint. Exits non-zero if the gold gate fails.
#
#   ./scripts/run_gold_gate.sh [OUT_DIR] [-- extra test args]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-results/gold_gate}"
shift || true
mkdir -p "$OUT_DIR"
DUMP_DIR="$(cd "$OUT_DIR" && pwd)/dump"
mkdir -p "$DUMP_DIR"

{
  echo "=== env fingerprint $(date -u +'%Y-%m-%d %H:%M:%S UTC') ==="
  echo "git_head: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  nvidia-smi --query-gpu=index,uuid,driver_version,memory.total --format=csv 2>/dev/null || true
  echo "python:  $(python3 --version)"
  echo "torch:   $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
  echo "cutlass: $(python3 -c 'import cutlass; print(getattr(cutlass, "__version__", "unknown"))')"
} > "$OUT_DIR/env.txt"

export CUTE_DSL_KEEP="ptx,cubin"
export CUTE_DSL_DUMP_DIR="$DUMP_DIR"
export CUTE_DSL_LINEINFO=1
export INCLUDE_LAYER0="${INCLUDE_LAYER0:-1}"
export MOE_SPLITK="${MOE_SPLITK:-0}"

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT:-29799}" \
    tests/test_gold_and_bench.py \
    --input_len "${INPUT_LEN:-128}" \
    --output_len "${OUTPUT_LEN:-1008}" \
    --warmup "${WARMUP:-2}" \
    --iters "${ITERS:-3}" \
    --output_json "$OUT_DIR/gate.json" \
    "$@" 2>&1 | tee "$OUT_DIR/gate.log"
