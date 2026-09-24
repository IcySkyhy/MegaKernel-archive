#!/usr/bin/env bash
# Headline decode benchmark: B=1, TP=EP=8, in=128 / out=1008.
# Reproduces 11.85 mean / 11.82 p50 ms/tok on 8x B200.
#
#   ./scripts/run_decode_bench.sh [OUT_DIR] [-- extra bench args]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-results/decode_bench}"
shift || true
mkdir -p "$OUT_DIR"

export INCLUDE_LAYER0="${INCLUDE_LAYER0:-1}"
export MOE_SPLITK="${MOE_SPLITK:-0}"

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT:-29793}" \
    bench/bench_decode.py \
    --input_len "${INPUT_LEN:-128}" \
    --output_len "${OUTPUT_LEN:-1008}" \
    --warmup "${WARMUP:-2}" \
    --iters "${ITERS:-3}" \
    --output_json "$OUT_DIR/run.json" \
    "$@" 2>&1 | tee "$OUT_DIR/run.log"

python3 -c "
import json
d = json.load(open('$OUT_DIR/run.json'))
print(f\"decode_mean_ms = {d['decode_mean_ms']:.3f}  p50 = {d['decode_p50_ms']:.3f}  \"
      f\"tok/s = {d['decode_tok_per_s']:.2f}\")
"
