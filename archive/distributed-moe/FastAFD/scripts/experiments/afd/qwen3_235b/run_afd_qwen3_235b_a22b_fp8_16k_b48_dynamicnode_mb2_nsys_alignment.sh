#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# Qwen3-235B-A22B-FP8 AFD AG/EG large-node 16k decode replay workload.
#
# This is intentionally a thin wrapper around the 8k dynamic-node runner.  The
# topology, vLLM scoring defaults, MegaMoE settings, cleanup, and Ray placement
# logic stay shared there; this file only changes the long-context workload
# shape.
#
# 16k doubles KV tokens/request relative to the 8k b96 workload, so the default
# per-attention-GPU batch is halved to 48.  With mb=2 this replays a 24-token
# per-microbatch graph, so AFD_DECODE_GRAPH_BS must include 24.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-FP8}"
export PROMPT_LEN="${PROMPT_LEN:-16384}"
export PER_ATTN_GPU_BSZ="${PER_ATTN_GPU_BSZ:-48}"
export MAX_TOKENS="${MAX_TOKENS:-16}"

export AFD_TOTAL_NODES="${AFD_TOTAL_NODES:-${NUM_NODES:-4}}"
export AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-512}"
export MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-16448}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MINISGL_MAX_SEQ_LEN}"
export AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-8,16,24,32,48,64}"
export AFD_MEMORY_RATIO="${AFD_MEMORY_RATIO:-0.82}"

PROMPT_BASE_16K_FILE="${PROMPT_BASE_16K_FILE:-$CALIB_DIR/prompts/prompts_512x16384_seed20260527.txt}"
if [[ -z "${PROMPT_BASE_FILE:-}" ]]; then
  if [[ ! -f "$PROMPT_BASE_16K_FILE" ]]; then
    echo "Generating 16k prompt base: $PROMPT_BASE_16K_FILE"
    PYTHONPATH="$CALIB_DIR/python${PYTHONPATH:+:$PYTHONPATH}" \
      python "$CALIB_DIR/scripts/data_gen/generate_realistic_long_prompts.py" \
        --model "$MODEL_PATH" \
        --target-tokens "$PROMPT_LEN" \
        --count 512 \
        --output "$PROMPT_BASE_16K_FILE" \
        --seed 20260527 \
        --cache-dir "$CALIB_DIR/cache/real_corpus_sources" \
        --min-source-tokens 20000
  fi
  export PROMPT_BASE_FILE="$PROMPT_BASE_16K_FILE"
fi

AFD_MEMORY_TAG="${AFD_MEMORY_RATIO/./}"
export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_qwen3_235b_a22b_fp8_16k_b${PER_ATTN_GPU_BSZ}_peragpu_${AFD_TOTAL_NODES}node_mem${AFD_MEMORY_TAG}_bucket${AFD_MAX_BATCHED_TOKENS}_wavefix_${TIMESTAMP}}"

exec bash "$SCRIPT_DIR/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh"
