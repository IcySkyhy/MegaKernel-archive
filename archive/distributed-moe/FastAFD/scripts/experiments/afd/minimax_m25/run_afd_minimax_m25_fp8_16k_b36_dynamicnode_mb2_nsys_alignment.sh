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

# MiniMax M2.5 FP8 AFD AG/EG large-node 16k decode replay workload.
#
# This is a thin wrapper around the MiniMax 8k dynamic-node runner.  The Ray
# placement, AFD topology, MegaMoE settings, nsys, and optional vLLM alignment
# logic stay shared there; this file only changes the long-context workload
# shape.
#
# 16k doubles KV tokens/request relative to the 8k b72 workload, so the default
# per-attention-GPU batch is halved to 36.  With mb=2 this replays an 18-request
# graph per attention GPU per microbatch.
DEFAULT_MODEL="$HOME/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/snapshots/f710177d938eff80b684d42c5aa84b382612f21f"
if [[ ! -d "$DEFAULT_MODEL" ]]; then
  DEFAULT_MODEL="MiniMaxAI/MiniMax-M2.5"
fi
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
export PROMPT_LEN="${PROMPT_LEN:-16384}"
export PER_ATTN_GPU_BSZ="${PER_ATTN_GPU_BSZ:-36}"
export MAX_TOKENS="${MAX_TOKENS:-16}"

export AFD_TOTAL_NODES="${AFD_TOTAL_NODES:-${NUM_NODES:-4}}"
export AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-512}"
export MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-16640}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MINISGL_MAX_SEQ_LEN}"
export AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-8,16,18,24,32,36,48,64}"
export AFD_MEMORY_RATIO="${AFD_MEMORY_RATIO:-0.79}"

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
export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_minimax_m25_fp8_16k_b${PER_ATTN_GPU_BSZ}_peragpu_${AFD_TOTAL_NODES}node_mem${AFD_MEMORY_TAG}_bucket${AFD_MAX_BATCHED_TOKENS}_${TIMESTAMP}}"

exec bash "$SCRIPT_DIR/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh"
