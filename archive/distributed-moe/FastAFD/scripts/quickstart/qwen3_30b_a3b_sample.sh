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
STAMP="$(date -u +%Y%m%d_%H%M%S)"

ENV_NAME="${ENV_NAME:-minisgl-cuda130}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PROMPT_FILE="${PROMPT_FILE:-$CALIB_DIR/prompts/qwen3_30b_alignment.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$CALIB_DIR/reports/qwen3_30b_a3b_minisgl_sample_${STAMP}}"
GPUS="${GPUS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
PROMPT_REPEAT="${PROMPT_REPEAT:-4}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-64}"
MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-64}"
MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-64}"
MINISGL_NUM_PAGES="${MINISGL_NUM_PAGES:-2048}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PORT="${PORT:-19193}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PRINT_SAMPLES="${PRINT_SAMPLES:-3}"
PREVIEW_CHARS="${PREVIEW_CHARS:-160}"
MOE_BACKEND="${MOE_BACKEND:-fused}"
MINISGL_LAUNCH_BACKEND="${MINISGL_LAUNCH_BACKEND:-mp}"
MINISGL_RAY_ADDRESS="${MINISGL_RAY_ADDRESS:-auto}"
CONTROL_HOST="${CONTROL_HOST:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Running Qwen3-30B-A3B mini-sgl sample"
echo "  env: $ENV_NAME"
echo "  model: $MODEL_PATH"
echo "  prompt_file: $PROMPT_FILE"
echo "  output_dir: $OUTPUT_DIR"
echo "  minisgl_launch_backend: $MINISGL_LAUNCH_BACKEND"
if [[ "$MINISGL_LAUNCH_BACKEND" == "ray" ]]; then
  echo "  minisgl_ray_address: $MINISGL_RAY_ADDRESS"
fi

cmd=(
  "$SCRIPT_ROOT/quickstart/sample_minisgl.sh"
  --env "$ENV_NAME"
  --model "$MODEL_PATH"
  --prompt-file "$PROMPT_FILE"
  --prompt-repeat "$PROMPT_REPEAT"
  --gpus "$GPUS"
  --tp-size "$TP_SIZE"
  --host "$SERVER_HOST"
  --port "$PORT"
  --max-tokens "$MAX_TOKENS"
  --sample-concurrency "$SAMPLE_CONCURRENCY"
  --max-running-requests "$MINISGL_MAX_RUNNING_REQUESTS"
  --cuda-graph-max-bs "$MINISGL_CUDA_GRAPH_MAX_BS"
  --num-pages "$MINISGL_NUM_PAGES"
  --max-seq-len "$MAX_SEQ_LEN"
  --moe-backend "$MOE_BACKEND"
  --launch-backend "$MINISGL_LAUNCH_BACKEND"
  --ray-address "$MINISGL_RAY_ADDRESS"
  --output-dir "$OUTPUT_DIR"
  --print-samples "$PRINT_SAMPLES"
  --preview-chars "$PREVIEW_CHARS"
)
if [[ -n "$CONTROL_HOST" ]]; then
  cmd+=(--control-host "$CONTROL_HOST")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  cmd+=(--extra-args "$EXTRA_ARGS")
fi

exec "${cmd[@]}"
