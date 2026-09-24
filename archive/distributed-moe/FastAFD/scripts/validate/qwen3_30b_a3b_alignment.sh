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
REPORT_DIR="$CALIB_DIR/reports"
mkdir -p "$REPORT_DIR"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PROMPT_FILE="${PROMPT_FILE:-$CALIB_DIR/prompts/qwen3_30b_alignment.txt}"
ENV_NAME="${ENV_NAME:-minisgl-cuda130}"
GPUS="${GPUS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
MINISGL_PORT="${MINISGL_PORT:-19193}"
VLLM_PORT="${VLLM_PORT:-18000}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-5}"
PROMPT_REPEAT="${PROMPT_REPEAT:-4}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-64}"
MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-64}"
MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-64}"
MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-2048}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
MINISGL_LAUNCH_BACKEND="${MINISGL_LAUNCH_BACKEND:-mp}"
MINISGL_RAY_ADDRESS="${MINISGL_RAY_ADDRESS:-auto}"
CONTROL_HOST="${CONTROL_HOST:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_JSON:-$REPORT_DIR/qwen3_30b_a3b_alignment_${TIMESTAMP}.json}"

if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* || "$MODEL_PATH" == ../* || "$MODEL_PATH" == ~* ]]; then
  if [[ ! -e "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH" >&2
    exit 1
  fi
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

BASE_MINISGL_EXTRA_ARGS="--moe-backend fused"
if [[ -n "$CONTROL_HOST" ]]; then
  BASE_MINISGL_EXTRA_ARGS+=" --control-host $CONTROL_HOST"
fi
if [[ -n "${MINISGL_EXTRA_ARGS:-}" ]]; then
  export MINISGL_EXTRA_ARGS="$BASE_MINISGL_EXTRA_ARGS ${MINISGL_EXTRA_ARGS}"
else
  export MINISGL_EXTRA_ARGS="$BASE_MINISGL_EXTRA_ARGS"
fi

echo "Running Qwen3-30B-A3B alignment report"
echo "  model: $MODEL_PATH"
echo "  prompts: $PROMPT_FILE"
echo "  prompt_repeat: $PROMPT_REPEAT"
echo "  gpus: $GPUS"
echo "  tp_size: $TP_SIZE"
echo "  minisgl_launch_backend: $MINISGL_LAUNCH_BACKEND"
if [[ "$MINISGL_LAUNCH_BACKEND" == "ray" ]]; then
  echo "  minisgl_ray_address: $MINISGL_RAY_ADDRESS"
fi
echo "  sample_concurrency: $SAMPLE_CONCURRENCY"
echo "  minisgl_max_running_requests: $MINISGL_MAX_RUNNING_REQUESTS"
echo "  minisgl_cuda_graph_max_bs: $MINISGL_CUDA_GRAPH_MAX_BS"
echo "  output_json: $OUTPUT_JSON"

exec "$SCRIPT_ROOT/validate/minisgl_vllm_alignment.sh" \
  --env "$ENV_NAME" \
  --model "$MODEL_PATH" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-repeat "$PROMPT_REPEAT" \
  --gpus "$GPUS" \
  --tp-size "$TP_SIZE" \
  --minisgl-launch-backend "$MINISGL_LAUNCH_BACKEND" \
  --minisgl-ray-address "$MINISGL_RAY_ADDRESS" \
  --host "$SERVER_HOST" \
  --minisgl-port "$MINISGL_PORT" \
  --vllm-port "$VLLM_PORT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --top-p 1 \
  --top-k 1 \
  --prompt-logprobs "$PROMPT_LOGPROBS" \
  --sample-concurrency "$SAMPLE_CONCURRENCY" \
  --minisgl-max-running-requests "$MINISGL_MAX_RUNNING_REQUESTS" \
  --minisgl-cuda-graph-max-bs "$MINISGL_CUDA_GRAPH_MAX_BS" \
  --minisgl-max-seq-len "$MINISGL_MAX_SEQ_LEN" \
  --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
  --output-json "$OUTPUT_JSON" \
  --keep-artifacts
