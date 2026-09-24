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
MINISGL_TP_SIZE="${MINISGL_TP_SIZE:-8}"
ATTN_TP_SIZE="${ATTN_TP_SIZE:-}"
MLP_TP_SIZE="${MLP_TP_SIZE:-}"
ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}"
MLP_DP_SIZE="${MLP_DP_SIZE:-1}"
AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-none}"
AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-deep_gemm}"
if [[ ! -v VLLM_TP_SIZE ]]; then
  if [[ "$MODEL_PATH" == *FP8* || "$MODEL_PATH" == *fp8* ]]; then
    VLLM_TP_SIZE="1"
  else
    VLLM_TP_SIZE="4"
  fi
fi
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
MINISGL_PORT="${MINISGL_PORT:-}"
VLLM_PORT="${VLLM_PORT:-}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-5}"
PROMPT_REPEAT="${PROMPT_REPEAT:-4}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-64}"
MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-}"
MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-64}"
MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-2048}"
MINISGL_PAGE_SIZE="${MINISGL_PAGE_SIZE:-1}"
MINISGL_LAUNCH_BACKEND="${MINISGL_LAUNCH_BACKEND:-ray}"
MINISGL_RAY_ADDRESS="${MINISGL_RAY_ADDRESS:-auto}"
AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-64}"
AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-}"
AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-8192}"
AFD_CACHE_TYPE="${AFD_CACHE_TYPE:-naive}"
AFD_DEVICE_COMM_NUM_SMS="${AFD_DEVICE_COMM_NUM_SMS:-4}"
NSYS="${NSYS:-0}"
NSYS_OUTPUT_PREFIX="${NSYS_OUTPUT_PREFIX:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_JSON:-$REPORT_DIR/qwen3_30b_a3b_afd_serve_alignment_tp8_ray_${TIMESTAMP}.json}"

if [[ "$MINISGL_LAUNCH_BACKEND" != "ray" ]]; then
  echo "AFD online wrapper currently only supports MINISGL_LAUNCH_BACKEND=ray" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

# Size the AFD running table to the whole workload, matching the large-scale
# presets (max_running_req = num_prompts). AFD prefill is globally prioritized,
# so if batch-submitted prompts exceed max_running_req the unadmitted prompts
# keep the scheduler in "prefill" forever and decode never runs -> deadlock.
# num_prompts = non-blank prompt lines x PROMPT_REPEAT (matches load_prompts()).
# Set MINISGL_MAX_RUNNING_REQUESTS explicitly to pin a smaller table.
if [[ -z "$MINISGL_MAX_RUNNING_REQUESTS" ]]; then
  _prompt_count="$(grep -cve '^[[:space:]]*$' "$PROMPT_FILE")"
  MINISGL_MAX_RUNNING_REQUESTS="$(( _prompt_count * PROMPT_REPEAT ))"
fi

if [[ -z "$ATTN_TP_SIZE" || -z "$MLP_TP_SIZE" ]]; then
  if (( MINISGL_TP_SIZE % 2 != 0 )); then
    echo "MINISGL_TP_SIZE must be even when auto-splitting to attention/model TP sizes" >&2
    exit 1
  fi
  half_tp=$((MINISGL_TP_SIZE / 2))
  ATTN_TP_SIZE="${ATTN_TP_SIZE:-$half_tp}"
  MLP_TP_SIZE="${MLP_TP_SIZE:-$half_tp}"
fi

echo "Running Qwen3-30B-A3B AFD online alignment report"
echo "  model: $MODEL_PATH"
echo "  prompts: $PROMPT_FILE"
echo "  prompt_repeat: $PROMPT_REPEAT"
echo "  gpus: $GPUS"
echo "  minisgl_tp_size: $MINISGL_TP_SIZE"
echo "  afd_attn_tp_size: $ATTN_TP_SIZE"
echo "  afd_mlp_tp_size: $MLP_TP_SIZE"
echo "  afd_attn_dp_size: $ATTN_DP_SIZE"
echo "  afd_mlp_dp_size: $MLP_DP_SIZE"
echo "  afd_mlp_ep_size: $AFD_MLP_EP_SIZE"
echo "  afd_moe_a2a_backend: $AFD_MOE_A2A_BACKEND"
echo "  afd_moe_runner_backend: $AFD_MOE_RUNNER_BACKEND"
echo "  vllm_tp_size: $VLLM_TP_SIZE"
echo "  minisgl_launch_backend: $MINISGL_LAUNCH_BACKEND"
echo "  minisgl_ray_address: $MINISGL_RAY_ADDRESS"
echo "  sample_concurrency: $SAMPLE_CONCURRENCY"
echo "  minisgl_max_running_requests: $MINISGL_MAX_RUNNING_REQUESTS"
echo "  minisgl_cuda_graph_max_bs: $MINISGL_CUDA_GRAPH_MAX_BS"
echo "  afd_decode_graph_bs: ${AFD_DECODE_GRAPH_BS:-<auto>}"
echo "  afd_max_batched_tokens: $AFD_MAX_BATCHED_TOKENS"
echo "  afd_cache_type: $AFD_CACHE_TYPE"
echo "  minisgl_page_size: $MINISGL_PAGE_SIZE"
echo "  afd_device_comm_num_sms: $AFD_DEVICE_COMM_NUM_SMS"
echo "  gin_n2m_protocol: split"
if [[ "$NSYS" == "1" ]]; then
  echo "  ray_nsys: 1"
  if [[ -n "$NSYS_OUTPUT_PREFIX" ]]; then
    echo "  ray_nsys_output_prefix: $NSYS_OUTPUT_PREFIX"
  fi
fi
echo "  output_json: $OUTPUT_JSON"

cmd=(
  "$SCRIPT_ROOT/validate/fastafd_vllm_alignment.sh"
  --env "$ENV_NAME"
  --model "$MODEL_PATH"
  --prompt-file "$PROMPT_FILE"
  --prompt-repeat "$PROMPT_REPEAT"
  --batch-size "$AFD_BATCH_SIZE"
  --max-new-tokens "$MAX_TOKENS"
  --cuda-graph-max-bs "$MINISGL_CUDA_GRAPH_MAX_BS"
  --max-batched-tokens "$AFD_MAX_BATCHED_TOKENS"
  --cache-type "$AFD_CACHE_TYPE"
  --afd-max-running-requests "$MINISGL_MAX_RUNNING_REQUESTS"
  --gpus "$GPUS"
  --afd-attn-dp-size "$ATTN_DP_SIZE"
  --afd-mlp-dp-size "$MLP_DP_SIZE"
  --attn-tp-size "$ATTN_TP_SIZE"
  --mlp-tp-size "$MLP_TP_SIZE"
  --afd-mlp-ep-size "$AFD_MLP_EP_SIZE"
  --afd-moe-a2a-backend "$AFD_MOE_A2A_BACKEND"
  --afd-moe-runner-backend "$AFD_MOE_RUNNER_BACKEND"
  --vllm-tp-size "$VLLM_TP_SIZE"
  --ray-address "$MINISGL_RAY_ADDRESS"
  --host "$SERVER_HOST"
  --afd-port "$MINISGL_PORT"
  --vllm-port "$VLLM_PORT"
  --prompt-logprobs "$PROMPT_LOGPROBS"
  --sample-concurrency "$SAMPLE_CONCURRENCY"
  --max-seq-len-override "$MINISGL_MAX_SEQ_LEN"
  --page-size "$MINISGL_PAGE_SIZE"
  --vllm-max-model-len "$VLLM_MAX_MODEL_LEN"
  --output-json "$OUTPUT_JSON"
  --keep-artifacts
)

if [[ -n "$AFD_DECODE_GRAPH_BS" ]]; then
  cmd+=(--decode-graph-bs "$AFD_DECODE_GRAPH_BS")
fi
cmd+=(--afd-device-comm-num-sms "$AFD_DEVICE_COMM_NUM_SMS")
if [[ "$NSYS" == "1" ]]; then
  cmd+=(--ray-nsys)
fi
if [[ -n "$NSYS_OUTPUT_PREFIX" ]]; then
  cmd+=(--ray-nsys-output-prefix "$NSYS_OUTPUT_PREFIX")
fi

exec "${cmd[@]}"
