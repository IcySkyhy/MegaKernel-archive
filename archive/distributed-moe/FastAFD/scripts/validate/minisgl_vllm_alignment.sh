#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_alignment_report.sh --model PATH --prompt-file FILE [options]

This runs a sequential pipeline on the same GPUs:
1. Start mini-sgl and sample deterministic outputs.
2. Stop mini-sgl.
3. Start vLLM and score the sampled tokens with prompt_logprobs.
4. Print the final report.

Options:
  --env NAME                  Expected active conda env name. Default: minisgl-cuda130
  --model PATH                Model path or HF repo. Required.
  --prompt-file FILE          UTF-8 prompt file. Required.
  --prompt-repeat N           Repeat the loaded prompt set in memory. Default: 1
  --gpus CSV                  CUDA_VISIBLE_DEVICES for both phases. Default: 0,1,2,3
  --tp-size N                 Tensor parallel size for both phases. Default: number of GPUs in --gpus
  --minisgl-tp-size N         Tensor parallel size for mini-sgl only. Overrides --tp-size.
  --vllm-tp-size N            Tensor parallel size for vLLM only. Overrides --tp-size.
  --host HOST                 Default: 127.0.0.1
  --minisgl-port PORT         Default: 19193
  --vllm-port PORT            Default: 18000
  --max-tokens N              Default: 64
  --temperature FLOAT         Default: 0
  --top-p FLOAT               Default: 1
  --top-k N                   Default: 1
  --prompt-logprobs N         Default: 5
  --sample-concurrency N      Concurrent mini-sgl sampling requests. Default: 1
  --minisgl-max-running-requests N
                              mini-sgl scheduler limit. Default: 1
  --minisgl-cuda-graph-max-bs N
                              mini-sgl CUDA graph capture ceiling. Default: 1
  --minisgl-max-seq-len N     Optional
  --minisgl-launch-backend NAME
                              Optional mini-sgl launch backend override
  --minisgl-ray-address ADDR  Optional Ray address when mini-sgl launch backend is ray
  --minisgl-ray-nsys          Enable Nsight Systems profiling for Ray mini-sgl workers
  --minisgl-ray-nsys-output-prefix PATH
                              Optional Nsight Systems output prefix for Ray mini-sgl workers
  --vllm-max-model-len N      Optional
  --output-json FILE          Optional final report JSON path
  --keep-artifacts            Keep temp sample/log files and print their paths

Environment passthrough:
  MINISGL_EXTRA_ARGS          Extra args appended to mini-sgl start command
  VLLM_EXTRA_ARGS             Extra args appended to vLLM start command

Artifacts under the temp/output directory:
  minisgl.log                 mini-sgl driver stdout/stderr
  ray_logs/                   Per-rank Ray scheduler logs when launch-backend=ray
  sample.log                  Sample command stdout/stderr
  sample.json                 Sample bundle written after successful sampling
  vllm.log                    vLLM stdout/stderr
EOF
}

count_csv_items() {
  local csv="$1"
  python - "$csv" <<'PY'
import sys
items = [x for x in sys.argv[1].split(",") if x]
print(len(items))
PY
}

wait_for_ready() {
  local url="$1"
  local name="$2"
  local log_file="$3"
  local pid="$4"
  local timeout_s="${5:-900}"
  local deadline=$((SECONDS + timeout_s))
  local last_error=""

  while (( SECONDS < deadline )); do
    if ! process_is_running "$pid"; then
      return 2
    fi
    if last_error="$(try_http_ready "$url" 2>&1)"; then
      echo "$name ready: $url"
      return 0
    fi
    sleep 2
  done

  echo "$name failed to become ready: $url" >&2
  echo "See log: $log_file" >&2
  if [[ -n "$last_error" ]]; then
    echo "Last error: $last_error" >&2
  fi
  return 1
}

process_is_running() {
  local pid="$1"
  local stat
  stat="$(ps -o stat= -p "$pid" 2>/dev/null | awk 'NR==1 {print $1}')"
  [[ -n "$stat" && "$stat" != Z* ]]
}

wait_child_status() {
  local pid="$1"
  local status
  set +e
  wait "$pid"
  status=$?
  set -e
  echo "$status"
}

print_log_tail() {
  local log_file="$1"
  local lines="${2:-120}"
  if [[ -f "$log_file" ]]; then
    echo "--- tail -n $lines $log_file ---" >&2
    tail -n "$lines" "$log_file" >&2 || true
    echo "--- end log tail ---" >&2
  fi
}

fail_with_logs() {
  local message="$1"
  local exit_code="$2"
  shift 2

  echo "$message" >&2
  for log_file in "$@"; do
    echo "See log: $log_file" >&2
    print_log_tail "$log_file"
  done
  exit "$exit_code"
}

try_http_ready() {
  local url="$1"
  python - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=5) as response:
    response.read()
PY
}

log_step() {
  local message="$1"
  printf '\n==> %s\n' "$message"
}

ENV_NAME="minisgl-cuda130"
MODEL=""
PROMPT_FILE=""
GPUS="0,1,2,3"
TP_SIZE=""
MINISGL_TP_SIZE=""
VLLM_TP_SIZE=""
SERVER_HOST="127.0.0.1"
MINISGL_PORT="19193"
VLLM_PORT="18000"
MAX_TOKENS="64"
TEMPERATURE="0"
TOP_P="1"
TOP_K="1"
PROMPT_LOGPROBS="5"
SAMPLE_CONCURRENCY="1"
PROMPT_REPEAT="1"
MINISGL_MAX_RUNNING_REQUESTS="1"
MINISGL_CUDA_GRAPH_MAX_BS="1"
MINISGL_MAX_SEQ_LEN=""
MINISGL_LAUNCH_BACKEND=""
MINISGL_RAY_ADDRESS=""
MINISGL_RAY_NSYS=0
MINISGL_RAY_NSYS_OUTPUT_PREFIX=""
VLLM_MAX_MODEL_LEN=""
OUTPUT_JSON=""
KEEP_ARTIFACTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --prompt-repeat) PROMPT_REPEAT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --minisgl-tp-size) MINISGL_TP_SIZE="$2"; shift 2 ;;
    --vllm-tp-size) VLLM_TP_SIZE="$2"; shift 2 ;;
    --host) SERVER_HOST="$2"; shift 2 ;;
    --minisgl-port) MINISGL_PORT="$2"; shift 2 ;;
    --vllm-port) VLLM_PORT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --prompt-logprobs) PROMPT_LOGPROBS="$2"; shift 2 ;;
    --sample-concurrency) SAMPLE_CONCURRENCY="$2"; shift 2 ;;
    --minisgl-max-running-requests) MINISGL_MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --minisgl-cuda-graph-max-bs) MINISGL_CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
    --minisgl-max-seq-len) MINISGL_MAX_SEQ_LEN="$2"; shift 2 ;;
    --minisgl-launch-backend) MINISGL_LAUNCH_BACKEND="$2"; shift 2 ;;
    --minisgl-ray-address) MINISGL_RAY_ADDRESS="$2"; shift 2 ;;
    --minisgl-ray-nsys) MINISGL_RAY_NSYS=1; shift ;;
    --minisgl-ray-nsys-output-prefix) MINISGL_RAY_NSYS_OUTPUT_PREFIX="$2"; shift 2 ;;
    --vllm-max-model-len) VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --keep-artifacts) KEEP_ARTIFACTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -v VLLM_EXTRA_ARGS ]]; then
  VLLM_EXTRA_ARGS="--enforce-eager"
fi

if [[ -z "$MODEL" || -z "$PROMPT_FILE" ]]; then
  echo "--model and --prompt-file are required" >&2
  usage >&2
  exit 1
fi

if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$(count_csv_items "$GPUS")"
fi
if [[ -z "$MINISGL_TP_SIZE" ]]; then
  MINISGL_TP_SIZE="$TP_SIZE"
fi
if [[ -z "$VLLM_TP_SIZE" ]]; then
  VLLM_TP_SIZE="$TP_SIZE"
fi

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
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
TMP_DIR="${TMP_DIR:-$REPORT_DIR/alignment_run_${TIMESTAMP}}"
mkdir -p "$TMP_DIR"
SAMPLE_JSON="$TMP_DIR/sample.json"
SAMPLE_LOG="$TMP_DIR/sample.log"
MINISGL_LOG="$TMP_DIR/minisgl.log"
VLLM_LOG="$TMP_DIR/vllm.log"
MINISGL_RAY_LOG_DIR="$TMP_DIR/ray_logs"
MINISGL_PID=""
VLLM_PID=""
SAMPLE_PID=""

cleanup() {
  set +e
  if [[ -n "$SAMPLE_PID" ]]; then
    if process_is_running "$SAMPLE_PID"; then
      stop_process_tree "$SAMPLE_PID"
    fi
    wait "$SAMPLE_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$MINISGL_PID" ]] && process_is_running "$MINISGL_PID"; then
    stop_process_tree "$MINISGL_PID"
    wait "$MINISGL_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$VLLM_PID" ]] && process_is_running "$VLLM_PID"; then
    stop_process_tree "$VLLM_PID"
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$KEEP_ARTIFACTS" -eq 0 ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

kill_tree() {
  local pid="$1"
  local signal_name="${2:-TERM}"
  local children
  children="$(pgrep -P "$pid" || true)"
  for child in $children; do
    kill_tree "$child" "$signal_name"
  done
  kill "-${signal_name}" "$pid" 2>/dev/null || true
}

stop_process_tree() {
  local pid="$1"
  local timeout_s="${2:-10}"
  local waited=0

  if ! process_is_running "$pid"; then
    return 0
  fi

  kill_tree "$pid" TERM
  while process_is_running "$pid"; do
    if (( waited >= timeout_s * 10 )); then
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done

  if process_is_running "$pid"; then
    kill_tree "$pid" KILL
    for _ in $(seq 1 50); do
      if ! process_is_running "$pid"; then
        break
      fi
      sleep 0.1
    done
  fi
}

ensure_active_env() {
  local expected="$1"
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Expected active conda env '$expected', but no conda env is activated." >&2
    exit 1
  fi
  if [[ "$CONDA_DEFAULT_ENV" != "$expected" ]]; then
    echo "Expected active conda env '$expected', got '$CONDA_DEFAULT_ENV'." >&2
    exit 1
  fi
}

ensure_active_env "$ENV_NAME"

print_sample_preview() {
  local sample_json="$1"
  python - <<'PY' "$sample_json"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

summary = payload.get("summary", {})
print("Sample Preview")
for key in ("num_prompts", "total_generated_tokens"):
    if key in summary:
        print(f"  {key}: {summary[key]}")

def shorten(text: str, limit: int = 160) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."

for idx, sample in enumerate(payload.get("samples", [])[:3], start=1):
    print(f"  [{idx}] prompt: {shorten(sample.get('prompt', ''))}")
    print(f"      output: {shorten(sample.get('generated_text', ''))}")
PY
}

minisgl_start_cmd=(
  "$SCRIPT_ROOT/serve/minisgl_server.sh"
  --env "$ENV_NAME"
  --model "$MODEL"
  --gpus "$GPUS"
  --host "$SERVER_HOST"
  --port "$MINISGL_PORT"
  --tp-size "$MINISGL_TP_SIZE"
  --max-running-requests "$MINISGL_MAX_RUNNING_REQUESTS"
  --cuda-graph-max-bs "$MINISGL_CUDA_GRAPH_MAX_BS"
  --num-pages "${MINISGL_NUM_PAGES:-2048}"
  --attention-backend trtllm
  --ray-log-dir "$MINISGL_RAY_LOG_DIR"
  --extra-args "${MINISGL_EXTRA_ARGS:-}"
)
if [[ -n "$MINISGL_LAUNCH_BACKEND" ]]; then
  minisgl_start_cmd+=(--launch-backend "$MINISGL_LAUNCH_BACKEND")
fi
if [[ -n "$MINISGL_RAY_ADDRESS" ]]; then
  minisgl_start_cmd+=(--ray-address "$MINISGL_RAY_ADDRESS")
fi
if [[ "$MINISGL_RAY_NSYS" -eq 1 ]]; then
  minisgl_start_cmd+=(--ray-nsys)
fi
if [[ -n "$MINISGL_RAY_NSYS_OUTPUT_PREFIX" ]]; then
  minisgl_start_cmd+=(--ray-nsys-output-prefix "$MINISGL_RAY_NSYS_OUTPUT_PREFIX")
fi
if [[ -n "$MINISGL_MAX_SEQ_LEN" ]]; then
  minisgl_start_cmd+=(--max-seq-len-override "$MINISGL_MAX_SEQ_LEN")
fi

log_step "Starting mini-sgl"
echo "  active_env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "  gpus: $GPUS"
if [[ -n "$MINISGL_LAUNCH_BACKEND" ]]; then
  echo "  minisgl_launch_backend: $MINISGL_LAUNCH_BACKEND"
fi
if [[ -n "$MINISGL_RAY_ADDRESS" ]]; then
  echo "  minisgl_ray_address: $MINISGL_RAY_ADDRESS"
fi
if [[ "$MINISGL_RAY_NSYS" -eq 1 ]]; then
  echo "  minisgl_ray_nsys: 1"
  if [[ -n "$MINISGL_RAY_NSYS_OUTPUT_PREFIX" ]]; then
    echo "  minisgl_ray_nsys_output_prefix: $MINISGL_RAY_NSYS_OUTPUT_PREFIX"
  fi
fi
echo "  minisgl_tp_size: $MINISGL_TP_SIZE"
echo "  minisgl_url: http://$SERVER_HOST:$MINISGL_PORT/v1"
echo "  minisgl_log: $MINISGL_LOG"
echo "  ray_log_dir: $MINISGL_RAY_LOG_DIR"
echo "  sample_log: $SAMPLE_LOG"
echo "  sample_json: $SAMPLE_JSON"
echo "  vllm_log: $VLLM_LOG"
if [[ -n "$OUTPUT_JSON" ]]; then
  echo "  output_json: $OUTPUT_JSON"
fi
"${minisgl_start_cmd[@]}" >"$MINISGL_LOG" 2>&1 &
MINISGL_PID=$!

set +e
wait_for_ready "http://$SERVER_HOST:$MINISGL_PORT/v1" "mini-sgl" "$MINISGL_LOG" "$MINISGL_PID"
ready_status=$?
set -e
if (( ready_status == 2 )); then
  minisgl_status="$(wait_child_status "$MINISGL_PID")"
  MINISGL_PID=""
  fail_with_logs \
    "mini-sgl exited before becoming ready (exit code $minisgl_status)." \
    1 \
    "$MINISGL_LOG"
elif (( ready_status != 0 )); then
  fail_with_logs \
    "mini-sgl failed to become ready within the startup timeout." \
    1 \
    "$MINISGL_LOG"
fi

log_step "Sampling deterministic outputs from mini-sgl"
echo "  sample_concurrency: $SAMPLE_CONCURRENCY"
echo "  prompt_repeat: $PROMPT_REPEAT"
python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" sample \
  --model "$MODEL" \
  --minisgl-url "http://$SERVER_HOST:$MINISGL_PORT" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-repeat "$PROMPT_REPEAT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --sample-concurrency "$SAMPLE_CONCURRENCY" \
  --sample-json "$SAMPLE_JSON" >"$SAMPLE_LOG" 2>&1 &
SAMPLE_PID=$!

sample_status=0
while true; do
  if ! process_is_running "$MINISGL_PID"; then
    minisgl_status="$(wait_child_status "$MINISGL_PID")"
    MINISGL_PID=""
    if process_is_running "$SAMPLE_PID"; then
      stop_process_tree "$SAMPLE_PID"
      set +e
      wait "$SAMPLE_PID" >/dev/null 2>&1 || true
      set -e
    fi
    SAMPLE_PID=""
    fail_with_logs \
      "mini-sgl exited during sampling (exit code $minisgl_status)." \
      1 \
      "$MINISGL_LOG" \
      "$SAMPLE_LOG"
  fi

  if ! process_is_running "$SAMPLE_PID"; then
    sample_status="$(wait_child_status "$SAMPLE_PID")"
    SAMPLE_PID=""
    break
  fi

  sleep 0.2
done

if (( sample_status != 0 )); then
  fail_with_logs \
    "Sampling command failed (exit code $sample_status)." \
    "$sample_status" \
    "$SAMPLE_LOG" \
    "$MINISGL_LOG"
fi

print_sample_preview "$SAMPLE_JSON"

stop_process_tree "$MINISGL_PID"
wait "$MINISGL_PID" >/dev/null 2>&1 || true
MINISGL_PID=""
log_step "mini-sgl stopped"

vllm_start_cmd=(
  "$SCRIPT_ROOT/serve/vllm_server.sh"
  --env "$ENV_NAME"
  --model "$MODEL"
  --gpus "$GPUS"
  --host "$SERVER_HOST"
  --port "$VLLM_PORT"
  --tp-size "$VLLM_TP_SIZE"
  --extra-args "${VLLM_EXTRA_ARGS:-}"
)
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  vllm_start_cmd+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi

log_step "Starting vLLM"
echo "  gpus: $GPUS"
echo "  vllm_tp_size: $VLLM_TP_SIZE"
echo "  vllm_url: http://$SERVER_HOST:$VLLM_PORT/v1/models"
echo "  vllm_log: $VLLM_LOG"
"${vllm_start_cmd[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

set +e
wait_for_ready "http://$SERVER_HOST:$VLLM_PORT/v1/models" "vLLM" "$VLLM_LOG" "$VLLM_PID"
vllm_ready_status=$?
set -e
if (( vllm_ready_status == 2 )); then
  vllm_status="$(wait_child_status "$VLLM_PID")"
  VLLM_PID=""
  fail_with_logs \
    "vLLM exited before becoming ready (exit code $vllm_status)." \
    1 \
    "$VLLM_LOG"
elif (( vllm_ready_status != 0 )); then
  fail_with_logs \
    "vLLM failed to become ready within the startup timeout." \
    1 \
    "$VLLM_LOG"
fi

log_step "Scoring sampled outputs with vLLM"
echo "  prompt_logprobs: $PROMPT_LOGPROBS"
score_cmd=(
  python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" score
  --sample-json "$SAMPLE_JSON"
  --model "$MODEL"
  --vllm-url "http://$SERVER_HOST:$VLLM_PORT"
  --prompt-logprobs "$PROMPT_LOGPROBS"
)
if [[ -n "$OUTPUT_JSON" ]]; then
  score_cmd+=(--output-json "$OUTPUT_JSON")
fi
"${score_cmd[@]}"
log_step "Alignment finished"

if [[ "$KEEP_ARTIFACTS" -eq 1 ]]; then
  echo "Artifacts kept under: $TMP_DIR"
  echo "  sample_log: $SAMPLE_LOG"
  echo "  sample_json: $SAMPLE_JSON"
  echo "  minisgl_log: $MINISGL_LOG"
  echo "  vllm_log: $VLLM_LOG"
  if [[ -n "$OUTPUT_JSON" ]]; then
    echo "  output_json: $OUTPUT_JSON"
  fi
fi
