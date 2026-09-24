#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_minisgl_sample.sh --env NAME --model PATH --prompt-file FILE [options]

This runs a mini-sgl-only sampling pipeline:
1. Start mini-sgl.
2. Sample prompts through the OpenAI-compatible API.
3. Stop mini-sgl.
4. Save sample.json and minisgl.log under an output directory.

Options:
  --env NAME                  Expected active conda env name. Required.
  --model PATH                Model path or HF repo. Required.
  --prompt-file FILE          UTF-8 prompt file. Required.
  --prompt-repeat N           Repeat the loaded prompt set in memory. Default: 4
  --gpus CSV                  CUDA_VISIBLE_DEVICES. Default: 0,1,2,3
  --tp-size N                 Tensor parallel size. Default: number of GPUs in --gpus
  --host HOST                 Default: 127.0.0.1
  --port PORT                 Default: 19193
  --max-tokens N              Default: 64
  --temperature FLOAT         Default: 0
  --top-p FLOAT               Default: 1
  --top-k N                   Default: 1
  --sample-concurrency N      Default: 64
  --max-running-requests N    mini-sgl scheduler limit. Default: 64
  --cuda-graph-max-bs N       mini-sgl CUDA graph capture ceiling. Default: 64
  --num-pages N               mini-sgl KV cache pages. Default: 2048
  --max-seq-len N             Optional
  --output-dir DIR            Directory to write artifacts. Required.
  --attention-backend NAME    Default: trtllm
  --moe-backend NAME          Optional
  --memory-ratio FLOAT        Optional
  --launch-backend NAME       Default: ray
  --ray-address ADDR          Default: auto
  --ray-log-dir DIR           Optional; default OUTPUT_DIR/ray_logs when launch-backend=ray
  --ray-nsys                  Enable Nsight Systems profiling for Ray workers
  --ray-nsys-output-prefix PATH
                              Optional Nsight Systems output prefix for Ray workers
  --control-host HOST         Optional; forwarded to mini-sgl via --extra-args
  --disable-pynccl            Pass through to mini-sgl
  --extra-args "..."          Extra args appended to mini-sgl start command
  --print-samples N           Number of sample previews to print. Default: 3
  --preview-chars N           Max chars per printed output preview. Default: 160

Artifacts:
  sample.json                 Sample bundle with prompts, generated_text, token_ids
  minisgl.log                 mini-sgl server stdout/stderr
  run_config.json             Resolved run configuration
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

fail_with_log() {
  local message="$1"
  local log_file="$2"
  local exit_code="${3:-1}"
  echo "$message" >&2
  echo "See log: $log_file" >&2
  print_log_tail "$log_file"
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

wait_for_ready() {
  local url="$1"
  local log_file="$2"
  local pid="$3"
  local timeout_s="${4:-900}"
  local deadline=$((SECONDS + timeout_s))
  local last_error=""

  while (( SECONDS < deadline )); do
    if ! process_is_running "$pid"; then
      return 2
    fi
    if last_error="$(try_http_ready "$url" 2>&1)"; then
      echo "mini-sgl ready: $url"
      return 0
    fi
    sleep 2
  done

  echo "mini-sgl failed to become ready: $url" >&2
  echo "See log: $log_file" >&2
  if [[ -n "$last_error" ]]; then
    echo "Last error: $last_error" >&2
  fi
  return 1
}

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

ENV_NAME=""
MODEL=""
PROMPT_FILE=""
PROMPT_REPEAT="4"
GPUS="0,1,2,3"
TP_SIZE=""
SERVER_HOST="127.0.0.1"
PORT="19193"
MAX_TOKENS="64"
TEMPERATURE="0"
TOP_P="1"
TOP_K="1"
SAMPLE_CONCURRENCY="64"
MAX_RUNNING_REQUESTS="64"
CUDA_GRAPH_MAX_BS="64"
NUM_PAGES="2048"
MAX_SEQ_LEN=""
OUTPUT_DIR=""
ATTENTION_BACKEND="trtllm"
MOE_BACKEND=""
MEMORY_RATIO=""
LAUNCH_BACKEND="ray"
RAY_ADDRESS="auto"
RAY_LOG_DIR=""
RAY_NSYS=0
RAY_NSYS_OUTPUT_PREFIX=""
CONTROL_HOST=""
DISABLE_PYNCCL=0
EXTRA_ARGS=""
PRINT_SAMPLES="3"
PREVIEW_CHARS="160"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --prompt-repeat) PROMPT_REPEAT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --host) SERVER_HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --sample-concurrency) SAMPLE_CONCURRENCY="$2"; shift 2 ;;
    --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --cuda-graph-max-bs) CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
    --num-pages) NUM_PAGES="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --moe-backend) MOE_BACKEND="$2"; shift 2 ;;
    --memory-ratio) MEMORY_RATIO="$2"; shift 2 ;;
    --launch-backend) LAUNCH_BACKEND="$2"; shift 2 ;;
    --ray-address) RAY_ADDRESS="$2"; shift 2 ;;
    --ray-log-dir) RAY_LOG_DIR="$2"; shift 2 ;;
    --ray-nsys) RAY_NSYS=1; shift ;;
    --ray-nsys-output-prefix) RAY_NSYS_OUTPUT_PREFIX="$2"; shift 2 ;;
    --control-host) CONTROL_HOST="$2"; shift 2 ;;
    --disable-pynccl) DISABLE_PYNCCL=1; shift ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --print-samples) PRINT_SAMPLES="$2"; shift 2 ;;
    --preview-chars) PREVIEW_CHARS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$ENV_NAME" || -z "$MODEL" || -z "$PROMPT_FILE" || -z "$OUTPUT_DIR" ]]; then
  echo "--env, --model, --prompt-file, and --output-dir are required" >&2
  usage >&2
  exit 1
fi

ensure_active_env "$ENV_NAME"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$(count_csv_items "$GPUS")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"
OUTPUT_DIR="$(python - <<'PY' "$OUTPUT_DIR"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
mkdir -p "$OUTPUT_DIR"
if [[ "$LAUNCH_BACKEND" == "ray" && -z "$RAY_LOG_DIR" ]]; then
  RAY_LOG_DIR="$OUTPUT_DIR/ray_logs"
fi

SAMPLE_JSON="$OUTPUT_DIR/sample.json"
MINISGL_LOG="$OUTPUT_DIR/minisgl.log"
RUN_CONFIG_JSON="$OUTPUT_DIR/run_config.json"
MINISGL_PID=""
SAMPLE_PID=""

cleanup() {
  set +e
  if [[ -n "$SAMPLE_PID" ]]; then
    if process_is_running "$SAMPLE_PID"; then
      stop_process_tree "$SAMPLE_PID"
    fi
    wait "$SAMPLE_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$MINISGL_PID" ]]; then
    if process_is_running "$MINISGL_PID"; then
      stop_process_tree "$MINISGL_PID"
    fi
    wait "$MINISGL_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export MINISGL_EXTRA_ARGS="$EXTRA_ARGS"

start_cmd=(
  "$SCRIPT_ROOT/serve/minisgl_server.sh"
  --env "$ENV_NAME"
  --model "$MODEL"
  --gpus "$GPUS"
  --host "$SERVER_HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
  --num-pages "$NUM_PAGES"
  --attention-backend "$ATTENTION_BACKEND"
  --launch-backend "$LAUNCH_BACKEND"
  --ray-address "$RAY_ADDRESS"
)
if [[ -n "$RAY_LOG_DIR" ]]; then
  start_cmd+=(--ray-log-dir "$RAY_LOG_DIR")
fi
if [[ "$RAY_NSYS" -eq 1 ]]; then
  start_cmd+=(--ray-nsys)
fi
if [[ -n "$RAY_NSYS_OUTPUT_PREFIX" ]]; then
  start_cmd+=(--ray-nsys-output-prefix "$RAY_NSYS_OUTPUT_PREFIX")
fi
if [[ -n "$MAX_SEQ_LEN" ]]; then
  start_cmd+=(--max-seq-len-override "$MAX_SEQ_LEN")
fi
if [[ -n "$MOE_BACKEND" ]]; then
  start_cmd+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "$MEMORY_RATIO" ]]; then
  start_cmd+=(--memory-ratio "$MEMORY_RATIO")
fi
if [[ "$DISABLE_PYNCCL" -eq 1 ]]; then
  start_cmd+=(--disable-pynccl)
fi
if [[ -n "$CONTROL_HOST" ]]; then
  start_cmd+=(--extra-args "--control-host $CONTROL_HOST${EXTRA_ARGS:+ $EXTRA_ARGS}")
elif [[ -n "$EXTRA_ARGS" ]]; then
  start_cmd+=(--extra-args "$EXTRA_ARGS")
fi

python - <<'PY' "$RUN_CONFIG_JSON" "$ENV_NAME" "$MODEL" "$PROMPT_FILE" "$PROMPT_REPEAT" "$GPUS" "$TP_SIZE" "$SERVER_HOST" "$PORT" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P" "$TOP_K" "$SAMPLE_CONCURRENCY" "$MAX_RUNNING_REQUESTS" "$CUDA_GRAPH_MAX_BS" "$NUM_PAGES" "$MAX_SEQ_LEN" "$ATTENTION_BACKEND" "$MOE_BACKEND" "$MEMORY_RATIO" "$LAUNCH_BACKEND" "$RAY_ADDRESS" "$RAY_LOG_DIR" "$CONTROL_HOST" "$DISABLE_PYNCCL" "$EXTRA_ARGS"
import json
import sys

path = sys.argv[1]
keys = [
    "env_name",
    "model",
    "prompt_file",
    "prompt_repeat",
    "gpus",
    "tp_size",
    "host",
    "port",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "sample_concurrency",
    "max_running_requests",
    "cuda_graph_max_bs",
    "num_pages",
    "max_seq_len",
    "attention_backend",
    "moe_backend",
    "memory_ratio",
    "launch_backend",
    "ray_address",
    "ray_log_dir",
    "control_host",
    "disable_pynccl",
    "extra_args",
]
values = sys.argv[2:]
with open(path, "w", encoding="utf-8") as f:
    json.dump(dict(zip(keys, values)), f, ensure_ascii=False, indent=2)
PY

echo "Starting mini-sgl"
echo "  active_env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "  server_url: http://$SERVER_HOST:$PORT/v1"
echo "  minisgl_log: $MINISGL_LOG"
echo "  num_pages: $NUM_PAGES"
if [[ -n "$RAY_LOG_DIR" ]]; then
  echo "  ray_log_dir: $RAY_LOG_DIR"
fi
if [[ "$RAY_NSYS" -eq 1 ]]; then
  echo "  ray_nsys: 1"
  if [[ -n "$RAY_NSYS_OUTPUT_PREFIX" ]]; then
    echo "  ray_nsys_output_prefix: $RAY_NSYS_OUTPUT_PREFIX"
  fi
fi
echo "  sample_json: $SAMPLE_JSON"
echo "  run_config: $RUN_CONFIG_JSON"
"${start_cmd[@]}" >"$MINISGL_LOG" 2>&1 &
MINISGL_PID=$!

set +e
wait_for_ready "http://$SERVER_HOST:$PORT/v1" "$MINISGL_LOG" "$MINISGL_PID"
ready_status=$?
set -e
if (( ready_status == 2 )); then
  minisgl_status="$(wait_child_status "$MINISGL_PID")"
  MINISGL_PID=""
  fail_with_log "mini-sgl exited before becoming ready (exit code $minisgl_status)." "$MINISGL_LOG"
elif (( ready_status != 0 )); then
  fail_with_log "mini-sgl failed to become ready within the startup timeout." "$MINISGL_LOG"
fi

echo "Sampling prompts with mini-sgl"
python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" sample \
  --model "$MODEL" \
  --minisgl-url "http://$SERVER_HOST:$PORT" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-repeat "$PROMPT_REPEAT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --sample-concurrency "$SAMPLE_CONCURRENCY" \
  --sample-json "$SAMPLE_JSON" &
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
    fail_with_log "mini-sgl exited during sampling (exit code $minisgl_status)." "$MINISGL_LOG"
  fi

  if ! process_is_running "$SAMPLE_PID"; then
    sample_status="$(wait_child_status "$SAMPLE_PID")"
    SAMPLE_PID=""
    break
  fi

  sleep 0.2
done

if (( sample_status != 0 )); then
  fail_with_log "Sampling command failed (exit code $sample_status)." "$MINISGL_LOG" "$sample_status"
fi

stop_process_tree "$MINISGL_PID"
wait "$MINISGL_PID" >/dev/null 2>&1 || true
MINISGL_PID=""

echo "Saved artifacts:"
echo "  sample_json: $SAMPLE_JSON"
echo "  minisgl_log: $MINISGL_LOG"
echo "  run_config: $RUN_CONFIG_JSON"

python - <<'PY' "$SAMPLE_JSON" "$PRINT_SAMPLES" "$PREVIEW_CHARS"
import json
import sys

sample_json = sys.argv[1]
print_samples = int(sys.argv[2])
preview_chars = int(sys.argv[3])

with open(sample_json, "r", encoding="utf-8") as f:
    payload = json.load(f)

summary = payload.get("summary", {})
print("Sample Summary:")
for key in ("num_prompts", "total_generated_tokens"):
    if key in summary:
        print(f"  {key}: {summary[key]}")

def shorten(text: str, limit: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

samples = payload.get("samples", [])
if print_samples > 0 and samples:
    print("Sample Preview:")
    for idx, sample in enumerate(samples[:print_samples], start=1):
        prompt = shorten(sample.get("prompt", ""), preview_chars)
        output = shorten(sample.get("generated_text", ""), preview_chars)
        print(f"  [{idx}] prompt: {prompt}")
        print(f"      output: {output}")
PY
