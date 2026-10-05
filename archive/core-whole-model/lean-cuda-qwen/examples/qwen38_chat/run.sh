#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model_dir=${QWEN_MODEL_DIR:-}
host=${QWEN_CHAT_HOST:-0.0.0.0}
port=${QWEN_CHAT_PORT:-8080}

if [[ -z $model_dir ]]; then
  echo "QWEN_MODEL_DIR must point to a published Qwen3.8 checkpoint" >&2
  exit 1
fi
if [[ ! -f $model_dir/tokenizer.json || ! -f $model_dir/model.safetensors.index.json ]]; then
  echo "QWEN_MODEL_DIR is not a published Qwen3.8 checkpoint: $model_dir" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Qwen3.8 chat requires uv for the tokenizer environment" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 ||
    ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq '12.1'; then
  echo "Qwen3.8 chat requires an sm_121 GPU" >&2
  exit 1
fi

worker=$($script_dir/build_worker.sh)
exec uv run --no-project --with 'tokenizers==0.23.1' python "$script_dir/server.py" \
  --worker "$worker" --model-dir "$model_dir" --host "$host" --port "$port"
