#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model_dir=${QWEN_MODEL_DIR:-}
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  if [[ ${arguments[index]} == --model-dir ]]; then
    if ((index + 1 >= ${#arguments[@]})); then
      echo "--model-dir requires a value" >&2
      exit 1
    fi
    model_dir=${arguments[index + 1]}
  fi
done

if [[ -z $model_dir ]]; then
  echo "QWEN_MODEL_DIR must point to a published Qwen3.8 checkpoint" >&2
  exit 1
fi
if [[ ! -f $model_dir/model.safetensors.index.json ]]; then
  echo "QWEN_MODEL_DIR is not a published Qwen3.8 checkpoint: $model_dir" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 ||
    ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Fxq '12.1'; then
  echo "Qwen3.8 train requires an sm_121 GPU" >&2
  exit 1
fi

worker=$($script_dir/build_worker.sh)
exec "$worker" --model-dir "$model_dir" "$@"
