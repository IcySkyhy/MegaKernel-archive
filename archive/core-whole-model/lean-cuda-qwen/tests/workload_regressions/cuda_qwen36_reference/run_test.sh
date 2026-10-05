#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

# NumPy oracle gates for the Qwen3.6 tiny configuration (no GPU, no Lean build):
#   1. oracle.py selftest  (delta-rule parity + finite-difference backward check)
#   2. fixture regeneration determinism (byte-compare against checked-in fixtures)
#   3. mixed-adapter LoRA selftest and byte-exact fixture regeneration
#   4. optional torch cross-check against the vendored HF reference (explicit
#      skip when no python with torch is available)

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/qwen36-oracle.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

py=${ORACLE_PYTHON:-}
if [[ -z $py ]]; then
  if python3 -c 'import numpy' >/dev/null 2>&1; then
    py=python3
  else
    echo "Skipping Qwen3.6 oracle gates: no python with numpy available"
    exit 0
  fi
fi
echo "Using python: $py ($("$py" -c 'import numpy; print("numpy", numpy.__version__)'))"

"$py" "$script_dir/oracle.py" selftest

"$py" "$script_dir/oracle.py" dump "$build_dir/fixtures"
"$py" "$script_dir/oracle.py" compare "$script_dir/fixtures" "$build_dir/fixtures"

"$py" "$script_dir/lora_oracle.py" selftest
"$py" "$script_dir/lora_oracle.py" dump "$build_dir/lora"
"$py" "$script_dir/lora_oracle.py" compare "$script_dir/fixtures/lora" "$build_dir/lora"

torch_py=""
if "$py" -c 'import torch' >/dev/null 2>&1; then
  torch_py=$py
fi
if [[ -n $torch_py ]]; then
  "$torch_py" "$script_dir/torch_reference_check.py" "$script_dir/fixtures"
else
  echo "Skipping torch reference cross-check: no python with torch available"
fi

echo "cuda_qwen36_reference: all mandatory gates passed"
