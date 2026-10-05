#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

suite_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$suite_dir/../../scripts/backend-env.sh"
lean_cuda_build_qwen_library
export LEAN="$(command -v lean)"
export LEANC="$(command -v leanc)"

if (( $# == 0 )); then
  mapfile -t tests < <(find "$suite_dir" -mindepth 1 -maxdepth 1 -type d -name 'cuda_*' -printf '%f\n' | sort)
else
  tests=("$@")
fi

for test_name in "${tests[@]}"; do
  if [[ $test_name == */* || ! -x "$suite_dir/$test_name/run_test.sh" ]]; then
    echo "Unknown workload regression: $test_name" >&2
    exit 1
  fi
  echo "==> $test_name"
  (cd "$suite_dir/$test_name" && ./run_test.sh)
done
