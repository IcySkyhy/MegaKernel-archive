#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle

# Pure-elaboration gate for the Qwen3.8 config provider; no GPU or nvcc needed. Uses the
# standalone Qwen library built by the suite wrapper, elaborates its provider assertions and runtime
# checks, and verifies that the
# malformed fixtures fail elaboration.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
lean_test_root=${LEAN_TEST_ROOT:-"$repo_root/.lake/packages/LeanTest"}
cd "$script_dir"

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/lean-cuda-qwen36-config.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

lean_bin=${LEAN:-"$(command -v lean)"}
lean_libdir=$($lean_bin --print-libdir)
cp -as "$lean_libdir/Lean" "$build_dir/"
cp -as "$lean_libdir"/Lean.olean* "$build_dir/"
mkdir -p "$build_dir/LeanTest"
export LEAN_PATH="$build_dir:${LEAN_PATH:-$lean_libdir}"

if [[ ! -f "$lean_test_root/LeanTest.lean" ]]; then
  echo "Qwen3.8 config suite requires LeanTest; set LEAN_TEST_ROOT" >&2
  exit 1
fi
for module in Basic Assert Attr Runner; do
  "$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest/$module.olean" \
    "$lean_test_root/LeanTest/$module.lean"
done
"$lean_bin" --root="$lean_test_root" -o "$build_dir/LeanTest.olean" \
  "$lean_test_root/LeanTest.lean"


"$lean_bin" --root="$script_dir" -o "$build_dir/Test.olean" Test.lean
"$lean_bin" --root="$script_dir" --run Driver.lean

if "$lean_bin" --root="$script_dir" Malformed.lean >"$build_dir/Malformed.out" 2>&1; then
  echo "Expected the divisibility-violating Qwen3.8 config to be rejected" >&2
  exit 1
fi
grep -Fq "hfconfig_type_provider" "$build_dir/Malformed.out"
grep -Fq "must be a multiple of num_key_value_heads" "$build_dir/Malformed.out"

if "$lean_bin" --root="$script_dir" MalformedMissing.lean >"$build_dir/MalformedMissing.out" 2>&1; then
  echo "Expected the Qwen3.8 config with a missing hidden_size to be rejected" >&2
  exit 1
fi
grep -Fq "hfconfig_type_provider" "$build_dir/MalformedMissing.out"
grep -Fq "missing field 'hidden_size'" "$build_dir/MalformedMissing.out"
