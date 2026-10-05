#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

suite_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$suite_dir/../.." && pwd)
# shellcheck source=../../scripts/backend-env.sh
source "$repo_root/scripts/backend-env.sh"
lean_cuda_build_qwen_library

if [[ ! -d "$suite_dir/.lake/packages/LeanTest" ]]; then
  (cd "$suite_dir" && lake update)
fi

export LEAN_PATH="$repo_root/.lake/build/lib/lean${LEAN_PATH:+:$LEAN_PATH}"
(cd "$suite_dir" && lake test -- "$@")
