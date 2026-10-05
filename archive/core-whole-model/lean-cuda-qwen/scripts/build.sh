#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=backend-env.sh
source "$repo_root/scripts/backend-env.sh"

if [[ ! -d "$repo_root/.lake/packages/LeanTest" ]]; then
  "$repo_root/scripts/setup-dependencies.sh"
fi
lean_cuda_build_qwen_library
