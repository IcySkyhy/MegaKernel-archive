#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh" "$@"
