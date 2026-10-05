#!/bin/bash
# The strongest form of the weight-sensitivity check: EVERY weight tensor in the
# model, not a sample.  Each one is zeroed, the output must move, and the bytes
# are restored from the checkpoint and asserted bit-identical.
#
# Usage: scripts/full_guard.sh <out_dir> <model_dir> [device]
set -u
cd "$(dirname "$0")/.."
source env.sh
export MKGEN_GPU_AUTO=0
V=$MKGEN_PY
exec $V harness/run.py "$1" --model "$2" --gate 0 --prefill 0 --decode 0 \
     --device "${3:-0}" --guard --weight-sample 0
