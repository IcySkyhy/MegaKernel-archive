#!/bin/bash
# Build + correctness-gate + interleaved benchmark for the sm120 sparse
# cooperative-WG phase probe. Run ON box (prefer through the persistent job
# runner); clock and power state are reverted even on error/interrupt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=${S:-8192}
ITERS=${ITERS:-120}
CHECK_S=${CHECK_S:-1024}
RUNS=${RUNS:-3}
BASE=${BASE:-/tmp/sparse_phase_base}
PHASE=${PHASE:-/tmp/sparse_phase_test}

cd "$HERE"
./build_sparse.sh sparse_probe.cu "$BASE" -DHASH -DGEMM_ONLY
./build_sparse.sh sparse_probe.cu "$PHASE" -DHASH -DGEMM_ONLY -DPHASE

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU busy; refusing to benchmark" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  exit 2
fi

restore_gpu() {
  sudo nvidia-smi -rgc >/dev/null 2>&1 || true
  sudo nvidia-smi -rmc >/dev/null 2>&1 || true
  sudo nvidia-smi -pl 300 >/dev/null 2>&1 || true
}
trap restore_gpu EXIT INT TERM
sudo nvidia-smi -pm 1 >/dev/null
sudo nvidia-smi -lgc 3090,3090 >/dev/null
sudo nvidia-smi -lmc 14001 >/dev/null
sudo nvidia-smi -pl 330 >/dev/null 2>&1 || true

# Correctness first: same deterministic CUTLASS inputs and unchanged MMA order
# must produce bit-identical bf16 output.
bcheck=$("$BASE" "$CHECK_S" 8)
pcheck=$("$PHASE" "$CHECK_S" 8)
bout=$(grep 'output hash:' <<<"$bcheck")
pout=$(grep 'output hash:' <<<"$pcheck")
echo "baseline $bout"
echo "phase    $pout"
[[ "${bout#output hash: }" == "${pout#output hash: }" ]] || {
  echo "HASH MISMATCH: phase schedule is not correct" >&2
  exit 3
}

# Warm each kernel, then alternate to cancel temperature/clock drift.
"$BASE" "$S" 30 >/dev/null
"$PHASE" "$S" 30 >/dev/null
for ((r=1; r<=RUNS; r++)); do
  echo "run $r baseline"
  stdbuf -oL "$BASE" "$S" "$ITERS" | grep -m1 '^sparse tile'
  echo "run $r phase"
  stdbuf -oL "$PHASE" "$S" "$ITERS" | grep -m1 '^sparse tile'
done

nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,utilization.gpu --format=csv,noheader
