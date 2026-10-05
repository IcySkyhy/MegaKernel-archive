#!/usr/bin/env bash
# Regenerate the whole M39 reference set + its evidence.
#
#   bash tools/make_reference.sh            # full set (~10 min single-core-ish)
#   bash tools/make_reference.sh quick      # m=1 only
#
# Everything is deterministic: same checkpoint, same seeds, same code -> same
# sha256. `evidence/` gets the command log, the per-file sha256 and the
# selfcheck transcript.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/workspace/venvs/baseline/bin/python3
MODE="${1:-full}"

# Pin the torch intra-op thread count: `moe.routed_out` is the ONE segment whose
# bits depend on the CPU GEMM's blocking (measured: 8 threads vs default differ
# on that segment only). Everything else is thread-independent. The archived
# sha256 are produced with this setting; override if you know what you are doing.
export M39_THREADS="${M39_THREADS:-8}"

mkdir -p evidence reference
LOG=evidence/reference_build.log
: > "$LOG"

run() {
  echo "### $*" | tee -a "$LOG"
  "$PY" run_reference.py "$@" 2>&1 | grep -v 'UserWarning\|torch.from_numpy\|^  t = torch' | tee -a "$LOG"
}

# --- layer 0 (GDN, linear_attention) ---------------------------------------
# decode m=1 at position 8, with the GDN conv/ssm state primed by 8 tokens so the
# dump is not the degenerate all-zero-state first step.
run --layer 0 --m 1 --warmup 8 --input embed --pending none  --tag layer0_decode_m1
# same, plus a synthetic pending block/injection -> exercises the fused
# attn_hc.combine_and_mix branch of the FIRST mixer as well
run --layer 0 --m 1 --warmup 8 --input embed --pending synth --tag layer0_decode_m1_pending

# --- layer 3 (QSA, full_attention) -----------------------------------------
# warmup 2100 -> dump at position 2100, so visible_blocks = 525 > block_topk = 512
# and the archive covers the top-k TRUNCATION path as well as the multi-block path.
run --layer 3 --m 1 --warmup 2100 --input synthetic --pending synth --tag layer3_decode_m1

if [ "$MODE" != "quick" ]; then
  # chunk 64 = one FLA/GDN chunk and a full QSA prefill micro-chunk
  run --layer 0 --m 64 --input embed     --pending synth --tag layer0_chunk_m64
  run --layer 3 --m 64 --input synthetic --pending none  --tag layer3_chunk_m64
  # identical settings, different seed -> the non-hollowness twin pair
  run --layer 3 --m 64 --input synthetic --seed 1 --pending none --tag layer3_chunk_m64_seed1
fi

# --- evidence ---------------------------------------------------------------
# `tools/make_evidence.sh` regenerates everything under evidence/ from the dumps
# above; keep a copy of the first build's hashes so the determinism check has a
# baseline to compare against.
if [ ! -f evidence/reference_sha256.first_run.txt ]; then
  "$PY" tools/hash_dumps.py reference > evidence/reference_sha256.first_run.txt
fi
bash tools/make_evidence.sh

echo "done; reference/ + evidence/ written"
