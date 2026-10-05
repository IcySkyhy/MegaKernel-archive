#!/bin/bash
#SBATCH --job-name=mkgen-final
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
# Everything that produces a number, on one exclusive GPU, from one build:
# recalibrate, compile and gate every model, then the serving-engine baselines.
set -u
cd "${MKGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source env.sh
export CUDA_VISIBLE_DEVICES=0 MKGEN_GPU_AUTO=0 HF_HUB_OFFLINE=1
# Freeze the compiler for the life of the job: the shared target directory is
# rebuilt by whoever is working, and a benchmark whose rows came from two
# different binaries is not a benchmark.
MK=$MKGEN_WORK/build/mkc-$SLURM_JOB_ID
cp "${CARGO_TARGET_DIR}/release/mkc" "$MK"
export MKC_BIN="$MK"
V=$MKGEN_PY
OUT=$MKGEN_WORK/out
LOG=$MKGEN_WORK/logs
M=$MKGEN_MODELS

echo "== node $(hostname)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

echo "== [1] calibrate"
# /home is read-only on the compute nodes, so the machine model is written to
# scratch and every compile in this job is pointed at it with MKC_HW.  Copy it
# into hw/ from a login node afterwards to make it the shipped one.
export MKC_HW=$MKGEN_WORK/hw.json
nvcc -O3 -arch=sm_90a -std=c++17 -Iruntime/include -o $MKGEN_WORK/build/calib calib/calib.cu
$MKGEN_WORK/build/calib "$MKC_HW" 40 > "$LOG/calib_final.log" 2>&1
grep -A5 "cooperative grid barrier" "$LOG/calib_final.log"; grep "^\[4\]" "$LOG/calib_final.log"
python3 - "$MKC_HW" <<'PY'
import json, sys, collections
h = json.load(open(sys.argv[1]))
print("   machine model schema", h.get("schema"), h["device"],
      "gemv", dict(collections.Counter(g["quant"] for g in h["gemv"])),
      "stage", dict(collections.Counter(g["quant"] for g in h["stage"])))
PY

echo "== [2] primitives"
nvcc -O2 -arch=sm_90a -std=c++17 -Iruntime/include -o $MKGEN_WORK/build/test_prims tests/test_prims.cu
$MKGEN_WORK/build/test_prims

echo "== [3] the zoo, one shot each"
rm -f "$MKGEN_RESULTS"
CTX=1024 DECODE=128 GATE=8 VERIFY=1 REF=cuda:0 scripts/zoo.sh \
  $M/Qwen3-0.6B-FP8 $M/Qwen3-0.6B $M/TinyLlama-1.1B-Chat-v1.0 \
  $M/Qwen2.5-1.5B-Instruct-AWQ $M/Qwen2.5-1.5B-Instruct $M/Qwen3-1.7B \
  $M/Qwen3-1.7B-FP8-dynamic $M/SmolLM2-1.7B-Instruct $M/gemma-2-2b-it \
  $M/gpt-oss-20b $M/Phi-3-mini-4k-instruct $M/Qwen1.5-MoE-A2.7B $M/Qwen3-8B
column -t -s$'\t' "$MKGEN_RESULTS"

echo "== [4] the big two"
for BIG in "$M/Qwen3-30B-A3B" "$MKGEN_MODEL_120B"; do
  NAME=$(basename "$BIG"); [ "$NAME" = "gpt-oss-120b-hf" ] && NAME=gpt-oss-120b
  $MK build "$BIG" -o "$OUT/$NAME" --ctx 1024 --name "$NAME" --verify \
     > "$LOG/$NAME.compile.log" 2>&1
  grep -E "verify |^ *predicted" "$LOG/$NAME.compile.log" | tail -8
  # Qwen3-30B-A3B is bf16: 60 GB for the reference and 60 GB for the engine do
  # not fit together, but --ref-first computes the reference and frees it before
  # the engine loads, so it CAN be gated.  gpt-oss-120b cannot: dequantised to
  # bf16 it is 240 GB, more than the card and more than the node's RAM.  Its
  # architecture is gated on gpt-oss-20b, which shares every code path; here it
  # gets the guard only, and the results table says so.
  G=0; RF=""
  [ "$NAME" = "Qwen3-30B-A3B" ] && { G=4; RF="--ref-first"; }
  $V harness/run.py "$OUT/$NAME" --model "$BIG" --gate $G $RF --ref-device cuda:0 \
     --prefill 1024 --decode 128 \
     --device 0 --guard --weight-sample 24 --json "$OUT/$NAME/result.json" 2>&1 \
     | grep -vE "it/s|loading" | tail -20
  # the same table as the zoo's: these two are run here, not by zoo.sh, and a
  # results file missing its two largest models is not the whole story
  $V scripts/row.py "$OUT/$NAME" "$NAME" "0" "$MKGEN_RESULTS"
done
column -t -s$'\t' "$MKGEN_RESULTS"

echo "== [4b] where the time actually goes (measured stage schedule)"
# The compiler's own feedback: the %globaltimer marks the kernel writes after
# every barrier, against what the planner predicted.  One dense model and one
# MoE model is enough to see which side of the cost model is drifting.
for PM in Qwen3-8B Qwen3-30B-A3B gpt-oss-120b; do
  [ -f "$OUT/$PM/build.json" ] || continue
  SRC="$M/$PM"; [ -d "$SRC" ] || SRC="$MKGEN_MODEL_120B"
  echo "-- $PM"
  $V harness/mkprofile.py "$OUT/$PM" --model "$SRC" --ctx 1024 --device 0 \
     --json "$OUT/$PM/profile.json" 2>&1 | grep -vE "it/s|loading"
done

# The serving-engine baselines now live in scripts/slurm_base.sh: they depend on
# two other projects' virtualenvs, they fail in ways that have nothing to do with
# this compiler, and a failure there should not cost a whole zoo run.
echo "== done"
