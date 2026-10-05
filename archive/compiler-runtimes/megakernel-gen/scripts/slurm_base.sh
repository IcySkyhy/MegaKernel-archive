#!/bin/bash
#SBATCH --job-name=mkgen-base
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:30:00
# The serving-engine baselines on their own, so a failure here does not cost a
# whole zoo run.  Same workload as harness/run.py's decode measurement: batch 1,
# a 1024-token prompt, 128 decode steps, prefill removed by N-vs-1 differencing.
set -u
cd "${MKGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source env.sh
export CUDA_VISIBLE_DEVICES=0 MKGEN_GPU_AUTO=0 HF_HUB_OFFLINE=1
# per-job log names: two baseline jobs can be in flight at once (they fail for
# different reasons and get resubmitted), and a shared path means the loser
# overwrites the winner
LOG=$MKGEN_WORK/logs/base-$SLURM_JOB_ID
mkdir -p "$LOG"
M=$MKGEN_MODELS
R=$MKGEN_ROOT

# Both engines JIT-compile kernels into `~/.cache/<engine>` and ignore
# XDG_CACHE_HOME; /home is read-only here, so the compile dies as a bare
# PermissionError long before a model loads.  Redirecting HOME for the engine
# processes catches every library that hardcodes a path under it, which is the
# only fix that does not have to be repeated per library.  MKGEN_{VLLM,SGLANG}_PY
# are absolute, so redirecting HOME does not lose the interpreters.
export HOME=$MKGEN_WORK/fakehome
mkdir -p "$HOME/.cache" "$HOME/.triton"

echo "== node $(hostname)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
# Both engines size a static pool and carve the KV cache out of what is left, so
# a 120 B model on an 80 GB card needs the fraction HIGH, not low: SGLang states
# its own minimum (0.880) and vLLM just says "no available memory for the cache
# blocks".  An earlier OOM during SGLang's MXFP4 swizzle looked like the opposite
# problem and sent this the wrong way twice.  They still want different values,
# so each gets its own.
# model : vllm-fraction : sglang-fraction
for SPEC in "$M/Qwen3-8B:0.85:0.85" "$M/gpt-oss-20b:0.85:0.85" \
            "$MKGEN_MODEL_120B:0.90:0.92"; do
  BM=${SPEC%%:*}; REST=${SPEC#*:}; VF=${REST%%:*}; SF=${REST##*:}; BN=$(basename "$BM")
  echo "-- vLLM $BN"
  VLLM_LOGGING_LEVEL=WARNING "$MKGEN_VLLM_PY" \
     "$R/bench/bench_vllm.py" --model "$BM" --mem "$VF" > "$LOG/vllm_$BN.log" 2>&1
  grep -E "RESULT" "$LOG/vllm_$BN.log" | tail -2 || tail -3 "$LOG/vllm_$BN.log"
  echo "-- SGLang $BN"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$MKGEN_SGLANG_PY" \
     "$R/bench/bench_sglang.py" --model "$BM" --mem "$SF" > "$LOG/sgl_$BN.log" 2>&1
  grep -E "RESULT" "$LOG/sgl_$BN.log" | tail -2 || tail -3 "$LOG/sgl_$BN.log"
done
echo "== done"
