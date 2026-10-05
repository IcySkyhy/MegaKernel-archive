#!/bin/bash
# source env.sh   -- toolchain for the megakernel compiler (mkc)
source /etc/profile.d/z-01-site.sh 2>/dev/null || true
for _mf in /etc/profile.d/*modules*.sh; do [ -e "$_mf" ] && source "$_mf" 2>/dev/null; done; unset _mf
module load StdEnv/2023 gcc/12.3 cuda/12.9 >/dev/null 2>&1

# The repository is wherever this script lives; nothing below is site-specific
# except through env.local.sh, which is untracked.  Copy env.local.sh.example to
# env.local.sh and set MKGEN_WORK (and, on a slurm cluster, MKGEN_ACCOUNT).
_mkgen_src="${BASH_SOURCE[0]:-$0}"
export MKGEN_ROOT="$(cd "$(dirname "$_mkgen_src")" && pwd)"; unset _mkgen_src
[ -f "$MKGEN_ROOT/env.local.sh" ] && source "$MKGEN_ROOT/env.local.sh"

# Everything derived -- caches, candidate build trees, checkpoints, results --
# lives under MKGEN_WORK, never in the repository and never under $HOME.
export MKGEN_WORK="${MKGEN_WORK:-${SCRATCH:-$HOME/.cache}/mkgen}"
export MKGEN_MODELS="${MKGEN_MODELS:-$MKGEN_WORK/models}"
export MKGEN_MODEL_120B="${MKGEN_MODEL_120B:-$MKGEN_MODELS/gpt-oss-120b}"
export MKGEN_PY="${MKGEN_PY:-$MKGEN_WORK/venv/bin/python}"
# The baselines need their own environments: vLLM and SGLang pin conflicting
# versions of torch and of each other's dependencies, so they cannot share one.
export MKGEN_VLLM_PY="${MKGEN_VLLM_PY:-$MKGEN_WORK/venv-vllm/bin/python}"
export MKGEN_SGLANG_PY="${MKGEN_SGLANG_PY:-$MKGEN_WORK/venv-sgl/bin/python}"
export MKGEN_ACCOUNT="${MKGEN_ACCOUNT:-}"

export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$CUDA_HOME/bin:$CARGO_HOME/bin:$MKGEN_ROOT/.venv/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-$MKGEN_WORK/hf}"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
# Every cache goes to scratch.  Two reasons, both learned the hard way:
#   * /home is read-only on the compute nodes, so a cache under it kills a
#     benchmark halfway through with a permission error
#   * /home has a 100 GiB quota that build and download caches will eat
export TRITON_CACHE_DIR="$MKGEN_WORK/triton_cache"
export XDG_CACHE_HOME="$MKGEN_WORK/cache"
export TORCHINDUCTOR_CACHE_DIR="$MKGEN_WORK/inductor_cache"
export VLLM_CACHE_ROOT="$MKGEN_WORK/vllm_cache"
export HF_HUB_OFFLINE=0
export OUTLINES_CACHE_DIR="$MKGEN_WORK/outlines_cache"
export PIP_CACHE_DIR="$MKGEN_WORK/pip_cache"
export UV_CACHE_DIR="$MKGEN_WORK/uv_cache"
export CCACHE_DIR="$MKGEN_WORK/ccache"
# /home is READ-ONLY on the compute nodes, so anything that writes under $HOME
# has to be redirected too -- vLLM's usage-stats file and the filelocks that
# guard it are the ones that actually bite.
export XDG_CONFIG_HOME="$MKGEN_WORK/config"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# flashinfer JIT-compiles kernels and takes a lock in its OWN cache directory,
# which ignores XDG_CACHE_HOME and defaults under $HOME -- read-only here, so the
# lock's ftruncate fails and vLLM's engine core dies before it loads a model.
export FLASHINFER_WORKSPACE_BASE="$MKGEN_WORK/flashinfer"
export FLASHINFER_CACHE_DIR="$MKGEN_WORK/flashinfer"
export FLASHINFER_JIT_DIR="$MKGEN_WORK/flashinfer/jit"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -n "$MKGEN_ACCOUNT" ]; then export SBATCH_ACCOUNT="$MKGEN_ACCOUNT"; fi
export SBATCH_OUTPUT="${SBATCH_OUTPUT:-$MKGEN_WORK/logs/%x-%j.out}"
mkdir -p "$MKGEN_WORK/logs" 2>/dev/null
export MKGEN_RESULTS="${MKGEN_RESULTS:-$MKGEN_WORK/results.tsv}"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$OUTLINES_CACHE_DIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
         "$CCACHE_DIR" "$XDG_CONFIG_HOME" "$FLASHINFER_JIT_DIR" 2>/dev/null
export TORCH_CUDA_ARCH_LIST="9.0a"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$MKGEN_WORK/cargo-target}"

# GPU selection.  "A benchmark on a shared GPU is not a measurement": pick the
# GPU with the most free memory AND the lowest utilisation, and export
# MKGEN_GPU_BUSY=1 if even the best one is not clean, so measurement scripts can
# refuse rather than quietly report a number that is 50% off.
if [ "${MKGEN_GPU_AUTO:-1}" = "1" ]; then
  _q=$(CUDA_VISIBLE_DEVICES= nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
         --format=csv,noheader,nounits)
  _sel=$(echo "$_q" | awk -F', ' -v want="${MKGEN_GPU:--1}" '
    { free=$3-$2; idx=$1; u=$4;
      if (idx==want) { print idx; found=1; exit }
      score = free - u*2000;
      if (score>best || best=="") { best=score; bidx=idx } }
    END { if (!found) print bidx }')
  # order every GPU best-first, so the harness can put the engine on device 0
  # and its reference on device 1 without them fighting for memory
  _all=$(echo "$_q" | awk -F', ' '{ printf "%d %d\n", ($3-$2) - $4*2000, $1 }' | sort -rn | awk '{printf "%s%s", (NR>1?",":""), $2}')
  export CUDA_VISIBLE_DEVICES="$_all"
  export MKGEN_GPU_ORDER="$_all"
  _busy=$(echo "$_q" | awk -F', ' -v s="$_sel" '$1==s { if ($2>6000 || $4>5) print 1; else print 0 }')
  export MKGEN_GPU_BUSY="$_busy"
  [ "$_busy" = "1" ] && echo "[env.sh] WARNING: GPU $_sel is not idle -- timings will be polluted" >&2
  unset _sel _q _busy _all
fi
export MKGEN_DEVICE_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
