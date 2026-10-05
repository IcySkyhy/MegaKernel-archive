#!/bin/bash
# M30: regenerate the raw evidence quoted in baseline_env/README.md.
# Read-only w.r.t. the system: it only prints facts (no installs, no device work).
#
#   bash baseline_env/scripts/collect_env_evidence.sh > baseline_env/evidence/01-... etc.
set -u
OUT=${1:-.}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# A failed network call must never clobber a previously good archive: every file is
# built in a temp file and only moved into place when the run completed (sentinel line
# present, no NETWORK_UNAVAILABLE marker).
net() {
  for _ in 1 2 3; do
    out=$(timeout 40 git "$@" 2>/dev/null) && [ -n "$out" ] && { printf '%s\n' "$out"; return 0; }
    sleep 5
  done
  echo "NETWORK_UNAVAILABLE: git $*"
  return 1
}
emit() { # emit <target> <tmpfile>
  local target=$1 tmp=$2
  if grep -q '^### EOF' "$tmp" && ! grep -q 'NETWORK_UNAVAILABLE' "$tmp"; then
    mv "$tmp" "$target"
  else
    echo "REFUSING to overwrite $target (incomplete run; kept at $tmp)"
    return 1
  fi
}

{
  echo "### collected: $(date -Is)"
  echo
  echo "## host"
  uname -a
  lscpu | grep -E "^Model name|^CPU\(s\)|^Socket|^NUMA node\(s\)"
  echo
  echo "## npu-smi"
  npu-smi info
  echo
  echo "## driver"
  cat /usr/local/Ascend/driver/version.info
  echo
  echo "## CANN / ATB versions"
  for f in /usr/local/Ascend/cann-9.1.0/version.cfg \
           /usr/local/Ascend/cann-9.1.0/ascend_toolkit_install.info \
           /usr/local/Ascend/nnal/atb/latest/version.info \
           /usr/local/Ascend/ascend-toolkit/latest/version.cfg; do
    echo "--- $f"; cat "$f" 2>&1 | head -20
  done
  echo
  echo "## ATB cxx ABI deployment"
  ls /usr/local/Ascend/nnal/atb/latest/atb/
  env | grep -E "ATB_CXX_ABI|ATB_HOME_PATH|ASCEND_HOME_PATH|ASCEND_OPP_PATH|ASCEND_TOOLKIT_HOME"
  echo
  echo "## python"
  /usr/local/python3.12.13/bin/python3 -V
  which -a python3 pip3
  echo
  echo "## memory cgroup limit / disk"
  cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
  echo "## host memory (frame view; NOT the container limit -- see cgroup line above)"
  free -g
  grep -E "^MemTotal|^MemAvailable" /proc/meminfo
  df -h /workspace
  echo
  echo "## model checkpoint"
  du -sh /workspace/Qwen3.8-Flash-Next-MXFP4
  ls /workspace/Qwen3.8-Flash-Next-MXFP4/*.safetensors | wc -l
  grep -m1 '"model_type"' /workspace/Qwen3.8-Flash-Next-MXFP4/config.json
  grep -m1 -A2 '"architectures"' /workspace/Qwen3.8-Flash-Next-MXFP4/config.json
  echo "### EOF"
} > "$OUT/.01-host-and-npu.tmp" 2>&1
emit "$OUT/01-host-and-npu.txt" "$OUT/.01-host-and-npu.tmp"

{
  echo "### collected: $(date -Is)"
  echo
  echo "## /workspace/vllm (upstream checkout, read-only)"
  git -C /workspace/vllm log --oneline -1
  git -C /workspace/vllm describe --tags
  echo
  echo "## /workspace/vllm-ascend (fork checkout, read-only)"
  git -C /workspace/vllm-ascend log --oneline -1
  git -C /workspace/vllm-ascend branch --show-current
  echo "--- newest local tags of the fork"
  git -C /workspace/vllm-ascend tag | sort -V | tail -6
  echo "--- branches on the fork remote (gitcode)"
  net ls-remote --heads https://gitcode.com/liruixin_dvc/vllm-ascend.git
  echo "--- ALL branches on the upstream project (github) — UNFILTERED, do not add a"
  echo "    regex filter here: a previous 'releases/v[0-9.]+$' pattern silently dropped"
  echo "    releases/v0.28.0rc and releases/v0.29.0rc and produced a wrong conclusion (see README §10)"
  net ls-remote --heads https://github.com/vllm-project/vllm-ascend.git | sed 's#refs/heads/##' | sort -k2
  echo "--- newest tags on the upstream project (github)"
  net ls-remote --tags https://github.com/vllm-project/vllm-ascend.git \
    | grep -oE 'refs/tags/v0\.(2[6-9]|3[0-9])[^^{}]*$' | sort -V | tail -6
  echo "--- newest tags in the local upstream vLLM checkout"
  git -C /workspace/vllm tag | sort -V | tail -6
  echo
  echo "## does a given upstream vLLM tag know the Qwen3.8-Flash-Next architecture (qwen4_exp)?"
  echo "## grep hit counts per tag, over the vllm/ python tree:"
  for t in v0.23.0 v0.25.1 v0.27.0 v0.28.0 v0.29.0 v0.30.0; do
    echo "--- $t"
    for pat in qwen4_exp Qwen4ExpForConditionalGeneration hc_count ple_embed_dim \
               ngram_vocab_size_base indexer_compress_ratio; do
      c=$(git -C /workspace/vllm grep -c "$pat" "$t" -- vllm 2>/dev/null | wc -l)
      echo "    $pat: $c"
    done
  done
  echo
  echo "## registry entries for qwen4_exp in vLLM main (8a2364605c)"
  git -C /workspace/vllm grep -n "qwen4_exp" 8a2364605c -- vllm/model_executor/models/registry.py
  echo
  echo "## platform dispatch inside vllm/models/qwen4_exp/__init__.py (main)"
  git -C /workspace/vllm show 8a2364605c:vllm/models/qwen4_exp/__init__.py | sed -n '20,48p'
  echo
  echo "## vLLM main does not ship an Ascend/NPU platform (plugin is mandatory)"
  git -C /workspace/vllm ls-tree --name-only 8a2364605c -- vllm/platforms/
  echo
  echo "## no NPU branch in vllm/models/qwen4_exp (main)"
  echo "--- file tree of vllm/models/qwen4_exp at main (dirs: see 'nvidia/' + 'amd/')"
  git -C /workspace/vllm ls-tree -r --name-only 8a2364605c -- vllm/models/qwen4_exp | sed 's#^vllm/models/qwen4_exp/##' | cut -d/ -f1 | sort -u
  echo "--- any 'npu' / 'ascend' token in that tree?"
  git -C /workspace/vllm grep -ilw "npu\|ascend" 8a2364605c -- vllm/models/qwen4_exp/ || echo "(none)"
  echo
  echo "## vllm-ascend v0.23.0 release compatibility matrix row"
  git -C /workspace/vllm-ascend show HEAD:docs/source/community/versioning_policy.md | sed -n '/^| vLLM Ascend/,/^| v0.20/p'
  echo
  echo "## vllm-ascend v0.23.0 pinned python deps"
  git -C /workspace/vllm-ascend show HEAD:requirements.txt | grep -E "^(torch|torch-npu|torchvision|torchaudio|triton-ascend)"
  echo
  echo "## vLLM main pinned torch"
  grep -n "torch" /workspace/vllm/pyproject.toml | head -5
  echo
  echo "## wheels published on the reachable indexes (x86_64, cp312)"
  echo "--- torch_npu (repo.huaweicloud.com pypi mirror)"
  curl -sS https://repo.huaweicloud.com/repository/pypi/simple/torch-npu/ \
    | grep -oE 'torch_npu-2\.1[0-9][^"<]*cp312[^"<]*_x86_64\.whl' | sort -u
  echo "--- torch (same mirror)"
  curl -sS https://repo.huaweicloud.com/repository/pypi/simple/torch/ \
    | grep -oE 'torch-2\.1[0-9]\.[0-9]+-cp312[^"<]*\.whl' | sort -u
  echo "--- triton-ascend (mirrors.huaweicloud.com/ascend/repos/pypi)"
  curl -sS https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/ \
    | grep -oE "triton_ascend-[0-9][^\"<']*cp312[^\"<']*_x86_64[^\"<']*\.whl" | sort -u
  echo "### EOF"
} > "$OUT/.02-version-matrix.tmp" 2>&1
emit "$OUT/02-version-matrix.txt" "$OUT/.02-version-matrix.tmp"

echo "wrote/kept: $OUT/01-host-and-npu.txt  $OUT/02-version-matrix.txt"
