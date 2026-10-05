#!/bin/bash
# M30: install the torch / torch_npu / triton-ascend stack for CANN 9.1.0 + Python 3.12 (x86_64)
# into an isolated venv. Never touches the system interpreter or /root/.bashrc.
#
# Version choice: vllm-ascend v0.23.0 (fork tag v0.23.0+ascend950) release compatibility
# matrix row -> vLLM v0.23.0 / Python >=3.10,<3.13 / CANN 9.1.0 / torch 2.10.0 /
# torch_npu 2.10.0.post4 / triton-ascend 3.2.2. See README.md section 2.
set -x

VENV=${VENV:-/workspace/venvs/baseline}
PY=$VENV/bin/python
PIP=$VENV/bin/pip

$PY -V || exit 1

export PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. build-time deps (both source trees below build with --no-build-isolation)
$PIP install -U "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" setuptools-scm wheel || exit 10

# 2. triton-ascend first: it is only published on the Ascend PyPI repo, and it must be
#    installed BEFORE the pinned CPU torch pair so the pinned versions win.
$PIP install triton-ascend==3.2.2 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi || exit 11

# 3. pinned CPU torch stack (the NPU side is provided by torch_npu). The small runtime
#    deps come from the huaweicloud mirror first: download.pytorch.org serves the big
#    wheels fine but stalled twice on small ones (0-byte unpack file, dead socket), so
#    torch itself is then installed with --no-deps against the wheel cache.
$PIP install --timeout 60 --retries 5 \
  filelock typing-extensions sympy networkx jinja2 fsspec numpy pillow || exit 12
$PIP install --no-deps --timeout 60 --retries 5 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu || exit 13

# 4. torch_npu matching CANN 9.1.0
$PIP install torch_npu==2.10.0.post4 --timeout 60 --retries 5 || exit 14

$PIP list
echo "INSTALL_TORCH_STACK_RC=0"
