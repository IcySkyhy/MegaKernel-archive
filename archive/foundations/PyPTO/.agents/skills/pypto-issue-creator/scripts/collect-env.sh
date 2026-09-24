#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# PyPTO 环境信息自动采集脚本
# 用法: bash scripts/collect-env.sh
# 输出: 格式化环境信息，可直接粘贴到 Bug Report 的 Environment 字段

set -euo pipefail

pick_python() {
    if command -v python >/dev/null 2>&1; then
        echo "python"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return
    fi
    echo ""
}

extract_semver() {
    local text="$1"
    local version=""
    version=$(printf '%s\n' "$text" | sed -n 's/.*\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
    printf '%s\n' "$version"
}

get_npu_model() {
    local device_ids
    if ! command -v lspci >/dev/null 2>&1; then
        echo "Unknown"
        return
    fi
    device_ids=$(lspci -n -D 2>/dev/null | grep -oE '19e5:d80[23]' || true)
    case "$device_ids" in
        *19e5:d802*) echo "Ascend 910B" ;;
        *19e5:d803*) echo "Ascend 910C" ;;
        *) echo "Unknown" ;;
    esac
}

get_cann_version() {
    local version=""
    local candidates=""
    version=$(printf '%s\n' "${ASCEND_HOME_PATH:-}" | sed -n 's/.*cann-\([0-9][0-9.]*\).*/\1/p' | head -1 || true)
    version=$(extract_semver "$version")
    if [ -z "$version" ]; then
        candidates=$(ls -d /usr/local/Ascend/ascend-toolkit/* 2>/dev/null || true)
        version=$(printf '%s\n' "$candidates" | sed -n 's/.*\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | sort -u | tail -n 1 || true)
    fi
    if [ -z "$version" ]; then
        if command -v npu-smi >/dev/null 2>&1; then
            version=$(npu-smi info 2>/dev/null | sed -n 's/.*Version[[:space:]]*:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -1 || true)
            version=$(extract_semver "$version")
        fi
    fi
    printf '%s\n' "${version:-Unknown}"
}

sanitize_single_line() {
    local raw="$1"
    local line=""
    local result=""
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        case "$line" in
            Warning\ :\ ASCEND_HOME_PATH\ environment\ variable\ is\ not\ set.) continue ;;
            *) result="$line" ;;
        esac
    done <<EOF
$raw
EOF
    printf '%s\n' "${result:-Unknown}"
}

get_pypto_version() {
    local head_hash head_time
    if ! command -v git >/dev/null 2>&1; then
        printf '%s\n' "Unknown"
        return
    fi

    head_hash=$(git rev-parse --short HEAD 2>/dev/null || true)
    head_time=$(git log -1 --format='%ci' HEAD 2>/dev/null || true)
    if [ -n "$head_hash" ] && [ -n "$head_time" ]; then
        printf '%s (%s)\n' "$head_hash" "$head_time"
        return
    fi

    if [ -n "$head_hash" ]; then
        printf '%s\n' "$head_hash"
        return
    fi

    printf '%s\n' "Unknown"
}

PYTHON_BIN=$(pick_python)
npu_model=$(get_npu_model)
cann_version=$(get_cann_version)
pypto_version=$(get_pypto_version)

if [ -n "$PYTHON_BIN" ]; then
    python_version=$(sanitize_single_line "$($PYTHON_BIN --version 2>&1 || true)")
    torch_version=$(sanitize_single_line "$($PYTHON_BIN -c "import importlib; m=importlib.import_module('torch'); print(getattr(m, '__version__', 'Unknown'))" 2>/dev/null || true)")
    torch_npu_version=$(sanitize_single_line "$($PYTHON_BIN -c "import importlib; m=importlib.import_module('torch_npu'); print(getattr(m, '__version__', 'Unknown'))" 2>/dev/null || true)")
else
    python_version="Unknown"
    torch_version="Unknown"
    torch_npu_version="Unknown"
fi

os_info=$(grep '^PRETTY_NAME=' /etc/os-release 2>/dev/null | cut -d'=' -f2- | tr -d '"' || true)
os_info=${os_info:-$(uname -s -r 2>/dev/null || printf '%s' "Unknown")}

cat <<EOF
- **服务器/NPU 型号**: ${npu_model}
- **PyPTO 版本/Commit**: ${pypto_version}
- **CANN 版本**: ${cann_version}
- **Python 版本**: ${python_version}
- **操作系统**: ${os_info}
- **torch 版本**: ${torch_version}
- **torch_npu 版本**: ${torch_npu_version}
EOF
