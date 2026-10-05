#!/usr/bin/env bash
#
# m26_moe_prefill 防回退自检（M170）：configure + build 本工程两个 target，任一失败即非零退出。
#
# 被咬的对象（finding 20261004-agent-flagreg-bug-m26-...-m85p-m26-tu.md）：
#   `m26_moe_prefill.asc` / `m26_consts.asc` 经 `m15_moe_prefill.h:66 → m15_layer_resources.h`
#   引入资源表，而该表 §4 的 `FLAG_SEQ[]`（`:728-737`）直接引用 `M85P::FLAG_B2/B3/B4`。
#   `M85P` 只由 `m15_ple_wire.h` 提供 ⇒ 两个 .asc 的 include 块里若缺 `#include "m15_ple_wire.h"`
#   （且必须在 `m15_moe_prefill.h` 之前），编译即报 `use of undeclared identifier 'M85P'`
#   及其级联错误（`sizeof` incomplete FlagStep[]、多条 `static_assert` 非整常量）⇒ 本脚本变红。
#
# 用法（仓库根目录或任意目录均可）：
#   bash m26_moe_prefill/check_build.sh
#
# 注意：本脚本只做 host 侧编译，不碰设备、不需要锁。

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJ_DIR}/build"

# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh

rc=0
echo "== configure =="
cmake -B "${BUILD_DIR}" -S "${PROJ_DIR}" -DCMAKE_BUILD_TYPE=Release || rc=$?

if [ "${rc}" -eq 0 ]; then
    echo "== build (default target: m26_moe_prefill) =="
    cmake --build "${BUILD_DIR}" -j4 || rc=$?
fi

if [ "${rc}" -eq 0 ]; then
    echo "== build --target m26_consts =="
    cmake --build "${BUILD_DIR}" -j4 --target m26_consts || rc=$?
fi

if [ "${rc}" -eq 0 ]; then
    echo "m26_moe_prefill self-check: PASS (rc=0)"
else
    echo "m26_moe_prefill self-check: FAIL (rc=${rc})"
fi
exit "${rc}"
