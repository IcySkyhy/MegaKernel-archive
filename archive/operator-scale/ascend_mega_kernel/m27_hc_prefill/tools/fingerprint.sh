#!/usr/bin/env bash
# M119（B5）证据指纹：把"这次判据到底判了什么"钉死成一份可复算的清单。
#
#   bash m27_hc_prefill/tools/fingerprint.sh [dump 目录 …]        # 缺省 /tmp/m27_out
#
# 三段：
#   A. **被测源码 + 判据源码 + read-only 依赖**的 sha256（含 `m20_hyperconn/check_ref.py` —— 判据的
#      参考就来自它；它一变，历史读数就不能再引用）；
#   B. **二进制**的 sha256（真机读数属于哪个二进制）；
#   C. **dump 产物**的 sha256 + 字节数（大文件只记大小与 sha256，不落库）。
#
# 用法示例（写进 evidence/sha256.txt）：
#   bash m27_hc_prefill/tools/fingerprint.sh /tmp/m27_out /tmp/m27_out_mt12 /tmp/m27_mut/rowsteal > m27_hc_prefill/evidence/sha256.txt
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
echo "# M119（B5）证据指纹（$(date -u +%Y-%m-%dT%H:%M:%SZ) UTC）"
echo "# 复现：bash m27_hc_prefill/tools/fingerprint.sh <dump 目录…>"
echo
echo "## A. 源码与 read-only 依赖"
for f in \
    m15_layer_loop/m15_hc_prefill.h \
    m15_layer_loop/m15_hc_prefill_host.h \
    m15_layer_loop/m15_hc_layer.h \
    m15_layer_loop/m15_hc_resources.h \
    m20_hyperconn/check_ref.py \
    m15_layer_loop/weights_manifest.txt \
    m27_hc_prefill/m27_hc_prefill.asc \
    m27_hc_prefill/check_ref.py \
    m27_hc_prefill/CMakeLists.txt \
    m27_hc_prefill/reproduce.sh \
    m27_hc_prefill/tools/selftest_dump.py ; do
    if [ -f "$REPO/$f" ]; then
        printf '%-52s %s\n' "$f" "$(sha256sum "$REPO/$f" | cut -d' ' -f1)"
    else
        printf '%-52s %s\n' "$f" "（缺失）"
    fi
done
echo
echo "## B. 二进制"
for b in m27_hc_prefill/build/m27_hc_prefill m27_hc_prefill/build/m27_hc_prefill_mt12; do
    if [ -f "$REPO/$b" ]; then
        printf '%-52s %s\n' "$b" "$(sha256sum "$REPO/$b" | cut -d' ' -f1)"
    else
        printf '%-52s %s\n' "$b" "（未构建）"
    fi
done
echo
echo "## C. dump 产物（<目录> <文件> <字节> <sha256>）"
for d in "$@"; do
    echo "### $d"
    if [ ! -d "$d" ]; then
        echo "（缺目录）"
        continue
    fi
    ( cd "$d" && find . -maxdepth 1 -type f \( -name '*.bin' -o -name '*.txt' -o -name '*.log' \) -printf '%f\n' \
        | sort | while read -r f; do
            printf '%-34s %10d %s\n' "$f" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
        done )
done
