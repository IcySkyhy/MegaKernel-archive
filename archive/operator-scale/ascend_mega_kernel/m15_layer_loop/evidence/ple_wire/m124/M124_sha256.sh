#!/usr/bin/env bash
# M124：sha256 归档（源码 / 产物 / 两个二进制 / 两个 kv 平面 / dump）
# 用法（仓库根）： bash m15_layer_loop/evidence/ple_wire/m124/M124_sha256.sh > m15_layer_loop/evidence/ple_wire/m124/M124_sha256.txt
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO" || exit 1
BASE_REV="${M124_BASE_REV:-9296e79}"
echo "# M124 sha256 归档（生成命令：M124_sha256.sh；BASE_REV=$BASE_REV）"
echo "# 时间（UTC）：$(date -u +%FT%TZ)  tip：$(git rev-parse --short HEAD)（+工作树未提交改动）"
echo
echo "## 1. 本 tip 的源码与产物"
sha256sum m15_layer_loop/m15_ple.asc m15_layer_loop/m15_ple_wire.h \
          m15_layer_loop/m15_ple_check.py m15_layer_loop/m15_ple_mutants.py \
          m15_layer_loop/m15_layer_kernel.h m15_layer_loop/m15_layer_resources.h \
          m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py 2>/dev/null
echo
echo "## 2. 基线 rev 的两个文件（改前的归档面）"
git show "$BASE_REV:m15_layer_loop/m15_ple.asc" | sha256sum | sed "s|-|$BASE_REV:m15_layer_loop/m15_ple.asc|"
git show "$BASE_REV:m15_layer_loop/m15_ple_wire.h" | sha256sum | sed "s|-|$BASE_REV:m15_layer_loop/m15_ple_wire.h|"
echo
echo "## 3. 二进制（改前 = 基线 rev 在 /tmp 重建；改后 = 本 tip 工作树）"
for pair in "/tmp/m124_base/m15_layer_loop/build/m15_ple|m15_ple(改前 $BASE_REV 重建)" \
            "build/m15_ple|m15_ple(改后 本 tip；本批的 build 目录在仓库根 build/)" \
            "build/m15_layer_loop|m15_layer_loop(改后 本 tip)"; do
  f="${pair%%|*}"; label="${pair##*|}"
  if [ -f "$f" ]; then printf "%s  %s\n" "$(sha256sum "$f" | cut -d' ' -f1)" "$label"; else echo "（缺 $f）"; fi
done
echo
echo "## 4. 设备 dump 的关键平面（改前 /tmp/m124_out_base、改后 /tmp/m124_out_new、M=1 /tmp/m124_out_new_m1）"
sha256sum /tmp/m124_out_base/B_kv.bin /tmp/m124_out_new/B_kv.bin /tmp/m124_out_new_m1/B_kv.bin 2>/dev/null
shopt -s nullglob
for f in /tmp/m124_out_base/B_*.bin /tmp/m124_out_new/B_*.bin; do sha256sum "$f"; done 2>/dev/null
