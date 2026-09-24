#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 火焰图对比脚本
# 功能：对比优化前后的火焰图，生成差异图
#
# 用法：
#   bash compare_flamegraphs.sh <优化前folded文件> <优化后folded文件> [输出目录]
#
# 示例：
#   bash compare_flamegraphs.sh ./flamegraphs/folded_before.txt ./flamegraphs/folded_after.txt
#   bash compare_flamegraphs.sh ./flamegraphs/folded_before.txt ./flamegraphs/folded_after.txt ./diff_output
#
# 输出文件：
#   - diff_flamegraph_{timestamp}.svg : 差异火焰图
#     红色/橙色 = 优化后增加的热点
#     蓝色/青色 = 优化后减少的热点
# -----------------------------------------------------------------------------------------------------------

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 参数解析
BEFORE_FOLDED=${1:-""}
AFTER_FOLDED=${2:-""}
OUTPUT_DIR=${3:-"./flamegraphs"}

# FlameGraph 工具路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAMEGRAPH_DIR="$SCRIPT_DIR/FlameGraph"

# 参数检查
if [ -z "$BEFORE_FOLDED" ] || [ -z "$AFTER_FOLDED" ]; then
    echo ""
    echo -e "${RED}错误：缺少必要参数${NC}"
    echo ""
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 <优化前folded文件> <优化后folded文件> [输出目录]"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0 ./flamegraphs/folded_20260313_143022.txt ./flamegraphs/folded_20260313_150335.txt"
    echo "  $0 ./flamegraphs/folded_before.txt ./flamegraphs/folded_after.txt ./diff_output"
    echo ""
    echo -e "${BLUE}说明:${NC}"
    echo "  folded 文件由 generate_flamegraph.sh 生成"
    echo "  对比图将显示："
    echo "    - 红色/橙色 = 优化后增加的热点（需要关注）"
    echo "    - 蓝色/青色 = 优化后减少的热点（优化有效）"
    echo ""
    echo -e "${YELLOW}查找可用的 folded 文件:${NC}"
    echo "  find ./flamegraphs -name 'folded_*.txt' -type f | sort"
    echo ""
    exit 1
fi

# 检查 FlameGraph 工具
if [ ! -d "$FLAMEGRAPH_DIR" ]; then
    echo ""
    echo -e "${RED}错误：FlameGraph 工具未找到${NC}"
    echo ""
    echo -e "${YELLOW}请先运行 generate_flamegraph.sh 自动下载 FlameGraph 工具${NC}"
    echo "  bash generate_flamegraph.sh 300 ./flamegraphs python3 test.py"
    echo ""
    exit 1
fi

# 检查 difffolded.pl 是否存在
if [ ! -f "$FLAMEGRAPH_DIR/difffolded.pl" ]; then
    echo ""
    echo -e "${RED}错误：difffolded.pl 工具未找到${NC}"
    echo ""
    echo -e "${YELLOW}FlameGraph 工具可能不完整，请删除后重新下载${NC}"
    echo "  rm -rf $FLAMEGRAPH_DIR"
    echo "  然后重新运行 generate_flamegraph.sh"
    echo ""
    exit 1
fi

# 检查输入文件
if [ ! -f "$BEFORE_FOLDED" ]; then
    echo ""
    echo -e "${RED}错误：优化前 folded 文件不存在${NC}"
    echo "  文件路径: $BEFORE_FOLDED"
    echo ""
    echo -e "${YELLOW}查找可用的 folded 文件:${NC}"
    echo "  find ./flamegraphs -name 'folded_*.txt' -type f | sort"
    echo ""
    exit 1
fi

if [ ! -f "$AFTER_FOLDED" ]; then
    echo ""
    echo -e "${RED}错误：优化后 folded 文件不存在${NC}"
    echo "  文件路径: $AFTER_FOLDED"
    echo ""
    echo -e "${YELLOW}查找可用的 folded 文件:${NC}"
    echo "  find ./flamegraphs -name 'folded_*.txt' -type f | sort"
    echo ""
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIFF_SVG="$OUTPUT_DIR/diff_flamegraph_${TIMESTAMP}.svg"

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}火焰图对比配置${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "  优化前数据: ${GREEN}${BEFORE_FOLDED}${NC}"
echo -e "  优化后数据: ${GREEN}${AFTER_FOLDED}${NC}"
echo -e "  输出目录: ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 显示文件大小信息
BEFORE_SIZE=$(du -h "$BEFORE_FOLDED" | cut -f1)
AFTER_SIZE=$(du -h "$AFTER_FOLDED" | cut -f1)
echo -e "${BLUE}文件大小:${NC}"
echo -e "  优化前: ${BEFORE_SIZE}"
echo -e "  优化后: ${AFTER_SIZE}"
echo ""

# 生成差异火焰图
echo -e "${YELLOW}生成差异火焰图...${NC}"
echo ""

"$FLAMEGRAPH_DIR/difffolded.pl" "$BEFORE_FOLDED" "$AFTER_FOLDED" | \
    "$FLAMEGRAPH_DIR/flamegraph.pl" \
    --title "PyPTO Pass Performance Diff Flame Graph" \
    --colors "aqua" > "$DIFF_SVG"

if [ ! -f "$DIFF_SVG" ]; then
    echo ""
    echo -e "${RED}✗ 错误：差异火焰图生成失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✓ 差异火焰图生成完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "${BLUE}输出文件：${NC}"
echo -e "  ${GREEN}🔥 差异火焰图：${NC}"
echo -e "     ${DIFF_SVG}"
echo ""
echo -e "${YELLOW}使用浏览器打开查看：${NC}"
echo -e "  firefox ${DIFF_SVG}"
echo -e "  或"
echo -e "  google-chrome ${DIFF_SVG}"
echo ""
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}🎨 颜色说明${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""
echo -e "  ${RED}🔴 红色/橙色${NC}  → 优化后增加的热点"
echo -e "                 ⚠️  需要关注，可能是新引入的性能问题"
echo ""
echo -e "  ${BLUE}🔵 蓝色/青色${NC}  → 优化后减少的热点"
echo -e "                 ✅ 优化有效，这些函数耗时减少"
echo ""
echo -e "  ⚪ 灰色        → 基本不变"
echo -e "                 优化对该函数影响不大"
echo ""
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}📊 对比分析要点${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""
echo "  1. 关注红色区域：优化后新出现或增加的热点"
echo "  2. 确认蓝色区域：验证优化是否对目标函数有效"
echo "  3. 记录主要函数优化前后的占比变化"
echo ""
