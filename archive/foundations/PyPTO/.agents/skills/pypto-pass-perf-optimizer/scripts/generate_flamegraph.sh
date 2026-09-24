#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 火焰图生成脚本
# 功能：使用 perf + FlameGraph 生成 CPU 火焰图
#
# 用法：
#   bash generate_flamegraph.sh <超时时间(秒)> <输出目录> <命令...>
#
# 示例：
#   bash generate_flamegraph.sh 300 ./flamegraphs python3 test.py
#   bash generate_flamegraph.sh 600 ./perf_data python3 my_operator.py --arg1 value1
#
# 输出文件：
#   - flamegraph_{timestamp}.svg  : 火焰图文件（用浏览器打开）
#   - folded_{timestamp}.txt      : 折叠数据文件（用于后续对比）
#   - perf_{timestamp}.data       : perf 原始数据文件
# -----------------------------------------------------------------------------------------------------------

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 参数解析
TIMEOUT=${1:-300}
OUTPUT_DIR=${2:-"./flamegraphs"}
SCRIPT_CMD="${@:3}"

# 检查命令参数
if [ -z "$SCRIPT_CMD" ]; then
    echo -e "${RED}错误：未指定要执行的命令${NC}"
    echo ""
    echo "用法: $0 <超时时间(秒)> <输出目录> <命令...>"
    echo ""
    echo "示例:"
    echo "  $0 300 ./flamegraphs python3 test.py"
    echo "  $0 600 ./perf_data python3 my_operator.py --arg1 value1"
    exit 1
fi

# FlameGraph 工具路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAMEGRAPH_DIR="$SCRIPT_DIR/FlameGraph"

# 自动下载 FlameGraph（如果不存在）
if [ ! -d "$FLAMEGRAPH_DIR" ]; then
    echo -e "${YELLOW}FlameGraph 工具未找到，正在自动下载...${NC}"
    echo ""
    git clone --depth 1 https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR"
    echo ""
    echo -e "${GREEN}✓ FlameGraph 工具下载完成${NC}"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERF_DATA="$OUTPUT_DIR/perf_${TIMESTAMP}.data"
SVG_FILE="$OUTPUT_DIR/flamegraph_${TIMESTAMP}.svg"
FOLDED_FILE="$OUTPUT_DIR/folded_${TIMESTAMP}.txt"

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}火焰图生成配置${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "  超时时间: ${GREEN}${TIMEOUT}s${NC}"
echo -e "  输出目录: ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "  执行命令: ${GREEN}${SCRIPT_CMD}${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# perf 采样（使用 run_with_timeout.sh 控制超时）
echo -e "${YELLOW}[1/3] 开始 perf 采样...${NC}"
echo ""

bash "$SCRIPT_DIR/run_with_timeout.sh" $TIMEOUT \
    perf record -F 99 -g -e cycles -o "$PERF_DATA" -- $SCRIPT_CMD

# 检查 perf 数据是否生成
if [ ! -f "$PERF_DATA" ]; then
    echo ""
    echo -e "${RED}✗ 错误：perf 数据文件未生成${NC}"
    echo -e "${YELLOW}  可能原因：${NC}"
    echo "    1. perf 命令执行失败"
    echo "    2. 权限不足（需要 root 或设置 /proc/sys/kernel/perf_event_paranoid）"
    echo "    3. 程序过快退出"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ perf 采样完成${NC}"
echo ""

# 生成折叠数据（用于后续对比）
echo -e "${YELLOW}[2/3] 生成折叠数据...${NC}"
echo ""

perf script -i "$PERF_DATA" | \
    "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" > "$FOLDED_FILE"

if [ ! -f "$FOLDED_FILE" ]; then
    echo -e "${RED}✗ 错误：折叠数据文件生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 折叠数据生成完成${NC}"
echo ""

# 生成火焰图
echo -e "${YELLOW}[3/3] 生成火焰图...${NC}"
echo ""

"$FLAMEGRAPH_DIR/flamegraph.pl" \
    --title "PyPTO Pass Performance Flame Graph" \
    "$FOLDED_FILE" > "$SVG_FILE"

if [ ! -f "$SVG_FILE" ]; then
    echo -e "${RED}✗ 错误：火焰图文件生成失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✓ 火焰图生成完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "${BLUE}输出文件：${NC}"
echo -e "  ${GREEN}🔥 火焰图文件：${NC}"
echo -e "     ${SVG_FILE}"
echo ""
echo -e "  ${BLUE}📊 折叠数据文件（用于对比）：${NC}"
echo -e "     ${FOLDED_FILE}"
echo ""
echo -e "  ${BLUE}📁 perf 原始数据：${NC}"
echo -e "     ${PERF_DATA}"
echo ""
echo -e "${YELLOW}使用浏览器打开火焰图：${NC}"
echo -e "  firefox ${SVG_FILE}"
echo -e "  或"
echo -e "  google-chrome ${SVG_FILE}"
echo ""
echo -e "${YELLOW}下次优化后，可使用对比功能：${NC}"
echo -e "  bash compare_flamegraphs.sh ${FOLDED_FILE} <优化后的folded文件>"
echo ""
