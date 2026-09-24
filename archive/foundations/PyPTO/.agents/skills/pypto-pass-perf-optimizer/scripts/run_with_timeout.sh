#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# run_with_timeout.sh - 带超时控制的命令执行脚本（进程组信号传播版）
# =============================================================================
# 功能：
#   1. 在独立进程组中执行命令（set -m 作业控制）
#   2. 监控执行时间
#   3. 超时后向整个进程组发送信号（Python 父进程 + C++ 子进程均能收到）
#   4. 三级信号升级：SIGINT(10s) -> SIGTERM(5s) -> SIGKILL
#   5. 返回合适的退出码
#
# 参数：
#   $1: 超时时间（秒），默认300秒（5分钟）
#   $2...: 要执行的命令及其参数
#
# 退出码：
#   0   - 成功完成或超时中断（允许继续后续步骤）
#   非0 - 执行失败（停止后续步骤）
#
# 使用示例：
#   bash run_with_timeout.sh 300 python3 test.py
#   bash run_with_timeout.sh 600 python3 test.py --run-mode npu
#
# === 信号传播原理 ===
# 原方案仅 kill -INT $CMD_PID，信号只发给 Python 父进程，
# C++ 编译子进程（fork 出来的）收不到信号继续运行。
#
# 本方案启用 set -m（作业控制），使后台命令运行在独立进程组中，
# 进程组 ID (PGID) == 命令 PID。通过 kill -INT -$PGID 向整个进程组
# 发送信号，Python 父进程和所有 C++ 子进程都能收到。
# =============================================================================

# 启用作业控制：后台命令自动获得独立进程组，PGID == PID
# 这是信号能传播到子进程的关键
set -m
set -e

# 参数处理
TIMEOUT_SECONDS=${1:-300}  # 默认5分钟
shift                       # 移除超时参数，剩余为命令
SCRIPT_CMD=("$@")           # 保存命令数组

echo "========================================="
echo "执行命令: ${SCRIPT_CMD[*]}"
echo "超时设置: ${TIMEOUT_SECONDS} 秒"
echo "进程组模式: 已启用 (set -m)"
echo "========================================="

# 在后台执行命令（作业控制确保它获得独立进程组）
"${SCRIPT_CMD[@]}" &
CMD_PID=$!
# set -m 下，后台作业的 PID == 进程组 ID (PGID)
PGID=$CMD_PID

# 监控进程（后台子 shell）
(
    sleep "$TIMEOUT_SECONDS"
    if kill -0 "$CMD_PID" 2>/dev/null; then
        echo ""
        echo "⚠️  算子执行已超过 ${TIMEOUT_SECONDS} 秒"
        echo "⚠️  向进程组 (PGID=$PGID) 发送 SIGINT 信号..."
        echo "✓  Pass 编译日志已保存，将继续性能分析"

        # 第1级：SIGINT 到整个进程组（优雅退出，Ctrl+C 等效）
        kill -INT -"$PGID" 2>/dev/null || true

        # 等待 10 秒让进程优雅退出（刷新日志、释放资源）
        sleep 10

        # 第2级：如果仍在运行，升级为 SIGTERM
        if kill -0 "$CMD_PID" 2>/dev/null; then
            echo "⚠️  SIGINT 后进程仍在运行，升级为 SIGTERM..."
            kill -TERM -"$PGID" 2>/dev/null || true
            sleep 5
        fi

        # 第3级：最终手段 SIGKILL（不可被忽略）
        if kill -0 "$CMD_PID" 2>/dev/null; then
            echo "⚠️  SIGTERM 后进程仍在运行，升级为 SIGKILL..."
            kill -KILL -"$PGID" 2>/dev/null || true
        fi

        wait "$CMD_PID" 2>/dev/null || true
    fi
) &
TIMEOUT_PID=$!

# 命令可能被超时信号中断，wait 返回非零属于预期，需捕获后判断
set +e
wait "$CMD_PID" 2>/dev/null
EXIT_CODE=$?
set -e

# 清理监控进程（如果命令正常完成，监控进程还在 sleep）
kill "$TIMEOUT_PID" 2>/dev/null || true
wait "$TIMEOUT_PID" 2>/dev/null || true

# 处理退出码
# 130 = SIGINT  (128+2), 137 = SIGKILL (128+9), 143 = SIGTERM (128+15)
if [ "$EXIT_CODE" -eq 130 ] || [ "$EXIT_CODE" -eq 137 ] || [ "$EXIT_CODE" -eq 143 ]; then
    echo ""
    echo "✓ 算子执行被中断（退出码: $EXIT_CODE）"
    echo "✓ Pass 编译日志已保存，可以继续性能分析"
    exit 0  # 正常退出，允许继续后续步骤
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "✗ 算子执行失败，退出码: $EXIT_CODE"
    exit "$EXIT_CODE"
else
    echo ""
    echo "✓ 算子执行完成"
    exit 0
fi
