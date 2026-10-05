#!/usr/bin/env bash
# probe_ub2l1/evidence/slot.sh —— 取一个设备槽，跑**一条**短命令（≤120s）。
#
# 设备纪律（塔裁，2026-10-04 收紧版）：
#   ① 一次 flock 只跑一条短命令；② flock -w ≤ 300；③ **进锁前先 flock -n 探锁**，探不到直接退，
#   不空等（退码 111 = SKIP_BUSY，调用方如实记「本轮未取到」并稍后重试）；
#   ④ 单进程、必带 timeout；⑤ 进锁后先 npu-smi 复查并把结果原样记进本档的 log。
#
# 用法：bash evidence/slot.sh <name> <timeout_s> <cmd...>
#   LOG_DIR=<dir>  每档 log 落在 <dir>/<name>.log（默认 evidence/logs）
# 退出码：0..124 = 那条命令自己的 rc（124 = timeout 到点未返回，即"挂死"）；111 = 锁被别人占着（未执行）。
#
# **只追加的尝试流水**：每次调用都会往 <LOG_DIR>/attempts.log 追加一行（时间/档名/结果）。
#   为什么需要它：每档 log 是 `>` 截断写的，同一档"先失败、后成功"时，失败那次的记录会被覆盖
#   （r2 复审 P2-2 正是踩到这个）。attempts.log 用 `>>` 追加，不会被覆盖。
set -u
NAME="$1"; TMO="$2"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$HERE/logs}"
LOCK=/tmp/npu0.lock
LOG="$LOG_DIR/$NAME.log"
mkdir -p "$LOG_DIR"

# 探锁：先非阻塞试一次（快路径）。探不到时按 M121_WAIT（默认 0 = 不等）做**有界**获取：
#   塔的纪律允许 `-w ≤300`；默认 0 表示严格"探不到就走"，只在明确要补齐读数时把它设成 ≤300。
ATTEMPTS="$LOG_DIR/attempts.log"
WAIT="${M121_WAIT:-0}"
if ! flock -n "$LOCK" true 2>/dev/null; then
    if [ "$WAIT" = "0" ]; then
        echo "$(date -Is) name=$NAME wait=$WAIT result=SKIP_BUSY（未执行；未取得读数）" >> "$ATTEMPTS"
        echo "rc=111 name=$NAME SKIP_BUSY（设备锁被别人持有；-n 探锁未过且 M121_WAIT=0，本轮不计入读数）"
        exit 111
    fi
    echo "PROBE_BUSY name=$NAME → 做一次有界获取 flock -w $WAIT（≤300）"
fi

{
    echo "### $NAME  timeout=${TMO}s  at=$(date -Is)"
    echo "### cmd: $*"
    echo "### 进锁后 npu-smi 复查："
    npu-smi info 2>&1
    echo "### 命令输出："
} > "$LOG"

# 真正执行：一次 flock 只放这一条命令。
# 关键：**把 timeout 放在锁里面**（flock ... bash -c 'timeout T cmd'），否则 timeout 会把"等锁"的时间
# 也算进去，等锁超时也会报 124 —— 那就和"挂死"（kill=2/5 的预期读数）混成一回事了，读数就废了。
W="$WAIT"; [ "$W" -gt 300 ] && W=300
CMD=""; for a in "$@"; do CMD="$CMD $(printf '%q' "$a")"; done
# 进锁后在 log 里打一个**唯一标记**：用来区分「没拿到锁（命令根本没跑）」与
# 「命令自己返回 1（例如 capi-rt 判定 MISMATCH 就是 rc=1）」——两者靠返回码分不开。
MARK="__SLOT_ENTERED_${NAME}_$$"
flock -w "$W" "$LOCK" bash -c "echo $MARK; timeout $TMO $CMD" >> "$LOG" 2>&1
RC=$?
if ! grep -q "$MARK" "$LOG" 2>/dev/null; then
    # 把"本条命令根本没跑"写进 log 本身，免得后人把"等锁没拿到"误读成"跑过/挂死"
    {
        echo "### 未执行：LOCK_ACQUIRE_FAILED —— 在 -w $W 内没拿到 /tmp/npu0.lock，命令没有运行。"
        echo "### 按塔的口径：这记「**未取得读数**」，不是「未复现」，也不是「挂死」。重跑本档再取。"
    } >> "$LOG"
    echo "$(date -Is) name=$NAME wait=$W result=LOCK_ACQUIRE_FAILED（未执行；未取得读数）" >> "$ATTEMPTS"
    echo "rc=1 name=$NAME LOCK_ACQUIRE_FAILED（-w $W 内没拿到锁，未执行；log 里没有进锁标记）"
    exit 1
fi
echo "$(date -Is) name=$NAME wait=$W result=rc$RC executed=yes" >> "$ATTEMPTS"
echo "rc=$RC name=$NAME executed=yes"
{
    echo "### 跑后 npu-smi（并发背景）："
    npu-smi info 2>&1
    echo "### rc=$RC"
} >> "$LOG"

echo "rc=$RC name=$NAME"
exit "$RC"
