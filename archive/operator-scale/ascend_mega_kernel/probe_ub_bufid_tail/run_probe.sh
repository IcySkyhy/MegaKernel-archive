#!/usr/bin/env bash
# run_probe.sh —— M126 探针证据生成（设备槽纪律：flock + 进锁先 npu-smi + 每条 timeout）
#
# 用法（本目录下）：
#   bash run_probe.sh tail 'base|multi3'          # 只跑名字匹配该正则的 tail 变体
#   REPS=3 bash run_probe.sh sp 'mte2|^v_'        # sp 变体
#   REPS=1 bash run_probe.sh all                  # 全部（tail + sp）
#
# 脚本自行取锁（PROBE_LOCKED 防重入）：`exec flock -w 300 /tmp/npu0.lock "$0" "$@"`。
# 每条变体每次运行都用 `timeout`；exit=124 记成「挂死（未取得内容读数）」，不写成「没复现」。
#
# 产物（进 git）：evidence/logs/<mode>_<variant>.log、evidence/logs/<mode>_matrix.txt、evidence/commands.txt
set -u

cd "$(dirname "$0")"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
OUT="$ROOT/out_tmp"
BIN="$ROOT/build/probe_ub_bufid_tail"
REPS="${REPS:-3}"
TO="${TO:-40}"          # 单次运行超时（秒）
FILTER="${FILTER:-.}"
TRACE="${TRACE:-}"      # 非空 ⇒ 每次运行加 --trace（kernel 内逐次 printf）

if [ -z "${PROBE_LOCKED:-}" ]; then
    export PROBE_LOCKED=1
    exec flock -w 300 /tmp/npu0.lock "$ROOT/run_probe.sh" "$@"
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [ ! -x "$BIN" ]; then
    echo "== 构建 =="
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release > "$OUT.cmake" 2>&1 || { cat "$OUT.cmake"; exit 1; }
    cmake --build build -j4 > "$OUT.build" 2>&1 || { tail -30 "$OUT.build"; exit 1; }
fi

mkdir -p "$LOGS" "$OUT"
MODE="${1:-all}"        # tail | sp | all
[ $# -ge 2 ] && FILTER="$2"

{
    echo "# probe_ub_bufid_tail 证据生成记录（M126）"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "npu-smi         :"
    npu-smi info 2>&1 | sed -n '1,12p'
    echo "compiler        : $(bisheng --version 2>&1 | head -2 | tr '\n' ' ')"
    echo "reps per variant: $REPS"
    echo "per-run timeout : ${TO}s"
    echo "trace           : ${TRACE:-0}"
    echo
    echo "## 复现命令（本目录执行；脚本自带 flock）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
    echo "bash run_probe.sh all          # 全部变体"
    echo "bash run_probe.sh tail base_N3 # 单变体"
    echo
    echo "## 源码 sha256"
    sha256sum probe_ub_bufid_tail.asc CMakeLists.txt run_probe.sh
} > "$EV/commands.txt"

echo "== 进锁成功 =="
npu-smi info 2>&1 | sed -n '5,7p'
echo "date=$(date -Is) mode=$MODE filter=$FILTER reps=$REPS timeout=${TO}s"

run_one() {
    local mode="$1" v="$2" tmo="$3"
    local log="$LOGS/${mode}_${v}.log"
    : > "$log"
    local i
    for i in $(seq 1 "$REPS"); do
        echo "--- run $i (timeout ${tmo}s) ---" >> "$log"
        timeout "$tmo" "$BIN" "$mode" "$v" "$i" "$OUT" ${TRACE:+--trace} >> "$log" 2>&1
        local rc=$?
        echo "exit=$rc" >> "$log"
        if [ "$rc" -eq 124 ]; then
            echo "SUMMARY mode=$mode variant=$v run=$i HANG=1 rc=124（超时未返回：按纪律记「未取得读数」，不记「没复现」）" >> "$log"
        fi
    done
    local n0 hang
    n0=$(grep -cE ' rc=0$' "$log")
    hang=$(grep -c 'rc=124' "$log")
    # 取一条内容读数
    local one
    one=$(grep -m1 '^SUMMARY' "$log")
    printf '%-34s rc==0 %d/%d  hang=%d  %s\n' "${mode}:${v}" "$n0" "$REPS" "$hang" "$one" | tee -a "$LOGS/${mode}_matrix.txt"
}

if [ "$MODE" = "tail" ] || [ "$MODE" = "all" ]; then
    echo "## matrix $(date -Is) mode=tail filter=$FILTER reps=$REPS" >> "$LOGS/tail_matrix.txt"
fi
if [ "$MODE" = "sp" ] || [ "$MODE" = "all" ]; then
    echo "## matrix $(date -Is) mode=sp filter=$FILTER reps=$REPS" >> "$LOGS/sp_matrix.txt"
fi

if [ "$MODE" = "tail" ] || [ "$MODE" = "all" ]; then
    echo "== tail 变体 =="
    while read -r v; do
        [ -z "$v" ] && continue
        echo "$v" | grep -Eq "$FILTER" || continue
        run_one tail "$v" "$TO"
    done < <("$BIN" list tail)
fi

if [ "$MODE" = "sp" ] || [ "$MODE" = "all" ]; then
    echo "== sp 变体 =="
    while read -r v; do
        [ -z "$v" ] && continue
        echo "$v" | grep -Eq "$FILTER" || continue
        run_one sp "$v" "$TO"
    done < <("$BIN" list sp)
fi

echo "完成。证据目录：$EV"
