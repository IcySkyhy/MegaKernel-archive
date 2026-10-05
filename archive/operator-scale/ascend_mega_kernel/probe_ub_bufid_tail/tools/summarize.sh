#!/usr/bin/env bash
# summarize.sh —— 从 evidence/logs/<mode>_<variant>.log 汇总矩阵（派生工件，可随时重跑，不占设备）
set -u
cd "$(dirname "$0")/.."
LOGS=evidence/logs

summarize() {
    local mode=$1
    local log v n0 hang tot one
    for log in "$LOGS"/${mode}_*.log; do
        [ -e "$log" ] || continue
        v=$(basename "$log" .log); v=${v#"${mode}"_}
        n0=$(grep -cE ' rc=0$' "$log")
        hang=$(grep -c 'rc=124' "$log")
        tot=$(grep -c '^--- run' "$log")
        one=$(grep -m1 '^SUMMARY' "$log")
        printf '%-34s rc==0 %d/%d hang=%d %s\n' "${mode}:${v}" "$n0" "$tot" "$hang" "$one"
    done
}

{
    echo "# tail matrix（由 tools/summarize.sh 从逐变体日志重算）"
    summarize tail
} > "$LOGS/tail_matrix.txt"
{
    echo "# sp matrix（由 tools/summarize.sh 从逐变体日志重算）"
    summarize sp
} > "$LOGS/sp_matrix.txt"
cat "$LOGS/tail_matrix.txt" "$LOGS/sp_matrix.txt"
