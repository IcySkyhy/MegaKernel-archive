#!/usr/bin/env bash
# summarize.sh —— 从逐次日志机械重算 matrix_N<N>.txt（不改写任何原始日志）
#
# 扫描 evidence/logs/ 下的 `<variant>_N<N>.log`（非 trace 档），取其中的 `AGG` 行，按 N 汇总。
# `--trace` 档的日志名带 `_trace` 后缀，不匹配 `*_N<N>.log`，故不进矩阵。
# 用法：bash tools/summarize.sh
set -u
cd "$(dirname "$0")/.."
LOGS=evidence/logs
for n in 4 16 32 64; do
    out="$LOGS/matrix_N${n}.txt"
    : > "$out"
    echo "# variant N=${n}：由 tools/summarize.sh 从逐次日志的 AGG 行重算（每档 3 次独立进程）" >> "$out"
    for f in "$LOGS"/*_N${n}.log; do
        [ -f "$f" ] || continue
        agg=$(grep -m1 '^AGG ' "$f" || true)
        [ -z "$agg" ] && continue
        tag=$(basename "$f" | sed "s/_N${n}.log//")
        printf '%-18s %s\n' "$tag" "${agg#AGG }" >> "$out"
    done
    echo "--- matrix_N${n}.txt ---"
    cat "$out"
done
