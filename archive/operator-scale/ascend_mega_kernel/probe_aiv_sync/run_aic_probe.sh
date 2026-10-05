#!/usr/bin/env bash
# run_aic_probe.sh —— AIC↔AIV 对照探针证据（AIV 主探针之外单独跑，避免与主 run_probe.sh 互相扰动）
# 用法：bash run_aic_probe.sh           # v0/v3/v4/v5 各 9 次；v1/v2（预期挂死）各 9 次、单次 20s 超时
set -u
cd "$(dirname "$0")"
ROOT="$PWD"; EV="$ROOT/evidence"; LOGS="$EV/logs"; TMP="$ROOT/out_tmp"
REPS="${REPS:-9}"; TO_HANG="${TO_HANG:-20}"; TO="${TO:-90}"
BIN="$ROOT/build/probe_aic_aiv"
mkdir -p "$LOGS" "$TMP"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
kill_mine() { pkill -f "^${BIN} " 2>/dev/null || true; }
: > "$LOGS/aic_aiv_matrix.txt"
for v in v0_chain_mix_standard v3_aiv_only v4_aic_only v5_aic_relay_nobar v1_aic_set_wrong_pipe v2_cross_group_mode2_wait; do
    log="$LOGS/aic_aiv_${v}.log"; : > "$log"
    tmo="$TO"; case "$v" in v1_*|v2_*) tmo="$TO_HANG";; esac
    for i in $(seq 1 "$REPS"); do
        timeout "$tmo" "$BIN" "$v" "$i" "$TMP" >> "$log" 2>&1
        rc=$?
        [ $rc -eq 124 ] && { echo "# $v run=$i 超时未返回（预期挂死变体；见 README §7.5）" >> "$log"; echo "SUMMARY variant=$v run=$i HANG=1 rc=124" >> "$log"; }
        [ $v = "v1_aic_set_wrong_pipe" -o $v = "v2_cross_group_mode2_wait" ] && kill_mine
    done
    n0=$(grep -c " rc=0$" "$log"); hang=$(grep -c "rc=124" "$log"); nf=$(grep -c "rc=2$" "$log")
    printf '%-26s rc0=%d/%d hang=%d launchfail=%d hit_full=%s final_head_ok=%s\n' "$v" "$n0" "$REPS" "$hang" "$nf" \
        "$(grep -o 'hit_full=[0-9]*/[0-9]*' "$log" | sort -u | tr '\n' ' ')" \
        "$(grep -o 'final_head_ok=[0-9]*' "$log" | sort -n | uniq -c | tr '\n' ' ')" | tee -a "$LOGS/aic_aiv_matrix.txt"
done
kill_mine
echo "done"
