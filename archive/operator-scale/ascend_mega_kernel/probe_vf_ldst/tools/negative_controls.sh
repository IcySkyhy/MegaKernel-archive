#!/usr/bin/env bash
# probe_vf_ldst/tools/negative_controls.sh —— summarize.py 的**负向对照**（按 tower 规则：
# "找一条你的匹配器本来就看不见的样本，确认它被报成不合格，而不是被当成通过"）。
#
# 三条对照，全部只在 /tmp 副本上做，不改 evidence/ 里的真证据：
#   ① 空目录（无 run_*.log、无 declared_groups.txt）      ⇒ 期望 RESULT: SKIPPED + exit 2
#   ② 删掉两个**整组**变体的日志（已知会被漏掉的那一态）  ⇒ 期望 RESULT: DIFF（缺组）+ exit 1
#   ③ 只留 1/5 个进程                                     ⇒ 期望 RESULT: DIFF（进程数不足）+ exit 1
#
# 归档：evidence/logs/negative_control_{nocompare,partial_missing,fewer_procs}.log
# 退出码：0 = 三条对照**全部符合预期**（对照本身通过）；1 = 有对照不符合预期；2 = 没得比（真证据缺失）
#
# 用法（本目录下）：bash tools/negative_controls.sh [procs]

set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"
PROCS="${1:-5}"

if [ ! -f "$LOGS/declared_groups.txt" ] || ! ls "$LOGS"/run_*.log >/dev/null 2>&1; then
    echo "SKIPPED：真证据还没生成（缺 declared_groups.txt 或 run_*.log）——先跑 run_probes.sh"
    exit 2
fi

# 固定（不用 mktemp）：随机临时目录名会写进归档 log 的「命令」行 ⇒ 每次重跑都产生非空 diff，
# 容易被后人误读成"证据被改动"。活值不进断言（tower 规则）。
NEG="/tmp/m45_neg_controls"
rm -rf "$NEG"
mkdir -p "$NEG"
trap 'rm -rf "$NEG"' EXIT
ok=0
total=3

run_case() {   # $1=名字  $2=说明  $3=期望退出码  $4=准备函数名
    local name="$1" desc="$2" want="$3" prep="$4"
    local dir="$NEG/$name"
    mkdir -p "$dir/logs"
    cp "$LOGS/declared_groups.txt" "$dir/logs/" 2>/dev/null || true
    cp "$LOGS"/run_*.log "$dir/logs/" 2>/dev/null || true
    $prep "$dir"
    local out="$LOGS/negative_control_$name.log"
    {
        echo "# 负向对照：$desc"
        echo "date    : $(date -Is)"
        echo "命令    : python3 tools/summarize.py $dir $PROCS"
        echo "期望    : RESULT 非 OK 且退出码 $want"
        echo
    } > "$out"
    "$PY" tools/summarize.py "$dir" "$PROCS" >> "$out" 2>&1
    local rc=$?
    echo "退出码  : $rc（期望 $want）" >> "$out"
    echo "判定    : $([ "$rc" -eq "$want" ] && echo '对照通过' || echo '对照不符！')" >> "$out"
    if [ "$rc" -eq "$want" ]; then
        echo "  OK    $name：exit=$rc  (期望 $want)  $desc"
        ok=$((ok + 1))
    else
        echo "  BAD   $name：exit=$rc  (期望 $want)  $desc"
    fi
}

prep_empty() {   # ① 空：连 run_*.log 都不给
    rm -f "$1"/logs/run_*.log "$1"/logs/declared_groups.txt
}
prep_partial() { # ② 删掉两个**整组**变体（用 p[0-9] 精确锚定，避免 p* 把 rel1_brc_perchunk 也吃掉）
    rm -f "$1"/logs/run_probe_b_rel1_brc_p[0-9].log "$1"/logs/run_probe_b_rel0_brc_p[0-9].log
}
prep_fewer() {   # ③ 只留 1/5 个进程
    for p in 1 2 3 4; do rm -f "$1"/logs/run_probe_b_rel1_norm_p$p.log; done
}

run_case nocompare       "空目录（无 run_*.log、无 declared_groups.txt）⇒ 不得发合格证" 2 prep_empty
run_case partial_missing "删掉两个整组变体的日志 ⇒ 必须报缺组而不是 OK"                  1 prep_partial
run_case fewer_procs     "只留 1/5 个进程 ⇒ 必须报进程数不足而不是 OK"                  1 prep_fewer

echo
if [ "$ok" -eq "$total" ]; then
    echo "RESULT: OK（负向对照 $ok/$total 全部符合预期；归档 evidence/logs/negative_control_*.log）"
    exit 0
fi
echo "RESULT: DIFF（负向对照只有 $ok/$total 符合预期）"
exit 1
