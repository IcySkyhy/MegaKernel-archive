#!/usr/bin/env bash
# probe_v_align/run_probes.sh —— 一条命令重跑 M57 的全矩阵并归档原始证据
#
# 用法（本目录下）：bash run_probes.sh [op_filter]
#   op_filter  只跑名字匹配该正则的 op（默认 '.' = 全部）。
#              **过滤模式写到 `evidence_partial/`，不动 `evidence/` 归档** —— 因为"部分刷新"会让
#              归档不再是一次完整运行（`tools/check_archive_matches_list.py` 的"与当前代码同序"
#              与 §10 的单次运行声明都会失效）。要更新归档就跑不带参数的整轮。
#
# 产物（全部进 git，供 review 追溯）：
#   evidence/commands.txt         本次运行的确切命令/环境/版本
#   evidence/logs/build*.log      完整构建输出
#   evidence/logs/run_*.log       逐变体完整 stdout+stderr（含 aclError / FAULT_MSG 原文）
#   evidence/logs/matrix.txt      逐变体一行的机器可读矩阵（含组内提前停的 STOPPED 行）
#   evidence/logs/ildl_*.log      probe_v_ildl（Interleave/DeInterleave 语义）逐用例日志
#   evidence/dumps/*.bin          每条变体 1024B 的 OUT arena 原始 dump（逐字节证据）
#   evidence/check_ref*.log       两个独立复核脚本的输出（三态退出码）
#
# 两条设计决定：
#  1. **一个变体一个进程**：对齐违规是设备侧异常（507035），且故障/挂死可能污染上下文；
#     逐变体独立进程保证 (a) 故障可干净归因、(b) 后续变体不受污染、(c) 每条变体有独立 aclError。
#  2. **`ad:` 分组的自适应降序 + 首个故障即停**：对齐要求必是 32 的 2 幂因子（1/2/4/8/16/32），
#     故降序 {16,8,4,2,1} 的**第一个故障点**就给出最小对齐（16 故障 ⇒ 32B；16 过、8 故障 ⇒ 16B；…）。
#     这样把每条 op 的故障次数从 5 降到 1。主族 vlds/vsts（`full:` 分组）仍做完整扫描 {0,1,2,4,8,16,24,32}
#     作为该分类假设的**完整性证据**（不提前停）。

set -u
cd "$(dirname "$0")"

FILTER="${1:-.}"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
DUMPS="$EV/dumps"
BUILD="$ROOT/build"

if [ "$FILTER" = "." ]; then
    EV="$ROOT/evidence"
else
    # 过滤模式：另起目录，绝不碰归档
    EV="$ROOT/evidence_partial"
fi
LOGS="$EV/logs"
DUMPS="$EV/dumps"
mkdir -p "$LOGS" "$DUMPS"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
BISHENG="$(command -v bisheng || echo /usr/local/Ascend/cann-9.1.0/bin/bisheng)"
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

{
    echo "# probe_v_align（M57）证据生成记录"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "npu             : $(npu-smi info 2>/dev/null | sed -n '6p' | tr -s ' ')"
    echo "CANN/compiler   : $BISHENG"
    echo "bisheng version : $($BISHENG --version 2>&1 | head -2 | tr '\n' ' ')"
    echo "ASC arch        : dav-3510"
    echo "cmake           : $(cmake --version | head -1)"
    echo "op filter       : $FILTER"
    echo "concurrency     : 采集时设备上的其它进程（fleet 规则：读数需带并发背景）"
    npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | sed 's/^/                  /' 
    echo
    echo "## 复现命令（本目录执行）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8"
    echo "./build/probe_v_align list                     # 打印全部变体（含分组标签）"
    echo "bash run_probes.sh                             # 本文（全矩阵 + ildl + 两个复核）"
    echo "$PY check_ref.py .                             # 逐变体逐字节复核（三态退出码）"
    echo "$PY check_ref_ildl.py .                        # Interleave/DeInterleave lane 映射复核"
} > "$EV/commands.txt"

echo "== cmake 配置 =="
cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_configure.log" 2>&1 || {
    echo "cmake 配置失败，见 $LOGS/cmake_configure.log"; exit 1; }
echo "== 构建 =="
cmake --build "$BUILD" -j8 > "$LOGS/build.log" 2>&1 || {
    echo "构建失败，见 $LOGS/build.log"; tail -20 "$LOGS/build.log"; exit 1; }

if [ "$FILTER" = "." ]; then
    rm -f "$LOGS"/run_*.log "$LOGS"/ildl_*.log "$DUMPS"/*.bin
fi
: > "$LOGS/matrix.txt"
if [ "$FILTER" != "." ]; then
    echo "!! 过滤模式：本次输出写入 $EV（**不动 evidence/ 归档**）；这不是一次完整运行。"
fi

echo "== 逐变体运行（每变体一个进程、独立 aclError；每变体超时 180s）=="
n_total=0; n_ok=0; n_wrong=0; n_fault=0; n_hang=0; n_stop=0; n_noout=0
declare -A GROUP_FAILED
while read -r op a b n group; do
    [ -n "${op:-}" ] || continue
    echo "$op" | grep -qE "$FILTER" || continue
    tag="${op}_a${a}_b${b}_n${n}"
    log="$LOGS/run_${tag}.log"
    # 只有 `ad:` 分组才"首个故障即停"；`full:` 分组做完整扫描（分类假设的完整性证据）
    if [ "${GROUP_FAILED[$group]:-0}" = "1" ] && [ "${group#ad:}" != "$group" ]; then
        printf '%-8s op=%-14s a=%-3s b=%-3s n=%-3s rc=- aclError=- window=- outside=- fnv=- (%s)\n' \
            "STOPPED" "$op" "$a" "$b" "$n" "$group（组内已见故障，按降序策略不再下探）" >> "$LOGS/matrix.txt"
        n_stop=$((n_stop + 1))
        continue
    fi
    # 设备共享：其它 agent 的 kernel 会长期占用 AIV，探测变体可能长时间排队。
    # 因此超时给到 600s，并把"无输出"重试 3 次；重试仍无输出则记为 NO-OUTPUT（**不算故障**，
    # 也不参与对齐推断——绝不能把"排队超时"写成"对齐要求"）。
    outcome=""
    for attempt in 1 2 3; do
        timeout 600 "$BUILD/probe_v_align" run "$op" "$a" "$b" "$n" "$DUMPS" > "$log" 2>&1
        rc=$?
        outcome="$(grep -m1 '^OUTCOME:' "$log" | awk '{print $2}')"
        [ -n "$outcome" ] && [ "$outcome" != "NOSYNC" ] && break
        echo "  (attempt $attempt 无输出，重试：op=$op a=$a b=$b n=$n)" >&2
        sleep 5
    done
    [ -n "$outcome" ] || outcome="NO-OUTPUT"
    aclerr="$(grep -m1 '^LAUNCH aclError=' "$log" | sed 's/.*aclError=//; s/ .*//')"
    win="$(grep -m1 '^COMPARE window_bytes=' "$log" | sed 's/.*window_bytes=//; s/ .*//')"
    out="$(grep -m1 '^COMPARE window_bytes=' "$log" | sed 's/.*bytes_outside_window_written=//; s/ .*//')"
    fnv="$(grep -m1 '^DUMP_FNV ' "$log" | awk '{print $2}')"
    [ -n "${win:-}" ] || win="-"
    [ -n "${out:-}" ] || out="-"
    [ -n "${fnv:-}" ] || fnv="-"
    printf '%-8s op=%-14s a=%-3s b=%-3s n=%-3s rc=%s aclError=%-7s window=%-4s outside=%-4s fnv=%s (%s)\n' \
        "$outcome" "$op" "$a" "$b" "$n" "$rc" "${aclerr:-NA}" "$win" "$out" "$fnv" "$group" | tee -a "$LOGS/matrix.txt"
    n_total=$((n_total + 1))
    case "$outcome" in
        OK)    n_ok=$((n_ok + 1)) ;;
        WRONG) n_wrong=$((n_wrong + 1)) ;;
        HANG)  n_hang=$((n_hang + 1)); case "$group" in ad:*) GROUP_FAILED[$group]=1 ;; esac ;;
        FAULT) n_fault=$((n_fault + 1)); case "$group" in ad:*) GROUP_FAILED[$group]=1 ;; esac ;;
        NO-OUTPUT|NOSYNC) n_noout=$((n_noout + 1)) ;;
        *)     n_fault=$((n_fault + 1)) ;;
    esac
done < <("$BUILD/probe_v_align" list | grep -vE '^#')

{
    echo "# 变体计数（判定项；guard/报告项见 README §5）"
    echo "total=$n_total ok=$n_ok wrong=$n_wrong fault=$n_fault hang=$n_hang stopped=$n_stop no_output=$n_noout"
} >> "$LOGS/matrix.txt"
echo "== 汇总 =="
echo "total=$n_total ok=$n_ok wrong=$n_wrong fault=$n_fault hang=$n_hang stopped=$n_stop no_output=$n_noout"

if [ "$FILTER" = "." ]; then
echo "== probe_v_ildl（Interleave/DeInterleave 语义）=="
bash "$ROOT/run_ildl.sh"

echo "== 归档 vs 当前代码（行序/log 列/guard② 一致性守卫）=="
"$PY" "$ROOT/tools/check_archive_matches_list.py" "$ROOT" > "$EV/check_archive_matches_list.log" 2>&1
echo "check_archive_matches_list rc=$?  → $EV/check_archive_matches_list.log"

echo "== 独立复核（check_ref.py，三态退出码）=="
"$PY" "$ROOT/check_ref.py" "$ROOT" > "$EV/check_ref.log" 2>&1
echo "check_ref rc=$?  → $EV/check_ref.log"
else
    echo "== 过滤模式：跳过 ildl 与两个归档级复核（它们的前提是完整归档）=="
fi
