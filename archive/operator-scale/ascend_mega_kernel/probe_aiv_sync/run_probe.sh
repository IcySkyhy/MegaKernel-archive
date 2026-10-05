#!/usr/bin/env bash
# run_probe.sh —— M64 探针证据生成（一条命令重跑全部变体并归档）
#
# 用法（本目录下）：
#   bash run_probe.sh              # 默认每个变体 9 次独立进程（mission 要求 ≥9）
#   REPS=3 bash run_probe.sh       # 冒烟
#
# 产物（全部进 git，供 review 追溯；原始大快照 .bin 只在本目录临时留存、分析后删除）：
#   evidence/logs/<probe>_<variant>.log     每变体 N 次独立进程的完整输出（含 SUMMARY/ROUND/COREVIOL）
#   evidence/logs/<probe>_matrix.txt        逐变体摘要（逐次一行）
#   evidence/logs/visibility_matrix.txt     可见性探针摘要
#   evidence/logs/m53_reverse.log           M53 原探针逐字副本的 9 次读数
#   evidence/diag_<variant>.txt             设备侧快照逐词归因 + 时序交叉验证（窗口未闭合证明）
#                                            + 设备侧 ROUND 行（与 host 归因**双源交叉核对**，见 tools/crosscheck_diag.py）
#   evidence/commands.txt                   本次运行的确切命令、版本、源码 sha256
#
# 清场纪律：只用 `timeout` + 按**本 worktree 内的确切二进制路径** pkill；不碰别的目录/别的 agent。
set -u
cd "$(dirname "$0")"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
TMP="$ROOT/out_tmp"
REPS="${REPS:-9}"
VAR_FILTER="${VAR_FILTER:-}"      # 只跑匹配该正则的主探针变体（续跑用）
SKIP_OTHERS="${SKIP_OTHERS:-0}"    # 1 = 跳过 M53 反向验证与可见性探针（续跑用）
SKIP_DIAG="${SKIP_DIAG:-0}"
BIN="$ROOT/build/probe_aiv_sync"
VIS="$ROOT/build/probe_visibility"
M53="$ROOT/build/probe_m53_reverse"
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"
TO="${TO:-120}"          # 单个变体单次运行的超时（秒）；C0_halfset 预期死锁
mkdir -p "$LOGS" "$TMP"

kill_mine() {  # 只杀本 worktree 内、名字精确匹配的探针进程
    pkill -f "^${BIN} " 2>/dev/null || true
    pkill -f "^${VIS} " 2>/dev/null || true
    pkill -f "^${M53}" 2>/dev/null || true
}
trap 'kill_mine' EXIT

source /usr/local/Ascend/ascend-toolkit/set_env.sh
{
    echo "# probe_aiv_sync 证据生成记录（M64）"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "npu             : $(npu-smi info 2>/dev/null | sed -n '6p' | tr -s ' ')"
    echo "CANN/compiler   : $(command -v bisheng)"
    echo "bisheng version : $(bisheng --version 2>&1 | head -2 | tr '\n' ' ')"
    echo "ASC arch        : dav-3510  (cmake CMAKE_ASC_ARCHITECTURES)"
    echo "cmake           : $(cmake --version | head -1)"
    echo "reps per variant: $REPS"
    echo "per-run timeout : ${TO}s"
    echo
    echo "## 复现命令（本目录执行）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
    echo "bash run_probe.sh                    # 本文件（REPS 默认 9）"
    echo
    echo "## 单变体手工复现"
    echo "  ./build/probe_aiv_sync <variantIdx|name> <runNo> [outdir] [--snap] [--trace]"
    echo "  ./build/probe_aiv_sync                       # 打印全部变体清单"
    echo "  ./build/probe_visibility <variantIdx|name> <runNo>"
    echo "  ./build/probe_m53_reverse                    # M53 原探针（逐字副本）"
    echo
    echo "## 源码 sha256"
    sha256sum probe_aiv_sync.asc probe_visibility.asc probe_aic_aiv.asc probe_m53_reverse.asc \
        run_probe.sh run_aic_probe.sh tools/analyze_snap.py tools/make_tables.py CMakeLists.txt 2>/dev/null
} > "$EV/commands.txt"

echo "== 构建 =="
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release > "$TMP/cmake_configure.log" 2>&1 || {
    echo "cmake 配置失败，见 $TMP/cmake_configure.log"; exit 1; }
cmake --build build -j4 > "$TMP/build.log" 2>&1 || {
    echo "构建失败，见 $TMP/build.log"; tail -20 "$TMP/build.log"; exit 1; }
echo "   构建 OK"

if [ "$SKIP_OTHERS" = "1" ]; then
    echo "== 跳过 M53 反向验证 / 可见性探针（SKIP_OTHERS=1）=="
else
echo "== M53 原探针反向验证（逐字副本，$REPS 次独立进程）=="
: > "$LOGS/m53_reverse.log"
(mkdir -p "$TMP/m53" && cd "$TMP/m53" && for i in $(seq 1 "$REPS"); do
    echo "--- run $i ---"
    timeout "$TO" "$M53" "$TMP/m53" 2>&1
    echo "exit=$?"
 done) >> "$LOGS/m53_reverse.log" 2>&1
grep -c "probe_bar" "$LOGS/m53_reverse.log" >/dev/null 2>&1
tail -2 "$LOGS/m53_reverse.log"

echo "== 可见性探针（flag 与可见性拆开测，$REPS 次/变体）=="
: > "$LOGS/visibility_matrix.txt"
VIS_LIST=$("$VIS" 2>&1 | sed -n 's/^ *[0-9]*  //p')
for v in $VIS_LIST; do
    log="$LOGS/visibility_${v}.log"
    : > "$log"
    for i in $(seq 1 "$REPS"); do
        timeout "$TO" "$VIS" "$v" "$i" "$TMP" >> "$log" 2>&1
        rc=$?
        if [ $rc -eq 124 ]; then echo "SUMMARY variant=$v run=$i HANG=1 rc=124" >> "$log"; fi
    done
    n0=$(grep -cE " rc=0( |$)" "$log")
    printf '%-28s rc==0 %d/%d   viol_head_ok=%s\n' "$v" "$n0" "$REPS" \
        "$(grep -o 'host_head_ok=[0-9]*' "$log" | sort | uniq -c | tr '\n' ' ')" \
        | tee -a "$LOGS/visibility_matrix.txt"
done

fi
echo "== 主探针（配方 × 参数，$REPS 次/变体；A15/C0 预期死锁置末尾、单次 20s 超时）=="
[ -z "$VAR_FILTER" ] && : > "$LOGS/main_matrix.txt"
VAR_LIST=$("$BIN" 2>&1 | sed -n 's/^ *[0-9]*  \([A-Za-z0-9_]*\) .*/\1/p')
[ -n "$VAR_FILTER" ] && VAR_LIST=$(echo "$VAR_LIST" | grep -E "$VAR_FILTER")
NORMAL=$(echo "$VAR_LIST" | grep -v '^C0_halfset$')
LAST=$(echo "$VAR_LIST" | grep  '^C0_halfset$')
for v in $NORMAL $LAST; do
    log="$LOGS/aiv_sync_${v}.log"
    : > "$log"
    tmo="$TO"
    case "$v" in A15_m2_close|C0_halfset) tmo=20;; esac   # 预期死锁：短超时
    for i in $(seq 1 "$REPS"); do
        timeout "$tmo" "$BIN" "$v" "$i" "$TMP" >> "$log" 2>&1
        rc=$?
        if [ $rc -eq 124 ]; then
            echo "# $v run=$i 超时未返回（预期：只让一半核 set ⇒ 会合永不满 / mode2 自 set 自 wait ⇒ 计数器永不满）" >> "$log"
            echo "SUMMARY variant=$v run=$i HANG=1 rc=124" >> "$log"
        fi
    done
    case "$v" in A15_m2_close|C0_halfset) kill_mine; sleep 2;; esac
    n0=$(grep -cE " rc=0( |$)" "$log")
    hang=$(grep -c "rc=124" "$log")
    printf '%-26s rc==0 %2d/%d  hang=%d  viol_min..max=%s  rounds_clean=%s\n' "$v" "$n0" "$REPS" "$hang" \
        "$(grep -o 'viol=[0-9]*' "$log" | cut -d= -f2 | sort -n | sed -n '1p;$p' | tr '\n' ' ')" \
        "$(grep -o 'rounds_clean=[0-9]*' "$log" | cut -d= -f2 | sort -n | uniq -c | tr '\n' ' ')" \
        | tee -a "$LOGS/main_matrix.txt"
done
kill_mine

[ "$SKIP_DIAG" = "1" ] && { echo "== 跳过诊断（SKIP_DIAG=1）=="; exit 0; }
echo "== 诊断证据（设备侧快照逐词归因 + 时序交叉验证）=="
for v in A2_open_only A3_open_close B4_dense4_openonly A14_m1_close A20_sepwins; do
    timeout "$TO" "$BIN" "$v" 1 "$TMP" --snap --trace > "$TMP/diag_${v}.log" 2>&1
    {
        echo "############ diag $v（run 1，--snap --trace）"
        grep -m1 "^SUMMARY" "$TMP/diag_${v}.log"
        echo
        echo "---- 设备侧自读数（本次运行的 ROUND 行；与下面的 host 逐词归因互证）----"
        grep "^ROUND" "$TMP/diag_${v}.log"
        "$PY" tools/analyze_snap.py snap "$TMP/snap_${v}_run1.bin" 2>&1 | head -60
        "$PY" tools/analyze_snap.py trace "$TMP/diag_${v}.log" 2>&1 | head -25
        "$PY" tools/crosscheck_diag.py "$TMP/diag_${v}.log" "$TMP/snap_${v}_run1.bin" 2>&1 | head -20
    } > "$EV/diag_${v}.txt" 2>&1
    rm -f "$TMP/snap_${v}_run1.bin"        # 原始大快照不入库：归因结果已落盘
    echo "   $v -> evidence/diag_${v}.txt"
done

echo "== 汇总 =="
"$PY" tools/analyze_snap.py matrix "$LOGS" 2>&1 | tee "$EV/logs/matrix_summary.txt" | head -45
echo
echo "完成。证据目录：$EV（logs/ 逐变体日志 + matrix + diag_*）"
