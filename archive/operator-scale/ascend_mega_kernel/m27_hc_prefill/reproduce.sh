#!/usr/bin/env bash
# M119（Wave B5）hc prefill 段体的**一条命令复现**（构建 → 设备跑 → 判据）
#
#   bash m27_hc_prefill/reproduce.sh [输出目录]        # 缺省 /tmp/m27_out
#   M27_SYNTH=1 bash m27_hc_prefill/reproduce.sh       # 不读 checkpoint（合成权重）
#   M27_CASES=m1,m33 bash m27_hc_prefill/reproduce.sh  # 只跑部分档
#
# 约定：设备槽用 `flock -w 900 /tmp/npu0.lock`（进锁后再 npu-smi 复查；不并发）。
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-/tmp/m27_out}
BUILD="$REPO/m27_hc_prefill/build"
LOG="$OUT"

echo "== [1/4] 构建（独立工程，不碰 m15_layer_loop/CMakeLists.txt）=="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B "$BUILD" -S "$REPO/m27_hc_prefill" -DCMAKE_BUILD_TYPE=Release > "$LOG/cmake.log" 2>&1 || { tail -20 "$LOG/cmake.log"; exit 1; }
cmake --build "$BUILD" -j4 > "$LOG/build.log" 2>&1 || { tail -40 "$LOG/build.log"; exit 1; }
echo "   OK：$BUILD/m27_hc_prefill（MT=8）与 $BUILD/m27_hc_prefill_mt12（MT=12）"

echo "== [0/4] 判据链路自检（**不占设备**：合成 dump ⇒ 期望 rc=0；弄坏一档 ⇒ 期望 rc=1）=="
rm -rf /tmp/m27_self
/usr/local/python3.12.13/bin/python3 "$REPO/m27_hc_prefill/tools/selftest_dump.py" /tmp/m27_self m33 33 1 > /dev/null
/usr/local/python3.12.13/bin/python3 "$REPO/m27_hc_prefill/check_ref.py" /tmp/m27_self > "$LOG/selftest_check.log" 2>&1
echo "   正向 rc=$?（期望 0）:: $(grep -E 'RESULT' "$LOG/selftest_check.log" | tail -1)"
rm -rf /tmp/m27_self_mut
/usr/local/python3.12.13/bin/python3 "$REPO/m27_hc_prefill/tools/selftest_dump.py" /tmp/m27_self_mut m33 33 1 --mutate a1.blk > /dev/null
/usr/local/python3.12.13/bin/python3 "$REPO/m27_hc_prefill/check_ref.py" /tmp/m27_self_mut > "$LOG/selftest_mut_check.log" 2>&1
echo "   弄坏 a1.blk rc=$?（期望 1）:: $(grep -E '未过' "$LOG/selftest_mut_check.log" | tail -1)"

echo "== [2/4] 设备跑（flock -w 900 /tmp/npu0.lock；dump 落 $OUT）=="
mkdir -p "$OUT"
rm -f "$OUT"/m27_*      # dump / meta / 逐 case 元数据 / 布局 / 日志都以此前缀（清成空目录再跑）
df -h / | tail -1
cd "$REPO"
SYNTH_ARG=""
[ -n "${M27_SYNTH:-}" ] && SYNTH_ARG="M27_SYNTH=$M27_SYNTH"
CASES_ARG=""
[ -n "${M27_CASES:-}" ] && CASES_ARG="M27_CASES=$M27_CASES"
flock -w 900 /tmp/npu0.lock bash -c "cd '$OUT' && M27_DUMP=1 M27_MANIFEST='$REPO/m15_layer_loop/weights_manifest.txt' $SYNTH_ARG $CASES_ARG '$BUILD/m27_hc_prefill'" > "$OUT/dump.log" 2>&1
drc=$?
echo "   设备 rc=$drc（dump 出 $(ls "$OUT"/m27_case_*.txt 2>/dev/null | wc -l) 个 case 的元数据）"

echo "== [3/4] 判据（独立 numpy float64 参考 = m20_hyperconn/check_ref.py）=="
/usr/local/python3.12.13/bin/python3 "$REPO/m27_hc_prefill/check_ref.py" "$OUT" > "$OUT/check.log" 2>&1
rc=$?
tail -25 "$OUT/check.log"
echo "   判据 rc=$rc（0 = 比过且通过；1 = 比过有差异；2 = 没得比）"

echo "== [4/4] 负向对照（把被测对象弄坏必变红；期望 rc=1）=="
for mut in rowsteal norel; do
    mout="$OUT/mut_$mut"
    mkdir -p "$mout"
    rm -f "$mout"/m27_*
    flock -w 900 /tmp/npu0.lock bash -c "cd '$mout' && M27_DUMP=1 M27_MANIFEST='$REPO/m15_layer_loop/weights_manifest.txt' M27_MUTANT=$mut M27_CASES=${M27_MUT_CASES:-m33} $SYNTH_ARG '$BUILD/m27_hc_prefill'" > "$mout/dump.log" 2>&1
    /usr/local/python3.12.13/bin/python3 "$REPO/m27_hc_prefill/check_ref.py" "$mout" > "$mout/check.log" 2>&1
    mrc=$?
    echo "   mutant=$mut 判据 rc=$mrc（期望 1）"
    grep -E "RESULT|未过" "$mout/check.log" | tail -3
done

echo "== 汇总：设备 rc=$drc 判据 rc=$rc；日志在 $OUT/{dump,check}.log =="
exit $rc
