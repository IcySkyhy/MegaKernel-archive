#!/usr/bin/env bash
# M148（m28_gdn_epilog）一条命令复现：构建 → 设备跑（flock 队）→ 判据 → 负向对照
#
#   bash m28_gdn_epilog/reproduce.sh [输出目录]        # 缺省 /tmp/m28_out
#   M28_CASES=4,4097 bash m28_gdn_epilog/reproduce.sh  # 只跑部分正向档
#
# 设备槽纪律（塔口径）：一次 flock 只跑一条短命令、`-w 300`、进锁先 `npu-smi`（快照落盘）、
# `timeout` 放在锁内；拿不到锁如实记「未取得读数」。锁文件 = /tmp/npu0.lock。
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-/tmp/m28_out}
BUILD="$REPO/m28_gdn_epilog/build"
PY=${PY:-/usr/local/python3.12.13/bin/python3}
CASES=${M28_CASES:-4,4097}
MUT_CASES=${M28_MUT_CASES:-4,4097}

mkdir -p "$OUT" || { echo "[FAIL] 无法创建输出目录 $OUT"; exit 1; }

echo "== [1/4] 构建（独立工程，不碰仓库顶层 CMakeLists.txt）=="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B "$BUILD" -S "$REPO/m28_gdn_epilog" -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1 \
    || { tail -20 "$OUT/cmake.log"; exit 1; }
cmake --build "$BUILD" -j4 > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
echo "   OK：$BUILD/m28_gdn_epilog"

# 一个 m 档 = 一次独立的 flock（拿锁 → npu-smi 快照 → timeout 内跑 → 立即释放）
run_locked() {
    local m=$1 tag=$2 mutant=$3
    local locklog="$OUT/${tag}_m${m}.log"
    flock -w 300 /tmp/npu0.lock bash -c "
        cd '$OUT' || exit 3
        echo '[lock] acquired' >&2
        npu-smi info > '${tag}_npu_smi_m${m}.txt' 2>&1
        M28_DUMP=1 M28_MUTANT='$mutant' timeout 180 '$BUILD/m28_gdn_epilog' $m
    " > "$locklog" 2>&1
    local rc=$?
    echo "   m=$m tag=$tag mutant='${mutant:-none}' rc=$rc :: $(grep -E '^\[M28\] ' "$locklog" | tail -1)"
    echo "      log=$locklog  smi=${tag}_npu_smi_m${m}.txt"
    return $rc
}

echo "== [2/4] 正向设备档（每档各自进一次锁）=="
pos_rc=0
for m in ${CASES//,/ }; do
    [ -z "$m" ] && continue
    run_locked "$m" "m28" "" || pos_rc=1
done

echo "== [3/4] 正向判据（numpy 参考，位级）=="
"$PY" "$REPO/m28_gdn_epilog/check_ref.py" "$OUT" --mode pos > "$OUT/check_pos.log" 2>&1
chk_rc=$?
tail -8 "$OUT/check_pos.log"
echo "   正向判据 rc=$chk_rc（0=比过且绿；1=有差异；2=没得比）"

echo "== [4/4] 负向对照（M28_MUTANT=headstride，判据必须变红）=="
mut_rc=0
for m in ${MUT_CASES//,/ }; do
    [ -z "$m" ] && continue
    run_locked "$m" "mut_m28" "headstride" || true   # 设备档本身预期 FAIL，rc 非 0 属正常
done
"$PY" "$REPO/m28_gdn_epilog/check_ref.py" "$OUT" --mode mut > "$OUT/check_mut.log" 2>&1
mutchk_rc=$?
tail -8 "$OUT/check_mut.log"
echo "   负向判据 rc=$mutchk_rc（0=确认已变红；1=没变红）"

echo "== 汇总：设备档 rc=$pos_rc 正向判据 rc=$chk_rc 负向判据 rc=$mutchk_rc；日志在 $OUT =="
[ "$chk_rc" -eq 0 ] && [ "$mutchk_rc" -eq 0 ] && exit 0 || exit 1
