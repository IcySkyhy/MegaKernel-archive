#!/usr/bin/env bash
# M149（m28_epilog_chain）一条命令复现：构建 → 设备跑（每档各自 flock）→ 三路对拍
#
#   bash m28_gdn_epilog/reproduce_chain.sh [输出目录]        # 缺省 /tmp/m28c_out
#   M28C_CASES=4,4097 M28C_MUT_CASES=4,4097 bash m28_gdn_epilog/reproduce_chain.sh
#
# 段：转位/落位（M148 段体，AIV）→ S5 RMSNormGated（m12 段体，AIV/VF）→ S6 out_proj（m11 段体，AIC/cube）
#     → hcAttnOut。判据三路：m12 check_ref（S5）、m11 check_ref（S6 精确域）、check_chain_ref.py
#     （链端到端，参考 = m21_layer_ref/ref/gdn.py）。
#
# 设备槽纪律（塔口径）：一次 flock 只跑一条短命令、`-w 300`、进锁先 `npu-smi`（快照落盘）、
# `timeout` 放在锁内；拿不到锁如实记「未取得读数」。锁文件 = /tmp/npu0.lock。
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-/tmp/m28c_out}
BUILD="$REPO/m28_gdn_epilog/build"
PY=${PY:-/usr/local/python3.12.13/bin/python3}
PYTORCH=${PYTORCH:-/workspace/venvs/baseline/bin/python3}
CASES=${M28C_CASES:-4,4097}
MUT_CASES=${M28C_MUT_CASES:-4,4097}
MUTS=${M28C_MUTS:-nogamma,zhead0}
LOCK=/tmp/npu0.lock

mkdir -p "$OUT" || { echo "[FAIL] 无法创建输出目录 $OUT"; exit 1; }

echo "== [1/5] 构建（独立工程；M148 转位段 target + M149 链段 target）=="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B "$BUILD" -S "$REPO/m28_gdn_epilog" -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1 \
    || { tail -20 "$OUT/cmake.log"; exit 1; }
cmake --build "$BUILD" -j4 > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
echo "   OK: $BUILD/m28_gdn_epilog（M148 转位段）+ $BUILD/m28_epilog_chain（M149 链段）"

# 一个 m 档 = 一次独立的 flock（拿锁 → npu-smi 快照 → timeout 内跑 → 立即释放）
run_locked() {  # $1=m  $2=输出目录  $3=mutant(none 传空)  $4=日志
    local m=$1 dir=$2 mut=$3 log=$4
    mkdir -p "$dir"
    flock -w 300 "$LOCK" bash -c "
        cd '$dir' || exit 3
        echo '[lock] acquired' >&2
        npu-smi info > 'npu_smi_m${m}.txt' 2>&1
        M28C_DUMP=1 M28C_OUT='$dir' M28C_MUT='$mut' timeout 280 '$BUILD/m28_epilog_chain' $m
    " > "$log" 2>&1
    local rc=$?
    echo "   m=$m mutant='${mut:-none}' rc=$rc :: $(grep -E '^\[M28C\] ' "$log" | tail -1)"
    return $rc
}

echo "== [2/5] 正向设备档（每档各自进一次锁）=="
POS="$OUT/pos"
pos_rc=0
for m in ${CASES//,/ }; do
    [ -z "$m" ] && continue
    run_locked "$m" "$POS" "" "$OUT/pos_m${m}.log" || pos_rc=1
done

echo "== [3/5] 负向对照设备档（S5 mutant；判据必须变红）=="
for mut in ${MUTS//,/ }; do
    for m in ${MUT_CASES//,/ }; do
        [ -z "$m" ] && continue
        run_locked "$m" "$OUT/mut_${mut}" "$mut" "$OUT/mut_${mut}_m${m}.log" || true  # 设备档预期 FAIL
    done
done

echo "== [4/5] 正向判据：链端到端（m21 参考）+ S5（m12 check_ref）+ S6（m11 check_ref）=="
"$PYTORCH" "$REPO/m28_gdn_epilog/check_chain_ref.py" "$POS" --mode pos > "$OUT/check_chain_pos.log" 2>&1
chain_pos_rc=$?
tail -4 "$OUT/check_chain_pos.log"

m12_rc=0
for m in ${CASES//,/ }; do
    [ -z "$m" ] && continue
    (cd "$POS" && "$PY" "$REPO/m12_rmsnorm_gated/check_ref.py" "$m" 0) > "$OUT/check_m12_m${m}.log" 2>&1 || m12_rc=1
    echo "   m12 m=$m :: $(tail -1 "$OUT/check_m12_m${m}.log")"
done

m11_rc=0
for m in ${CASES//,/ }; do
    [ -z "$m" ] && continue
    (cd "$POS/m${m}_s6exact" && "$PY" "$REPO/m11_bf16_gemm/check_ref.py" 6144 2560 "$m") > "$OUT/check_m11_m${m}.log" 2>&1 || m11_rc=1
    echo "   m11 m=$m :: $(tail -1 "$OUT/check_m11_m${m}.log")"
done

echo "== [5/5] 负向判据：链端到端必须变红；S5 的 m12 check_ref 亦须 FAIL =="
mut_rc=0
for mut in ${MUTS//,/ }; do
    [ -z "$mut" ] && continue
    "$PYTORCH" "$REPO/m28_gdn_epilog/check_chain_ref.py" "$OUT/mut_${mut}" --mode mut > "$OUT/check_chain_mut_${mut}.log" 2>&1
    cm_rc=$?
    tail -3 "$OUT/check_chain_mut_${mut}.log"
    [ "$cm_rc" -eq 0 ] || mut_rc=1
    for m in ${MUT_CASES//,/ }; do
        [ -z "$m" ] && continue
        # m12 check_ref 对 mutant 档应 FAIL（rc=1）；这里记录读数，不因它非 0 而判复现失败
        (cd "$OUT/mut_${mut}" && "$PY" "$REPO/m12_rmsnorm_gated/check_ref.py" "$m" 0) > "$OUT/check_m12_mut_${mut}_m${m}.log" 2>&1
        echo "   m12(mut=$mut) m=$m rc=$? :: $(grep -E 'NPU-dev|max rel' "$OUT/check_m12_mut_${mut}_m${m}.log" | tail -1)"
    done
done

echo "== 汇总：正向设备 rc=$pos_rc；链端到端 rc=$chain_pos_rc；S5(m12) rc=$m12_rc；S6(m11) rc=$m11_rc；负向 rc=$mut_rc =="
echo "   日志在 $OUT"
[ "$chain_pos_rc" -eq 0 ] && [ "$m12_rc" -eq 0 ] && [ "$m11_rc" -eq 0 ] && [ "$mut_rc" -eq 0 ] && exit 0 || exit 1
