#!/usr/bin/env bash
# M163：GDN chunk 扫描**设备侧数值稳定化**的设备档复算脚本（可复算、能传播失败）。
#
# 纪律：每档各自进一次 flock（-w 300）、进锁先 `npu-smi`（快照落盘）、`timeout` 在锁内、
#        一次进锁一条短命令。等锁没拿到 ⇒ 该档记「未取得读数」并以非零退出（不写成"未复现"）。
#        所有离线对拍（NaN-aware 判据）在锁外做。
#
# 六个档 + 期望（rc 不符即 FAIL，脚本以非零退出传播）：
#   base  m=1    —— 修前（BASE 提交的 header，exp(ĝ)·exp(−ĝ)）⇒ 有限 ⇒ 判据 PASS
#   base  m=4097 —— 修前 ⇒ o/ht 非有限 ⇒ 判据**必须 FAIL**（判据不瞎的证据）
#   fix   m=1    —— 修后 ⇒ 有限 ⇒ 判据 PASS
#   fix   m=4097 —— 修后 ⇒ 0 非有限 ⇒ 判据 PASS
#   nc1   m=4097 —— 负向对照：`-DM15GP_STAB_NAIVE=1` 退回 Γ 旧乘法 ⇒ 判据**必须 FAIL**
#   nc2   m=4097 —— 负向对照：`-DM15GP_STAB_BADIDX=1` 指数差下标错一位 ⇒ 判据**必须 FAIL**
#
# 用法：bash m15_layer_loop/evidence/m163_stab/reproduce.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
MAN="$ROOT/m15_layer_loop/weights_manifest.txt"
LOCK=/tmp/npu0.lock
PY=/usr/local/python3.12.13/bin/python3
BASE_SHA=0cd7fe57d85310d079eccf7354dc4fc1e2970087   # 本分支 base（不可变 rev）
TMP="${M163_TMP:-/tmp/m163_stab}"
HDR_REL=m15_layer_loop/m15_gdn_prefill.h
mkdir -p "$HERE/logs"

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
rc_all=0

# run_dump <tag> <binary> <m> <timeout_s>：进一次锁、跑一档、dump 进 $HERE/dumps_<tag>
run_dump() {
  local tag="$1" bin="$2" m="$3" to="$4"
  local dir="$HERE/dumps_$tag" log="$HERE/logs/$tag.log"
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[m163] 设备档 $tag（m=$m）→ $log"
  flock -w 300 "$LOCK" bash -c "
    { echo '=== lock acquired '\$(date -Is)' tag=$tag m=$m ===';
      echo '--- npu-smi (lock entry) ---'; npu-smi info;
      timeout $to env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
        M15_PREFILL_PHASES=3 M15_PREFILL_PROLOG=1 M15_PREFILL_M=$m M15_PREFILL_DUMPDIR=$dir \
        '$bin' '$MAN' prefill;
      drc=\$?;
      echo \"exit=\$drc\";
      exit \$drc;
    } > '$log' 2>&1
  "
  local frc=$?
  if [ $frc -ne 0 ]; then
    if grep -q "lock acquired" "$log"; then
      echo "[m163] 设备档 $tag 退出码 $frc（见 $log）"
    else
      echo "[m163] 未取得读数：锁未拿到（$tag）"
    fi
    rc_all=1
    return 3
  fi
  grep -q "lock acquired" "$log" || { echo "[m163] 未取得读数（$tag：无锁内快照）"; rc_all=1; return 3; }
  return 0
}

# expect_judge <tag> <expect_rc>：对 dumps_<tag> 跑 NaN-aware 判据，rc 必须等于 expect_rc
expect_judge() {
  local tag="$1" exp="$2" r
  PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/check_stab_nan.py" "$HERE/dumps_$tag" \
    > "$HERE/logs/judge_$tag.log" 2>&1
  r=$?
  if [ "$r" = "$exp" ]; then
    echo "[m163] 判据 $tag：rc=$r（期望 $exp）✓"
  else
    echo "[m163] 判据 $tag：rc=$r（期望 $exp）✗ —— 见 logs/judge_$tag.log"; rc_all=1
  fi
}

# expect_script_rc <tag> <expect_rc> <script> [args...]：跑外部脚本，rc 必须等于 expect_rc
expect_script_rc() {
  local tag="$1" exp="$2"; shift 2
  PYTHONDONTWRITEBYTECODE=1 "$PY" "$@" > "$HERE/logs/$tag.log" 2>&1
  local r=$?
  if [ "$r" = "$exp" ]; then
    echo "[m163] $tag：rc=$r（期望 $exp）✓"
  else
    echo "[m163] $tag：rc=$r（期望 $exp）✗ —— 见 logs/$tag.log"; rc_all=1
  fi
}

# --- 构建修后（当前工作树）---
echo "[m163] 构建 fix（工作树 header）"
cmake -B "$ROOT/m15_layer_loop/build" -S "$ROOT/m15_layer_loop" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build "$ROOT/m15_layer_loop/build" -j16 --target m15_layer_loop > "$HERE/logs/build_fix.log" 2>&1 || \
  { echo "[m163] fix 构建失败"; rc_all=1; }

# --- 构建修前（BASE 提交的源码树，不动工作树）---
echo "[m163] 构建 base（git archive $BASE_SHA，不改工作树）"
rm -rf "$TMP/src_before" "$TMP/build_before"; mkdir -p "$TMP/src_before"
git -C "$ROOT" archive "$BASE_SHA" m15_layer_loop | tar -x -C "$TMP/src_before" || rc_all=1
cmake -B "$TMP/build_before" -S "$TMP/src_before/m15_layer_loop" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build "$TMP/build_before" -j16 --target m15_layer_loop > "$HERE/logs/build_base.log" 2>&1 || \
  { echo "[m163] base 构建失败"; rc_all=1; }

# --- 构建两个负向对照（编译期宏，工作树源码）---
for nc in nc1 nc2; do
  case "$nc" in
    nc1) D="-DM15GP_STAB_NAIVE=1";;
    nc2) D="-DM15GP_STAB_BADIDX=1";;
  esac
  echo "[m163] 构建 $nc（$D）"
  rm -rf "$TMP/build_$nc"
  ASCFLAGS="$D" cmake -B "$TMP/build_$nc" -S "$ROOT/m15_layer_loop" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build "$TMP/build_$nc" -j16 --target m15_layer_loop > "$HERE/logs/build_$nc.log" 2>&1 || \
    { echo "[m163] $nc 构建失败"; rc_all=1; }
done

# --- 设备档 + 判据 ---
run_dump base_m1      "$TMP/build_before/m15_layer_loop" 1     240; [ $? = 0 ] || rc_all=1
run_dump base_m4097   "$TMP/build_before/m15_layer_loop" 4097  290; [ $? = 0 ] || rc_all=1
run_dump fix_m1       "$ROOT/m15_layer_loop/build/m15_layer_loop" 1    240; [ $? = 0 ] || rc_all=1
run_dump fix_m4097    "$ROOT/m15_layer_loop/build/m15_layer_loop" 4097 290; [ $? = 0 ] || rc_all=1
run_dump nc1_m4097    "$TMP/build_nc1/m15_layer_loop" 4097 290; [ $? = 0 ] || rc_all=1
run_dump nc2_m4097    "$TMP/build_nc2/m15_layer_loop" 4097 290; [ $? = 0 ] || rc_all=1

echo "[m163] ---- NaN-aware 判据（参考 = check_stab_nan.py 的稳定化 fp64）----"
expect_judge base_m1    0
expect_judge base_m4097 1
expect_judge fix_m1     0
expect_judge fix_m4097  0
expect_judge nc1_m4097  1
expect_judge nc2_m4097  1

# --- 交叉：m23 既有判据（naive、NaN 盲）在 base 与 fix 上都应 PASS（检测缺口的正面证据）---
for t in base_m4097 fix_m4097; do
  PYTHONDONTWRITEBYTECODE=1 "$PY" "$ROOT/m23_gdn_prefill/check_ref.py" --dir "$HERE/dumps_$t" --allow-stale \
    > "$HERE/logs/m23judge_$t.log" 2>&1
  echo "[m163] m23 判据 $t：rc=$?（NaN 盲；见 logs/m23judge_$t.log）"
done

# --- 交叉：M151 订正后的 NaN 脚本（默认指数差参考；旧式保留为 --naive-ref 对照）---
#   fix 应 PASS(0)；base 应 FAIL(1)；fix 上加 --naive-ref 应 FAIL(1)（参考自洽守卫咬住旧式参考）。
M151NAN="$ROOT/m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_phaseA_nan.py"
expect_script_rc m151nan_fix_m1       0 "$M151NAN" "$HERE/dumps_fix_m1"
expect_script_rc m151nan_fix_m4097    0 "$M151NAN" "$HERE/dumps_fix_m4097"
expect_script_rc m151nan_base_m4097   1 "$M151NAN" "$HERE/dumps_base_m4097"
expect_script_rc m151nan_fix_naiveref 1 "$M151NAN" "$HERE/dumps_fix_m4097" --naive-ref

# --- m=1 修前/修后非逐位对照（max 标量逐位相等、张量有差异但量级不变；F1 要求写进证据）---
expect_script_rc m1_bitdiff 0 "$HERE/check_m1_bitdiff.py" "$HERE/dumps_base_m1" "$HERE/dumps_fix_m1"

echo "[m163] ==== reproduce 汇总：$([ $rc_all -eq 0 ] && echo ALL EXPECTATIONS MET || echo FAILURES PRESENT) ===="
exit $rc_all
