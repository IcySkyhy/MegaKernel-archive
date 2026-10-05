#!/usr/bin/env bash
# M151：预填充 prolog 链路（S2 in_proj + S3 conv1d/l2norm/gating）的设备档复算脚本。
#
# 纪律：**每档各自进一次 flock**（-w ≤300）、进锁先 `npu-smi` 并把快照写进该档日志、
# `timeout` 在锁内、一次进锁一条短命令。等锁没拿到 ⇒ 该档记「未取得读数」并以非零退出
# （不写成"未复现"）。所有离线对拍在锁外做。
#
# 用法：
#   bash m15_layer_loop/evidence/m151_prefill_prolog_wiring/reproduce.sh          # 全量
#   bash .../reproduce.sh --offline-only                                          # 只跑离线对拍（用已有 dump）
#
# 相位 A 参考判据的期望口径（M167 订正；详见 README §5）：M151 归档 dump 是 **M163 前**的 naive 核产物。
#   · m=1    ：设备有限、修正后的指数差参考也有限 ⇒ 期望 rc=0；
#   · m=4097 ：设备 o 真有 8,374,400 个非有限（旧式 exp(ĝ)·exp(−ĝ)=0·inf），修正后的指数差参考
#              **正确地判红** ⇒ 期望 rc=1（**预期红**，不是判据回退、不是新回归）。
#   `--offline-only` 按归档核对（m=4097 期望红）；全量（M163 后二进制重生成）按 m=4097 期望绿核对。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BIN="$ROOT/m15_layer_loop/build/m15_layer_loop"
MAN="$ROOT/m15_layer_loop/weights_manifest.txt"
LOCK=/tmp/npu0.lock
PY=/usr/local/python3.12.13/bin/python3
mkdir -p "$HERE/logs"
rc_all=0

run_one() {  # $1=prolog $2=m $3=name $4=timeout
  local prolog="$1" m="$2" name="$3" to="$4"
  local dir="$HERE/dumps_$name" log="$HERE/logs/run_$name.log"
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[m151] 设备档 $name（prolog=$prolog m=$m）→ $log"
  flock -w 300 "$LOCK" bash -c "
    { echo '=== lock acquired '\$(date -Is)' prolog=$prolog m=$m ===';
      echo '--- npu-smi (lock entry) ---'; npu-smi info;
      timeout $to env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
        M15_PREFILL_PHASES=3 M15_PREFILL_PROLOG=$prolog M15_PREFILL_M=$m M15_PREFILL_DUMPDIR=$dir \
        '$BIN' '$MAN' prefill;
      echo \"exit=\$?\";
    } > '$log' 2>&1
  " || { echo "[m151] 未取得读数：锁未拿到（$name）"; rc_all=1; return 1; }
  if ! grep -q "ALL PASS" "$log"; then echo "[m151] $name 设备判据未全过（见 $log）"; rc_all=1; fi
}

# expect_rc <期望rc> <标签> <命令...>：rc 与期望分开核对，不符才置 rc_all=1。
# 判据自身的输出照常打印（不吞、不静默），期望写在调用处 —— 避免"红被当成绿、绿被当成红"。
expect_rc() {
  local exp="$1" label="$2"; shift 2
  PYTHONDONTWRITEBYTECODE=1 "$@"; local r=$?
  if [ "$r" = "$exp" ]; then echo "[m151] $label：rc=$r（期望 $exp）✓"
  else echo "[m151] $label：rc=$r（期望 $exp）✗"; rc_all=1; fi
}

offline() {  # $1 = m=4097 两个参考判据的期望 rc：归档 dump（M163 前 naive 核）⇒ 1；M163 后重生成 ⇒ 0
  local exp_m4097="${1:-1}"
  echo "[m151] ---- 离线对拍（m=4097 相位 A 参考判据期望 rc=$exp_m4097；口径见 README §5）----"
  ( cd "$HERE" && PYTHONDONTWRITEBYTECODE=1 "$PY" check_prolog.py dumps_m1_clean dumps_m4097_clean ) || rc_all=1
  bash "$HERE/verify_lift.sh" || rc_all=1
  # 相位 A 参考判据（逐档各自期望）：
  #   m=1    ：设备有限 + 修正后的指数差参考有限 ⇒ 期望 rc=0；
  #   m=4097 ：归档是 naive 核产物，指数差参考**正确地**判红 ⇒ 期望 rc=$exp_m4097（预期红）。
  # 两条判据都跑：check_phaseA_nan.py（NaN 掩码感知，M162 订正）、m23 check_ref.py（有限子集 T3 + 非有限分类，M157/M165）。
  expect_rc 0 "check_phaseA_nan.py dumps_m1_clean" \
    "$PY" "$HERE/check_phaseA_nan.py" "$HERE/dumps_m1_clean"
  expect_rc "$exp_m4097" "check_phaseA_nan.py dumps_m4097_clean" \
    "$PY" "$HERE/check_phaseA_nan.py" "$HERE/dumps_m4097_clean"
  expect_rc 0 "m23 check_ref.py dumps_m1_clean" \
    "$PY" "$ROOT/m23_gdn_prefill/check_ref.py" --dir "$HERE/dumps_m1_clean" --allow-stale
  expect_rc "$exp_m4097" "m23 check_ref.py dumps_m4097_clean" \
    "$PY" "$ROOT/m23_gdn_prefill/check_ref.py" --dir "$HERE/dumps_m4097_clean" --allow-stale
  # 见证：相位 A 的输入确实来自设备（prolog 档 vs 宿主合成档的 m23 q/k/v/g/β 逐面 DIFFER）
  echo "[m151] ---- 相位 A 输入来源见证（device vs host-synth）----"
  for t in q k v g beta; do
    a=$(sha256sum "$HERE/dumps_m1_clean/m23_Pf.gdn_$t.bin" | cut -c1-16)
    b=$(sha256sum "$HERE/dumps_m1_hostsynth/m23_Pf.gdn_$t.bin" | cut -c1-16)
    if [ "$a" = "$b" ]; then echo "  $t: SAME（**异常**：prolog 档应与宿主合成不同）"; rc_all=1;
    else echo "  $t: device=$a hostsynth=$b DIFFER ✓"; fi
  done
}

if [ "${1:-}" = "--offline-only" ]; then
  offline 1     # 已有 dump = M151 归档（M163 前 naive 核）⇒ m=4097 期望红
else
  run_one 1 1     m1_clean      240
  run_one 1 4097  m4097_clean   300
  run_one 0 1     m1_hostsynth  240
  offline 0     # 刚由当前（M163 后）二进制重生成 ⇒ m=4097 期望绿
fi

echo "[m151] ==== reproduce 汇总：$([ $rc_all -eq 0 ] && echo ALL PASS || echo FAILURES PRESENT) ===="
exit $rc_all
