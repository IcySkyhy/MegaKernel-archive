#!/usr/bin/env bash
# M164 reproduce: prefill epilog chain (transpose -> S5 -> S6 -> hcAttnOut) mounted into the
# prefill four-phase path (host-orchestrated split: A | epilog | H2|B).
#
# Discipline: one flock per arm (-w <= 300), npu-smi snapshot written into that arm's log *inside*
# the lock, `timeout` inside the lock, one short command per lock acquisition. Lock not acquired =>
# that arm is recorded as "未取得读数" and this script exits non-zero (NOT reported as "not
# reproduced"). All offline comparisons run outside the lock.
#
# Usage:
#   bash reproduce.sh                 # full device matrix + offline checks
#   bash reproduce.sh --offline-only  # offline checks on existing dumps_*/ (no device)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BIN="$ROOT/m15_layer_loop/build/m15_layer_loop"
MAN="$ROOT/m15_layer_loop/weights_manifest.txt"
LOCK=/tmp/npu0.lock
PY=/usr/local/python3.12.13/bin/python3
mkdir -p "$HERE/logs"
rc_all=0

# run_one <name> <epilog> <epilog_mut> <prolog> <m-list> <timeout>
#   prolog=1 -> epilog z-source (pfQkvzba) has a producer; M15_PREFILL_PHASES=7 = H1|A|H2
#   (avoids phase B's known m=1 red item; H2 is needed for criterion (c)).
run_one() {
  local name="$1" epilog="$2" emut="$3" prolog="$4" mlist="$5" to="$6"
  local dir="$HERE/dumps_$name" log="$HERE/logs/run_$name.log"
  rm -rf "$dir"; mkdir -p "$dir"
  if flock -w 300 "$LOCK" bash -c "
      { echo '=== lock acquired '\$(date -Is)' name=$name epilog=$epilog emut=$emut prolog=$prolog m=$mlist ===';
        echo '--- npu-smi (lock entry) ---'; npu-smi info;
        cd '$ROOT' && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null;
        timeout $to env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
          M15_PREFILL_PHASES=7 M15_PREFILL_PROLOG=$prolog M15_PREFILL_EPILOG=$epilog \
          M15_PREFILL_EPILOG_MUT=$emut M15_PREFILL_M=$mlist M15_PREFILL_DUMPDIR=$dir \
          '$BIN' '$MAN' prefill;
        echo \"exit=\$?\";
      } > '$log' 2>&1
    "; then
    :
  else
    echo "[m164] 未取得读数：锁未拿到（$name）"; rc_all=1; return 1
  fi
  if grep -q "ALL PASS" "$log"; then
    echo "[m164] $name: ALL PASS"
  else
    echo "[m164] $name: FAILURES PRESENT（见 $log）"; rc_all=1
  fi
}

offline() {
  echo "[m164] ---- offline：lift 见证 ----"
  PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/verify_lift.py" || rc_all=1

  echo "[m164] ---- offline：epilog 链 numpy 对拍（T1/T3 + 出处 + 负向期望）----"
  ( cd "$HERE" && PYTHONDONTWRITEBYTECODE=1 "$PY" check_epilog.py \
      dumps_m1_clean dumps_m4_clean dumps_m4097_clean dumps_m1_mut1 dumps_m4_mut1 \
      dumps_m1_mut2 dumps_m1_mut4 ) || rc_all=1

  echo "[m164] ---- offline：判据 (c) EPILOG=0/1 的 H2 输出必须不同 ----"
  if [ -f "$HERE/dumps_m1_clean/m140_Pf.gdn_h2_blk.bin" ] && [ -f "$HERE/dumps_m1_epilog0/m140_Pf.gdn_h2_blk.bin" ]; then
    A=$(sha256sum "$HERE/dumps_m1_clean/m140_Pf.gdn_h2_blk.bin" | cut -c1-16)
    B=$(sha256sum "$HERE/dumps_m1_epilog0/m140_Pf.gdn_h2_blk.bin" | cut -c1-16)
    echo "[m164]   epilog=1 h2_blk $A ; epilog=0 h2_blk $B"
    if [ "$A" = "$B" ]; then
      echo "[m164]   H2 输出两档相同 ⇒ epilog 未改变 H2 的 bo 输入 ⇒ FAIL"; rc_all=1
    else
      echo "[m164]   DIFFER ✓（H2 确实吃到了新的 bo）"
    fi
  else
    echo "[m164]   缺 h2_blk dump（dumps_m1_clean / dumps_m1_epilog0）⇒ 未取得读数"; rc_all=1
  fi

  echo "[m164] ---- offline：判据 (d) 相位 A 的 m23 对拍仍 PASS ----"
  for d in dumps_m1_clean dumps_m4097_clean; do
    PYTHONDONTWRITEBYTECODE=1 "$PY" "$ROOT/m23_gdn_prefill/check_ref.py" --dir "$HERE/$d" --allow-stale || rc_all=1
  done

  echo "[m164] ---- offline：回归档（PROLOG=0 EPILOG=0，M136 基线）----"
  if grep -q "ALL PASS" "$HERE/logs/run_m1_hostsynth.log" 2>/dev/null; then
    echo "[m164]   hostsynth(prolog=0,epilog=0): ALL PASS ✓"
  else
    echo "[m164]   hostsynth 档未取得读数或非 ALL PASS"; rc_all=1
  fi
}

if [ "${1:-}" = "--offline-only" ]; then
  offline
else
  run_one m1_clean      1 0 1 1    240
  run_one m4_clean      1 0 1 4    240
  run_one m4097_clean   1 0 1 4097 300
  run_one m1_epilog0    0 0 1 1    240
  run_one m1_mut1       1 1 1 1    240
  run_one m4_mut1       1 1 1 4    240
  run_one m1_mut2       1 2 1 1    240
  run_one m1_mut4       1 4 1 1    240
  run_one m1_hostsynth  0 0 0 1    240
  offline
fi
echo "[m164] ==== reproduce 汇总：$([ $rc_all -eq 0 ] && echo ALL PASS || echo FAILURES PRESENT) ===="
exit $rc_all
