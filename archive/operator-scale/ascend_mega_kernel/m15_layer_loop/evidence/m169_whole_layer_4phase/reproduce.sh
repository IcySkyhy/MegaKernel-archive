#!/usr/bin/env bash
# M169：整层预填充 —— **四相位同时生效** + 两条段间链路（prolog + epilog）的新基线设备档 +
# 离线对拍。判据全部复用现成参考（不自己造第二套数学）。
#
# 档矩阵（公共 env：M15_LAYERS=1、真 checkpoint 权重、M15_SKIP_WCHECK=1、M15_PREFILL_WIRE=1、
# M15_PREFILL_KIND=1 —— KIND_ATTN 在 wire=1 下响亮失败，见任务 1 的盘点）：
#   m1_p15_links      m=1    PHASES=0xF(全四相位) PROLOG=1 EPILOG=1   ← **新基线（两条链路都开）**
#   m4097_p15_links   m=4097 同上
#   m1_p15_nolinks    m=1    PHASES=0xF             PROLOG=0 EPILOG=0   ← 四相位回归（旧基线，两条链路关）
#   m4097_p15_nolinks m=4097 同上
#   m1_p12_iso        m=1    PHASES=0xC(H2|B)       PROLOG=0 EPILOG=0   ← 既有红项 m1_p12 的隔离档
#
# 纪律（塔口径）：每档各自进一次 flock -w 300 /tmp/npu0.lock、进锁先 npu-smi（快照写进该档日志）、
#   timeout 在锁内、一次进锁一条短命令；等锁没拿到 ⇒ 该档记「未取得读数」并以非零退出（不写成
#   "未复现"）。所有离线对拍在锁外做。
#
# 用法：
#   bash reproduce.sh                 # 全矩阵设备档 + 锁外离线对拍
#   bash reproduce.sh --offline-only  # 只跑离线对拍（用已有 dumps_*/；需先跑过设备档）
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BIN="$ROOT/m15_layer_loop/build/m15_layer_loop"
MAN="$ROOT/m15_layer_loop/weights_manifest.txt"
LOCK=/tmp/npu0.lock
PY=/usr/local/python3.12.13/bin/python3
M140="$ROOT/m15_layer_loop/evidence/m140_prefill_full_layer"
M151="$ROOT/m15_layer_loop/evidence/m151_prefill_prolog_wiring"
M164="$ROOT/m15_layer_loop/evidence/m164_epilog_mount"
mkdir -p "$HERE/logs"
rc_all=0

# run_one <name> <mask> <m> <prolog> <epilog> <timeout>
run_one() {
  local name="$1" mask="$2" m="$3" prolog="$4" epilog="$5" to="$6"
  local dir="$HERE/dumps_$name" log="$HERE/logs/run_$name.log"
  rm -rf "$dir"; mkdir -p "$dir"
  echo "[m169] 设备档 $name（mask=0x$(printf '%X' "$mask") m=$m prolog=$prolog epilog=$epilog）→ $log"
  flock -w 300 "$LOCK" bash -c "
    { echo '=== lock acquired '\$(date -Is)' name=$name mask=$mask m=$m prolog=$prolog epilog=$epilog ===';
      echo '--- npu-smi (lock entry) ---'; npu-smi info;
      echo '--- command (verbatim, run from '$ROOT') ---';
      echo 'M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_PHASES=$mask M15_PREFILL_PROLOG=$prolog M15_PREFILL_EPILOG=$epilog M15_PREFILL_M=$m M15_PREFILL_DUMPDIR=$dir $BIN $MAN prefill';
      cd '$ROOT' && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null;
      timeout $to env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
        M15_PREFILL_PHASES=$mask M15_PREFILL_PROLOG=$prolog M15_PREFILL_EPILOG=$epilog \
        M15_PREFILL_M=$m M15_PREFILL_DUMPDIR=$dir '$BIN' '$MAN' prefill;
      echo \"exit=\$?\";
    } > '$log' 2>&1
  " || { echo "[m169] 未取得读数：锁未拿到（$name）"; rc_all=1; return 1; }
  if grep -q "lock acquired" "$log"; then :; else echo "[m169] 未取得读数（$name：无锁内快照）"; rc_all=1; return 1; fi
}

# kernel_expect <name> <regex>：该档日志末尾的汇总行必须匹配 regex，否则 FAIL
kernel_expect() {
  local name="$1" want="$2"
  local got
  got="$(grep -oE 'ALL PASS（checks=[0-9]+, guards=[0-9]+, fails=[0-9]+）|FAILURES PRESENT（checks=[0-9]+, guards=[0-9]+, fails=[0-9]+）' "$HERE/logs/run_$name.log" | tail -1)"
  if echo "$got" | grep -qE "$want"; then
    echo "[m169] $name 进程内判据：$got （期望 /$want/）✓"
  else
    echo "[m169] $name 进程内判据：'$got' 不匹配 /$want/ ✗（见 logs/run_$name.log）"; rc_all=1
  fi
}

# judge <desc> <expect_rc> <cmd...>：离线判据，rc 必须等于 expect_rc
judge() {
  local desc="$1" exp="$2"; shift 2
  local slog="$HERE/logs/judge_$(echo "$desc" | tr ' /' '__').log"
  PYTHONDONTWRITEBYTECODE=1 "$@" > "$slog" 2>&1
  local r=$?
  if [ "$r" = "$exp" ]; then echo "[m169] 判据 $desc：rc=$r（期望 $exp）✓"
  else echo "[m169] 判据 $desc：rc=$r（期望 $exp）✗ —— 见 $slog"; rc_all=1; fi
}

# judge_red_h2blk <desc> <dir> <m> <expect_ulp>：m=4097 的 h1/h2 判据里 h2.blk 是**登记红项**；
# 期望 rc=1 且 h2.blk 的 ulpMax 恰为 <expect_ulp>（移动/变绿都判 ✗，以传播基线漂移）。
judge_red_h2blk() {
  local desc="$1" D="$2" m="$3" exp="$4"
  local slog="$HERE/logs/judge_$(echo "$desc" | tr ' /' '__').log"
  PYTHONDONTWRITEBYTECODE=1 "$PY" "$M140/check_full_layer.py" "$D" Pf.gdn "$m" 15 > "$slog" 2>&1
  local r=$? ulp
  ulp="$(grep -E 'h2.blk' "$slog" | grep -oE 'ulpMax=[0-9]+' | head -1 | cut -d= -f2)"
  if [ "$r" = 1 ] && [ "$ulp" = "$exp" ]; then
    echo "[m169] 判据 $desc：rc=1、h2.blk ulpMax=$ulp（登记红项；期望 ulpMax=$exp）✓"
  else
    echo "[m169] 判据 $desc：rc=$r、h2.blk ulpMax='$ulp' 与登记（rc=1 / ulpMax=$exp）不符 ✗ —— 见 $slog"; rc_all=1
  fi
}

offline() {
  local D
  echo "[m169] ==== 离线对拍（锁外；参考全部为现成脚本）===="

  for spec in "m1_p15_links 1" "m4097_p15_links 4097"; do
    set -- $spec; local nm="$1" m="$2"; D="$HERE/dumps_$nm"
    echo "[m169] ---- $nm（m=$m，四相位 + 两条链路）----"
    if [ "$m" = 4097 ]; then
      judge_red_h2blk "$nm hc(H1/H2 m20ref)" "$D" "$m" 9
    else
      judge "$nm hc(H1/H2 m20ref)" 0 "$PY" "$M140/check_full_layer.py" "$D" Pf.gdn "$m" 15
    fi
    judge "$nm phaseA(m23)"       0 "$PY" "$ROOT/m23_gdn_prefill/check_ref.py" --dir "$D" --allow-stale
    judge "$nm phaseB(m26)"       0 "$PY" "$ROOT/m26_moe_prefill/check_ref.py" "$D/m26_Pf.gdn"
    judge "$nm phaseA_nan(m151)"  0 "$PY" "$M151/check_phaseA_nan.py" "$D"
    judge "$nm epilog(m164)"      0 "$PY" "$M164/check_epilog.py" "$D"
    judge "$nm prolog(m151 S2/S3)" 0 "$PY" "$M151/check_prolog.py" "$D"
  done

  for spec in "m1_p15_nolinks 1" "m4097_p15_nolinks 4097"; do
    set -- $spec; local nm="$1" m="$2"; D="$HERE/dumps_$nm"
    echo "[m169] ---- $nm（m=$m，四相位、两条链路关：旧基线回归）----"
    if [ "$m" = 4097 ]; then
      judge_red_h2blk "$nm hc(H1/H2 m20ref)" "$D" "$m" 11
    else
      judge "$nm hc(H1/H2 m20ref)" 0 "$PY" "$M140/check_full_layer.py" "$D" Pf.gdn "$m" 15
    fi
    judge "$nm phaseA(m23)"       0 "$PY" "$ROOT/m23_gdn_prefill/check_ref.py" --dir "$D" --allow-stale
    judge "$nm phaseB(m26)"       0 "$PY" "$ROOT/m26_moe_prefill/check_ref.py" "$D/m26_Pf.gdn"
    judge "$nm phaseA_nan(m151)"  0 "$PY" "$M151/check_phaseA_nan.py" "$D"
  done

  echo "[m169] ---- m1_p12_iso（H2|B，H1 关）----"
  judge "m1_p12_iso phaseB(m26)" 0 "$PY" "$ROOT/m26_moe_prefill/check_ref.py" "$HERE/dumps_m1_p12_iso/m26_Pf.gdn"

  # ---- 见证①：prolog 链路确实生效（相位 A 的 q/k/v/g/β 由设备产出 vs 宿主合成）----
  echo "[m169] ---- 见证①（prolog）：device 档 vs nolinks 档的相位 A 输入面 sha256 ----"
  for t in q k v g beta; do
    a=$(sha256sum "$HERE/dumps_m1_p15_links/m23_Pf.gdn_$t.bin" 2>/dev/null | cut -c1-16)
    b=$(sha256sum "$HERE/dumps_m1_p15_nolinks/m23_Pf.gdn_$t.bin" 2>/dev/null | cut -c1-16)
    if [ -z "$a" ] || [ -z "$b" ]; then echo "  $t: 缺 dump ⇒ 未取得读数"; rc_all=1;
    elif [ "$a" = "$b" ]; then echo "  $t: SAME（**异常**：prolog 档应与宿主合成不同）"; rc_all=1;
    else echo "  $t: device=$a nolinks=$b DIFFER ✓"; fi
  done

  # ---- 见证②：epilog 链路确实生效（H2 的 bo 输入 = epilog 设备产出 vs 宿主合成）----
  echo "[m169] ---- 见证②（epilog）：H2 的 bo/blk 面 sha256（links vs nolinks）----"
  for t in h2_bo h2_blk; do
    a=$(sha256sum "$HERE/dumps_m1_p15_links/m140_Pf.gdn_$t.bin" 2>/dev/null | cut -c1-16)
    b=$(sha256sum "$HERE/dumps_m1_p15_nolinks/m140_Pf.gdn_$t.bin" 2>/dev/null | cut -c1-16)
    if [ -z "$a" ] || [ -z "$b" ]; then echo "  $t: 缺 dump ⇒ 未取得读数"; rc_all=1;
    elif [ "$a" = "$b" ]; then echo "  $t: SAME（**异常**：epilog 应改变 H2 的 bo 输入及其 blk 输出）"; rc_all=1;
    else echo "  $t: links=$a nolinks=$b DIFFER ✓"; fi
  done
  # 两条链路在同一档内同时挂载的直接见证：该档同时存在 S2/S3（prolog）与 epilog 段产物
  for f in m151_s2_Pf.gdn m151_s3_Pf.gdn m164_Pf.gdn; do
    [ -d "$HERE/dumps_m1_p15_links/$f" ] && echo "  档内产物存在：$f/ ✓" || { echo "  档内产物缺失：$f/ ✗"; rc_all=1; }
  done
}

if [ "${1:-}" = "--offline-only" ]; then
  offline
else
  run_one m1_p15_links      15 1    1 1 240
  run_one m1_p15_nolinks    15 1    0 0 240
  run_one m1_p12_iso        12 1    0 0 240
  run_one m4097_p15_links   15 4097 1 1 300
  run_one m4097_p15_nolinks 15 4097 0 0 300

  # 进程内判据期望：四相位 clean 档全绿（checks 数随档不同）；m1_p12_iso 是既有红项（fails=3）
  kernel_expect m1_p15_links      "ALL PASS"
  kernel_expect m1_p15_nolinks    "ALL PASS"
  kernel_expect m4097_p15_links   "ALL PASS"
  kernel_expect m4097_p15_nolinks "ALL PASS"
  kernel_expect m1_p12_iso        "FAILURES PRESENT（checks=[0-9]+, guards=[0-9]+, fails=3）"

  offline
fi
echo "[m169] ==== reproduce 汇总：$([ $rc_all -eq 0 ] && echo ALL PASS || echo FAILURES PRESENT) ===="
exit $rc_all
