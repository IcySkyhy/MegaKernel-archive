#!/usr/bin/env bash
# M175 reproduce: `M15_PREFILL_EPILOG_MUT` 越界「假绿」脚枪 —— 修前（base）vs 修后（tip）
# 两个二进制、同一档矩阵（MUT 0..7），内建期望与失败非零退出。
#
# 要复现的两个断言：
#   ① 修前：MUT 0..7 全部 `ALL PASS`（越界值被 `& 7u` 静默当 clean ⇒ 脚枪现场）。
#   ② 修后：MUT 0..4 `ALL PASS（checks=89, guards=9, fails=0）` rc=0；
#            MUT 5..7 `FAILURES PRESENT（checks=2, guards=9, fails=5）` rc=1，且进程内先有一行 WARN。
#   ③ 修前/修后 MUT 0..4 的判据读数逐行相同（剔除墙钟「耗时」行后 diff 必须为空）。
#
# 纪律（塔口径）：每档各自进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（快照落进该档日志）、
#   `timeout` 在锁内、一次进锁一条短命令；等锁没拿到 / 没取到读数 ⇒ 记「未取得读数」并以非零退出
#   （不写成"未复现"）。所有归档日志落在本目录 `logs/`（已按 §9.12 把临时路径改写成 WORK/REPO 前缀）。
#
# 用法：
#   bash reproduce.sh                 # 构建修前+修后两个二进制，跑 16 档设备矩阵 + 逐字比对
#   bash reproduce.sh --no-build      # 复用已有二进制（见下 M175_BASE_BIN / M175_FIX_BIN）
#
# 可覆盖的环境变量（都有缺省，缺省路径即可跑）：
#   M175_BASE_SHA  修前源码的 rev（缺省 = 本 mission 的 base，即 tip 的父提交）
#   M175_BASE_SRC  已解出的修前源码树（设了就不再用 git archive）
#   M175_BASE_BIN / M175_FIX_BIN  已构建好的二进制（设了就不重建那一个）
#   M175_WORK      临时构建/日志目录（缺省 = mktemp -d）
set -u
shopt -s nullglob
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LOCK=/tmp/npu0.lock
MAN="$ROOT/m15_layer_loop/weights_manifest.txt"

# 修前 = 本 mission 的 base（tip 的父提交 449c5ea）；可用 M175_BASE_SHA 覆盖。
BASE_SHA="${M175_BASE_SHA:-449c5ea9f5ed5b887c241adba535d043e7e820a9}"

WORK="${M175_WORK:-$(mktemp -d /tmp/m175_repro.XXXXXX)}"
RAW="$WORK/logs"          # 原始日志（不入仓）
LOGS="$HERE/logs"         # 归档日志（入仓；已 sanitize）
DUMPS="$WORK/dumps"
mkdir -p "$RAW" "$LOGS" "$DUMPS"

rc_all=0
verdict_re='ALL PASS（checks=[0-9]+, guards=[0-9]+, fails=[0-9]+）|FAILURES PRESENT（checks=[0-9]+, guards=[0-9]+, fails=[0-9]+）'

say()  { echo "[m175] $*"; }
fail() { echo "[m175] ✗ $*"; rc_all=1; }

# ---------- 构建 ----------
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

build_pre() {
  local src="$1"
  cmake -B "$src/m15_layer_loop/build" -S "$src/m15_layer_loop" -DCMAKE_BUILD_TYPE=Release > "$WORK/cmake_pre.log" 2>&1 \
    && cmake --build "$src/m15_layer_loop/build" -j4 --target m15_layer_loop >> "$WORK/cmake_pre.log" 2>&1
}

PRE_BIN="${M175_BASE_BIN:-}"
FIX_BIN="${M175_FIX_BIN:-}"

if [ "${1:-}" != "--no-build" ]; then
  if [ -z "$PRE_BIN" ]; then
    if [ -n "${M175_BASE_SRC:-}" ]; then
      PRE_SRC="$M175_BASE_SRC"
    else
      PRE_SRC="$WORK/pre_src"
      mkdir -p "$PRE_SRC"
      say "解出修前源码（rev $BASE_SHA）→ $PRE_SRC"
      git -C "$ROOT" archive "$BASE_SHA" | tar -x -C "$PRE_SRC" \
        || { say "✗ 解不出修前源码（rev $BASE_SHA 不可达？）"; exit 1; }
    fi
    say "构建修前二进制 …"
    if build_pre "$PRE_SRC"; then
      PRE_BIN="$PRE_SRC/m15_layer_loop/build/m15_layer_loop"
    else
      say "✗ 修前构建失败 —— 见 $WORK/cmake_pre.log"; exit 1
    fi
  fi
  if [ -z "$FIX_BIN" ]; then
    say "构建修后二进制 …"
    cmake -B "$WORK/fix_build" -S "$ROOT/m15_layer_loop" -DCMAKE_BUILD_TYPE=Release > "$WORK/cmake_fix.log" 2>&1 \
      && cmake --build "$WORK/fix_build" -j4 --target m15_layer_loop >> "$WORK/cmake_fix.log" 2>&1 \
      || { say "✗ 修后构建失败 —— 见 $WORK/cmake_fix.log"; exit 1; }
    FIX_BIN="$WORK/fix_build/m15_layer_loop"
  fi
fi

for b in "$PRE_BIN" "$FIX_BIN"; do
  [ -n "$b" ] && [ -x "$b" ] || { say "✗ 缺可执行二进制（PRE_BIN='$PRE_BIN' FIX_BIN='$FIX_BIN'）"; exit 1; }
done
say "修前二进制 = $PRE_BIN"
say "修后二进制 = $FIX_BIN"

# 编译告警数（仅在做构建时有意义）：修后不得多于修前（README §3 的"告警逐条相同"由此可核）
wp="n/a"; wf="n/a"
if [ -f "$WORK/cmake_pre.log" ] && [ -f "$WORK/cmake_fix.log" ]; then
  wp="$(grep -c 'warning:' "$WORK/cmake_pre.log")"
  wf="$(grep -c 'warning:' "$WORK/cmake_fix.log")"
  if [ "$wp" = "$wf" ]; then say "✓ 构建告警数 修前=$wp 修后=$wf（相同）"
  else fail "构建告警数 修前=$wp 修后=$wf（不相同）"; fi
fi

# ---------- 设备档 ----------
# run_arm <tag> <bin> <mut> <timeout>：写 $RAW/<tag>_mut<mut>.log；取不到读数即返回 1
run_arm() {
  local tag="$1" bin="$2" mut="$3" to="$4"
  local log="$RAW/${tag}_mut${mut}.log"
  local dump="$DUMPS/mut${mut}"          # 修前/修后同路径（逐档串行）⇒ 日志里该路径逐字相同
  rm -rf "$dump"; mkdir -p "$dump"
  if ! flock -w 300 "$LOCK" bash -c "
    {
      echo '=== lock acquired '\$(date -Is)' tag=$tag mut=$mut ===';
      echo '--- npu-smi (lock entry) ---';
      npu-smi info | head -12;
      echo '--- command (verbatim) ---';
      echo 'M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_PHASES=7 M15_PREFILL_PROLOG=1 M15_PREFILL_EPILOG=1 M15_PREFILL_EPILOG_MUT=$mut M15_PREFILL_M=1 M15_PREFILL_DUMPDIR=$dump $bin $MAN prefill';
      echo '--- program output ---';
      cd '$ROOT' && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null;
      timeout $to env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
        M15_PREFILL_PHASES=7 M15_PREFILL_PROLOG=1 M15_PREFILL_EPILOG=1 M15_PREFILL_EPILOG_MUT=$mut \
        M15_PREFILL_M=1 M15_PREFILL_DUMPDIR=$dump '$bin' '$MAN' prefill;
      echo '--- exit='\$?' ---';
    } > '$log' 2>&1
  "; then
    say "未取得读数：锁未拿到（$tag mut=$mut）"; return 1
  fi
  if grep -q 'lock acquired' "$log" && grep -q -- '--- exit=' "$log"; then
    return 0
  fi
  say "未取得读数（$tag mut=$mut：无锁内快照 / 无程序输出）"; return 1
}

# expect <desc> <log> <verdict-regex> <expected-rc>
expect() {
  local desc="$1" log="$2" re="$3" want_rc="$4"
  local got rc
  got="$(grep -oE "$verdict_re" "$log" | tail -1)"
  rc="$(grep -oE 'exit=[0-9]+' "$log" | tail -1 | cut -d= -f2)"
  if [ "$rc" != "$want_rc" ]; then fail "$desc：rc=$rc（期望 $want_rc）—— $log"; return; fi
  if echo "$got" | grep -qE "$re"; then say "✓ $desc：$got（rc=$rc）"
  else fail "$desc：'$got' 不匹配 /$re/（rc=$rc）—— $log"; fi
}

# norm <log>：只留程序输出段，剔除墙钟「耗时」行
norm() {
  awk '/^--- program output ---$/{p=1;next} /^--- exit=/{p=0} p' "$1" | grep -v '耗时'
}

say "==== 设备档：修前 MUT 0..7（期望全部 ALL PASS，rc=0）===="
for m in 0 1 2 3 4 5 6 7; do
  run_arm pre "$PRE_BIN" "$m" 240 || { fail "修前 MUT=$m 未取得读数"; continue; }
  expect "修前 MUT=$m" "$RAW/pre_mut$m.log" 'ALL PASS' 0
done

say "==== 设备档：修后 MUT 0..7 ===="
for m in 0 1 2 3 4 5 6 7; do
  run_arm fix "$FIX_BIN" "$m" 240 || { fail "修后 MUT=$m 未取得读数"; continue; }
  if [ "$m" -le 4 ]; then
    expect "修后 MUT=$m" "$RAW/fix_mut$m.log" '^ALL PASS（checks=89, guards=9, fails=0）$' 0
  else
    expect "修后 MUT=$m" "$RAW/fix_mut$m.log" '^FAILURES PRESENT（checks=2, guards=9, fails=5）$' 1
    grep -qE '\[m15\]\[WARN\] M15_PREFILL_EPILOG_MUT=[0-9]+ 越界' "$RAW/fix_mut$m.log" \
      && say "✓ 修后 MUT=$m 先有一行越界 WARN" \
      || fail "修后 MUT=$m 未见越界 WARN 行 —— $RAW/fix_mut$m.log"
  fi
done

say "==== 逐字比对：修前/修后 MUT 0..4（剔墙钟「耗时」行后 diff 必须为空）===="
diff_ok=1
for m in 0 1 2 3 4; do
  out="$LOGS/normdiff_mut$m.txt"
  if diff <(norm "$RAW/pre_mut$m.log") <(norm "$RAW/fix_mut$m.log") > "$out"; then
    say "✓ MUT=$m 判据读数逐行相同（差异 0 行）"
  else
    fail "MUT=$m 存在差异（见 $(basename "$out")）"; diff_ok=0
  fi
done

# ---------- 归档（sanitize：临时路径 → WORK/REPO 前缀）----------
for f in "$RAW"/*.log; do
  sed -e "s#$WORK#WORK#g" -e "s#$ROOT#REPO#g" "$f" > "$LOGS/$(basename "$f")"
done
{
  echo "M175 reproduce 汇总 @ $(date -Is)"
  echo "base rev   = $BASE_SHA"
  echo "pre  bin   = $(sha256sum "$PRE_BIN" | cut -d' ' -f1)"
  echo "fix  bin   = $(sha256sum "$FIX_BIN" | cut -d' ' -f1)"
  echo "build warn = pre:$wp / fix:$wf（grep -c 'warning:'）"
  echo "---- 期望与实测（见 pre_mut*/fix_mut*.log）----"
  echo "pre  MUT 0..7 : 全部 ALL PASS / rc=0"
  echo "fix  MUT 0..4 : ALL PASS（checks=89, guards=9, fails=0）/ rc=0"
  echo "fix  MUT 5..7 : FAILURES PRESENT（checks=2, guards=9, fails=5）/ rc=1（先有一行越界 WARN）"
  echo "MUT 0..4 修前/修后剔耗时逐字比对 : $( [ "$diff_ok" = 1 ] && echo '差异 0 行' || echo '见 normdiff_*.txt' )"
} > "$LOGS/summary.txt"

say "归档 → $LOGS/（16 档日志 + normdiff_mut0..4.txt + summary.txt）"
say "==== reproduce 汇总：$([ "$rc_all" = 0 ] && echo 'ALL PASS' || echo 'FAILURES PRESENT') ===="
exit "$rc_all"
