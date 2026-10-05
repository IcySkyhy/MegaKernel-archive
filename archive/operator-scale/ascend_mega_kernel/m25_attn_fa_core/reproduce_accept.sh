#!/usr/bin/env bash
# ============================================================
# reproduce_accept.sh —— M196：m25 attention FA core 的**生产形态设备验收**一键复跑
#
# 两个验收形态 **就是 harness 的默认档**（无需新参数）：
#   M25FA_M 不设 ⇒ "4097,1"
#     m=4097 → ctx=4097, posBase=0     （prefill）
#     m=1    → ctx=4097, posBase=4096  （decode）
#   M25FA_BLOCKS 不设 ⇒ nBlk = aclrtGetDeviceInfo(AICORE_CORE_NUM)（本机 28）
#
# 判据 = **自建 fp64 dense-causal 参考**（`check_ref.py`）+ 合成确定性 bf16 数据；
# **不是**真实 checkpoint 权重 / 真实 KV 的 parity（那需要另造 harness）。
#
# 内置期望（任一不符 ⇒ 退出码非零）：
#   · `core` 两档 VERDICT=PASS；
#   · 同一 `m` 多次运行的 `out.bin` sha256 一致（确定性）；
#   · 三个负向对照在 `M25FA_NEG_M`（默认 4097，prefill）VERDICT=FAIL；
#   · nTiles 按段体 `FacMakeWork` 公式自算 = 33（m=4097 的上界 / m=1 的常量）。
#
# 设备纪律：`flock -w 300 /tmp/npu0.lock`；进锁先 `npu-smi`（落盘）；`timeout` 在锁内。
# 超时（timeout 杀进程）**记「未取得读数」、不算 FAIL**，本脚本按缺 out.bin 中止。
#
# 环境变量：
#   M25FA_ACCEPT_OUT  落盘根（默认 /tmp/m196_accept）
#   M25FA_REPEAT      core 连跑次数（默认 2）
#   M25FA_M           跑哪些 m（默认 "4097,1"）
#   M25FA_NEG_M       负向对照用的 m（默认 4097）
#   M25FA_TIMEOUT     每档 timeout 秒（默认 120）
#   M25FA_BUILD=1     先 clean build
#   M25FA_TIMING=0    跳过墙钟计时（默认 1，用 time_core.py 量设备核时间）
# ============================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BIN="$HERE/build/m25_attn_fa_core"
OUT="${M25FA_ACCEPT_OUT:-/tmp/m196_accept}"
REPEAT="${M25FA_REPEAT:-2}"
MS="${M25FA_M:-4097,1}"
NEG_M="${M25FA_NEG_M:-4097}"
TMO="${M25FA_TIMEOUT:-120}"
PY=/usr/local/python3.12.13/bin/python3
[[ -x "$PY" ]] || PY=python3

mkdir -p "$OUT/logs"

if [[ "${M25FA_BUILD:-0}" == "1" ]]; then
  echo "=== clean build ==="
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  rm -rf "$HERE/build"
  cmake -B "$HERE/build" -S "$HERE" -DCMAKE_BUILD_TYPE=Release || exit 1
  cmake --build "$HERE/build" -j4 || exit 1
fi
[[ -x "$BIN" ]] || { echo "[reproduce_accept][FAIL] 缺可执行 $BIN（M25FA_BUILD=1 或先手动构建）"; exit 1; }

echo "=== 基线：M189 是否在 HEAD 的祖先里 ==="
if git -C "$REPO" merge-base --is-ancestor f66ad1e HEAD 2>/dev/null; then
  echo "M189 merge f66ad1e IS ancestor of $(git -C "$REPO" rev-parse --short HEAD)"
else
  echo "[reproduce_accept][FAIL] M189 merge f66ad1e 不在 HEAD 祖先里 —— 基线不对，拒绝跑验收"
  exit 1
fi

echo
echo "=== nTiles（按 m15_attn_fa_core.h::FacMakeWork 公式自算；内置期望 33）==="
"$PY" - "$MS" <<'PYEOF'
import math, sys
P, SIN, NH = 64, 128, 24
bad = 0
for tok in sys.argv[1].split(','):
    m = int(tok)
    ctx = 4097 if m == 1 else m
    posBase = (ctx - 1) if m == 1 else 0
    vals = []
    for qT in range(math.ceil(m / P)):
        hi = min(posBase + qT * P + P, ctx)
        vals.append(math.ceil(hi / SIN))
    mx = max(vals)
    print(f"  m={m} ctx={ctx} posBase={posBase} nQTile={len(vals)} nTiles(min/max)={min(vals)}/{mx}")
    if mx != 33:
        print(f"  [FAIL] m={m} nTiles max={mx} != 33")
        bad = 1
sys.exit(bad)
PYEOF
[[ $? -eq 0 ]] || { echo "[reproduce_accept][FAIL] nTiles 与内置期望 33 不符"; exit 1; }

export BIN OUT TMO

echo
echo "=== 设备：core × $REPEAT（M25FA_M=$MS）==="
for i in $(seq 1 "$REPEAT"); do
  export LABEL="rep$i" M25FA_M="$MS" M25FA_OUT="$OUT/rep$i"
  flock -w 300 /tmp/npu0.lock bash -c '
    npu-smi info -t common -i 0 | head -8
    echo "[cmd] core $LABEL M25FA_M=$M25FA_M M25FA_OUT=$M25FA_OUT timeout '"$TMO"'"
    timeout "$TMO" "$BIN" core
    echo "EXIT_CORE_$LABEL=$?"
  ' 2>&1 | tee "$OUT/logs/core_$LABEL.log"
done

echo
echo "=== 设备：负向对照 × 3（M25FA_M=$NEG_M；m=1 上不咬，见 evidence/M196_readings.txt）==="
for mode in negmask negshift negstart; do
  export M25FA_M="$NEG_M" M25FA_OUT="$OUT/neg"
  flock -w 300 /tmp/npu0.lock bash -c '
    npu-smi info -t common -i 0 | head -8
    echo "[cmd] '"$mode"' M25FA_M=$M25FA_M M25FA_OUT=$M25FA_OUT timeout '"$TMO"'"
    timeout "$TMO" "$BIN" '"$mode"'
    echo "EXIT_'"$mode"'=$?"
  ' 2>&1 | tee "$OUT/logs/neg_${NEG_M}_${mode}.log"
done

RC=0
echo
echo "=== 判定：core / 负向对照（check_ref.py）==="
for i in $(seq 1 "$REPEAT"); do
  for m in ${MS//,/ }; do
    d="$OUT/rep$i/m${m}_core"
    if [[ -f "$d/out.bin" ]]; then
      line=$("$PY" "$HERE/check_ref.py" "$d" --quiet 2>&1 | tail -1)
      echo "[core rep$i m=$m] $line"
      [[ "$line" == VERDICT=PASS* ]] || RC=1
    else
      echo "[core rep$i m=$m] 缺 out.bin（timeout 或跑挂 ⇒ 未取得读数）"; RC=1
    fi
  done
done
for mode in negmask negshift negstart; do
  d="$OUT/neg/m${NEG_M}_${mode}"
  if [[ -f "$d/out.bin" ]]; then
    line=$("$PY" "$HERE/check_ref.py" "$d" --quiet 2>&1 | tail -1)
    echo "[$mode m=$NEG_M] $line"
    [[ "$line" == VERDICT=FAIL* ]] || { echo "  ^^ 负向对照没变红 —— 判据在该档空洞"; RC=1; }
  else
    echo "[$mode m=$NEG_M] 缺 out.bin"; RC=1
  fi
done

echo
echo "=== 确定性：同一 m 多次 out.bin sha256 ==="
for m in ${MS//,/ }; do
  n=$(for i in $(seq 1 "$REPEAT"); do sha256sum "$OUT/rep$i/m${m}_core/out.bin" 2>/dev/null | cut -d' ' -f1; done | sort -u | wc -l)
  echo "m=$m 的不同 sha256 数 = $n（期望 1）"
  [[ "$n" == "1" ]] || RC=1
done

if [[ "${M25FA_TIMING:-1}" == "1" ]]; then
  echo
  echo "=== 墙钟（time_core.py：只读既有 harness 的 flushed 打印，不新增探针）==="
  flock -w 300 /tmp/npu0.lock bash -c '
    npu-smi info -t common -i 0 | head -8
    for m in 4097 1; do
      "'"$PY"'" "'"$HERE"'/time_core.py" "'"$BIN"'" core "$m" "'"$OUT"'/timing"
    done
  ' 2>&1 | tee "$OUT/logs/timing.log"
fi

echo
echo "REPRODUCE_ACCEPT_RC=$RC"
exit $RC
