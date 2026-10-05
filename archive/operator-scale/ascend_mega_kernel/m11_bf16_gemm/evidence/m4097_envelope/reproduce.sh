#!/usr/bin/env bash
# M146 —— m11 bf16 GEMM m=4097 包络取证：可复算脚本。
#
# 默认只做离线复算（sha256 + check_ref 逐位对拍 + 负向对照）。
#   bash m11_bf16_gemm/evidence/m4097_envelope/reproduce.sh
# 置 M11_DEVICE=1 先重跑设备档再复算（需 source /usr/local/Ascend/ascend-toolkit/set_env.sh）：
#   M11_DEVICE=1 bash m11_bf16_gemm/evidence/m4097_envelope/reproduce.sh
#
# 失败语义：脚本汇总每项检查，任一失败即 exit 1。缺大 bin 的入库默认状态下会**响亮失败**
# （打印「未取得读数（需 M11_DEVICE=1 重建）」）。负向对照拆成两个独立检查（见下），
# 「参考因缺文件崩溃」走失败路径，不会被当成「判据如期变红」。
#
# 纪律（同 m15_layer_loop/evidence 惯例）：每档独立一次 flock -w 300；进锁先 npu-smi（快照落盘）；
# timeout 在锁内；未取得锁时写「未取得读数」，不伪造读数。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-/usr/local/python3.12.13/bin/python3}"
BIN="$REPO/m11_bf16_gemm/build/m11_bf16_gemm"
CHKREF="$REPO/m11_bf16_gemm/check_ref.py"

FAIL=0
fail() { echo "  [FAIL] $*" >&2; FAIL=1; }

need_bins() {  # $1=dump dir；三个 bin 缺任一即响亮失败
  local d="$1" f
  for f in a.bin b.bin c_device.bin; do
    if [ ! -f "$d/$f" ]; then
      fail "缺 $d/$f —— 未取得读数（需 M11_DEVICE=1 重建）"
      return 1
    fi
  done
  return 0
}

run_case() {  # $1=name $2=K $3=N $4=m
  local name="$1" k="$2" n="$3" m="$4"
  local dump="$HERE/dumps_${name}_m${m}"
  mkdir -p "$dump"
  local log="$HERE/logs/${name}_m${m}.run.log"
  flock -w 300 /tmp/npu0.lock bash -c "
    cd '$dump'
    echo '--- npu-smi (lock entry) ---'; npu-smi info
    echo '--- cmd: m11_bf16_gemm dump $k $n $m ---'
    timeout 280 '$BIN' dump $k $n $m
  " > "$log" 2>&1
  local rc=$?
  echo "lock_exit=$rc" >> "$log"
  if [ "$rc" -ne 0 ]; then
    fail "[${name} m=$m] 未取得读数（flock/timeout rc=$rc；见 $log）"
  fi
}

check_case() {  # $1=name $2=K $3=N $4=m
  local name="$1" k="$2" n="$3" m="$4"
  local dump="$HERE/dumps_${name}_m${m}"
  local log="$HERE/logs/${name}_m${m}.check_ref.log"
  echo "== check_ref: $name K=$k N=$n m=$m =="
  if ! need_bins "$dump"; then return; fi
  ( cd "$dump" && "$PY" "$CHKREF" "$k" "$n" "$m" ) | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then fail "check_ref $name m=$m 退出码 $rc（期望 0）"; fi
}

if [ "${M11_DEVICE:-0}" = "1" ]; then
  echo "== 重跑设备档（m=4097，两形状；每档一次 flock，npu-smi 在锁内）=="
  cmake -B "$REPO/m11_bf16_gemm/build" -S "$REPO/m11_bf16_gemm" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$REPO/m11_bf16_gemm/build" -j4 >/dev/null
  run_case in_proj  2560 16480 4097
  run_case out_proj 6144 2560  4097
fi

echo "== sha256 校验（入库档：大 bin 不入库，按上面设备档重建后即可核对）=="
for name in in_proj out_proj; do
  dump="$HERE/dumps_${name}_m4097"
  if need_bins "$dump"; then
    ( cd "$dump" && sha256sum -c sha256sums.txt ) || fail "sha256 $name 校验失败"
  fi
done

check_case in_proj  2560 16480 4097
check_case out_proj 6144 2560  4097

echo "== 负向对照（两个独立检查；m11 无内建 mutant，用数据侧篡改）=="
NEG="$(mktemp -d)"
trap 'rm -rf "$NEG"' EXIT
SRC="$HERE/dumps_in_proj_m4097"
if need_bins "$SRC"; then
  cp "$SRC/a.bin" "$SRC/b.bin" "$SRC/c_device.bin" "$NEG/"

  # -- 检查 (a)：参考自身必须先在未篡改副本上跑通（exit 0 且 device C: True）--
  echo "-- (a) 参考自检（未篡改副本，必须 True）--"
  ( cd "$NEG" && "$PY" "$CHKREF" 2560 16480 4097 ) > "$HERE/logs/negative_control_ref.log" 2>&1
  rc_ref=$?
  cat "$HERE/logs/negative_control_ref.log"
  if [ "$rc_ref" -ne 0 ] || ! grep -q "== device C: True" "$HERE/logs/negative_control_ref.log"; then
    fail "负控(a)：参考自身未跑通（rc=$rc_ref）—— 这是复算环境问题，不是判据变红"
  fi

  # -- 检查 (b)：篡改后必须变红。用「输出里出现 device C: False 且 mismatch>0」判定，
  #    与「check_ref 崩溃（traceback / 无 False 行）」区分开。--
  echo "-- (b) 篡改 offset 2 的 bit0（第 2 个 bf16 元素），必须变红 --"
  "$PY" - "$NEG/b.bin" <<'EOF'
import sys
p = sys.argv[1]
# b.bin 为 uint16 小端 bf16 位型：元素 i 占字节 [2i, 2i+1]。
# seek(2) 即元素 index 1（第 2 个元素）的低字节，异或 0x01 = 翻 1 个 bf16 ULP。
with open(p, "r+b") as f:
    f.seek(2)
    low = f.read(1)[0]
    f.seek(2)
    f.write(bytes([low ^ 0x01]))
EOF
  ( cd "$NEG" && "$PY" "$CHKREF" 2560 16480 4097 ) > "$HERE/logs/negative_control.log" 2>&1
  rc_mut=$?
  cat "$HERE/logs/negative_control.log"
  if [ "$rc_mut" -eq 0 ]; then
    fail "负控(b)：篡改后仍为 True —— 判据没有牙"
  elif ! grep -q "== device C: False" "$HERE/logs/negative_control.log"; then
    fail "负控(b)：篡改后未出现 device C: False（疑似崩溃，rc=$rc_mut）—— 不作为达标"
  elif ! grep -qE 'mismatches: [1-9][0-9]*' "$HERE/logs/negative_control.log"; then
    fail "负控(b)：篡改后 mismatch 计数非正"
  else
    echo "  [OK] 负控(b)：篡改后 device C: False 且 mismatch>0"
  fi
fi

echo "== 结果 =="
if [ "$FAIL" -ne 0 ]; then
  echo "REPRODUCE FAILED"
  exit 1
fi
echo "REPRODUCE OK"
