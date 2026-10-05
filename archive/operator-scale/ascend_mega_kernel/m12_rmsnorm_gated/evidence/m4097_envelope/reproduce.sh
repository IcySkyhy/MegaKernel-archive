#!/usr/bin/env bash
# M150 —— m12 RMSNormGated m 包络设备取证（m=257 / m=4097）：可复算脚本。
#
# 默认只做离线复算（sha256 + 两档 check_ref 容差对拍 + 负向对照三检查）。
#   bash m12_rmsnorm_gated/evidence/m4097_envelope/reproduce.sh
# 置 M12_DEVICE=1 先重跑设备档再复算（需 source /usr/local/Ascend/ascend-toolkit/set_env.sh）：
#   M12_DEVICE=1 bash m12_rmsnorm_gated/evidence/m4097_envelope/reproduce.sh
#
# 失败语义：脚本汇总每项检查，任一失败即 exit 1。大 bin 不入库，缺任一 bin 的入库默认
# 状态下**响亮失败**（打印「未取得读数（需 M12_DEVICE=1 重建）」），不会被当成通过。
# 负向对照拆成三个独立检查（见下）：「参考因缺文件崩溃」走失败路径，不会被当成「判据如期变红」。
#
# 纪律（同 m15_layer_loop/evidence、m11 evidence 惯例）：每档独立一次 flock -w 300；进锁先
# npu-smi（快照落盘）；timeout 在锁内；未取得锁时写「未取得读数」，不伪造读数。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-/usr/local/python3.12.13/bin/python3}"
BIN="$REPO/m12_rmsnorm_gated/build/m12_rmsnorm_gated"
CHKREF="$REPO/m12_rmsnorm_gated/check_ref.py"
mkdir -p "$HERE/logs"

FAIL=0
fail() { echo "  [FAIL] $*" >&2; FAIL=1; }

bins_of() { echo "$1/m${2}_s0_o.bin $1/m${2}_s0_z.bin $1/m${2}_s0_gamma.bin $1/m${2}_s0_y.bin $1/m${2}_s0_y_device.bin"; }

need_bins() {  # $1=dump dir $2=m；5 个 bin 缺任一即响亮失败
  local d="$1" m="$2" f
  for f in $(bins_of "$d" "$m"); do
    if [ ! -f "$f" ]; then
      fail "缺 $f —— 未取得读数（需 M12_DEVICE=1 重建）"
      return 1
    fi
  done
  return 0
}

run_case() {  # $1=m
  local m="$1"
  local dump="$HERE/dumps_m${m}_s0"
  mkdir -p "$dump"
  local log="$HERE/logs/m${m}_s0.run.log"
  flock -w 300 /tmp/npu0.lock bash -c "
    cd '$dump'
    echo '--- npu-smi (lock entry) ---'; npu-smi info
    echo '--- cmd: m12_rmsnorm_gated dump $m 0 ---'
    timeout 280 '$BIN' dump $m 0
  " > "$log" 2>&1
  local rc=$?
  echo "lock_exit=$rc" >> "$log"
  if [ "$rc" -ne 0 ]; then
    fail "[m=$m] 未取得读数（flock/timeout rc=$rc；见 $log）"
  elif ! grep -q ': PASS' "$log"; then
    fail "[m=$m] 设备档未打印 PASS（见 $log）"
  fi
}

check_case() {  # $1=m
  local m="$1"
  local dump="$HERE/dumps_m${m}_s0"
  local log="$HERE/logs/m${m}_s0.check_ref.log"
  echo "== check_ref: m=$m =="
  if ! need_bins "$dump" "$m"; then return; fi
  ( cd "$dump" && "$PY" "$CHKREF" "$m" 0 ) | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then fail "check_ref m=$m 退出码 $rc（期望 0）"; fi
}

# 篡改 gamma 的单个 bf16 元素：$1=path $2=byte offset $3=xor mask
tamper_byte() {
  "$PY" - "$1" "$2" "$3" <<'EOF'
import sys
p, off, mask = sys.argv[1], int(sys.argv[2]), int(sys.argv[3], 0)
with open(p, "r+b") as f:
    f.seek(off); b = f.read(1)[0]
    f.seek(off); f.write(bytes([b ^ mask]))
EOF
}

# 负向对照单检查：$1=tag $2=byte offset $3=xor mask $4=logfile
# 期望：check_ref 变红（rc!=0）+ NPU-dev 行 False + 有 >0 条 mismatch。
neg_red() {
  local tag="$1" off="$2" mask="$3" log="$4" rc
  cp "$HERE/dumps_m4097_s0/"m4097_s0_{o,z,gamma,y,y_device}.bin "$NEG/" 2>/dev/null || { fail "负控($tag)：拷 bin 失败"; return; }
  tamper_byte "$NEG/m4097_s0_gamma.bin" "$off" "$mask"
  ( cd "$NEG" && "$PY" "$CHKREF" 4097 0 ) > "$log" 2>&1
  rc=$?
  cat "$log"
  if [ "$rc" -eq 0 ]; then
    fail "负控($tag)：篡改后仍 rc=0（判据没有牙）"
  elif ! grep -q "NPU-dev vs numpy: out rows all within 0.01 rel: False" "$log"; then
    fail "负控($tag)：未出现 NPU-dev False（疑似崩溃，rc=$rc）—— 不作为达标"
  elif ! grep -qE 'total [1-9][0-9]*' "$log"; then
    fail "负控($tag)：mismatch 计数非正"
  else
    echo "  [OK] 负控($tag)：rc=$rc，NPU-dev False 且 mismatch>0"
  fi
  rm -f "$NEG/m4097_s0_"*.bin
}

if [ "${M12_DEVICE:-0}" = "1" ]; then
  echo "== 重跑设备档（m=257 / m=4097，各一次 flock，npu-smi 在锁内）=="
  cmake -B "$REPO/m12_rmsnorm_gated/build" -S "$REPO/m12_rmsnorm_gated" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$REPO/m12_rmsnorm_gated/build" -j4 >/dev/null
  run_case 257
  run_case 4097
fi

echo "== sha256 校验（入库档：大 bin 不入库，按上面设备档重建后即可核对）=="
for m in 257 4097; do
  dump="$HERE/dumps_m${m}_s0"
  if need_bins "$dump" "$m"; then
    ( cd "$dump" && sha256sum -c sha256sums.txt ) || fail "sha256 m=$m 校验失败"
  fi
done

check_case 257
check_case 4097

echo "== 负向对照（三个独立检查；m12 无内建 mutant，用数据侧篡改）=="
NEG="$(mktemp -d)"
trap 'rm -rf "$NEG"' EXIT
if need_bins "$HERE/dumps_m4097_s0" 4097; then
  # (a) 参考自身必须先在未篡改副本上跑通（exit 0 且 NPU-dev True）
  echo "-- (a) 参考自检（未篡改副本，必须 True）--"
  cp "$HERE/dumps_m4097_s0/"m4097_s0_{o,z,gamma,y,y_device}.bin "$NEG/"
  ( cd "$NEG" && "$PY" "$CHKREF" 4097 0 ) > "$HERE/logs/negative_control_ref.log" 2>&1
  rc_ref=$?
  cat "$HERE/logs/negative_control_ref.log"
  if [ "$rc_ref" -ne 0 ] || ! grep -q "NPU-dev vs numpy: out rows all within 0.01 rel: True" "$HERE/logs/negative_control_ref.log"; then
    fail "负控(a)：参考自身未跑通（rc=$rc_ref）—— 这是复算环境问题，不是判据变红"
  fi
  rm -f "$NEG/m4097_s0_"*.bin
  # (b) gamma[1] 低字节 bit0 = 1 个 bf16 ULP（字节偏移 2）：必须变红（M146 同款手法）
  echo "-- (b) 篡改 gamma[1] 1 bf16 ULP（byte offset 2 bit0），必须变红 --"
  neg_red "1ulp" 2 0x01 "$HERE/logs/negative_control_1ulp.log"
  # (c) gamma[1] 符号位（byte offset 3 bit7，-1.0 -> +1.0）：把该列整体取反，必须强红
  echo "-- (c) 篡改 gamma[1] 符号位（byte offset 3 bit7），必须变红 --"
  neg_red "sign" 3 0x80 "$HERE/logs/negative_control.log"
fi

echo "== 结果 =="
if [ "$FAIL" -ne 0 ]; then
  echo "REPRODUCE FAILED"
  exit 1
fi
echo "REPRODUCE OK"
