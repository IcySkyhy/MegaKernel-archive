#!/usr/bin/env bash
# M136（Wave D）GDN prefill 相位 A 的**证据复算脚本**。
# 默认只做离线复算（check_ref + sha256）；置 M136_DEVICE=1 先重跑设备档再复算。
#
#   bash m15_layer_loop/evidence/m136_prefill_gdn/reproduce.sh              # 离线复算
#   M136_DEVICE=1 bash m15_layer_loop/evidence/m136_prefill_gdn/reproduce.sh # 重跑设备 + 复算
#
# 设备档前置：source /usr/local/Ascend/ascend-toolkit/set_env.sh
#            cmake --build m15_layer_loop/build -j8 --target m15_layer_loop
# 纪律：flock -w ≤300、进锁先 npu-smi、timeout 在锁内、一次进锁一条短命令。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-/usr/local/python3.12.13/bin/python3}"
BIN="$REPO/m15_layer_loop/build/m15_layer_loop"
MAN="$REPO/m15_layer_loop/weights_manifest.txt"

if [ "${M136_DEVICE:-0}" = "1" ]; then
  echo "== 重跑设备档（真权重 checkpoint 切片；M15_SKIP_WCHECK=1 只跳来源校验，不改权重口径）=="
  for M in 1 4097; do
    flock -w 300 /tmp/npu0.lock bash -c "
      echo '--- npu-smi (lock entry) ---'; npu-smi info
      timeout 150 env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
        M15_PREFILL_M=$M M15_PREFILL_DUMPDIR=$HERE/dumps_m$M \
        $BIN $MAN prefill"
  done
  flock -w 300 /tmp/npu0.lock bash -c "
    echo '--- npu-smi (lock entry) ---'; npu-smi info
    timeout 120 env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_M=1,4097 $BIN $MAN prefill"
fi

echo "== check_ref：clean / --mutant 1 × m=1 / m=4097（m23 的 numpy float64 逐句参考，rtol 2e-3）=="
for M in 1 4097; do
  echo "-- m=$M clean --"; "$PY" "$REPO/m23_gdn_prefill/check_ref.py" --dir "$HERE/dumps_m$M"
  echo "-- m=$M --mutant 1（期望变红）--"
  "$PY" "$REPO/m23_gdn_prefill/check_ref.py" --dir "$HERE/dumps_m$M" --mutant 1
done

echo "== dumps_m1 的 sha256 校验（入库档，含 bin）=="
( cd "$HERE/dumps_m1" && sha256sum -c sha256sums.txt )
echo "== dumps_m4097 的 sha256 校验（只入 meta + sha256；bin 需按上面设备档重建）=="
( cd "$HERE/dumps_m4097" && sha256sum -c sha256sums.txt )
