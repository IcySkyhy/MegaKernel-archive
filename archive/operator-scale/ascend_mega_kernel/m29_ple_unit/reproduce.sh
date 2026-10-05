#!/usr/bin/env bash
# M161 PLE 单元：一条命令复现（构建 → 数据 → 设备跑 → 判据 → 负向对照）
#
#   bash m29_ple_unit/reproduce.sh [输出目录]        # 缺省 /tmp/m29_repro
#
# 设备槽纪律（塔口径）：**每档各自进一次** `flock -w 300 /tmp/npu0.lock`；进锁先 `npu-smi`
#   （快照落盘）；`timeout` 放在锁内；等锁仍未取得读数则命令直接失败（日志里的 rc 即如实反映）。
#   锁文件 = `/tmp/npu0.lock`。
#
# 档位：synth-flat（合成缩减表）/ real-full（真实表多槽全覆盖）/ real-miss（真实表越窗档）/
#       负向对照 3 条（表基址偏 1 行 / 不搬跨 step 状态 / 多槽路径行偏移错）。
# 退出码：0 = 全部符合预期；1 = 有档不符预期。
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
OUT=${1:-/tmp/m29_repro}
BUILD="$HERE/build"
PY=/usr/local/python3.12.13/bin/python3
LOCK=/tmp/npu0.lock
mkdir -p "$OUT"

echo "== [1/5] 构建（独立工程，不碰 m15_layer_loop/CMakeLists.txt）=="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B "$BUILD" -S "$HERE" -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1 || { tail -20 "$OUT/cmake.log"; exit 1; }
cmake --build "$BUILD" -j4 > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
echo "   built $BUILD/m29_ple_unit"

echo "== [2/5] 数据（synth / real / real_miss）=="
$PY "$HERE/gen_unit_data.py" --mode synth --out "$HERE/data_synth" > "$OUT/gen_synth.log" 2>&1 || { tail -20 "$OUT/gen_synth.log"; exit 1; }
$PY "$HERE/gen_unit_data.py" --mode real --out "$HERE/data_real" > "$OUT/gen_real.log" 2>&1 || { tail -20 "$OUT/gen_real.log"; exit 1; }
$PY "$HERE/gen_unit_data.py" --mode real --miss --out "$HERE/data_real_miss" > "$OUT/gen_real_miss.log" 2>&1 || { tail -20 "$OUT/gen_real_miss.log"; exit 1; }

FAIL=0

# 正档：设备 rc=0 且判据 rc=0
run_positive() {
    local name=$1 data=$2
    local od="$OUT/$name"; mkdir -p "$od"
    flock -w 300 "$LOCK" bash -c "
        cd '$od' || exit 3
        npu-smi info > npu_smi.txt 2>&1
        M29_OUT='$od' M29_DATA='$data' timeout 180 '$BUILD/m29_ple_unit'
    " > "$od/run.log" 2>&1
    local drc=$?
    $PY "$HERE/check_ref.py" "$od" "$data" > "$od/check.log" 2>&1
    local crc=$?
    echo "   [$name] dev rc=$drc  check rc=$crc  $(grep -o 'VERDICT=[A-Z]*' "$od/check.log" | tail -1)"
    { [ $drc -eq 0 ] && [ $crc -eq 0 ]; } || FAIL=1
}

# 负档：设备 rc=0 且判据 rc=1（必须变红）；envs = 逐字 env 赋值串
run_negative() {
    local name=$1 data=$2 envs=$3
    local od="$OUT/$name"; mkdir -p "$od"
    flock -w 300 "$LOCK" bash -c "
        cd '$od' || exit 3
        npu-smi info > npu_smi.txt 2>&1
        env M29_OUT='$od' M29_DATA='$data' $envs timeout 180 '$BUILD/m29_ple_unit'
    " > "$od/run.log" 2>&1
    local drc=$?
    $PY "$HERE/check_ref.py" "$od" "$data" > "$od/check.log" 2>&1
    local crc=$?
    echorc=$(grep -c 'FAIL' "$od/check.log")
    echo "   [$name] dev rc=$drc  check rc=$crc（期望 1，红档数=$echorc）"
    { [ $drc -eq 0 ] && [ $crc -eq 1 ]; } || FAIL=1
}

echo "== [3/5] 正档（synth 合成表 / real 真实表多槽全覆盖）=="
run_positive "synth"    "$HERE/data_synth"
run_positive "real"     "$HERE/data_real"

echo "== [4/5] 真实表越窗档（只 stage 1 个分片的槽 ⇒ 设备 miss 计数 > 0；covered 行仍须逐字节正确）=="
run_positive "real_miss" "$HERE/data_real_miss"

echo "== [5/5] 负向对照（必须变红）=="
run_negative "neg_table"     "$HERE/data_synth" "M29_NEG_TABLE=1"
run_negative "neg_state"     "$HERE/data_synth" "M29_NEG_STATE=1"
# 多槽路径负向对照（本次核心增量）：真实表 + 槽内行偏移错（M29_MUT bit14=0x4000）
run_negative "neg_slot_real" "$HERE/data_real"  "M29_MUT=16384"

echo "== 汇总：$( [ $FAIL -eq 0 ] && echo ALL-AS-EXPECTED || echo MISMATCH )  （日志在 $OUT/*/{run.log,check.log,npu_smi.txt}）=="
exit $FAIL
