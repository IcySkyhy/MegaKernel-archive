#!/usr/bin/env bash
# M147 / m9_gdn_prolog m>1 段体复算脚本
#
# 纪律：每档各自进一次 flock -w 300；进锁先 npu-smi（快照落 evidence/logs/）；
#       timeout 放在锁内；锁未取得时命令会直接失败，读不到 dump 就记「未取得读数」。
#
# 用法（在仓库根目录或本目录均可）：
#   bash m9_gdn_prolog/reproduce.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/evidence/logs"
mkdir -p "$LOGS"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cmake -B "$HERE/build" -S "$HERE" -DCMAKE_BUILD_TYPE=Release || exit 1
cmake --build "$HERE/build" -j4 || exit 1

cd "$HERE/build" || exit 1

run_one() {  # $1 = M9_ONLY 取值, $2 = 档名（日志文件名）
    echo "=== [$2] M9_ONLY=$1  $(date -Is) ==="
    flock -w 300 /tmp/npu0.lock bash -c "
        npu-smi info > '$LOGS/$2'_npu_smi.txt 2>&1
        M9_ONLY=$1 timeout 280 ./m9_gdn_prolog > '$LOGS/$2'_run.log 2>&1
        echo \"[$2] run rc=\$?\"
    "
}

# m=1 历史档（5 个 blockDim）+ m>1 档（各档各自进锁）
run_one m1  m1_all
run_one mt0 mt0_m1
run_one mt1 mt1_m4
run_one mt2 mt2_m65
run_one mt3 mt3_m257
run_one mt4 mt4_m65_b4
run_one mt5 mt5_m65_b16
run_one mt6 mt6_m65_mut1
run_one mt7 mt7_m257_mut2

# 判据（零设备）：默认档 + 两个反向对照（期望变红）
/usr/local/python3.12.13/bin/python3 "$HERE/check_ref.py"           | tee "$LOGS/check_default.log"
/usr/local/python3.12.13/bin/python3 "$HERE/check_ref.py" --mutant 1 | tee "$LOGS/check_mut1.log"
/usr/local/python3.12.13/bin/python3 "$HERE/check_ref.py" --mutant 2 | tee "$LOGS/check_mut2.log"
