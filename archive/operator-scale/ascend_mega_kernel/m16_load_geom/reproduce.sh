#!/usr/bin/env bash
# M27（m16_load_geom）一键复现：构建 → 2D 探针 ×5（确定性）→ 3D 探针 → BMM2 全模式
#                            → 独立 numpy 复核 → 隔离运行判定实验 → 归档校验和（信息性）。
#
# 注意：
#   * 约 6 分钟，且**必须单进程串行**——多个进程同时占用 NPU 会互相影响读数（本 mission 踩过一次）。
#   * 本脚本只写 build/（每个阶段的输出同时 tee 到 build/<阶段>.log）与 stdout；
#     要连 evidence/ 归档一起刷新（跑 + 复制 + 重算校验和 + **强制校验**），用
#     `tools/refresh_evidence.sh`。
set -euo pipefail
cd "$(dirname "$0")"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

PY=/usr/local/python3.12.13/bin/python3      # 干净 shell 里的 python3 没有 numpy
$PY -c "import numpy; print('[repro] numpy', numpy.__version__)"

rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cd build

echo "=== [1/10] 2D 标定探针 ×5（49 conf/次；逐 conf 一次 launch + launch 前清零 L0B）==="
# 第 1 次的 dump 归档为 m16_geom_2d_raw_run1.*（与 geom2d_run1.log 同一次运行 ⇒ 命名与内容一致）
./m16_geom 2d > geom2d_run1.log 2>&1
cp m16_geom_2d_raw.bin m16_geom_2d_raw_run1.bin
cp m16_geom_2d_raw.bin.tsv m16_geom_2d_raw_run1.bin.tsv
for r in 2 3 4 5; do ./m16_geom 2d > geom2d_run$r.log 2>&1; done
$PY ../tools/check_determinism.py          # 生成 geom2d_determinism.txt

echo "=== [2/10] 3D 探针（29 conf；期望：回读窗口内未观察到写入）==="
./m16_geom 3d 2>&1 | tee geom3d.log

echo "=== [3/10] BMM2 合成数据（期望 0/4096 逐位不等）==="
./m16_pv 2>&1 | tee pv_synth.log

echo "=== [4/10] BMM2 map（P=I ⇒ 期望解码命中 4096/4096）==="
./m16_pv map 2>&1 | tee pv_map.log

echo "=== [5/10] BMM2 kmap（K 轴探针；期望 C=1+kOff）==="
./m16_pv kmap 2>&1 | tee pv_kmap.log

echo "=== [6/10] BMM2 dump（合成数据落盘，供 check_ref.py 复核）==="
./m16_pv dump 2>&1 | tee pv_dump.log
cp m16_pv_c_device.bin m16_pv_synth_c_device.bin
cp m16_pv_c_ref.bin m16_pv_synth_c_ref.bin

echo "=== [7/10] BMM2 baddst 反例（期望 device 报错 507015）==="
if ./m16_pv baddst 2>&1 | tee pv_baddst.log; then
    echo "[repro][WARN] baddst 竟然成功了 —— 对照失效，请检查"
else
    echo "[repro] baddst 如期失败（device 报错 507015）"
fi

echo "=== [8/10] BMM2 M24 真实 dump（期望 0/4096 逐位不等 + 逐位一致）==="
./m16_pv real ../evidence/m24_s256_P_unit0_par0.bin ../evidence/m24_s256_V_n2_0.bin 2>&1 | tee pv_real.log
cp m16_pv_c_device.bin m16_pv_real_c_device.bin
cp m16_pv_c_ref.bin m16_pv_real_c_ref.bin

echo "=== [9/10] 独立 numpy 复核 ==="
$PY ../check_ref.py pv 2>&1 | tee check_ref_pv.log
$PY ../check_ref.py geom2d m16_geom_2d_raw_run1.bin m16_geom_2d_raw_run1.bin.tsv 2>&1 | tee check_ref_geom2d.log
$PY ../check_ref.py geom3d m16_geom_3d_raw.bin m16_geom_3d_raw.bin.tsv 2>&1 | tee check_ref_geom3d.log

echo "=== [10/10] P2-2 判定实验（单 conf 隔离运行；约 3 分钟）==="
$PY ../tools/check_isolated.py 2>&1 | tee geom2d_isolated_run.txt

echo "=== 归档数据产物校验和（信息性）==="
# dump_sha256.txt 只覆盖数据产物（切片/raw dump/sidecar/.bin），不含 log；路径相对工程根。
# 这里只做信息性检查：本脚本只写 build/、不改 evidence/，所以"归档没刷新"会出现不匹配
# ——那不是本脚本失败，而是归档落后于树；**强制校验**放在 tools/refresh_evidence.sh 里。
if (cd .. && sha256sum -c evidence/dump_sha256.txt >/dev/null 2>&1); then
    echo "[repro] 归档数据产物校验和 OK"
else
    echo "[repro][WARN] 归档数据产物校验和不匹配 —— 归档未刷新；用 tools/refresh_evidence.sh"
fi
echo "[repro] done"
