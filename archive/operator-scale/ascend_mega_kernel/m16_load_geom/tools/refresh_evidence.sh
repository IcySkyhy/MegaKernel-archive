#!/usr/bin/env bash
# 刷新 evidence/ 归档 —— 一次性完成「跑 → 复制 → 重算校验和 → 校验」，避免归档与树脱节
# （round-2 review P2-1 的根因：树是绿的但归档是旧运行的日志）。
#
#   cd m16_load_geom && bash tools/refresh_evidence.sh
#
# 串行执行，约 6 分钟；**不要在别处同时跑任何 m16_* 程序**（并发占 NPU 会让读数不可信，见 reproduce.sh 注释）。
set -euo pipefail
cd "$(dirname "$0")/.."          # 工程根 = m16_load_geom/
PY=/usr/local/python3.12.13/bin/python3

mkdir -p evidence

echo "=== [1/4] 跑 reproduce.sh（其输出即归档用的 reproduce_run.log）==="
bash reproduce.sh 2>&1 | tee evidence/reproduce_run.log | tail -5

echo "=== [2/4] 复制 build/ 产物与日志进 evidence/ ==="
# 每个阶段的日志都由 reproduce.sh 自己 tee 到 build/<阶段>.log —— 归档里不会出现"脚本没生成过的日志"
cp build/geom2d_run1.log build/geom2d_run2.log build/geom2d_run3.log build/geom2d_run4.log build/geom2d_run5.log \
   build/geom2d_determinism.txt build/geom2d_isolated_run.txt build/geom3d.log \
   build/pv_synth.log build/pv_map.log build/pv_kmap.log build/pv_dump.log build/pv_real.log build/pv_baddst.log \
   build/check_ref_pv.log build/check_ref_geom2d.log build/check_ref_geom3d.log \
   evidence/
cp build/m16_geom_2d_raw_run1.bin build/m16_geom_2d_raw_run1.bin.tsv \
   build/m16_geom_3d_raw.bin build/m16_geom_3d_raw.bin.tsv \
   build/m16_pv_synth_c_device.bin build/m16_pv_synth_c_ref.bin \
   build/m16_pv_real_c_device.bin build/m16_pv_real_c_ref.bin \
   build/m16_pv_map_device.bin \
   evidence/

echo "=== [3/4] 重算 evidence/dump_sha256.txt ==="
$PY tools/make_sha256.py

echo "=== [4/4] 校验归档 ==="
sha256sum -c evidence/dump_sha256.txt | tail -3
echo "[refresh] done —— 归档与树已同步（reproduce_run.log 以 '[repro] done' 结尾）"
