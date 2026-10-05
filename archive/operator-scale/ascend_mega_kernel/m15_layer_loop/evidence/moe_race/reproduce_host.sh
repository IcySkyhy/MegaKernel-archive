#!/bin/bash
# M105 · host 侧复现（干净树 → 借 include → 四档构建）。
# 这条序列是本 mission 实跑过的（读数见 ab_generator.log / runs_summary.txt）；
# 本脚本自身也在 2026-09-27 被整脚本重跑过一次（见 README §7 的「整脚本重跑」读数）。
#
# 用法：  bash reproduce_host.sh [WORKDIR]      # 默认 WORKDIR=/tmp/m105_repro
# 依赖：  /usr/local/Ascend/ascend-toolkit/set_env.sh（cmake + ASC 编译器）
set -euo pipefail
WT=$(cd "$(dirname "$0")/../../.." && pwd)       # = <wt-105>（本脚本在 <wt-105>/m15_layer_loop/evidence/moe_race/）
WORK=${1:-/tmp/m105_repro}
BASE_FLIP=c9f97e3      # M100 的 d63ef3a 的基座（M102 定的“翻面基座”）
BASE_MAIN=20bd20d      # 本 mission 的 base = 当时的 main

echo "WT=$WT  WORK=$WORK"
df -h / | tail -1

# ---- 1. 两棵干净树 × {未修, 修后} 四档 ----
rm -rf "$WORK"; mkdir -p "$WORK"
for d in c9_unf c9_fix mn_unf mn_fix; do mkdir -p "$WORK/$d"; done
( cd "$WT" && git archive "$BASE_FLIP" | tar -x -C "$WORK/c9_unf" )
( cd "$WT" && git archive "$BASE_FLIP" | tar -x -C "$WORK/c9_fix" )
( cd "$WT" && git archive "$BASE_MAIN" | tar -x -C "$WORK/mn_unf" )
( cd "$WT" && git archive "$BASE_MAIN" | tar -x -C "$WORK/mn_fix" )

# ---- 2. 「借 include」配方（M102 的复现配方；**无任何接线代码**）----
#   #include "m15_hc_layer.h"  →  #define main … / #include "m15_ple.asc" / #undef main
#   （m15_ple.asc 自己 include m15_hc_layer.h ⇒ 等价于「原来的 hc 段 + 整段 PLE device 代码」）
#   `Ctx C;` → `M15Run::Ctx C;`（借入后匿名 namespace 的 Ctx 会歧义）
python3 - "$WORK" <<'PYEOF'
import pathlib, sys
w = pathlib.Path(sys.argv[1])
for d in ("c9_unf", "c9_fix", "mn_unf", "mn_fix"):
    p = w / d / "m15_layer_loop" / "m15_layer_loop.asc"
    t = p.read_text()
    old = '#include "m15_hc_layer.h"      // hc 边界段 device 代码（复制改造自 m20，见 lift_hc_segment.py）\n'
    new = ('#define main m15_ple_standalone_main_unused\n'
           '#include "m15_ple.asc"\n'
           '#undef main\n')
    assert t.count(old) == 1, (d, t.count(old))
    t = t.replace(old, new)
    assert t.count("    Ctx C;\n") == 1, (d, "Ctx C;")
    t = t.replace("    Ctx C;\n", "    M15Run::Ctx C;\n")
    p.write_text(t)
    print("patched", p)
PYEOF

# ---- 3. 修后档换成本 mission 的生成物（未修档留基座原样的 m15_moe_layer.h）----
cp "$WT/m15_layer_loop/m15_moe_layer.h" "$WORK/c9_fix/m15_layer_loop/m15_moe_layer.h"
cp "$WT/m15_layer_loop/m15_moe_layer.h" "$WORK/mn_fix/m15_layer_loop/m15_moe_layer.h"
for d in c9_unf c9_fix mn_unf mn_fix; do sha256sum "$WORK/$d/m15_layer_loop/m15_moe_layer.h"; done

# ---- 4. 构建（-j32；host 侧，不占 NPU）----
source /usr/local/Ascend/ascend-toolkit/set_env.sh
for d in c9_unf c9_fix mn_unf mn_fix; do
  ( cd "$WORK/$d/m15_layer_loop"
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release > "$WORK/$d.cmake.log" 2>&1
    cmake --build build -j32 --target m15_layer_loop > "$WORK/$d.build.log" 2>&1 )
  echo "=== $d build rc=$? ==="
  sha256sum "$WORK/$d/m15_layer_loop/build/m15_layer_loop"
done

# ---- 5. 设备档（**不在这里跑**：每条设备命令各自 `flock -w 900 /tmp/npu0.lock` 排队）----
cat <<'EOD'
设备档命令（逐条；读数见本目录 runs_summary.txt / run_*.log）：
  # 本轮的纪律（r1 复审后收紧）：一次进锁只放一条**短**设备命令、锁内先复查 npu-smi、命令包 timeout
  flock -w 300 /tmp/npu0.lock <inner.sh> <tree>/m15_layer_loop/build/m15_layer_loop <tree>/m15_layer_loop/weights_manifest.txt
  # inner.sh 内：npu-smi 复查（有人在跑就退出、不自旋）→
  #   timeout 600 env M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 "$@" <bin> <manifest> all
带落盘档再加： M15_DUMP=1 M15_DUMP_MOE=0,1（或列更多层）
EOD
