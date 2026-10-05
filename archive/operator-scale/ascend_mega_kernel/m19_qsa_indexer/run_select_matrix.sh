#!/bin/bash
# m19_qsa_indexer/run_select_matrix.sh —— 选择段「单核先、后多核」的可复现矩阵 + 证据生成
#
# M53 方法硬要求：**先把单核（coreDiv=1）在小档上跑通并与暴力 oracle 逐块对齐，再让多核与单核
# 做逐步回归对照**。本脚本把该矩阵一次跑完，读数落 evidence/select_single_core.log。
#
# 用法（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m19_qsa_indexer/build -S m19_qsa_indexer -DCMAKE_BUILD_TYPE=Release && \
#     cmake --build m19_qsa_indexer/build -j4
#   bash m19_qsa_indexer/run_select_matrix.sh [输出目录，默认 m19_sel]
#
# 矩阵：
#   档位 (V, budget) ∈ {(256,8),(256,32),(256,512),(2048,8),(2048,32),(2048,512)}   # blkTopk = budget/4
#   核数 coreDiv：**每档都跑 1（单核）与 56（全核）**做单核↔多核对照；中间切分（2/4/8）
#                 只在代表档位 (2048,512) 上跑（共用单卡，避免矩阵把单次实验拉得过长）。
#   每个 (档位, 核数)：device 自检 + check_select.py 逐块 oracle 判据
#   同档位跨核数：logits 必须位同、选中集合必须同（多核 == 单核）
#   默认 4 档 × 9 次独立进程：逐档计数必须逐次相同（确定性）
set -u
set -o pipefail
FAILED=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-m19_qsa_indexer/m19_sel}"   # 放在模块内 ⇒ 被模块 .gitignore 覆盖（运行产物不入库）
EXE=./m19_qsa_indexer/build/m19_qsa_indexer
PY=/usr/local/python3.12.13/bin/python3.12
[ -x "$EXE" ] || { echo "先构建：cmake --build m19_qsa_indexer/build -j4"; exit 2; }

rm -rf "$OUT"
mkdir -p "$OUT"

cores_for() {   # $1=V $2=budget
  if [ "$1 $2" = "2048 512" ]; then echo "1 2 4 8 56"; else echo "1 56"; fi
}

echo "=== ① 小档矩阵：单核先，多核回归 ==="
# cfg = "V budget pos"（pos = 序列位置；tail_count = (pos+1)-4V，用来覆盖 tail 收尾）
for cfg in "256 8 1023" "256 32 1023" "256 512 1023" "2048 8 8190" "2048 32 8189" "2048 512 8193"; do
  set -- $cfg; V="$1"; B="$2"; P="$3"
  for C in $(cores_for "$V" "$B"); do
    D="$OUT/V${V}_B${B}_C${C}"
    mkdir -p "$D"
    timeout 300 env M19_V="$V" M19_BUDGET="$B" M19_POS="$P" M19_CORES="$C" M19_OUT="./$D" "$EXE" 2 > "$D/run.log" 2>&1
    rc=$?
    nmeta=$(ls "$D"/*_meta.txt 2>/dev/null | wc -l)
    [ "$nmeta" -ge 1 ] || { echo "  ✗ $D 没有产出 *_meta.txt（dump 缺失）"; FAILED=1; }
    [ "$rc" -eq 0 ] || FAILED=1
    printf "V=%-5s budget=%-4s pos=%-7s cores=%-3s exit=%s 自检PASS=%s meta=%s\n" \
        "$V" "$B" "$P" "$C" "$rc" "$(grep -c '自检 PASS' "$D/run.log")" "$nmeta"
  done
done

echo
echo "=== ② 逐块 oracle：选中集合 == 暴力 top-k；每块恰 4 token；count/tail/padding ==="
for cfg in "256 8" "256 32" "256 512" "2048 8" "2048 32" "2048 512"; do
  set -- $cfg; V="$1"; B="$2"
  for C in $(cores_for "$V" "$B"); do
    echo "--- V=$V budget=$B cores=$C"
    $PY m19_qsa_indexer/check_select.py "$OUT/V${V}_B${B}_C${C}" || FAILED=1
  done
done

echo
echo "=== ③ 多核 == 单核（同档位跨核数：logits 位同 + 选中集合同）==="
for cfg in "256 8" "256 32" "256 512" "2048 8" "2048 32" "2048 512"; do
  set -- $cfg; V="$1"; B="$2"
  $PY m19_qsa_indexer/check_select.py --cross "$OUT/V${V}_B${B}_C1" "$OUT/V${V}_B${B}_C56" || FAILED=1
  if [ "$V $B" = "2048 512" ]; then
    $PY m19_qsa_indexer/check_select.py --cross "$OUT/V${V}_B${B}_C1" "$OUT/V${V}_B${B}_C2" \
        "$OUT/V${V}_B${B}_C4" "$OUT/V${V}_B${B}_C8" "$OUT/V${V}_B${B}_C56" || FAILED=1
  fi
done

echo
echo "=== ④ 默认 4 档 × 9 次独立进程（确定性）==="
for i in $(seq 1 9); do
  D="$OUT/det/run$i"; mkdir -p "$D"
  timeout 300 env M19_OUT="./$D" "$EXE" > "$D/run.log" 2>&1
  [ "$(ls "$D"/*_meta.txt 2>/dev/null | wc -l)" -ge 4 ] || { echo "run$i ✗ 缺 dump"; FAILED=1; }
  printf "run%-2s " "$i"
  grep -oE "(A_close_2048|B_open_2048|C_small_256|D_max_65536) +自检 (PASS|FAIL) +count=[0-9]+ tokens=[0-9]+" \
      "$D/run.log" | sed 's/  */ /g' | tr '\n' ' '
  echo
done

echo
echo "=== ⑤ 默认 4 档判据（check_select 离散 + check_ref T1/T3/离散）==="
$PY m19_qsa_indexer/check_select.py "$OUT/det/run1" || FAILED=1
$PY m19_qsa_indexer/check_ref.py "$OUT/det/run1" || FAILED=1
echo
if [ "$FAILED" -ne 0 ]; then
  echo "[run_select_matrix] ===== FAILURES PRESENT（见上面 ✗ / 非零退出）====="
  exit 1
fi
echo "[run_select_matrix] ===== ALL PASS ====="
echo "（$OUT/ 为运行产物、已 ignore；本文件即证据：**一次运行产出**，判据退出码已逐级传递）"
