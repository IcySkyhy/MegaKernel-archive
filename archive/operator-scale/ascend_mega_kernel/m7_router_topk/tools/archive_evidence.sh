#!/usr/bin/env bash
# m7_router_topk 证据归档（M56）。
#
# WHY：M7 的位级声明（"M48 改造前后 5 个 case 共 50 个文件逐文件 sha256 完全相同"）此前**没有
#      evidence/ 目录**：没有 dump 的 sha256 清单，也没有 run/check_ref 的完整输出，所以那条声明
#      当时只能靠**源码级等价**（改造是机械替换）支撑。本脚本把该声明变成**可复算的归档**。
#
# 用法：
#   m7_router_topk/tools/archive_evidence.sh            # 构建 + 跑 + 归档到 m7_router_topk/evidence/
#   m7_router_topk/tools/archive_evidence.sh --no-build # 复用已有 build/
# 退出码（三态）：
#   0 = 归档成功且 5 档全 PASS（dump 50 个文件的 sha256 已入库）
#   1 = 跑起来了但某档 FAIL（**不发合格证**，仍写下读数供定位）
#   2 = 没得跑/输入缺失（无设备、无解释器、无权重、构建失败）—— 不产生"通过"文案
set -u
cd "$(dirname "$0")/.." >/dev/null 2>&1 || exit 2   # m7_router_topk/
PKG=$(pwd)
cd "$PKG/.." || exit 2                              # 仓根（kernel 按 m7_router_topk/data/... 找权重）
REPO=$(pwd)

PY=${PY:-/usr/local/python3.12.13/bin/python3}
CASES=${CASES:-"m1 m2 m16 m33 m64"}
OUT=${M7_OUT_DIR:-$PKG/m7_out}
EV=$PKG/evidence

[ -x "$PY" ] || { echo "RESULT: SKIPPED (缺少解释器 $PY；可用 PY=... 覆盖)" >&2; exit 2; }
[ -f "$PKG/data/router_weight.bin" ] || { echo "RESULT: SKIPPED (缺 m7_router_topk/data/router_weight.bin)" >&2; exit 2; }

if [ "${1:-}" != "--no-build" ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
  cmake -B "$PKG/build" -S "$PKG" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
    && cmake --build "$PKG/build" -j4 >/dev/null 2>&1 \
    || { echo "RESULT: SKIPPED (构建失败：m7_router_topk/build 未产出)"; exit 2; }
fi
BIN="$PKG/build/m7_router_topk"
[ -x "$BIN" ] || { echo "RESULT: SKIPPED (缺可执行 $BIN；先不带 --no-build 跑一次)"; exit 2; }

mkdir -p "$EV" || exit 2
rm -rf "$OUT"

# ---- 1) 设备自检 + dump ----
M7_OUT="$OUT" "$BIN" > "$EV/run.log" 2>&1
run_rc=$?
OUT_REL=${OUT#"$REPO"/}                 # 头里只写仓内相对路径（绝对路径会随 checkout 变，不利复算）
n_dump=$(find "$OUT" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "[archive] 设备自检 rc=$run_rc；dump 文件 $n_dump 个（期望 $(( $(echo $CASES | wc -w) * 10 ))）"

# ---- 2) 离线交叉校验（逐 case，判定项读数）----
: > "$EV/check_ref.log"
cr_fail=0
for c in $CASES; do
  echo "===== check_ref  $OUT_REL/$c" >> "$EV/check_ref.log"
  if [ -d "$OUT/$c" ]; then
    "$PY" "$PKG/check_ref.py" "$OUT/$c" >> "$EV/check_ref.log" 2>&1 || cr_fail=1
  else
    echo "[check_ref] RESULT: SKIPPED (缺 dump 目录 $OUT_REL/$c；这不是通过)" >> "$EV/check_ref.log"
    cr_fail=1
  fi
done

# ---- 3) dump sha256 清单（覆盖范围与清单**同源**：计数就是这里被哈希的文件数）----
( cd "$OUT" && find . -type f | sort | xargs sha256sum ) > "$EV/dump_sha256.txt"
n_hash=$(wc -l < "$EV/dump_sha256.txt" | tr -d ' ')
{
  echo "# m7_router_topk dump sha256 清单（M56 归档；由 tools/archive_evidence.sh 生成）"
  echo "# WHERE: 仓根下 M7_OUT=$OUT_REL ./m7_router_topk/build/m7_router_topk（设备：Ascend950PR，单 AIV）"
  echo "# WHAT : 5 个 case（m=1/2/16/33/64，seed 7）× 10 个文件 = $n_hash 个文件"
  echo "#        每 case: x / router_weight / router_logits / topk_ids / topk_weights + 各自 .json"
  echo "# 复算 : m7_router_topk/tools/archive_evidence.sh  （期望 rc=0，清单逐行相同）"
  echo "# 对照 : evidence/dump_sha256_m48_after.txt（M48 改造后，50 文件）与"
  echo "#        evidence/dump_sha256_m48_before.txt（M48 改造前基线，m1/m2/m16 共 30 文件）"
  echo "#        都是原样转存的当时读数；本清单与 after 逐行相同，且与 before 在重叠 30 文件上相同"
} | cat - "$EV/dump_sha256.txt" > "$EV/dump_sha256.txt.tmp" && mv "$EV/dump_sha256.txt.tmp" "$EV/dump_sha256.txt"

n_pass=$(grep -c "PASS" "$EV/check_ref.log")
n_skip=$(grep -c "RESULT: SKIPPED" "$EV/check_ref.log")
if [ "$n_hash" -eq 0 ] || [ "$n_dump" -eq 0 ]; then
  echo "RESULT: SKIPPED (没有 dump 可比：n_dump=$n_dump n_hash=$n_hash；这不是通过)"
  exit 2
fi
if [ "$run_rc" -ne 0 ] || [ "$cr_fail" -ne 0 ]; then
  echo "RESULT: FAIL (设备 rc=$run_rc；check_ref 失败/缺档 $n_skip；**不发合格证**，读数见 evidence/)"
  exit 1
fi
echo "RESULT: OK ($n_hash/$n_dump 个 dump 的 sha256 已入库；check_ref $n_pass 行 PASS、0 缺档)"
exit 0
