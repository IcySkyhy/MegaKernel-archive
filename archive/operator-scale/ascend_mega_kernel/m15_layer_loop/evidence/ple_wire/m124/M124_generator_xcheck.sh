#!/usr/bin/env bash
# M124：生成器 ↔ 产物的**双向交叉对拍**（形态照 M105 已受审的先例）
#
# 被验的两件：
#   · 生成器 `m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py`（本 mission **未改**）
#   · 源     `m15_layer_loop/m15_ple.asc`（本 mission 改了 ⇒ 它是唯一的自变量）
#   · 产物   `m15_layer_loop/m15_ple_wire.h`（机械生成物，hand-edit 必被 `--check` 判红）
#
# 判据（M105 先例 = `reviewer-m105` r1 的验收项，逐字「生成器↔产物双向交叉对拍：新生成器放旧树→
#   新 artifact、旧生成器放新树→旧 artifact ⇒ 产物差异完全由生成器产生」+「两棵树 --check rc=0」）：
#   ① 2×2 四条腿的产物 sha256 分别等于**对应源**的归档值：
#        oldgen+oldsrc → 旧 main（`M124_BASE_REV`）的 `m15_ple_wire.h` sha256
#        newgen+newsrc → 本 tip 的 `m15_ple_wire.h` sha256
#        oldgen+newsrc / newgen+oldsrc → 同上两者（⇒ 产物只是**源**的函数，与生成器无关）
#   ② 四条腿的 `--check` 全部 rc=0；
#   ③ 负向对照：手改产物一个 token ⇒ `--check` 必须 rc=1（「产物非手改」的咬合力）；
#   ④ 生成器在区间内 sha256 未变（`--diff` 打印），源在区间内 sha256 变了（⇒ 差异只可能来自源）。
#
# 用法（仓库根目录）：
#   bash m15_layer_loop/evidence/ple_wire/m124/M124_generator_xcheck.sh
#   M124_BASE_REV=<rev>（默认 9296e79 = M124 开工时的 main）
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO" || exit 1
BASE_REV="${M124_BASE_REV:-9296e79}"
GEN=m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py
SRC=m15_layer_loop/m15_ple.asc
ART=m15_layer_loop/m15_ple_wire.h
X=${M124_XCHECK_DIR:-/tmp/m124_xcheck.sh.$$}

echo "== 0. 区间内的源与生成器 =="
echo "BASE_REV=$BASE_REV"
echo "old gen sha256: $(git show "$BASE_REV:$GEN" | sha256sum | cut -d' ' -f1)"
echo "new gen sha256: $(sha256sum "$GEN" | cut -d' ' -f1)"
echo "old src sha256: $(git show "$BASE_REV:$SRC" | sha256sum | cut -d' ' -f1)"
echo "new src sha256: $(sha256sum "$SRC" | cut -d' ' -f1)"
echo "老产物归档 sha256: $(git show "$BASE_REV:$ART" | sha256sum | cut -d' ' -f1)"
echo "新产物归档 sha256: $(sha256sum "$ART" | cut -d' ' -f1)"

echo
echo "== 1. 2×2 交叉对拍 =="
for gen in old new; do
  for src in old new; do
    d="$X/${gen}gen_${src}src"
    rm -rf "$d"; mkdir -p "$d/m15_layer_loop/evidence/ple_wire"
    if [ "$src" = old ]; then git show "$BASE_REV:$SRC" > "$d/$SRC"; else cp "$SRC" "$d/$SRC"; fi
    if [ "$gen" = old ]; then git show "$BASE_REV:$GEN" > "$d/$GEN"; else cp "$GEN" "$d/$GEN"; fi
    ( cd "$d" && python3 "$GEN" > gen.log 2>&1; echo "gen_rc=$?" >> gen.log
      python3 "$GEN" --check > check.log 2>&1; echo "check_rc=$?" >> check.log ) || true
    printf "%-16s gen_rc=%s check_rc=%s artifact=%s\n" \
      "${gen}gen_${src}src" \
      "$(grep -o 'gen_rc=[0-9]*' "$d/gen.log" | cut -d= -f2)" \
      "$(grep -o 'check_rc=[0-9]*' "$d/check.log" | cut -d= -f2)" \
      "$(sha256sum "$d/$ART" | cut -d' ' -f1)"
  done
done

echo
echo "== 2. 负向对照：手改产物一个 token ⇒ --check 必须 rc=1 =="
d="$X/newgen_newsrc"
sed -i 's/namespace M85P {/namespace M85Q {/' "$d/$ART"
( cd "$d" && python3 "$GEN" --check; echo "negctl_rc=$?" )
cp "$SRC" "$d/$SRC"
( cd "$d" && python3 "$GEN" >/dev/null 2>&1 && python3 "$GEN" --check; echo "restored_check_rc=$?" )

echo
echo "（证据归档 = M124_generator_xcheck.log，由本脚本的输出重定向而成）"
