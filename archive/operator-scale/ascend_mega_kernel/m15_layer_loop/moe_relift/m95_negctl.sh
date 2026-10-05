#!/usr/bin/env bash
# M95 的**常备负向对照**（第四/第五变体纪律：判据的「通过」必须难拿到）。
#
# 每条对照都把**被测对象**弄坏（不是改参考侧、不是改判据），然后跑 `check_stream.py`，
# **期望 rc=1**。其中 nc9a/nc9b/nc9c 是「注释绕过」对照：把破坏处的文本用注释补回去 ——
# 判据若扫**含注释的原文**就会被"打回绿"（r1 复审实测出过这个洞），所以这三条必须仍然红。
#
# 用法：
#   m15_layer_loop/moe_relift/m95_negctl.sh                        # 用 tip 的 check_stream.py
# 对照清单（13 条 = 12 条破坏 + 1 条健全性正对照；脚本结尾会自报这个数）：
#   · nc_row / nc_b6（资源表）、nc_tree / nc_a11 / nc_a12 / nc_a13（实现）、nc_a2（旧形态回归）；
#   · nc9a_com / nc9b_com / nc9c_com —— **注释绕过（正向：注释补回被匹配文本）**，必须仍红；
#   · nc11_survivor / nc11b_survivor —— **注释绕过（反向：块注释的 `*/` 与 `//` 同行 ⇒ 注释文本存活）**，
#     两趟扫描器下这两条会假绿（r2 复审实测），单趟扫描器下必须仍红；
#   · nc10_benign —— 正对照：无害块注释 ⇒ 读数不许变，rc 必须仍为 0。
#   m15_layer_loop/moe_relift/m95_negctl.sh --checker <path>       # 换 checker（复现修复前的假绿）
# 退出码：全部对照 rc=1 ⇒ 0；**任一条被绕过（rc=0）⇒ 1**（即「对照失效」本身就报错）。
#
# 所有变异只落在 `mktemp -d` 的 /tmp 副本里，worktree 不被触碰。
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
M15=$(cd "$HERE/.." && pwd)
REPO=$(cd "$M15/.." && pwd)
CHECKER="$HERE/check_stream.py"
if [ "${1:-}" = "--checker" ]; then
  CHECKER="$2"
fi
echo "# checker = $CHECKER"
echo "# 被测头文件 = $M15/m15_moe_layer.h + $M15/m15_moe_resources.h"
TMP=$(mktemp -d /tmp/m95_negctl.XXXXXX)
LAYER="$M15/m15_moe_layer.h"
RES="$M15/m15_moe_resources.h"

# 生成变异副本：$1 = 输出名，$2 = python 表达式片段（对文本做替换）
mutate() {
  local out="$1" patch="$2"
  python3 - "$LAYER" "$out" "$patch" <<'PY'
import pathlib, sys
src, out, patch = sys.argv[1], sys.argv[2], sys.argv[3]
t = pathlib.Path(src).read_text()
ns = {"t": t}
exec(patch, ns)
new = ns["t"]
assert new != t, "变异没有生效（目标串未命中）"
pathlib.Path(out).write_text(new)
PY
}

fails=0
n_break=0
n_pos=0
run() {   # $1 = 名字, $2 = 期望 rc, $3.. = checker 参数
  local name="$1" want="$2"; shift 2
  if [ "$want" = "1" ]; then n_break=$((n_break + 1)); else n_pos=$((n_pos + 1)); fi
  # PYTHONPATH 指向本目录：--checker 指向 /tmp 下的副本时，它 import 的 check_prep 仍能解析
  PYTHONPATH="$HERE:${PYTHONPATH:-}" LC_ALL=C python3 "$CHECKER" "$@" > "$TMP/$name.out" 2>&1
  local rc=$?
  local line
  line=$(grep -m1 -E '^\[check\] FAIL' "$TMP/$name.out" || true)
  if [ "$rc" = "$want" ]; then
    printf 'ok    %-8s rc=%s  %s\n' "$name" "$rc" "${line:-（无 FAIL 行）}"
  else
    printf 'BAD   %-8s rc=%s（期望 %s）  %s\n' "$name" "$rc" "$want" "${line:-（无 FAIL 行）}"
    fails=$((fails + 1))
  fi
}

# ---- 1. 资源表侧 ----
python3 - "$RES" "$TMP/res_row.h" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]).read_text()
q = p.replace("constexpr uint32_t RT_ROWL = RT_SORTLN;", "constexpr uint32_t RT_ROWL = 64;")
assert q != p
pathlib.Path(sys.argv[2]).write_text(q)
PY
python3 - "$RES" "$TMP/res_b6.h" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]).read_text()
q = p.replace("constexpr uint32_t RT_WROW = 64;", "constexpr uint32_t RT_WROW = 4096;")
assert q != p
pathlib.Path(sys.argv[2]).write_text(q)
PY
run nc_row   1 --header "$LAYER" --res "$TMP/res_row.h"
run nc_b6    1 --header "$LAYER" --res "$TMP/res_b6.h"

# ---- 2. 生成物侧：破坏实现 ----
mutate "$TMP/h_tree.h"   't = t.replace("        MergeTree();", "        // [NC] MergeTree() 调用被删")'
mutate "$TMP/h_a11.h"    't = t.replace("UB_RT_LOG + r * RT_ROWL * 4) + e0;", "UB_RT_LOG + r * 512 * 4) + e0;")'
mutate "$TMP/h_a12.h"    't = t.replace("for (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {\n                RegTensor<int32_t> rg;", "for (uint16_t c = 0; c < 1; ++c) {\n                RegTensor<int32_t> rg;")'
mutate "$TMP/h_a13.h"    't = t.replace("        for (uint32_t g = 0; g < 4; ++g) {   // level2", "        for (uint32_t g = 0; g < 0; ++g) {   // level2")'
run nc_tree  1 --header "$TMP/h_tree.h" --res "$RES"
run nc_a11   1 --header "$TMP/h_a11.h" --res "$RES"
run nc_a12   1 --header "$TMP/h_a12.h" --res "$RES"
run nc_a13   1 --header "$TMP/h_a13.h" --res "$RES"

# ---- 3. 「注释绕过」对照：破坏实现 + 用注释把文本补回去（必须仍红）----
#    nc9a：A12 的 chunk 循环文本（跨行 ⇒ 用块注释）
mutate "$TMP/h_nc9a.h" 't = t.replace("for (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {\n                RegTensor<int32_t> rg;", "for (uint16_t c = 0; c < 1; ++c) {\n                RegTensor<int32_t> rg;") + "\n/* 注释绕过对照（A12）：\nfor (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {\n                RegTensor<int32_t> rg;\n*/\n"'
#    nc9b：A11 的行距文本（单行 ⇒ 行注释）
mutate "$TMP/h_nc9b.h" 't = t.replace("UB_RT_LOG + r * RT_ROWL * 4) + e0;", "UB_RT_LOG + r * 512 * 4) + e0;") + "\n// 注释绕过对照（A11）：UB_RT_LOG + r * RT_ROWL * 4\n"'
#    nc9c：删掉 MergeTree 真定义、把签名留在注释里
mutate "$TMP/h_nc9c.h" 't = t.replace("    __aicore__ inline void MergeTree()\n", "/* 注释绕过对照（A10）：__aicore__ inline void MergeTree() */\n")'
run nc9a_com  1 --header "$TMP/h_nc9a.h" --res "$RES"
run nc9b_com  1 --header "$TMP/h_nc9b.h" --res "$RES"
run nc9c_com  1 --header "$TMP/h_nc9c.h" --res "$RES"

# ---- 4. A2 的非空洞性（旧形态回归到代码里必须被咬住） ----
mutate "$TMP/h_a2.h" 't = t.replace("        LoadWRow(0, 0);", "        PrecastWeights();\n        LoadWRow(0, 0);")'
run nc_a2     1 --header "$TMP/h_a2.h" --res "$RES"

# ---- 4b. 「反方向」注释绕过对照（r2 复审实测的洞：块注释的 `*/` 与 `//` 同行）----
#   两趟扫描器（先 `//` 后 `/*...*/`）会把 `*/` 与 `//` 同行时的那一趟吃掉 `*/`，留下未闭合 `/*`
#   ⇒ 块注释那趟什么都不删 ⇒ **注释文本残留进代码视图** ⇒ 破坏实现 + 补一条注释又能拿到 PASS。
#   单趟扫描器（check_prep.strip_comments，r3）必须让这两条仍红。
mutate "$TMP/h_nc11.h" 't = t.replace("UB_RT_LOG + r * RT_ROWL * 4) + e0;", "UB_RT_LOG + r * 512 * 4) + e0;") + "\n/* 反方向对照（A11）：\nUB_RT_LOG + r * RT_ROWL * 4\nclose // */\n"'
mutate "$TMP/h_nc11b.h" 't = t.replace("    __aicore__ inline void MergeTree()\n", "/* 反方向对照（A10）：\n    __aicore__ inline void MergeTree()\nclose // */\n")'
run nc11_survivor  1 --header "$TMP/h_nc11.h" --res "$RES"
run nc11b_survivor 1 --header "$TMP/h_nc11b.h" --res "$RES"

# ---- 5. 剥注释器的**健全性**正对照：加一条无害块注释 ⇒ 判据读数不许变（仍 rc=0） ----
#      若剥注释器会把 `/*` 当块注释起点一路吞掉代码（check_prep.strip_comments 在本段生成物上就会），
#      这条会**假红**；`code_view`（先按行去 `//` 再去 `/*...*/`）必须让它保持 rc=0。
mutate "$TMP/h_nc10.h" 't = t + "\n/* 健全性正对照（nc10）：无害块注释\n   UB_RT_LOG + r * RT_ROWL * 4 的说明文本\n*/\n"'
run nc10_benign 0 --header "$TMP/h_nc10.h" --res "$RES"

echo "# 对照总数 = $((n_break + n_pos))（破坏 $n_break 条，健全性正对照 $n_pos 条）"
echo "# 副本目录：$TMP"
if [ "$fails" -eq 0 ]; then
  echo "# 全部对照按预期变红（对照有效）"
  exit 0
fi
echo "# 有 $fails 条对照**没有**变红 ⇒ 判据在那条路径上可被绕过"
exit 1
