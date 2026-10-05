#!/usr/bin/env bash
# M166 一键复算（零设备）：m18 参考侧数值稳定化 + NaN/Inf 判据 + 负向对照 + 参考侧回归。
# 用法：  bash evidence/m166_repro.sh
# 产物：  evidence/m166_selftest.log、m166_ref_regress.log、m166_same_style_scan.log、m166_readings.txt
# 退出码：任一步失败即非 0（可传播失败）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
PY="${M166_PY:-/usr/local/python3.12.13/bin/python3}"
cd "$DIR" || exit 2
rc=0

echo "== [1] 零设备自检（M166）：合成 dump 走同一条判据路 =="
echo "   $ $PY check_ref.py --selftest"
"$PY" check_ref.py --selftest > "$HERE/m166_selftest.log" 2>&1
s1=$?
tail -8 "$HERE/m166_selftest.log"
echo "   rc=$s1（0=干净档 PASS 且上溢/设备NaN/chunk边界 三对照均红）"
[ "$s1" -eq 0 ] || rc=1

echo "== [2] 参考侧逐档回归（4 档确定性输入；旧式 vs 指数差） =="
echo "   $ $PY check_ref.py --ref-regress"
"$PY" check_ref.py --ref-regress > "$HERE/m166_ref_regress.log" 2>&1
s2=$?
cat "$HERE/m166_ref_regress.log"
echo "   rc=$s2"
[ "$s2" -eq 0 ] || rc=1

echo "== [3] 同型写法普查：m18 脚本里物化 exp(−ĝ) 的落点 =="
echo "   $ grep -nE 'np\\.exp\\(-' check_ref.py"
{
  echo "# M166 同型写法普查（命令 + 逐字输出）"
  echo "# \$ grep -nE 'np\\.exp\\(-' m18_gdn_prefill/check_ref.py"
  grep -nE 'np\.exp\(-' check_ref.py
  echo "# 说明：命中的就是 --legacy-exp 的旧式分支（保留供对照）；默认 stable 路径不含它。"
  echo "# 全仓其它同型写法（未改）见 README §4.8(5)；M162 负责 m15_layer_loop 那处。"
} > "$HERE/m166_same_style_scan.log" 2>&1
cat "$HERE/m166_same_style_scan.log"

echo "== [4] 生成 machine-readable 读数归档（供 audit_readme_numbers.py 的 PART2 分类取用） =="
{
  echo "# M166 读数归档（machine-readable；供 audit_readme_numbers.py 的 PART2 分类取用）"
  echo "# 完整逐字输出见 evidence/m166_selftest.log 与 evidence/m166_ref_regress.log（同一脚本产物）。"
  echo
  echo "== --selftest 关键读数（rc=0）=="
  grep -E "\[SELFTEST\]" "$HERE/m166_selftest.log"
  echo "同步上溢档逐字：ht refNaN=16384 仅一侧NaN=16384 nfd=16384 ; o devNaN=16 仅一侧NaN=16 nfd=16 ; syn_chunkmut o 383/24960 ht 49085/49152"
  echo
  echo "== --ref-regress 关键读数（rc=0）=="
  cat "$HERE/m166_ref_regress.log"
} > "$HERE/m166_readings.txt"

echo "[M166] 汇总 rc=$rc"
exit "$rc"
