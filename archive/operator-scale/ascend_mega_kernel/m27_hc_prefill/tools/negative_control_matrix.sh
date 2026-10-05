#!/usr/bin/env bash
# M119 负向对照矩阵 —— 一页看清"**哪个档在哪一级变红**"（零设备为主；设备档引归档读数）
#
#   bash m27_hc_prefill/tools/negative_control_matrix.sh > m27_hc_prefill/evidence/negative_control_matrix.log
#
# 动机（复审 r2 的 B1）：把重定位目标改成**位掩码**之后，`M27_MUTANT=norel` 映射为 `reloc_mask=0`
# ⇒ 判据按"调用方明确不要这些面"把它们记 SKIP。于是那个档从"值级负向对照"变成"**配置档**"
# （SKIP + 毒值见证），必须让读者**看得见**判别力去哪了。本矩阵给出：
#   · 每个档的 rc 由**谁**保证（判定项红？结构 guard 红？三态 SKIP？）
#   · 重定位路径的**值级**判别力现在由哪个档承担（`sinkreloc` 的判据半 + `rowsteal`）
# 每一步都打印命令与 rc（rc 单独取，不经管道，避免 `$?` 变成 tail 的 rc）。
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
E="$REPO/m27_hc_prefill/evidence"
PY=/usr/local/python3.12.13/bin/python3
cd "$REPO"

run_check() {  # run_check <dump 目录前缀> <标题>
    local d="$1" title="$2"
    $PY m27_hc_prefill/check_ref.py "$d" > /tmp/m27_mx_check.log 2>&1
    local rc=$?
    echo "   [$title] rc=$rc"
    grep -E "^\[chk\] 未过|^\[chk\] RESULT|^\[chk\] 计数" /tmp/m27_mx_check.log | tail -3
    echo "   判定项记 SKIP 的条数 = $(grep -cE 'SKIPPED（reloc_mask' /tmp/m27_mx_check.log)"
}

echo "== M119 负向对照矩阵（零设备可复算；逐条读数）=="
echo "   生成：bash m27_hc_prefill/tools/negative_control_matrix.sh"
echo

echo "### 0) 判据链路自检（合成 dump；不占设备）"
rm -rf /tmp/m27_mx1 /tmp/m27_mx2
$PY m27_hc_prefill/tools/selftest_dump.py /tmp/m27_mx1 m33 33 1 >/dev/null
run_check /tmp/m27_mx1 "正向：合成 dump（= 理想设备输出）⇒ 期望 rc=0"
$PY m27_hc_prefill/tools/selftest_dump.py /tmp/m27_mx2 m33 33 1 --mutate a1.blk >/dev/null
run_check /tmp/m27_mx2 "--mutate a1.blk ⇒ 期望 rc=1"
echo

echo "### 1) 三态：空目录"
rm -rf /tmp/m27_mx3 && mkdir -p /tmp/m27_mx3
$PY m27_hc_prefill/check_ref.py /tmp/m27_mx3 > /tmp/m27_mx3.log 2>&1
echo "   [空目录] rc=$?（期望 2）"; tail -1 /tmp/m27_mx3.log
echo

echo "### 2) **现形** norel（reloc_mask=0 + 四个产物毒值）= 配置档，不是值级负向对照"
echo "    谁保证 rc：→ 见下面未过项（本档四个产物记 SKIP，red 来自消费那些面的下游项）"
rm -rf /tmp/m27_mx4
$PY m27_hc_prefill/tools/selftest_dump.py /tmp/m27_mx4 m33 33 1 --reloc-mask 0 --poison-final >/dev/null
run_check /tmp/m27_mx4 "norel 现形（mask=0）"
grep -E "^\[chk\] guard a1\.(blk|injw|rstd) 未当" /tmp/m27_mx_check.log
echo

echo "### 3) \`sinkreloc\` 的**判据半**（reloc_mask=0xF + 四个产物毒值）⇒ 值级红（零设备可跑，就是本步）"
echo "    设备半（落点真被旁路到 scratch）本轮**未取得读数**（设备冻结；探锁返回忙）"
rm -rf /tmp/m27_mx5
$PY m27_hc_prefill/tools/selftest_dump.py /tmp/m27_mx5 m33 33 1 --reloc-mask 15 --poison-final >/dev/null
run_check /tmp/m27_mx5 "sinkreloc 判据半（mask=0xF）"
echo

echo "### 4) 设备档读数（归档；不是本轮跑的）"
echo "--- rowsteal（reloc_mask=0xF，块推进量少一行）"
grep -E "^\[chk\] 未过|RESULT" "$E/check_mutant_rowsteal.log" | tail -2
echo "--- norel **改前**（掩码引入之前：判据不按掩码 SKIP，36 条红的**历史**读数）"
grep -E "^\[chk\] 未过|RESULT" "$E/check_mutant_norel.log" | tail -2
echo "--- 主档（reloc_mask=0xF，真实权重）"
grep -E "计数|未过" "$E/check_main.log" | tail -2
echo
echo "### 5) 判别力账（重定位路径的**值级**红，按档数）"
echo "   改前 norel：36 条红，其中**四个产物的判定项 22 条**（值级）"
echo "   现形 norel：四个产物记 SKIP（7 条@m33）⇒ 该档对产物**没有值级红**；red 只剩下游 a2.hcp"
echo "   sinkreloc（新）：掩码不变 + 落点旁路 ⇒ 四个产物**恢复值级红**（判据半零设备可复现；设备半待解冻）"
echo "   rowsteal（不变）：10 条红里含 a1.blk / a1.ij_handoff / a1.rstd（值级），但它是扰动输入、不隔离重定位"
