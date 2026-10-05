#!/usr/bin/env bash
# [M129] 一键重建本目录全部读数（在仓根跑；**零设备**）：bash docs/evidence/quote_refs/repro.sh
#
# 工具本体 = docs/scan_quote_refs.py（本 mission 新增）。所有读数由**同一次运行、同一套匹配器**产出
# （docs/17 §9.2 规则②第 1 条）。改前/改后这一栏没有 —— 本工具是本 mission 从零建的，不存在「旧版」，
# 故只落当前读；语义对照由 `--selftest` 的 14 条正/负/盲区对照承担。
set -u
cd "$(dirname "$0")/../../.."
E=docs/evidence/quote_refs

# ---- 1. 工具层自检：12 条对照（正、负、M125 真实回归、盲区登记）----
{
  echo "# [M129] scan_quote_refs.py --selftest（工具层权威）"
  echo "# 命令：cd <repo root> && python3 docs/scan_quote_refs.py --selftest ; echo rc=\$?"
  echo
  echo "\$ python3 docs/scan_quote_refs.py --selftest"
  python3 docs/scan_quote_refs.py --selftest; echo "rc=$?"
} > "$E/selftest.log" 2>&1

# ---- 2. 主扫描（语料层）：汇总 + finding/分类 ----
{
  echo "# [M129] docs 引文保真主扫描（语料层）"
  echo "# 命令：cd <repo root> && python3 docs/scan_quote_refs.py ; echo rc=\$?"
  echo
  echo "\$ python3 docs/scan_quote_refs.py"
  python3 docs/scan_quote_refs.py; echo "rc=$?"
} > "$E/scan.log" 2>&1

# ---- 3. 逐条：finding + unverified + 全部成对样本（报告 (a) 的原始素材）----
{
  echo "# [M129] docs 引文保真扫描 —— 逐对明细（--dump-pairs）"
  echo "# 命令：cd <repo root> && python3 docs/scan_quote_refs.py --dump-pairs ; echo rc=\$?"
  echo
  echo "\$ python3 docs/scan_quote_refs.py --dump-pairs"
  python3 docs/scan_quote_refs.py --dump-pairs; echo "rc=$?"
} > "$E/scan_pairs.log" 2>&1

# ---- 4. `.tower/` 路径引用清单（报告 (b)）----
{
  echo "# [M129] docs 里 \`.tower/\` 路径引用清单（事实；交付后可达性由塔裁决）"
  echo "# 命令：cd <repo root> && python3 docs/scan_quote_refs.py --tower-refs"
  echo "# 说明：\`.tower/**\` 按主检出根解析（git worktree list 的第一项）—— 本仓 \`git ls-files .tower\` 的输出行数为 0。"
  echo
  echo "\$ python3 docs/scan_quote_refs.py --tower-refs"
  python3 docs/scan_quote_refs.py --tower-refs
} > "$E/tower_refs.md" 2>&1

echo "wrote: $(cd "$E" && ls -1 *.log tower_refs.md | tr '\n' ' ')"
