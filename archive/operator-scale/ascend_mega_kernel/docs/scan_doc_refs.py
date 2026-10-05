#!/usr/bin/env python3
"""docs 引用完整性扫描器 —— `docs/17-verification-standard.md` §10 的可跑附件（M55 交付）。

## 它真正检查什么（覆盖范围，与匹配器同源 —— docs/17 §9.2 第 1 条）

| 栏 | 内容 |
|---|---|
| **覆盖** | `docs/*.md` 里的三类引用：① `docs/NN` / `docs/NN-name.md`（doc→doc 定位；**不受包裹符影响** —— `DOCREF` 里没有"路径必须紧邻某字符"的约束，`` `docs/17` `` / `**docs/17**` 这类写法本来就照匹配，M78 故未动它）；② `docs/NN §X`（章节点位；**实际匹配面 = `docs/NN` 与 `§` 之间只隔空白、「的」、以及一个可选反引号** —— M78 起容忍反引号包裹；此形态在 M78 之前**一条都不匹配**，读数与重基线见下面「M78 放宽匹配面」段；**判定（目标标题那一侧）M114 起容忍前导 `§`** —— `### §3.2 …` 与 `### 3.2 …` 同等对待，见下面「M114 放宽判定面」段）；③ 本仓路径式文件引用（`m<编号>_*/`、`tools/`、`baseline_env/` … 仅判"路径是否存在"） |
| **未覆盖（unverified）** | ① 各模块 `m*/README.md`（不在 docs 扫描面上）；② **外部快照**（`/workspace/vllm/**`、`ops-*` donor、CANN/Bisheng 头）—— 其中呈**本仓路径形态**的那些，必须逐条登记在下面的 `ALLOWLIST` 里才不 gate；③ 行号引用（由 docs/17 §8.2 单独处置）；④ **裸 `§`-号引用**（`§X` 前不带 `docs/NN`）—— `SECREF` 要求 `docs/NN` 紧邻 `§`，故两个子形态都**不在覆盖面内**：**(a) 同文档内**的错引（`docs/17` 里写「见本文 §99.7」这种）；**(b) 省略 `docs/NN` 前缀的跨文档引用**（实例：`docs/05:379` 的「`docs/17` §9.5、§8.1」—— 第二个 `§8.1` 前没有 `docs/NN`；finding `20260926-reviewer-m63-…` 当年举的是同一行的 `§9.1`，那处**已随 `b158e49` 改为 `§8.1`**，形态不变）。理由见下 |
| **未覆盖（M78 追加登记）** | ⑤ **路径与 `§` 之间夹了别的 token**（版本号那种：`docs/05-megakernel-design.md v1.2 §2/§6`）—— 本轮**刻意不**放宽到这里（再放宽就滑向"任意位置都算引用"），故它**仍不被匹配**。M78 在 `08df83e` 上实测 **2 处**（**2 个文件的抬头行各 1 行**：`docs/09-moe-donor-map.md:4` 与 `docs/11-attn-analysis.md:4`，两行都是 `docs/05-megakernel-design.md v1.2 §2/§6`，两个目标在 `docs/05` 里都存在；M83 在本 mission base `8e496c9` 上复跑同为 2 处）；`「」`/`'`/`**` 等**其它包裹符**的写法实测 **0 处** ⇒ 本轮只加反引号这一种包裹符，不加**没有样本**的字符类 |
| **out-of-range** | 指向不存在目标的引用。**三类引用全部 gate**：逐条打印并**计入 `RESULT` 与退出码**。唯一的例外是 `ALLOWLIST` 条目 —— 它们**单列打印**（`allowlisted` + 依据）并且**不静默** |

`docs/17-verification-standard.md` **默认被排除**：该文件 §10 自身引述了待查编号（`docs/07`、`docs/08`）
与自检样本，若把它算进来，报出的 out-of-range 全部来自它自己。用 `--include-self` 可复现"含自引"的读数。

**汇总里每一栏都带分母**（§9.2 第 2 条的三态分栏；分母与上面的匹配器**同源**）：
`doc refs=NN (unresolved=M)`（doc→doc 定位数，与章节点位两栏不重叠）/ `section refs=NN (unresolved=M)` /
`in-repo path refs=NN  gating=G (distinct=…)  allowlisted=A (distinct=…)`。
「查过多少」与「命中多少」由**同一次运行**产出，故不存在「一份正则匹配、另一份手写数字」（§9.2 第 1 条）。

**分子与分母同源（M63 r1 修正，P2-2）**：`doc refs` 的分母**不含**带 `§` 的那种 `docs/NN`，
因此它的分子 `unresolved=M` 也**只数不带 `§` 的** `DOCREF` 命中（`scan_docrefs()` 与 `docref_counts()`
用**同一份 `_sec_starts()`** 划分子集）。带 `§` 的坏引用由 `section refs` 那一栏登记 ——
于是 **①比例不可能 >1，②一条坏引用不会被数成两条**（此前它同时进 `doc` 与 `sec`，正是 §9.2 第 1 条
要治的「分母与分子不同源」）。

## 三态退出码（docs/17 §8.3）与「谁是权威」

**为什么「裸 `§`-号」这一栏不在覆盖面内（M63 追加，选择「显式声明 + 负向对照」而不是扩检查器；M70 登记 (b) 与「反引号」两个子形态；M78 把「反引号」收进覆盖面，故现只剩 (a)(b)）**：
- 裸 `§X` 号**有歧义** —— 它常指**另一份**文档的章节（本文件自己 `docs/17` §10 里就有「这与 §11.5 …是同一条纪律」，
  而 `§11.5` 在 `docs/05`），也会以**省略 `docs/NN` 前缀**的形态出现（实例：`docs/05:379` 的「`docs/17` §9.5、§8.1」——
  第二个 `§8.1` 前没有 `docs/NN`；finding `20260926-reviewer-m63-…` 当年举的是同一行的 `§9.1`，那处**已随 `b158e49` 改为 `§8.1`**）。
  若把裸 `§X` 一律当同文档引用去判，会**大面积误报**（把合法的跨文档引用报成错引），
  那比"不查这一栏"更糟。
- ⇒ 本工具**明确声明**这一栏在覆盖面外，并用 `--selftest` 的 **NC-4**（构造一条同文档错引，核它**确实不报**）
  把这条盲区**登记成可见的一栏**（§9.2 第 3 条「已知会被漏掉」的对照）。
  要查这一类请用别的办法（人工或专门的一致性检查），**不要**把这里的 `0 out-of-range` 读成"裸 `§`-号引用也查过了"。

**M78 放宽匹配面：`SECREF` 容忍一个反引号（**不得**放宽到"任意位置都算引用"）**
- **改前**：`SECREF` 要求 `docs/NN` 与 `§` 之间只有空白/「的」，于是本仓主流写法（`` `docs/05` §6.1 ``，路径写成行内代码）
  **一条都不匹配** —— 实测 `main` `05e4662`：`95 → 143`（加一个可选反引号后），多出的 48 处**全部可解析**；
  本轮在 `08df83e`（本分支 base）复测为 `95 → 145`（多出 50 处，其中 2 处来自 `05e4662` 之后落库的 `docs/05` 改动）。
  ⇒ 那是**匹配面偏窄**，不是语料坏（M70 的 finding `…docs-scan-doc-refs-py-secref-…-95-vs-14.md`）。
- **改法**：只加一个**可选反引号**（`_SEC_PATH + '`?' + _SEC_TAIL`），间隔仍是「可选反引号 + 空白 + 可选「的」」——
  **没有**任何"路径与 `§` 之间可以有任意文本"的分支。若再加宽（例如容忍版本号），请先按 docstring 上面的第 ⑤ 栏登记证据。
- **判定阈一格未动**：`docs/NN` 必须存在、`§X` 必须在目标文档的标题里存在、退出码三态、`ALLOWLIST` 准入规则**全部照旧**。
  本轮放宽后**没有**浮出新的坏引用（读数见 `docs/17` §10）；**若将来浮出，那是发现问题 —— 报给塔，不许加进 `ALLOWLIST`**。
- ⇒ 覆盖表 ② 的 `section refs=N` 现在**含**反引号写法；NC-5 是这个匹配面的常驻**正/负**对照。

**M114 放宽判定面：目标标题可选带前导 `§`（**匹配面与分母未动**）**
- **病**：`docs/20-kernel-compliance-sweep.md` 的章节标题逐字写成 `### §3.2 …` / `## §5 …`（数字前一个字面 `§`），
  而判定是 `re.match(rf'^{section}(?!\d)', heading)` ⇒ 带 `§` 的标题**恒不匹配** ⇒ **任何** `docs/20 §X` 都判 unresolved
  （M113 实测：在 `docs/05` 写了 4 处 `docs/20 §3.2`/`§3.4` ⇒ `SEC-REF ×4`、`rc=1`；它只能改成内容锚点才绿）。
  ⚠ **性质要说准**：这不是"漏掉"（假阴性），而是**恒判负**（假阳性）—— 引用是好的，坏的是判定。
- **改法**：只把**目标标题那一侧**的判定放宽成 `^§?<num>(?!\d)`（`_has_section()`）。**匹配面（哪些文本算引用）一字未动**
  ⇒ `section refs` 分母不变、**无需重基线**；`docs/NN` 必须存在、`§X` 必须在目标文档标题里存在、退出码三态、
  `ALLOWLIST` 准入规则**全部照旧**。M114 之前的判定留成 `_has_section_legacy()`，只给 NC-6 (c) 当"改前"对照。
- **不凭猜扩集**：仓内 `§` 实测只有 U+00A7 一种（1515 处），无全角/半角变体、无 `第N节` 写法；`§` 与数字之间带空白的标题
  **0 例** ⇒ 只加 `§` 本身（与 M78「不加没有样本的字符类」同一纪律）。
- **影响面**（量尺 = `docs/evidence/scan_doc_refs/measure_sec_refs.py`，读数落同目录）：带前导 `§` 的标题 **28 条、全在 `docs/20`**；
  今天指向它们的 parsed 引用 **0 条**（M113 当场绕开了）⇒ 修后**当天语料**的五组计数与 rc **不变**，但这一类从"恒红"变"可用"。
  常驻对照 = **NC-6**（正 + 负，并含"同一份样本在改前判定下必须全被判负"的非空洞对照）。

**主扫描 —— 语料层的唯一权威**：
`0` = 跑过、且三类引用都是 0 out-of-range（`allowlisted` 命中另计，不 gate）；
`1` = 跑过且有 out-of-range（**含未登记的本仓路径式引用**）；
`2` = 没得比（`docs/*.md` 不存在）—— **不发合格证**。

**`--selftest` —— 工具层的权威**，用同一套退出码报告 **6 条负向对照**是否成立：
`0` = 六条都成立（这套检查器"还在咬"；两处**盲区**已分别登记为 NC-4 与 NC-5 的 (c) 子项，
NC-6 是 M114 新增的"目标标题带前导 `§`"正/负对照）；
`1` = 有一条不成立；`2` = 没得比。

**两条命令回答的是不同问题，刻意互不派生**（这一条是 M55 r3 评审的 P2：旧版 NC-3 的判定条件
比它自己打印的描述更强，导致"注入一条真坏路径"或"某条 allowlist 未被引用"时 `--selftest` 报 FAIL、
而主扫描给另一个信号）：

| 状态 | 主扫描 rc | `--selftest` rc |
|---|---|---|
| 干净 | 0 | 0 |
| **语料里有一条未登记的本仓路径式引用** | **1** | 0（检查器没坏） |
| `ALLOWLIST` 里某条未被任何 docs 引用 | 0 | 0 |
| `--ignore-allowlist`（把表当空表） | 1 | 0 |

⇒ **不要**用 `--selftest` 的退出码判断语料干不干净，反之亦然。

## `ALLOWLIST` 与 `allowlist-unused`

只有**外部快照**（随工具链 / 上游仓库分发、本仓不持有的文件）可以进 `ALLOWLIST`，且**每条必须带依据注释**。
`allowlist-unused`（某条目未被任何 docs 引用到）是 **advisory：既不 gate、也不影响任何退出码**——
它只说明"这条豁免可以删了"，**不是语料缺陷**。脚本、本 docstring、docs/17 §10 的覆盖表与退出码注释**四处同一口径**。

## 用法

```bash
python3 docs/scan_doc_refs.py                    # 默认：排除本文件自身
python3 docs/scan_doc_refs.py --include-self     # 含自引（读数见 docs/17 §10）
python3 docs/scan_doc_refs.py --ignore-allowlist # 把 ALLOWLIST 当空表：证明它不是"免死金牌"
python3 docs/scan_doc_refs.py --selftest         # 6 条负向对照（doc/section；坏路径必须进门；allowlist 真在干活；同文档 §-号盲区如实登记；反引号形态的正/负 + 版本号夹层的盲区登记；目标标题带前导 § 的正/负）
```
"""

import argparse
import collections
import glob
import os
import re
import subprocess
import sys

SELF = 'docs/17-verification-standard.md'

HEAD = re.compile(r'^#{1,6}\s+(.*)$')
DOCREF = re.compile(r'docs/([0-9]{2})(?:-[A-Za-z0-9_.-]+)?(?:\.md)?')
# `SECREF` = 路径 + 可选包裹符 + 间隔 + `§X`。**间隔只有三种形态**（M78 放宽前后都在这一条正则可控）：
# ① 空白；② 空白 + 「的」；③ 上面两种任一之前再加**一个反引号**（M78 新增：本仓主流写法是把路径写成行内代码，
# 如 `` `docs/05` §6.1 ``）。**没有**"路径与 `§` 之间可以是任意文本"的分支 —— 那会大面积误报。
# `_SEC_PATH` / `_SEC_TAIL` 拆开是为了让 `--selftest` 的 NC-5 能重建**放宽前**的匹配面（= 去掉 `` `? `` 那一段）。
_SEC_PATH = r'docs/([0-9]{2})(?:-[A-Za-z0-9_.-]+)?(?:\.md)?'
_SEC_TAIL = r'\s*(?:的\s*)?§\s*([0-9]+(?:\.[0-9]+)*)'
SECREF = re.compile(_SEC_PATH + r'`?' + _SEC_TAIL)

# 目标标题那一侧（M114）：**可选前导 `§`**。`docs/20` 把章节标题逐字写成 `### §3.2 …`，其余 docs 写成 `### 3.2 …`；
# M114 之前判定用 `^<num>` ⇒ 带 `§` 的标题**恒不匹配** ⇒ `docs/20 §X` 一律被判 out-of-range（M113 实测 4 处、rc=1）。
# 只放宽**标题**这一侧：匹配面（哪些文本算引用）与 path 部分**一字未动** ⇒ `section refs` 分母不变。
# 只加 `§` 本身：仓内 `§` 实测只有 U+00A7 一种（1515 处），无全角/半角变体、无 `第N节` 写法，
# `§` 与数字之间带空白的标题 **0 例** ⇒ 不凭猜扩集（同 M78「不加没有样本的字符类」）。
_SEC_HEAD_LEAD = '§?'


def _has_section(heading, sec):
    """标题 `heading` 里是否存在章节点位 `sec`（容忍标题的**前导 `§`**：`§3.2 …` 与 `3.2 …` 同等对待）。"""
    return bool(re.match(rf'^{_SEC_HEAD_LEAD}{re.escape(sec)}(?!\d)', heading))


def _has_section_legacy(heading, sec):
    """M114 **之前**的标题判定（`^<num>`）。**只给 `--selftest` 的 NC-6 (c) 当「改前」对照**，生产路径不用。"""
    return bool(re.match(rf'^{re.escape(sec)}(?!\d)', heading))


PATHREF = re.compile(
    r'(?<![A-Za-z0-9_./-])('
    r'(?:m[0-9]{1,2}_[A-Za-z0-9_]+|tools|baseline_env|probe_sync_quirks|build|data|evidence|logs|golden|weights)'
    r'/[A-Za-z0-9_./-]+\.(asc|h|hpp|cpp|cu|cuh|py|sh|md|json|toml|txt|log|cmake|yaml|yml|cfg|ini)'
    r')(?![A-Za-z0-9])')

# 负向对照样本（doc→doc 与章节点位）：三处**不带 § 的坏 doc 引用**（不存在的名字/编号 + 一处纯名字）
# 与三处**带 § 的坏章节点位**。两栏必须各自登记、且**互不重叠**（分子/分母同源，见 scan_docrefs 的 docstring）。
SAMPLE = "see docs/99-nothing.md §1 and docs/11 §5.1 ; also docs/07 §2 ; bare doc refs docs/88-nowhere.md and docs/77"
SAMPLE_EXPECT_DOC = ['docs/88-nowhere.md', 'docs/77']
SAMPLE_EXPECT_SEC = ['docs/99-nothing.md §1', 'docs/11 §5.1', 'docs/07 §2']

# 负向对照样本（本仓路径式引用）：两条呈**本仓路径形态**但**不存在**的路径。
# 它们 NOT 在 ALLOWLIST 里 ⇒ 必须**进门**（进 RESULT 与退出码），而不是被当 advisory。
SAMPLE_PATH = "see m99_bogus/nope_file.py and tools/does_not_exist.py"
SAMPLE_EXPECT_PATH = ['m99_bogus/nope_file.py', 'tools/does_not_exist.py']

# 负向/正向对照样本（章节点位，**反引号包裹**形态；M78 新增 —— 这一形态在 M78 之前一条都不匹配）：
#   (a) 负向：`` `docs/99-nothing.md` §1 ``（文档不存在）与 `` `docs/07` §2 ``（编号不存在）**必须**被登记；
#   (b) 正向：`` `docs/17` §10 `` 与 `` `docs/17` §1 `` 都**可解析**（`docs/17` 是本文、§1/§10 都在）**必须不**被登记。
# 期望值里的反引号是**匹配器吞掉的那个**（`SECREF` 的 `` `? `` 段），故命中文本形如 `` docs/07` §2 `` ——
# 它同时是"走了放宽后分支"的可见证据。四项都进 `SAMPLE_Q_EXPECT_SEC`（= 放宽后的匹配面）。
SAMPLE_Q = ("wrap-bad: `docs/99-nothing.md` §1 and `docs/07` §2 ; "
            "wrap-good: `docs/17` §10 and `docs/17` §1")
SAMPLE_Q_EXPECT_SEC = ['docs/99-nothing.md` §1', 'docs/07` §2', 'docs/17` §10', 'docs/17` §1']
SAMPLE_Q_EXPECT_BAD = ['docs/99-nothing.md` §1', 'docs/07` §2']

# 盲区登记样本（M78，与 NC-4 同型：「报 0 命中」才是 PASS）：路径与 `§` 之间**夹了版本号**的写法
# （docstring 覆盖表的「未覆盖」第 ⑤ 栏）。这条引用**本身是好的**（`docs/05` 的 §2/§6 都存在，
# 下面当场核过）—— 正因为它好、却被漏掉，才必须把缺口登记成可见的一栏，而不是让它伪装成"没发现问题"。
SAMPLE_V = 'version-gap: docs/05-megakernel-design.md v1.2 §2/§6'
SAMPLE_V_SECS = ('2', '6')

# 正向/负向对照样本（章节点位，**目标标题带前导 `§`**；M114 新增 —— 这一形态的判定在 M114 之前**恒判负**）：
#   目标 = `docs/20`：它的标题逐字是 `### §3.2 …` / `## §5 …`（仓内 28 条 §-前导标题**全在这一个文档**里，
#   量尺读见 `docs/evidence/scan_doc_refs/`）。故：
#   (a) **负向**：`docs/20 §99.7` 与 `docs/20 §10`（`docs/20` 里都没有）**必须**被登记 —— 这条是防"改宽成永真"的闸；
#       `§10` 特意选它：`docs/17` **有** §10 ⇒ 它同时证明"容忍前导 `§`"没有变成"任意数字都算"。
#       另附 `docs/17 §99.7`（目标标题**不带** `§` 的坏引用）⇒ 证明"不带 `§` 的一半"也仍然会咬。
#   (b) **正向**：`docs/20 §3.2` 与 `docs/20 §5`（`docs/20` 里都有）**必须不**被登记，且**确实被匹配到**
#       （否则这条正向对照会因为"压根没匹配上"而**假通过** —— 正是 M114 要防的形态）。
#   (c) **非空洞对照**：同一份样本用 `_has_section_legacy()`（M114 之前的判定）跑一遍，(b) 的两条**必须全被判负**
#       ⇒ 证明 NC-6 测的是新判定面；并当场核 `docs/20` 里确实有以 `§` 开头的标题。
SAMPLE_H = ('sign-heading: docs/20 §3.2 and docs/20 §5 ; '
            'bad: docs/20 §99.7 and docs/20 §10 and docs/17 §99.7')
SAMPLE_H_EXPECT_BAD = ['docs/20 §99.7', 'docs/20 §10', 'docs/17 §99.7']
SAMPLE_H_EXPECT_SECS = ['10', '3.2', '5', '99.7']    # `SECREF` 命中的**去重**章节点位（字符串序）
SAMPLE_H_EXPECT_SECTIONS = ('3.2', '5')              # 正向那两条（= 改前判定必被判负的两条）

# ---- 显式 allowlist：本仓路径式引用里**允许 miss** 的条目，每条必须写明"为什么它不是本仓文件" ----
# 准入规则：**只有外部快照**（随工具链 / 上游仓库分发、本仓不持有的文件）可以进这里。
#   * 命中会**单列打印**（`allowlisted`，含依据）—— 不静默；
#   * 本表**不是免死金牌**：`--selftest` 的第 3 条负向对照会把本表清空重扫，证明这些条目随即变成 gate；
#   * 从未命中的条目会打印 `allowlist-unused`（advisory，不 gate）—— 防止本表悄悄长胖。
ALLOWLIST = {
    'tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h':
        '外部快照：Bisheng 编译器自带头（随 ASC 工具链分发在 /usr/local/Ascend 下），本仓不持有；'
        'docs/13 引它是为了指出 clang 内置宏的定义处',
    'tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_defines.h':
        '外部快照：同上（docs/18 引它指出 `__clang_cce_*` 宏的定义处）',
}


def load_index():
    files = sorted(glob.glob('docs/*.md'))
    if not files:
        print('RESULT: SKIPPED (docs/*.md 不存在 —— 没得比，不发合格证)', file=sys.stderr)
        sys.exit(2)
    path_of = {os.path.basename(f)[:2]: f for f in files}
    heads = {}
    for f in files:
        with open(f, encoding='utf-8') as fh:
            heads[f] = [m.group(1).strip() for m in (HEAD.match(l.strip()) for l in fh) if m]
    return files, path_of, heads


def _sec_starts(text):
    """所有 `docs/NN §X` 命中的**起始位置**（用于把同一处引用归到「章节点位」那一栏）。"""
    return {m.start() for m in SECREF.finditer(text)}


def scan_docrefs(text, path_of, heads):
    """返回 `(bad_doc, bad_sec, n_sec)`。

    **`bad_doc` 只含不带 `§` 的 `docs/NN` 命中** —— 与 `docref_counts()` 的 `doc refs` 分母是
    **同一子集**（§9.2 第 1 条：分子分母同源）。带 `§` 的那种由 `bad_sec` 单独登记，
    因此一条坏引用只进一栏：既不会出现「分子掉在分母之外」（比例 >1），也不会被数成两条。

    章节点位是否存在，用 `_has_section()` 判（M114 起**容忍目标标题的前导 `§`**：`### §3.2 …` 与 `### 3.2 …` 同等对待）。
    """
    sec_starts = _sec_starts(text)
    bad_doc = [m.group(0) for m in DOCREF.finditer(text)
               if m.start() not in sec_starts and m.group(1) not in path_of]
    bad_sec = []
    for m in SECREF.finditer(text):
        tgt = path_of.get(m.group(1))
        if tgt is None or not any(_has_section(h, m.group(2)) for h in heads[tgt]):
            bad_sec.append(m.group(0))
    return bad_doc, bad_sec, len(sec_starts)


def docref_counts(text):
    """两类引用的**分母**（"查过多少条"）：(doc→doc 定位, 章节点位)。

    doc→doc 定位 = 不带章的 `docs/NN` 出现次数（带 `§` 的那种归入章节点位，与 §10 覆盖表的
    ①② 两栏一一对应，两栏不重叠）；计数与上面的 `DOCREF` / `SECREF` **同一套匹配器**（§9.2 第 1 条），
    且与 `scan_docrefs()` 的 `bad_doc` **同一子集**（分子/分母同源）。
    """
    sec_starts = _sec_starts(text)
    doc_only = sum(1 for m in DOCREF.finditer(text) if m.start() not in sec_starts)
    return doc_only, len(sec_starts)


def git_files():
    return set(subprocess.run(['git', 'ls-files'], capture_output=True, text=True).stdout.split())


def pathref_all(text):
    """本仓路径式引用的**全部**出现次数（分母）。"""
    return [m.group(1) for m in PATHREF.finditer(text)]


def pathref_miss(text, own):
    """本仓路径式引用里"路径不存在"的那一类（**未**过 allowlist）。"""
    return [m.group(1) for m in PATHREF.finditer(text)
            if not (os.path.exists(m.group(1)) or m.group(1) in own)]


def scan_pathrefs(files, allowlist):
    """返回 (gating, allowed)：前者进 RESULT/退出码，后者单列打印。"""
    own = git_files()
    gating = collections.Counter()
    allowed = collections.Counter()
    for f in files:
        with open(f, encoding='utf-8') as fh:
            for p in pathref_miss(fh.read(), own):
                (allowed if p in allowlist else gating)[p] += 1
    return gating, allowed


def main():
    ap = argparse.ArgumentParser(description='docs 引用完整性扫描（docs/17 §10）')
    ap.add_argument('--selftest', action='store_true',
                    help='6 条负向对照：doc/section、坏路径必须进门、allowlist 真的在干活、'
                         '同文档 §-号盲区登记、反引号形态的正/负 + 版本号夹层的盲区登记（M78）、'
                         '目标标题带前导 § 的正/负（M114）')
    ap.add_argument('--include-self', action='store_true',
                    help='把 docs/17 自身也算进被扫文件（复现"自引"读数；预期 FAIL）')
    ap.add_argument('--ignore-allowlist', action='store_true',
                    help='把 ALLOWLIST 当空表跑（复现"删掉 allowlist 条目会怎样"）')
    a = ap.parse_args()

    allowlist = {} if a.ignore_allowlist else ALLOWLIST
    files, path_of, heads = load_index()

    if a.selftest:
        # NC-1：doc→doc 与章节点位 —— 不存在的名字 / 编号 / 章节都必须被登记。
        #       同时证明**两栏互不重叠**（同一处 §-式引用只进 bad_sec，不进 bad_doc）：
        #       分子（bad_doc）与分母（doc refs）同源，比例不可能 >1。
        bad_doc, bad_sec, n_sec = scan_docrefs(SAMPLE, path_of, heads)
        n_doc_only, n_sec_only = docref_counts(SAMPLE)
        doc_only_set = {m.group(0) for m in DOCREF.finditer(SAMPLE)
                        if m.start() not in _sec_starts(SAMPLE)}
        sec_set = {m.group(0) for m in SECREF.finditer(SAMPLE)}
        c1 = (bad_doc == SAMPLE_EXPECT_DOC and bad_sec == SAMPLE_EXPECT_SEC
              and set(bad_doc) <= doc_only_set and set(bad_sec) <= sec_set
              and len(bad_doc) <= n_doc_only and len(bad_sec) <= n_sec_only)
        print(f'NC-1 doc/section   SAMPLE = {SAMPLE}')
        print(f'     bad_doc = {bad_doc}   （分母 doc refs={n_doc_only}，分子 ⊆ 分母）')
        print(f'     bad_sec = {bad_sec}    （分母 section refs={n_sec_only}，分子 ⊆ 分母）')
        print(f'     两栏不重叠：`docs/99-nothing.md §1` 只进 bad_sec、`docs/88-nowhere.md` 只进 bad_doc'
              f'    -> {"PASS" if c1 else "FAIL"}')

        # NC-2：本仓路径式引用 —— 坏路径必须**进门**（不是 advisory）
        got = pathref_miss(SAMPLE_PATH, git_files())
        c2 = got == SAMPLE_EXPECT_PATH
        print(f'NC-2 path-ref      SAMPLE = {SAMPLE_PATH}')
        print(f'     missed  = {got}    -> {"PASS" if c2 else "FAIL"}')
        print('     （这两条不在 ALLOWLIST 里 ⇒ 必须进 RESULT 与退出码）')

        # NC-3：allowlist 不是免死金牌。**判定 = 两次运行的差集**（不依赖语料里有没有别的坏引用，
        #      也不要求每条 allowlist 条目都被引用到）：
        #        (清空后 gating) == (清空前 gating) ∪ (清空前 allowlisted)
        #      判定条件**就是打印出来的那条等式**（M55 r4 评审 follow-up）。旧版另带两个合取项
        #      （`清空后没有 allowlisted`、`清空前 allowlisted ⊆ ALLOWLIST`），它们在构造上**恒真**
        #      （空表不可能产出 allowlisted；scan_pathrefs 只从 ALLOWLIST 取键）⇒ 已删，免得读者再推导一次。
        base = [f for f in files if f != SELF]          # 与主扫描同一集合口径
        g_all, a_all = scan_pathrefs(base, ALLOWLIST)
        g_none, a_none = scan_pathrefs(base, {})
        c3 = set(g_none) == set(g_all) | set(a_all)
        print('NC-3 allowlist     含 ALLOWLIST：gating=%s  allowlisted=%s'
              % (sorted(g_all), sorted(a_all)))
        print('                   清空 ALLOWLIST：gating=%s  allowlisted=%s'
              % (sorted(g_none), sorted(a_none)))
        print('     判定：(清空后 gating) == (清空前 gating) ∪ (清空前 allowlisted)')
        print('     -> %s   （本次 %d 条豁免全部改走 gate；allowlist 不是免死金牌）'
              % ('PASS' if c3 else 'FAIL', len(a_all)))

        # NC-4 覆盖**盲区**（"已知会被漏掉"的一条 —— §9.2 第 3 条）：**裸 §-号引用不在覆盖面内**。
        #      样本 `§99.7` 在本文（docs/17）里**不存在**（下面当场核过），而扫描器的 SECREF 要求
        #      `docs/NN` 前缀 ⇒ 它对这种引用**一概不匹配、必须报 0 命中**。
        #      这条对照的作用是把盲区**如实登记成可见的一栏**（而不是让它伪装成"没发现问题"）——
        #      这就是 coverage 声明本身。盲区的**理由**见 docstring「未覆盖」栏（裸 § 号有歧义：
        #      同一个 `§11.5` 可能指另一份文档；另一子形态 = 省略 `docs/NN` 前缀的跨文档引用）。
        #      该栏的**第三**个子形态（`docs/NN` 带反引号）原是本栏的一部分，**M78 已把它收进覆盖面**，
        #      并另立 **NC-5** 作它的常驻正/负对照；本栏现只剩 (a) 同文档、(b) 省略前缀两个子形态。
        self_sec = '同上，见本文 §99.7（本文件内的错引）与本文 §1.2（正确引用）'
        bd4, bs4, ns4 = scan_docrefs(self_sec, path_of, heads)
        sec_exists = any(_has_section(h, '99.7') for h in heads[SELF])
        c4 = (bd4 == [] and bs4 == [] and ns4 == 0) and (not sec_exists)
        print(f'NC-4 same-doc §-ref SAMPLE = {self_sec}')
        print(f'     `§99.7` 在本文中是否存在 = {sec_exists}（False = 它确实是一条**错引**）')
        print(f'     scanner reported: bad_doc={bd4} bad_sec={bs4} n_sec={ns4}  <- 全 0 = **不报**（盲区，如实登记）')
        print(f'     判据：错引确实存在 且 扫描器确实不报 ⇒ 这条对照**就是**「同文档 §-号不在覆盖面内」的声明'
              f'    -> {"PASS" if c4 else "FAIL"}')

        # NC-5（M78 新增）：**反引号包裹**的 `docs/NN §X`。三个子项一次登记放宽后的匹配面边界：
        #   (a) **负向**：反引号包裹的**坏**引用必须被登记（与 NC-1 同口径）；
        #   (b) **正向**：反引号包裹的**可解析**引用必须**不**被登记 —— 且它**确实被匹配到**
        #       （否则这条"正向对照"会因为"压根没匹配上"而**假通过**，那正好是本 mission 要防的形态）；
        #   (c) **盲区登记**（与 NC-4 同型、以「报 0 命中」为 PASS）：路径与 `§` 之间夹**版本号**的写法
        #       （docstring 覆盖表「未覆盖」第 ⑤ 栏）**仍不匹配**。
        # 另核 `legacy`（= 放宽前的匹配面）对这 4 条**一条都匹配不到** ⇒ 本对照确实在测**新**匹配面。
        # 注：`bad_sec` 里的命中会带上**被吞掉的那个反引号**（形如 `` docs/07` §2 ``），见 SAMPLE_Q 的注释。
        legacy = re.compile(_SEC_PATH + _SEC_TAIL)
        bad_doc_q, bad_sec_q, n_sec_q = scan_docrefs(SAMPLE_Q, path_of, heads)
        sec_set_q = {m.group(0) for m in SECREF.finditer(SAMPLE_Q)}
        n_legacy_q = len(legacy.findall(SAMPLE_Q))
        n_v = len(SECREF.findall(SAMPLE_V))
        v_05 = path_of.get('05')
        v_ok = bool(v_05) and all(any(_has_section(h, s) for h in heads[v_05])
                                  for s in SAMPLE_V_SECS)
        c5 = (bad_doc_q == [] and bad_sec_q == SAMPLE_Q_EXPECT_BAD
              and sec_set_q == set(SAMPLE_Q_EXPECT_SEC) and n_sec_q == len(SAMPLE_Q_EXPECT_SEC)
              and n_legacy_q == 0 and n_v == 0 and v_ok)
        print(f'NC-5 backtick §-ref SAMPLE = {SAMPLE_Q}')
        print(f'     放宽前的匹配面（= 去掉 `` `? ``）在本样本上命中 = {n_legacy_q} 条'
              f'（0 = 旧正则对这 4 条一条都不匹配 ⇒ 这正是被放宽的缺口）')
        print(f'     放宽后命中 = {n_sec_q} 条：{sorted(sec_set_q)}')
        print(f'     (a) 负向 bad_sec = {bad_sec_q}   （期望 {SAMPLE_Q_EXPECT_BAD}）')
        print(f'     (b) 正向：上面 4 条里可解析的那 2 条（`docs/17` §10 / §1）未进 bad_sec，'
              f'且它们**确实被匹配到**（否则就是假通过）')
        print(f'     (c) 盲区 SAMPLE = {SAMPLE_V}')
        print(f'         `docs/05` 的 §{"/§".join(SAMPLE_V_SECS)} 都存在 = {v_ok}（引用本身是好的）'
              f'，而扫描器命中 = {n_v} 条（0 = **不报**，缺口如实登记）')
        print(f'     -> {"PASS" if c5 else "FAIL"}')

        # NC-6（M114 新增）：**目标标题带前导 `§`** 的章节点位（`docs/20` 的 `### §3.2 …` 形态）。
        #   本仓 28 条 §-前导标题**全在这一个文档**里（量尺读数见 `docs/evidence/scan_doc_refs/`）。
        #   (a) **负向**：`docs/20 §99.7` 与 `docs/20 §10`（`docs/20` 里都没有）必须被登记 —— 这是防"改宽成永真"的闸；
        #       `§10` 特意选它：`docs/17` **有** §10 ⇒ 同时证明"容忍前导 `§`"没有变成"任意数字都算"。
        #       另附 `docs/17 §99.7`（目标标题**不带** `§` 的坏引用）⇒ 证明另一半也仍然会咬。
        #   (b) **正向**：`docs/20 §3.2` / `docs/20 §5` 必须**不**被登记，且**确实被匹配到**（防"没匹配上"的假通过）；
        #   (c) **非空洞**：同一份样本改用 `_has_section_legacy()`（= M114 之前的判定）跑，(b) 那 2 条必须**全被判负**
        #       ⇒ 证明 NC-6 测的确实是新判定面；并当场核 `docs/20` 里确实有以 `§` 开头的标题。
        h_20 = path_of.get('20')
        sign20 = list(heads[h_20]) if h_20 else []
        n_sign20 = sum(1 for h in sign20 if h.startswith('§'))
        bd6, bs6, ns6 = scan_docrefs(SAMPLE_H, path_of, heads)
        secs6 = sorted({m.group(2) for m in SECREF.finditer(SAMPLE_H)})
        pre6 = [s for s in SAMPLE_H_EXPECT_SECTIONS
                if h_20 and not any(_has_section_legacy(h, s) for h in sign20)]
        c6 = (h_20 is not None and n_sign20 > 0
              and bd6 == [] and bs6 == SAMPLE_H_EXPECT_BAD
              and ns6 == 5 and secs6 == SAMPLE_H_EXPECT_SECS
              and pre6 == list(SAMPLE_H_EXPECT_SECTIONS))
        print(f'NC-6 §-heading     SAMPLE = {SAMPLE_H}')
        print(f'     `docs/20`（{h_20}）里以 `§` 开头的标题 = {n_sign20} 条 —— 仓内 `§`-前导标题的唯一出处')
        print(f'     改后判定命中 = {ns6} 条：{secs6}')
        print(f'     (a) 负向 bad_sec = {bs6}   （期望 {SAMPLE_H_EXPECT_BAD}）')
        print(f'     (b) 正向：`docs/20` §3.2 / §5 未进 bad_sec，且它们**确实被匹配到**（否则就是假通过）')
        print(f'     (c) 非空洞：同一份样本改用**改前**判定（`^<num>`）时，正向条目里被判处负的 = {pre6}'
              f'   （期望 {list(SAMPLE_H_EXPECT_SECTIONS)} ⇒ 这条对照测的是新判定面）')
        print(f'     -> {"PASS" if c6 else "FAIL"}')

        ok = c1 and c2 and c3 and c4 and c5 and c6
        print('RESULT: ' + ('OK (6 条负向对照全部成立)' if ok else 'FAIL (负向对照未按预期成立)'))
        print('说明：`--selftest` 判的是**这套检查器还在不在咬**（工具层），'
              '它刻意不随语料状态变化；语料的干净与否由**主扫描**的退出码回答（语料层）。')
        return 0 if ok else 1

    scanned = files if a.include_self else [f for f in files if f != SELF]
    nd = ns = nt = n_doc = n_path = 0
    for f in scanned:
        with open(f, encoding='utf-8') as fh:
            text = fh.read()
        bad_doc, bad_sec, _ = scan_docrefs(text, path_of, heads)
        n_doc_only, n_sec = docref_counts(text)
        nd += len(bad_doc)
        ns += len(bad_sec)
        nt += n_sec
        n_doc += n_doc_only
        n_path += len(pathref_all(text))
        for x in bad_doc:
            print(f'DOC-REF  {os.path.basename(f)}: {x}')
        for x in bad_sec:
            print(f'SEC-REF  {os.path.basename(f)}: {x}')

    gating, allowed = scan_pathrefs(scanned, allowlist)
    # 三态分栏（§9.2 第 2 条）：每一栏都带**分母**（查过多少条），计数与上面的匹配器同源。
    print(f'docs scanned={len(scanned)}  doc refs={n_doc} (unresolved={nd})  '
          f'section refs={nt} (unresolved={ns})')
    print(f'in-repo path refs={n_path}  gating={sum(gating.values())} (distinct={len(gating)})  '
          f'allowlisted={sum(allowed.values())} (distinct={len(allowed)})')
    for k, v in sorted(gating.items(), key=lambda x: -x[1]):
        print(f'  {v:3d}  {k}')
    for k, v in sorted(allowed.items(), key=lambda x: -x[1]):
        print(f'  {v:3d}  {k}')
        print(f'       ↳ 依据：{ALLOWLIST[k]}')
    unused = sorted(set(ALLOWLIST) - set(allowed))
    if unused:
        print(f'allowlist-unused (advisory，不 gate): {unused}')

    nbad = nd + ns + sum(gating.values())
    if nbad == 0:
        print(f'RESULT: OK ({len(scanned)} files scanned, {n_doc} doc refs, {nt} section refs, '
              f'{n_path} path refs, {sum(allowed.values())} allowlisted, 0 out-of-range)')
        return 0
    print(f'RESULT: FAIL ({nd} doc + {ns} sec + {sum(gating.values())} path out-of-range)')
    return 1


if __name__ == '__main__':
    sys.exit(main())
