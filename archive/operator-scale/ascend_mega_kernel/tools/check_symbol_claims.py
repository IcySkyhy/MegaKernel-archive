#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/check_symbol_claims.py —— **静态标注核对**：把「文档里对符号的用途/归属说法」与
「符号声明处的注释原文 + 代码引用次数」**并排摆出来**。

起因（M56 里连续出现的同一形态）：**新写的静态文字没有被自身核对**。三处实例**一律引 commit
（轮次标签不可核，故不写）**：
  * `tools/scan_stale_runtime_prints.py` 的 `[refused]` 文案说"仍会扫描"、代码没做到 —— 该版见
    `96e7e96`（修复见其后的 `daf3344`）；
  * `tools/golden/selfcheck.py` 运行时注记里"3 处命中都在 `MxQuantComputeScale` 内"只对 2/3 个文件成立
    —— 该版见 `96e7e96`（原始那句见 `0c163b5:tools/golden/selfcheck.py` 的 `[note]`）；
  * `m20_hyperconn/README.md` §3 把 `BUF_AIV_GJ(8)` 标成「injW 预转」—— 该版见
    `cecf0ae:m20_hyperconn/README.md`；而它**声明处的注释**（`m20_resources.h` 的 `BUF_AIV_GJ`）写的是
    `预转: hc_norm bf16 行 MTE2 -> V（整核一次）`（`injW` 是 `BUF_AIV_IW` 的用途）。
规则：**凡在文档里对符号的用途/归属下判断，必须引用声明处原文，或附当场计数；不许自造用途描述。**
**历史叙述一律引 commit，不引轮次标签**（轮次标签不可核；这一条与"引用要能被独立定位"同族）。

════════════════════════════════════════════════════════════════════════
**定位：这是"把事实并排摆出来的辅助工具"，不是判定器。** 用途是否一致**大多无法自动判定**。
每行按下面四态贴出来（判据都是机械的、可复算）：

| 每行状态 | 机械判据（可复算） | 能自动咬住什么 / 不能 |
|---|---|---|
| `冲突（借用他符号词汇）` | 同一文档行里还提到了另一个**已声明符号 Y**；该行出现了 token T：T **不在**本符号 X 的声明注释里、**在 Y 的声明注释里**、T 不是任何已声明符号的名字、且 T 像用途名词（**驼峰 / 含 `_` / 连续两个大写字母** —— 与代码 `distinctive()` 一致） | **正是 `cecf0ae:m20_hyperconn/README.md` 那处**：文档把 `BUF_AIV_GJ` 说成「injW 预转」，而 `injW` 只在 `BUF_AIV_IW` 的注释里 ⇒ 与声明注释冲突。**⚠ 这是启发式"候选"，不是判定**：实测 repo 级有假阳性（见下） |
| `一致（词汇包含）` | 未借他符号词汇，且该行与该符号的声明注释**至少有 1 个标识符 token 重叠** | 弱正向：说明行文与声明注释用了同一套词汇；**不保证语义正确** |
| `需人工核对（无机械信号）` | 上面两条都不成立 | **已知盲区 A**：把用途**改写**成另一套说法（不借用任何他符号词汇）时本工具**看不见** ⇒ 必须人工读并排的两栏 |
| （不参与判定）`未覆盖·已知盲区` | 文档里**没加反引号**的已声明符号名 | **已知盲区 B**：本工具只认反引号，纯文本提到的符号它根本不看 ⇒ 单独列出、不当作通过 |

**实测的假阳性（不要在评审里当成缺陷，它是设计的一部分）**：repo 级一次（42 个文档 / 2000 个声明）
报 4 个 `冲突` 候选，人工读后**全部是假阳性**——形态都是「一行里并列枚举多个符号、各自带注释」，
于是邻居符号的词汇被算到了某一符号头上（`m7_router_topk/README.md:204` 的 `BUF_LOG` 即此类）。
故：**`冲突` 的作用是"把它指给你读"，不是替你做结论**；真判定必须人读「文档该行」与「声明处注释原文」两栏。

⇒ **只有 `冲突` 会让本工具返回 1**；其余三态只被**显式列出**（不当作通过，也不当作失败）。

## 它机械核到的其它事实（计数，与匹配器同源）

* `代码引用次数`（= `decl_refs`）—— 该符号**剥掉注释后**在 `--code` 范围内的出现次数，**再减去它自己
  声明处那一行**（声明行只算"声明"，不算引用）；为 0 即**死声明**。注释里的提及**不算引用**，单列成
  `注释提及`（`comment_refs`），并在输出里与「声明行」分开摆出来：
  * `出现在注释里` ⇒ `注释提及 > 0`、但**仍可能是死声明**；
  * `出现在自己的声明行` ⇒ 声明行本身，只作定位，不计入引用；
  * `真的被代码引用` ⇒ `代码引用次数 > 0`。
  **这是本工具修复前的一个缺陷**：旧版把注释提及也算进唯一口径 `decl_refs`，于是"给死声明补一条
  `⚠ 未使用` 注释"就能让它们在判据里静默消失（m20 树在加注释前后，旧口径的死声明数从 9 掉到 2；
  对照读数与各自所在的 commit 见 `tools/evidence/symbol_claims_check.log` 读数 B/C）——
  工具看不见"注释里点过名"与"真的被引用"的区别。
  仍存在的**已知盲区（漏判方向）**：预处理条件里的提及（`#if X` / `#ifdef X`）是文本不是注释，会被
  算作代码引用，`#if 0` 关掉的块同理；字符串字面量里的符号名也一样 ⇒ 这些符号**不会**被判死，必须人读。
  另注意：若某符号只被**同一头文件里的别的声明**引用（如偏移链 `A = B + 1`），
  它就不是这个口径下的死声明，但仍可能 kernel 侧 0 引用 —— 需要另按 `grep -c` 在具体实现文件里核。
* `未解析` —— 文档反引号里出现、但在扫描范围内**找不到声明**的标识符（可能是别的仓库/库的名字，
  故只报数，不判错）。
* 反引号符号的**首词**才会被当作符号，且**长度 ≥ 3**：`u`/`x`/`m`/`S1` 这类短名在别的模块里
  常被声明成局部量，repo 级运行时会制造假阳性（这也是实测结论）。

## 退出码（三态）

  0 = 比过且通过：至少解析出 1 个文档符号（符号模式）/ 至少认出 1 次读数（`--numbers`），
      且**没有** `冲突`（**不等于正确**，见各模式的已知盲区）
  1 = 比过且有差异：存在 ≥1 个 `冲突` 行（**启发式候选**，需人读）
  2 = 没得比：没有文档、没有代码根、扫描范围内没有任何声明，或什么都没解析出来

## 用法

    python3 tools/check_symbol_claims.py                       # 默认：仓根 + 仓内全部 *.md
    python3 tools/check_symbol_claims.py --code m20_hyperconn --docs m20_hyperconn/README.md
    python3 tools/check_symbol_claims.py --numbers --docs <file…>   # 数字对账模式（见下节）
    python3 tools/check_symbol_claims.py --self-test           # 合成用例的负向对照（见 self_test）
    # 相对路径一律按 --root 解析（不跟 cwd），便于把"另一份 checkout 的文档"拿来对同一个 --code 跑

════════════════════════════════════════════════════════════════════════
## `--numbers`：数字对账模式（M56 新增；首次随 `6c48163`）

**规则**：**重跑任何读数块之后，本文件里引用同一读数的其它数字必须同步改**。
本模式找的就是「**同一读数在两处数值不一致**」，并且**限定在同一文件 + 同一节**内比较
（节 = 最近的 `^#{1,3} ` 标题）—— 因为不同节本来可能是不同环境/不同档的读数，跨节比较会制造假阳性。

**机械判据**（三条都在同一遍遍历里产出，计数与匹配器同源）：
* **M1 单读数**：`标签 + 数字 + 量词`（量词如 `个/处/行/条/次/个文件/…` 是"这是个读数"的锚）；
  两个出现**可比较**当且仅当它们的标签里存在**长度 ≥3 的中文片段**（3-gram）相同。
* **M2 复合读数**：`数字+量词 / 数字+量词`（如 `8 处 / 17 个`），按**量词对**比较。
* 同一条线上出现 ≥2 个不同数值 ⇒ `冲突`（**候选，需人读**）。

**已知盲区（必须人工读，模式看不见）**：
1. 标签**措辞完全不同**的同义读数（如「源文件 7 个」vs「输入文件 8 个」）连不上线；
2. 只出现**一次**的读数没有兄弟可比（coverage 里报数）；
3. 数字写成中文/拼写形态认不出。
⇒ 与符号模式同一立场：**它是"把不一致的两处并排指给你读"的辅助工具，不是判定器。**

**实测的假阳性（目标文件 `tools/evidence/stale_runtime_print_scan.log` 上 8 个候选里 2 真 6 假）**，
形态只有两种，读的时候直接跳过：
* 「**同一名词、不同谓语**」：`扫描源文件 8 个` 与 `跳过非源文件 330 个` 共享"源文件"这个 3-gram；
* 「**同一张表、不同行**」：如 §5 表里两行各自的「各 3 处 / 各 5 处」，本来就是不同符号的读数。
（这两条已写进本节，避免后来者把候选当结论。）
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))

DECL_RE = re.compile(r"^\s*(?:constexpr\s+[\w:<>]+\s+|#define\s+|(?:static\s+)?(?:const\s+)?"
                     r"(?:uint\d+_t|int\d+_t|size_t|float|double|bool|char))\s*([A-Za-z_]\w*)\s*[=;(]")
SYM_IN_DOC = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_.])[^`]*`")


def doc_symbols(line):
    """文档行里**当作"符号"来核对**的反引号首词：长度 ≥ 3 才收。

    短名（`u` / `x` / `m` / `S1`）在别的模块里常被声明成局部量，repo 级运行时会把它们
    误解析成符号并制造假阳性 —— 实测见归档 log §3 的 repo 级读数。
    """
    return [m.group(1) for m in SYM_IN_DOC.finditer(line) if len(m.group(1)) >= 3]
ID_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
BACKTICK_SPAN = re.compile(r"`[^`]*`")
COMMENT = re.compile(r"//\s*(.*)$")
CODE_EXT = (".asc", ".h", ".hpp", ".c", ".cc", ".cpp", ".cxx", ".py")
SKIP_DIRS = {".git", ".tower", "build", "__pycache__", "node_modules", "evidence"}

CONFLICT, AGREE, NEEDS_READ, UNRESOLVED = "冲突（借用他符号词汇）", "一致（词汇包含）", "需人工核对（无机械信号）", "未解析"


def _iter_files(root, exts, skip_evidence=True):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not (skip_evidence and d == "evidence")]
        for f in sorted(filenames):
            if f.endswith(exts):
                yield os.path.join(dirpath, f)


_CODE_LEX = re.compile(
    r'"(?:\\.|[^"\\\n])*"'          # 双引号字符串（不跨行）
    r"|'(?:\\.|[^'\\\n])*'"         # 字符字面量（不跨行）
    r"|//[^\n]*"                    # 行注释
    r"|/\*.*?\*/",                  # 块注释（可跨行）
    re.S)
_PY_LEX = re.compile(
    r'"""(?:\\.|[^\\])*?"""'        # 三引号字符串（可跨行）
    r"|'''(?:\\.|[^\\])*?'''"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|#[^\n]*",                    # Python 行注释（`#` 在 C 里是预处理指令、不是注释）
    re.S)
PY_EXT = (".py",)


def split_code_comments(text, ext):
    """把源码文本切成 (code, comment) 两份**等长**文本（行结构一致、逐 token 可对齐）。

    粒度是**字符位置**而不是整行：同一行里常同时躺着「注释提及」与「真代码引用」，
    按行排除（`grep -v '//'`）会把真引用一起吃掉。这里只把**注释跨度**内的字符换成空格
    （换行保留），并显式跳过字符串/字符字面量 —— 否则 `"http://host/SYM"` 里的 `//`
    会被当成注释起点、把后半行的真引用吞掉。

    已知取舍（都写在 docstring 的「它机械核到的其它事实」一节）：预处理条件（`#if X`）、
    `#if 0` 关掉的块、以及字符串字面量里的符号名都留在 code 里 ⇒ 会被算作引用。
    """
    code, comment = list(text), list(text)
    for i, c in enumerate(comment):
        if c != "\n":
            comment[i] = " "
    pat = _PY_LEX if ext in PY_EXT else _CODE_LEX
    for m in pat.finditer(text):
        s = m.group(0)
        if s[0] in "/*#":                     # 注释跨度；字面量跨度落在 else（留在 code 里）
            for k in range(m.start(), m.end()):
                if code[k] != "\n":
                    code[k] = " "
                comment[k] = text[k]
    return "".join(code), "".join(comment)


def collect_decls(paths):
    """返回 {name: {"file","line","comment","code_refs","comment_refs"}}。

    * `code_refs` —— **剥掉注释后**代码内该符号的出现次数，再减去声明行自身那一次；
      为 0 即**死声明**。
    * `comment_refs` —— 该符号在**注释**里的提及次数，**不计入引用**（只说明"这个名字被写过"）。
      旧版把两者混在同一个口径里，于是"给死声明补一条 `⚠ 未使用` 注释"就能让它们从判据里消失。
    """
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += list(_iter_files(p, CODE_EXT))
        elif os.path.exists(p):
            files.append(p)
    decls, texts = {}, {}
    for f in files:
        try:
            raw = open(f, errors="ignore").read()
        except OSError:
            continue
        lines = raw.split("\n")               # 声明探测仍用原行（DECL_RE / 注释抽取口径不变）
        texts[f] = split_code_comments(raw, os.path.splitext(f)[1])
        for i, line in enumerate(lines):
            m = DECL_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            cm = COMMENT.search(line)
            comment = cm.group(1).strip() if cm else ""
            if not comment:                       # 声明行无注释 ⇒ 取上一条 // 注释块（最多上溯 2 行）
                for j in range(i - 1, max(-1, i - 3), -1):
                    if DECL_RE.match(lines[j]):
                        break                      # 撞到**另一条声明** ⇒ 那句注释不属于本符号
                    c2 = COMMENT.search(lines[j])
                    if c2:
                        comment = c2.group(1).strip()
                        break
            if name not in decls:
                decls[name] = {"file": f, "line": i + 1, "comment": comment,
                               "code_refs": 0, "comment_refs": 0}
    # 引用计数（除声明行本身）：**一次遍历所有 token** 建 Counter，避免 O(声明数 × 行数)
    # （repo 级根目录下后者要几分钟，前者亚秒级 —— 这是本工具能在任意 root 上跑的前提）。
    # 代码引用与注释提及**分两个 Counter**：注释提及不再污染死声明判据（M71 修的盲点）。
    from collections import Counter
    counts, comment_counts = Counter(), Counter()
    decl_line_of = {(d["file"], d["line"]): n for n, d in decls.items()}
    for f, (code_text, comment_text) in texts.items():
        for i, line in enumerate(code_text.split("\n")):
            for t in WORD.findall(line):
                counts[t] += 1
            own = decl_line_of.get((f, i + 1))
            if own:
                counts[own] -= 1               # 声明行上那次是符号名本身
        for line in comment_text.split("\n"):
            for t in WORD.findall(line):
                comment_counts[t] += 1
    for name, d in decls.items():
        d["code_refs"] = counts.get(name, 0)
        d["comment_refs"] = comment_counts.get(name, 0)
    return decls


def decl_tokens(decls):
    """{name: set(标识符 token)} —— 来自声明注释 + 符号名本身。"""
    return {n: set(ID_TOKEN.findall(d["comment"])) | set(ID_TOKEN.findall(n))
            for n, d in decls.items()}


def distinctive(tok):
    """该 token 是否"像用途名词"（够特别，值得当机械信号）：驼峰混合大小写，或含下划线。"""
    if "_" in tok:
        return True
    return bool(re.search(r"[a-z][A-Z]", tok) or re.search(r"[A-Z][A-Z]", tok))


def classify(doc_line, name, decls, tks):
    """对一次「文档提到 name」的分类：返回 (state, 明细)。

    `冲突` 的机械判据（刻意收窄，宁少勿滥）：
      同一个文档行里同时提到了另一个**已声明符号 Y**，而该行出现了 token T：
      T 不在本符号 X 的声明注释里、却在 Y 的声明注释里，且 T 不是任何已声明符号的名字、
      T 看起来像用途名词（驼峰/下划线）。
      ⇒ 形态就是「把 Y 的用途词汇写到了 X 名下」——正是要抓的那一处。
    """
    if name not in decls:
        return UNRESOLVED, []
    own = tks[name]
    mentioned = {n for n in doc_symbols(doc_line) if n in decls and n != name}
    seen = set(ID_TOKEN.findall(doc_line)) - set(decls)
    bad = []
    for t in sorted(seen - own):
        if not distinctive(t):
            continue
        owners = sorted(y for y in mentioned if t in tks[y])
        if owners:
            bad.append("%s ← 来自同行提到的 %s 的声明注释" % (t, ",".join(owners)))
    if bad:
        return CONFLICT, bad
    if seen & own:
        return AGREE, sorted(seen & own)
    return NEEDS_READ, []


def scan(root, code_paths, doc_paths, limit):
    decls = collect_decls(code_paths)
    tks = decl_tokens(decls)
    n_docs = n_mentions = 0
    rows, unresolved, unbackticked = [], [], {}
    for doc in doc_paths:
        try:
            lines = open(doc, errors="ignore").read().split("\n")
        except OSError:
            continue
        n_docs += 1
        for i, line in enumerate(lines, 1):
            for name in doc_symbols(line):
                n_mentions += 1
                state, detail = classify(line, name, decls, tks)
                if state == UNRESOLVED:
                    unresolved.append((doc, i, name))
                    continue
                rows.append((state, name, doc, i, detail, decls[name]))
            # **已知盲区**：文档里**没用反引号**包裹的已声明符号名 —— 本工具看不见它附近的用途说法。
            # 实现要点：先把该行的词抽成集合，再与声明表求交（O(1) 级），**不要**逐声明做正则扫描
            # （那是 O(文档行数 × 声明数)，repo 级根目录下要几分钟）。
            plain = BACKTICK_SPAN.sub(" ` ", line)
            for name in set(WORD.findall(plain)) & decls.keys():
                unbackticked.setdefault(name, []).append("%s:%d" % (os.path.relpath(doc, root), i))
    return decls, rows, unresolved, unbackticked, n_docs, n_mentions


def _rel2root(root, p):
    """相对路径一律按 `--root` 解析（否则会跟着 cwd 跑，读者复算结果不同）。"""
    return p if os.path.isabs(p) else os.path.join(root, p)


def report(args, write=print):
    root = os.path.abspath(args.root)
    code_paths = [_rel2root(root, p) for p in args.code] if args.code else [root]
    docs = [_rel2root(root, d) for d in args.docs] if args.docs else \
        list(_iter_files(root, (".md",), skip_evidence=True))
    docs = [d for d in docs if os.path.exists(d)]
    if not os.path.isdir(root):
        write("RESULT: SKIPPED (扫描根不存在：%s；这不是通过)" % root)
        return 2
    if not docs:
        write("RESULT: SKIPPED (没有可核对的文档；这不是通过)")
        return 2
    decls, rows, unresolved, unbackticked, n_docs, n_mentions = scan(root, code_paths, docs,
                                                                     args.limit)
    if not decls:
        write("RESULT: SKIPPED (扫描范围内没有任何声明可核；这不是通过)")
        return 2
    if not rows:
        # 没得比也要把**覆盖范围**与**已知盲区**打出来，否则读者只看到一个 SKIPPED
        write("[coverage] 文档 %d 个；反引号符号出现 %d 次（解析到声明 %d）"
              % (n_docs, n_mentions, len(rows)))
        write("[coverage] 声明 %d 个（来源：%s）"
              % (len(decls), ",".join(os.path.relpath(p, root) for p in code_paths)))
        _write_unbackticked(root, unbackticked, write)
        write("RESULT: SKIPPED (文档里没有一个反引号符号能解析到声明 ⇒ 没得比；"
              "**这不是通过**，而且上面「未覆盖」栏里的符号本工具根本没看)" )
        return 2
    dead = sorted(n for n, d in decls.items() if d["code_refs"] == 0)
    dead_mentioned = [n for n in dead if decls[n]["comment_refs"] > 0]
    conflicts = [r for r in rows if r[0] == CONFLICT]
    agree = [r for r in rows if r[0] == AGREE]
    needs = [r for r in rows if r[0] == NEEDS_READ]

    write("[coverage] 文档 %d 个；反引号符号出现 %d 次（解析到声明 %d、未解析 %d）"
          % (n_docs, n_mentions, len(rows), len(unresolved)))
    write("[coverage] 声明 %d 个（来源：%s）；**死声明（剥注释后代码引用 0）%d 个**"
          "（其中在注释里被点过名的 %d 个 —— **注释提及不算引用**，故它们仍是死声明）"
          % (len(decls), ",".join(os.path.relpath(p, root) for p in code_paths),
             len(dead), len(dead_mentioned)))
    write("[verdict] 冲突 %d / 一致 %d / 需人工核对 %d —— **只有「冲突」参与判定**"
          % (len(conflicts), len(agree), len(needs)))
    for state, name, doc, ln, detail, d in conflicts + agree + needs:
        if len(rows) > args.limit and state != CONFLICT:
            continue
        write("  [%s] %s  ←  %s:%d" % (state, name, os.path.relpath(doc, root), ln))
        write("      文档该行      : %s" % _short(open(doc, errors="ignore")
                                                  .read().split("\n")[ln - 1]))
        write("      声明处(%s:%d): %s   // %s"
              % (os.path.relpath(d["file"], root), d["line"],
                 _decl_head(d), d["comment"] or "（无注释）"))
        write("      代码引用次数  : %d（剥注释后的代码内，已减去声明行；注释提及不计）"
              % d["code_refs"])
        write("      注释提及      : %d（**不算引用**，只说明这个名字被写过）"
              % d["comment_refs"])
        if detail:
            write("      机械信号      : %s" % "; ".join(detail))
    if unresolved:
        write("  [%s] %d 个（只报数，不判错 —— 可能是别的仓库/库的名字）：%s"
              % (UNRESOLVED, len(unresolved),
                 ", ".join(sorted({u[2] for u in unresolved})[:10])))
    if dead:
        write("  [死声明] %s%s"
              % (", ".join(dead[:20]),
                 "" if len(dead) <= 20 else "（共 %d 个，此处只列前 20）" % len(dead)))
        if dead_mentioned:
            write("  [死声明·注释点过名] %s%s —— 它们**仍判死**（注释提及不是引用）"
                  % (", ".join(dead_mentioned[:20]),
                     "" if len(dead_mentioned) <= 20 else "（共 %d 个，此处只列前 20）"
                     % len(dead_mentioned)))
    if unbackticked:
        _write_unbackticked(root, unbackticked, write)
    else:
        write("  [未覆盖·已知盲区] 无（文档里提到已声明符号时都加了反引号 ⇒ 本工具看得到）")

    if conflicts:
        write("RESULT: CONFLICT (%d 个冲突行 / 核了 %d 个文档符号出现、%d 个声明；"
              "其余 %d 一致、%d 需人工核对、%d 个未覆盖)"
              % (len(conflicts), len(rows), len(decls), len(agree), len(needs),
                 len(unbackticked)))
        return 1
    write("RESULT: OK (**0 个机械冲突信号** / 核了 %d 个文档符号出现、%d 个声明；%d 一致、"
          "%d 需人工核对、%d 个未覆盖 —— **这不等于用途正确**：需人工核对与未覆盖两类才是"
          "判定依据，本工具只把事实并排摆出来)"
          % (len(rows), len(decls), len(agree), len(needs), len(unbackticked)))
    return 0


def _short(s, n=150):
    s = s.strip()
    return s[:n] + (" …" if len(s) > n else "")


def _decl_head(d):
    return "line %d" % d["line"]


def _write_unbackticked(root, unbackticked, write):
    write("  [未覆盖·已知盲区] **文档里没加反引号**的已声明符号 %d 个 —— 本工具看不见它们旁边的用途说法，"
          "必须人工读：" % len(unbackticked))
    for n, locs in sorted(unbackticked.items())[:10]:
        write("      %-18s %s" % (n, ", ".join(locs[:4])))


# ================================================================== --numbers
# 规则：**重跑任何读数块之后，本文件里引用同一读数的其它数字必须同步**。
# 这一模式做的是「同一读数在两处不一致」的机械交叉核对：
#   M1 单读数  `标签 + 数字 + 量词`（量词是"这是个读数"的锚），按**标签里长度 ≥3 的中文片段**连线；
#   M2 复合读数 `数字+量词 / 数字+量词`（如 `8 处 / 17 个`），按**量词对**连线。
# 同一条线里出现 ≥2 个不同数值 ⇒ `冲突`（候选，需人读；与符号模式同一立场）。
# **已知盲区**：标签完全不同（如「源文件 7 个」vs「输入文件 8 个」）时连不上线；
# 只出现一次的读数没有兄弟可比；数字写成中文/拼写形态也认不出。这些都在 coverage 里报数。
UNIT = (r"(?:个|处|行|档|条|次|对|项|块|根|页|字节|文件|命中|列|位|"
        r"files?|hits?|lines?|items?|rows?|ms|us|µs|ns|s|B|KB|KiB|MB|GB|%)")
READ_M1 = re.compile(r"([^\d\n]{1,20}?)(\d+(?:\.\d+)?)\s*(" + UNIT + r")")
READ_M2 = re.compile(r"(\d+(?:\.\d+)?)\s*(" + UNIT + r")\s*/\s*(\d+(?:\.\d+)?)\s*(" + UNIT + r")")
CJK3 = re.compile(r"[\u4e00-\u9fff]{3,}")
FENCE = re.compile(r"^\s*```")


def _cjk3_keys(label):
    """标签里所有长度 ≥3 的连续中文片段（取其 3-gram，便于两个措辞不同的标签连线）。"""
    keys = set()
    for run in CJK3.findall(label):
        for i in range(len(run) - 2):
            keys.add(run[i:i + 3])
    return keys


def numbers_scan(paths):
    """返回 (occurrences, n_lines)。occurrence = dict(key/kind/value/line/fenced/section/text)。

    **可比范围限定在「同一文件 + 同一章节」**（章节 = 最近的 `^#{1,3} ` 标题；无标题则整文件）
    —— 这是为了把"同一节里重跑过的读数与其周边引用"当成一组，避免把**不同环境/不同节**里
    本来就该不同的读数（如真树 8 个 vs /tmp 副本 9 个）误报成冲突。
    """
    occ, n_lines = [], 0
    for p in paths:
        try:
            lines = open(p, errors="ignore").read().split("\n")
        except OSError:
            continue
        n_lines += len(lines)
        in_fence, section = False, "(no-heading)"
        for i, line in enumerate(lines, 1):
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            if not in_fence and re.match(r"^#{1,3} ", line):
                section = line.strip()
            for m in READ_M2.finditer(line):
                occ.append({"kind": "M2", "key": "M2:%s/%s" % (m.group(2), m.group(4)),
                            "value": "%s %s / %s %s" % (m.group(1), m.group(2), m.group(3), m.group(4)),
                            "line": i, "fenced": in_fence, "doc": p, "section": section, "text": line})
            for m in READ_M1.finditer(line):
                label = re.sub(r"\s+", " ", m.group(1)).strip("|*`（()【[：:，,、。 ").strip()
                for k in _cjk3_keys(label):
                    occ.append({"kind": "M1", "key": "M1:%s:%s" % (k, m.group(3)),
                                "value": "%s %s" % (m.group(2), m.group(3)),
                                "line": i, "fenced": in_fence, "doc": p, "section": section,
                                "text": line, "label": label})
    return occ, n_lines


def report_numbers(args, write=print):
    root = os.path.abspath(args.root)
    docs = [_rel2root(root, d) for d in args.docs] if args.docs else \
        list(_iter_files(root, (".md", ".log", ".txt"), skip_evidence=False))
    docs = [d for d in docs if os.path.exists(d)]
    if not docs:
        write("RESULT: SKIPPED (没有可对账的文件；这不是通过)")
        return 2
    occ, n_lines = numbers_scan(docs)
    if not occ:
        write("RESULT: SKIPPED (这些文件里没有认得出的读数（标签+数字+量词）；这不是通过)")
        return 2
    groups = {}
    for o in occ:
        groups.setdefault((o["doc"], o["section"], o["key"]), []).append(o)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    conflicts = {k: v for k, v in multi.items() if len({x["value"] for x in v}) >= 2}
    singles = sum(1 for k, v in groups.items() if len(v) == 1)
    write("[coverage] 文件 %d 个 / %d 行；认出读数出现 %d 次，按「**同一文件 + 同一节**内的"
          "（M1 标签 3-gram / M2 量词对，再加量词）」归成 %d 条线；**其中可交叉核对的线 %d 条**（≥2 次出现）"
          % (len(docs), n_lines, len(occ), len(groups), len(multi)))
    for k in sorted(conflicts):
        vs = conflicts[k]
        write("  [冲突] 同一节内同一条线出现不同数值：%s  ← %s「%s」"
              % (k[2], os.path.relpath(k[0], root), k[1][:60]))
        for o in vs:
            write("      %-9s %s:%d %s" % (o["value"], os.path.relpath(o["doc"], root), o["line"],
                                          "（代码块内）" if o["fenced"] else "（散文）"))
            write("            行文: %s" % _short(o["text"]))
    write("  [未覆盖·已知盲区] 只出现一次的读数线 %d 条（没有兄弟可比）、"
          "以及标签措辞完全不同的同义读数（连不上线）—— 本模式都看不见，必须人工读" % singles)
    if conflicts:
        write("RESULT: CONFLICT (%d 条线数值不一致 / 交叉核对了 %d 条线、%d 次读数出现) —— "
              "**候选，需人读**" % (len(conflicts), len(multi), len(occ)))
        return 1
    write("RESULT: OK (0 条线数值不一致 / 交叉核对了 %d 条线、%d 次读数出现；"
          "%d 条线只有单次出现 ⇒ 未覆盖 —— **这不等于文件里的数字都对**)" %
          (len(multi), len(occ), singles))
    return 0


# ------------------------------------------------------------------ 负向对照
def self_test(tmpdir):
    """四个文档合成用例：机械能咬的、弱正向、**两个已知盲区**（改写说法 / 不加反引号）。

    另带两组计数对照：`_self_test_numbers`（`--numbers`）与
    `_self_test_dead_refs`（`decl_refs` 的死声明口径：注释提及不算引用）。
    """
    code = os.path.join(tmpdir, "code")
    docs = os.path.join(tmpdir, "docs")
    os.makedirs(code)
    os.makedirs(docs)
    with open(os.path.join(code, "res.h"), "w") as f:
        f.write("constexpr uint32_t BUF_A_GJ = 8;    // 预转: hc_norm bf16 行 MTE2 -> V\n"
                "constexpr uint32_t BUF_A_IW = 3;    // S1: injW 32B MTE2 -> V\n")
    cases = {
        # 1) **复现 `cecf0ae:m20_hyperconn/README.md` 的现场形态**：一行里同时提到两个符号（含简写 `SYM(3)`），
        #    却把第二个符号的用途词汇 injW 挂到了第一个符号名下 ⇒ 必须判「冲突」
        "borrow.md": "| `BUF_A_IW`(3)、`BUF_A_GJ`(8) | 后两个是 injW 预转 |\n",
        # 2) 与该符号自己的声明注释共用词汇 hc_norm ⇒ 判「一致」（弱正向）
        "agree.md": "| `BUF_A_GJ` | 预转 hc_norm bf16 行 |\n",
        # 3) 已知盲区 A：把用途改写成另一套说法，不借任何他符号词汇 ⇒ 必须报「需人工核对」，
        #    既不能判「冲突」（无机械信号）也不能判「一致」（不许假装它核对过）
        "blindspot_paraphrase.md": "| `BUF_A_GJ` | 门控混合用的常驻表 |\n",
        # 4) 已知盲区 B：符号**不加反引号** ⇒ 本工具看不见它旁边的用途说法，
        #    必须出现在「未覆盖·已知盲区」栏里（而不是被当成没提到）
        "blindspot_plain.md": "| BUF_A_GJ | 门控混合用的常驻表 |\n",
    }
    for n, s in cases.items():
        with open(os.path.join(docs, n), "w") as f:
            f.write(s)
    want = {"borrow.md": CONFLICT, "agree.md": AGREE,
            "blindspot_paraphrase.md": NEEDS_READ, "blindspot_plain.md": "未覆盖"}
    print("--self-test（负向对照：四个合成用例）--")
    ok = True
    for n, expected in want.items():
        args = argparse.Namespace(root=tmpdir, code=[code], docs=[os.path.join(docs, n)],
                                  limit=50)
        import io
        buf = io.StringIO()
        rc = report(args, write=lambda s: buf.write(s + "\n"))
        out = buf.getvalue()
        if "[%s]" % CONFLICT in out:
            got = CONFLICT
        elif "[%s]" % AGREE in out:
            got = AGREE
        elif "[%s]" % NEEDS_READ in out:
            got = NEEDS_READ
        elif "[未覆盖·已知盲区]" in out:
            got = "未覆盖"
        else:
            got = "?"
        good = (got == expected)
        ok = ok and good
        print("  %-26s 判为 %-22s 期望 %-22s rc=%d %s"
              % (n, got, expected, rc, "OK" if good else "MISMATCH"))
    print("RESULT: %s（borrow=机械咬住 / agree=弱正向 / paraphrase=需人工核对 / "
          "plain=落进「未覆盖·已知盲区」栏）" % ("OK" if ok else "FAIL"))
    ok = _self_test_numbers(docs) and ok
    ok = _self_test_dead_refs(tmpdir) and ok
    return 0 if ok else 1


def _self_test_dead_refs(tmpdir):
    """引用计数口径的负向对照（M71）：**注释提及不算引用**，同行里的真引用也不许被按行吃掉。

    合成用例（全写进调用方给的临时目录，不碰 repo）：
      * `ONLY_COMMENT` / `ONLY_BLOCK` —— 只在行注释 / 块注释里被点名 ⇒ **必须仍判死**（正对照：
        修复前的旧口径会把它们算成"有引用"，这正是 `m20_hyperconn` 上 9 → 2 的成因）；
      * `REAL_REF` —— 真被代码引用 ⇒ **不得判死**（负对照）；
      * `MIXED` —— **同一行**既有注释提及又有真引用 ⇒ 真引用必须仍被算到（粒度是字符位置不是整行）；
      * `AFTER_STR` —— 出现在**字符串里的 `//`** 之后 ⇒ 不许被当成注释起点吞掉；
      * `COND_SYM` —— **已知盲区（漏判方向）**：`#if COND_SYM` 是文本不是注释 ⇒ 会被算作引用、
        本工具**不会**判死它。这一例是用来钉住"它会被漏掉"的，不是要它判死。
    """
    code = os.path.join(tmpdir, "refcode")
    os.makedirs(code)
    with open(os.path.join(code, "refs.h"), "w") as f:
        f.write("constexpr uint32_t ONLY_COMMENT = 1;   // ONLY_COMMENT 只在注释里被点名\n"
                "constexpr uint32_t REAL_REF = 2;       // 真引用见 use.asc\n"
                "constexpr uint32_t MIXED = 3;          // MIXED 在注释里也提一次\n"
                "constexpr uint32_t AFTER_STR = 4;\n"
                "constexpr uint32_t COND_SYM = 5;\n"
                "/* ONLY_BLOCK 只在块注释里被点名\n"
                "   ONLY_BLOCK 第二处 */\n"
                "constexpr uint32_t ONLY_BLOCK = 6;\n")
    with open(os.path.join(code, "use.asc"), "w") as f:
        f.write('uint32_t a = REAL_REF;\n'
                'uint32_t b = MIXED;            // MIXED 在本行注释里又提一次\n'
                'const char* url = "http://h/AFTER_STR";\n'
                '#if COND_SYM\n'
                'uint32_t c = 1;\n'
                '#endif\n')
    decls = collect_decls([code])
    print("--self-test（引用计数口径：注释提及 vs 真引用）--")
    # (符号, 期望 code_refs, 期望 comment_refs 下限, 说明)
    cases = [
        ("ONLY_COMMENT", 0, 1, "只在行注释里点名 ⇒ 仍判死"),
        ("ONLY_BLOCK",   0, 2, "只在块注释里点名 ⇒ 仍判死"),
        ("REAL_REF",     1, 0, "真被代码引用 ⇒ 不判死"),
        ("MIXED",        1, 1, "同行既注释提及又真引用 ⇒ 真引用仍被算到"),
        ("AFTER_STR",    1, 0, "字符串里的 // 不是注释起点 ⇒ 真引用没被吞"),
        ("COND_SYM",     1, 0, "**已知盲区**：预处理条件里的提及算引用 ⇒ 不会判死"),
    ]
    ok = True
    for name, want_code, want_cmt, why in cases:
        d = decls.get(name)
        if d is None:
            print("  %-14s 未解析到声明 ⇒ MISMATCH（%s）" % (name, why))
            ok = False
            continue
        good = (d["code_refs"] == want_code and d["comment_refs"] >= want_cmt)
        ok = ok and good
        print("  %-14s code_refs=%-2d comment_refs=%-2d 期望 code=%d comment>=%d %s（%s）"
              % (name, d["code_refs"], d["comment_refs"], want_code, want_cmt,
                 "OK" if good else "MISMATCH", why))
    print("RESULT: %s（ONLY_COMMENT/ONLY_BLOCK=注释提及仍判死 / REAL_REF=真引用不判死 / "
          "MIXED/AFTER_STR=同行或字符串里的真引用没被吞 / COND_SYM=**已知会被漏掉**的盲区）"
          % ("OK" if ok else "FAIL"))
    return ok


def _self_test_numbers(docs_dir):
    """`--numbers` 的负向对照：① 同一读数两处不一致必须报出；② 已知盲区（标签措辞不同）必须看不见
    但**被计入「只出现一次的线」**，不许当成通过。"""
    cases = {
        # M1：散文说 7 个、代码块说 8 个 —— 必须连成同一条线并报冲突
        "num_hit.md": "关键读数：**源文件仍是 7 个**。\n\n```\n[coverage] 扫描源文件 8 个；跳过 330 个\n```\n",
        # M2：`8 处 / 17 个` vs `9 处 / 18 个` —— 必须按量词对连线并报冲突
        "num_hit2.md": "见下表：**9 处 / 18 个**文本文件。\n\n```\n归档命中 8 处 / 17 个文本文件\n```\n",
        # 已知盲区：同义读数但标签措辞完全不同（源文件 vs 输入文件）⇒ 连不上线
        "num_blind.md": "关键读数：**输入文件仍是 7 个**。\n\n```\n[coverage] 扫描源文件 8 个\n```\n",
    }
    for n, s in cases.items():
        with open(os.path.join(docs_dir, n), "w") as f:
            f.write(s)
    print("--self-test（--numbers：三个合成用例）--")
    ok = True
    want = {"num_hit.md": "冲突", "num_hit2.md": "冲突", "num_blind.md": "未覆盖"}
    for n, expected in want.items():
        import io
        buf = io.StringIO()
        args = argparse.Namespace(root=docs_dir, docs=[os.path.join(docs_dir, n)], code=None,
                                  numbers=True, limit=50)
        rc = report_numbers(args, write=lambda s: buf.write(s + "\n"))
        out = buf.getvalue()
        got = "冲突" if "[冲突]" in out else "未覆盖" if "[未覆盖·已知盲区]" in out else "?"
        good = (got == expected)
        ok = ok and good
        print("  %-16s 判为 %-8s 期望 %-8s rc=%d %s"
              % (n, got, expected, rc, "OK" if good else "MISMATCH"))
    print("RESULT: %s（num_hit/num_hit2=同一条线两处数值不一致必须报出 / "
          "num_blind=**已知盲区**：标签措辞不同 ⇒ 只进「未覆盖」栏，不许当成通过）"
          % ("OK" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO, help="扫描根（默认 = 本脚本所在仓）")
    ap.add_argument("--code", nargs="*", default=None, help="声明来源（文件/目录，默认 = --root）")
    ap.add_argument("--docs", nargs="*", default=None,
                    help="要核对的文档（符号模式默认 root 下全部 *.md；--numbers 模式默认 *.md/*.log/*.txt）")
    ap.add_argument("--limit", type=int, default=200, help="最多展示多少行")
    ap.add_argument("--numbers", action="store_true",
                    help="数字对账模式：找「同一读数在两处数值不一致」（见 docstring 的 --numbers 节）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            return self_test(d)
    if args.numbers:
        return report_numbers(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
