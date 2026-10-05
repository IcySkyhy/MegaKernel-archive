#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""docs 引文保真扫描器（M129 交付）—— 逐对核 `(所引文件:行号, 引文子串)`。

## 它真正检查什么（覆盖范围声明；按 `docs/17-verification-standard.md` §9.2 规则②自报）

扫 `docs/*.md`，抽出「引文 ↔ 所引 `文件:行`」的**成对样本**，断言引文是所引行的
**连续子串**（`**` 加粗标记按下面的归一化规则处置）。三类计数（verified / unverified /
finding）与分母**由同一次运行、同一套匹配器产出**（§9.2 第 1 条）。

| 栏 | 内容 |
|---|---|
| **覆盖（成对样本）** | ① **同行紧邻**：`「引文」` 与 `文件:行` 之间只隔分隔符字符（空白、反引号、圆/方括号、`：:`、`，,`、`、`、`的`、`*`、`·`），两个方向都算；② **表内相邻单元格**：引文所在单元格去掉所有 `「…」` 后**只剩空白/分隔符**（纯引文列），且左/右**紧邻**单元格含 `文件:行` ⇒ 该引文与该单元格内的每个 `文件:行` 成对；③ **源侧跨行 `「…」`**（开 `「` 与闭 `」` 分处两行，逐行匹配看不见）：整份文本匹配后映射回行号，取**开行左侧最近一个** `文件:行` 作出处；引文按「去换行缩进 / 折成空格」两种拼法各核一次（续行的 markdown 引用标记 `> ` 拼前去掉）。**另**：一对样本若在同一份被引文件里列了多行（本仓写法「`:18`（同 `:20`）」），引文再对**这些行的合并文本**判一次，取更优的那个判定（记 `cite_kind=merged-lines`） |
| **路径解析范围** | 只解析**带 `/` 的仓内相对路径**，外加 `docs/NN:line` 这一简写（唯一匹配 `docs/NN-*.md`）。`.tower/**` 按**主检出根**（`git worktree list` 的第一项）解析 —— 在 worktree 里跑时 `.tower` 不在本 worktree，但在主检出下存在 |
| **未覆盖（unverified / 逐类跳过）** | ① 引文所在处与任何 `文件:行` 都不构成上面三种关系（本工具**不抽**这对，计入 `quote-unpaired` 分母；**源侧跨行引文也在这个分母里**，见 `quote_crossline` 单独计数）；② 所引路径解析不到：外部前缀（`[asc-main]`/`[asc-fork]`/`[vLLM]`）、绝对路径、裸文件名（`README.md`/`m15_ple.asc`，含唯一同名 —— 按 M39 的教训不猜）、简写模块名（`m14`）、裸行号（`:419`）且**同行**没有前置 `文件:行`、`docs/NN` 匹配到多份或 0 份；③ 引文为空 |
| **已知盲区（本工具不覆盖，不得读成通过）** | ⓵ **一对多/多对一的语义配对**：单元格级绑定是**集合式**的（引文只要在**该单元格所列任一 `文件:行`** 里逐字出现即记 OK），**不**强制「第 i 条引文 ↔ 第 i 个出处」的位置对应。选集合式而非位置式，是因为 M125 修正后的真实行会让位置式**误判**（见下「为什么是集合式」）；② 引文所在**行/单元格内**若同时并置多条引文与多个出处，本工具只保证「集合命中」，不保证语义归属正确；③ 引文与出处分处**不相邻**的表格列（中间隔了别的列）不配对；④ 裸行号 `:xx` 只回看**同一行**的最近前置 `文件:行`，跨行/跨单元格的指代本工具不解析；⑤ **只识别 `「…」` 定界** —— ASCII 引号 `"…"`（含 `docs/18-vector-api-audit.md:436` 那种）**不被扫**，即同一处失配若只以 ASCII 引号写出，本工具不报；⑥ 源侧跨行引文**只**在**开行左侧**找出处 —— 出处写在闭行、或写在开行之前的行时，该引文计入 `quote-unpaired`（**可见的分母**），不被检查 |

**判定档（引文 vs 所引行）**：

| 档 | 条件 | 归入 |
|---|---|---|
| `OK` | 引文逐字是所引行（区间，含 `:a-b`）的连续子串 | verified |
| `OK-EMPH` | 逐字不中，但把**引文与所引行双方的 `**` 去掉后**中 | verified（另一栏列出） |
| `OK-PARTIAL` | 引文含省略号（`…`/`...`），按省略号切段后**各段按序**都能在所引行里找到 | verified（另一栏列出） |
| `OK-MULTILINE` | 单行不中，但在**整份被引文件**里能中，且命中区间**覆盖所引行**（引文跨行、含所引行） | verified（另一栏列出） |
| `FIND-NOTINFILE` | 引文既不在所引行、也不在被引文件任何位置 | finding |
| `FIND-ELSEWHERE` | 引文（或其省略号切段）在**被引文件**里出现，但命中区间**不覆盖所引行**（挂在别的行上） | finding |
| `FIND-PARTIAL` | 含省略号的引文，切段后**有段**在所引行里找不到，且在被引文件里也找不到 | finding |

**为什么是集合式（M125 实证）**：`docs/17` §9.9 触发事件表末行修正后，同一单元格里并置了
两条出处与一条**带出处注记的引文**（「前者：…（逐字）、…（作「…」）；后者：…（逐字）」）。
位置式配对会把「第二处引文」与「第二处出处」硬绑，从而**误报**修正后的正确行。
集合式把「引文是否能在该单元格所列出处里逐字找到」作为判据，既不误报修正行，又能抓住**修前列错**那一版
（修前第二处引文在两个所列出处里都找不到）；代价就是上面「已知盲区」⓵。

## 三态退出码（沿用本仓惯例，`docs/17` §8.3）

主扫描：`0` = 跑过且无 finding；`1` = 跑过且有 finding；`2` = 没得比（`docs/*.md` 一个都找不到）。
`--selftest`：`0` = 全部对照成立；`1` = 有对照不成立；`2` = 没得比（脚本自检样例无法构造）。

## `--selftest` 的正负两侧

负向对照合成一条**引文改错**的样本 ⇒ 必须 `rc=1`；正向对照合成一条**引文正确**的样本 ⇒ 必须 `rc=0`。
另有 M125 B1 的**真实回归样例**：`docs/17` §9.9 末行修正**前**的那一版（引文挂错出处）必须判红，
修正**后**的那一版必须判绿；两版文本逐字取自 `.tower/comms/reviews/` 的评审记录与 M125 分支的 `docs/17`。
还含盲区登记对照：省略号引文、裸行号、外部前缀、裸文件名、引文无紧邻出处、`docs/NN` 简写解析、
`FIND-ELSEWHERE`、**源侧跨行引文（进入检查 / 计入分母）**各一条。
`--selftest` 共 14 条对照，全部成立才 `rc=0`。

## 用法

    python3 docs/scan_quote_refs.py                  # 扫 docs/*.md
    python3 docs/scan_quote_refs.py --selftest       # 工具层自检
    python3 docs/scan_quote_refs.py --dump-pairs     # 打印所有成对样本与判定（校准用）
    python3 docs/scan_quote_refs.py --tower-refs     # 打印 docs 里 `.tower/` 路径引用清单

`--docs-dir` / `--repo-root` 供自检在临时目录里复现，正常使用不必给。
"""

import argparse
import bisect
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

QUOTE_RE = re.compile(r"「([^「」]*)」")

# `path:lines` —— path 可缺省（裸行号），lines 支持 `87` / `452-462` / `19/135/143` / `87,192-202`。
CITE_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./][A-Za-z0-9_./+*@-]*)?"
    r":(?P<lines>\d+(?:-\d+)?(?:[,/]\d+(?:-\d+)?)*)"
)

EXTERNAL_PREFIX_RE = re.compile(r"\[(asc-main|asc-fork|vLLM|vllm)\]")

# 紧邻关系里允许夹在引文与出处之间的字符（不含 `|`，故跨单元格不会被判为紧邻）。
SEP_RE = re.compile(r"^[\s`（）()\[\]{}<>:：,，、;；.。的「」*·'\"~—-]*$")

MODULE_SHORTHAND_RE = re.compile(r"^m\d{1,2}$")


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - 读不到就是没得比
        return None


def find_quotes(line):
    out = []
    for m in QUOTE_RE.finditer(line):
        out.append({"text": m.group(1), "start": m.start(), "end": m.end()})
    return out


def find_cites(line):
    out = []
    for m in CITE_RE.finditer(line):
        before = line[max(0, m.start() - 24):m.start()]
        ext = EXTERNAL_PREFIX_RE.search(before)
        path = m.group("path")
        if path is not None and not re.search(r"[A-Za-z0-9_]", path):
            path = None  # 形如 `/` 的裸行号前缀
        out.append({
            "path": path,
            "lines": m.group("lines"),
            "start": m.start(),
            "end": m.end(),
            "external": ext is not None,
        })
    return out


def parse_linespec(spec):
    ranges = []
    for part in re.split(r"[,/]", spec):
        if "-" in part:
            a, b = part.split("-", 1)
            ranges.append((int(a), int(b)))
        else:
            ranges.append((int(part), int(part)))
    return ranges


def cell_spans(line):
    spans = []
    start = 0
    for m in re.finditer(r"\|", line):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(line)))
    return spans


def cell_index(spans, pos):
    for i, (a, b) in enumerate(spans):
        if a <= pos < b:
            return i
    return len(spans) - 1


def is_table_row(line):
    return line.lstrip().startswith("|")


def main_worktree_root(repo_root):
    """主检出的根 —— 在 git worktree 里跑时，`.tower/**` 仍在主检出下。

    `docs` 里的 `.tower/...` 引用指向塔的工件，它们**不在** worktree 里
    （`git ls-files .tower` = 0，且 worktree 只含被跟踪文件）。故 `.tower/**` 一律
    按主检出根解析，其余路径按 `--repo-root` 解析；两种运行位置读数一致。
    """
    try:
        cp = subprocess.run(["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
                            capture_output=True, text=True)
        for line in cp.stdout.splitlines():
            if line.startswith("worktree "):
                return Path(line[len("worktree "):].strip())
    except OSError:
        pass
    return Path(repo_root)


class Resolver:
    """把 `文件:行` 的路径部分解析成本机可读文件。

    只解析**带 `/` 的仓内相对路径**（含 `.tower/**`）。**裸文件名一律记为 unverified** ——
    唯一同名解析在 M39 已实测会拿错同名文件并静默通过（`docs/17` §9.2 触发事件 1），
    故本工具不采用该启发式（宁可少核，不拿错文件）。
    """

    def __init__(self, repo_root, tower_root):
        self.repo_root = Path(repo_root)
        self.tower_root = Path(tower_root)

    def resolve(self, path):
        """返回 (绝对 Path 或 None, 分类字符串)。"""
        if path is None or not re.search(r"[A-Za-z0-9_]", path):
            return None, "shorthand-no-path"
        if path.startswith("/"):
            return None, "absolute-external"
        if "/" in path:
            clean = re.sub(r"^\./", "", path)
            root = self.tower_root if clean.startswith(".tower/") else self.repo_root
            dm = re.match(r"^docs/(\d{1,2})$", clean)
            if dm:
                cands = sorted((self.repo_root / "docs").glob("%s-*.md" % dm.group(1)))
                cands += sorted((self.repo_root / "docs").glob("%s.md" % dm.group(1)))
                cands = [c for c in cands if c.is_file()]
                if len(cands) == 1:
                    return cands[0], "doc-shorthand"
                return None, ("doc-shorthand-ambiguous" if cands else "repo-path-missing")
            cand = root / clean
            if cand.exists():
                return cand, ("tower-path" if clean.startswith(".tower/") else "repo-path")
            return None, "repo-path-missing"
        if MODULE_SHORTHAND_RE.match(path):
            return None, "module-shorthand"
        return None, "bare-name"


def strip_emph(s):
    return s.replace("**", "")


def _segments_in_order(segs, hay):
    pos = 0
    for s in segs:
        if not s:
            continue
        i = hay.find(s, pos)
        if i < 0:
            return False
        pos = i + len(s)
    return True


def check_quote(raw, seg_text, file_text, region):
    """region = (start, end) 所引行在 file_text 里的字符区间。返回 (verdict, detail)。"""
    q = raw.strip()
    if not q:
        return "SKIP-EMPTY", "引文去空白后为空"
    if q in seg_text:
        return "OK", "逐字落在所引行"
    qe, se = strip_emph(q), strip_emph(seg_text)
    if qe and qe in se:
        return "OK-EMPH", "去掉 ** 后落在所引行"
    if ("…" in q) or ("..." in q):
        segs = [s.strip() for s in re.split(r"…|\.\.\.", q) if s.strip()]
        if segs:
            hay = se
            pos, ok = 0, True
            for s in segs:
                i = hay.find(strip_emph(s), pos)
                if i < 0:
                    ok = False
                    break
                pos = i + len(strip_emph(s))
            if ok:
                return "OK-PARTIAL", "省略号切段后各段按序在所引行命中"
            # 切段在所引行里不全 —— 再看整份被引文件：若各段按序能在文件里找到，
            # 那是「引文片段在文件里但不在所引行」（行号漂移），比笼统的 FIND-PARTIAL 更准。
            if file_text is not None and _segments_in_order([strip_emph(s) for s in segs], strip_emph(file_text)):
                return "FIND-ELSEWHERE", "省略号切段在所引行里不全；但各段按序在被引文件的其它位置"
            return "FIND-PARTIAL", "省略号切段后有所引行里找不到的段"
    if file_text is not None:
        for probe in (q, qe):
            if not probe:
                continue
            hay = file_text if probe == q else strip_emph(file_text)
            i = hay.find(probe)
            if i >= 0:
                if i < region[1] and i + len(probe) > region[0]:
                    return "OK-MULTILINE", "命中区间覆盖所引行（引文跨行）"
                return "FIND-ELSEWHERE", "引文在本文件里，但不在所引行上"
    return "FIND-NOTINFILE", "所引行与被引文件里都找不到该引文"


def check_quote_variants(spellings, seg_text, file_text, region):
    """对同一引文的多种拼法（原样 / 去掉换行缩进 / 换行折成空格）取**最优**判定。

    源侧跨行引文在本仓是「为显示折行」的写法（源处常是一整行），故拼接后再核。
    """
    best = None
    for s in spellings:
        v, d = check_quote(s, seg_text, file_text, region)
        if best is None or _rank(v) < _rank(best[0]):
            best = (v, d)
        if _rank(best[0]) == 0:
            break
    return best


def spellings_for(qtext):
    if "\n" not in qtext:
        return [qtext]
    # 折行的续行常带 markdown 引用标记 `> `（渲染产物，不是引文本身）—— 拼前先去掉。
    joined = "\n".join(re.sub(r"^[ \t]*>[ \t]?", "", ln) for ln in qtext.split("\n"))
    return [qtext,
            re.sub(r"[ \t]*\n[ \t]*", "", joined),
            re.sub(r"[ \t]*\n[ \t]*", " ", joined)]


def line_offsets(lines):
    offs, p = [], 0
    for ln in lines:
        offs.append(p)
        p += len(ln) + 1
    return offs


def find_crossline_quotes(text, lines, offs):
    """逐行 `QUOTE_RE` 看不见的**源侧跨行** `「…」`（开闭不在同一行）。

    逐行匹配只覆盖**单行内**闭合的引文；开 `「` 与闭 `」` 分处两行时两行都匹配不到 ⇒
    会从 `quote-unpaired` 分母里漏掉。本函数用**整份文本**匹配再映射回行号，把这一类捞出来。
    """
    out = []
    for m in QUOTE_RE.finditer(text):
        sl = bisect.bisect_right(offs, m.start())
        el = bisect.bisect_right(offs, m.end())
        if sl != el:
            out.append({"text": m.group(1), "start": m.start(), "end": m.end(),
                        "sl": sl, "el": el})
    return out


class Scan:
    def __init__(self, repo_root, docs_dir, tower_root=None):
        self.repo_root = Path(repo_root).resolve()
        self.docs_dir = Path(docs_dir).resolve()
        self.tower_root = Path(tower_root).resolve() if tower_root else main_worktree_root(self.repo_root)
        self.resolver = Resolver(self.repo_root, self.tower_root)
        self.file_cache = {}

    def _load(self, path):
        key = str(path)
        if key not in self.file_cache:
            text = read_text(path)
            self.file_cache[key] = text
        return self.file_cache[key]

    def _cite_target(self, cite, line):
        path = cite["path"]
        note = ""
        if path is None:
            # 裸行号 `:xx`：先回看同行最近一个带路径的 `文件:行`……
            prev = [c for c in find_cites(line[:cite["start"]]) if c["path"] is not None]
            if prev:
                path = prev[-1]["path"]
                note = "裸行号→%s" % path
            else:
                # ……否则回看同行最近的 `docs/NN` 提法（本仓常见：`` `docs/05` §2（`:19`） ``）。
                m = None
                for mm in re.finditer(r"docs/(\d{1,2})", line[:cite["start"]]):
                    m = mm
                if not m:
                    return None, "shorthand-no-path"
                path = "docs/%s" % m.group(1)
                note = "裸行号→%s（同行最近 docs/NN）" % path
        if cite["external"]:
            return None, "external-prefix"
        target, kind = self.resolver.resolve(path)
        if target is None:
            return None, kind
        return (target, note), kind

    def _pair_line(self, line, fname, lineno):
        quotes = find_quotes(line)
        cites = find_cites(line)
        pairs = []          # (quote, list[cite])
        quote_used = set()
        cite_used = set()
        # Rule A：同行紧邻（两个方向）
        for qi, q in enumerate(quotes):
            for ci, c in enumerate(cites):
                if c["end"] <= q["start"]:
                    gap = line[c["end"]:q["start"]]
                elif q["end"] <= c["start"]:
                    gap = line[q["end"]:c["start"]]
                else:
                    continue
                if SEP_RE.match(gap):
                    pairs.append((q, [c]))
                    quote_used.add(qi)
                    cite_used.add(ci)
        # Rule B：表内相邻单元格。只对**纯引文单元格**生效 ——
        # 该单元格去掉所有「…」后只剩空白/分隔符。这样「散文里顺带提到一句引文」的列
        # （如 docs/17:491 那格）不会被硬绑到相邻的出处列上，避免误报。
        if is_table_row(line) and cites:
            spans = cell_spans(line)
            cites_by_cell = {}
            for ci, c in enumerate(cites):
                cites_by_cell.setdefault(cell_index(spans, c["start"]), []).append(c)
            for qi, q in enumerate(quotes):
                if qi in quote_used:
                    continue
                qc = cell_index(spans, q["start"])
                if qc in cites_by_cell:
                    continue
                a, b = spans[qc]
                if not SEP_RE.match(QUOTE_RE.sub("", line[a:b])):
                    continue
                adj = []
                for nc in (qc - 1, qc + 1):
                    adj.extend(cites_by_cell.get(nc, []))
                if adj:
                    pairs.append((q, adj))
                    quote_used.add(qi)
        meta = {
            "file": fname,
            "line": lineno,
            "n_quotes": len(quotes),
            "n_cites": len(cites),
            "quotes_paired": len(quote_used),
            "cites_paired": len(cite_used),
            "cites_used_starts": [cites[ci]["start"] for ci in sorted(cite_used)],
        }
        return pairs, meta

    def _eval_pair(self, qtext, spellings, clist, line, fname, lineno):
        """判一对样本，返回 (record, bump_key)。record 的 verdict 恒有值（UNVERIFIED 也算一栏）。"""
        resolved = []
        skip_reasons = []
        for c in clist:
            got, kind = self._cite_target(c, line)
            if got is None:
                skip_reasons.append(kind)
            else:
                resolved.append((got, c, kind))
        base = {"file": fname, "line": lineno, "quote": qtext,
                "sources": ["%s:%s" % (c["path"] or "(裸行号)", c["lines"]) for c in clist]}
        if not resolved:
            reason = "/".join(sorted(set(skip_reasons))) or "unresolved"
            return dict(base, verdict="UNVERIFIED", reason=reason), "unverified:" + reason
        per_cite = []
        for (target, note), c, kind in resolved:
            ttext = self._load(target)
            if ttext is None:
                skip_reasons.append("target-read-fail")
                continue
            tlines = ttext.split("\n")
            idxs = []
            for a, b in parse_linespec(c["lines"]):
                for k in range(a, b + 1):
                    if 1 <= k <= len(tlines):
                        idxs.append(k)
            if not idxs:
                skip_reasons.append("line-out-of-range")
                continue
            if kind == "tower-path":
                note = (note + " " if note else "") + "目标在 .tower/（本机存在、git 未跟踪）"
            per_cite.append({"target": target, "c": c, "kind": kind, "note": note,
                             "ttext": ttext, "tlines": tlines, "idxs": idxs})
        verdicts = []
        for pc in per_cite:
            tlines, idxs = pc["tlines"], pc["idxs"]
            seg_text = "\n".join(tlines[k - 1] for k in sorted(set(idxs)))
            rstart = sum(len(x) + 1 for x in tlines[:min(idxs) - 1])
            rend = rstart + len(seg_text)
            verdict, detail = check_quote_variants(spellings, seg_text, pc["ttext"], (rstart, rend))
            verdicts.append({
                "verdict": verdict, "detail": detail,
                "source": "%s:%s" % (str(pc["target"]), pc["c"]["lines"]),
                "cite_kind": pc["kind"], "note": pc["note"],
            })
        # 同一份被引文件内、多条 `文件:行` 并列为同一引文来源时（本仓写法
        # 「`:18`（同 `:20`）」），引文可能**横跨所列各行的合并文本** ——
        # 再对「本对样本所引行的合并」判一次，取更优的那个判定。
        if len(per_cite) > 1 and len({str(p["target"]) for p in per_cite}) == 1:
            merged = []
            for pc in per_cite:
                merged.extend(sorted(set(pc["idxs"])))
            allidx = sorted(set(merged))
            tl = per_cite[0]["tlines"]
            seg_text = "\n".join(tl[k - 1] for k in allidx)
            rstart = sum(len(x) + 1 for x in tl[:min(allidx) - 1])
            rend = rstart + len(seg_text)
            verdict, detail = check_quote_variants(spellings, seg_text, per_cite[0]["ttext"],
                                                   (rstart, rend))
            verdicts.append({
                "verdict": verdict,
                "detail": detail + "（本对样本所列多行的合并文本）",
                "source": "%s（合并 %s）" % (str(per_cite[0]["target"]),
                                           "/".join(pc["c"]["lines"] for pc in per_cite)),
                "cite_kind": "merged-lines", "note": "",
            })
        if not verdicts:
            reason = "/".join(sorted(set(skip_reasons))) or "unresolved"
            return dict(base, verdict="UNVERIFIED", reason=reason), "unverified:" + reason
        worst = min(verdicts, key=lambda v: _rank(v["verdict"]))
        rec = dict(base, verdict=worst["verdict"], detail=worst["detail"],
                   matched=worst["source"], cite_kind=worst["cite_kind"],
                   note=worst["note"], n_alternatives=len(verdicts))
        key = ("ok:" if _rank(worst["verdict"]) <= 1 else "finding:") + worst["verdict"]
        return rec, key

    def run(self):
        docs = sorted(p for p in self.docs_dir.glob("*.md") if p.is_file())
        out = {
            "docs_scanned": len(docs),
            "pairs": [],
            "skips": {},
            "files": [],
            "quote_total": 0, "quote_bound": 0, "quote_unpaired": 0,
            "quote_crossline": 0, "quote_crossline_bound": 0,
            "cite_total": 0, "cite_bound": 0, "cite_unpaired": 0,
        }
        cite_keys = set()

        def bump(k):
            out["skips"][k] = out["skips"].get(k, 0) + 1

        for doc in docs:
            text = self._load(doc)
            if text is None:
                out["files"].append((doc.name, "READ-FAIL"))
                continue
            lines = text.split("\n")
            offs = line_offsets(lines)
            out["files"].append((doc.name, "scanned"))
            cross = find_crossline_quotes(text, lines, offs)
            out["quote_crossline"] += len(cross)
            out["quote_total"] += len(cross)
            for i, line in enumerate(lines, 1):
                if "「" not in line and ":" not in line:
                    continue
                pairs, meta = self._pair_line(line, doc.name, i)
                out["quote_total"] += meta["n_quotes"]
                out["quote_bound"] += meta["quotes_paired"]
                out["cite_total"] += meta["n_cites"]
                for st in meta["cites_used_starts"]:
                    cite_keys.add((doc.name, i, st))
                for q, clist in pairs:
                    rec, key = self._eval_pair(q["text"], [q["text"]], clist, line, doc.name, i)
                    out["pairs"].append(rec)
                    bump(key)
            # 源侧跨行 `「…」`：开行左侧**最近一个** `文件:行` 作为出处；没有则只计入
            # `quote-unpaired`（可见的分母），不静默。
            for cq in cross:
                oline = lines[cq["sl"] - 1]
                col = cq["start"] - offs[cq["sl"] - 1]
                left = [c for c in find_cites(oline) if c["end"] <= col]
                if not left:
                    continue
                c = left[-1]
                out["quote_crossline_bound"] += 1
                out["quote_bound"] += 1
                cite_keys.add((doc.name, cq["sl"], c["start"]))
                rec, key = self._eval_pair(cq["text"], spellings_for(cq["text"]), [c],
                                           oline, doc.name, cq["sl"])
                rec["crossline"] = "%s:%d-%d（源侧跨行引文，按拼接后核）" % (doc.name, cq["sl"], cq["el"])
                out["pairs"].append(rec)
                bump(key)
        out["quote_unpaired"] = out["quote_total"] - out["quote_bound"]
        out["cite_bound"] = len(cite_keys)
        out["cite_unpaired"] = out["cite_total"] - out["cite_bound"]
        return out


_ORDER = {"OK": 0, "OK-EMPH": 1, "OK-PARTIAL": 1, "OK-MULTILINE": 1,
          "UNVERIFIED": 2, "FIND-ELSEWHERE": 3, "FIND-PARTIAL": 3, "FIND-NOTINFILE": 3}


def _rank(v):
    return _ORDER.get(v, 2)


def is_finding(v):
    return v.startswith("FIND")


def summarize(res):
    findings = [p for p in res["pairs"] if is_finding(p["verdict"])]
    unverified = [p for p in res["pairs"] if p["verdict"] == "UNVERIFIED"]
    verified = [p for p in res["pairs"] if _rank(p["verdict"]) <= 1]
    return verified, unverified, findings


def print_report(res, dump=False):
    verified, unverified, findings = summarize(res)
    print("== docs 引文保真扫描（scan_quote_refs.py） ==")
    print("docs scanned=%d  read-fail=%d" % (res["docs_scanned"],
                                             sum(1 for _, s in res["files"] if s == "READ-FAIL")))
    print("成对样本(pairs)=%d  verified=%d  unverified=%d  finding=%d"
          % (len(res["pairs"]), len(verified), len(unverified), len(findings)))
    print("引文「…」总数=%d  其中成对受检=%d  未成对(计入 quote-unpaired)=%d"
          % (res["quote_total"], res["quote_bound"], res["quote_unpaired"]))
    print("源侧跨行引文(开闭不在同一行)=%d ｜ 开行有出处→成对检查=%d ｜ 无→计入 quote-unpaired=%d"
          % (res["quote_crossline"], res["quote_crossline_bound"],
             res["quote_crossline"] - res["quote_crossline_bound"]))
    print("文件:行 总数=%d  其中成对受检=%d  未成对(计入 cite-unpaired)=%d"
          % (res["cite_total"], res["cite_bound"], res["cite_unpaired"]))
    if res["skips"]:
        print("-- 分类计数（分母 = 上面的成对样本 %d） --" % len(res["pairs"]))
        for k in sorted(res["skips"]):
            print("   %-34s %d" % (k, res["skips"][k]))
    if findings:
        print("-- finding（逐条：判据 + 出处） --")
        for p in findings:
            print("   FIND %s:%d [%s]%s" % (p["file"], p["line"], p["verdict"],
                                           ("  " + p["crossline"]) if p.get("crossline") else ""))
            print("        引文：%s" % _clip(p["quote"]))
            print("        所引：%s" % "; ".join(p["sources"]))
            print("        判据：%s%s" % (p.get("detail", ""),
                                        ("  ｜" + p["note"]) if p.get("note") else ""))
    if dump:
        print("-- verified（逐条：判定 + 命中处） --")
        for p in verified:
            print("   %-14s %s:%d%s  引文=%s  命中=%s"
                  % (p["verdict"], p["file"], p["line"],
                     ("（跨行）" if p.get("crossline") else ""),
                     _clip(p["quote"]), p.get("matched", "")))
    if unverified and dump:
        print("-- unverified（未抽为 finding 的成对样本） --")
        for p in unverified:
            print("   UNVERIFIED %s:%d%s  所引=%s  原因=%s"
                  % (p["file"], p["line"], ("（跨行）" if p.get("crossline") else ""),
                     "; ".join(p["sources"]), p.get("reason", "")))
    if dump:
        print("-- 全部成对样本 --")
        for p in res["pairs"]:
            print("   %-14s %s:%d  引文=%s" % (p["verdict"], p["file"], p["line"], _clip(p["quote"])))
    return len(findings)


def _clip(s, n=64):
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"


def tower_inventory(repo_root, docs_dir, tower_root=None):
    docs = sorted(p for p in Path(docs_dir).glob("*.md") if p.is_file())
    tower_root = Path(tower_root).resolve() if tower_root else main_worktree_root(repo_root)
    tracked = set()
    try:
        cp = subprocess.run(["git", "-C", str(tower_root), "ls-files", ".tower"],
                            capture_output=True, text=True)
        tracked = set(cp.stdout.split("\n"))
    except OSError:
        pass
    rows = []
    for doc in docs:
        text = read_text(doc)
        if text is None:
            continue
        for i, line in enumerate(text.split("\n"), 1):
            # `*`（不是 `+`）：把**裸 `.tower/`**（其后无路径字符）也计进来 —— 否则少计一处。
            for m in re.finditer(r"\.tower/[A-Za-z0-9_./-]*", line):
                tok = m.group(0)
                if tok != ".tower/":
                    tok = tok.rstrip("/")
                bare = (tok == ".tower/")
                p = Path(tower_root) / tok
                rows.append({
                    "file": doc.name,
                    "line": i,
                    "token": tok,
                    "bare": bare,
                    "exists": p.exists(),
                    "tracked": tok in tracked,
                })
    return rows


def print_tower_inventory(rows, repo_root, tower_root=None):
    tower_root = Path(tower_root).resolve() if tower_root else main_worktree_root(repo_root)
    n_bare = sum(1 for r in rows if r["bare"])
    n_path = len(rows) - n_bare
    print("== docs 里 `.tower/` 路径引用清单（事实；可达性由塔裁决） ==")
    print("被扫 docs：%s" % repo_root)
    print("`.tower/` 所在主检出：%s" % tower_root)
    print("引用处数(occurrences)=%d ｜ 带路径 %d + 裸 `.tower/` %d ｜ 去重路径(distinct)=%d"
          % (len(rows), n_path, n_bare, len({r["token"] for r in rows})))
    print()
    print("| 文件:行 | `.tower/` 路径 | 本机是否存在 | git ls-files 是否跟踪 |")
    print("|---|---|---|---|")
    for r in rows:
        print("| %s:%d | `%s` | %s | %s |"
              % (r["file"], r["line"], r["token"],
                 "是" if r["exists"] else "否", "是" if r["tracked"] else "否"))
    print()
    try:
        cp = subprocess.run(["git", "-C", str(tower_root), "ls-files", ".tower"],
                            capture_output=True, text=True)
        n = len([x for x in cp.stdout.split("\n") if x])
        print("`git ls-files .tower` 输出行数 = %d" % n)
    except OSError:
        print("`git ls-files .tower` 无法执行")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------

M125_PREFIX_ROW = (
    "| 评审反复给出的合并建议 | 「带登记残留合并」「由作者下次顺手改 1 行，不需要再走完整复审」 | "
    "`.tower/comms/reviews/review-feat-m116-b2-attention-prefill-front-end-and-reviewer-m116-r3.md:107`；"
    "`.tower/comms/reviews/review-feat-m123-scalar-computation-sweep-reviewer-m123-r2.md:88` |"
)
M125_FIXED_ROW = (
    "| 评审反复给出的合并建议 | 「带登记残留合并」；"
    "「可由作者下一次顺手改成 1 行文字——**不需要再走一轮完整复审**」 | 前者："
    "`.tower/comms/reviews/review-feat-m116-b2-attention-prefill-front-end-and-reviewer-m116-r3.md:107`"
    "（逐字）、`.tower/comms/reviews/review-feat-m123-scalar-computation-sweep-reviewer-m123-r2.md:88`"
    "（作「带 p2-1/p2-2 登记残留合并」）；后者："
    "`.tower/comms/reviews/review-feat-m115-b1-gdn-prefill-chunk-scan-reviewer-l0sync-r5.md:70`（逐字） |"
)


def _run_on(docs_dir, repo_root):
    scan = Scan(repo_root, docs_dir)
    return scan.run()


def selftest():
    checks = []

    def add(name, ok, detail):
        checks.append((name, ok, detail))

    # ---- 合成正/负对照（临时 mini-repo，不与真实 docs 语料耦合） ----
    with tempfile.TemporaryDirectory(prefix="quote_refs_st") as td:
        mini = Path(td)
        (mini / "docs").mkdir()
        (mini / "targets").mkdir()
        (mini / "targets" / "sample.txt").write_text(
            "alpha beta gamma\nsecond line holds the expected phrase omega here\n", encoding="utf-8")
        (mini / "docs" / "good.md").write_text(
            "# good\n| item | 「the expected phrase omega」 | `targets/sample.txt:2` |\n", encoding="utf-8")
        (mini / "docs" / "bad.md").write_text(
            "# bad\n| item | 「a phrase that is not there at all」 | `targets/sample.txt:2` |\n", encoding="utf-8")
        good = _run_on(mini / "docs", mini)
        bad = _run_on(mini / "docs", mini)
        # same dir contains both; separate them by file name
        g_pairs = [p for p in good["pairs"] if p["file"] == "good.md"]
        b_pairs = [p for p in bad["pairs"] if p["file"] == "bad.md"]
        add("NC-1 合成引文正确 ⇒ 判绿",
            g_pairs and not is_finding(g_pairs[0]["verdict"]),
            "verdict=%s" % (g_pairs[0]["verdict"] if g_pairs else "no-pair"))
        add("NC-2 合成引文改错 ⇒ 判红",
            b_pairs and is_finding(b_pairs[0]["verdict"]),
            "verdict=%s" % (b_pairs[0]["verdict"] if b_pairs else "no-pair"))

        # 盲区登记：省略号引文 / 裸行号 / 外部前缀 / 裸文件名同名多处 / 引文无紧邻出处
        (mini / "docs" / "blind.md").write_text(
            "# blind\n"
            "| a | 「second line holds … omega here」 | `targets/sample.txt:2` |\n"
            "| b | 「alpha beta gamma」 | `:1` |\n"
            "| c | `[asc-main] targets/sample.txt:1`「alpha beta gamma」 |\n"
            "| d | 「alpha beta gamma」 | `sample.txt:1` |\n"
            "| e | 「unpaired label」 | plain text with no citation |\n",
            encoding="utf-8")
        blind = _run_on(mini / "docs", mini)
        bp = {p["line"]: p for p in blind["pairs"] if p["file"] == "blind.md"}
        add("NC-3 省略号引文按段核（可核时给 OK-PARTIAL）",
            bp.get(2, {}).get("verdict") == "OK-PARTIAL",
            "verdict=%s" % bp.get(2, {}).get("verdict"))
        add("NC-4 裸行号且同行无前置路径 ⇒ unverified（不当通过）",
            bp.get(3, {}).get("verdict") == "UNVERIFIED",
            "verdict=%s reason=%s" % (bp.get(3, {}).get("verdict"), bp.get(3, {}).get("reason")))
        add("NC-5 外部前缀 ⇒ unverified（不当通过）",
            bp.get(4, {}).get("verdict") == "UNVERIFIED"
            and bp.get(4, {}).get("reason") == "external-prefix",
            "verdict=%s reason=%s" % (bp.get(4, {}).get("verdict"), bp.get(4, {}).get("reason")))
        add("NC-6 裸文件名 ⇒ unverified（不按唯一同名猜测，避免 M39 的拿错文件）",
            bp.get(5, {}).get("verdict") == "UNVERIFIED"
            and bp.get(5, {}).get("reason") == "bare-name",
            "verdict=%s reason=%s" % (bp.get(5, {}).get("verdict"), bp.get(5, {}).get("reason")))
        add("NC-7 引文无紧邻出处 ⇒ 不抽为成对样本（盲区可见，不是静默通过）",
            all(6 not in (p["line"],) for p in blind["pairs"])
            and blind["quote_unpaired"] >= 1,
            "pairs=%d quote-unpaired=%d" % (len(blind["pairs"]), blind["quote_unpaired"]))

        # 解析档：docs/NN 简写应能解析；引文在文件里但不在所引行 ⇒ FIND-ELSEWHERE
        (mini / "docs" / "99-sample.md").write_text("shorthand target line one here\n", encoding="utf-8")
        (mini / "docs" / "usen.md").write_text(
            "# usen\n| item | 「shorthand target line one here」 | `docs/99:1` |\n", encoding="utf-8")
        (mini / "docs" / "elsewhere.md").write_text(
            "# elsewhere\n| item | 「second line holds the expected phrase omega」 | `targets/sample.txt:1` |\n",
            encoding="utf-8")
        more = _run_on(mini / "docs", mini)
        rp = {p["file"]: p for p in more["pairs"]}
        add("NC-11 `docs/NN:line` 简写应解析（正确引文判绿，非 unverified）",
            rp.get("usen.md") and rp["usen.md"]["verdict"] == "OK",
            "verdict=%s" % (rp.get("usen.md", {}).get("verdict")))
        add("NC-12 引文在被引文件里但不在所引行 ⇒ FIND-ELSEWHERE（不误记为 OK）",
            rp.get("elsewhere.md", {}).get("verdict") == "FIND-ELSEWHERE",
            "verdict=%s" % rp.get("elsewhere.md", {}).get("verdict"))

    # ---- 源侧跨行 `「…」`：进入检查 or 计入分母，不得静默零输出 ----
    with tempfile.TemporaryDirectory(prefix="quote_refs_xline") as td:
        x = Path(td)
        (x / "docs").mkdir()
        (x / "targets").mkdir()
        (x / "targets" / "sample.txt").write_text(
            "alpha beta gamma\nsecond line holds the expected phrase omega here\n", encoding="utf-8")
        (x / "docs" / "xline.md").write_text(
            "# xline\n"
            "`targets/sample.txt:2` 逐字：「second line holds the expected\nphrase omega here」\n"
            "`targets/sample.txt:2` 逐字：「a phrase that is not on that line at\nall nothing here」\n"
            "「a cross line quote with no citation\nanywhere on its opening line」\n",
            encoding="utf-8")
        xr = _run_on(x / "docs", x)
        xp = xr["pairs"]
        xok = [p for p in xp if p.get("crossline") and not is_finding(p["verdict"])]
        xbad = [p for p in xp if p.get("crossline") and is_finding(p["verdict"])]
        add("NC-13 源侧跨行引文进入检查（正确判绿、改错判红，不是静默零输出）",
            len(xp) == 2 and len(xok) == 1 and len(xbad) == 1,
            "crossline_records=%d ok=%d find=%d" % (len(xp), len(xok), len(xbad)))
        add("NC-14 源侧跨行且开行无出处 ⇒ 计入 quote-unpaired 分母（不是零输出）",
            xr["quote_crossline"] == 3 and xr["quote_crossline_bound"] == 2
            and xr["quote_unpaired"] == 1,
            "crossline=%d bound=%d quote-unpaired=%d"
            % (xr["quote_crossline"], xr["quote_crossline_bound"], xr["quote_unpaired"]))

    # ---- M125 B1 真实回归样例（外层用真实 repo_root 解析 .tower/**） ----
    with tempfile.TemporaryDirectory(prefix="quote_refs_m125") as td:
        tdocs = Path(td) / "docs"
        tdocs.mkdir()
        (tdocs / "prefix.md").write_text("# M125 pre-fix\n" + M125_PREFIX_ROW + "\n", encoding="utf-8")
        (tdocs / "fixed.md").write_text("# M125 post-fix\n" + M125_FIXED_ROW + "\n", encoding="utf-8")
        pre = _run_on(tdocs, REPO_ROOT)
        # 分开跑：先 pre-fix
        pre_pairs = [p for p in pre["pairs"] if p["file"] == "prefix.md"]
        pre_findings = [p for p in pre_pairs if is_finding(p["verdict"])]
        add("NC-8 M125 修前版本 ⇒ 判红（引文挂错出处）",
            len(pre_findings) >= 1,
            "findings=%d %s" % (len(pre_findings), pre_findings[0]["quote"][:24] if pre_findings else ""))
        # 再 post-fix：把 prefix.md 拿走，只留 fixed.md
        (tdocs / "prefix.md").unlink()
        post = _run_on(tdocs, REPO_ROOT)
        post_pairs = [p for p in post["pairs"] if p["file"] == "fixed.md"]
        post_findings = [p for p in post_pairs if is_finding(p["verdict"])]
        add("NC-9 M125 修后版本 ⇒ 判绿（且确有被核的成对样本）",
            len(post_pairs) >= 1 and len(post_findings) == 0,
            "pairs=%d findings=%d" % (len(post_pairs), len(post_findings)))
        add("NC-10 M125 修后样本的引文确实被核到 l0sync-r5.md:70",
            any("l0sync-r5.md:70" in (p.get("matched", "") + " ".join(p.get("sources", [])))
                for p in post_pairs),
            "sources=%s" % [p.get("sources") for p in post_pairs])

    fails = [c for c in checks if not c[1]]
    print("== scan_quote_refs.py --selftest ==")
    for name, ok, detail in checks:
        print("  [%s] %s   (%s)" % ("PASS" if ok else "FAIL", name, detail))
    print("对照总数=%d  成立=%d  不成立=%d" % (len(checks), len(checks) - len(fails), len(fails)))
    if not checks:
        return 2
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="docs 引文保真扫描器（M129）")
    ap.add_argument("--docs-dir", default=str(REPO_ROOT / "docs"))
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--tower-root", default=None,
                    help="`.tower/**` 所在的主检出（默认经 git worktree list 自动取）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-pairs", action="store_true")
    ap.add_argument("--tower-refs", action="store_true")
    args = ap.parse_args(argv)

    tower_root = args.tower_root or main_worktree_root(args.repo_root)

    if args.selftest:
        return selftest()

    docs = sorted(Path(args.docs_dir).glob("*.md"))
    if not docs:
        print("没得比：%s 下找不到 docs/*.md" % args.docs_dir, file=sys.stderr)
        return 2

    if args.tower_refs:
        print_tower_inventory(tower_inventory(args.repo_root, args.docs_dir, tower_root),
                              args.repo_root, tower_root)
        return 0

    scan = Scan(args.repo_root, args.docs_dir, tower_root)
    res = scan.run()
    n_find = print_report(res, dump=args.dump_pairs)
    return 1 if n_find else 0


if __name__ == "__main__":
    sys.exit(main())
