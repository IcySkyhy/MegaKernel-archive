#!/usr/bin/env python3
"""M125 —— docs/17 §9.7–§9.11 触发事件「引文 ↔ 出处」逐条复核器（清单 + 判定，一份文件）。

背景：r2 复审指出作者在复审请求里报的「28/28 严格子串」口径不可第三方复现。
本文件把那次复扫的 `(path, lineno, substring)` 清单**落成仓内工件**，并给出
**可照跑的 N/M/K 口径**，让第三方按同一清单得到同一组数。

抽取方法（人抽，一次抽一份清单；runner 只做判定）：
  逐行读 `docs/17-verification-standard.md` §9.7–§9.11 的五张触发事件表，取每张表里
  **对源文加了「」引号的那一段文字**作为一个 `(src_path, src_lineno, quote)` 条目。
  - 同一格引了多个来源时，按来源拆成多条（如 §9.9 表末行拆成 3 条）。
  - §9.7 的表**不含对源文的「」引文**（只有读数值与释义），故不进清单。
  - 跨行的引文（§9.10 表 r2/r3）把所引的每个行号都列出，按「所引行的拼接」判。

判定（每个条目落进唯一一栏，三栏互斥 —— 与 docs/17 §9.2「三态分栏」同族）：
  N = **逐字连续子串**：quote 原文（`substring in line`）出现在所引某一行的原文里；
  M = **归一化后吻合**：quote 不是逐字连续子串，但把 `*`/反引号/`/`/空白去掉、
      并把全角 `：` 折成半角后，是所引行（拼接）的子串；
  K = **已显式标注的省略或脱敏**：quote 里带 `…`（省略标记）或 `「…」`（脱敏），
      把它按 `…` 切开后每一段（同样归一化后）都能在所引行（拼接）里按序找到。
  `MISMATCH` = 三栏都落不进的条目 —— **应当为 0**；非 0 表示引文与出处对不上，要改文档。

用法（在仓库根跑）：
    python3 m27_hc_prefill/evidence/m125_citation_check.py
退出码照 docs/17 §8.3 三态：0 = 比过且无 MISMATCH；2 = 源文件缺失（没得比，不发合格证）。

注：源文件是塔内 `.tower/comms/**` 的评审/收件归档，只在塔工作区存在；本清单核对的是
「引文是否逐字/per-marked 地出现在所引的 `.tower` 行上」。
"""
import os
import subprocess
import sys


def _main_root():
    """本脚本可能在 worktree 里跑，而 `.tower/**` 只在主检出下；用 git-common-dir 找回主检出根。"""
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return os.path.dirname(os.path.abspath(common))
    except Exception:
        return os.getcwd()


MAIN_ROOT = _main_root()


def resolve(path):
    for cand in (path, os.path.join(MAIN_ROOT, path)):
        if os.path.exists(cand):
            return cand
    return None

# (anchor, [(src_path, src_lineno), ...], quote)
MANIFEST = [
    ("§9.8 表 r1", [(".tower/comms/inbox/20260927-tower-all-flock-120s-w-300s-flock-n.md", 26)],
     """ "等锁没拿到"必须报成"未取得读数"，不得报成"没复现" """.strip()),
    ("§9.8 表 r2", [(".tower/comms/inbox/20260927-tower-all-item-2.md", 16)],
     """等不到就放弃这一档，如实报"未取得读数"（不是"没复现"）"""),
    ("§9.8 表 r3", [(".tower/comms/reviews/review-feat-m119-b5-hc-prefill-mtile-reviewer-m119-r3.md", 67)],
     """未跑档如实: sinkreloc/ONLY_IJFLAT/ijfromoh 设备半记'未取得读数'"""),
    ("§9.8 表 r4", [("m27_hc_prefill/README.md", 442)],
     """本轮**未取得读数**：探锁 `flock -n` 返回忙"""),
    ("§9.9 表 r1", [(".tower/comms/reviews/review-feat-m123-scalar-computation-sweep-reviewer-m123-r3.md", 54)],
     """## p2-1（纯文字/自洽）—— :1332 用了「…」这个禁用的字面"""),
    ("§9.9 表 r3", [(".tower/comms/inbox/20261004-reviewer-m101-agent-attncore-review-result-r3-p2-1items-merge-3-1-residual.md", 35)],
     """不算 finding，只作记录"""),
    ("§9.9 表 r4a", [(".tower/comms/reviews/review-feat-m116-b2-attention-prefill-front-end-and-reviewer-m116-r3.md", 107)],
     """带登记残留合并"""),
    ("§9.9 表 r4b", [(".tower/comms/reviews/review-feat-m115-b1-gdn-prefill-chunk-scan-reviewer-l0sync-r5.md", 70)],
     """可由作者下一次顺手改成 1 行文字——**不需要再走一轮完整复审**"""),
    ("§9.9 表 r4c", [(".tower/comms/reviews/review-feat-m123-scalar-computation-sweep-reviewer-m123-r2.md", 88)],
     """带 p2-1/p2-2 登记残留合并"""),
    ("§9.10 表 r1", [(".tower/comms/inbox/20261004-reviewer-m116-agent-b2attn-review-result-r3-m116-p2-5items-a-0.md", 15)],
     """**按塔要的两分类**：**A 类（影响判定的）= 0 条；B 类（纯文字 / 自洽 + 1 条验证设计残留）= 5 条，每条都不影响本 mission 任何结论**"""),
    ("§9.10 表 r2", [(".tower/comms/reviews/review-feat-m123-scalar-computation-sweep-reviewer-m123-r2.md", 18),
                     (".tower/comms/reviews/review-feat-m123-scalar-computation-sweep-reviewer-m123-r2.md", 20)],
     """**影响判定的：0 条。**…**纯文字 / 自洽的：2 条**…两条的明确判断都是 —— **不影响本文件任何结论**"""),
    ("§9.10 表 r3", [(".tower/comms/reviews/review-feat-m116-b2-attention-prefill-front-end-and-reviewer-m116-r5.md", 18),
                     (".tower/comms/reviews/review-feat-m116-b2-attention-prefill-front-end-and-reviewer-m116-r5.md", 20)],
     """## A 类（影响判定的）—— **0 条** / ## B 类（纯文字 / 自洽）—— 1 条"""),
    ("§9.10 表 r4", [(".tower/comms/reviews/review-feat-m123-scalar-computation-sweep-reviewer-m123-r3.md", 62)],
     """**是否影响本文件任何结论：否**"""),
    ("§9.11 表 r1", [(".tower/comms/reviews/review-feat-m119-b5-hc-prefill-mtile-reviewer-m119-r2.md", 33)],
     """**已解决为「举证」**（**门限未被改松**）"""),
    ("§9.11 表 r2", [(".tower/comms/reviews/review-feat-m119-b5-hc-prefill-mtile-reviewer-m119-r2.md", 34)],
     """`m20_hyperconn/check_ref.py` 全分支一字未改（sha256 = `ab56de0a…`，…）；m27 `check_ref.py` 的容差常量 `1e-6`…/`1e-4`…未变"""),
    ("§9.11 表 r3", [(".tower/comms/reviews/review-feat-m119-b5-hc-prefill-mtile-reviewer-m119-r2.md", 37)],
     """口径变更是**"提出、交复审/塔复核"**，不是单方面落地 —— 符合塔的「举证，不许直接改松」"""),
    ("§9.11 表 r4", [(".tower/comms/reviews/review-feat-m119-b5-hc-prefill-mtile-reviewer-m119-r3.md", 24)],
     """**没偷偷改门限 / 判定语义**：本轮 `74eee0e..34f439b` **未碰** `check_ref.py`…值级红走的是**既有 `judge`**，没有新门限"""),
]

_DROP = ["*", "`", "/", " ", "\u3000", "\t", "\n", "\r"]


def norm(s: str) -> str:
    s = s.replace("：", ":")
    for ch in _DROP:
        s = s.replace(ch, "")
    return s


def classify(lines, quote):
    joined = "\n".join(lines)
    if any(quote in ln for ln in lines):
        return "N"
    if "…" not in quote and norm(quote) in norm(joined):
        return "M"
    if "…" in quote:
        frags = [f for f in quote.split("…") if f]
        pos = 0
        ok = True
        hay = norm(joined)
        for f in frags:
            idx = hay.find(norm(f), pos)
            if idx < 0:
                ok = False
                break
            pos = idx + len(norm(f))
        return "K" if ok else "MISMATCH"
    return "MISMATCH"


def main():
    missing = []
    buckets = {"N": [], "M": [], "K": [], "MISMATCH": []}
    cache = {}
    for anchor, srcs, quote in MANIFEST:
        lines = []
        for path, lineno in srcs:
            key = path
            if key not in cache:
                real = resolve(path)
                cache[key] = None if real is None else open(real, encoding="utf-8").read().splitlines()
                if cache[key] is None:
                    missing.append(path)
            if cache[key] is None or lineno > len(cache[key]):
                print(f"[SKIP] {anchor}: 缺 {path}:{lineno}")
                continue
            lines.append(cache[key][lineno - 1])
        if not lines:
            continue
        kind = classify(lines, quote)
        buckets[kind].append((anchor, srcs, quote))
        tag = {"N": "逐字", "M": "归一化", "K": "已标省略/脱敏", "MISMATCH": "!!! 对不上"}[kind]
        loc = ", ".join(f"{p}:{n}" for p, n in srcs)
        print(f"[{kind:8}] {tag:10} {anchor:12} -> {loc}")

    if missing:
        print(f"RESULT: SKIPPED (源文件缺失，没得比：{sorted(set(missing))})", file=sys.stderr)
        return 2

    n, m, k, bad = len(buckets["N"]), len(buckets["M"]), len(buckets["K"]), len(buckets["MISMATCH"])
    total = n + m + k + bad
    print()
    print(f"清单条目 = {total}（N 逐字连续子串 = {n} / M 归一化后吻合 = {m} / K 已标省略或脱敏 = {k} / MISMATCH = {bad}）")
    if bad:
        for anchor, srcs, _ in buckets["MISMATCH"]:
            print(f"  MISMATCH: {anchor} -> {srcs}")
        print(f"RESULT: FAIL ({bad} 条引文与出处对不上)")
        return 1
    print("RESULT: OK (引文清单逐条落进 N/M/K 三栏，MISMATCH = 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
