#!/usr/bin/env python3
"""M114 影响面量尺：仓内 `*.md` 的 `§`-号章节引用，按**当前扫描器的实际处置**分栏。

用法（在仓根跑）：`python3 docs/evidence/scan_doc_refs/measure_sec_refs.py [repo_root] [scanner_path]`
  * `scanner_path` 默认 `docs/scan_doc_refs.py`；量「改前」时传 `git show HEAD:docs/scan_doc_refs.py` 落盘的副本
    （`run_repro` 里就是这么做的）⇒ 改前/改后两份 log 除"口径一致性"那一行外应**逐字相同**。

匹配器与 `docs/scan_doc_refs.py` **同源**（import 它，不另写正则）—— §9.2 第 1 条：分子分母同源。
**本脚本只统计，不推断**：凡是"这条 `§` 指向哪个文档"要靠猜的地方，就**不猜**，改成报该号的歧义普查
（裸 `§` 有歧义正是扫描器把它登记为盲区第 ④ 栏的理由；猜出来的目标会把读数变成假结论）。

分域（决定读数怎么读）：
  * `gate`    = 主扫描真正判的文件 = `docs/*.md` 去掉 `SELF`（`docs/17`）。**只有这一域决定 rc**；
  * `self`    = `docs/17` 自己（默认不扫；它的正文里逐字引着负向对照样本）；
  * `outside` = 其余 tracked `*.md`（各 `m*/README.md` 等）—— 不在 `docs/*.md` 扫描面上。

处置分栏（只按**同行可见的证据**判）：
  * `parsed`     —— 落在 `SECREF` 命中里 ⇒ 进 `section refs` 分母，会被判 out-of-range（目标唯一）；
  * `gap_narrow` —— `§` 之前同行最近的 `docs/NN` token 与它之间只夹了一段**短 ASCII 夹层**
                    （形如 `docs/05-….md v1.2 §2`，即扫描器 docstring 登记的第 ⑤ 栏）。夹层写成
                    `[ \\t]*[A-Za-z0-9_.\\-/~]{0,12}[ \\t]*` 且**不含第二个 `§`** —— 比"同行有任何路径就算"
                    紧得多，用来**对齐 M78 的历史读数（2 处 = 2 行）**当标定；
  * `other`      —— 其余（同行 `§` 前没有 `docs/NN` token，或夹层不是上面那种）。这一栏**不推断目标**，
                    只给「该号在几个文档里是标题」的普查（`0` / `1` / `>=2`；`>=2` = 真歧义）。

标题判定两套口径并列，用于量出「M114 修 vs 不修」的差：
  * `old` = `^<num>(?!\\d)`   —— M114 之前的实现（写死在 `scan_docrefs()` 里）；
  * `new` = `^§?<num>(?!\\d)` —— M114 落进 `scan_doc_refs.py` 的口径。
  两者在模块里若都有对应实现，本脚本会核对是否一致（防量尺与工具口径漂移）。

**本 mission 自己写在本目录下的证据 `.md` 不计入语料**（`SELF_EVIDENCE`）：那些文字是**引用样本的转录**
（满篇 `§X`），把它算进来会让"扫描面外"那一栏的读数随本报告的措辞漂移 —— 读数必须只取决于语料。
"""

import bisect
import collections
import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
SCANNER = sys.argv[2] if len(sys.argv) > 2 else 'docs/scan_doc_refs.py'
os.chdir(ROOT)

_spec = importlib.util.spec_from_file_location('scan_doc_refs', SCANNER)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

SEC_MARK = re.compile(r'§\s*([0-9]+(?:\.[0-9]+)*)')
# 本 mission 自己的证据目录（其 `.md` 是引用样本的转录，不算语料；见 docstring 末段）
SELF_EVIDENCE = 'docs/evidence/scan_doc_refs/'
HEAD_SIGN = re.compile(r'^§')
GAP = re.compile(r'[ \t]*[A-Za-z0-9_.\-/~]{0,12}[ \t]*$')


def head_old(head, sec):
    return bool(re.match(rf'^{re.escape(sec)}(?!\d)', head))


def head_new(head, sec):
    return bool(re.match(rf'^§?{re.escape(sec)}(?!\d)', head))


def domains(tracked):
    out = collections.defaultdict(list)
    for f in tracked:
        if os.path.dirname(f) != 'docs':
            out['outside'].append(f)
        elif f == S.SELF:
            out['self'].append(f)
        else:
            out['gate'].append(f)
    return out


def check_module_agrees(heads):
    fn = getattr(S, '_has_section', None)
    if fn is None:
        return 'N/A（模块里还没有 _has_section）'
    bad = [(f, h, sec) for f, hs in heads.items() for h in hs
           for sec in ('1', '3.2', '0.1', '99.7', '10')
           if bool(fn(h, sec)) != head_new(h, sec)]
    return 'AGREE' if not bad else f'MISMATCH {bad[:3]}'


def main():
    all_md = subprocess.run(['git', 'ls-files', '*.md'], capture_output=True, text=True).stdout.split()
    own = sorted(f for f in all_md if f.startswith(SELF_EVIDENCE))
    tracked = [f for f in all_md if f not in own]
    dom = domains(tracked)
    files, path_of, heads = S.load_index()

    sign_heads = {f: [h for h in heads[f] if HEAD_SIGN.match(h)] for f in files}
    resolver = {}
    for f in files:
        for h in heads[f]:
            m = HEAD_SIGN.match(h)
            body = h[m.end():] if m else h
            n = re.match(r'([0-9]+(?:\.[0-9]+)*)', body)
            if n:
                resolver.setdefault(n.group(1), set()).add(f)

    st = collections.Counter()
    parsed_rows, gap_rows = [], []
    for f in sorted(tracked):
        d = 'outside' if os.path.dirname(f) != 'docs' else ('self' if f == S.SELF else 'gate')
        text = open(f, encoding='utf-8').read()
        lines = text.split('\n')
        offs = [0]
        for ln in lines:
            offs.append(offs[-1] + len(ln) + 1)
        spans = sorted((m.start(), m.end(), m.group(1), m.group(2)) for m in S.SECREF.finditer(text))
        starts = [s[0] for s in spans]
        for m in SEC_MARK.finditer(text):
            off, sec = m.start(), m.group(1)
            ln = bisect.bisect_right(offs, off)
            line = lines[ln - 1]
            col = off - offs[ln - 1]          # 行内列号（切 `line` 用行内坐标，别拿全局 offset 切）
            st[f'markers.{d}'] += 1
            i = bisect.bisect_right(starts, off) - 1
            inside = i >= 0 and spans[i][1] > off
            if inside:
                doc = spans[i][2]
                st[f'parsed.{d}'] += 1
                hs = heads[path_of[doc]] if doc in path_of else []
                ok_new = any(head_new(h, sec) for h in hs)
                ok_old = any(head_old(h, sec) for h in hs)
                st[f'parsed_{"ok" if ok_new else "bad"}.{d}'] += 1
                if ok_new and not ok_old:
                    st[f'parsed_rescued.{d}'] += 1
                if not ok_new:
                    parsed_rows.append(('BAD', d, f, ln, sec, doc))
                if doc in path_of and sign_heads[path_of[doc]]:
                    st[f'parsed_target_sign.{d}'] += 1
                    parsed_rows.append(('TARGET-SIGN', d, f, ln, sec, doc))
                continue
            docrefs = [x for x in S.DOCREF.finditer(line) if x.end() <= col]
            gap = line[docrefs[-1].end():col] if docrefs else ''
            if docrefs and '§' not in gap and GAP.match(gap):
                st[f'gap_narrow.{d}'] += 1
                doc = docrefs[-1].group(1)
                hs = heads[path_of[doc]] if doc in path_of else []
                ok = any(head_new(h, sec) for h in hs)
                st[f'gap_narrow_{"ok" if ok else "bad"}.{d}'] += 1
                gap_rows.append((d, f, ln, sec, doc, gap, ok))
            else:
                st[f'other.{d}'] += 1
                n = len(resolver.get(sec, ()))
                st[f'other_amb_{"0" if n == 0 else ("1" if n == 1 else "many")}.{d}'] += 1

    print(f'# repo root: {ROOT}')
    print(f'# tracked *.md = {len(tracked)}  ->  gate(docs/*.md 去 SELF)={len(dom["gate"])} '
          f'self(docs/17)={len(dom["self"])} outside={len(dom["outside"])}')
    sign_docs = [os.path.basename(f) for f, hs in sign_heads.items() if hs]
    print(f'# 带前导 § 的标题 = {sum(len(v) for v in sign_heads.values())} 条，'
          f'分布在 {len(sign_docs)} 个文档 = {sign_docs}')
    print(f'# 量尺 vs 模块口径一致性: {check_module_agrees(heads)}')
    print(f'# 本目录自己的证据 .md（**不计入语料**，n={len(own)}）= {own}')
    print()
    print('## A. §-号标记处置（按域）')
    print(f'{"域":<8}{"markers":>8}{"parsed":>8}{"gap_narrow":>11}{"other":>7}')
    for d in ('gate', 'self', 'outside'):
        print(f'{d:<8}{st[f"markers.{d}"]:>8}{st[f"parsed.{d}"]:>8}'
              f'{st[f"gap_narrow.{d}"]:>11}{st[f"other.{d}"]:>7}')
    print()
    print('## B. parsed 那栏：标题口径 old vs new（这一栏的目标由 SECREF 唯一确定，无推断）')
    print(f'{"域":<8}{"可解析(new)":>12}{"其中改前误判":>14}{"仍 out-of-range":>16}'
          f'{"目标文档有带§标题":>17}')
    for d in ('gate', 'self', 'outside'):
        print(f'{d:<8}{st[f"parsed_ok.{d}"]:>12}{st[f"parsed_rescued.{d}"]:>14}'
              f'{st[f"parsed_bad.{d}"]:>16}{st[f"parsed_target_sign.{d}"]:>17}')
    print('  逐条（BAD = 改后仍 out-of-range；TARGET-SIGN = 目标文档里有带前导 § 的标题）:')
    for tag, d, f, ln, sec, doc in parsed_rows:
        print(f'    [{tag}][{d}] {f}:{ln}  §{sec} -> docs/{doc}')
    print()
    print('## C. skipped 那栏（本 mission **不改匹配面** ⇒ 这些是待裁决量，不是 M114 的暴露面）')
    print(f'{"域":<8}{"gap_narrow":>11}{"其中悬空":>10}{"other":>7}{"other该号0文档":>14}'
          f'{"other该号1文档":>14}{"other该号>=2文档":>15}')
    for d in ('gate', 'self', 'outside'):
        print(f'{d:<8}{st[f"gap_narrow.{d}"]:>11}{st[f"gap_narrow_bad.{d}"]:>10}{st[f"other.{d}"]:>7}'
              f'{st[f"other_amb_0.{d}"]:>14}{st[f"other_amb_1.{d}"]:>14}{st[f"other_amb_many.{d}"]:>15}')
    print(f'  gap_narrow 逐条（第 ⑤ 栏那种「path + 短 ASCII 夹层 + §」；M78 的历史读数是 **2 行** '
          f'= docs/09:4 与 docs/11:4，本脚本应复现；见下）:')
    for d, f, ln, sec, doc, gap, ok in gap_rows:
        print(f'    [{"可解析" if ok else "悬空"}][{d}] {f}:{ln}  §{sec} -> docs/{doc}  夹层={gap!r}')
    print(f'  行口径（去重后的行数）: gate={len({(f, ln) for d, f, ln, s, c, g, o in gap_rows if d == "gate"})} '
          f'repo={len({(f, ln) for _, f, ln, s, c, g, o in gap_rows})}')
    print()
    print('## D. 标题形态普查（解引用口径 `^<num>` 的隐含前提 —— 找"同一个问题"的其它形态）')
    form = collections.Counter()
    examples = collections.defaultdict(list)
    for f in files:
        for h in heads[f]:
            if HEAD_SIGN.match(h):
                k = '§<num>…（docs/20 形态）'
            elif re.match(r'[0-9]', h):
                k = '<num>…（其余 docs 形态）'
            else:
                k = '其它（不以数字开头）'
                if len(examples[k]) < 12:
                    examples[k].append(f'{os.path.basename(f)}: {h[:28]}')
            form[k] += 1
    for k, v in form.most_common():
        print(f'  {v:>4}  {k}')
        for e in examples[k]:
            print(f'         例：{e}')
    sign_with_refs = [f for f in sign_heads if sign_heads[f]]
    print(f'  其中「§<num>…」形态只在 {len(sign_with_refs)} 个文档里出现，'
          f'而这些标题**今天收到多少条 parsed 引用** = '
          f'{st["parsed_target_sign.gate"] + st["parsed_target_sign.self"] + st["parsed_target_sign.outside"]}')


if __name__ == '__main__':
    sys.exit(main())
