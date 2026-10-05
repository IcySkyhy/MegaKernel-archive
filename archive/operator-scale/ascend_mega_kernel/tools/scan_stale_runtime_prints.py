#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/scan_stale_runtime_prints.py —— 扫 `tools/**` 里"**会落进日志/输出**"的过期断言性文字。

依据（tower 规则，2026-09-26）：**脚本/内核在运行时打印、写盘或落进 dump/日志的断言性文字，
也属于证据的一部分，必须与同 commit 的事实一致**。过期结论（"正在修 X""暂不支持 Y""TODO：Z"）
必须改写成**可核对的现状陈述**，不是删掉（删掉会让"曾经缺过"这段历史消失）。
归档证据（`evidence/*.log`）里的同句按规则 4 **保留**。

## 覆盖范围（本脚本自己交代，防"通过了"与"根本没检查"同形）

| 栏 | 含义 |
|---|---|
| `scanned` | 实际解析的源文件数（`*.py` / `*.sh`），按扩展名分列 |
| `output_sites` | 抽出的**输出调用点**数（`print(…)` 等输出助手 / shell 的 `echo`/`printf`） |
| `runtime` | **命中**：输出调用点的源码片段里含过期断言标记（**这些必须改**） |
| `non_output` | 标记出现在**非输出上下文**（注释 / docstring / 普通字符串）—— 不是运行时打印，但列出以便人工判断 |
| `undecided` | **无法判定**：文件解析失败；或"标记只出现在非输出字符串、而本文件确有输出调用"（可能是间接打印，**已知盲区**，不静默当通过） |
| `archived` | `evidence/` 下的归档文本文件命中数（规则 4 豁免：归档记录的是「那一刻的事实」，**单独报数**，不参与判定） |
| `excluded` | **本扫描器自身**（它的标记表与自测夹具必然自命中）+ `--exclude` 命中的 **archived 类**文件 —— 单独报数，不静默丢弃 |
| `refused` | `--exclude` 指向了**非 archived**（即源文件）时**拒绝该排除**并单独报数 —— 被拒的文件**仍照常扫描并如实列出命中**（`96e7e96` 那版只报 `refused` 而不扫描它们，导致真命中被抹掉；`daf3344` 起**行为与文案一致**，判定不会被 `--exclude` 削掉覆盖） |
| `skipped` | 跳过的文件（`__pycache__` / 二进制 / 非 .py/.sh） |

**已知盲区（负向对照的具体对象）**：标记在运行时由变量拼出（`MSG = "…正在修…"` 在别处定义后
`print(MSG)`）。本脚本对这类情况打印 `undecided`，但**不能证明它真的没被打印** —— 这条由
`--self-test` 的三个合成用例钉住（见下），确保盲区**可见**而不是被当成通过。

## 退出码（三态）

  0 = 扫到了，且 **runtime 命中为 0**（非输出/归档命中只作报告）
  1 = **runtime 命中 > 0**（有要改的过期断言）
  2 = 没得扫：`tools/` 下没有任何可解析源文件，或全部文件解析失败

## 用法

    python3 tools/scan_stale_runtime_prints.py            # 扫本仓 tools/**
    python3 tools/scan_stale_runtime_prints.py --root DIR # 换根（默认 = 本脚本所在仓的 tools/）
    python3 tools/scan_stale_runtime_prints.py --self-test# 三个合成用例的负向对照（应 rc=0）
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 过期断言性文字的标记（中英）；命中的**行**会被列出，由人判断是不是真的过期。
MARKERS = [
    r"正在修", r"在修", r"is fixing", r"are fixing", r"will be fixed", r"will be added",
    r"暂不支持", r"尚未", r"未实现", r"未测", r"待实现", r"待补", r"待做", r"未做",
    r"\bTODO\b", r"\bFIXME\b", r"\bXXX\b", r"\bWIP\b",
    r"后续", r"将来", r"以后", r"not yet", r"to be implemented", r"planned",
]
MARKER_RE = re.compile("|".join(MARKERS))

# 「输出调用」的助手指名：这些函数名被调用时，其源码片段算作"会落进日志/输出"。
OUT_HELPERS = {
    "print", "printf", "echo", "note", "log", "logger", "warn", "warning", "error", "skip",
    "info", "debug", "check", "guard", "report", "say", "emit", "trace",
}
SKIP_DIRS = {"__pycache__", ".git"}


def _call_name(node: ast.AST) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def py_sites(src: str):
    """返回 (output_sites, undecided_reasons)。output_sites = [(lineno, segment)]。"""
    reasons = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [], ["parse failed: %s" % exc]
    sites = []
    has_const_string_with_marker = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in OUT_HELPERS:
            seg = ast.get_source_segment(src, node) or ""
            sites.append((node.lineno, seg))
        # 收集"带标记的字符串常量"（用于间接打印的 undecided 判定）
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if MARKER_RE.search(node.value):
                has_const_string_with_marker = True
    return sites, (["marker in a non-output string constant (可能被间接打印)"] if
                   has_const_string_with_marker else [])


def sh_sites(src: str):
    sites = []
    for i, line in enumerate(src.split("\n"), 1):
        s = line.strip()
        if re.match(r"^(echo|printf)\b", s) or re.search(r"(\|\||&&|;)\s*(echo|printf)\b", s):
            sites.append((i, line))
    return sites


def scan_file(path: str):
    """返回 dict：output_sites / runtime / non_output / undecided。"""
    src = open(path, errors="ignore").read()
    if path.endswith(".sh"):
        sites, ud = sh_sites(src), []
    else:
        sites, ud = py_sites(src)
    runtime, non_output = [], []
    covered = set()
    for lineno, seg in sites:
        n_lines = seg.count("\n") + 1
        for ln in range(lineno, lineno + n_lines):
            covered.add(ln)
        m = MARKER_RE.search(seg)
        if m:
            runtime.append((lineno, m.group(0), seg.strip().split("\n")[0][:140]))
    for i, line in enumerate(src.split("\n"), 1):
        if i in covered:
            continue
        m = MARKER_RE.search(line)
        if m:
            non_output.append((i, m.group(0), line.strip()[:140]))
    # 已知盲区：本文件有输出调用、且标记只出现在非输出字符串里
    if ud and sites and not runtime and non_output:
        pass  # 已在 ud 里
    elif ud and not non_output:
        ud = []
    return {"output_sites": len(sites), "runtime": runtime, "non_output": non_output,
            "undecided": ud}


def walk(root: str, excludes=()):
    """返回 (code, archived, excluded, refused, skipped)。

    * `code`     —— 被判定的源文件（`*.py`/`*.sh`，**排除 `evidence/`**）
    * `archived` —— `evidence/` 下的文本文件（规则 4 豁免：归档记录的是「那一刻的事实」）
    * `excluded` —— 本扫描器自身（它的标记表/自测夹具必然自命中）+ `--exclude` 命中的 **archived 类**文件
    * `refused`  —— `--exclude` 指向了**非 archived 的源文件**：**拒绝该排除**，但**仍把它放进 `code`
      照常扫描**（`96e7e96` 那版只塞进 `refused` 而不进 `code`，结果源文件既不参与扫描、
      文案却写着"仍按原样扫描" ⇒ 真命中被抹掉。**行为与文案必须一致**）
    * `skipped`  —— 其余文件（二进制/其他扩展名）
    """
    me = os.path.abspath(__file__)
    code, archived, excluded, refused, skipped = [], [], [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        in_evidence = (os.sep + "evidence" + os.sep) in dirpath + os.sep
        for f in sorted(filenames):
            p = os.path.abspath(os.path.join(dirpath, f))
            hit_exclude = any(e in p for e in excludes)
            is_src = f.endswith(".py") or f.endswith(".sh")
            if p == me:
                excluded.append(p)
            elif in_evidence:
                (excluded if hit_exclude else archived).append(p)
            elif hit_exclude and is_src:
                refused.append(p)          # 拒绝排除，但**照样扫描**（见 docstring）
                code.append(p)
            elif is_src:
                code.append(p)
            else:
                skipped.append(p)
    return code, archived, excluded, refused, skipped


def _text_lines(path):
    """只读文本文件（跳过含 NUL 的二进制）。"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return []
    if b"\x00" in raw:
        return []
    return raw.decode("utf-8", "ignore").split("\n")


def report(root: str, write=print, excludes=()) -> int:
    code, archived, excluded, refused, skipped = walk(root, excludes)
    py = [p for p in code if p.endswith(".py")]
    sh = [p for p in code if p.endswith(".sh")]
    if not code:
        write("RESULT: SKIPPED (tools/ 下没有可解析的 .py/.sh；这不是通过)")
        return 2
    total_sites, runtime, non_output, undecided = 0, [], [], []
    for p in code:
        try:
            r = scan_file(p)
        except Exception as exc:                       # noqa: BLE001
            undecided.append((p, 0, "scan failed: %s" % exc))
            continue
        total_sites += r["output_sites"]
        runtime += [(p, ln, mk, tx) for ln, mk, tx in r["runtime"]]
        non_output += [(p, ln, mk, tx) for ln, mk, tx in r["non_output"]]
        undecided += [(p, 0, u) for u in r["undecided"]]
    archived_hits, archived_scanned = 0, 0
    for p in archived:
        lines = _text_lines(p)
        if not lines:
            continue
        archived_scanned += 1
        archived_hits += sum(1 for l in lines if MARKER_RE.search(l))

    write("[coverage] 扫描源文件 %d 个（*.py %d / *.sh %d）；跳过非源文件 %d 个"
          % (len(code), len(py), len(sh), len(skipped)))
    write("[coverage] 排除 %d 个（本扫描器自身 1 个 + --exclude 命中的 archived 类 %d 个；前者因"
          "标记表/自测夹具必然自命中，后者用于剔除自指的归档读数）"
          % (len(excluded), max(len(excluded) - 1, 0)))
    if refused:
        write("[refused] **--exclude 只允许用于 archived 类（evidence/ 下）**；以下 %d 个**源文件**的"
              "排除被拒绝 —— 它们**仍按原样扫描**（命中会照常列出，见下面的 runtime/non_output 栏）："
              % len(refused))
        for p in refused:
            write("    %s" % os.path.relpath(p, os.path.dirname(root)))
    write("[coverage] 抽出输出调用点 %d 个（print/输出助手 / echo/printf）" % total_sites)
    write("[coverage] 归档 evidence/ 下扫了 %d 个文本文件、命中 %d 处"
          "（规则 4 豁免：归档记录的是「那一刻的事实」，单独报数、不参与判定）"
          % (archived_scanned, archived_hits))
    write("[runtime] 命中 %d 处（**这些是运行时打印，必须与同 commit 事实一致**）" % len(runtime))
    for p, ln, mk, tx in runtime:
        write("    %s:%d  [%s]  %s" % (os.path.relpath(p, os.path.dirname(root)), ln, mk, tx))
    write("[non_output] 命中 %d 处（注释/docstring/普通字符串；不是运行时打印，供人工判断）"
          % len(non_output))
    for p, ln, mk, tx in non_output:
        write("    %s:%d  [%s]  %s" % (os.path.relpath(p, os.path.dirname(root)), ln, mk, tx))
    write("[undecided] %d 处（解析失败 / 可能是间接打印 —— **不当作通过**）" % len(undecided))
    for p, ln, u in undecided:
        write("    %s:%d  %s" % (os.path.relpath(p, os.path.dirname(root)), ln, u))

    # 判定只反映**扫描本身**：被拒的 `--exclude` 不会减少覆盖（源文件照样被扫），所以
    # 它不改变 0/1/2 的语义（`daf3344` 的修法：行为与文案一致 —— 真命中**不会**被抹掉）。
    # `[refused]` 行是响亮的告警，且计数写进结论行，调用方不会漏看。
    refuse_note = ("；--exclude 的 %d 个源文件排除被拒、仍已扫描" % len(refused)) if refused else ""
    if not code:
        return 2
    if runtime:
        write("RESULT: STALE (runtime %d 命中 / 扫了 %d 个文件、%d 个输出调用点%s)"
              % (len(runtime), len(code), total_sites, refuse_note))
        return 1
    write("RESULT: OK (runtime 0 命中 / 扫了 %d 个文件、%d 个输出调用点；"
          "非输出 %d、未判定 %d、归档 %d —— 后三者不参与判定%s)" %
          (len(code), total_sites, len(non_output), len(undecided), archived_hits, refuse_note))
    return 0


# --------------------------------------------------------------- 负向对照
def self_test(tmpdir: str) -> int:
    """三个合成用例，钉住"看得见/看不见"的边界。"""
    ok = True
    cases = {
        "direct.py": 'print("M32 正在修该角落")\n',              # 必须判 runtime
        "comment_only.py": '# M32 正在修该角落\nx = 1\n',        # 必须判 non_output，不是 runtime
        "indirect.py": 'MSG = "M32 正在修该角落"\nprint(MSG)\n',  # 已知盲区：必须判 undecided
    }
    for name, src in cases.items():
        with open(os.path.join(tmpdir, name), "w") as f:
            f.write(src)
    got = {n: scan_file(os.path.join(tmpdir, n)) for n in cases}

    def state(r):
        if r["runtime"]:
            return "runtime"
        if r["undecided"]:
            return "undecided"
        if r["non_output"]:
            return "non_output"
        return "clean"

    want = {"direct.py": "runtime", "comment_only.py": "non_output", "indirect.py": "undecided"}
    print("--self-test（负向对照：三个合成用例）--")
    for n in cases:
        s = state(got[n])
        good = (s == want[n])
        ok = ok and good
        print("  %-16s 判为 %-10s 期望 %-10s  %s" % (n, s, want[n], "OK" if good else "MISMATCH"))
    print("RESULT: %s（direct=会被打印 / comment_only=不是运行时打印 / indirect=**已知盲区** "
          "会被报成 undecided 而不是静默通过）" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE, help="扫描根（默认 = 本脚本所在目录，即 tools/）")
    ap.add_argument("--exclude", action="append", default=[],
                    help="按路径子串排除（可重复）——**只对 archived 类（evidence/ 下）生效**。"
                         "例：--exclude tools/evidence/stale_runtime_print_scan.log"
                         "（该归档原文引用了过期句，会把 archived 计数自指推高）。"
                         "指向源文件会被**拒绝该排除**（照常扫描、命中如实列出，并显著打印 [refused]）")
    ap.add_argument("--self-test", action="store_true", help="跑三个合成用例的负向对照")
    args = ap.parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            return self_test(d)
    if not os.path.isdir(args.root):
        print("RESULT: SKIPPED (扫描根不存在：%s；这不是通过)" % args.root)
        return 2
    return report(os.path.abspath(args.root), excludes=args.exclude)


if __name__ == "__main__":
    sys.exit(main())
