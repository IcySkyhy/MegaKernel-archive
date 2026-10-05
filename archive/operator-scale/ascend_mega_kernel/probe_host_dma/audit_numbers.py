#!/usr/bin/env python3
"""audit_numbers.py —— README 的**逐数字冒烟测试**（启发式；**不是可复现性证明**）

定位（请连同下面的"能力边界"一起读）：
  它把「README 里出现了一个数字」变成一条可跑判据，能自动抓出**最粗的一类**问题——
  "某个数在整仓任何 log 里都找不到，也没说明它是什么"。它**不能**替代人工核查，
  更**不能**证明"写进仓库的计数都可复现"。`LABELED` 桶是**作者自证**（作者声称该数是推算/常量/外部读数），
  脚本无法验证这个声称是否为真。

判据（对 README 里每个 ≥4 位有效数字的 token）：
  1. 结构引用（`文件:行号`、`§n`、`n=5`、`dav-3510`、年份…）                       -> FILE-REF
  2. 该 token（或其四舍五入形式）能在 `evidence/` 的文本里找到                     -> IN-EVIDENCE
  3. **紧贴**该 token 处有 `[外部]` / `[git史]` 逃逸标记（见下"紧贴"定义）          -> ESCAPE-TAGGED
  4. 同行出现**语义标注词**（推算/示意/常量/错误码/版本/几何/定义/边界…）          -> LABELED
  5. 以上都不是                                                                   -> FLAG

"紧贴"的定义（这是 R3 复审指出的"整行出现即豁免"洞的修法）：
  标记必须**直接挨着这个数**。跳过时不看的只有 `` ` `` `*` `_` `~` 与空白这 5 类字符
  （即 `DECOR_RE`；**比早期 docstring 写的更严**——`，, : ： ）)` 并不豁免，R4 复审实测
  `[外部] 12345.678`（标记在前）与 `12345.678，[外部]`（只隔逗号）都是 FLAG=1）。
  或者该数在同一个 `[...]` 组内而该组的 `]` 紧跟着标记。
  **不再**是"同行出现即豁免"：`| 注册耗时 12345.678 ms [外部] |` 这种写法**不再**被豁免（中间隔着 `ms`）。

**能力边界（已知绕过口，R3 复审实测过）**：
  * IN-EVIDENCE 只表示"这个字符串出现在 evidence/ 下某个文件里"，**不等于可复现**：
    手工把数字塞进任意命名普通的文件即可命中（复审用 `evidence/zz_fake.txt` 验证过）。
    本脚本会额外列出 evidence/ 下**未被 git 跟踪**的文件作为提示，但**不**据此判 FAIL。
  * LABELED 是作者自证：**行内出现任一保留语义词（推算/示意/常量/错误码/几何/定义/边界…）
    就会把该行所有数归入"作者自证"桶** —— 作者写一个词即可豁免一整行，成本极低。
  * 逃逸标记同理：作者可以把 `[外部]` 紧贴任意数写。
  * （R4 已修）"数字后紧贴字母数字即被当结构引用跳过"的口子（`12345.678ms`）：
    旧实现只看"后随字母数字"就整段跳过，现已删除该条；粘单位不再能豁免。
  故：**rc=0 只说明"没有出现最粗的形态"，不构成任何保证。**

用法（本目录下）：
  python3 audit_numbers.py [README路径] [evidence目录]
  rc=0 无 FLAG；rc=1 有 FLAG（列出明细）；rc=2 用法/路径错
"""

import os
import re
import subprocess
import sys

# 只保留"语义"标注：明确声明该数**不是**本批实测读数（是推算/常量/契约/几何定义等）。
# 注意**不要**把统计/近似词（中位、区间、区间宽度、比值、之比、≈、约、量级…）放进来——
# 它们恰恰标记的是「本该能复现」的数（R3 复审实测：假数只要同行出现"中位/区间"就被放行，
# 而 r1/r2 出事的形态正是"假中位数"）。
LABEL_MARKERS = [
    "推算", "示意", "环境", "常量", "错误码", "版本", "几何", "定义", "边界", "规范",
    "仅作参考", "量级参考",
]
# 注意：`[外部]`/`[git史]` **不在** LABEL_MARKERS 里 —— 它们只走 is_escape_adjacent()，即
# **必须紧贴该数**。R3 复审的绕过②正是"整行出现即豁免"；把它们放回 LABEL_MARKERS 会重新打开那个洞。

# 逃逸标记：必须**紧贴**数字（见 docstring）
ESCAPES = ["[外部]", "[git史]"]

# markdown 装饰/可忽略字符（判定"紧贴"时跳过）。
# 注意**不要**把 `]` 放进来：`[外部]`/`[git史]` 自带 `]`，跳过它会把自己的标记拆掉。
DECOR_RE = re.compile(r"[`*_~ \t]")
DECOR_STR = "`*_~ \t]"


def is_structural(tok, line, start, end):
    if tok.startswith("0x"):
        return True
    if re.fullmatch(r"(19|20)\d\d", tok):        # 年份
        return True
    if re.fullmatch(r"\d{1,3}", tok):            # 序号/计数/版本
        return True
    # `文件:行号` 的简写形式 `:2426`（R3 复审指出：旧的子串规则漏掉了这种写法）
    if start > 0 and line[start - 1] == ":":
        return True
    # 比值 `3.6x`：定义上是由别的数算出来的，不是独立读数
    rest = line[end:end + 2]
    if rest.startswith("×") or rest.startswith("x"):
        return True
    # 是更大标识符的**尾部**（dav-3510 的 3510；`-`/`_`/字母数字紧贴其前）
    # 注意：**只看前面**。R4 复审实测出"数字后紧贴字母数字即被整段跳过"的口子
    # （`12345.678ms` / `12345.678GB/s` -> FILE-REF 桶 -> FLAG=0），已把"后随字母数字"
    # 这条删除；粘单位不再能豁免（`950PR` 这类 3 位数字本来就会被 <4 位规则跳过）。
    if start > 0 and (line[start - 1].isalnum() or line[start - 1] in "-_"):
        return True
    # `文件:行号`、`§n`
    if re.search(r"[A-Za-z_./-]+:" + re.escape(tok), line):
        return True
    if re.search(r"§" + re.escape(tok), line):
        return True
    return False


def is_escape_adjacent(line, start, end):
    """逃逸标记是否**紧贴**该 token（见 docstring 的"紧贴"定义）。"""
    before = DECOR_RE.sub("", line[max(0, start - 30):start])
    if any(before.endswith(mk) for mk in ESCAPES):
        return True
    after = DECOR_RE.sub("", line[end:end + 30])
    if any(after.startswith(mk) for mk in ESCAPES):
        return True
    # 该 token 在同一个 [...] 组内，且该组的 ']' 紧跟标记：`[6.780, 72.969] [git史]`
    lb = line.rfind("[", 0, start)
    if lb != -1:
        rb = line.find("]", end)
        if rb != -1:
            grp_after = line[rb + 1:rb + 30].lstrip(DECOR_STR)
            if any(grp_after.startswith(mk) for mk in ESCAPES):
                return True
    return False


def _skip(fn):
    # 守卫**自己的输出**不能被当成证据，否则自指：它注入的负向对照数字会被自己的输出"找回来"。
    return "audit" in fn.lower()


def load_evidence(evdir):
    text_chunks, nums, files = [], set(), []
    for root, _dirs, fns in os.walk(evdir):
        for fn in fns:
            if _skip(fn):
                continue
            p = os.path.join(root, fn)
            files.append(p)
            try:
                txt = open(p, "r", errors="replace").read()
            except OSError:
                continue
            text_chunks.append(txt)
            for m in re.finditer(r"\d+(?:\.\d+)?", txt):
                nums.add(m.group(0))
    return "\n".join(text_chunks), nums, files


def rounded_match(tok, ev_nums):
    dec = len(tok.split(".")[1]) if "." in tok else 0
    for v in ev_nums:
        try:
            fv = float(v)
        except ValueError:
            continue
        if f"{fv:.{dec}f}" == tok:
            return True
    return False


def untracked_evidence(evdir):
    """列出 evidence/ 下未被 git 跟踪的文件（提示：它们还不属于交付物，因而不算"可复现证据"）。"""
    try:
        out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", evdir],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return None
        return [l for l in out.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    readme = sys.argv[1] if len(sys.argv) > 1 else "README.md"
    evdir = sys.argv[2] if len(sys.argv) > 2 else "evidence"
    if not os.path.isfile(readme):
        print(f"FAIL: {readme} 不存在")
        return 2
    if not os.path.isdir(evdir):
        print(f"FAIL: {evdir} 不存在")
        return 2

    evidence, ev_nums, ev_files = load_evidence(evdir)
    lines = open(readme, encoding="utf-8").read().split("\n")

    n_total = n_in = n_rnd = n_esc = n_lab = n_str = 0
    flags = []
    for ln, line in enumerate(lines, 1):
        for m in re.finditer(r"\d+(?:\.\d+)?", line):
            tok = m.group(0)
            if len(tok.replace(".", "")) < 4:
                continue
            n_total += 1
            if is_structural(tok, line, m.start(), m.end()):
                n_str += 1
                continue
            if tok in evidence:
                n_in += 1
                continue
            if rounded_match(tok, ev_nums):
                n_rnd += 1
                continue
            if is_escape_adjacent(line, m.start(), m.end()):
                n_esc += 1
                continue
            if any(mk in line for mk in LABEL_MARKERS):
                n_lab += 1
                continue
            flags.append((ln, tok, line.strip()[:150]))

    print("== README 逐数字冒烟测试（audit_numbers.py）==")
    print("   注意：这是**启发式**冒烟测试，rc=0 **不构成**可复现性保证（见脚本 docstring 的能力边界）。")
    print(f"readme   : {readme}")
    print(f"evidence : {evdir}（{len(ev_files)} 个文件，排除含 'audit' 的）")
    print(f"扫描 token（≥4 位有效数字）: {n_total}")
    print(f"  IN-EVIDENCE   : {n_in}")
    print(f"  IN-EVIDENCE(四舍五入): {n_rnd}")
    print(f"  ESCAPE-TAGGED : {n_esc}   （`[外部]`/`[git史]` 紧贴该数）")
    print(f"  LABELED       : {n_lab}   （**作者自证**，脚本无法验证其真假）")
    print(f"  FILE-REF/结构  : {n_str}")
    print(f"  FLAG          : {len(flags)}")

    ut = untracked_evidence(evdir)
    if ut:
        print()
        print(f"== 提示：evidence/ 下有 {len(ut)} 个文件未被 git 跟踪（尚不属于交付物）==")
        print("   （'IN-EVIDENCE' 只表示字符串在某个文件里出现过，**不等于可复现**；")
        print("     若这些是本次新增的证据文件，请先 `git add` 后再重跑本脚本。）")
        for p in ut[:20]:
            print(f"     {p}")
        if len(ut) > 20:
            print(f"     … 其余 {len(ut) - 20} 个")

    if flags:
        print()
        print("== FLAG 明细（既不在 evidence/ 里，也没有紧贴逃逸标记或语义标注）==")
        for ln, tok, text in flags:
            print(f"  {readme}:{ln}  token={tok}")
            print(f"      {text}")
    print()
    print("结论: " + ("PASS（无 FLAG）" if not flags else f"NEEDS-FIX（{len(flags)} 个 FLAG）"))
    return 0 if not flags else 1


if __name__ == "__main__":
    sys.exit(main())
