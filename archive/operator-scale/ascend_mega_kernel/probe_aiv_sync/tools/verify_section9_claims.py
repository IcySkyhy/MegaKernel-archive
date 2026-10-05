#!/usr/bin/env python3
"""verify_section9_claims.py —— §9「核验」列的**非空洞**自检（三个独立来源互证）

要防的东西（r3 复审 F1 的教训）：**不能拿同一份自产元数据互比**。
本脚本对每个 `#tag` 段做**三源互证 + 内容级检查**：

  源①  **实际 stdout 行数**：从 evidence/section9_greps.txt 的块里**真数**（`$ cmd` 之后、`[exit=…]` 之前的所有行）。
  源②  **dump 页脚** `[完整输出行数=M]`：由生成脚本自报。
  源③  **README §9 声明** `#tag（**完整 N 行**）`：文档正文里的声明。

  断言（任一不成立即 FAIL，非零退出）：
   V1  源① == 源②            # 抓"dump 里被增删了一行、页脚却不变"（M2）
   V2  源② == 源③            # 抓"声明与实测不符"（r2 缺陷 / M1）
   V3  exit == 0              # 抓"命令失败却当成证据"（F2 / M6）
   V4  stdout 不出现错误特征行（`No such file`/`command not found`/`Permission denied`）——
                              # 独立于页脚/exit 这两个自产元数据的**内容级**判据（抓 M6'：把 exit 2 改成 0）
   V5  README 该格声明「列出 K 行：`:a`、`:b`…」时：每个行号必须在该块 stdout 里出现、且个数等于 K（抓 M5）
   V6  每个 dump 段都必须被 README 引用（抓"多出来的段无人认领"）

用法：
  python3 tools/verify_section9_claims.py            # 默认 root = 本工程目录
  python3 tools/verify_section9_claims.py --root /tmp/x   # 变异对照用
退出码：0 = 全过；1 = 有 FAIL。
"""
import argparse
import os
import re
import sys

TAGS_RE = re.compile(r'^=== \[(\w+)\] tree=(\S+) commit=(\S+)')
FOOT_RE = re.compile(r'^\[exit=(-?\d+)\] \[完整输出行数=(\d+)\]\s*$')
DECL_RE = re.compile(r'#(\w+)`?（\*\*完整 (\d+) 行\*\*(?:；列出 (\d+) 行：([^）]*))?）')
ERR_RE = re.compile(r'No such file or directory|command not found|Permission denied|Is a directory')


def parse_dump(path):
    """返回 {tag: dict(stdout=[...], exit=int, footer=int, tree=..., commit=...)}"""
    blocks, cur = {}, None
    for ln in open(path, encoding='utf-8'):
        ln = ln.rstrip('\n')
        m = TAGS_RE.match(ln)
        if m:
            cur = {'tag': m.group(1), 'tree': m.group(2), 'commit': m.group(3),
                   'cmd': None, 'stdout': [], 'exit': None, 'footer': None, 'done': False}
            blocks[m.group(1)] = cur
            continue
        if cur is None or cur['done']:
            continue
        if cur['cmd'] is None and ln.startswith('$ '):
            cur['cmd'] = ln[2:]
            continue
        f = FOOT_RE.match(ln)
        if f and cur['cmd'] is not None:
            cur['exit'], cur['footer'], cur['done'] = int(f.group(1)), int(f.group(2)), True
            continue
        if cur['cmd'] is not None:
            cur['stdout'].append(ln)
    return blocks


def parse_readme(path):
    """返回 §9 里 (tag, 声明N or None, 该格里列出的行号集合) 的列表"""
    txt = open(path, encoding='utf-8').read()
    sec = txt[txt.index('## 9. 对'):txt.index('## 10. 覆盖范围')]
    out = []
    for m in DECL_RE.finditer(sec):
        tag = m.group(1)
        listed = [int(x) for x in re.findall(r':(\d+)', m.group(4) or '')]
        out.append((tag, int(m.group(2)), listed, int(m.group(3)) if m.group(3) else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    dump_p = os.path.join(a.root, 'evidence', 'section9_greps.txt')
    readme_p = os.path.join(a.root, 'README.md')

    blocks = parse_dump(dump_p)
    claims = parse_readme(readme_p)
    bad = 0
    print(f"{'tag':24s} {'stdout':>6s} {'footer':>6s} {'decl':>5s} {'exit':>4s}  "
          f"{'V1':>3s} {'V2':>3s} {'V3':>3s} {'V4':>3s} {'V5':>3s}  结论")
    tags_in_readme = set()
    for tag, decl, listed, kdecl in claims:
        if tag in tags_in_readme:
            continue
        tags_in_readme.add(tag)
        b = blocks.get(tag)
        if b is None:
            print(f"{tag:24s} {'-':>6s} {'-':>6s} {str(decl):>5s} {'-':>4s}   ❌ 段不存在 → FAIL")
            bad += 1
            continue
        n_stdout = len([l for l in b['stdout']])
        v1 = (n_stdout == b['footer'])
        v2 = (b['footer'] == decl)
        v3 = (b['exit'] == 0)
        v4 = not any(ERR_RE.search(l) for l in b['stdout'])
        block_nums = set()
        for l in b['stdout']:
            m = re.match(r'^(\d+)[:-]', l)
            if m:
                block_nums.add(int(m.group(1)))
        missing = sorted(set(listed) - block_nums)
        v5 = (not missing) and (kdecl is None or kdecl == len(listed))
        ok = v1 and v2 and v3 and v4 and v5
        if not ok:
            bad += 1
        why = []
        if not v1:
            why.append('footer≠stdout')
        if not v2:
            why.append('decl≠footer')
        if not v3:
            why.append(f"exit={b['exit']}")
        if not v4:
            why.append('错误文本')
        if not v5:
            if missing:
                why.append(f"声明行号不在块内{missing}")
            else:
                why.append(f"列出计数不符(K={kdecl}, 实际{len(listed)})")
        print(f"{tag:24s} {n_stdout:6d} {b['footer']:6d} {str(decl):>5s} {str(b['exit']):>4s}  "
              f"{'OK' if v1 else 'X':>3s} {'OK' if v2 else 'X':>3s} {'OK' if v3 else 'X':>3s} "
              f"{'OK' if v4 else 'X':>3s} {'OK' if v5 else 'X':>3s}  {'OK' if ok else 'FAIL: ' + ','.join(why)}")
    unref = sorted(set(blocks) - tags_in_readme)
    v6 = not unref
    if unref:
        bad += 1
    print()
    print(f"段数: {len(blocks)}；README 引用: {len(tags_in_readme)}；未引用段: {len(unref)} "
          f"-> {' '.join(unref) if unref else '(none)'}  [V6 {'OK' if v6 else 'FAIL'}]")
    print(f"FAIL 数: {bad}   (V1=行数一致 V2=声明一致 V3=exit0 V4=无错误文本 V5=列出行长在块内 V6=无未引用段)")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
