#!/usr/bin/env python3
"""verify_controls.py —— `verify_section9_claims.py` 的**变异正对照**（证明这个自检不是空洞的）

塔的纪律：「若判据做不到 FAIL，按拦项处理」+「参考与实现同源 ⇒ 判据无咬合力」。
本脚本在 **/tmp 的副本**里逐个注入变异，然后跑自检，核对它的 rc 是否与预期一致：

  N0 保持原样（负向对照）                        → 期望 rc=0（PASS）—— 证明对照不是"全都 FAIL"
  M1 README 把某格声明的完整行数改错              → 期望 rc≠0（V2：decl≠footer）
  M2 ★ 从 dump 块里**删掉一行真实 stdout**（页脚不动）→ 期望 rc≠0（V1：footer≠stdout）
  M3 改 dump 页脚的计数                          → 期望 rc≠0（V1）
  M4 删掉某格的 `（**完整 N 行**…）` 标注          → 期望 rc≠0（V6：该段变成无人认领）
  M5 删掉某格「列出」里的一个行号（K 不变）        → 期望 rc≠0（V5：列出计数不符）
  M6 把某段 exit 翻成非 0                        → 期望 rc≠0（V3：exit≠0）
  M6' 把一段的 stdout 换成错误文本、exit 保持 0    → 期望 rc≠0（V4：内容级错误特征，独立于 exit/页脚）

用法（本目录下）：python3 tools/verify_controls.py
退出码：0 = 全部对照行为符合预期；1 = 有对照不符合（说明自检仍有洞）。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable or 'python3'


def make_copy(dst):
    os.makedirs(os.path.join(dst, 'evidence'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'tools'), exist_ok=True)
    shutil.copy(os.path.join(ROOT, 'README.md'), dst)
    shutil.copy(os.path.join(ROOT, 'evidence', 'section9_greps.txt'),
                os.path.join(dst, 'evidence', 'section9_greps.txt'))
    shutil.copy(os.path.join(ROOT, 'tools', 'verify_section9_claims.py'),
                os.path.join(dst, 'tools', 'verify_section9_claims.py'))


def read(p):
    return open(p, encoding='utf-8').read()


def write(p, s):
    open(p, 'w', encoding='utf-8').write(s)


# ---- 变异实现（都只改副本） ----
def n0(d):  # 原样
    return True


def m1(d):
    p = os.path.join(d, 'README.md')
    s = read(p)
    s2 = re.sub(r'(#m15_state_access`（\*\*完整 )4( 行\*\*)', r'\g<1>3\g<2>', s, count=1)
    write(p, s2)
    return s2 != s


def m2(d):
    """删掉 dump 里一行真实 stdout（保留页脚）——这是复审要求必须 FAIL 的那条"""
    p = os.path.join(d, 'evidence', 'section9_greps.txt')
    lines = read(p).split('\n')
    out, in_blk, deleted = [], False, False
    for ln in lines:
        if ln.startswith('=== [m20_processaiv_dws]'):
            in_blk = True
        elif in_blk and re.match(r'^\[exit=', ln):
            in_blk = False
        elif in_blk and ln.startswith('1707:') and not deleted:
            deleted = True
            continue          # 删掉这一行真实输出
        out.append(ln)
    write(p, '\n'.join(out))
    return deleted


def m3(d):
    p = os.path.join(d, 'evidence', 'section9_greps.txt')
    s = read(p)
    s2 = s.replace('[exit=0] [完整输出行数=8]', '[exit=0] [完整输出行数=9]', 1)
    write(p, s2)
    return s2 != s


def m4(d):
    p = os.path.join(d, 'README.md')
    s = read(p)
    s2 = re.sub(r'`#m15_head_loop`（\*\*完整 2 行\*\*；列出 2 行：`:\d+`、`:\d+`）',
                '`#m15_head_loop`', s, count=1)
    write(p, s2)
    return s2 != s


def m5(d):
    """删掉「列出」里的一个行号（K 保持 4）⇒ 期望 V5 抓住 K 与实际列出数不符"""
    p = os.path.join(d, 'README.md')
    s = read(p)
    s2 = re.sub(r'(`#m15_state_access`（\*\*完整 4 行\*\*；列出 4 行：)`:1066`、', r'\1', s, count=1)
    write(p, s2)
    return s2 != s


def m6(d):
    p = os.path.join(d, 'evidence', 'section9_greps.txt')
    s = read(p)
    s2 = s.replace('[exit=0] [完整输出行数=8]', '[exit=2] [完整输出行数=8]', 1)
    write(p, s2)
    return s2 != s


def m6p(d):
    """把一段的 stdout 首行换成错误文本（页脚/exit 同步保持一致），期望 V4 抓住"""
    p = os.path.join(d, 'evidence', 'section9_greps.txt')
    lines = read(p).split('\n')
    out, in_blk = [], False
    for ln in lines:
        if ln.startswith('=== [m15_nl_def]'):
            in_blk = True
        elif in_blk and re.match(r'^\[exit=', ln):
            in_blk = False
        elif in_blk and ln.startswith('42:'):
            out.append('grep: m15_layer_loop/m15_loop_layout.h: No such file or directory')
            continue
        out.append(ln)
    write(p, '\n'.join(out))
    return True


CASES = [
    ('N0 原样（负向对照）', n0, 0),
    ('M1 README 声明行数改错', m1, 1),
    ('M2 ★删一行真实 stdout（页脚不动）', m2, 1),
    ('M3 改 dump 页脚计数', m3, 1),
    ('M4 删掉完整行数标注', m4, 1),
    ('M5 删掉列出里的一个行号', m5, 1),
    ('M6 exit 翻成非 0', m6, 1),
    ("M6' stdout 换错误文本（exit 仍 0）", m6p, 1),
]


def main():
    tmp = tempfile.mkdtemp(prefix='m64_controls_')
    print(f"# 变异正对照（全部在 /tmp 副本里做）：{tmp}")
    print(f"{'对照':38s} {'注入生效':>8s} {'自检 rc':>7s} {'期望':>4s}  结论")
    bad = 0
    for name, fn, expect_nonzero in CASES:
        d = os.path.join(tmp, re.sub(r'\W+', '_', name))
        make_copy(d)
        injected = fn(d)
        r = subprocess.run([PY, os.path.join(d, 'tools', 'verify_section9_claims.py'), '--root', d],
                           capture_output=True, text=True)
        got = 0 if r.returncode == 0 else 1
        ok = (got == expect_nonzero) and injected
        if not ok:
            bad += 1
        print(f"{name:38s} {str(injected):>8s} {r.returncode:>7d} {expect_nonzero:>4d}  "
              f"{'OK' if ok else '对照不符合预期 ✗'}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n不符合预期的对照数：{bad}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
