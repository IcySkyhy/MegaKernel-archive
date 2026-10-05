#!/usr/bin/env python3
"""M165 一键复现（零设备）：自检 + 上溢对照 + 官方对齐 + 归档重跑，逐项核对关键读数。

外部归档 / 官方 m21 dump 缺失的档报 **SKIP**（不计 FAIL）；判据不符 **rc=1**；全过 **rc=0**。
路径可用 `--m151/--native/--ref` 覆盖（默认是本次取证用的快照路径）。

用法：
  /workspace/venvs/baseline/bin/python3 m165_repro.py \
      --m151 <m151_prefill_prolog_wiring> --native /tmp/m115_verify \
      --ref /tmp/m165_ref/m165_layer0_m1 --ref /tmp/m165_ref/m165_layer0_m64 \
      --ref /tmp/m165_ref/m165_layer0_m4097
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CHECK = os.path.join(ROOT, "check_ref.py")
ALIGN = os.path.join(HERE, "m165_official_align.py")
PY = os.environ.get("M165_PY", "/usr/local/python3.12.13/bin/python3")

DEFAULT_M151 = ("/workspace/ascend_mega_kernel/.tower/worktrees/wt-151/"
                "m15_layer_loop/evidence/m151_prefill_prolog_wiring")


def run(args):
    return subprocess.run([PY] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)


def nsum(out, field):
    return sum(int(m) for m in re.findall(field + r"=(\d+)", out))


def section(n, title):
    print("\n=== %d. %s ===" % (n, title))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m151", default=DEFAULT_M151)
    ap.add_argument("--native", default="/tmp/m115_verify")
    ap.add_argument("--ref", action="append", default=[],
                    help="官方 m21 reference 目录（可多次；缺失则 SKIP）")
    args = ap.parse_args()

    rc = 0
    skips = []

    # ---- 1. 零设备自检（含合成上溢档：指数差 PASS / 旧式 FAIL）----
    section(1, "check_ref.py --selftest")
    p = run([CHECK, "--selftest"])
    print(p.stdout)
    if p.returncode != 0 or "各档符合期望" not in p.stdout:
        print("!! selftest 未达期望"); rc = 1

    # ---- 2. 修前/修后对照：指数差 0 NaN / 旧式 8,374,272 NaN ----
    section(2, "--stab-demo（M151 m4097 clean；旧式 vs 指数差）")
    d = os.path.join(args.m151, "dumps_m4097_clean")
    if not os.path.isdir(d):
        print("SKIP：%s 不存在" % d); skips.append("stab-demo")
    else:
        p = run([CHECK, "--stab-demo", "--dir", d])
        print(p.stdout)
        has_legacy = "旧式       float32 | o NaN = 8374272" in p.stdout
        has_stable = re.search(r"指数差\s+float32\s*\|\s*o NaN = 0", p.stdout) is not None
        if not (p.returncode == 0 and has_legacy and has_stable):
            print("!! stab-demo 关键读数不符（旧式 fp32 应 8374272、指数差 fp32 应 0）"); rc = 1

    # ---- 3. 与官方 m21 torch core_out 对齐 ----
    section(3, "m165_official_align.py（官方 core_out）")
    refs = [r for r in args.ref if os.path.isdir(r)]
    if not refs:
        print("SKIP：未给 --ref 或目录不存在"); skips.append("official-align")
    else:
        p = run([ALIGN] + refs)
        print(p.stdout)
        if p.returncode != 0:
            print("!! official-align 未过"); rc = 1

    # ---- 4. 归档重跑：修正后的真实结论 ----
    section(4, "归档重跑（NaN-aware 判据；stable vs legacy）")
    m151ok = os.path.isdir(os.path.join(args.m151, "dumps_m4097_clean"))
    if not m151ok:
        print("SKIP：M151 归档不存在"); skips.append("archive-m151")
    else:
        p = run([CHECK, "--dir", os.path.join(args.m151, "dumps_m4097_clean"), "--allow-stale"])
        print(p.stdout)
        refnan = nsum(p.stdout, "refNaN"); onesided = nsum(p.stdout, "仅一侧NaN")
        # 修正后的真实结论：参考侧**零** NaN（不再是 3,670,912），设备 NaN 全部单侧
        if not (p.returncode == 1 and refnan == 0 and onesided > 0):
            print("!! M151 clean（stable）期望 rc=1 / 参考NaN=0 / 仅一侧NaN>0，实得 rc=%d refNaN=%d oneSided=%d"
                  % (p.returncode, refnan, onesided)); rc = 1
        p = run([CHECK, "--dir", os.path.join(args.m151, "dumps_m4097_clean"), "--allow-stale", "--legacy-exp"])
        print(p.stdout)
        refnan_l = nsum(p.stdout, "refNaN"); both_l = nsum(p.stdout, "双侧NaN")
        if not (p.returncode == 1 and refnan_l > 0 and both_l > 0):
            print("!! M151 clean（legacy）期望 rc=1 / 参考NaN>0 / 双侧NaN>0，实得 rc=%d refNaN=%d bothNaN=%d"
                  % (p.returncode, refnan_l, both_l)); rc = 1
        p = run([CHECK, "--dir", os.path.join(args.m151, "dumps_m1_clean"), "--allow-stale"])
        print(p.stdout)
        if not (p.returncode == 0 and nsum(p.stdout, "refNaN") == 0):
            print("!! M151 m1 clean 期望 rc=0 / 参考NaN=0"); rc = 1

    if os.path.isdir(args.native):
        p = run([CHECK, "--dir", args.native, "--allow-stale"])
        print(p.stdout)
        if not (p.returncode == 0 and nsum(p.stdout, "refNaN") == 0
                and nsum(p.stdout, "非有限不匹配") == 0):
            print("!! native clean 期望 rc=0 / 无非有限不匹配"); rc = 1
    else:
        print("SKIP：native 归档 %s 不存在" % args.native); skips.append("archive-native")

    print("\n[M165-REPRO] rc=%d%s" % (rc, ("（SKIP: %s）" % ", ".join(skips)) if skips else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
