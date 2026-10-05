#!/usr/bin/env python3
"""P2-2 判定实验：对"非确定集合"里的 conf 做**单 conf 隔离运行**，验证
「窗口内 in-range 槽位逐位一致、不确定性只出现在越界槽位」。

用法（在 build 目录内，且已构建好 m16_geom）：
  /usr/local/python3.12.13/bin/python3 ../tools/check_isolated.py

做法：对每个候选 conf N，各跑两次 `./m16_geom 2d N N`（每次进程 + 单 conf launch，
launch 前显式清零 L0B），从 dump 里取 conf N 的 16x64 网格，用 check_ref.py 的同一套模型
把槽位分成 in-range / 越界（junk）两类，分别比对两次运行的逐位一致性。

输出同时写 `geom2d_isolated_run.txt`（供归档为证据）。
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import check_ref as CR  # noqa: E402

M, N = CR.M, CR.N
CANDIDATES = [24, 27, 28, 29, 30, 36, 38, 41, 42, 43, 44]


def run_once(conf, tag):
    subprocess.run(["./m16_geom", "2d", str(conf), str(conf)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = np.fromfile("m16_geom_2d_raw.bin", dtype=np.float32).reshape(-1, M, N)
    return raw[conf].copy()


def main():
    tsv = "m16_geom_2d_raw.bin.tsv"
    label = {}
    for line in open(tsv):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            label[int(parts[0])] = (parts[1], parts[2])
    lines = ["# P2-2 判定实验：非确定 conf 的单 conf 隔离运行（各 2 次）",
             "# 判据：模型定义的 in-range 槽位必须两次逐位一致；越界槽位允许不同",
             ""]
    all_ok = True
    for c in CANDIDATES:
        tag, fields = label.get(c, ("?", ""))
        pred = CR.predict_grid(fields)
        inrange = [kn for kn, (pr, pc) in pred.items() if pr >= 0]
        junk = [kn for kn, (pr, pc) in pred.items() if pr < 0]
        a = run_once(c, "a")
        b = run_once(c, "b")
        if inrange:
            ia = np.array([a[k, n] for (k, n) in inrange])
            ib = np.array([b[k, n] for (k, n) in inrange])
            in_same = bool(np.array_equal(ia, ib))
        else:
            in_same = None
        if junk:
            ja = np.array([a[k, n] for (k, n) in junk])
            jb = np.array([b[k, n] for (k, n) in junk])
            junk_same = bool(np.array_equal(ja, jb))
        else:
            junk_same = None
        ok = (in_same is not False)
        all_ok = all_ok and ok
        lines.append(f"conf {c:2d} {tag[:46]:<46} in-range {len(inrange):4d} 槽 "
                     f"两次一致={in_same}  |  越界槽 {len(junk):4d} 两次一致={junk_same}")
    lines += ["",
              "结论：" + ("全部候选 conf 的 in-range 槽位两次逐位一致；"
                        "不一致只出现在越界（junk）槽位 ⇒ §3.4 里 mStart/kStartPosition 的结论"
                        "在 in-range 范围内成立。" if all_ok else "有 conf 的 in-range 槽位也不一致！"),
              "注意：本条只用于说明『越界槽位不可用』，不改变 §2.3 的规则——"
              "整 conf 的 FNV 在这些 conf 上本就非确定，不能拿整 conf 的读数去推断。"]
    out = "\n".join(lines) + "\n"
    print(out)
    open("geom2d_isolated_run.txt", "w").write(out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
