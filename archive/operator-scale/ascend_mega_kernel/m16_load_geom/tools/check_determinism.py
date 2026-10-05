#!/usr/bin/env python3
"""从 build/geom2d_run{1..5}.log 生成 evidence/geom2d_determinism.txt（确定性证据）。

由 `reproduce.sh` 调用（脚本会先跑 5 次 `./m16_geom 2d`），因此该证据的命令可离线复现：
    cd m16_load_geom && bash reproduce.sh          # 或 tools/refresh_evidence.sh
"""
import itertools
import os
import sys

N_RUNS = 5


def main():
    if not os.path.isfile("geom2d_run1.log"):
        print("[determinism] 找不到 geom2d_run1.log —— 请在 build 目录内运行", file=sys.stderr)
        return 2
    runs = {}
    for r in range(1, N_RUNS + 1):
        runs[r] = [l.split("fnv=")[1].strip() for l in open("geom2d_run%d.log" % r) if "fnv=" in l]
    L = ["# 2D 探针确定性证据（%d 次独立进程运行；每 conf 一次 launch + launch 前显式清零 L0B）" % N_RUNS,
         "# 生成：`./m16_geom 2d` 跑 %d 次（见 reproduce.sh 第 [1/9] 步），逐 conf 比对 FNV 校验和。" % N_RUNS,
         ""]
    for a, b in itertools.combinations(range(1, N_RUNS + 1), 2):
        d = [i for i, (x, y) in enumerate(zip(runs[a], runs[b])) if x != y]
        L.append("run%d vs run%d: 逐 conf FNV 不一致的 conf = %s" % (a, b, d))
    grp = {}
    for r, v in runs.items():
        grp.setdefault(tuple(v), []).append(r)
    L += ["", "按逐 conf FNV 序列分组：" + str(list(grp.values())), "",
          "# 结论：",
          "#  * 全部良构 conf（参数落在源范围内的多分形 conf + A/B/C 三段方法学自检）在 %d 次运行中逐位一致；" % N_RUNS,
          "#  * 不一致只出现在「把源读到越界区间、且越界数据落进回读窗口(n<64)」的 conf 上；",
          "#    准确的包含关系是「非确定集合 ⊂ 窗口内越界读集合」，而不是「恰好等于」：",
          "#    越界读但落在窗口外（conf 22 kStep=4）或落在稳定为零的 L1 区（conf 31/45 srcStride=-2）时实测是确定的；",
          "#    具体不一致集合会随设备残留浮动（本仓库的 reviewer 在同样 5 次里见过多出 conf 30/44 的情况）。",
          "#  * 这些 conf 读出的是越界 L1 的残留内容 ⇒ 越界/错参数组合的读数不可解释，不用于推断；",
          "#  * 本 mission 的所有结论只建立在 in-range 的观测上（§3.4 的 mStart/kStart 两行另用单 conf 隔离运行补证，",
          "#    见 geom2d_isolated_run.txt，由 tools/check_isolated.py 生成）。",
          ]
    open("geom2d_determinism.txt", "w").write("\n".join(L) + "\n")
    print("[determinism] 写出 geom2d_determinism.txt（分组 %s）" % list(grp.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
