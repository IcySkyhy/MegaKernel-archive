# SPDX-License-Identifier: Apache-2.0
"""M22 router 的结构性代价模型（任务 2「每级代价」+ 任务 3「设计输入类估算」）。

全部是**指令/字节计数**（编译期可推导），不含调优；实测时间只作报告项，且**从归档日志现场解析**
（`evidence/run_mode{2,4}.log`，共用机器活值——不写死字面量），日志缺失时明确报缺。
用法：/usr/local/python3.12.13/bin/python3.12 tools/cost_model.py
"""

from __future__ import annotations

import os
import re

HIDDEN = 2560
E = 512
KTOPK = 10
RB = 8
EGRP = 8
KCHUNK = 64
NCHUNK = HIDDEN // KCHUNK
PAIR_BYTES = 8          # 1 对 = {fp32 value, u32 index}（probe_sync_quirks §2 实证）

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(os.path.dirname(HERE), "evidence")


def measured_m4097_ms():
    """m=4097 档的实测时间区间，**从归档日志现场解析**（不写字面量）。

    这是在共用机器上测出的**活值**：同一档跨次运行会漂（本机 10+ worker 共用）。所以本模型
    只做"结构性判断"，实测时间一律从 `evidence/run_mode{2,4}.log` 现取；归档缺失时明确说
    缺，而不是回落到某个记下来的数（M51 修订：旧版这两行曾硬编码 107.5ms / 12.5 GB/s，
    在日志更新成 106.3ms 之后就成了过期值）。
    """
    times = []
    for name in ("run_mode2.log", "run_mode4.log"):
        try:
            with open(os.path.join(EVID, name), encoding="utf-8", errors="ignore") as fh:
                m = re.search(r"real_m4097 .*?t=([\d.]+)ms", fh.read())
        except OSError:
            continue
        if m:
            times.append(float(m.group(1)))
    return (min(times), max(times)) if times else None


def floor1(v: float) -> float:
    """1 位小数向下取整——README §4.3/§9.1 的 "8.3~12.6 GB/s" 就是这个口径。"""
    return int(v * 10) / 10


def merge_tree_cost(dummy: int = 0) -> None:
    """归并树每级代价：Sort32 得 16 块 × 32 对，之后每级"取输入前 32 对、输出 Σlen 对"。

    2 路树（m7 现行）= level1 8 次、level2 4 次、level3 2 次、level4 1 次；
    4 路树 = level1 4 次、level2 1 次。
    正确性不变量：top-10(并集) ⊆ top-32(各输入) ⇒ 逐级归纳，top-10 严格精确。
    """
    print("=" * 78)
    print("归并树每级代价（Sort32 输出 = 16 块 × 32 对；elementLengths 单位 = 8B 对）")
    print("=" * 78)
    print(f"{'树':<14}{'级':<6}{'MrgSort 次数':>12}{'每级读(对)':>12}{'每级写(对)':>12}"
          f"{'累计读':>9}{'累计写':>9}")
    for name, levels in (("2 路 (m7)", [(8, 2, 32, 64), (4, 2, 32, 64), (2, 2, 32, 64), (1, 2, 32, 64)]),
                         ("4 路", [(4, 4, 32, 128), (1, 4, 32, 128)])):
        cum_r = cum_w = 0
        for i, (calls, way, srcLen, outLen) in enumerate(levels, 1):
            rd = calls * way * srcLen      # 每路只取输入的前 srcLen 对
            wr = calls * outLen            # 明文输出 Σlen = way*srcLen 对
            cum_r += rd
            cum_w += wr
            print(f"{name:<14}{i:<6}{calls:>12}{rd:>12}{wr:>12}{cum_r:>9}{cum_w:>9}")
        print(f"{'':<14}{'合计':<6}{sum(l[0] for l in levels):>12}"
              f"{cum_r:>12}{cum_w:>12}{cum_r:>9}{cum_w:>9}"
              f"   UB 流量 {(cum_r + cum_w) * PAIR_BYTES} B")
    print("\nSort32：1 次调用 / 16 repeat（512 对），输出 16 块 × 32 对 = 512 对 = 4KB。")
    print("Extract：1 次，从 merge 输出取前 64 对（只用前 10 对做 renorm/topk）。")
    print("4 路相对 2 路：MrgSort 调用 15 → 5（-67%）、UB 归并对流量 1920 → 1280 对（-33%）；")
    print("实测两棵树在本 mission 的全部 10 档上**逐字节等价**（evidence/merge_mode_equiv.txt）。")


def gemv_cost() -> None:
    """GEMV 代价模型（向量指令计数 + 权重字节流量）。

    内层（1 个专家组 8 专家 × 1 个 K 分块 64）：x UNPACK 载入 1 + Cast 1
      + 8 × w LoadAlign + 8 × MulAddDst = 18 条向量指令，承载 8×64 = 512 MAC。
    """
    print()
    print("=" * 78)
    print("GEMV 代价模型（向量指令计数；RB=%d, EGRP=%d, KCHUNK=%d）" % (RB, EGRP, KCHUNK))
    print("=" * 78)
    instr_per_step = 2 + 2 * EGRP               # x UNPACK + Cast + EGRP×LoadAlign + EGRP×MulAddDst
    mac_per_step = EGRP * KCHUNK
    instr_per_row = (E // EGRP) * NCHUNK * instr_per_step
    print(f"内层一步：{instr_per_step} 条向量指令承载 {mac_per_step} MAC"
          f"（{instr_per_step / mac_per_step:.4f} 条/MAC）")
    print(f"每行（512 专家 × K=2560）：{E * HIDDEN} MAC = {(E // EGRP) * NCHUNK} 步"
          f" = {instr_per_row} 条向量指令")
    print()
    print(f"{'m':>6}{'MAC':>14}{'向量指令':>14}{'权重字节流量':>16}{'估算 @1 指令/cyc,1.4GHz':>26}")
    for m in (1, 16, 64, 4097):
        mac = m * E * HIDDEN
        instr = m * instr_per_row
        nblk = -(-m // RB)
        wtraf = nblk * E * HIDDEN * 2            # 每个 row-block 重读全部 512 行权重（bf16）
        print(f"{m:>6}{mac:>14}{instr:>14}{wtraf:>16}{instr / 1.4e9 * 1e3:>22.1f} ms")
    print()
    print("结论（结构判断，非调优）：本实现的 router 是**向量指令发射受限**，不是带宽受限——")
    # 与上表 m=4097 行同源（不写字面量）
    nblk = -(-4097 // RB)
    wtraf = nblk * E * HIDDEN * 2
    instr = 4097 * instr_per_row
    est_ms = instr / 1.4e9 * 1e3                     # 假设 1 指令/cycle @1.4GHz
    meas = measured_m4097_ms()
    if meas:
        lo, hi = meas
        bw_lo = floor1(wtraf / 1e9 / (hi / 1e3))     # 两端时间 → 两端带宽
        bw_hi = floor1(wtraf / 1e9 / (lo / 1e3))
        print(f"  * m=4097 时权重流量 {wtraf:.4e} B / 归档实测 {lo:.1f}~{hi:.1f}ms"
              f" ≈ {bw_lo:.1f}~{bw_hi:.1f} GB/s（时间现场解析自 "
              f"evidence/run_mode{{2,4}}.log，共用机器活值），远低于 HBM 峰值；")
        print(f"  * 同一档估算 {instr / 1e8:.2f}e8 条向量指令 / 1.4GHz ≈ {est_ms:.0f}ms，"
              f"与归档实测 {lo:.1f}ms 同量级 ⇒ 指令发射是瓶颈。")
    else:
        print("  * 实测时间：evidence/run_mode{2,4}.log 缺失，无法现场解析 —— 本模型不给字面量；")
        print(f"  * 同一档估算 {instr / 1e8:.2f}e8 条向量指令 / 1.4GHz ≈ {est_ms:.0f}ms"
              f" ⇒ 指令发射是瓶颈。")
    print("  * 因此后续提速的杠杆是「专家维切到多个 AIV」（或换 cube），而不是加大 RB 省权重流量；")
    print("    RB 受 UB 限制（x 块 = RB×5120B），RB=8 时 x 块 40KB、整核 UB 178.6KB。")


if __name__ == "__main__":
    merge_tree_cost()
    gemv_cost()
