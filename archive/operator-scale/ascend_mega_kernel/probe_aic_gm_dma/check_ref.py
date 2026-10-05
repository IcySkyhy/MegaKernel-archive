#!/usr/bin/env python3
"""check_ref.py —— M141 探针的**独立于 C++ 实现**的宿主侧复核。

读 probe_aic_gm_dma 落盘的 GM 出口字节（out_<variant>_run<N>.bin，4096 个 fp32），
按每个变体的规格**另写一份**期望，逐位/逐元素重算。与 probe_aic_gm_dma.asc 里的 C++ 判据
（同一份 spec）互为独立实现：C++ 那份用 float 比较，这份用 struct 解包 + 分桶计数。

用法：
  check_ref.py <variant> <binfile>
输出（一行）：
  variant=<v> total=4096 mismatched=<n> sentinel=<n> landed=<n> pat_hits=<n> fixp_hits=<n> rc=<0|1>
  rc=0 = 与该变体规格一致；rc=1 = 有差异（负对照 neg_fixp_short 期望就是 rc=1）
"""
import struct
import sys

OUT_ELEMS = 64 * 64
SENTINEL = -12345.0
FIXP_VAL = 128.0
LANDING = {"fixp_l0c2gm", "neg_fixp_short", "aiv_ub2gm"}


def pat(i):
    return 1.0 + float(i)


def expected(variant, i):
    if variant in ("fixp_l0c2gm", "neg_fixp_short"):
        return FIXP_VAL
    if variant in LANDING:
        return pat(i)
    return SENTINEL  # 其余「预期不落盘」的档：全 sentinel


def main():
    if len(sys.argv) != 3:
        print("usage: check_ref.py <variant> <binfile>", file=sys.stderr)
        return 2
    variant, path = sys.argv[1], sys.argv[2]
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != OUT_ELEMS * 4:
        print(f"variant={variant} FILE_BAD bytes={len(raw)} expected={OUT_ELEMS * 4} rc=2")
        return 2
    vals = struct.unpack("<%df" % OUT_ELEMS, raw)
    mism = sentinel = pat_hits = fixp_hits = 0
    for i, got in enumerate(vals):
        if got != expected(variant, i):
            mism += 1
        if got == SENTINEL:
            sentinel += 1
        if got == pat(i):
            pat_hits += 1
        if got == FIXP_VAL:
            fixp_hits += 1
    landed = OUT_ELEMS - sentinel
    rc = 0 if mism == 0 else 1
    print(f"variant={variant} total={OUT_ELEMS} mismatched={mism} sentinel={sentinel} landed={landed} "
          f"pat_hits={pat_hits} fixp_hits={fixp_hits} rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
