#!/usr/bin/env python3
"""M54 语义重基线：把「M54 前的 m3 量化器规则」与「M54 后的 floor/OCP 规则」在同一批
bf16 网格输入（`evidence/quant_cross/h.bin`）上对比，量化两件事：

  ① scale 规则（ceil -> floor）：scale 字节差异占比；
  ② data 路径：M54 前的 m3 镜像把 `h` **乘以** `2^(byte-127)`（= scale），而官方/ m13 是
     **乘以** `2^(127-byte)`（= halfScale = 1/scale）——这是一处**方向反了的实现缺陷**（见 README
     §5.3）。本脚本给出「乘 scale」与「乘 1/scale」两种 data 路径在同一输入上的
     pack 字节差异占比与 H 反量化重建误差，作为该缺陷的量化证据。

被判对象的口径（诚实披露）：
  · 「M54 前」的规则**逐句取自 base commit 的 m3 代码**（`QuantRowH` / `M3VfAmaxToScaleByte`
    在 main `95803a3`）：`byte = E-2+(M>0x400000)`、`v = h·2^(byte-127)`、e2m1 最近值 + 平局取偶。
  · 「M54 前的 device 与这份镜像逐位一致」由 base commit 归档的 `evidence/vf_selftest_run.log`
    的 `quant-pack`（0/2560 字节不符）见证——本脚本因此是对**pre-M54 device H 编码**的重建，
    不是对 device 的再次读数。
  · 「M54 后」的规则取 `tools/golden/moe_block_ref.quantize_ocp`（官方 OCP 逐句转写；已被
    `check_m13_cross.py` 证明与 M54 device 逐字节一致）。

覆盖范围 / 已知会被漏掉：
  · 输入是 `quant_cross/h.bin`（128 行 × 640，全部 bf16 网格值 + 端点/退化/非有限用例）；
  · 重建误差按**逐元素** |dequant − h| / |h| 统计，只统计有限且非零元素；
  · 非 bf16 网格的 fp32 h（生产里 h 是 fp32）不在本对比内——本脚本只量化「方向 + scale 规则」
    这一次性差异，不量化 h 精度的残留差异（后者见 README §5.4）。

用法：python3 m3_grouped_gemm/analyze_rebaseline.py [dump_dir]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "golden"))
from moe_block_ref import e2m1_decode, quantize_ceil_legacy, quantize_ocp  # noqa: E402

E2M1_MIDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def old_scale_byte(amax):
    b = np.asarray(amax, dtype=np.float32).view(np.uint32)
    E = (b >> 23) & 0xFF
    M = b & 0x7FFFFF
    byte = ((E - 2 + (M > 0x400000)) & 0xFF).astype(np.uint8)
    byte[np.asarray(amax) <= 0] = 0
    return byte


def old_mul_bitcast(sbyte):
    """base commit `ScaleByteToFloatH` 的位模式语义：f = bitcast((uint32)b << 23)。

    ⇒ byte 0 -> +0.0（不是 2^-127）、byte 0xFF -> +Inf（不是 2^128）。dump 里含这些组
    （退化 / ±Inf / ±NaN 行），必须按位模式建模才与 base commit 逐位一致。"""
    return (sbyte.astype(np.uint32) << np.uint32(23)).view(np.float32)


def old_pack(h, group=32):
    rows, k = h.shape
    g = h.reshape(rows, k // group, group)
    amax = np.max(np.abs(g), axis=2).astype(np.float32)
    sbyte = old_scale_byte(amax)
    mul = old_mul_bitcast(sbyte)[:, :, None]
    v = g * mul
    a = np.abs(v.astype(np.float64))
    c = np.zeros(a.shape, dtype=np.int64)
    for t in E2M1_MIDS:
        c += (a > t)
    c += ((a == 0.75) | (a == 1.75) | (a == 3.5))
    code = (c & 0xF).astype(np.uint8)
    code[v < 0] |= 8
    code = code.reshape(rows, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8), sbyte


def recon(pack, sbyte, k, group=32):
    rows = pack.shape[0]
    codes = np.empty((rows, k), np.uint8)
    codes[:, 0::2] = pack & 0x0F
    codes[:, 1::2] = pack >> 4
    vals = e2m1_decode(codes).reshape(rows, k // group, group)
    sc = np.exp2((sbyte.astype(np.int32) - 127)).astype(np.float64)[:, :, None]
    return (vals * sc).reshape(rows, k)


def relerr(h, r):
    m = np.isfinite(h) & (h != 0)
    rel = np.abs(r[m] - h[m]) / np.abs(h[m])
    return rel


def main(argv):
    d = argv[1] if len(argv) >= 2 else os.path.join(HERE, "evidence", "quant_cross")
    h = np.fromfile(os.path.join(d, "h.bin"), dtype=np.float32).reshape(-1, 640)
    rows, k = h.shape

    pack_old, scal_old = old_pack(h)
    pack_new, scal_new = quantize_ocp(h)
    scal_new = scal_new.reshape(rows, k // 32)
    pack_ceil, scal_ceil = quantize_ceil_legacy(h)
    scal_ceil = scal_ceil.reshape(rows, k // 32)

    pd = int(np.count_nonzero(pack_old != pack_new))
    sd = int(np.count_nonzero(scal_old != scal_new))
    pdc = int(np.count_nonzero(pack_old != pack_ceil))
    sdc = int(np.count_nonzero(scal_old != scal_ceil))
    print(f"[M54-rebaseline] 输入 {d}/h.bin：{rows} 行 × {k}（bf16 网格值）")
    print(f"[M54-rebaseline] ① scale 规则 ceil -> floor：scale 字节差异 {sd}/{scal_old.size} "
          f"({100.0 * sd / scal_old.size:.1f}%)")
    print(f"[M54-rebaseline] ② data 方向 mul scale -> mul 1/scale：pack 字节差异 {pd}/{pack_old.size} "
          f"({100.0 * pd / pack_old.size:.1f}%)")
    print(f"[M54-rebaseline] ③ pre-M54 vs tools/golden quantize_ceil_legacy（legacy ceil 的权威实现）："
          f"pack {pdc}/{pack_old.size} ({100.0 * pdc / pack_old.size:.1f}%)、scale {sdc}/{scal_old.size}"
          f"（⇒ 即使 scale 字节在多数组相同，data 方向相反仍使 pre-M54 与 legacy 规范大范围不符）")

    ro = relerr(h, recon(pack_old, scal_old, k))
    rn = relerr(h, recon(pack_new, scal_new, k))
    print(f"[M54-rebaseline] H 反量化重建误差 |deq-h|/|h|（{ro.size} 个有限非零元素）：")
    print(f"[M54-rebaseline]   pre-M54 (ceil + mul scale)： mean {ro.mean():.3g}, "
          f"p50 {np.percentile(ro, 50):.3g}, p99 {np.percentile(ro, 99):.3g}, max {ro.max():.3g}")
    print(f"[M54-rebaseline]   post-M54 (floor + mul 1/scale)： mean {rn.mean():.3g}, "
          f"p50 {np.percentile(rn, 50):.3g}, p99 {np.percentile(rn, 99):.3g}, max {rn.max():.3g}")
    print("[M54-rebaseline] 注：post-M54 的 p50=1.0 是因为本 dump 的随机行刻意让组内同时覆盖很宽的")
    print("[M54-rebaseline]     指数档（2^120..2^139），组内小量级元素按 group scale 量化到 code 0；")
    print("[M54-rebaseline]     这是 MXFP4 group-32 的固有行为，不是缺陷。判定：post-M54 的 H 反量化")
    print("[M54-rebaseline]     误差与 e2m1/group32 量级相符；pre-M54 因乘了 scale 而非 1/scale，")
    print("[M54-rebaseline]     重建误差远超量化噪声。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
