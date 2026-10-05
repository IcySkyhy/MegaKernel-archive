# SPDX-License-Identifier: Apache-2.0
"""Cross-check for the m5_swiglu_quant host/device outputs vs the numpy reference.

The m5 kernel fuses SwiGLU (bf16 gate|up [m,1280] -> bf16 [m,640]) with OCP MXFP4
quantization (qx [m,320] u8 lohi nibbles + scale [m,20] u8 E8M0, group 32), using the
same quant semantics as m2_mxfp4_quant (npu_dynamic_mx_quant round_mode="round",
scale_alg=0/OCP). This script is the numpy side of the validation chain:

  SwiGLU reference : silu(g)*u computed in float32 from bf16 inputs, RNE back to bf16
                     (RNE bf16 helpers from tools/golden/moe_block_ref.py).
  Quant reference  : ocp_numpy below -- a numpy transcription of quant_ref.py::
                     ocp_dynamic_mx_quant (same algorithm m2's check_ref.py verified),
                     with the e2m1 sign taken from signbit(xs) to match hardware
                     CAST_ROUND(-0) = -0 (m5 exercises -0 via SwiGLU underflow;
                     quant_ref.py's literal xs<0 rule was never validated there -- m2
                     avoided the corner by generating only +0 zero groups).

Checks (all required to pass):
  (a1) C host silu ref   == numpy silu      within 1 bf16 ULP (bf16-grid tolerance)
  (a2) device silu       == numpy silu      within 1 bf16 ULP
  (b1) C host quant ref  == numpy quant(silu_ref)    byte-exact
  (b2) device qx/scale   == numpy quant(silu_device) byte-exact
Plus an informational full-chain agreement: numpy quant(numpy silu) vs device qx.

Usage (from a dir containing the dumps of case (m, seed, inf); see m5 README):
    ../../build/m5_swiglu_quant dump <m> <seed> [inf]  # m{m}_s{seed}_i{inf}_{x,swiglu_ref,qx_ref,scale_ref}.bin
    M5_DUMP=1 ../../build/m5_swiglu_quant               # adds the *_device.bin for every case, incl. inf=1..3
    /usr/local/python3.12.13/bin/python3 check_ref.py <m> <seed> [inf]
inf = 0 baseline / 1 whole-group +Inf / 2 partial ±Inf / 3 NaN group (the ①②③ cases of mission M32).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "golden"))
from moe_block_ref import bf16_bits_to_f32  # 位精确 bf16->f32（bf16 即 fp32 高 16 位）

FP4_MIDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def f32_to_bf16_bits_keep_nan(x: np.ndarray) -> np.ndarray:
    """同 moe_block_ref.f32_to_bf16_bits 的 RNE 位运算，但**不把 NaN 规范化成 0x7FC0**。
    设备的 e2m1 Cast 对 NaN 给 ±0，符号位取自乘出来的 NaN（硬件实测：输入 -NaN 的组
    出 0x8、+NaN 出 0x0），C 参考的 F32ToBf16 亦保留符号位；规范化会丢掉符号位，
    使非有限用例的 numpy 判据与 C 参考/设备不一致。"""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return rounded.astype(np.uint16)


def silu_numpy(x_bits: np.ndarray):
    """x_bits: [M, 1280] uint16 bf16 gate|up -> swiglu bf16 bits [M, 640] uint16."""
    gate = bf16_bits_to_f32(x_bits[:, :640])
    up = bf16_bits_to_f32(x_bits[:, 640:])
    y = gate / (1.0 + np.exp(-gate)) * up   # float32 throughout
    return f32_to_bf16_bits_keep_nan(y.astype(np.float32))


def ocp_numpy(x_bits: np.ndarray):
    """x_bits: [M, K] uint16 bf16 bit patterns -> (qx [M,K/2] u8, scale [M,K/32] u8)."""
    m, k = x_bits.shape
    g = k // 32
    xf = bf16_bits_to_f32(x_bits)

    maxexp = (x_bits.astype(np.uint32) & 0x7F80).reshape(m, g, 32).max(axis=2)
    is_inf = maxexp == 0x7F80
    maxexp = np.maximum(maxexp, 0x0100)
    shared = maxexp - 0x0100
    scale_flat = (shared >> 7).astype(np.uint8)
    scale_flat = np.where(is_inf, 0xFF, scale_flat)
    hs_bits = (0x7F00 - shared).astype(np.uint16)
    hs_bits = np.where(shared == 0, 0, hs_bits)
    hs_bits = np.where(is_inf, 0x7F81, hs_bits)

    # 官方 :413 语义：非有限组 halfScale = 0x7F81（bf16 NaN）-> Mul(±Inf/NaN, NaN) = NaN
    # -> Cast(NaN) = 0.0。位精确解码下 0x7F81 就是 NaN，numpy 的 NaN 传播与设备一致：
    #   ±Inf * NaN -> +NaN -> +0；-NaN * NaN -> -NaN -> -0(0x8)；有限值 * NaN -> +NaN -> +0
    hs_f = bf16_bits_to_f32(hs_bits)[:, :, None]
    xs32 = xf.reshape(m, g, 32) * hs_f
    xs_bf16 = bf16_bits_to_f32(f32_to_bf16_bits_keep_nan(xs32.astype(np.float32)))  # RNE round trip

    a = np.abs(xs_bf16)
    idx = np.digitize(a, FP4_MIDS)                      # round half away from zero, tie -> upper
    idx = np.where(np.isnan(xs_bf16), 0, idx)           # Cast(NaN) = 0.0（digitize(NaN) 给 7，是错的）
    sign = np.where(np.signbit(xs_bf16), np.uint8(0x8), np.uint8(0))  # hardware keeps -0 / -NaN
    nib = idx.astype(np.uint8) | sign

    lo = nib[:, :, 0::2]
    hi = nib[:, :, 1::2]
    qx = (lo | (hi << 4)).astype(np.uint8).reshape(m, k // 2)
    scale = scale_flat.reshape(m, g)
    return qx, scale


def bf16_ulp_dist(a: np.ndarray, b: np.ndarray):
    """uint16 bf16 bit arrays -> (max ulp distance, exact count); +-0 counts as equal.
    非有限值：同类即等（NaN 的符号/payload 与 ±Inf 的符号归属不参与，只要求两边同类）；
    一边 NaN 一边有限/Inf 判为明确不同（0xFFFF），避免掩盖分类错误。"""
    both_zero = ((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0)
    a_inf = (a & 0x7FFF) == 0x7F80
    b_inf = (b & 0x7FFF) == 0x7F80
    a_nan = ((a & 0x7F80) == 0x7F80) & ((a & 0x007F) != 0)
    b_nan = ((b & 0x7F80) == 0x7F80) & ((b & 0x007F) != 0)
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    d = np.where(both_zero, 0, d)
    d = np.where(a_inf & b_inf, np.where(a == b, 0, 1), d)
    d = np.where(a_nan & b_nan, 0, d)
    d = np.where(a_nan ^ b_nan, 0xFFFF, d)
    return int(d.max()), int((d == 0).sum()), int(a.size)


def check_silu(tag, ref_bits, y_np):
    d_max, exact, total = bf16_ulp_dist(ref_bits, y_np)
    ok = d_max <= 1
    print(f"(a{tag}) silu bf16: maxULP={d_max} exact={exact}/{total} : {'PASS' if ok else 'FAIL'}")
    return ok


def check_quant(tag, x_bits, qx_expected, s_expected):
    qx, s = ocp_numpy(x_bits)
    ok_q = np.array_equal(qx.reshape(-1), qx_expected)
    ok_s = np.array_equal(s.reshape(-1), s_expected)
    print(f"(b{tag}) mxfp4 byte-exact: qx={ok_q} scale={ok_s} "
          f"({qx_expected.size} B + {s_expected.size} B) : {'PASS' if (ok_q and ok_s) else 'FAIL'}")
    if not ok_q:
        bad = np.nonzero(qx.reshape(-1) != qx_expected)[0]
        print(f"  qx mismatches={len(bad)}, first at byte {bad[0]}: "
              f"got=0x{qx_expected[bad[0]]:02x} expect=0x{qx.reshape(-1)[bad[0]]:02x}")
    if not ok_s:
        bad = np.nonzero(s.reshape(-1) != s_expected)[0]
        print(f"  scale mismatches={len(bad)}, first at byte {bad[0]}: "
              f"got=0x{s_expected[bad[0]]:02x} expect=0x{s.reshape(-1)[bad[0]]:02x}")
    return ok_q and ok_s


def main():
    # 用法：check_ref.py <m> <seed> [infKind]（infKind 0/1/2/3 = 基线 / 整组 +Inf / 部分 ±Inf / NaN 组）
    m, seed = int(sys.argv[1]), int(sys.argv[2])
    inf = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    p = f"m{m}_s{seed}_i{inf}"
    x = np.fromfile(f"{p}_x.bin", dtype=np.uint16).reshape(m, 1280)
    y_ref = np.fromfile(f"{p}_swiglu_ref.bin", dtype=np.uint16).reshape(m, 640)
    y_dev = np.fromfile(f"{p}_swiglu_device.bin", dtype=np.uint16).reshape(m, 640)
    qx_ref = np.fromfile(f"{p}_qx_ref.bin", dtype=np.uint8)
    s_ref = np.fromfile(f"{p}_scale_ref.bin", dtype=np.uint8)
    qx_dev = np.fromfile(f"{p}_qx_device.bin", dtype=np.uint8)
    s_dev = np.fromfile(f"{p}_scale_device.bin", dtype=np.uint8)

    y_np = silu_numpy(x)

    ok = True
    ok &= check_silu("1 C-ref", y_ref, y_np)
    ok &= check_silu("2 device", y_dev, y_np)
    ok &= check_quant("1 C-ref", y_ref, qx_ref, s_ref)
    ok &= check_quant("2 device", y_dev, qx_dev, s_dev)

    # informational full chain: numpy silu -> numpy quant vs device qx
    qx_np, _ = ocp_numpy(y_np)
    agree = int((qx_np.reshape(-1) == qx_dev).sum())
    print(f"info: full-chain numpy-silu -> numpy-quant vs device qx: {agree}/{qx_dev.size} bytes agree")
    print("===== ALL PASS =====" if ok else "===== FAILURES PRESENT =====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
