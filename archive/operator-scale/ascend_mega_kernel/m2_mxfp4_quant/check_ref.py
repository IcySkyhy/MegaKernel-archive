# SPDX-License-Identifier: Apache-2.0
"""Cross-check for the m2_mxfp4_quant host reference vs the quant_ref.py algorithm.

Literal numpy transcription of quant_ref.py::ocp_dynamic_mx_quant (the reference
verified bit-exact against torch_npu.npu_dynamic_mx_quant, round_mode="round",
scale_alg=0/OCP, group=32). torch is unavailable in this environment, so the two
torch calls are replaced by numpy equivalents with identical semantics:
  * x.view(torch.int16)           -> direct uint16 load of the bf16 bits
  * torch f32 -> bf16(RNE) -> f32 -> moe_block_ref.f32_to_bf16 (pure numpy RNE)

Usage (from a dir containing x.bin/qx.bin/scale.bin produced by
`m2_mxfp4_quant dump <K> <m> <seed>`):
    /usr/local/python3.12.13/bin/python3 check_ref.py <K> <m> <seed>
"""
import sys

import numpy as np

sys.path.insert(0, "/workspace/ascend_mega_kernel/tools/golden")
from moe_block_ref import f32_to_bf16  # pure-numpy RNE bf16 round trip

FP4_MIDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    """位精确 bf16 -> float32：bf16 就是 fp32 的高 16 位，左移 16 位后按 fp32 重解释。
    （早先的算术式解码会把 bf16 NaN 0x7F81/0x7FC0 解成 ±Inf，非有限用例下会算错符号。）"""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def ocp_numpy(x_bits: np.ndarray):
    """x_bits: [M, K] uint16 bf16 bit patterns -> (qx [M,K/2] u8, scale [M,K/32] u8)."""
    m, k = x_bits.shape
    g = k // 32
    xf = bf16_bits_to_fp32(x_bits)

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
    # -> Cast(NaN) = 0.0，组内所有 nibble 归零（±Inf 组实测与设备逐字节一致）。
    hs_f = bf16_bits_to_fp32(hs_bits)[:, :, None]
    xs32 = xf.reshape(m, g, 32) * hs_f
    xs_bf16 = f32_to_bf16(xs32.astype(np.float32))

    a = np.abs(xs_bf16)
    idx = np.digitize(a, FP4_MIDS)
    idx = np.where(np.isnan(xs_bf16), 0, idx)   # Cast(NaN) = 0.0（digitize(NaN) 给 7，是错的）
    # 符号取 signbit：与设备一致（硬件 Cast(-NaN) = -0 保留 NaN 的符号位；C 参考的
    # `xs < 0` 在 NaN 上同样表现为符号位，二者与本判据在生成集上逐字节一致）
    sign = np.where(np.signbit(xs_bf16), np.uint8(0x8), np.uint8(0))
    nib = idx.astype(np.uint8) | sign

    lo = nib[:, :, 0::2]
    hi = nib[:, :, 1::2]
    qx = (lo | (hi << 4)).astype(np.uint8).reshape(m, k // 2)
    scale = scale_flat.reshape(m, g)
    return qx, scale


def main():
    k, m = int(sys.argv[1]), int(sys.argv[2])
    x = np.fromfile("x.bin", dtype=np.uint16).reshape(m, k)
    c_qx = np.fromfile("qx.bin", dtype=np.uint8)
    c_s = np.fromfile("scale.bin", dtype=np.uint8)

    py_qx, py_s = ocp_numpy(x)
    py_qx = py_qx.reshape(-1)
    py_s = py_s.reshape(-1)

    ok_q = np.array_equal(c_qx, py_qx)
    ok_s = np.array_equal(c_s, py_s)
    print(f"C-ref qx == quant_ref-algorithm qx: {ok_q}")
    print(f"C-ref scale == quant_ref-algorithm scale: {ok_s}")
    if not ok_q:
        bad = np.nonzero(c_qx != py_qx)[0]
        print(f"qx mismatches: {len(bad)}, first at byte {bad[0]}: "
              f"C=0x{c_qx[bad[0]]:02x} py=0x{py_qx[bad[0]]:02x}")
    if not ok_s:
        bad = np.nonzero(c_s != py_s)[0]
        print(f"scale mismatches: {len(bad)}, first at byte {bad[0]}: "
              f"C=0x{c_s[bad[0]]:02x} py=0x{py_s[bad[0]]:02x}")
    sys.exit(0 if (ok_q and ok_s) else 1)


if __name__ == "__main__":
    main()
