# SPDX-License-Identifier: Apache-2.0
"""Cross-check for m12_rmsnorm_gated: C host reference & NPU device outputs vs the
numpy RMSNormGated reference (fp32 accumulation, bf16-grid 1e-2 relative tolerance).

numpy reference (semantics identical to RefRmsNormGated in m12_rmsnorm_gated.asc,
field-by-field aligned with vllm RMSNormGated, norm_before_gate=True, act=sigmoid):
    per head h of 48 (head_v_dim = 128, vllm variance dim=-1):
        var   = sum(o^2) / 128            (fp32 accumulation, eps dim = 128 not 6144)
        rstd  = 1 / sqrt(var + 1e-6)      (vllm config.rms_norm_eps = 1e-6)
        out   = bf16( ((o * rstd) * gamma) * sigmoid(z) )   (RNE)
    gamma is [128] shared by all heads (vllm norm.weight shape).

Usage (from a dir containing the bins produced by
`m12_rmsnorm_gated dump <m> <seed>` and/or `M12_DUMP=1 ./m12_rmsnorm_gated`):
    /usr/local/python3.12.13/bin/python3 check_ref.py <m> <seed>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # pure-numpy bf16 helpers

HEADS = 48
HEAD = 128
HIDDEN = HEADS * HEAD
EPS = np.float32(1e-6)


def numpy_ref(o: np.ndarray, z_bits: np.ndarray, gamma_bits: np.ndarray):
    """o: [m, 6144] float32; z_bits: [m, 6144] uint16 bf16 bits; gamma_bits: [128]
    uint16 -> y_bits [m, 6144] uint16."""
    ov = o.astype(np.float32).reshape(-1, HEADS, HEAD)
    zv = bf16_bits_to_f32(z_bits).astype(np.float32).reshape(-1, HEADS, HEAD)
    gf = bf16_bits_to_f32(gamma_bits).astype(np.float32)
    var = np.sum(ov * ov, axis=2, dtype=np.float32) / np.float32(HEAD)  # fp32 累加
    rstd = (np.float32(1.0) / np.sqrt(var + EPS)).astype(np.float32)
    with np.errstate(over="ignore"):  # z=-8192 饱和行 exp 上溢 → sigmoid 精确 0（两侧一致）
        sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-zv, dtype=np.float32))
    out = ((ov * rstd[:, :, None]) * gf[None, None, :]) * sig
    return f32_to_bf16_bits(out.reshape(-1, HIDDEN)).astype(np.uint16)


def y_rel_err(got_bits: np.ndarray, exp_bits: np.ndarray) -> np.ndarray:
    got = bf16_bits_to_f32(got_bits.astype(np.uint16))
    exp = bf16_bits_to_f32(exp_bits.astype(np.uint16))
    num = np.abs(got - exp)
    den = np.abs(exp)
    return np.where(den > 0, num / np.maximum(den, np.float32(1e-38)), np.where(got == 0, 0.0, 1.0))


def compare(tag, y_bits, exp_y, tol=1e-2):
    ok_y = np.array_equal(y_bits.shape, exp_y.shape)
    rel = y_rel_err(y_bits, exp_y) if ok_y else None
    y_ok = ok_y and bool(np.all(rel <= tol))
    print(f"{tag}: out rows all within {tol} rel: {y_ok} (max rel {rel.max() if ok_y else float('nan'):.3g})")
    if not y_ok and ok_y:
        bad = np.argwhere(rel > tol)
        i, j = bad[0]
        print(f"  first out mismatch at [{i}][{j}]: got {bf16_bits_to_f32(y_bits[i, j])!r} "
              f"expect {bf16_bits_to_f32(exp_y[i, j])!r} rel {rel[i, j]:.3g}, total {len(bad)}")
    return y_ok


def main():
    m, seed = int(sys.argv[1]), int(sys.argv[2])
    o = np.fromfile(f"m{m}_s{seed}_o.bin", dtype=np.float32).reshape(m, HIDDEN)
    z = np.fromfile(f"m{m}_s{seed}_z.bin", dtype=np.uint16).reshape(m, HIDDEN)
    gamma = np.fromfile(f"m{m}_s{seed}_gamma.bin", dtype=np.uint16)

    exp_y = numpy_ref(o, z, gamma)

    ok = True
    c_y = f"m{m}_s{seed}_y.bin"
    if os.path.exists(c_y):
        ok &= compare("C-ref   vs numpy", np.fromfile(c_y, dtype=np.uint16).reshape(m, HIDDEN), exp_y)
    else:
        print(f"skip C-ref check ({c_y} not present; run `m12_rmsnorm_gated dump {m} {seed}` first)")

    d_y = f"m{m}_s{seed}_y_device.bin"
    if os.path.exists(d_y):
        ok &= compare("NPU-dev vs numpy", np.fromfile(d_y, dtype=np.uint16).reshape(m, HIDDEN), exp_y)
    else:
        print(f"skip NPU-device check ({d_y} not present; run with M12_DUMP=1 first)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
