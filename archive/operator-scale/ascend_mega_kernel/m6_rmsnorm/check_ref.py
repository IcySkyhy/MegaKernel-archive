# SPDX-License-Identifier: Apache-2.0
"""Cross-check for m6_rmsnorm: C host reference & NPU device outputs vs the numpy
Add+RMSNorm reference (fp32 accumulation, bf16-grid 1e-2 relative tolerance on y,
bit-exact on the fp32 residual output).

numpy reference (semantics identical to RefAddRmsNorm in m6_rmsnorm.asc):
    xAdd      = bf16(x) + bf16(res)              (fp32)
    resOut    = xAdd                             (fp32, bit-exact expected)
    rstd      = 1 / sqrt(mean(xAdd^2) + eps)     (fp32 accumulation, eps=1e-6)
    y         = bf16((xAdd * rstd) * gamma)      (RNE)

Usage (from a dir containing the bins produced by
`m6_rmsnorm dump <m> <seed>` and/or `M6_DUMP=1 ./m6_rmsnorm`):
    /usr/local/python3.12.13/bin/python3 check_ref.py <m> <seed>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # pure-numpy bf16 helpers

HIDDEN = 2560
EPS = np.float32(1e-6)


def numpy_ref(x_bits: np.ndarray, res_bits: np.ndarray, gamma_bits: np.ndarray):
    """x/res: [m, 2560] uint16 bf16 bits; gamma: [2560] uint16 -> (y_bits u16, resOut f32)."""
    xf = bf16_bits_to_f32(x_bits).astype(np.float32)
    rf = bf16_bits_to_f32(res_bits).astype(np.float32)
    gf = bf16_bits_to_f32(gamma_bits).astype(np.float32)
    xAdd = xf + rf                                    # fp32 残差加
    ss = np.sum(xAdd * xAdd, axis=1, dtype=np.float32)  # fp32 累加
    rstd = (np.float32(1.0) / np.sqrt(ss / np.float32(HIDDEN) + EPS)).astype(np.float32)
    y = f32_to_bf16_bits((xAdd * rstd[:, None]) * gf[None, :])
    return y.astype(np.uint16), xAdd


def y_rel_err(got_bits: np.ndarray, exp_bits: np.ndarray) -> np.ndarray:
    got = bf16_bits_to_f32(got_bits.astype(np.uint16))
    exp = bf16_bits_to_f32(exp_bits.astype(np.uint16))
    num = np.abs(got - exp)
    den = np.abs(exp)
    return np.where(den > 0, num / np.maximum(den, np.float32(1e-38)), np.where(got == 0, 0.0, 1.0))


def compare(tag, y_bits, res_out, exp_y, exp_res_out, tol=1e-2):
    ok_y = np.array_equal(y_bits.shape, exp_y.shape)
    rel = y_rel_err(y_bits, exp_y) if ok_y else None
    y_ok = ok_y and bool(np.all(rel <= tol))
    ro_ok = bool(np.array_equal(res_out, exp_res_out))
    print(f"{tag}: y rows all within {tol} rel: {y_ok} (max rel {rel.max() if ok_y else float('nan'):.3g}), "
          f"resOut bit-exact: {ro_ok}")
    if not y_ok and ok_y:
        bad = np.argwhere(rel > tol)
        i, j = bad[0]
        print(f"  first y mismatch at [{i}][{j}]: got {bf16_bits_to_f32(y_bits[i, j])!r} "
              f"expect {bf16_bits_to_f32(exp_y[i, j])!r} rel {rel[i, j]:.3g}, total {len(bad)}")
    if not ro_ok:
        bad = np.argwhere(res_out != exp_res_out)
        i, j = bad[0]
        print(f"  first resOut mismatch at [{i}][{j}]: got {res_out[i, j]!r} expect {exp_res_out[i, j]!r}, "
              f"total {len(bad)}")
    return y_ok and ro_ok


def main():
    m, seed = int(sys.argv[1]), int(sys.argv[2])
    x = np.fromfile(f"m{m}_s{seed}_x.bin", dtype=np.uint16).reshape(m, HIDDEN)
    res = np.fromfile(f"m{m}_s{seed}_res.bin", dtype=np.uint16).reshape(m, HIDDEN)
    gamma = np.fromfile(f"m{m}_s{seed}_gamma.bin", dtype=np.uint16)

    exp_y, exp_res_out = numpy_ref(x, res, gamma)

    ok = True
    c_y = f"m{m}_s{seed}_y.bin"
    c_ro = f"m{m}_s{seed}_resout.bin"
    if os.path.exists(c_y) and os.path.exists(c_ro):
        ok &= compare("C-ref   vs numpy", np.fromfile(c_y, dtype=np.uint16).reshape(m, HIDDEN),
                      np.fromfile(c_ro, dtype=np.float32).reshape(m, HIDDEN), exp_y, exp_res_out)
    else:
        print(f"skip C-ref check ({c_y}/{c_ro} not present; run `m6_rmsnorm dump {m} {seed}` first)")

    d_y = f"m{m}_s{seed}_y_device.bin"
    d_ro = f"m{m}_s{seed}_resout_device.bin"
    if os.path.exists(d_y) and os.path.exists(d_ro):
        ok &= compare("NPU-dev vs numpy", np.fromfile(d_y, dtype=np.uint16).reshape(m, HIDDEN),
                      np.fromfile(d_ro, dtype=np.float32).reshape(m, HIDDEN), exp_y, exp_res_out)
    else:
        print(f"skip NPU-device check ({d_y}/{d_ro} not present; run with M6_DUMP=1 first)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
