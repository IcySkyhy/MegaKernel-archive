# SPDX-License-Identifier: Apache-2.0
"""Cross-check for m11_bf16_gemm: numpy fp32 reference vs device output.

The ASC host binary's own reference (exact-integer domain, fp32 accumulate) is
checked in-binary bit-exactly. This script independently recomputes the golden
from a.bin/b.bin (dump mode) with numpy fp32 matmul — on the generated data
(small integers, |sum| <= K*64 < 2^24) fp32 accumulation is rounding-free in any
order, so the numpy fp32 result equals both the hardware cube accumulation and
the C host reference bit-for-bit before bf16 rounding; comparison is on bf16
bits after RNE rounding (f32_to_bf16_bits, same grid as hardware F322BF16).

Usage (from a dir containing a.bin/b.bin/c_device.bin produced by
`m11_bf16_gemm dump <K> <N> <m>`):
    python3 check_ref.py <K> <N> <m>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits


def main():
    k, n, m = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    a = np.fromfile("a.bin", dtype=np.uint16).reshape(-1, k)  # mAlloc = max(m,2) rows
    b = np.fromfile("b.bin", dtype=np.uint16).reshape(n, k)
    c = np.fromfile("c_device.bin", dtype=np.uint16).reshape(m, n)

    a32 = bf16_bits_to_f32(a)[:m].astype(np.float32)  # [m,K] fp32
    b32 = bf16_bits_to_f32(b).astype(np.float32)      # [N,K] fp32
    ref = (a32 @ b32.T).astype(np.float32)            # fp32 matmul (BLAS, exact on this domain)
    ref_bits = f32_to_bf16_bits(ref)

    ok = np.array_equal(ref_bits, c)
    print(f"numpy fp32 ref == device C: {ok}")
    if not ok:
        bad = np.argwhere(ref_bits != c)
        print(f"mismatches: {len(bad)}, first 5:")
        for i, j in bad[:5]:
            print(f"  C[{i}][{j}]: got 0x{c[i, j]:04x} expect 0x{ref_bits[i, j]:04x}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
