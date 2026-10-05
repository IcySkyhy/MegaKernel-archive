# SPDX-License-Identifier: Apache-2.0
"""Cross-check for m7_router_topk device outputs vs the moe_block_ref.py golden.

The definitive numerical check required by the M7 mission: compare device dumps
(produced by `M7_OUT=<dir> ./m7_router_topk`) against
tools/golden/moe_block_ref.py::router_topk (the CPU golden for the routing
semantics: softmax over 512 experts -> top-10 descending (ties -> lower id) ->
renormalize by norm_topk_prob).

Checks:
  * topk_ids   : EXACT equality (int32)
  * topk_weights: device bf16 vs golden fp32 rounded to bf16(RNE), |ulp| <= 1
                  (golden weights that are themselves fp32-denormal may flush
                  to zero on device; accepted, |w| < 1e-38)
  * router_logits: golden returns max-shifted logits (logits - row max); the
                  kernel outputs the same shifted form. Tolerance: absolute
                  2e-2 on shifted logits (fp32 accumulation-order noise on
                  near-zero shifted values dominates relative error).

Usage:
    M7_OUT=./m7_out ./build/m7_router_topk
    /usr/local/python3.12.13/bin/python3.12 m7_router_topk/check_ref.py ./m7_out/m1
    /usr/local/python3.12.13/bin/python3.12 m7_router_topk/check_ref.py ./m7_out/m33
"""
import os
import sys

import numpy as np

# 优先用**本仓相对**的 tools/golden（worktree / 主 checkout / 独立 clone 都适用），
# 只在不存在的旧布局下才回落到绝对路径（此前硬编码绝对路径 ⇒ 换 checkout 即 ImportError）。
_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tools", "golden")
if not os.path.isdir(_GOLDEN):
    _GOLDEN = "/workspace/ascend_mega_kernel/tools/golden"
sys.path.insert(0, _GOLDEN)
from moe_block_ref import f32_to_bf16, f32_to_bf16_bits, router_topk  # noqa: E402

HIDDEN = 2560
E = 512
KTOPK = 10


def load_case(d: str):
    x = np.fromfile(f"{d}/x.bin", dtype=np.uint16).astype(np.uint32)
    x = (x.reshape(-1, HIDDEN) << 16).view(np.float32)
    w = np.fromfile(f"{d}/router_weight.bin", dtype=np.uint16).astype(np.uint32)
    w = (w.reshape(E, HIDDEN) << 16).view(np.float32)
    logits = np.fromfile(f"{d}/router_logits.bin", dtype=np.float32).reshape(-1, E)
    ids = np.fromfile(f"{d}/topk_ids.bin", dtype=np.int32).reshape(-1, KTOPK)
    wbits = np.fromfile(f"{d}/topk_weights.bin", dtype=np.uint16).reshape(-1, KTOPK)
    wdev = (wbits.astype(np.uint32) << 16).view(np.float32)
    return x, w, logits, ids, wdev, wbits


def main() -> int:
    d = sys.argv[1]
    x, w, logits_dev, ids_dev, wdev, wbits_dev = load_case(d)
    m = x.shape[0]
    logits_ref, ids_ref, wref = router_topk(x, w, KTOPK)

    ok = True

    ids_ok = np.array_equal(ids_dev, ids_ref)
    print(f"topk_ids exact:      {ids_ok}")
    if not ids_ok:
        bad = np.argwhere(ids_dev != ids_ref)
        print(f"  mismatches: {len(bad)}; first: {bad[:5].tolist()}")
    ok &= ids_ok

    wref_bf = f32_to_bf16_bits(wref)
    ulp = np.abs(wbits_dev.astype(np.int32) - wref_bf.astype(np.int32))
    tiny = np.abs(wref) < 1e-38
    w_ok = bool((ulp <= 1).all() or ((ulp[~tiny] <= 1).all() and (wdev[tiny] == 0).all()))
    print(f"topk_weights bf16-grid (<=1 ulp, denormal-flush allowed): {w_ok} "
          f"(max ulp {int(ulp.max())})")
    ok &= w_ok

    dmax = float(np.abs(logits_dev - logits_ref).max())
    l_ok = dmax <= 2e-2
    print(f"router_logits (max-shifted) abs diff <= 2e-2: {l_ok} (max {dmax:.3e})")
    ok &= l_ok

    print(f"[check_ref] m={m} : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
