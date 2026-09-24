"""Micro-bench a single GEMV across block configs to find achieved bandwidth.

Batch-1 GEMV is pure weight streaming, so achieved GB/s vs the 1008 GB/s peak is
the only number that matters. The two shapes that dominate Qwen2.5-1.5B decode:
gate/up (N=8960, K=1536) and down (N=1536, K=8960). down is the trap -- few
output rows means few programs unless the block is small enough to fill the SMs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(x_ptr, w_ptr, o_ptr, K, N, sw_n, sw_k, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        mk = ok < K
        xv = tl.load(x_ptr + ok, mask=mk, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * sw_n + ok[None, :] * sw_k,
                    mask=(offs_n < N)[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    tl.store(o_ptr + offs_n, acc.to(tl.bfloat16), mask=offs_n < N)


def bench(N, K, BN, BK, warps, reps=200):
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(N, device="cuda", dtype=torch.bfloat16)
    grid = (triton.cdiv(N, BN),)
    fn = lambda: _gemv[grid](x, W, o, K, N, W.stride(0), W.stride(1), BN, BK, num_warps=warps)
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        t.append(st.elapsed_time(en))
    t.sort()
    ms = t[len(t) // 2]
    gb = N * K * 2 / 1e9
    return ms, gb / (ms / 1e3)


if __name__ == "__main__":
    shapes = [("gate/up", 8960, 1536), ("down", 1536, 8960), ("qkv", 2048, 1536), ("o", 1536, 1536)]
    for name, N, K in shapes:
        print(f"\n{name}  [{N}x{K}]  ({N*K*2/1e6:.1f} MB), grid=cdiv(N,BN)")
        best = None
        for BN in (8, 16, 32, 64):
            for BK in (256, 512, 1024):
                for warps in (2, 4, 8):
                    ms, bw = bench(N, K, BN, BK, warps)
                    if best is None or bw > best[0]:
                        best = (bw, BN, BK, warps, ms)
        bw, BN, BK, warps, ms = best
        print(f"  best: {bw:.0f} GB/s ({100*bw/1008:.0f}% peak)  "
              f"BN={BN} BK={BK} warps={warps}  {ms*1000:.1f} us")
