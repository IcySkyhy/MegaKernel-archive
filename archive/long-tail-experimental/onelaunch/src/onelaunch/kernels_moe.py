"""MoE-specific Triton kernels for OLMoE batch-1 decode.

Two things a dense decode doesn't have: QK-norm (an RMSNorm over the full q and k
projections before RoPE) and a top-8-of-64 expert FFN. The expert kernels read
weights through an *indirect* index -- the 8 chosen expert ids live in a GPU
tensor the router fills each step -- so only the active 8/64 experts are ever
touched, and the launch stays shape-stable for CUDA-graph capture even though
*which* experts run changes token to token. The dense GEMV/attention kernels are
reused from kernels.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernels import _gemv_configs


# ---- QK-norm + RoPE + KV-cache write ---------------------------------------
# The q/k RMS scales are single reductions over the full 2048-wide projection,
# so they're computed in torch (2 tiny ops) and passed in; this kernel applies
# scale * per-element norm weight, then rotate_half RoPE, then writes k/v cache.
@triton.jit
def _qknorm_rope_write(qkv_ptr, qn_ptr, kn_ptr, qs_ptr, ks_ptr, cos_ptr, sin_ptr,
                       qo_ptr, kc_ptr, vc_ptr, pos_ptr, qd, kd, sk_t, sk_h,
                       NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.arange(0, HD)
    half = HD // 2
    cos = tl.load(cos_ptr + d).to(tl.float32)
    sin = tl.load(sin_ptr + d).to(tl.float32)
    sgn = tl.where(d < half, -1.0, 1.0)
    roti = tl.where(d < half, d + half, d - half)
    if pid < NH:
        base = pid * HD
        scale = tl.load(qs_ptr)
        g = tl.load(qn_ptr + base + d).to(tl.float32)
        x = tl.load(qkv_ptr + base + d).to(tl.float32) * scale * g
        xr = (tl.load(qkv_ptr + base + roti).to(tl.float32)
              * scale * tl.load(qn_ptr + base + roti).to(tl.float32))
        tl.store(qo_ptr + base + d, (x * cos + sgn * xr * sin).to(tl.bfloat16))
    else:
        kk = pid - NH
        pos = tl.load(pos_ptr)
        scale = tl.load(ks_ptr)
        kbase = kk * HD
        g = tl.load(kn_ptr + kbase + d).to(tl.float32)
        x = tl.load(qkv_ptr + qd + kbase + d).to(tl.float32) * scale * g
        xr = (tl.load(qkv_ptr + qd + kbase + roti).to(tl.float32)
              * scale * tl.load(kn_ptr + kbase + roti).to(tl.float32))
        dst = pos * sk_t + kk * sk_h + d
        tl.store(kc_ptr + dst, (x * cos + sgn * xr * sin).to(tl.bfloat16))
        v = tl.load(qkv_ptr + qd + kd + kk * HD + d)
        tl.store(vc_ptr + dst, v)


def qknorm_rope_write(qkv, qn, kn, qscale, kscale, cos, sin, kc_layer, vc_layer,
                      pos, nh, nkv, hd):
    qo = torch.empty(nh, hd, device=qkv.device, dtype=torch.bfloat16)
    _qknorm_rope_write[(nh + nkv,)](qkv, qn, kn, qscale, kscale, cos, sin,
                                    qo, kc_layer, vc_layer, pos, nh * hd, nkv * hd,
                                    kc_layer.stride(0), kc_layer.stride(1), nh, nkv, hd)
    return qo


# ---- expert gate/up GEMV + SwiGLU (indirect-indexed, top-k experts) --------
# grid (TOP_K, tiles of I): program (j, tile) runs the SwiGLU for the j-th chosen
# expert on a tile of intermediate rows, gathering that expert's fused
# gate_up_proj[e] by the id read from topi[j]. Writes act[j, :I] (fp32).
@triton.autotune(configs=_gemv_configs(), key=["H", "I"])
@triton.jit
def _expert_swiglu(x_ptr, gu_ptr, topi_ptr, act_ptr, H, I, se, sr, sa_j,
                   BN: tl.constexpr, BK: tl.constexpr):
    j = tl.program_id(0)
    tile = tl.program_id(1)
    e = tl.load(topi_ptr + j)
    offs_r = tile * BN + tl.arange(0, BN)
    mr = offs_r < I
    gate_row = e * se + offs_r[:, None] * sr           # gate rows [0, I)
    up_row = e * se + (I + offs_r)[:, None] * sr        # up rows   [I, 2I)
    accg = tl.zeros((BN,), tl.float32)
    accu = tl.zeros((BN,), tl.float32)
    for k0 in range(0, H, BK):
        ok = k0 + tl.arange(0, BK)
        mk = ok < H
        xv = tl.load(x_ptr + ok, mask=mk, other=0.0).to(tl.float32)
        wg = tl.load(gu_ptr + gate_row + ok[None, :], mask=mr[:, None] & mk[None, :], other=0.0).to(tl.float32)
        wu = tl.load(gu_ptr + up_row + ok[None, :], mask=mr[:, None] & mk[None, :], other=0.0).to(tl.float32)
        accg += tl.sum(wg * xv[None, :], axis=1)
        accu += tl.sum(wu * xv[None, :], axis=1)
    silu = accg * (1.0 / (1.0 + tl.exp(-accg)))
    tl.store(act_ptr + j * sa_j + offs_r, silu * accu, mask=mr)


def expert_swiglu(x, gate_up, topi, top_k, inter, act=None):
    E, twoI, H = gate_up.shape
    I = inter
    if act is None:
        act = torch.empty(top_k, I, device=x.device, dtype=torch.float32)
    grid = lambda m: (top_k, triton.cdiv(I, m["BN"]))
    _expert_swiglu[grid](x, gate_up, topi, act, H, I,
                         gate_up.stride(0), gate_up.stride(1), act.stride(0))
    return act


# ---- expert down-proj + gated combine + residual ---------------------------
# grid (tiles of H): each program sums, over the TOP_K experts, gate[j] * (down[e]
# @ act[j]) for its tile of output rows, gathering down_proj[e] indirectly, then
# adds the residual. One kernel produces the full MoE block output.
@triton.autotune(configs=_gemv_configs(), key=["H", "I"])
@triton.jit
def _expert_down(act_ptr, dn_ptr, topi_ptr, topv_ptr, res_ptr, o_ptr, H, I, sa_j, se, sr,
                 TOP_K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    offs_n = tile * BN + tl.arange(0, BN)
    mn = offs_n < H
    acc = tl.zeros((BN,), tl.float32)
    for j in range(TOP_K):
        e = tl.load(topi_ptr + j)
        wgt = tl.load(topv_ptr + j).to(tl.float32)
        row = e * se + offs_n[:, None] * sr
        partial = tl.zeros((BN,), tl.float32)
        for k0 in range(0, I, BK):
            ok = k0 + tl.arange(0, BK)
            mk = ok < I
            av = tl.load(act_ptr + j * sa_j + ok, mask=mk, other=0.0).to(tl.float32)
            wd = tl.load(dn_ptr + row + ok[None, :], mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
            partial += tl.sum(wd * av[None, :], axis=1)
        acc += wgt * partial
    acc += tl.load(res_ptr + offs_n, mask=mn, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs_n, acc.to(tl.bfloat16), mask=mn)


def expert_down(act, down, topi, topv, residual, top_k):
    E, H, I = down.shape
    out = torch.empty(H, device=act.device, dtype=torch.bfloat16)
    grid = lambda m: (triton.cdiv(H, m["BN"]),)
    _expert_down[grid](act, down, topi, topv, residual, out, H, I,
                       act.stride(0), down.stride(0), down.stride(1), top_k)
    return out
