"""Triton kernels for batch-1 decode.

At batch 1 every projection is a GEMV (matrix-vector), and the whole step is
memory-bound on the weights: each weight matrix must be read from HBM exactly
once, and that read *is* the latency floor. So these kernels do two things —
read each weight once at high bandwidth, and fuse the cheap glue (RMSNorm,
bias, SwiGLU, the residual add) into the same launch so activations never make
a separate HBM round-trip. Accumulate in fp32, store the residual stream in
bf16 to match the reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Batch-1 GEMV is weight-streaming: fill the SMs with many small programs (low
# BN) and read K in wide chunks (high BK). Autotune picks per (N,K); the search
# runs during graph warmup, so capture only ever replays the winning config.
def _gemv_configs():
    return [triton.Config({"BN": bn, "BK": bk}, num_warps=w)
            for bn in (4, 8, 16, 32) for bk in (256, 512, 1024, 2048) for w in (2, 4, 8)]


# ---- fused RMSNorm + GEMV (+ optional bias) --------------------------------
@triton.autotune(configs=_gemv_configs(), key=["N", "K"])
@triton.jit
def _rmsnorm_gemv(h_ptr, g_ptr, w_ptr, b_ptr, o_ptr, K, N, eps,
                  sw_n, sw_k, HAS_BIAS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    # pass 1: RMSNorm scale from the full input vector (cheap; H is small)
    ss = tl.zeros((1,), tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        hv = tl.load(h_ptr + ok, mask=ok < K, other=0.0).to(tl.float32)
        ss += tl.sum(hv * hv, axis=0)
    scale = 1.0 / tl.sqrt(ss / K + eps)
    # pass 2: GEMV against x_hat = h * scale * g
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        mk = ok < K
        hv = tl.load(h_ptr + ok, mask=mk, other=0.0).to(tl.float32)
        gv = tl.load(g_ptr + ok, mask=mk, other=0.0).to(tl.float32)
        xh = hv * scale * gv
        w = tl.load(w_ptr + offs_n[:, None] * sw_n + ok[None, :] * sw_k,
                    mask=(offs_n < N)[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * xh[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs_n, acc.to(tl.bfloat16), mask=offs_n < N)


def rmsnorm_gemv(h, g, W, bias, eps):
    N, K = W.shape
    out = torch.empty(N, device=h.device, dtype=torch.bfloat16)
    grid = lambda m: (triton.cdiv(N, m["BN"]),)
    _rmsnorm_gemv[grid](h, g, W, bias if bias is not None else h, out, K, N, eps,
                        W.stride(0), W.stride(1), bias is not None)
    return out


# ---- plain GEMV (+ optional bias, + optional residual) ---------------------
@triton.autotune(configs=_gemv_configs(), key=["N", "K"])
@triton.jit
def _gemv(x_ptr, w_ptr, b_ptr, r_ptr, o_ptr, K, N, sw_n, sw_k,
          HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    if HAS_RES:
        acc += tl.load(r_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs_n, acc.to(tl.bfloat16), mask=offs_n < N)


def gemv(x, W, bias=None, residual=None):
    N, K = W.shape
    out = torch.empty(N, device=x.device, dtype=torch.bfloat16)
    grid = lambda m: (triton.cdiv(N, m["BN"]),)
    _gemv[grid](x, W, bias if bias is not None else x, residual if residual is not None else x,
                out, K, N, W.stride(0), W.stride(1), bias is not None, residual is not None)
    return out


# ---- fused RMSNorm + gate/up GEMV + SwiGLU ---------------------------------
@triton.autotune(configs=_gemv_configs(), key=["I", "K"])
@triton.jit
def _swiglu_gemv(h_ptr, g_ptr, wg_ptr, wu_ptr, o_ptr, K, I, eps,
                 sg_n, sg_k, su_n, su_k, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    ss = tl.zeros((1,), tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        hv = tl.load(h_ptr + ok, mask=ok < K, other=0.0).to(tl.float32)
        ss += tl.sum(hv * hv, axis=0)
    scale = 1.0 / tl.sqrt(ss / K + eps)
    gate = tl.zeros((BN,), tl.float32)
    up = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        mk = ok < K
        hv = tl.load(h_ptr + ok, mask=mk, other=0.0).to(tl.float32)
        gv = tl.load(g_ptr + ok, mask=mk, other=0.0).to(tl.float32)
        xh = hv * scale * gv
        wg = tl.load(wg_ptr + offs_n[:, None] * sg_n + ok[None, :] * sg_k,
                     mask=(offs_n < I)[:, None] & mk[None, :], other=0.0).to(tl.float32)
        wu = tl.load(wu_ptr + offs_n[:, None] * su_n + ok[None, :] * su_k,
                     mask=(offs_n < I)[:, None] & mk[None, :], other=0.0).to(tl.float32)
        gate += tl.sum(wg * xh[None, :], axis=1)
        up += tl.sum(wu * xh[None, :], axis=1)
    silu = gate * (1.0 / (1.0 + tl.exp(-gate)))
    tl.store(o_ptr + offs_n, (silu * up).to(tl.bfloat16), mask=offs_n < I)


def swiglu_gemv(h, g, Wg, Wu, eps):
    I, K = Wg.shape
    out = torch.empty(I, device=h.device, dtype=torch.bfloat16)
    grid = lambda m: (triton.cdiv(I, m["BN"]),)
    _swiglu_gemv[grid](h, g, Wg, Wu, out, K, I, eps,
                       Wg.stride(0), Wg.stride(1), Wu.stride(0), Wu.stride(1))
    return out


# ---- fused RoPE + KV-cache write -------------------------------------------
# Replaces ~14 tiny torch ops/layer (rotate_half's split/neg/cat, the cos/sin
# mul-add, two index_copy_ cache writes) with one launch. One program per head:
# q heads write roped q to a scratch buffer for attention; kv heads write roped
# k and raw v straight into this layer's cache at `pos`. rotate_half is done with
# a gathered "other half" index so no cat is needed.
@triton.jit
def _rope_write(qkv_ptr, cos_ptr, sin_ptr, qo_ptr, kc_ptr, vc_ptr, pos_ptr,
                qd, kd, sk_t, sk_h, NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.arange(0, HD)
    half = HD // 2
    cos = tl.load(cos_ptr + d).to(tl.float32)
    sin = tl.load(sin_ptr + d).to(tl.float32)
    sgn = tl.where(d < half, -1.0, 1.0)                 # rotate_half sign
    roti = tl.where(d < half, d + half, d - half)       # rotate_half gather index
    if pid < NH:
        base = pid * HD
        x = tl.load(qkv_ptr + base + d).to(tl.float32)
        xr = tl.load(qkv_ptr + base + roti).to(tl.float32)
        tl.store(qo_ptr + base + d, (x * cos + sgn * xr * sin).to(tl.bfloat16))
    else:
        kk = pid - NH
        pos = tl.load(pos_ptr)
        kbase = qd + kk * HD
        x = tl.load(qkv_ptr + kbase + d).to(tl.float32)
        xr = tl.load(qkv_ptr + kbase + roti).to(tl.float32)
        dst = pos * sk_t + kk * sk_h + d
        tl.store(kc_ptr + dst, (x * cos + sgn * xr * sin).to(tl.bfloat16))
        v = tl.load(qkv_ptr + qd + kd + kk * HD + d)
        tl.store(vc_ptr + dst, v)


def rope_write(qkv, cos, sin, kc_layer, vc_layer, pos, nh, nkv, hd):
    """Roped q [nh,hd] returned; roped k and raw v written into the cache at pos."""
    qo = torch.empty(nh, hd, device=qkv.device, dtype=torch.bfloat16)
    _rope_write[(nh + nkv,)](qkv, cos, sin, qo, kc_layer, vc_layer, pos,
                             nh * hd, nkv * hd, kc_layer.stride(0), kc_layer.stride(1),
                             nh, nkv, hd)
    return qo


# ---- batch-1 GQA decode attention: split-K flash-decode ---------------------
# One program per head walking the whole KV timeline uses only n_heads (=12) SMs
# and serializes 600+ steps -- occupancy-bound, ~24 GB/s. Instead split the
# timeline into N_SPLIT chunks: grid (nh, N_SPLIT) fills the SMs, each program
# does a local online-softmax over its chunk, then a cheap combine kernel merges
# the partials with the standard exp(m_local - m_global) rescale. A chunk fully
# past the live length contributes m=-inf and is zeroed by the combine weight.
# T_MAX / CHUNK / N_SPLIT are all constexpr, so both launches stay graph-stable.
@triton.jit
def _attn_split(q_ptr, k_ptr, v_ptr, pm_ptr, pl_ptr, pa_ptr, len_ptr, GROUP, scale,
                sk_t, sk_h, sq_h, spm_h, spa_h, spa_s,
                HD: tl.constexpr, BT: tl.constexpr, CHUNK: tl.constexpr, N_SPLIT: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    kvh = h // GROUP
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + h * sq_h + d).to(tl.float32)
    L = tl.load(len_ptr)
    m_i = -1e30
    l_i = 0.0
    acc = tl.zeros((HD,), tl.float32)
    hi = s * CHUNK + CHUNK
    for t0 in range(s * CHUNK, hi, BT):
        ot = t0 + tl.arange(0, BT)
        mt = (ot < L) & (ot < hi)     # stay inside this split's chunk
        k = tl.load(k_ptr + ot[:, None] * sk_t + kvh * sk_h + d[None, :],
                    mask=mt[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(k * q[None, :], axis=1) * scale
        sc = tl.where(mt, sc, -1e30)
        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        p = tl.exp(sc - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(v_ptr + ot[:, None] * sk_t + kvh * sk_h + d[None, :],
                    mask=mt[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(pm_ptr + h * spm_h + s, m_i)
    tl.store(pl_ptr + h * spm_h + s, l_i)
    tl.store(pa_ptr + h * spa_h + s * spa_s + d, acc)


@triton.jit
def _attn_combine(pm_ptr, pl_ptr, pa_ptr, o_ptr, spm_h, spa_h, spa_s, so_h,
                  HD: tl.constexpr, N_SPLIT: tl.constexpr):
    h = tl.program_id(0)
    d = tl.arange(0, HD)
    ss = tl.arange(0, N_SPLIT)
    mvals = tl.load(pm_ptr + h * spm_h + ss)
    lvals = tl.load(pl_ptr + h * spm_h + ss)
    g = tl.max(mvals, axis=0)
    w = tl.exp(mvals - g)                                    # [N_SPLIT]
    denom = tl.sum(w * lvals, axis=0)
    a = tl.load(pa_ptr + h * spa_h + ss[:, None] * spa_s + d[None, :])  # [N_SPLIT, HD]
    out = tl.sum(w[:, None] * a, axis=0) / denom
    tl.store(o_ptr + h * so_h + d, out.to(tl.bfloat16))


def decode_attn(q, Kc, Vc, group, scale, cur_len=None, n_split=8, BT=32):
    """q [nh, hd]; Kc/Vc [T_MAX, nkv, hd]. cur_len: int32[1] GPU tensor of the
    live sequence length (defaults to full T_MAX for the incremental path)."""
    nh, hd = q.shape
    T = Kc.shape[0]
    if cur_len is None:
        cur_len = torch.full((1,), T, device=q.device, dtype=torch.int32)
    chunk = triton.cdiv(T, n_split)
    pm = torch.empty(nh, n_split, device=q.device, dtype=torch.float32)
    pl = torch.empty(nh, n_split, device=q.device, dtype=torch.float32)
    pa = torch.empty(nh, n_split, hd, device=q.device, dtype=torch.float32)
    out = torch.empty(nh, hd, device=q.device, dtype=torch.bfloat16)
    _attn_split[(nh, n_split)](q, Kc, Vc, pm, pl, pa, cur_len, group, scale,
                               Kc.stride(0), Kc.stride(1), q.stride(0),
                               pm.stride(0), pa.stride(0), pa.stride(1), hd, BT, chunk, n_split)
    _attn_combine[(nh,)](pm, pl, pa, out, pm.stride(0), pa.stride(0), pa.stride(1),
                         out.stride(0), hd, n_split)
    return out
