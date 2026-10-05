#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle
"""Pure-NumPy fp32 oracle for the tiny Qwen3.6 (Qwen3_5 text) gate configuration.

Reference anchor for all cuda_qwen36_* Lean gates per docs/QWEN36_MEGAKERNEL.md.
Implements the exact equations of the vendored HF excerpt
(modeling_qwen3_5_excerpt.py): hybrid Gated DeltaNet + gated full attention,
pre-norm decoder, (1+w) RMSNorm, zero-init norms, unit-init gated norm.

CLI:
  oracle.py dump OUTDIR        write fixtures (<name>.<dims>.f32.bin) + manifest.json
  oracle.py selftest           internal consistency gates (no fixtures needed):
                               chunked vs recurrent delta rule parity, decode-step
                               parity, and central-finite-difference checks of the
                               hand-written backward (eps 1e-3, tol 1e-3)
  oracle.py compare DIRA DIRB  byte-compare two fixture trees (regeneration gate)

No torch dependency. Backward is hand-derived reverse mode over the same
equations (see README.md); it is cross-checked against central finite
differences in `selftest`.
"""

import hashlib
import json
import os
import sys

import numpy as np

SEED = 0xC0FFEE
INIT_STD = 0.02  # HF-style initializer_range for all matrices/embeddings

# Tiny gate configuration (docs/QWEN36_MEGAKERNEL.md).
CFG = dict(
    hidden_size=256,
    layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention"),
    vocab_size=512,
    intermediate_size=512,
    linear_num_key_heads=2,
    linear_key_head_dim=64,
    linear_num_value_heads=4,
    linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    chunk_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    partial_rotary_factor=0.25,
    rope_theta=1e7,
    rms_norm_eps=1e-6,
    l2norm_eps=1e-6,
    seq_len=8,
)

# ---------------------------------------------------------------------------
# Init (single rng, fixed documented draw order; see README.md)
# ---------------------------------------------------------------------------


def init_params_and_inputs(cfg=CFG, seed=SEED, dtype=np.float32):
    rng = np.random.default_rng(seed)

    def normal(shape):
        return (rng.standard_normal(shape, dtype=np.float32) * np.float32(INIT_STD)).astype(dtype)

    H, V, I = cfg["hidden_size"], cfg["vocab_size"], cfg["intermediate_size"]
    nk, dk = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"]
    nv, dv = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
    kd, vd = nk * dk, nv * dv
    conv_dim = 2 * kd + vd
    nq, nkv, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]

    P = {}
    P["model.embed_tokens.weight"] = normal((V, H))
    for i, lt in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        P[p + "input_layernorm.weight"] = np.zeros(H, dtype)
        if lt == "linear_attention":
            P[p + "linear_attn.in_proj_qkv.weight"] = normal((conv_dim, H))
            P[p + "linear_attn.in_proj_z.weight"] = normal((vd, H))
            P[p + "linear_attn.in_proj_b.weight"] = normal((nv, H))
            P[p + "linear_attn.in_proj_a.weight"] = normal((nv, H))
            P[p + "linear_attn.conv1d.weight"] = normal((conv_dim, cfg["linear_conv_kernel_dim"]))
            P[p + "linear_attn.A_log"] = np.log(rng.uniform(0.0, 16.0, size=(nv,)).astype(np.float32)).astype(dtype)
            P[p + "linear_attn.dt_bias"] = np.ones(nv, dtype)
            P[p + "linear_attn.norm.weight"] = np.ones(dv, dtype)
            P[p + "linear_attn.out_proj.weight"] = normal((H, vd))
        else:
            P[p + "self_attn.q_proj.weight"] = normal((nq * dh * 2, H))
            P[p + "self_attn.k_proj.weight"] = normal((nkv * dh, H))
            P[p + "self_attn.v_proj.weight"] = normal((nkv * dh, H))
            P[p + "self_attn.o_proj.weight"] = normal((H, nq * dh))
            P[p + "self_attn.q_norm.weight"] = np.zeros(dh, dtype)
            P[p + "self_attn.k_norm.weight"] = np.zeros(dh, dtype)
        P[p + "post_attention_layernorm.weight"] = np.zeros(H, dtype)
        P[p + "mlp.gate_proj.weight"] = normal((I, H))
        P[p + "mlp.up_proj.weight"] = normal((I, H))
        P[p + "mlp.down_proj.weight"] = normal((H, I))
    P["model.norm.weight"] = np.zeros(H, dtype)
    P["lm_head.weight"] = normal((V, H))

    T = cfg["seq_len"]
    inputs = {
        "tokens": rng.integers(0, V, size=T),
        "targets": rng.integers(0, V, size=T),
        "probe_hidden": rng.standard_normal((T, H), dtype=np.float32).astype(dtype),
        "decode_token": rng.integers(0, V, size=1),
    }
    return P, inputs


# ---------------------------------------------------------------------------
# Primitives (forward returns (out, ctx); backward takes (dout, ctx))
# ---------------------------------------------------------------------------


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x):
    return x * sigmoid(x)


def dsilu(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def softplus(x):
    return np.logaddexp(x, np.zeros((), x.dtype))


def linear(x, w):
    # y = x @ w^T, w stored (out, in) HF-style
    return x @ w.T, (x, w)


def linear_bwd(dy, ctx):
    x, w = ctx
    return dy @ w, dy.reshape(-1, dy.shape[-1]).T @ x.reshape(-1, x.shape[-1])


def rms_norm_w(x, w, eps):
    # Qwen3_5RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w)
    s = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x * s * (1.0 + w), (x, s, w, eps)


def rms_norm_w_bwd(dy, ctx):
    x, s, w, eps = ctx
    n = x.shape[-1]
    dz = dy * (1.0 + w)
    dx = s * dz - x * (s**3) / n * (dz * x).sum(-1, keepdims=True)
    dw = (dy * (x * s)).reshape(-1, n).sum(0)
    return dx, dw


def gated_rms_norm(x, gate, w, eps):
    # Qwen3_5RMSNormGated: y = w * (x * rsqrt(mean(x^2)+eps)) * silu(gate); w unit-init
    s = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    z = silu(gate)
    return w * (x * s) * z, (x, s, gate, z, w, eps)


def gated_rms_norm_bwd(dy, ctx):
    x, s, gate, z, w, eps = ctx
    n = x.shape[-1]
    dz = dy * w * (x * s)
    dgate = dz * dsilu(gate)
    dw = (dy * (x * s) * z).reshape(-1, n).sum(0)
    dxn = dy * w * z
    dx = s * dxn - x * (s**3) / n * (dxn * x).sum(-1, keepdims=True)
    return dx, dgate, dw


def l2norm(x, eps):
    # FLA-aligned: y = x * rsqrt(sum(x^2) + eps)
    s = 1.0 / np.sqrt((x * x).sum(-1, keepdims=True) + eps)
    return x * s, (x, s, eps)


def l2norm_bwd(dy, ctx):
    x, s, eps = ctx
    return s * dy - x * (s**3) * (dy * x).sum(-1, keepdims=True)


def causal_conv1d_silu(x, w):
    # Depthwise causal conv (kernel K, left-pad K-1) + SiLU. x (T, C), w (C, K).
    K = w.shape[-1]
    xp = np.pad(x, ((K - 1, 0), (0, 0)))
    raw = np.zeros_like(x)
    for i in range(K):
        raw += w[:, i][None, :] * xp[i : i + x.shape[0]]
    return silu(raw), (x, raw, w)


def causal_conv1d_silu_bwd(dy, ctx):
    x, raw, w = ctx
    T, C = x.shape
    K = w.shape[-1]
    draw = dy * dsilu(raw)
    xp = np.pad(x, ((K - 1, 0), (0, 0)))
    dxp = np.zeros_like(xp)
    dw = np.zeros_like(w)
    for i in range(K):
        dxp[i : i + T] += w[:, i][None, :] * draw
        dw[:, i] = (draw * xp[i : i + T]).sum(0)
    return dxp[K - 1 :], dw


def causal_conv1d_update(conv_state, x_new, w):
    # Decode step: conv_state (K-1, C) raw pre-conv rows, x_new (C,) -> silu out, new state.
    win = np.concatenate([conv_state, x_new[None, :]], axis=0)
    out = silu((w.T * win).sum(0))
    return out, win[1:].copy()


def rope_tables(cfg, positions, dtype):
    rot = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    theta = np.float32(cfg["rope_theta"])
    inv_freq = np.float32(1.0) / (
        theta ** (np.arange(0, rot, 2, dtype=np.float32) / np.float32(rot))
    )
    freqs = np.outer(positions.astype(np.float32), inv_freq)  # (T, rot/2)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rotate_half_t(y):
    # adjoint of rotate_half (skew-symmetric permutation)
    h = y.shape[-1] // 2
    return np.concatenate([y[..., h:], -y[..., :h]], axis=-1)


def apply_rope(x, cos, sin):
    # x (T, h, dh); cos/sin (T, rot) broadcast over heads; partial rotary.
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    out = np.concatenate([xr * cos[:, None, :] + rotate_half(xr) * sin[:, None, :], xp], axis=-1)
    return out, rot


def apply_rope_bwd(dout, cos, sin, rot):
    dr = dout[..., :rot]
    dxr = dr * cos[:, None, :] + rotate_half_t(dr * sin[:, None, :])
    return np.concatenate([dxr, dout[..., rot:]], axis=-1)


# ---------------------------------------------------------------------------
# Gated delta rule (chunked prefill + recurrent decode), fp32 math
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule(q, k, v, g, beta, S0, chunk_size):
    """NumPy port of torch_chunk_gated_delta_rule (l2norm applied by caller).

    q, k: (nh, T, dk) post-l2norm; v: (nh, T, dv); g, beta: (nh, T);
    S0: (nh, dk, dv) or None. Returns (out (nh, T, dv), S1 (nh, dk, dv), cache).
    """
    nh, T, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    q = np.pad(q, ((0, 0), (0, pad), (0, 0)))
    k = np.pad(k, ((0, 0), (0, pad), (0, 0)))
    v = np.pad(v, ((0, 0), (0, pad), (0, 0)))
    beta = np.pad(beta, ((0, 0), (0, pad)))
    g = np.pad(g, ((0, 0), (0, pad)))
    # Keep the recurrence in the caller's requested dtype.  A NumPy Float64
    # scalar here promotes `q` (and therefore every cached recurrent state),
    # which makes the serialized Float32 stage fixtures internally
    # inconsistent with their backward caches.
    scale = np.asarray(1.0 / np.sqrt(dk), dtype=q.dtype)
    q = q * scale

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    nc = (T + pad) // chunk_size
    C = chunk_size
    q, k, v, k_beta, v_beta = [
        x.reshape(nh, nc, C, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)
    ]
    beta = beta.reshape(nh, nc, C)
    g = g.reshape(nh, nc, C)

    g = g.cumsum(-1)  # in-chunk cumulative log-decay
    diff = g[..., :, None] - g[..., None, :]
    decay = np.tril(np.exp(np.tril(diff)))  # exp(g_i - g_j) for j <= i else 0
    strict_lower = np.tril(np.ones((C, C), bool), -1)

    attn = -((k_beta @ k.swapaxes(-1, -2)) * decay)
    attn = attn * strict_lower  # masked_fill(triu incl. diagonal, 0)
    # UT transform: forward substitution -> attn + I = (I - A)^{-1}
    for i in range(1, C):
        row = attn[..., i, :i].copy()
        sub = attn[..., :i, :i].copy()
        attn[..., i, :i] = row + (row[..., :, None] * sub).sum(-2)
    M_inv = attn + np.eye(C, dtype=q.dtype)
    value = M_inv @ v_beta
    P = k_beta * np.exp(g)[..., None]
    k_cum = M_inv @ P

    S = np.zeros((nh, dk, dv), q.dtype) if S0 is None else S0.copy()
    out = np.zeros_like(value)
    S_in_per_chunk, att_per_chunk, v_new_per_chunk = [], [], []
    for c in range(nc):
        S_in_per_chunk.append(S)
        att = q[:, c] @ k[:, c].swapaxes(-1, -2) * decay[:, c]
        v_new = value[:, c] - k_cum[:, c] @ S
        out[:, c] = (q[:, c] * np.exp(g[:, c])[..., None]) @ S + att @ v_new
        S = S * np.exp(g[:, c, -1])[:, None, None] + (
            (k[:, c] * np.exp(g[:, c, -1, None] - g[:, c])[..., None]).swapaxes(-1, -2) @ v_new
        )
        att_per_chunk.append(att)
        v_new_per_chunk.append(v_new)

    cache = dict(
        q=q, k=k, v=v, beta=beta, v_beta=v_beta, k_beta=k_beta, P=P, g=g, decay=decay,
        M_inv=M_inv, k_cum=k_cum, att=att_per_chunk, v_new=v_new_per_chunk,
        S_in=S_in_per_chunk, T=T, nc=nc, C=C, scale=scale,
    )
    out = out.reshape(nh, T + pad, dv)[:, :T]
    return out, S, cache


def chunk_gated_delta_rule_bwd(dout, dS1, cache):
    """Hand-derived adjoint of chunk_gated_delta_rule (same equations reversed).

    dout: (nh, T, dv); dS1: (nh, dk, dv). Returns dq, dk, dv, dg, dbeta, dS0 where
    dq/dk are w.r.t. the post-l2norm pre-scale inputs and dg is w.r.t. the raw
    (pre-cumsum) log-decay, all unpadded to (nh, T, ...) / (nh, T).
    """
    q, k, v, beta = cache["q"], cache["k"], cache["v"], cache["beta"]
    v_beta, k_beta, P = cache["v_beta"], cache["k_beta"], cache["P"]
    g, decay, M_inv, k_cum = cache["g"], cache["decay"], cache["M_inv"], cache["k_cum"]
    T, nc, C, scale = cache["T"], cache["nc"], cache["C"], cache["scale"]
    nh, dk = q.shape[0], q.shape[-1]
    dv = dout.shape[-1]

    dout_c = np.zeros((nh, nc, C, dv), dout.dtype)
    dout_c.reshape(nh, nc * C, dv)[:, :T] = dout

    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dg = np.zeros_like(g)  # w.r.t. in-chunk cumulative g
    ddecay = np.zeros_like(decay)
    dM_inv = np.zeros_like(M_inv)
    dv_beta = np.zeros_like(v_beta)
    dk_beta = np.zeros_like(k_beta)

    dS = dS1.copy()
    for c in range(nc - 1, -1, -1):
        S_in = cache["S_in"][c]
        att, v_new = cache["att"][c], cache["v_new"][c]
        q_c, k_c, g_c = q[:, c], k[:, c], g[:, c]
        eg = np.exp(g_c)
        eg_last = np.exp(g_c[:, -1])
        do = dout_c[:, c]

        # S_out = S_in * e^{g_last} + (k * e^{g_last - g})^T @ v_new
        dS_in = dS * eg_last[:, None, None]
        dg_last = (dS * S_in).sum((-1, -2)) * eg_last
        kd = k_c * np.exp(g_c[:, -1, None] - g_c)[..., None]
        dv_new = kd @ dS
        dkd = v_new @ dS.swapaxes(-1, -2)
        dgf = (dkd * kd).sum(-1)  # d/d(g_last - g_i) per position
        dg[:, c] -= dgf
        dg[:, c, -1] += dg_last + dgf.sum(-1)
        dk[:, c] += dkd * np.exp(g_c[:, -1, None] - g_c)[..., None]

        # out = (q e^g) @ S_in + att @ v_new
        qe = q_c * eg[..., None]
        dS_in = dS_in + qe.swapaxes(-1, -2) @ do
        dqe = do @ S_in.swapaxes(-1, -2)
        dg[:, c] += (dqe * qe).sum(-1)
        dq[:, c] += dqe * eg[..., None]
        datt = do @ v_new.swapaxes(-1, -2)
        dv_new = dv_new + att.swapaxes(-1, -2) @ do

        # att = (q k^T) * decay
        dA_raw = datt * decay[:, c]
        ddecay[:, c] += datt * (q_c @ k_c.swapaxes(-1, -2))
        dq[:, c] += dA_raw @ k_c
        dk[:, c] += dA_raw.swapaxes(-1, -2) @ q_c

        # v_new = value - k_cum @ S_in ; value = M_inv @ v_beta ; k_cum = M_inv @ P
        dM_inv[:, c] += dv_new @ v_beta[:, c].swapaxes(-1, -2)
        dv_beta[:, c] += M_inv[:, c].swapaxes(-1, -2) @ dv_new
        dk_cum = -(dv_new @ S_in.swapaxes(-1, -2))
        dM_inv[:, c] += dk_cum @ P[:, c].swapaxes(-1, -2)
        dP_c = M_inv[:, c].swapaxes(-1, -2) @ dk_cum
        dS_in = dS_in - k_cum[:, c].swapaxes(-1, -2) @ dv_new

        # P = k_beta * e^g (uses cumulative g)
        dk_beta[:, c] += dP_c * eg[..., None]
        dg[:, c] += (dP_c * P[:, c]).sum(-1)

        dS = dS_in

    dS0 = dS

    # M_inv = (I - A)^{-1}, A strictly lower: dA = tril(M_inv^T dM_inv M_inv^T, -1)
    dA = np.tril(M_inv.swapaxes(-1, -2) @ dM_inv @ M_inv.swapaxes(-1, -2), -1)
    # A[i,j] = -(k_beta_i . k_j) decay[i,j]
    dk_beta = dk_beta - (dA * decay) @ k
    dk = dk - (dA * decay).swapaxes(-1, -2) @ k_beta
    ddecay = ddecay - dA * (k_beta @ k.swapaxes(-1, -2))

    # decay[i,j] = e^{g_i - g_j} (j <= i); decay == 0 above the diagonal
    D = ddecay * decay
    dg = dg + D.sum(-1) - D.sum(-2)

    # k_beta = k * beta ; v_beta = v * beta
    dk = dk + dk_beta * beta[..., None]
    dv = dv_beta * beta[..., None]
    dbeta = (dk_beta * k).sum(-1) + (dv_beta * v).sum(-1)

    # adjoint of the in-chunk cumsum, then unpad / undo the q scale
    dg_raw = dg[..., ::-1].cumsum(-1)[..., ::-1]

    def unpad(x):
        return x.reshape(x.shape[0], nc * C, *x.shape[3:])[:, :T]

    return unpad(dq) * scale, unpad(dk), unpad(dv), unpad(dg_raw), unpad(dbeta), dS0


def recurrent_gated_delta_rule(q, k, v, g, beta, S0):
    """NumPy port of torch_recurrent_gated_delta_rule (l2norm applied by caller).

    q, k: (nh, T, dk) post-l2norm; v: (nh, T, dv); g, beta: (nh, T);
    S0: (nh, dk, dv). Returns (out (nh, T, dv), S1).
    """
    nh, T, dk = k.shape
    dv = v.shape[-1]
    scale = np.asarray(1.0 / np.sqrt(dk), dtype=q.dtype)
    q = q * scale
    S = S0.copy()
    out = np.zeros((nh, T, dv), q.dtype)
    for t in range(T):
        S = S * np.exp(g[:, t])[:, None, None]
        kv = (S * k[:, t, :, None]).sum(1)
        delta = (v[:, t] - kv) * beta[:, t, None]
        S = S + k[:, t, :, None] * delta[:, None, :]
        out[:, t] = (S * q[:, t, :, None]).sum(1)
    return out, S


# ---------------------------------------------------------------------------
# Stage-level forward (returns flat dump dict + backward cache)
# ---------------------------------------------------------------------------


def mlp_forward(P, p, h, eps):
    n2, n2ctx = rms_norm_w(h, P[p + "post_attention_layernorm.weight"], eps)
    g_, ctxg = linear(n2, P[p + "mlp.gate_proj.weight"])
    u, ctxu = linear(n2, P[p + "mlp.up_proj.weight"])
    act = silu(g_) * u
    out, ctxd = linear(act, P[p + "mlp.down_proj.weight"])
    return out, dict(
        n2=n2, n2ctx=n2ctx, ctxg=ctxg, ctxu=ctxu, g_=g_, u=u, act=act, ctxd=ctxd
    )


def mlp_bwd(dy, lc):
    dact, dWd = linear_bwd(dy, lc["ctxd"])
    dg_ = dact * lc["u"] * dsilu(lc["g_"])
    du = dact * silu(lc["g_"])
    dn2, dWg = linear_bwd(dg_, lc["ctxg"])
    dn2u, dWu = linear_bwd(du, lc["ctxu"])
    dx, dwn = rms_norm_w_bwd(dn2 + dn2u, lc["n2ctx"])
    return (
        dx,
        {"gate_proj.weight": dWg, "up_proj.weight": dWu, "down_proj.weight": dWd,
         "post_attention_layernorm.weight": dwn},
        {"output": dy, "act": dact, "gate": dg_, "up": du, "norm2": dn2 + dn2u,
         "input": dx},
    )


def deltanet_forward(cfg, P, i, x, D=None, prefix=None):
    """Gated DeltaNet layer on x (T, H). Returns (mixer_out (T, H), cache)."""
    eps = cfg["rms_norm_eps"]
    nk, dk = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"]
    nv, dv = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
    kd, vd = nk * dk, nv * dv
    T = x.shape[0]
    lp = f"model.layers.{i}."
    p = lp + "linear_attn."

    n1, n1ctx = rms_norm_w(x, P[lp + "input_layernorm.weight"], eps)
    qkv, ctx_qkv = linear(n1, P[p + "in_proj_qkv.weight"])
    z, ctx_z = linear(n1, P[p + "in_proj_z.weight"])
    b, ctx_b = linear(n1, P[p + "in_proj_b.weight"])
    a, ctx_a = linear(n1, P[p + "in_proj_a.weight"])
    qkv_conv, convctx = causal_conv1d_silu(qkv, P[p + "conv1d.weight"])

    q = qkv_conv[:, :kd].reshape(T, nk, dk)
    k_ = qkv_conv[:, kd : 2 * kd].reshape(T, nk, dk)
    vh = qkv_conv[:, 2 * kd :].reshape(T, nv, dv)
    beta = sigmoid(b)
    g = -np.exp(P[p + "A_log"]) * softplus(a + P[p + "dt_bias"][None, :])
    rep = nv // nk
    qr = np.repeat(q, rep, axis=1)  # repeat_interleave
    kr = np.repeat(k_, rep, axis=1)
    ql, qlctx = l2norm(qr, cfg["l2norm_eps"])
    kl, klctx = l2norm(kr, cfg["l2norm_eps"])

    out, S, gdr_cache = chunk_gated_delta_rule(
        ql.transpose(1, 0, 2), kl.transpose(1, 0, 2), vh.transpose(1, 0, 2),
        g.T.copy(), beta.T.copy(), None, cfg["chunk_size"],
    )
    delta_out = out.transpose(1, 0, 2)  # (T, nv, dv)
    gated, gctx = gated_rms_norm(
        delta_out.reshape(-1, dv), z.reshape(-1, dv), P[p + "norm.weight"], eps
    )
    gated_flat = gated.reshape(T, vd)
    mixer, ctx_out = linear(gated_flat, P[p + "out_proj.weight"])

    if D is not None:
        D[prefix + "qkv_pre_conv"] = qkv
        D[prefix + "q_post_conv"] = qkv_conv[:, :kd]
        D[prefix + "k_post_conv"] = qkv_conv[:, kd : 2 * kd]
        D[prefix + "v_post_conv"] = qkv_conv[:, 2 * kd :]
        D[prefix + "beta"] = beta
        D[prefix + "g"] = g
        D[prefix + "q_l2"] = ql
        D[prefix + "k_l2"] = kl
        D[prefix + "v_heads"] = vh
        D[prefix + "delta_out"] = delta_out
        D[prefix + "S_final"] = S
        D[prefix + "gated_out"] = gated_flat
        D[prefix + "mixer_out"] = mixer
    cache = dict(
        n1ctx=n1ctx, ctx_qkv=ctx_qkv, ctx_z=ctx_z, ctx_b=ctx_b, ctx_a=ctx_a,
        convctx=convctx, qlctx=qlctx, klctx=klctx, gdr_cache=gdr_cache, gctx=gctx,
        ctx_out=ctx_out, beta=beta, a=a, ql=ql, kl=kl, vh=vh, g=g, P=p, lp=lp,
    )
    return mixer, cache


def deltanet_bwd(cfg, P, i, dmixer, lc):
    nk, dk = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"]
    nv, dv = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
    kd, vd = nk * dk, nv * dv
    T = dmixer.shape[0]
    p, lp = lc["P"], lc["lp"]
    grads = {}

    dgated_flat, dWout = linear_bwd(dmixer, lc["ctx_out"])
    grads[p + "out_proj.weight"] = dWout
    ddelta, dz, dgnw = gated_rms_norm_bwd(dgated_flat.reshape(-1, dv), lc["gctx"])
    grads[p + "norm.weight"] = dgnw

    dql, dkl, dvh, dg, dbeta, _ = chunk_gated_delta_rule_bwd(
        ddelta.reshape(T, nv, dv).transpose(1, 0, 2),
        np.zeros((nv, dk, dv), dmixer.dtype),
        lc["gdr_cache"],
    )
    dqr = l2norm_bwd(dql.transpose(1, 0, 2), lc["qlctx"])
    dkr = l2norm_bwd(dkl.transpose(1, 0, 2), lc["klctx"])
    rep = nv // nk
    dqh = dqr.reshape(T, nk, rep, dk).sum(2)  # repeat_interleave adjoint
    dkh = dkr.reshape(T, nk, rep, dk).sum(2)
    dqkv_conv = np.concatenate(
        [dqh.reshape(T, kd), dkh.reshape(T, kd), dvh.transpose(1, 0, 2).reshape(T, vd)], axis=-1
    )
    dqkv, dWconv = causal_conv1d_silu_bwd(dqkv_conv, lc["convctx"])
    grads[p + "conv1d.weight"] = dWconv

    beta = lc["beta"]
    db = dbeta.T * beta * (1.0 - beta)  # sigmoid adjoint
    da = dg.T * (-np.exp(P[p + "A_log"])) * sigmoid(lc["a"] + P[p + "dt_bias"][None, :])
    grads[p + "A_log"] = (dg.T * lc["g"]).sum(0)
    grads[p + "dt_bias"] = da.sum(0)

    dn1, dWqkv = linear_bwd(dqkv, lc["ctx_qkv"])
    grads[p + "in_proj_qkv.weight"] = dWqkv
    dxz, grads[p + "in_proj_z.weight"] = linear_bwd(dz.reshape(T, vd), lc["ctx_z"])
    dn1 = dn1 + dxz
    dxb, grads[p + "in_proj_b.weight"] = linear_bwd(db, lc["ctx_b"])
    dn1 = dn1 + dxb
    dxa, grads[p + "in_proj_a.weight"] = linear_bwd(da, lc["ctx_a"])
    dn1 = dn1 + dxa
    dx, dwn1 = rms_norm_w_bwd(dn1, lc["n1ctx"])
    grads[lp + "input_layernorm.weight"] = dwn1
    debug = {
        "output": dmixer,
        "gated_out": dgated_flat,
        "delta_out": ddelta.reshape(T, nv, dv),
        "z": dz.reshape(T, vd),
        "q_l2_repeated": dql.transpose(1, 0, 2),
        "k_l2_repeated": dkl.transpose(1, 0, 2),
        "v_heads": dvh.transpose(1, 0, 2),
        "g": dg.T,
        "beta": dbeta.T,
        "q_repeated": dqr,
        "k_repeated": dkr,
        "q_post_conv": dqh,
        "k_post_conv": dkh,
        "qkv_post_conv": dqkv_conv,
        "qkv_pre_conv": dqkv,
        "b": db,
        "a": da,
        "norm1": dn1,
        "input": dx,
    }
    return dx, grads, debug


def attention_forward(cfg, P, i, x, cos, sin, D=None, prefix=None):
    """Gated full-attention layer on x (T, H). Returns (mixer_out (T, H), cache)."""
    eps = cfg["rms_norm_eps"]
    nq, nkv, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    T = x.shape[0]
    lp = f"model.layers.{i}."
    p = lp + "self_attn."

    n1, n1ctx = rms_norm_w(x, P[lp + "input_layernorm.weight"], eps)
    qg, ctxq = linear(n1, P[p + "q_proj.weight"])
    qg_h = qg.reshape(T, nq, 2 * dh)
    q = qg_h[..., :dh]
    gate = qg_h[..., dh:].reshape(T, nq * dh)
    k_, ctxk = linear(n1, P[p + "k_proj.weight"])
    v_, ctxv = linear(n1, P[p + "v_proj.weight"])
    qn, qnctx = rms_norm_w(q, P[p + "q_norm.weight"], eps)
    kn, knctx = rms_norm_w(k_.reshape(T, nkv, dh), P[p + "k_norm.weight"], eps)
    vh = v_.reshape(T, nkv, dh)
    qr, rot = apply_rope(qn, cos, sin)
    kr, _ = apply_rope(kn, cos, sin)

    qhT = qr.transpose(1, 0, 2)  # (nq, T, dh)
    khT = kr.transpose(1, 0, 2)
    vhT = vh.transpose(1, 0, 2)
    rep = nq // nkv
    krep = np.repeat(khT, rep, axis=0)
    vrep = np.repeat(vhT, rep, axis=0)
    scaling = np.asarray(dh ** -0.5, dtype=qhT.dtype)
    scores = qhT @ krep.transpose(0, 2, 1) * scaling
    causal = np.zeros((T, T), dtype=scores.dtype)
    causal[np.triu_indices(T, 1)] = -np.inf
    scores = scores + causal
    e = np.exp(scores - scores.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    o = probs @ vrep  # (nq, T, dh)
    pre = o.transpose(1, 0, 2).reshape(T, nq * dh)
    post = pre * sigmoid(gate)
    mixer, ctxo = linear(post, P[p + "o_proj.weight"])

    if D is not None:
        D[prefix + "qg_proj"] = qg
        D[prefix + "q_pre_norm"] = q
        D[prefix + "k_pre_norm"] = k_.reshape(T, nkv, dh)
        D[prefix + "v_proj"] = v_
        D[prefix + "q_norm"] = qn
        D[prefix + "k_norm"] = kn
        D[prefix + "q_rope"] = qr
        D[prefix + "k_rope"] = kr
        D[prefix + "v_heads"] = vh
        D[prefix + "gate"] = gate
        D[prefix + "attn_probs"] = probs
        D[prefix + "attn_out_pre_gate"] = pre
        D[prefix + "attn_out_post_gate"] = post
        D[prefix + "mixer_out"] = mixer
    cache = dict(
        n1ctx=n1ctx, ctxq=ctxq, ctxk=ctxk, ctxv=ctxv, qnctx=qnctx, knctx=knctx,
        cos=cos, sin=sin, rot=rot, probs=probs, qhT=qhT, krep=krep, vrep=vrep, gate=gate,
        pre=pre, ctxo=ctxo, scaling=scaling, P=p, lp=lp,
    )
    return mixer, cache


def attention_bwd(cfg, P, i, dmixer, lc):
    nq, nkv, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    T = dmixer.shape[0]
    p, lp = lc["P"], lc["lp"]
    grads = {}

    dpost, grads[p + "o_proj.weight"] = linear_bwd(dmixer, lc["ctxo"])
    sg = sigmoid(lc["gate"])
    dpre = dpost * sg
    dgate = dpost * lc["pre"] * sg * (1.0 - sg)
    do = dpre.reshape(T, nq, dh).transpose(1, 0, 2)  # (nq, T, dh)

    probs, krep, vrep = lc["probs"], lc["krep"], lc["vrep"]
    dprobs = do @ vrep.transpose(0, 2, 1)
    ds = probs * (dprobs - (dprobs * probs).sum(-1, keepdims=True))  # softmax VJP
    ds = ds * lc["scaling"]
    dqhT = ds @ krep
    dkrep = ds.transpose(0, 2, 1) @ lc["qhT"]
    dvrep = probs.transpose(0, 2, 1) @ do
    rep = nq // nkv
    dkhT = dkrep.reshape(nkv, rep, T, dh).sum(1)
    dvhT = dvrep.reshape(nkv, rep, T, dh).sum(1)

    dqn = apply_rope_bwd(dqhT.transpose(1, 0, 2), lc["cos"], lc["sin"], lc["rot"])
    dkn = apply_rope_bwd(dkhT.transpose(1, 0, 2), lc["cos"], lc["sin"], lc["rot"])
    dq, dwnq = rms_norm_w_bwd(dqn, lc["qnctx"])
    grads[p + "q_norm.weight"] = dwnq
    dkh, dwnk = rms_norm_w_bwd(dkn, lc["knctx"])
    grads[p + "k_norm.weight"] = dwnk

    dqg = np.concatenate([dq, dgate.reshape(T, nq, dh)], axis=-1).reshape(T, nq * 2 * dh)
    dn1, grads[p + "q_proj.weight"] = linear_bwd(dqg, lc["ctxq"])
    dxk, grads[p + "k_proj.weight"] = linear_bwd(dkh.reshape(T, nkv * dh), lc["ctxk"])
    dn1 = dn1 + dxk
    dxv, grads[p + "v_proj.weight"] = linear_bwd(dvhT.transpose(1, 0, 2).reshape(T, nkv * dh), lc["ctxv"])
    dn1 = dn1 + dxv
    dx, dwn1 = rms_norm_w_bwd(dn1, lc["n1ctx"])
    grads[lp + "input_layernorm.weight"] = dwn1
    debug = {
        "output": dmixer,
        "post_gate": dpost,
        "pre_gate": dpre,
        "gate": dgate,
        "probs": dprobs,
        "scores": ds,
        "q_rope": dqhT.transpose(1, 0, 2),
        "k_rope": dkhT.transpose(1, 0, 2),
        "v_heads": dvhT.transpose(1, 0, 2),
        "q_norm": dqn,
        "k_norm": dkn,
        "q_pre_norm": dq,
        "k_pre_norm": dkh,
        "qg_proj": dqg,
        "norm1": dn1,
        "input": dx,
    }
    return dx, grads, debug


# ---------------------------------------------------------------------------
# Full model (embed -> 4 decoder layers -> final norm -> lm_head)
# ---------------------------------------------------------------------------


def forward_model(cfg, P, tokens, dtype=np.float32):
    """Prefill forward over tokens (T,). Returns (dumps D, backward cache C)."""
    D, C = {}, {"layers": []}
    eps = cfg["rms_norm_eps"]
    T = tokens.shape[0]
    x = P["model.embed_tokens.weight"][tokens].astype(dtype)
    D["embed_out"] = x
    cos, sin = rope_tables(cfg, np.arange(T), dtype)
    D["rope.cos"] = cos
    D["rope.sin"] = sin

    for i, lt in enumerate(cfg["layer_types"]):
        lp = f"model.layers.{i}."
        prefix = f"layer{i}."
        D[prefix + "hidden_in"] = x
        if lt == "linear_attention":
            mixer, mc = deltanet_forward(cfg, P, i, x, D, prefix)
        else:
            mixer, mc = attention_forward(cfg, P, i, x, cos, sin, D, prefix)
        h = x + mixer
        mlp_out, mlc = mlp_forward(P, lp, h, eps)
        D[prefix + "norm1_out"] = mc["n1ctx"][0] * mc["n1ctx"][1] * (1.0 + mc["n1ctx"][2])
        D[prefix + "mixer_residual"] = h
        D[prefix + "norm2_out"] = mlc["n2"]
        D[prefix + "mlp_gate"] = mlc["g_"]
        D[prefix + "mlp_up"] = mlc["u"]
        D[prefix + "mlp_act"] = mlc["act"]
        D[prefix + "mlp_out"] = mlp_out
        x = h + mlp_out
        D[prefix + "hidden_out"] = x
        C["layers"].append(dict(type=lt, mixer=mc, mlp=mlc))

    fh, fhctx = rms_norm_w(x, P["model.norm.weight"], eps)
    logits, lmctx = linear(fh, P["lm_head.weight"])
    D["final_hidden"] = fh
    D["logits"] = logits
    C["fhctx"] = fhctx
    C["lmctx"] = lmctx
    return D, C


def loss_and_dlogits(logits, targets):
    """L = mean_t CE(logits_t, target_t), fp log-softmax. Returns (L, dL/dlogits)."""
    T, V = logits.shape
    z = logits - logits.max(-1, keepdims=True)
    logZ = np.log(np.exp(z).sum(-1))
    ll = z[np.arange(T), targets] - logZ
    L = -ll.mean()
    e = np.exp(z)
    probs = e / e.sum(-1, keepdims=True)
    dlogits = probs
    dlogits[np.arange(T), targets] -= 1.0
    dlogits /= T
    return L, dlogits


def backward_model(cfg, P, C, dlogits):
    """Full-model reverse mode. Returns flat grads dict (HF parameter names)."""
    G = {}
    dfh, dWlm = linear_bwd(dlogits, C["lmctx"])
    G["grad.logits"] = dlogits
    G["grad.final_hidden"] = dfh
    G["lm_head.weight"] = dWlm
    dx, dwn = rms_norm_w_bwd(dfh, C["fhctx"])
    G["model.norm.weight"] = dwn
    for i in range(len(C["layers"]) - 1, -1, -1):
        lc = C["layers"][i]
        lp = f"model.layers.{i}."
        dxmlp, dmlp_g, dmlp_d = mlp_bwd(dx, lc["mlp"])
        for name, value in dmlp_d.items():
            G[f"grad.layer{i}.mlp_{name}"] = value
        for k_, v_ in dmlp_g.items():
            G[lp + "mlp." + k_ if "layernorm" not in k_ else lp + k_] = v_
        dh = dx + dxmlp
        dmixer = dh
        dx = dh  # residual
        if lc["type"] == "linear_attention":
            dxm, mg, md = deltanet_bwd(cfg, P, i, dmixer, lc["mixer"])
            for name, value in md.items():
                G[f"grad.layer{i}.deltanet_{name}"] = value
        else:
            dxm, mg, md = attention_bwd(cfg, P, i, dmixer, lc["mixer"])
            for name, value in md.items():
                G[f"grad.layer{i}.attention_{name}"] = value
        G.update(mg)
        dx = dx + dxm
    G["grad.embed_out"] = dx
    return G


# ---------------------------------------------------------------------------
# Decode case: one token through the linear_attention layers on top of prefill
# state (conv state = last K-1 raw qkv rows; recurrent state S from prefill).
# ---------------------------------------------------------------------------


def decode_case(cfg, P, inputs, D, C, dtype=np.float32):
    """Emits decode.* dumps for each linear_attention layer (token after prefill)."""
    DD = {}
    nk, dk = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"]
    nv, dv = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
    kd, vd = nk * dk, nv * dv
    eps = cfg["rms_norm_eps"]
    T = cfg["seq_len"]
    tok = int(inputs["decode_token"][0])
    x = P["model.embed_tokens.weight"][tok].astype(dtype)  # (H,)

    for i, lt in enumerate(cfg["layer_types"]):
        if lt != "linear_attention":
            continue
        lp = f"model.layers.{i}."
        p = lp + "linear_attn."
        prefix = f"decode.layer{i}."
        n1 = rms_norm_w(x[None, :], P[lp + "input_layernorm.weight"], eps)[0][0]
        qkv = n1 @ P[p + "in_proj_qkv.weight"].T
        z = n1 @ P[p + "in_proj_z.weight"].T
        b = n1 @ P[p + "in_proj_b.weight"].T
        a = n1 @ P[p + "in_proj_a.weight"].T

        conv_state_in = D[f"layer{i}.qkv_pre_conv"][T - 3 :].copy()  # (K-1, conv_dim)
        qkv_conv, conv_state_out = causal_conv1d_update(conv_state_in, qkv, P[p + "conv1d.weight"])
        q = qkv_conv[:kd].reshape(nk, dk)
        k_ = qkv_conv[kd : 2 * kd].reshape(nk, dk)
        vh = qkv_conv[2 * kd :].reshape(nv, dv)
        beta = sigmoid(b)
        g = -np.exp(P[p + "A_log"]) * softplus(a + P[p + "dt_bias"])
        rep = nv // nk
        ql = l2norm(np.repeat(q, rep, axis=0), cfg["l2norm_eps"])[0]
        kl = l2norm(np.repeat(k_, rep, axis=0), cfg["l2norm_eps"])[0]
        S_in = D[f"layer{i}.S_final"]
        out, S_out = recurrent_gated_delta_rule(
            ql[:, None, :], kl[:, None, :], vh[:, None, :], g[:, None], beta[:, None], S_in
        )
        delta_out = out[:, 0, :]  # (nv, dv)
        gated = gated_rms_norm(delta_out, z.reshape(nv, dv), P[p + "norm.weight"], eps)[0]
        gated_flat = gated.reshape(vd)
        mixer = gated_flat @ P[p + "out_proj.weight"].T

        h = x + mixer
        mlp_out, _ = mlp_forward(P, lp, h[None, :], eps)
        y = h + mlp_out[0]

        DD[prefix + "hidden_in"] = x
        DD[prefix + "norm1_out"] = n1
        DD[prefix + "qkv_pre_conv"] = qkv
        DD[prefix + "conv_state_in"] = conv_state_in
        DD[prefix + "conv_state_out"] = conv_state_out
        DD[prefix + "q_post_conv"] = qkv_conv[:kd]
        DD[prefix + "k_post_conv"] = qkv_conv[kd : 2 * kd]
        DD[prefix + "v_post_conv"] = qkv_conv[2 * kd :]
        DD[prefix + "q_l2"] = ql
        DD[prefix + "k_l2"] = kl
        DD[prefix + "v_heads"] = vh
        DD[prefix + "beta"] = beta
        DD[prefix + "g"] = g
        DD[prefix + "S_in"] = S_in
        DD[prefix + "delta_out"] = delta_out
        DD[prefix + "S_out"] = S_out
        DD[prefix + "gated_out"] = gated_flat
        DD[prefix + "mixer_out"] = mixer
        DD[prefix + "mlp_out"] = mlp_out[0]
        DD[prefix + "hidden_out"] = y
        x = y
    return DD


# ---------------------------------------------------------------------------
# Fixture assembly / IO
# ---------------------------------------------------------------------------

# Backward probes checked in as fixtures (docs/QWEN36_MEGAKERNEL.md minimum set).
GRAD_FIXTURES = (
    "grad.logits",
    "grad.final_hidden",
    "grad.model.layers.0.linear_attn.in_proj_qkv.weight",
    "grad.model.layers.0.linear_attn.in_proj_z.weight",
    "grad.model.layers.0.linear_attn.in_proj_b.weight",
    "grad.model.layers.0.linear_attn.in_proj_a.weight",
    "grad.model.layers.0.linear_attn.conv1d.weight",
    "grad.model.layers.0.linear_attn.A_log",
    "grad.model.layers.0.linear_attn.dt_bias",
    "grad.model.layers.0.linear_attn.norm.weight",
    "grad.model.layers.0.linear_attn.out_proj.weight",
    "grad.model.layers.0.input_layernorm.weight",
    "grad.layer0.deltanet_output",
    "grad.layer0.deltanet_gated_out",
    "grad.layer0.deltanet_delta_out",
    "grad.layer0.deltanet_z",
    "grad.layer0.deltanet_q_l2_repeated",
    "grad.layer0.deltanet_k_l2_repeated",
    "grad.layer0.deltanet_v_heads",
    "grad.layer0.deltanet_g",
    "grad.layer0.deltanet_beta",
    "grad.layer0.deltanet_q_repeated",
    "grad.layer0.deltanet_k_repeated",
    "grad.layer0.deltanet_q_post_conv",
    "grad.layer0.deltanet_k_post_conv",
    "grad.layer0.deltanet_qkv_post_conv",
    "grad.layer0.deltanet_qkv_pre_conv",
    "grad.layer0.deltanet_b",
    "grad.layer0.deltanet_a",
    "grad.layer0.deltanet_norm1",
    "grad.layer0.deltanet_input",
    "grad.model.layers.0.post_attention_layernorm.weight",
    "grad.model.layers.0.mlp.gate_proj.weight",
    "grad.model.layers.0.mlp.up_proj.weight",
    "grad.model.layers.0.mlp.down_proj.weight",
    "grad.layer0.mlp_output",
    "grad.layer0.mlp_act",
    "grad.layer0.mlp_gate",
    "grad.layer0.mlp_up",
    "grad.layer0.mlp_norm2",
    "grad.layer0.mlp_input",
    "grad.model.layers.3.self_attn.q_proj.weight",
    "grad.model.layers.3.self_attn.k_proj.weight",
    "grad.model.layers.3.self_attn.v_proj.weight",
    "grad.model.layers.3.self_attn.o_proj.weight",
    "grad.model.layers.3.self_attn.q_norm.weight",
    "grad.model.layers.3.self_attn.k_norm.weight",
    "grad.model.layers.3.input_layernorm.weight",
    "grad.layer3.attention_post_gate",
    "grad.layer3.attention_output",
    "grad.layer3.attention_pre_gate",
    "grad.layer3.attention_gate",
    "grad.layer3.attention_probs",
    "grad.layer3.attention_scores",
    "grad.layer3.attention_q_rope",
    "grad.layer3.attention_k_rope",
    "grad.layer3.attention_v_heads",
    "grad.layer3.attention_q_norm",
    "grad.layer3.attention_k_norm",
    "grad.layer3.attention_q_pre_norm",
    "grad.layer3.attention_k_pre_norm",
    "grad.layer3.attention_qg_proj",
    "grad.layer3.attention_norm1",
    "grad.layer3.attention_input",
)


def build_fixtures(cfg=CFG, dtype=np.float32):
    P, inputs = init_params_and_inputs(cfg, dtype=dtype)
    tokens = inputs["tokens"].astype(np.int64)
    D, C = forward_model(cfg, P, tokens, dtype)
    L, dlogits = loss_and_dlogits(D["logits"], inputs["targets"].astype(np.int64))
    G = backward_model(cfg, P, C, dlogits)
    DD = decode_case(cfg, P, inputs, D, C, dtype)

    F = dict(P)
    F["tokens"] = inputs["tokens"].astype(np.float32)
    F["targets"] = inputs["targets"].astype(np.float32)
    F["decode_token"] = inputs["decode_token"].astype(np.float32)
    F["probe_hidden"] = inputs["probe_hidden"]
    F.update(D)
    F["loss"] = np.asarray([L], np.float32)
    embedding_gradient = np.zeros_like(P["model.embed_tokens.weight"])
    np.add.at(embedding_gradient, tokens, G["grad.embed_out"])
    G["model.embed_tokens.weight"] = embedding_gradient
    # Keep every analytical activation and parameter VJP as a fixture.  The
    # smaller GRAD_FIXTURES tuple remains the finite-difference/self-test
    # contract, while this complete set lets the Lean full-model gate prove
    # every residual edge and trainable tensor across all decoder layers.
    for name, value in G.items():
        fixture_name = name if name.startswith("grad.") else "grad." + name
        F[fixture_name] = value
    F.update(DD)
    return F


def fixture_filename(name, shape):
    dims = "x".join(str(d) for d in shape)
    return f"{name}.{dims}.f32.bin"


def dump_fixtures(outdir, cfg=CFG):
    F = build_fixtures(cfg)
    os.makedirs(outdir, exist_ok=True)
    tensors = {}
    for name in sorted(F):
        arr = np.ascontiguousarray(np.asarray(F[name], dtype=np.float32))
        fname = fixture_filename(name, arr.shape)
        data = arr.tobytes(order="C")
        with open(os.path.join(outdir, fname), "wb") as f:
            f.write(data)
        tensors[name] = {
            "shape": list(arr.shape),
            "file": fname,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = {
        "meta": {
            "generator": "oracle.py (pure NumPy fp32)",
            "numpy_version": np.__version__,
            "seed": SEED,
            "init_std": INIT_STD,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "loss": "L = mean_t CE(logits_t, targets_t); grads are dL/d(.)",
            "format": "raw little-endian fp32, C (row-major) order; one tensor per file",
        },
        "tensors": tensors,
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def load_fixtures(dirpath):
    with open(os.path.join(dirpath, "manifest.json")) as f:
        manifest = json.load(f)
    out = {}
    for name, rec in manifest["tensors"].items():
        arr = np.fromfile(os.path.join(dirpath, rec["file"]), dtype="<f4").reshape(rec["shape"])
        out[name] = arr
    return manifest, out


def compare_dirs(dira, dirb):
    ma, _ = load_fixtures(dira)
    mb, _ = load_fixtures(dirb)
    if ma["tensors"].keys() != mb["tensors"].keys():
        missing = ma["tensors"].keys() ^ mb["tensors"].keys()
        print(f"COMPARE FAIL: tensor sets differ: {sorted(missing)}")
        return 1
    nfail = 0
    for name in sorted(ma["tensors"]):
        ra, rb = ma["tensors"][name], mb["tensors"][name]
        if ra["shape"] != rb["shape"]:
            print(f"COMPARE FAIL: {name}: shape {ra['shape']} vs {rb['shape']}")
            nfail += 1
            continue
        with open(os.path.join(dira, ra["file"]), "rb") as f:
            ba = f.read()
        with open(os.path.join(dirb, rb["file"]), "rb") as f:
            bb = f.read()
        if ba != bb:
            aa = np.frombuffer(ba, dtype="<f4")
            bb_ = np.frombuffer(bb, dtype="<f4")
            err = np.abs(aa - bb_).max()
            print(f"COMPARE FAIL: {name}: bytes differ, max abs err {err:.3e}")
            nfail += 1
    if nfail:
        print(f"COMPARE FAIL: {nfail} tensor(s) differ")
        return 1
    print(f"COMPARE PASS: {len(ma['tensors'])} tensors byte-identical")
    return 0


# ---------------------------------------------------------------------------
# Self-test: delta-rule parity + finite-difference backward check
# ---------------------------------------------------------------------------


def selftest(cfg=CFG):
    ok = True

    def report(name, err, tol):
        nonlocal ok
        status = "PASS" if err <= tol else "FAIL"
        ok = ok and err <= tol
        print(f"  [{status}] {name}: max abs err {err:.3e} (tol {tol:.0e})")

    P, inputs = init_params_and_inputs(cfg, dtype=np.float32)
    tokens = inputs["tokens"].astype(np.int64)
    targets = inputs["targets"].astype(np.int64)
    D, C = forward_model(cfg, P, tokens, np.float32)

    # 1. chunked vs recurrent gated delta rule (mathematically identical)
    err = 0.0
    for i, lc in enumerate(C["layers"]):
        if lc["type"] != "linear_attention":
            continue
        mc = lc["mixer"]
        out_r, S_r = recurrent_gated_delta_rule(
            mc["ql"].transpose(1, 0, 2), mc["kl"].transpose(1, 0, 2),
            mc["vh"].transpose(1, 0, 2), mc["g"].T.copy(), mc["beta"].T.copy(),
            np.zeros((cfg["linear_num_value_heads"], cfg["linear_key_head_dim"],
                      cfg["linear_value_head_dim"]), np.float32),
        )
        err = max(err, np.abs(out_r.transpose(1, 0, 2) - D[f"layer{i}.delta_out"]).max())
        err = max(err, np.abs(S_r - D[f"layer{i}.S_final"]).max())
    report("chunked vs recurrent delta rule (prefill)", err, 1e-5)

    # 2. decode step vs 9-token recurrent reference (layer 0, end-to-end state)
    DD = decode_case(cfg, P, inputs, D, C, np.float32)
    tok9 = np.concatenate([tokens, inputs["decode_token"].astype(np.int64)])
    x9 = P["model.embed_tokens.weight"][tok9].astype(np.float32)
    _, mc9 = deltanet_forward(cfg, P, 0, x9)
    out9, S9 = recurrent_gated_delta_rule(
        mc9["ql"].transpose(1, 0, 2), mc9["kl"].transpose(1, 0, 2),
        mc9["vh"].transpose(1, 0, 2), mc9["g"].T.copy(), mc9["beta"].T.copy(),
        np.zeros((cfg["linear_num_value_heads"], cfg["linear_key_head_dim"],
                  cfg["linear_value_head_dim"]), np.float32),
    )
    report(
        "decode delta step vs 9-token recurrent (layer 0, out)",
        np.abs(out9[:, -1] - DD["decode.layer0.delta_out"]).max(), 1e-5,
    )
    report(
        "decode delta step vs 9-token recurrent (layer 0, S)",
        np.abs(S9 - DD["decode.layer0.S_out"]).max(), 1e-5,
    )

    # 3. finite-difference check of the hand-written backward (fp64, central, eps 1e-3)
    P64, inputs64 = init_params_and_inputs(cfg, dtype=np.float64)
    t64 = inputs64["tokens"].astype(np.int64)
    tg64 = inputs64["targets"].astype(np.int64)
    D64, C64 = forward_model(cfg, P64, t64, np.float64)
    _, dlogits64 = loss_and_dlogits(D64["logits"], tg64)
    G64 = backward_model(cfg, P64, C64, dlogits64)

    def loss_fn(Pcur):
        Dcur, _ = forward_model(cfg, Pcur, t64, np.float64)
        return loss_and_dlogits(Dcur["logits"], tg64)[0]

    eps = 1e-3
    rng = np.random.default_rng(0xFD)
    worst = 0.0
    for name, count in (
        ("model.layers.0.linear_attn.in_proj_qkv.weight", 24),
        ("model.layers.3.self_attn.q_proj.weight", 24),
        ("model.layers.0.linear_attn.in_proj_b.weight", 8),
        ("model.layers.0.linear_attn.in_proj_a.weight", 8),
    ):
        W = P64[name]
        grad = G64[name]
        err = 0.0
        for _ in range(count):
            ix = tuple(int(rng.integers(0, s)) for s in W.shape)
            Wp = W.copy()
            Wp[ix] += eps
            Wm = W.copy()
            Wm[ix] -= eps
            fd = (loss_fn({**P64, name: Wp}) - loss_fn({**P64, name: Wm})) / (2 * eps)
            err = max(err, abs(fd - grad[ix]))
        worst = max(worst, err)
        report(f"finite-difference dL/d({name}) [{count} entries]", err, 1e-3)

    # 4. fp32 analytic grads vs fp64 reference (fixture precision sanity)
    _, dlogits32 = loss_and_dlogits(D["logits"], targets)
    G32 = backward_model(cfg, P, C, dlogits32)
    err = 0.0
    for name in GRAD_FIXTURES:
        key = name if not name.startswith("grad.model") else name[len("grad."):]
        err = max(err, np.abs(G32[key] - G64[key]).max())
    report("fp32 vs fp64 analytic grads (fixture probes)", err, 1e-3)

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "dump":
        manifest = dump_fixtures(argv[2])
        print(f"DUMP PASS: wrote {len(manifest['tensors'])} tensors + manifest.json to {argv[2]}")
        return 0
    if cmd == "selftest":
        return selftest()
    if cmd == "compare":
        return compare_dirs(argv[2], argv[3])
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
