#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
"""Generate the NumPy oracle fixtures for the cuda_qwen36_primitives gate.

Pure NumPy; every tensor is computed in float64 and stored little-endian fp32 row-major as
`<name>.<dim1>x<dim2>.f32.bin` plus a `manifest.json`, the layout of
docs/QWEN36_MEGAKERNEL.md. Tiny configuration: hidden 256, head 64, rotary 16, seq 8.

Usage: python3 gen_fixtures.py [output-dir]   (default: ./fixtures)
"""

import hashlib
import json
import os
import sys

import numpy as np

HIDDEN = 256
ROWS = 8
HEAD_DIM = 64
HEADS = 4
ROTARY = 16
HALF = ROTARY // 2
SEQ = 8
VOCAB = 512
EPS = 1e-6
THETA = 1e7
POSITION = 5.0
SEED = 0xC0FFEE


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def silu_grad(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def bf16_round_f32(a):
    """Round-to-nearest-even bf16 emulation, returned widened to fp32."""
    bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.astype(np.uint32).view(np.float32)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    manifest = {}

    def save(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).astype("<f4")
        suffix = "x".join(str(d) for d in a.shape)
        fname = f"{name}.{suffix}.f32.bin" if suffix else f"{name}.f32.bin"
        data = a.tobytes()
        with open(os.path.join(out, fname), "wb") as handle:
            handle.write(data)
        manifest[name] = {
            "shape": list(a.shape),
            "file": fname,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    # (1 + w) RMSNorm, rows x hidden.
    x = rng.standard_normal((ROWS, HIDDEN))
    w = rng.uniform(-0.25, 0.25, HIDDEN)
    dy = rng.standard_normal((ROWS, HIDDEN))
    ms = (x**2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + EPS)
    y = x * inv * (1.0 + w)
    dot = (dy * (1.0 + w) * x).sum(-1, keepdims=True)
    dx = inv * dy * (1.0 + w) - x * dot * inv**3 / HIDDEN
    dw = (dy * (x * inv)).sum(0)
    save("rms_x", x)
    save("rms_w", w)
    save("rms_dy", dy)
    save("rms_y", y)
    save("rms_inv", inv[:, 0])
    save("rms_dx", dx)
    save("rms_dw", dw)

    # Unit-weight gated RMSNorm, rows x head_dim.
    gx = rng.standard_normal((ROWS, HEAD_DIM))
    gz = rng.standard_normal((ROWS, HEAD_DIM))
    gdy = rng.standard_normal((ROWS, HEAD_DIM))
    gms = (gx**2).mean(-1, keepdims=True)
    ginv = 1.0 / np.sqrt(gms + EPS)
    normed = gx * ginv
    gy = normed * silu(gz)
    gdot = (gdy * silu(gz) * gx).sum(-1, keepdims=True)
    gdx = ginv * gdy * silu(gz) - gx * gdot * ginv**3 / HEAD_DIM
    gdz = gdy * normed * silu_grad(gz)
    save("gated_x", gx)
    save("gated_z", gz)
    save("gated_dy", gdy)
    save("gated_y", gy)
    save("gated_inv", ginv[:, 0])
    save("gated_dx", gdx)
    save("gated_dz", gdz)

    # Partial RoPE (GPT-NeoX rotate-half) at POSITION.
    inv_freq = 1.0 / (THETA ** (np.arange(HALF, dtype=np.float64) * (2.0 / ROTARY)))
    angle = POSITION * inv_freq
    cos = np.cos(angle)
    sin = np.sin(angle)
    q = rng.standard_normal((HEADS, HEAD_DIM))
    k = rng.standard_normal((HEADS, HEAD_DIM))
    dq = rng.standard_normal((HEADS, HEAD_DIM))
    dk = rng.standard_normal((HEADS, HEAD_DIM))

    def rope_fwd(t):
        out = t.copy()
        out[:, :HALF] = t[:, :HALF] * cos - t[:, HALF:ROTARY] * sin
        out[:, HALF:ROTARY] = t[:, HALF:ROTARY] * cos + t[:, :HALF] * sin
        return out

    def rope_bwd(g):
        out = g.copy()
        out[:, :HALF] = g[:, :HALF] * cos + g[:, HALF:ROTARY] * sin
        out[:, HALF:ROTARY] = g[:, HALF:ROTARY] * cos - g[:, :HALF] * sin
        return out

    save("rope_invfreq", inv_freq)
    save("rope_q", q)
    save("rope_k", k)
    save("rope_q_out", rope_fwd(q))
    save("rope_k_out", rope_fwd(k))
    save("rope_dq", dq)
    save("rope_dk", dk)
    save("rope_dq_pre", rope_bwd(dq))
    save("rope_dk_pre", rope_bwd(dk))

    # Causal depthwise conv1d (k=4) with SiLU, HF weight layout [channels, 4].
    cx = rng.standard_normal((SEQ, HIDDEN))
    cw = rng.standard_normal((HIDDEN, 4)) * 0.5
    cdy = rng.standard_normal((SEQ, HIDDEN))
    u = np.zeros((SEQ, HIDDEN))
    for j in range(4):
        u[3 - j:, :] += cw[:, j] * cx[: SEQ - 3 + j, :]
    cy = silu(u)
    ds = cdy * silu_grad(u)
    cdx = np.zeros((SEQ, HIDDEN))
    cdw = np.zeros((HIDDEN, 4))
    for j in range(4):
        cdx[: SEQ - 3 + j, :] += cw[:, j] * ds[3 - j:, :]
        cdw[:, j] = (cx[: SEQ - 3 + j, :] * ds[3 - j:, :]).sum(0)
    save("conv_x", cx)
    save("conv_w", cw)
    save("conv_dy", cdy)
    save("conv_y", cy)
    save("conv_dx", cdx)
    save("conv_dw", cdw)

    # Decode step at t = SEQ - 1: state carries x[SEQ-4 .. SEQ-2]. yt == conv_y[-1] and
    # dxt == conv_dx[-1] by construction, cross-checked in the gate.
    state = cx[SEQ - 4 : SEQ - 1].copy()
    xt = cx[SEQ - 1].copy()
    u_t = cw[:, 0] * state[0] + cw[:, 1] * state[1] + cw[:, 2] * state[2] + cw[:, 3] * xt
    yt = silu(u_t)
    state_out = np.stack([state[1], state[2], xt])
    dyt = cdy[SEQ - 1].copy()
    ds_t = dyt * silu_grad(u_t)
    dxt = cw[:, 3] * ds_t
    dstate = np.stack([cw[:, 0] * ds_t, cw[:, 1] * ds_t, cw[:, 2] * ds_t])
    dw_step = np.stack(
        [state[0] * ds_t, state[1] * ds_t, state[2] * ds_t, xt * ds_t], axis=1
    )
    save("conv_state", state)
    save("conv_xt", xt)
    save("conv_yt", yt)
    save("conv_state_out", state_out)
    save("conv_dyt", dyt)
    save("conv_dxt", dxt)
    save("conv_dstate", dstate)
    save("conv_dw_step", dw_step)

    # Per-head l2 normalization, heads x head_dim.
    lx = rng.standard_normal((HEADS, HEAD_DIM))
    den = np.sqrt((lx**2).sum(-1, keepdims=True) + EPS)
    ly = lx / den
    ldy = rng.standard_normal((HEADS, HEAD_DIM))
    ldot = (ldy * ly).sum(-1, keepdims=True)
    ldx = (ldy - ly * ldot) / den
    save("l2_x", lx)
    save("l2_y", ly)
    save("l2_dy", ldy)
    save("l2_dx", ldx)

    # SiLU and SwiGLU, rows x hidden.
    sx = rng.standard_normal((ROWS, HIDDEN))
    sdy = rng.standard_normal((ROWS, HIDDEN))
    save("silu_x", sx)
    save("silu_y", silu(sx))
    save("silu_dy", sdy)
    save("silu_dx", sdy * silu_grad(sx))

    gate = rng.standard_normal((ROWS, HIDDEN))
    up = rng.standard_normal((ROWS, HIDDEN))
    dh = rng.standard_normal((ROWS, HIDDEN))
    save("swiglu_gate", gate)
    save("swiglu_up", up)
    save("swiglu_dh", dh)
    save("swiglu_h", silu(gate) * up)
    save("swiglu_dgate", dh * up * silu_grad(gate))
    save("swiglu_dup", dh * silu(gate))

    # Embedding gather, vocab x hidden table.
    table = rng.standard_normal((VOCAB, HIDDEN))
    ids = rng.integers(0, VOCAB, ROWS)
    save("embed_table", table)
    save("embed_ids", ids.astype(np.float64))
    save("embed_out", table[ids])

    # Casts: expected tensors are the RNE bf16 rounding widened back to fp32.
    vals = rng.standard_normal((ROWS, HIDDEN)) * 3.0
    save("cast_x", vals)
    save("cast_bf16", bf16_round_f32(vals))

    manifest["_meta"] = {
        "seed": SEED,
        "eps": EPS,
        "rope_theta": THETA,
        "rope_position": POSITION,
        "hidden": HIDDEN,
        "rows": ROWS,
        "head_dim": HEAD_DIM,
        "heads": HEADS,
        "rotary": ROTARY,
        "seq": SEQ,
        "vocab": VOCAB,
        "generator": "gen_fixtures.py (numpy float64 oracle, fp32 storage)",
    }
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"wrote {len(manifest) - 1} fixtures to {out}")


if __name__ == "__main__":
    main()
