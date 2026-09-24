"""Fused batch-1 decode using the Triton kernels, checked against model_ref.

Per layer this is 6 launches (rmsnorm+qkv, attention, o+residual, rmsnorm+
gate/up+swiglu, down+residual, plus torch rope/cache-write) instead of the ~40
eager ops. RoPE / cache-write / final norm stay in torch for now -- negligible
traffic; the point is that every weight read goes through one memory-optimal
kernel and the residual stream doesn't round-trip HBM between the big ops.
"""

from __future__ import annotations

import torch

from .kernels import decode_attn, gemv, rmsnorm_gemv, swiglu_gemv
from .model_ref import KVCache, apply_rope, load_weights, rms_norm, rope_cos_sin


def build_fused(model_id="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
    w = load_weights(model_id, device=device)
    hd = w.cfg["head_dim"]
    for ly in w.layers:
        ly["wqkv"] = torch.cat([ly["wq"], ly["wk"], ly["wv"]], 0).contiguous()
        ly["bqkv"] = torch.cat([ly["bq"], ly["bk"], ly["bv"]], 0).contiguous()
    return w


@torch.no_grad()
def decode_step_fused(w, tok: int, kv: KVCache, pos: int) -> torch.Tensor:
    cfg = w.cfg
    H, nh, nkv, hd, eps = cfg["hidden"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["eps"]
    group, scale = nh // nkv, hd ** -0.5
    h = w.embed[tok].clone()
    cos, sin = rope_cos_sin(w.inv_freq, pos, h.dtype)
    qd, kd = nh * hd, nkv * hd
    for i, ly in enumerate(w.layers):
        res = h
        qkv = rmsnorm_gemv(h, ly["ln1"], ly["wqkv"], ly["bqkv"], eps)      # [qd+2kd]
        q = apply_rope(qkv[:qd].view(nh, hd), cos, sin).contiguous()
        k = apply_rope(qkv[qd:qd + kd].view(nkv, hd), cos, sin)
        v = qkv[qd + kd:].view(nkv, hd)
        kv.k[i, pos] = k
        kv.v[i, pos] = v
        attn = decode_attn(q, kv.k[i, : pos + 1], kv.v[i, : pos + 1], group, scale)
        h = gemv(attn.reshape(H), ly["wo"], residual=res)                 # o_proj + res
        res = h
        act = swiglu_gemv(h, ly["ln2"], ly["wgate"], ly["wup"], eps)      # [I]
        h = gemv(act, ly["wdown"], residual=res)                          # down + res
    h = rms_norm(h, w.final_norm, eps)
    return h @ w.embed.T


@torch.no_grad()
def verify(model_id="Qwen/Qwen2.5-1.5B-Instruct", ctx=32, steps=8):
    """Fused vs the plain-torch reference oracle: logits close, argmax identical."""
    from .model_ref import decode_step as ref_step

    w = build_fused(model_id)
    ids = torch.randint(0, w.embed.shape[0], (ctx,), device="cuda")

    kv_ref = KVCache(w, ctx + steps + 1)
    kv_fus = KVCache(w, ctx + steps + 1)
    for pos in range(ctx):
        lr = ref_step(w, int(ids[pos]), kv_ref, pos)
        lf = decode_step_fused(w, int(ids[pos]), kv_fus, pos)
    d = (lr.float() - lf.float()).abs()
    print(f"prefill last-token max|Δ vs ref| = {d.max().item():.4f}   "
          f"argmax ref={int(lr.argmax())} fused={int(lf.argmax())}")

    tok = int(lr.argmax())
    match = 0
    for s in range(steps):
        pos = ctx + s
        lr = ref_step(w, tok, kv_ref, pos)
        lf = decode_step_fused(w, tok, kv_fus, pos)
        match += int(lr.argmax()) == int(lf.argmax())
        tok = int(lr.argmax())
    print(f"greedy argmax agreement (fused vs ref) over {steps} steps: {match}/{steps}")


@torch.no_grad()
def bench_eager(model_id="Qwen/Qwen2.5-1.5B-Instruct", ctx=512, steps=64):
    """Rough eager latency of the fused step (Python + launch overhead included).
    The graph-captured number is the real one; this just sanity-checks the path."""
    import time

    w = build_fused(model_id)
    kv = KVCache(w, ctx + steps + 1)
    tok = 1
    for pos in range(ctx):  # prefill via the fused step
        tok = int(decode_step_fused(w, tok, kv, pos).argmax())
    torch.cuda.synchronize()
    t = []
    for s in range(steps):
        pos = ctx + s
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        tok = int(decode_step_fused(w, tok, kv, pos).argmax())
        en.record(); torch.cuda.synchronize()
        t.append(st.elapsed_time(en))
    t.sort()
    print(f"fused eager decode: {t[len(t)//2]:.3f} ms/token (median, Python overhead included)")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        bench_eager()
    else:
        verify()
