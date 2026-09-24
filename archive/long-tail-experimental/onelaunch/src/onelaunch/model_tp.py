"""Tensor-parallel sharding of the dense megakernel + single-GPU correctness sim.

Batch-1 decode is weight-bandwidth-bound, so splitting the weights across R GPUs
divides the bandwidth floor by R while the only cross-GPU traffic is an all-reduce
of the H-wide residual (a few KB) after o-proj and after down-proj -- tiny, so TP
pushes batch-1 latency *below* the single-GPU floor. Sharding scheme (Megatron):
  - QKV, gate, up : column-parallel (split output) -> each rank owns nh/R heads
    and I/R intermediate channels, computes its slice locally.
  - o, down       : row-parallel (split input) -> each rank produces a partial
    H-vector; an all-reduce sums them into the full residual.
  - RMSNorm/RoPE/residual/embed : replicated (both ranks hold the full residual).
  - lm_head       : column-parallel -> partial logits, all-gathered for argmax.

This file verifies the sharding *math* on one GPU by running all R ranks in one
process and summing partials by hand (no NCCL). model_tp_dist.py runs it for real
across GPUs with torchrun. The kernels are Part 1's, unchanged.
"""

from __future__ import annotations

import torch

from .kernels import decode_attn, gemv, rmsnorm_gemv, rope_write, swiglu_gemv
from .model_fused import build_fused
from .model_ref import KVCache, rms_norm, rope_cos_sin


def shard_weights(w, world):
    """Split each layer's weights into `world` row/column-parallel shards.
    Returns a list (per rank) of layer-dicts holding that rank's slices."""
    cfg = w.cfg
    H, nh, nkv, hd, I = cfg["hidden"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], w.layers[0]["wgate"].shape[0]
    assert nh % world == 0 and nkv % world == 0 and I % world == 0
    nh_r, nkv_r, I_r = nh // world, nkv // world, I // world
    qd_r, kd_r = nh_r * hd, nkv_r * hd
    ranks = [[] for _ in range(world)]
    for ly in w.layers:
        wq, wk, wv = ly["wq"], ly["wk"], ly["wv"]
        bq, bk, bv = ly["bq"], ly["bk"], ly["bv"]
        for r in range(world):
            qsl, ksl = slice(r * qd_r, (r + 1) * qd_r), slice(r * kd_r, (r + 1) * kd_r)
            isl = slice(r * I_r, (r + 1) * I_r)
            wqkv = torch.cat([wq[qsl], wk[ksl], wv[ksl]], 0).contiguous()   # column-parallel QKV
            bqkv = torch.cat([bq[qsl], bk[ksl], bv[ksl]], 0).contiguous()
            ranks[r].append({
                "ln1": ly["ln1"], "ln2": ly["ln2"],                        # replicated norms
                "wqkv": wqkv, "bqkv": bqkv,
                "wo": ly["wo"][:, r * qd_r:(r + 1) * qd_r].contiguous(),    # row-parallel o (split input=heads)
                "wgate": ly["wgate"][isl].contiguous(),                     # column-parallel gate
                "wup": ly["wup"][isl].contiguous(),
                "wdown": ly["wdown"][:, isl].contiguous(),                  # row-parallel down (split input=I)
            })
    lm_r = w.embed.shape[0] // 1  # lm_head tied to embed [V,H]; column-parallel by vocab handled in sim below
    meta = dict(nh_r=nh_r, nkv_r=nkv_r, hd=hd, H=H, I_r=I_r, group=nh // nkv, eps=cfg["eps"],
                n_layers=cfg["n_layers"])
    return ranks, meta


@torch.no_grad()
def verify_tp_sim(model_id="Qwen/Qwen2.5-1.5B-Instruct", world=2, ctx=32, steps=8):
    """Run `world` ranks in one process, summing partials by hand, and check the
    result is argmax-identical to the single-GPU oracle."""
    from .model_ref import decode_step as ref_step

    w = build_fused(model_id)
    shards, m = shard_weights(w, world)
    nh_r, nkv_r, hd, H, I_r, group, eps = (m["nh_r"], m["nkv_r"], m["hd"], m["H"],
                                           m["I_r"], m["group"], m["eps"])
    scale = hd ** -0.5
    # each rank keeps its own KV cache (only its heads)
    kvs = [KVCache.__new__(KVCache) for _ in range(world)]
    for r in range(world):
        kvs[r].k = torch.zeros(m["n_layers"], ctx + steps + 1, nkv_r, hd, device="cuda", dtype=torch.bfloat16)
        kvs[r].v = torch.zeros_like(kvs[r].k)

    def tp_step(tok, pos):
        h = w.embed[tok].clone()
        cos, sin = rope_cos_sin(w.inv_freq, pos, h.dtype)
        for i in range(m["n_layers"]):
            res = h
            # --- attention: each rank does its heads, o-proj partials all-reduced ---
            o_partials = []
            for r in range(world):
                ly = shards[r][i]
                qkv = rmsnorm_gemv(h, ly["ln1"], ly["wqkv"], ly["bqkv"], eps)
                q = rope_write(qkv, cos, sin, kvs[r].k[i], kvs[r].v[i], torch.tensor([pos], device="cuda"),
                               nh_r, nkv_r, hd)
                attn = decode_attn(q, kvs[r].k[i, : pos + 1], kvs[r].v[i, : pos + 1], group, scale)
                o_partials.append(gemv(attn.reshape(nh_r * hd), ly["wo"]).float())
            h = res + sum(o_partials).to(h.dtype)                          # all-reduce (sum)
            res = h
            # --- MLP: each rank does its I/R channels, down partials all-reduced ---
            d_partials = []
            for r in range(world):
                ly = shards[r][i]
                act = swiglu_gemv(h, ly["ln2"], ly["wgate"], ly["wup"], eps)
                d_partials.append(gemv(act, ly["wdown"]).float())
            h = res + sum(d_partials).to(h.dtype)                          # all-reduce (sum)
        h = rms_norm(h, w.final_norm, eps)
        return h @ w.embed.T

    kv_ref = KVCache(w, ctx + steps + 1)
    ids = torch.randint(0, w.embed.shape[0], (ctx,), device="cuda")
    for pos in range(ctx):
        lr = ref_step(w, int(ids[pos]), kv_ref, pos)
        lt = tp_step(int(ids[pos]), pos)
    d = (lr.float() - lt.float()).abs()
    print(f"[world={world}] prefill max|Δ vs single-GPU oracle| = {d.max().item():.4f}  "
          f"argmax ref={int(lr.argmax())} tp={int(lt.argmax())}")

    tok = int(lr.argmax())
    match = 0
    for s in range(steps):
        pos = ctx + s
        lr = ref_step(w, tok, kv_ref, pos)
        lt = tp_step(tok, pos)
        match += int(lr.argmax()) == int(lt.argmax())
        tok = int(lr.argmax())
    print(f"[world={world}] greedy argmax agreement (TP sim vs oracle) over {steps} steps: {match}/{steps}")


if __name__ == "__main__":
    verify_tp_sim(world=2)
