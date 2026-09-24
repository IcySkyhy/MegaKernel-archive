"""Hand-rolled batch-1 decode for Qwen2.5-style models, in plain PyTorch.

This is the correctness oracle and the surface the fused kernels replace. It is
deliberately written op-by-op (RMSNorm, QKV, RoPE, GQA attention, O-proj, SwiGLU,
down-proj) so each op can be swapped for a Triton kernel and checked against HF
logits at every step. No batching, no cache abstraction beyond flat tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Weights:
    embed: torch.Tensor            # [vocab, hidden]  (tied to lm_head)
    layers: list                   # list of dicts of per-layer tensors
    final_norm: torch.Tensor       # [hidden]
    inv_freq: torch.Tensor         # [head_dim/2]
    cfg: dict


def load_weights(model_id: str, device="cuda", dtype=torch.bfloat16) -> Weights:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device).eval()
    c = m.config
    hidden = c.hidden_size
    nkv = c.num_key_value_heads
    hd = hidden // c.num_attention_heads
    sd = m.model.state_dict()
    layers = []
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        layers.append({
            "ln1": sd[p + "input_layernorm.weight"],
            "wq": sd[p + "self_attn.q_proj.weight"], "bq": sd[p + "self_attn.q_proj.bias"],
            "wk": sd[p + "self_attn.k_proj.weight"], "bk": sd[p + "self_attn.k_proj.bias"],
            "wv": sd[p + "self_attn.v_proj.weight"], "bv": sd[p + "self_attn.v_proj.bias"],
            "wo": sd[p + "self_attn.o_proj.weight"],
            "ln2": sd[p + "post_attention_layernorm.weight"],
            "wgate": sd[p + "mlp.gate_proj.weight"],
            "wup": sd[p + "mlp.up_proj.weight"],
            "wdown": sd[p + "mlp.down_proj.weight"],
        })
    # inv_freq from the model's rotary embedding when present (exact convention),
    # else derived from rope_theta -- transformers moved rotary_emb across versions.
    inv_freq = None
    for obj in (getattr(m.model, "rotary_emb", None),
                getattr(m.model.layers[0].self_attn, "rotary_emb", None)):
        if obj is not None and hasattr(obj, "inv_freq"):
            inv_freq = obj.inv_freq.detach().to(device); break
    if inv_freq is None:
        theta = getattr(c, "rope_theta", 1000000.0)
        inv_freq = (1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd)))
    w = Weights(
        embed=m.model.embed_tokens.weight.detach(),
        layers=layers,
        final_norm=m.model.norm.weight.detach(),
        inv_freq=inv_freq,
        cfg=dict(hidden=hidden, n_heads=c.num_attention_heads, n_kv=nkv, head_dim=hd,
                 eps=c.rms_norm_eps, n_layers=c.num_hidden_layers),
    )
    del m
    torch.cuda.empty_cache()
    return w


def rms_norm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * weight.float()).to(x.dtype)


def rope_cos_sin(inv_freq, pos: int, dtype):
    ang = torch.arange(1, device=inv_freq.device).float() * 0 + pos  # scalar pos
    freqs = pos * inv_freq                                           # [hd/2]
    emb = torch.cat([freqs, freqs])                                  # [hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: [heads, head_dim]; rotate_half convention
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class KVCache:
    def __init__(self, w: Weights, max_len: int, device="cuda", dtype=torch.bfloat16):
        L, nkv, hd = w.cfg["n_layers"], w.cfg["n_kv"], w.cfg["head_dim"]
        self.k = torch.zeros(L, max_len, nkv, hd, device=device, dtype=dtype)
        self.v = torch.zeros(L, max_len, nkv, hd, device=device, dtype=dtype)
        self.len = 0


@torch.no_grad()
def decode_step(w: Weights, tok: int, kv: KVCache, pos: int) -> torch.Tensor:
    """One token in, logits [vocab] out. Writes K/V into the cache at `pos`."""
    cfg = w.cfg
    H, nh, nkv, hd, eps = cfg["hidden"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["eps"]
    group = nh // nkv
    h = w.embed[tok].clone()                                   # [H]
    cos, sin = rope_cos_sin(w.inv_freq, pos, h.dtype)
    scale = hd ** -0.5
    for i, ly in enumerate(w.layers):
        res = h
        x = rms_norm(h, ly["ln1"], eps)
        q = (x @ ly["wq"].T + ly["bq"]).view(nh, hd)
        k = (x @ ly["wk"].T + ly["bk"]).view(nkv, hd)
        v = (x @ ly["wv"].T + ly["bv"]).view(nkv, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        kv.k[i, pos] = k
        kv.v[i, pos] = v
        K = kv.k[i, : pos + 1]                                 # [T, nkv, hd]
        V = kv.v[i, : pos + 1]
        out = torch.empty(nh, hd, device=h.device, dtype=torch.float32)
        for hh in range(nh):
            kvh = hh // group
            scores = (q[hh].float() @ K[:, kvh].float().T) * scale   # [T]
            p = torch.softmax(scores, dim=-1)
            out[hh] = p @ V[:, kvh].float()
        attn = out.to(h.dtype).view(H)
        h = res + attn @ ly["wo"].T
        res = h
        x = rms_norm(h, ly["ln2"], eps)
        gate = x @ ly["wgate"].T
        up = x @ ly["wup"].T
        act = torch.nn.functional.silu(gate) * up
        h = res + act @ ly["wdown"].T
    h = rms_norm(h, w.final_norm, eps)
    return h @ w.embed.T                                       # tied lm_head -> [vocab]


@torch.no_grad()
def verify(model_id="Qwen/Qwen2.5-1.5B-Instruct", ctx=32, steps=8):
    """Prefill with HF, then check our decode_step logits match HF's, token by token."""
    from transformers import AutoModelForCausalLM

    w = load_weights(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    ids = torch.randint(0, cfg_vocab(hf), (1, ctx), device="cuda")

    # our reference: replay the prompt through decode_step to fill the cache
    kv = KVCache(w, ctx + steps + 1)
    for pos in range(ctx):
        logits_ours = decode_step(w, int(ids[0, pos]), kv, pos)
    # HF logits for the same full prefill
    hf_out = hf(ids, use_cache=True)
    logits_hf = hf_out.logits[0, -1].float()
    diff = (logits_ours.float() - logits_hf).abs()
    print(f"prefill last-token logit max|Δ| = {diff.max().item():.4f}  "
          f"argmax ours={int(logits_ours.argmax())} hf={int(logits_hf.argmax())}")

    # continue greedily and compare argmax each step
    past = hf_out.past_key_values
    match = 0
    tok = int(logits_hf.argmax())
    for s in range(steps):
        pos = ctx + s
        lo = decode_step(w, tok, kv, pos)
        hf_step = hf(torch.tensor([[tok]], device="cuda"), past_key_values=past, use_cache=True)
        past = hf_step.past_key_values
        lh = hf_step.logits[0, -1].float()
        a_o, a_h = int(lo.argmax()), int(lh.argmax())
        match += (a_o == a_h)
        tok = a_h
    print(f"greedy argmax agreement over {steps} steps: {match}/{steps}")


def cfg_vocab(m):
    return m.config.vocab_size


if __name__ == "__main__":
    verify()
