"""Hand-rolled batch-1 decode for OLMoE-1B-7B, in plain PyTorch.

Correctness oracle for the fused MoE megakernel. OLMoE specifics vs a dense
model: (1) QK-norm -- an RMSNorm over the full q and k projections before RoPE;
(2) a top-8-of-64 router with raw softmax gates (norm_topk_prob=False, so the
top-8 probabilities are used un-renormalized); (3) experts stored fused as
gate_up_proj [E, 2I, H] and down_proj [E, H, I]; (4) untied lm_head. At batch 1
only the 8 selected experts per layer are ever read -- that is the whole point:
the memory floor is the *active* weights, not the 7B total.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model_ref import apply_rope, rms_norm, rope_cos_sin


@dataclass
class MoEWeights:
    embed: torch.Tensor
    lm_head: torch.Tensor
    layers: list
    final_norm: torch.Tensor
    inv_freq: torch.Tensor
    cfg: dict


def weights_from_model(m):
    """Build MoEWeights referencing an already-loaded HF model's tensors (no copy)."""
    c = m.config
    hidden = c.hidden_size
    hd = hidden // c.num_attention_heads
    sd = m.model.state_dict()
    layers = []
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        layers.append({
            "ln1": sd[p + "input_layernorm.weight"],
            "wq": sd[p + "self_attn.q_proj.weight"], "wk": sd[p + "self_attn.k_proj.weight"],
            "wv": sd[p + "self_attn.v_proj.weight"], "wo": sd[p + "self_attn.o_proj.weight"],
            "qn": sd[p + "self_attn.q_norm.weight"], "kn": sd[p + "self_attn.k_norm.weight"],
            "ln2": sd[p + "post_attention_layernorm.weight"],
            "router": sd[p + "mlp.gate.weight"],                     # [E, H]
            "gate_up": sd[p + "mlp.experts.gate_up_proj"],           # [E, 2I, H]
            "down": sd[p + "mlp.experts.down_proj"],                 # [E, H, I]
        })
    w = MoEWeights(
        embed=m.model.embed_tokens.weight.detach(),
        lm_head=m.lm_head.weight.detach(),
        layers=layers,
        final_norm=m.model.norm.weight.detach(),
        inv_freq=m.model.rotary_emb.inv_freq.detach().to(m.model.embed_tokens.weight.device),
        cfg=dict(hidden=hidden, n_heads=c.num_attention_heads, n_kv=c.num_key_value_heads,
                 head_dim=hd, eps=c.rms_norm_eps, n_layers=c.num_hidden_layers,
                 n_experts=c.num_experts, top_k=c.num_experts_per_tok, inter=c.intermediate_size),
    )
    return w


def load_weights_moe(model_id="allenai/OLMoE-1B-7B-0924-Instruct", device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device).eval()
    return weights_from_model(m)


class KVCacheMoE:
    def __init__(self, w, max_len, device="cuda", dtype=torch.bfloat16):
        L, nkv, hd = w.cfg["n_layers"], w.cfg["n_kv"], w.cfg["head_dim"]
        self.k = torch.zeros(L, max_len, nkv, hd, device=device, dtype=dtype)
        self.v = torch.zeros(L, max_len, nkv, hd, device=device, dtype=dtype)


@torch.no_grad()
def decode_step_moe(w, tok, kv, pos):
    cfg = w.cfg
    H, nh, nkv, hd, eps = cfg["hidden"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["eps"]
    I, top_k = cfg["inter"], cfg["top_k"]
    group, scale = nh // nkv, hd ** -0.5
    h = w.embed[tok].clone()
    cos, sin = rope_cos_sin(w.inv_freq, pos, h.dtype)
    for i, ly in enumerate(w.layers):
        res = h
        x = rms_norm(h, ly["ln1"], eps)
        q = rms_norm(x @ ly["wq"].T, ly["qn"], eps).view(nh, hd)     # QK-norm over full 2048
        k = rms_norm(x @ ly["wk"].T, ly["kn"], eps).view(nkv, hd)
        v = (x @ ly["wv"].T).view(nkv, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        kv.k[i, pos] = k
        kv.v[i, pos] = v
        K, V = kv.k[i, : pos + 1], kv.v[i, : pos + 1]
        out = torch.empty(nh, hd, device=h.device, dtype=torch.float32)
        for hh in range(nh):
            kvh = hh // group
            scores = (q[hh].float() @ K[:, kvh].float().T) * scale
            p = torch.softmax(scores, dim=-1)
            out[hh] = p @ V[:, kvh].float()
        h = res + (out.to(h.dtype).view(H) @ ly["wo"].T)
        res = h
        x = rms_norm(h, ly["ln2"], eps)
        # router: top-8 of 64, raw softmax gates (norm_topk_prob=False)
        logits = x @ ly["router"].T                                 # [E]
        probs = torch.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(top_k)
        moe = torch.zeros(H, device=h.device, dtype=torch.float32)
        for j in range(top_k):
            e = int(topi[j])
            gu = ly["gate_up"][e] @ x                               # [2I]
            gate, up = gu[:I], gu[I:]
            act = torch.nn.functional.silu(gate) * up               # [I]
            moe += topv[j] * (ly["down"][e] @ act).float()
        h = res + moe.to(h.dtype)
    h = rms_norm(h, w.final_norm, eps)
    return h @ w.lm_head.T


@torch.no_grad()
def verify(model_id="allenai/OLMoE-1B-7B-0924-Instruct", ctx=32, steps=8):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    w = weights_from_model(hf)                       # share storage -- one 14GB copy
    ids = torch.randint(0, hf.config.vocab_size, (1, ctx), device="cuda")

    kv = KVCacheMoE(w, ctx + steps + 1)
    for pos in range(ctx):
        lo = decode_step_moe(w, int(ids[0, pos]), kv, pos)
    hf_out = hf(ids, use_cache=True)
    lh = hf_out.logits[0, -1].float()
    d = (lo.float() - lh).abs()
    print(f"prefill last-token logit max|Δ| = {d.max().item():.4f}  "
          f"argmax ours={int(lo.argmax())} hf={int(lh.argmax())}")

    past = hf_out.past_key_values
    tok = int(lh.argmax())
    match = 0
    for s in range(steps):
        pos = ctx + s
        lo = decode_step_moe(w, tok, kv, pos)
        hs = hf(torch.tensor([[tok]], device="cuda"), past_key_values=past, use_cache=True)
        past = hs.past_key_values
        lh = hs.logits[0, -1].float()
        match += int(lo.argmax()) == int(lh.argmax())
        tok = int(lh.argmax())
    print(f"greedy argmax agreement over {steps} steps: {match}/{steps}")


if __name__ == "__main__":
    verify()
