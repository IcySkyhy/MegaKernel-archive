"""CUDA-graph-captured fused MoE decode for OLMoE-1B-7B.

Same idea as the dense Part 1 megakernel (model_graph.py) but for a sparse MoE:
the router picks 8 of 64 experts per token, and only those experts' weights are
read. The data-dependent expert ids live in a static GPU tensor (topi) that the
router overwrites each step, so the captured graph replays byte-identically while
routing to different experts. Everything -- QKV, QK-norm+RoPE+cache-write,
split-K attention, router, expert SwiGLU, gated expert-down + residual -- is a
Triton kernel; the only torch glue is the two QK-norm scale reductions and the
router softmax/top-k.
"""

from __future__ import annotations

import torch

from .kernels import decode_attn, gemv, rmsnorm_gemv
from .kernels_moe import expert_down, expert_swiglu, qknorm_rope_write
from .model_ref_moe import KVCacheMoE, load_weights_moe, weights_from_model
from .model_ref import rms_norm


def build_fused_moe(model_id="allenai/OLMoE-1B-7B-0924-Instruct", w=None):
    if w is None:
        w = load_weights_moe(model_id)
    for ly in w.layers:
        ly["wqkv"] = torch.cat([ly["wq"], ly["wk"], ly["wv"]], 0).contiguous()  # [3H, H]
    return w


def active_floor_ms(w, peak_gbs=1008.0):
    """Bytes actually read per decode step: attention + router + top-k experts + lm_head."""
    cfg = w.cfg
    H, top_k, I, E = cfg["hidden"], cfg["top_k"], cfg["inter"], cfg["n_experts"]
    per_layer = 4 * H * H + E * H + top_k * (2 * I * H + H * I)   # attn + router + active experts
    total = cfg["n_layers"] * per_layer + w.lm_head.shape[0] * H
    gb = total * 2 / 1e9
    return gb, gb / peak_gbs * 1e3


class GraphDecoderMoE:
    def __init__(self, w, max_len):
        self.w = w
        self.kv = KVCacheMoE(w, max_len)
        self.tok = torch.zeros(1, dtype=torch.long, device="cuda")
        self.pos = torch.zeros(1, dtype=torch.long, device="cuda")
        self.out = torch.empty(1, w.lm_head.shape[0], device="cuda", dtype=torch.bfloat16)
        self.graph = None

    def _body(self):
        w = self.w
        cfg = w.cfg
        H, nh, nkv, hd, eps = cfg["hidden"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["eps"]
        I, top_k = cfg["inter"], cfg["top_k"]
        group, scale = nh // nkv, hd ** -0.5
        h = w.embed.index_select(0, self.tok).squeeze(0)
        freqs = self.pos.to(torch.float32) * w.inv_freq
        emb = torch.cat([freqs, freqs])
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)
        cur_len = (self.pos + 1).to(torch.int32)
        for i, ly in enumerate(w.layers):
            res = h
            qkv = rmsnorm_gemv(h, ly["ln1"], ly["wqkv"], None, eps)          # [3H]
            qf, kf = qkv[:H].float(), qkv[H:2 * H].float()
            qscale = torch.rsqrt(qf.pow(2).mean() + eps).reshape(1)          # QK-norm scales
            kscale = torch.rsqrt(kf.pow(2).mean() + eps).reshape(1)
            q = qknorm_rope_write(qkv, ly["qn"], ly["kn"], qscale, kscale, cos, sin,
                                  self.kv.k[i], self.kv.v[i], self.pos, nh, nkv, hd)
            attn = decode_attn(q, self.kv.k[i], self.kv.v[i], group, scale, cur_len=cur_len)
            h = gemv(attn.reshape(H), ly["wo"], residual=res)
            res = h
            x = rms_norm(h, ly["ln2"], eps)                                  # shared: router + experts
            logits = gemv(x, ly["router"]).float()                          # [E]
            probs = torch.softmax(logits, dim=-1)
            topv, topi = probs.topk(top_k)                                   # raw gates (no renorm)
            topi = topi.to(torch.int32)
            act = expert_swiglu(x, ly["gate_up"], topi, top_k, I)
            h = expert_down(act, ly["down"], topi, topv, res, top_k)
        h = rms_norm(h, w.final_norm, eps)
        torch.matmul(h.unsqueeze(0), w.lm_head.T, out=self.out)

    def capture(self, prime_pos=8):
        self.tok.fill_(1)
        self.pos.fill_(prime_pos)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    def step(self, tok, pos):
        self.tok.fill_(tok)
        self.pos.fill_(pos)
        self.graph.replay()
        return self.out


@torch.no_grad()
def verify_graph(model_id="allenai/OLMoE-1B-7B-0924-Instruct", ctx=32, steps=12):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .model_ref_moe import decode_step_moe

    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    w = build_fused_moe(w=weights_from_model(hf))
    tk = AutoTokenizer.from_pretrained(model_id)
    ids = tk("The capital of France is Paris. The largest planet in the solar system is",
             return_tensors="pt").input_ids.cuda()[0][:ctx]
    ctx = ids.shape[0]

    dec = GraphDecoderMoE(w, ctx + steps + 1)
    dec.capture(prime_pos=0)
    kv_ref = KVCacheMoE(w, ctx + steps + 1)
    for pos in range(ctx):
        lr = decode_step_moe(w, int(ids[pos]), kv_ref, pos)
        lg = dec.step(int(ids[pos]), pos)
    d = (lr.float() - lg.float()).abs()
    print(f"prefill max|Δ vs ref| = {d.max().item():.4f}  argmax ref={int(lr.argmax())} graph={int(lg.argmax())}")

    tok = int(lr.argmax())
    match = 0
    for s in range(steps):
        pos = ctx + s
        lr = decode_step_moe(w, tok, kv_ref, pos)
        lg = dec.step(tok, pos)
        match += int(lr.argmax()) == int(lg.argmax())
        tok = int(lr.argmax())
    print(f"greedy argmax agreement (graph vs ref) over {steps} steps: {match}/{steps}")


@torch.no_grad()
def bench_graph(model_id="allenai/OLMoE-1B-7B-0924-Instruct", ctx=512, steps=128):
    w = build_fused_moe(model_id)
    gb, floor = active_floor_ms(w)
    dec = GraphDecoderMoE(w, ctx + steps + 1)
    dec.capture(prime_pos=8)
    tok = 1
    for pos in range(ctx):
        tok = int(dec.step(tok, pos).argmax())
    torch.cuda.synchronize()
    t = []
    for s in range(steps):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); dec.graph.replay(); en.record(); torch.cuda.synchronize()
        t.append(st.elapsed_time(en))
    t.sort()
    med = t[len(t) // 2]
    print(f"\n=== fused MoE megakernel (CUDA graph) ===")
    print(f"model OLMoE-1B-7B  active weights {gb:.2f} GB/step  floor {floor:.3f} ms")
    print(f"decode {med:.3f} ms/token   ({1e3/med:.0f} tok/s)   {med/floor:.2f}x active floor")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        bench_graph()
    else:
        verify_graph()
