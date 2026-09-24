"""CUDA-graph-captured fused decode -- the real Part 1 number.

The eager fused step is already memory-optimal per kernel, but at batch 1 each
of the ~170 launches/step costs more in CPU launch latency than the kernel
spends moving bytes. Capturing the whole step into one CUDA graph collapses all
of it into a single replay: the CPU issues one call, the GPU walks the recorded
launch list back-to-back. Everything that changes per step (the input token, the
position, the KV write slot, the attention length) lives in static GPU tensors
the graph reads on replay, so the same recording serves every decode step as the
sequence grows.
"""

from __future__ import annotations

import torch

from .kernels import decode_attn, gemv, rmsnorm_gemv, rope_write, swiglu_gemv
from .model_fused import build_fused
from .model_ref import KVCache, rms_norm


class GraphDecoder:
    def __init__(self, w, max_len: int):
        self.w = w
        self.max_len = max_len
        self.kv = KVCache(w, max_len)
        self.tok = torch.zeros(1, dtype=torch.long, device="cuda")
        self.pos = torch.zeros(1, dtype=torch.long, device="cuda")
        self.out = torch.empty(1, w.embed.shape[0], device="cuda", dtype=torch.bfloat16)
        self.graph = None

    def _body(self):
        w = self.w
        cfg = w.cfg
        H, nh, nkv, hd, eps = cfg["hidden"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["eps"]
        group, scale = nh // nkv, hd ** -0.5
        qd, kd = nh * hd, nkv * hd
        h = w.embed.index_select(0, self.tok).squeeze(0)            # [H]
        freqs = self.pos.to(torch.float32) * w.inv_freq             # [hd/2]
        emb = torch.cat([freqs, freqs])
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)
        cur_len = (self.pos + 1).to(torch.int32)                   # attention length
        for i, ly in enumerate(w.layers):
            res = h
            qkv = rmsnorm_gemv(h, ly["ln1"], ly["wqkv"], ly["bqkv"], eps)
            q = rope_write(qkv, cos, sin, self.kv.k[i], self.kv.v[i], self.pos, nh, nkv, hd)
            attn = decode_attn(q, self.kv.k[i], self.kv.v[i], group, scale, cur_len=cur_len)
            h = gemv(attn.reshape(H), ly["wo"], residual=res)
            res = h
            act = swiglu_gemv(h, ly["ln2"], ly["wgate"], ly["wup"], eps)
            h = gemv(act, ly["wdown"], residual=res)
        h = rms_norm(h, w.final_norm, eps)
        torch.matmul(h.unsqueeze(0), w.embed.T, out=self.out)

    def capture(self, prime_pos: int = 8):
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

    def step(self, tok: int, pos: int) -> torch.Tensor:
        self.tok.fill_(tok)
        self.pos.fill_(pos)
        self.graph.replay()
        return self.out


@torch.no_grad()
def verify_graph(model_id="Qwen/Qwen2.5-1.5B-Instruct", ctx=32, steps=8):
    """Graph decode vs the plain-torch oracle: greedy argmax must agree."""
    from .model_ref import decode_step as ref_step

    w = build_fused(model_id)
    dec = GraphDecoder(w, ctx + steps + 1)
    dec.capture(prime_pos=0)

    kv_ref = KVCache(w, ctx + steps + 1)
    ids = torch.randint(0, w.embed.shape[0], (ctx,), device="cuda")
    for pos in range(ctx):
        lr = ref_step(w, int(ids[pos]), kv_ref, pos)
        lg = dec.step(int(ids[pos]), pos)
    d = (lr.float() - lg.float()).abs()
    print(f"prefill last-token max|Δ vs ref| = {d.max().item():.4f}   "
          f"argmax ref={int(lr.argmax())} graph={int(lg.argmax())}")

    tok = int(lr.argmax())
    match = 0
    for s in range(steps):
        pos = ctx + s
        lr = ref_step(w, tok, kv_ref, pos)
        lg = dec.step(tok, pos)
        match += int(lr.argmax()) == int(lg.argmax())
        tok = int(lr.argmax())
    print(f"greedy argmax agreement (graph vs ref) over {steps} steps: {match}/{steps}")


@torch.no_grad()
def bench_graph(model_id="Qwen/Qwen2.5-1.5B-Instruct", ctx=512, steps=128):
    w = build_fused(model_id)
    dec = GraphDecoder(w, ctx + steps + 1)
    dec.capture(prime_pos=8)
    # prime the cache with a fake prefill (values irrelevant to latency)
    tok = 1
    for pos in range(ctx):
        tok = int(dec.step(tok, pos).argmax())
    torch.cuda.synchronize()
    t = []
    for s in range(steps):
        pos = ctx + s
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        dec.graph.replay()
        en.record(); torch.cuda.synchronize()
        t.append(st.elapsed_time(en))
    t.sort()
    med = t[len(t) // 2]
    floor = 3.063
    print(f"\n=== fused megakernel (CUDA graph) ===")
    print(f"decode {med:.3f} ms/token   ({1e3/med:.0f} tok/s)")
    print(f"vs floor {floor} ms ({med/floor:.2f}x floor) | vs vLLM 4.414 ms "
          f"({4.414/med:.2f}x {'faster' if med < 4.414 else 'slower'})")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        bench_graph()
    else:
        verify_graph()
