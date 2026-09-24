"""Distributed tensor-parallel megakernel decode -- the real multi-GPU Part 3.

Run with:  torchrun --nproc_per_node=2 -m onelaunch.model_tp_dist --bench
Each rank loads only its shard, so it reads half the weights per step; the two
NCCL all-reduces per layer (o-proj, down-proj partials, H-wide) and the lm_head
all-gather are captured *inside* the CUDA graph, so a decode step is still one
replay. Target: batch-1 latency below the single-GPU 3.06 ms weight-bandwidth
floor, because each GPU now streams only ~1.5 GB.

Correctness of the sharding math is checked separately and cheaply on one GPU in
model_tp.py (verify_tp_sim); this file is about the multi-GPU latency number.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from .kernels import decode_attn, gemv, rmsnorm_gemv, rope_write, swiglu_gemv
from .model_fused import build_fused
from .model_ref import rms_norm, rope_cos_sin
from .model_tp import shard_weights


class TPDecoder:
    def __init__(self, w, shard, meta, rank, world, max_len, nocomm=False):
        self.w = w
        self.ly = shard
        self.m = meta
        self.rank, self.world = rank, world
        self.nocomm = nocomm
        nkv_r, hd = meta["nkv_r"], meta["hd"]
        L = meta["n_layers"]
        self.k = torch.zeros(L, max_len, nkv_r, hd, device="cuda", dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.tok = torch.zeros(1, dtype=torch.long, device="cuda")
        self.pos = torch.zeros(1, dtype=torch.long, device="cuda")
        V = w.embed.shape[0]
        self.Vr = V // world
        self.vsl = slice(rank * self.Vr, (rank + 1) * self.Vr)
        self.logit_part = torch.empty(self.Vr, device="cuda", dtype=torch.bfloat16)
        self.logits = torch.empty(world * self.Vr, device="cuda", dtype=torch.bfloat16)
        self.o_buf = torch.empty(meta["H"], device="cuda", dtype=torch.float32)
        self.d_buf = torch.empty(meta["H"], device="cuda", dtype=torch.float32)
        self.graph = None

    def _body(self):
        w, m = self.w, self.m
        nh_r, nkv_r, hd, H, group, eps = m["nh_r"], m["nkv_r"], m["hd"], m["H"], m["group"], m["eps"]
        scale = hd ** -0.5
        h = w.embed.index_select(0, self.tok).squeeze(0)
        freqs = self.pos.to(torch.float32) * w.inv_freq
        emb = torch.cat([freqs, freqs])
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)
        for i in range(m["n_layers"]):
            ly = self.ly[i]
            res = h
            qkv = rmsnorm_gemv(h, ly["ln1"], ly["wqkv"], ly["bqkv"], eps)
            q = rope_write(qkv, cos, sin, self.k[i], self.v[i], self.pos, nh_r, nkv_r, hd)
            attn = decode_attn(q, self.k[i], self.v[i], group, scale, cur_len=(self.pos + 1).to(torch.int32))
            self.o_buf.copy_(gemv(attn.reshape(nh_r * hd), ly["wo"]).float())
            if not self.nocomm:
                dist.all_reduce(self.o_buf)                                # sum o-proj partials
            h = res + self.o_buf.to(h.dtype)
            res = h
            act = swiglu_gemv(h, ly["ln2"], ly["wgate"], ly["wup"], eps)
            self.d_buf.copy_(gemv(act, ly["wdown"]).float())
            if not self.nocomm:
                dist.all_reduce(self.d_buf)                                # sum down-proj partials
            h = res + self.d_buf.to(h.dtype)
        h = rms_norm(h, w.final_norm, eps)
        self.logit_part.copy_(h @ w.embed[self.vsl].T)                     # column-parallel lm_head
        if not self.nocomm:
            dist.all_gather_into_tensor(self.logits, self.logit_part)

    def comm_bench(self, reps=200):
        """Eager latency of one decode step's collectives: 2*n_layers all_reduce(H)
        + 1 all_gather(V). Captures the comm cost the graph can't (NCCL won't
        capture on this stack), to add to the graphed no-comm compute time."""
        for _ in range(10):
            for _ in range(self.m["n_layers"]):
                dist.all_reduce(self.o_buf); dist.all_reduce(self.d_buf)
            dist.all_gather_into_tensor(self.logits, self.logit_part)
        dist.barrier(); torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(self.m["n_layers"]):
                dist.all_reduce(self.o_buf); dist.all_reduce(self.d_buf)
            dist.all_gather_into_tensor(self.logits, self.logit_part)
            en.record(); torch.cuda.synchronize()
            t.append(st.elapsed_time(en))
        t.sort()
        return t[len(t) // 2]

    def capture(self, prime_pos=8):
        self.tok.fill_(1); self.pos.fill_(prime_pos)
        init = dist.is_initialized()
        if init:
            dist.barrier()                       # align ranks before warmup
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(8):                   # warmup: autotune + prime NCCL comms
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        if init:
            dist.barrier()                       # both ranks enter capture together
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    def step(self, tok, pos):
        self.tok.fill_(tok); self.pos.fill_(pos)
        self.graph.replay()
        return self.logits


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eager", action="store_true",
                    help="time the eager _body loop instead of a captured graph "
                         "(NCCL collectives don't capture cleanly; eager keeps the "
                         "per-rank overhead identical across TP sizes for a fair ratio)")
    args = ap.parse_args()

    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    w = build_fused(args.model)
    shards, meta = shard_weights(w, world)
    # nocomm: graph-capture the half-weight compute WITHOUT collectives (NCCL won't
    # capture on this stack); comm is measured separately and added. TP=1 needs no
    # comm so it's a clean end-to-end graph number either way.
    nocomm = world > 1
    dec = TPDecoder(w, shards[rank], meta, rank, world, args.ctx + args.steps + 1, nocomm=nocomm)

    def run_eager(pos):
        dec.tok.fill_(1); dec.pos.fill_(pos)
        dec._body()

    if args.eager:
        for pos in range(8, 8 + 16):          # warmup (autotune + NCCL prime)
            run_eager(pos)
        dist.barrier(); torch.cuda.synchronize()
        t = []
        for s in range(args.steps):
            pos = 8 + s
            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
            st.record(); run_eager(pos); en.record(); torch.cuda.synchronize()
            t.append(st.elapsed_time(en))
        mode = "eager"
    else:
        dec.capture(prime_pos=8)
        tok = 1
        for pos in range(args.ctx):
            tok = int(dec.step(tok, pos).argmax())
        torch.cuda.synchronize()
        t = []
        for s in range(args.steps):
            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
            st.record(); dec.graph.replay(); en.record(); torch.cuda.synchronize()
            t.append(st.elapsed_time(en))
        mode = "cudagraph-nocomm" if nocomm else "cudagraph"
    t.sort()
    med = t[len(t) // 2]
    comm = dec.comm_bench() if (world > 1 and not args.eager) else 0.0
    if rank == 0:
        total = med + comm
        print(f"\n=== TP={world} fused megakernel ({mode}) ===")
        print(f"compute {med:.3f} ms  + comm {comm:.3f} ms  = {total:.3f} ms/token "
              f"({1e3/total:.0f} tok/s)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
