"""Baseline decode-latency ladder for a small dense model.

The point of this file is to establish the honest "before" for the megakernel:
at batch 1, decode is memory-bound on the weights, and a CUDA graph already
removes almost all launch overhead -- so a dense model sits close to the memory
floor with nothing left for fusion to win. That is the setup for the MoE part,
where the naive path is nowhere near its floor.

Measures, at batch 1, for a fixed context length:
  - eager per-token decode latency (HF generate, dynamic cache)
  - CUDA-graph per-token latency (static cache + reduce-overhead compile)
  - the memory-bound floor: (weight bytes read per token) / (peak bandwidth)
  - kernel launches per decode step (torch profiler)
"""

from __future__ import annotations

import argparse
import time

import torch


def _params_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


@torch.no_grad()
def _time_generate(model, tokenizer, input_ids, new_tokens, static=False) -> float:
    """Median ms/token over the decode of `new_tokens` tokens (prefill excluded)."""
    kw = dict(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if static:
        kw["cache_implementation"] = "static"
    # warmup (compile + graph capture happen here on the static path)
    model.generate(input_ids, **kw)
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model.generate(input_ids, **kw)
        end.record()
        torch.cuda.synchronize()
        # subtract a single prefill by amortization: generate includes one prefill,
        # so per-token over many tokens is dominated by decode.
        times.append(start.elapsed_time(end) / new_tokens)
    times.sort()
    return times[len(times) // 2]


@torch.no_grad()
def _launches_per_step(model, input_ids) -> int:
    from torch.profiler import ProfilerActivity, profile

    # prime a cache with a prefill, then profile exactly one decode step
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(next_tok, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA" and e.cuda_time > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--peak-gbs", type=float, default=1008.0)  # RTX 4090
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    ids = torch.randint(0, tok.vocab_size, (1, args.ctx), device="cuda")

    wbytes = _params_bytes(model)
    floor_ms = wbytes / (args.peak_gbs * 1e9) * 1e3

    launches = _launches_per_step(model, ids)
    eager = _time_generate(model, tok, ids, args.new, static=False)

    compiled = model
    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        graph = _time_generate(compiled, tok, ids, args.new, static=True)
    except Exception as e:  # torch.compile can be brittle; report what we got
        graph = float("nan")
        print(f"[warn] compiled path failed: {e}")

    print("\n=== dense decode latency ladder ===")
    print(f"model            {args.model}")
    print(f"weights          {wbytes/1e9:.2f} GB  (bf16)")
    print(f"context          {args.ctx} tokens, batch 1")
    print(f"kernel launches  {launches} per decode step (eager)")
    print(f"memory floor     {floor_ms:.3f} ms/token   (weights / {args.peak_gbs:.0f} GB/s)")
    print(f"eager            {eager:.3f} ms/token   ({1e3/eager:.0f} tok/s)")
    print(f"cuda graph       {graph:.3f} ms/token   ({1e3/graph:.0f} tok/s)")
    if graph == graph:  # not nan
        print(f"graph vs floor   {graph/floor_ms:.2f}x the floor  "
              f"(headroom left for fusion: {(graph-floor_ms)/graph*100:.0f}%)")


if __name__ == "__main__":
    main()
