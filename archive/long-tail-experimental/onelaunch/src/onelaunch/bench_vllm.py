"""SOTA baseline: vLLM batch-1 decode latency, same config as bench_dense.py.

vLLM runs CUDA graphs + fused kernels + an optimized attention/paged-KV path,
so this is the honest "what a production engine delivers" point on the ladder.
Batch-1 single-stream decode latency is exactly the regime a megakernel is meant
to contest, so this is the number to beat -- or to report honestly if we don't.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=False,          # CUDA graphs on -- the real SOTA path
        gpu_memory_utilization=0.85,
        max_model_len=args.ctx + args.new + 16,
        disable_log_stats=True,
    )
    # a fixed ~ctx-token prompt (token ids -> text via a dummy; vLLM tokenizes)
    prompt = "the " * args.ctx
    sp_decode = SamplingParams(temperature=0.0, max_tokens=args.new, ignore_eos=True)
    sp_prefill = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    # warmup (graph capture)
    llm.generate([prompt], sp_decode, use_tqdm=False)

    def timed(sp):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        n = len(out[0].outputs[0].token_ids)
        return dt, n

    # isolate decode: (prefill+N) - (prefill+1), per remaining token
    dec = []
    for _ in range(args.reps):
        t_full, n_full = timed(sp_decode)
        t_pre, n_pre = timed(sp_prefill)
        dec.append((t_full - t_pre) / max(1, n_full - n_pre) * 1e3)
    dec.sort()
    med = dec[len(dec) // 2]
    print("\n=== vLLM SOTA baseline ===")
    print(f"model       {args.model}")
    print(f"context     {args.ctx}, decode {args.new}, batch 1, CUDA graphs on")
    print(f"vllm decode {med:.3f} ms/token   ({1e3/med:.0f} tok/s)")


if __name__ == "__main__":
    main()
