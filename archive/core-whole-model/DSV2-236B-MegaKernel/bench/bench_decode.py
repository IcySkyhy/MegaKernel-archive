"""Decode-latency benchmark: ms/token and tok/s for the megakernel at B=1.

    torchrun --nproc_per_node=8 bench/bench_decode.py --input_len 128 --output_len 1008
"""
import os, sys, time, json, argparse
import torch
import torch.distributed as dist

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv2mk.paths import model_snapshot


SNAP = model_snapshot()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_len", type=int, default=128,
                    help="prompt length (tokens) — only affects KV starting context")
    ap.add_argument("--output_len", type=int, default=64,
                    help="number of tokens to decode")
    ap.add_argument("--warmup", type=int, default=2, help="warmup iters")
    ap.add_argument("--iters", type=int, default=3, help="measurement iters")
    ap.add_argument("--output_json", type=str, default=None)
    args = ap.parse_args()

    assert "WORLD_SIZE" in os.environ, "Launch with torchrun --nproc_per_node=N"
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    include_layer0 = os.environ.get("INCLUDE_LAYER0", "1") == "1"

    if rank == 0:
        print(f"=== MK bench: TP=EP={ws}, input_len={args.input_len}, output_len={args.output_len}, "
              f"warmup={args.warmup}, iters={args.iters}, include_layer0={include_layer0} ===", flush=True)
        print(f"Free GPU mem: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB", flush=True)

    from dsv2mk.runtime.engine import DSv2InKernelTPEP
    t0 = time.time()
    engine = DSv2InKernelTPEP(
        SNAP, S_max=max(args.input_len + args.output_len + 16, 512),
        device=device,
        tp_size=ws, ep_size=ws,
        use_fp8_q1=True, use_fp8_q2=True, use_fp8_routed=True,
        skip_hf_load=True,
        include_layer0=include_layer0,
    )
    if rank == 0:
        print(f"[rank 0] Loaded in {time.time() - t0:.1f}s", flush=True)
        print(f"[rank 0] Free GPU mem now: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB", flush=True)

    torch.manual_seed(0)
    prompt_ids = torch.randint(1000, 50000, (args.input_len,), dtype=torch.long).tolist()
    if rank == 0:
        print(f"[rank 0] synthetic prompt of {len(prompt_ids)} tokens", flush=True)

    inline_path = bool(getattr(engine, "use_inline_lm_head", False))

    for w in range(args.warmup):
        engine.reset()
        last = None
        for tid in prompt_ids:
            last = engine.step(int(tid))
        if inline_path:
            for _ in range(min(args.output_len, 8)):
                engine.step(None)
        else:
            for _ in range(min(args.output_len, 8)):
                last = engine.step(int(last.argmax().item()))
        torch.cuda.synchronize()
        if rank == 0:
            print(f"[rank 0]   warmup {w+1}/{args.warmup} done", flush=True)

    results = []
    for it in range(args.iters):
        engine.reset()
        torch.cuda.synchronize()
        t_p = time.perf_counter()
        last = None
        for tid in prompt_ids:
            last = engine.step(int(tid))
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t_p
        decode_step_s = []
        for d in range(args.output_len):
            if inline_path:
                t_s = time.perf_counter()
                engine.step(None)
                _ = int(engine.next_token_buf.item())
                torch.cuda.synchronize()
            else:
                tid = int(last.argmax().item())
                t_s = time.perf_counter()
                last = engine.step(tid)
                torch.cuda.synchronize()
            decode_step_s.append(time.perf_counter() - t_s)
        decode_total_s = sum(decode_step_s)
        results.append({
            "iter": it,
            "prefill_s": prefill_s,
            "decode_total_s": decode_total_s,
            "decode_step_s": decode_step_s,
        })
        if rank == 0:
            mean_ms = 1000 * decode_total_s / len(decode_step_s)
            tps = len(decode_step_s) / decode_total_s
            print(f"[rank 0]   iter {it+1}/{args.iters}: prefill={prefill_s*1000:.1f}ms, "
                  f"decode={args.output_len}t in {decode_total_s*1000:.1f}ms = {mean_ms:.2f}ms/t = {tps:.2f} tok/s",
                  flush=True)

    if rank == 0:
        all_steps = [s for r in results for s in r["decode_step_s"]]
        mean_ms = 1000 * sum(all_steps) / len(all_steps)
        sorted_ms = sorted(s * 1000 for s in all_steps)
        p50 = sorted_ms[len(sorted_ms) // 2]
        p90 = sorted_ms[int(len(sorted_ms) * 0.9)]
        p99 = sorted_ms[int(len(sorted_ms) * 0.99)]
        tps_mean = 1000.0 / mean_ms
        summary = {
            "tp": ws, "ep": ws, "batch_size": 1,
            "input_len": args.input_len, "output_len": args.output_len,
            "include_layer0": include_layer0,
            "class_b_inline": inline_path,
            "iters": args.iters,
            "decode_mean_ms": mean_ms,
            "decode_p50_ms": p50,
            "decode_p90_ms": p90,
            "decode_p99_ms": p99,
            "decode_tok_per_s": tps_mean,
            "n_steps_total": len(all_steps),
            "per_iter": [{"prefill_s": r["prefill_s"],
                          "decode_total_s": r["decode_total_s"]}
                         for r in results],
        }
        print(f"\n=== SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"mean decode latency: {mean_ms:.2f} ms/tok  p50={p50:.2f}  p90={p90:.2f}  p99={p99:.2f}", flush=True)
        print(f"sustained throughput: {tps_mean:.2f} tok/s/sequence (B=1)", flush=True)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[rank 0] wrote {args.output_json}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

