"""End-to-end smoke test: greedy-generate 16 tokens and print the text."""
import os, sys, time
import torch
import torch.distributed as dist

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv2mk.paths import model_snapshot


SNAP = model_snapshot()


def main():
    assert "WORLD_SIZE" in os.environ, "Launch with torchrun --nproc_per_node=N"
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    include_layer0 = os.environ.get("INCLUDE_LAYER0", "0") == "1"

    if rank == 0:
        print(f"=== DSv2-236B in-kernel TP={ws} EP={ws} FP8 (include_layer0={include_layer0}) ===")
        print(f"Snapshot: {SNAP}")
        print(f"Free GPU mem at start: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    from dsv2mk.runtime.engine import DSv2InKernelTPEP
    t0 = time.time()
    engine = DSv2InKernelTPEP(
        SNAP, S_max=512, device=device,
        tp_size=ws, ep_size=ws,
        use_fp8_q1=True, use_fp8_q2=True, use_fp8_routed=True,
        skip_hf_load=True,
        include_layer0=include_layer0,
    )
    t_load = time.time() - t0
    if rank == 0:
        print(f"[rank 0] Loaded in {t_load:.1f}s; "
              f"Free GPU mem now: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    tok = engine.tokenizer
    prompt = "Hello, my name is"
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    max_new = 16

    if rank == 0:
        print(f"[rank 0] prompt='{prompt}' prompt_ids={prompt_ids} max_new={max_new}")

    t1 = time.time()
    out_ids = engine.generate(prompt_ids, max_new_tokens=max_new, greedy=True)
    t_gen = time.time() - t1

    if rank == 0:
        out_text = tok.decode(out_ids)
        print(f"[rank 0] DSv2-236B generated {len(out_ids) - len(prompt_ids)} tokens "
              f"in {t_gen:.2f}s ({(len(out_ids) - len(prompt_ids)) / max(t_gen, 1e-6):.2f} tok/s)")
        print(f"[rank 0] tokens: {out_ids}")
        print(f"[rank 0] text:   {out_text!r}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

