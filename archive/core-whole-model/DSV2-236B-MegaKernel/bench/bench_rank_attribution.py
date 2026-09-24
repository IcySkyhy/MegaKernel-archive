"""Per-stage, per-rank attribution: which of the 8 ranks is the straggler.

Reports min/median/max across ranks for every stage in `STAGE_DEFS`, separating
compute stages from the TP/EP collectives, which is how cross-rank skew is
distinguished from genuine stage cost.
"""
import os, sys, time, json, argparse
import torch
import torch.distributed as dist

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv2mk.paths import model_snapshot


SNAP = model_snapshot()


STAGE_DEFS = [
    ("A+B (input_norm + qkva)",         0,  1,  "attn"),
    ("  A+B work",                      0,  35, "attn"),
    ("  A+B sync1",                     35, 1,  "sync"),
    ("C+D+E+F (q_a/q_b/kv_a/k_pe)",     1,  2,  "attn"),
    ("  C-F work",                      1,  36, "attn"),
    ("  C-F sync2",                     36, 2,  "sync"),
    ("G+H (q_pe + W_UK)",               2,  3,  "attn"),
    ("  G+H work",                      2,  37, "attn"),
    ("  G+H sync3",                     37, 3,  "sync"),
    ("I-split (MLA partials)",          3,  4,  "attn"),
    ("  I.setup (7 pipelines)",         3,  32, "attn"),
    ("  I.qpack (Q->smem)",             32, 33, "attn"),
    ("  I.mma (QK+softmax+PV+wr)",      33, 34, "attn"),
    ("  I.gridsync4",                   34, 4,  "attn"),
    ("I-combine (partial reduce)",      4,  5,  "attn"),
    ("  I-comb work",                   4,  38, "attn"),
    ("  I-comb sync5",                  38, 5,  "sync"),
    ("J (o_per_head)",                  5,  6,  "attn"),
    ("  J work",                        5,  39, "attn"),
    ("  J sync6",                       39, 6,  "sync"),
    ("K (attn_proj + TP signal)",       6,  7,  "attn"),
    ("  K work",                        6,  40, "attn"),
    ("  K sync7(overlap)",              40, 7,  "sync"),
    ("TP barrier",                      7,  26, "moe_collective"),
    ("Stage L (TP all-reduce consume)", 26, 27, "moe_collective"),
    ("Stage M+N (norm + router)",       27, 8,  "moe_collective"),
    ("Stage O (top-K + zero MoE)",      8,  9,  "moe_serial_compute"),
    ("Q1 (shared gate+up+SiLU)",        22, 23, "shared_expert"),
    ("Q2 (shared down)",                23, 10, "shared_expert"),
    ("EP barrier",                      10, 24, "moe_collective"),
    ("Stage R (distributed ld_reduce + bcast)", 24, 25, "moe_collective"),
]


def delta_us_per_layer(stage_ts_per_rank, slot_a, slot_b, cyc_to_us):
    a = stage_ts_per_rank[..., slot_a].to(torch.int64)
    b = stage_ts_per_rank[..., slot_b].to(torch.int64)
    d = b - a
    d = torch.where(d < 0, d + (1 << 32), d)
    return d.to(torch.float64) * cyc_to_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n_steps", type=int, default=8)
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

    sm_clock_hz = torch.cuda.get_device_properties(rank).clock_rate * 1_000
    cyc_to_us = 1e6 / sm_clock_hz
    if rank == 0:
        print(f"=== Per-stage per-rank attribution: TP=EP={ws}, SM clock = {sm_clock_hz/1e9:.3f} GHz ===", flush=True)

    from dsv2mk.runtime.engine import DSv2InKernelTPEP
    from dsv2mk.kernels.megakernel import get_last_stage_ts

    t0 = time.time()
    engine = DSv2InKernelTPEP(
        SNAP, S_max=max(args.input_len + 64 + 16, 512),
        device=device,
        tp_size=ws, ep_size=ws,
        use_fp8_q1=True, use_fp8_q2=True, use_fp8_routed=True,
        skip_hf_load=True,
        include_layer0=include_layer0,
    )
    if rank == 0:
        print(f"[rank 0] Loaded in {time.time() - t0:.1f}s", flush=True)

    torch.manual_seed(0)
    prompt_ids = torch.randint(1000, 50000, (args.input_len,), dtype=torch.long).tolist()
    inline_path = bool(getattr(engine, "use_inline_lm_head", False))

    for w in range(args.warmup):
        engine.reset()
        for tid in prompt_ids:
            engine.step(int(tid))
        for _ in range(4):
            engine.step(None) if inline_path else engine.step(int(prompt_ids[-1]))
        torch.cuda.synchronize()
        if rank == 0:
            print(f"[rank 0]   warmup {w+1}/{args.warmup} done", flush=True)

    engine.reset()
    for tid in prompt_ids:
        engine.step(int(tid))
    torch.cuda.synchronize()

    K_topk = engine.cfg.num_experts_per_tok
    all_steps_stage_ts_gathered = []

    for s in range(args.n_steps):
        engine.step(None) if inline_path else engine.step(int(prompt_ids[-1]))
        torch.cuda.synchronize()
        stage_ts = get_last_stage_ts()
        assert stage_ts is not None
        stage_ts_cuda = stage_ts.to(torch.int64).contiguous()
        L, NS = stage_ts_cuda.shape

        gather_buf = [torch.zeros(L, NS, dtype=torch.int64, device=device) for _ in range(ws)]
        dist.all_gather(gather_buf, stage_ts_cuda)
        stage_ts_per_rank = torch.stack(gather_buf, dim=0).cpu()
        all_steps_stage_ts_gathered.append(stage_ts_per_rank)

        if rank == 0:
            print(f"[rank 0]   step {s+1}/{args.n_steps}: gathered", flush=True)

    if rank == 0:
        stage_ts_all = torch.stack(all_steps_stage_ts_gathered, dim=0)
        S, WS, L, NS = stage_ts_all.shape
        print(f"\n=== Per-stage per-rank duration analysis ({S} steps × {L} layers × {WS} ranks) ===\n", flush=True)

        print(f"{'stage':<35} {'cat':<22} {'min_rank_us':>13} {'med_us':>10} {'max_rank_us':>13} {'r4_us':>10} {'r4-med':>10} {'lag_class':<8}", flush=True)

        attribution_summary = {}
        per_stage_results = {}

        for label, slot_a, slot_b, cat in STAGE_DEFS:
            try:
                d = delta_us_per_layer(stage_ts_all, slot_a, slot_b, cyc_to_us)
            except IndexError:
                continue
            per_rank_mean = d.permute(1, 0, 2).reshape(WS, S * L).mean(dim=1)
            r4 = per_rank_mean[4].item()
            r7 = per_rank_mean[7].item()
            others = torch.cat([per_rank_mean[:4], per_rank_mean[5:7]]).tolist()
            med_others = sorted(others)[len(others)//2]
            min_other = min(others)
            max_other = max(others)
            r4_lag = r4 - med_others

            if r4 > 0 and med_others > 0:
                rel = r4 / med_others
                if rel > 1.2:
                    lag_class = "++"
                elif rel > 1.1:
                    lag_class = "+"
                elif rel < 0.5:
                    lag_class = "--"
                elif rel < 0.9:
                    lag_class = "-"
                else:
                    lag_class = "~"
            else:
                lag_class = "?"

            per_stage_results[label] = {
                "category": cat,
                "per_rank_mean_us": per_rank_mean.tolist(),
                "median_of_others_us": med_others,
                "rank4_us": r4,
                "rank4_lag_vs_median_us": r4_lag,
                "lag_class": lag_class,
            }

            print(f"{label:<35} {cat:<22} {min_other:>13.2f} {med_others:>10.2f} {max_other:>13.2f} {r4:>10.2f} {r4_lag:>+10.2f} {lag_class:<8}", flush=True)

        p1_total = delta_us_per_layer(stage_ts_all, 9, 16, cyc_to_us)
        p2_total = delta_us_per_layer(stage_ts_all, 16, 22, cyc_to_us)

        for label, d in [("P1 routed experts (6 total)", p1_total),
                          ("P2 routed experts (6 total)", p2_total)]:
            per_rank_mean = d.permute(1, 0, 2).reshape(WS, S * L).mean(dim=1)
            r4 = per_rank_mean[4].item()
            others = torch.cat([per_rank_mean[:4], per_rank_mean[5:7]]).tolist()
            med_others = sorted(others)[len(others)//2]
            min_other = min(others)
            max_other = max(others)
            r4_lag = r4 - med_others
            rel = r4 / med_others if med_others > 0 else 1.0
            if rel > 1.2:
                lag_class = "++"
            elif rel > 1.1:
                lag_class = "+"
            elif rel < 0.5:
                lag_class = "--"
            elif rel < 0.9:
                lag_class = "-"
            else:
                lag_class = "~"
            per_stage_results[label] = {
                "category": "moe_serial_compute",
                "per_rank_mean_us": per_rank_mean.tolist(),
                "median_of_others_us": med_others,
                "rank4_us": r4,
                "rank4_lag_vs_median_us": r4_lag,
                "lag_class": lag_class,
            }
            print(f"{label:<35} {'moe_serial_compute':<22} {min_other:>13.2f} {med_others:>10.2f} {max_other:>13.2f} {r4:>10.2f} {r4_lag:>+10.2f} {lag_class:<8}", flush=True)

        print(f"\n=== ATTRIBUTION ===", flush=True)
        cat_lag = {}
        for label, info in per_stage_results.items():
            cat = info["category"]
            cat_lag.setdefault(cat, []).append((label, info["rank4_lag_vs_median_us"], info["lag_class"]))

        for cat, items in cat_lag.items():
            total_lag = sum(x[1] for x in items)
            print(f"\n  [{cat}] rank 4 cumulative lag = {total_lag:+.2f} µs/layer", flush=True)
            for label, lag, lag_class in items:
                if lag_class in ("+", "++"):
                    print(f"     {lag_class:<3} {label:<35} {lag:+.2f} µs/layer", flush=True)

        attn_lag_us = sum(x[1] for x in cat_lag.get("attn", []))
        moe_serial_lag_us = sum(x[1] for x in cat_lag.get("moe_serial_compute", []))
        moe_collective_lag_us = sum(x[1] for x in cat_lag.get("moe_collective", []))
        shared_lag_us = sum(x[1] for x in cat_lag.get("shared_expert", []))

        print(f"\n  Attn cumulative lag        : {attn_lag_us:+10.2f} µs/layer", flush=True)
        print(f"  MoE serial (P1+P2+O) lag   : {moe_serial_lag_us:+10.2f} µs/layer", flush=True)
        print(f"  MoE collective (barriers)  : {moe_collective_lag_us:+10.2f} µs/layer", flush=True)
        print(f"  Shared expert (Q1+Q2)      : {shared_lag_us:+10.2f} µs/layer", flush=True)
        print(f"  TOTAL ATTRIBUTABLE LAG     : {attn_lag_us + moe_serial_lag_us + moe_collective_lag_us + shared_lag_us:+10.2f} µs/layer", flush=True)

        print(f"\n=== DECISION ===", flush=True)
        if abs(moe_serial_lag_us) > 5 and abs(attn_lag_us) < 5:
            print(f"  → ROUTED EXPERT IMBALANCE dominates. Fix = expert-id reshuffle "
                  f"(2-3 wks) or MoE pipelining (3-4 wks).", flush=True)
        elif abs(attn_lag_us) > 5 and abs(moe_serial_lag_us) > 5:
            print(f"  → UNIFORM SILICON BINNING (both attn and MoE affected). "
                  f"Fix = different pod (1 day).", flush=True)
        elif abs(attn_lag_us) > 5 and abs(moe_serial_lag_us) < 5:
            print(f"  → ATTN-ONLY lag — unusual. Could be NUMA/topology. "
                  f"Try different rank-to-GPU pinning.", flush=True)
        elif abs(shared_lag_us) > 5:
            print(f"  → SHARED EXPERT (Q1/Q2) lag — unusual. Investigate FP8 GEMM "
                  f"on rank 4 specifically.", flush=True)
        else:
            print(f"  → MIXED or BELOW THRESHOLD — review table manually.", flush=True)

        if args.output_json:
            out = {
                "tp": ws, "ep": ws, "L": L, "n_steps": S, "K_topk": K_topk,
                "sm_clock_hz": sm_clock_hz,
                "per_stage_results": per_stage_results,
                "category_cumulative_lag_us_per_layer": {
                    "attn": attn_lag_us,
                    "moe_serial_compute": moe_serial_lag_us,
                    "moe_collective": moe_collective_lag_us,
                    "shared_expert": shared_lag_us,
                },
            }
            with open(args.output_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n[rank 0] wrote {args.output_json}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

