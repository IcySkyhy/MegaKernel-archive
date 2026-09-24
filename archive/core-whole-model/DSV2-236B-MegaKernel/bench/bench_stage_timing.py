"""Per-layer stage breakdown from the kernel's in-place clock64 timestamps.

Attributes each layer's microseconds to the stages listed in `stage_breakdown`,
so a change can be located rather than merely observed at the wall.
"""
import os, sys, time, json, argparse
import torch
import torch.distributed as dist

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv2mk.paths import model_snapshot


SNAP = model_snapshot()


def stage_breakdown(stage_ts_cpu, sm_clock_hz, K_topk=6):
    L = stage_ts_cpu.shape[0]
    cyc_to_us = 1e6 / sm_clock_hz

    def delta_us(a, b, l):
        d = int(stage_ts_cpu[l, b]) - int(stage_ts_cpu[l, a])
        if d < 0:
            d += (1 << 32)
        return d * cyc_to_us

    fixed_stages = [
        ("A+B   (input_norm, qkva)",                  0, 1),
        ("C+D+E+F (q_a_norm, q_b, kv_a_norm, k_pe)",  1, 2),
        ("G+H   (q_pe RoPE, W_UK)",                   2, 3),
        ("I-split (MLA attn partials)",               3, 4),
        ("I-combine (partial reduce)",                4, 5),
        ("J     (o_per_head)",                        5, 6),
        ("K + TP_signal_setup (attn_proj+grid_sync)",  6, 7),
        ("O     (top-K + zero MoE)",                   8, 9),
    ]
    totals = {label: 0.0 for label, _, _ in fixed_stages}
    for label, a, b in fixed_stages:
        for l in range(L):
            totals[label] += delta_us(a, b, l)

    has_lmn_bisect = stage_ts_cpu.shape[1] >= 28 and bool((stage_ts_cpu[:, 26] != 0).any())
    has_rs_bisect = (
        stage_ts_cpu.shape[1] >= 30
        and bool((stage_ts_cpu[:, 28] != 0).any())
        and bool((stage_ts_cpu[:, 29] != 0).any())
    )
    has_spin_diag = (
        stage_ts_cpu.shape[1] >= 32
        and bool((stage_ts_cpu[:, 30] != 0).any())
        and bool((stage_ts_cpu[:, 31] != 0).any())
    )
    if has_lmn_bisect:
        tp_bar_total = 0.0
        stage_l_total = 0.0
        stage_mn_total = 0.0
        stage_l1_total = 0.0
        stage_l_sync_total = 0.0
        stage_l2_total = 0.0
        stage_r0_total = 0.0
        spin_pre_total = 0.0
        spin_body_total = 0.0
        spin_wake_total = 0.0
        for l in range(L):
            tp_bar_total += delta_us(7, 26, l)
            stage_l_total += delta_us(26, 27, l)
            stage_mn_total += delta_us(27, 8, l)
            if has_rs_bisect:
                stage_l1_total += delta_us(26, 28, l)
                stage_l_sync_total += delta_us(28, 29, l)
                stage_l2_total += delta_us(29, 27, l)
            if has_spin_diag:
                spin_pre_total += delta_us(7, 30, l)
                spin_body_total += delta_us(30, 31, l)
                spin_wake_total += delta_us(31, 26, l)
        totals["TP barrier (multimem.red + per-CTA spin)"] = tp_bar_total
        totals["Stage L (multimem.ld_reduce + sCarry)"] = stage_l_total
        totals["Stage M+N (RMSNorm + router GEMV)"] = stage_mn_total
        if has_rs_bisect:
            totals["  └─ Stage L pre (slot 26->28, R1: ~0)"] = stage_l1_total
            totals["  └─ Stage L body (slot 28->29, R1: local read + sCarry)"] = stage_l_sync_total
            totals["  └─ Stage L post (slot 29->27, R1: ~0)"] = stage_l2_total
        if has_spin_diag:
            totals["  ⟦DIAG⟧ TP spin: slot 7->30 (pre-spin warp7 entry)"] = spin_pre_total
            totals["  ⟦DIAG⟧ TP spin: slot 30->31 (warp7 fabric wait body)"] = spin_body_total
            totals["  ⟦DIAG⟧ TP spin: slot 31->26 (warp7 fence + barrier wake)"] = spin_wake_total
            tp_bar_cp = spin_pre_total + spin_body_total
            slot7_27 = tp_bar_total + stage_l_total
            stage_l_clean = max(0.0, slot7_27 - tp_bar_cp)
            totals.pop("TP barrier (multimem.red + per-CTA spin)", None)
            totals.pop("Stage L (multimem.ld_reduce + sCarry)", None)
            totals.pop(
                "  ⟦DIAG⟧ TP spin: slot 31->26 (warp7 fence + barrier wake)",
                None,
            )
            for k in list(totals.keys()):
                if k.startswith("  └─ Stage L "):
                    totals.pop(k, None)
            totals["TP barrier critical-path (slot 7->31 warp7, CORRECTED)"] = tp_bar_cp
            totals["Stage L compute (slot 7->27 minus TP-cp, CORRECTED)"] = stage_l_clean
    else:
        lmn_total = 0.0
        for l in range(L):
            lmn_total += delta_us(7, 8, l)
        totals["L+M+N (residual, post-norm, router)"] = lmn_total

    p1_total = 0.0
    p2_total = 0.0
    p1_starts = [9] + [11 + k for k in range(K_topk - 1)]
    p1_ends   = [11 + k for k in range(K_topk)]
    for (a, b) in zip(p1_starts, p1_ends):
        for l in range(L):
            p1_total += delta_us(a, b, l)
    p2_starts = [11 + K_topk - 1] + [17 + k for k in range(K_topk - 1)]
    p2_ends   = [17 + k for k in range(K_topk)]
    for (a, b) in zip(p2_starts, p2_ends):
        for l in range(L):
            p2_total += delta_us(a, b, l)
    totals[f"P1 routed-expert ({K_topk} experts total)"] = p1_total
    totals[f"P2 routed-expert ({K_topk} experts total)"] = p2_total
    import sys as _sys
    for _l in (5, 30, 55):
        _p1 = [delta_us(p1_starts[k], p1_ends[k], _l) for k in range(K_topk)]
        _p2 = [delta_us(p2_starts[k], p2_ends[k], _l) for k in range(K_topk)]
        _spin1 = delta_us(11 + K_topk - 1, 43, _l)
        _p2c0  = delta_us(43, 17, _l)
        print(f"[per-expert-slot µs, layer {_l}]  P1={['%.1f'%x for x in _p1]}  P2={['%.1f'%x for x in _p2]}  | EOP1-spin={_spin1:.1f}  P2compute0={_p2c0:.1f}", file=_sys.stderr, flush=True)

    q1_total = 0.0
    q2_total = 0.0
    for l in range(L):
        q1_total += delta_us(22, 23, l)
        q2_total += delta_us(23, 10, l)
    totals["Q1 shared-expert (gate+up+SiLU)"] = q1_total
    totals["Q2 shared-expert (down only, EP excluded)"] = q2_total

    has_q2_diag = (
        stage_ts_cpu.shape[1] >= 43
        and bool((stage_ts_cpu[:, 41] != 0).any())
        and bool((stage_ts_cpu[:, 42] != 0).any())
    )
    if has_q2_diag:
        q2_quant_total = 0.0
        q2_main_total = 0.0
        q2_sync_total = 0.0
        for l in range(L):
            q2_quant_total += delta_us(23, 41, l)
            q2_main_total += delta_us(41, 42, l)
            q2_sync_total += delta_us(42, 10, l)
        totals["  ⟦DIAG⟧ Q2: slot 23->41 (quant mInterTmp BF16->FP8)"] = q2_quant_total
        totals["  ⟦DIAG⟧ Q2: slot 41->42 (TMA+MMA+EPI mainloop)"] = q2_main_total
        totals["  ⟦DIAG⟧ Q2: slot 42->10 (grid-sync wait cross-CTA)"] = q2_sync_total

    has_bisect = stage_ts_cpu.shape[1] >= 26 and bool((stage_ts_cpu[:, 24] != 0).any())
    if has_bisect:
        ep_bar_total = 0.0
        stage_r_total = 0.0
        loop_overhead_total = 0.0
        for l in range(L):
            ep_bar_total += delta_us(10, 24, l)
            stage_r_total += delta_us(24, 25, l)
        for l in range(L - 1):
            d = int(stage_ts_cpu[l + 1, 0]) - int(stage_ts_cpu[l, 25])
            if d < 0:
                d += (1 << 32)
            loop_overhead_total += d * cyc_to_us
        totals["EP barrier (multimem.red + spin)"] = ep_bar_total
        totals["Stage R (multimem.ld_reduce flood + sCarry update)"] = stage_r_total
        totals[f"Inter-layer loop overhead ({L-1} layers)"] = loop_overhead_total
    else:
        ep_r_total = 0.0
        for l in range(L - 1):
            d = int(stage_ts_cpu[l + 1, 0]) - int(stage_ts_cpu[l, 10])
            if d < 0:
                d += (1 << 32)
            ep_r_total += d * cyc_to_us
        totals[f"EP_allreduce + stage R + loop overhead ({L-1} layers)"] = ep_r_total

    return totals, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--n_steps", type=int, default=5, help="number of measurement decode steps to aggregate")
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
    if rank == 0:
        print(f"=== Stage timing: TP=EP={ws}, SM clock = {sm_clock_hz/1e9:.3f} GHz ===", flush=True)

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

    K_topk = engine.cfg.num_experts_per_tok
    aggregate = None
    step_total_ms_list = []
    engine.reset()
    for tid in prompt_ids:
        engine.step(int(tid))
    torch.cuda.synchronize()

    for s in range(args.n_steps):
        t_p = time.perf_counter()
        engine.step(None) if inline_path else engine.step(int(prompt_ids[-1]))
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t_p) * 1000
        step_total_ms_list.append(step_ms)
        stage_ts = get_last_stage_ts()
        assert stage_ts is not None, "stage_ts buffer missing — was _clock() compiled in?"
        stage_ts_cpu = stage_ts.cpu()
        per_stage_us, L = stage_breakdown(stage_ts_cpu, sm_clock_hz, K_topk=K_topk)
        if aggregate is None:
            aggregate = {k: [v] for k, v in per_stage_us.items()}
            aggregate["_L"] = L
        else:
            for k, v in per_stage_us.items():
                aggregate[k].append(v)
        if rank == 0:
            print(f"[rank 0]   step {s+1}/{args.n_steps}: wall={step_ms:.2f} ms", flush=True)

    if rank == 0:
        L = aggregate.pop("_L")
        mean_wall = sum(step_total_ms_list) / len(step_total_ms_list)
        print("", flush=True)
        print(f"=== Per-stage µs averaged over {args.n_steps} decode steps "
              f"(L={L} layers, sum across all layers per step) ===", flush=True)
        rows = []
        sum_us = 0.0
        for label, samples in aggregate.items():
            mean_us = sum(samples) / len(samples)
            rows.append((mean_us, label))
            sum_us += mean_us
        rows.sort(reverse=True)
        print(f"{'stage':<55} {'sum_us/step':>12} {'%':>6} {'us/layer':>10}", flush=True)
        for us, label in rows:
            pct = 100 * us / sum_us if sum_us > 0 else 0.0
            per_layer = us / L
            print(f"{label:<55} {us:>12.1f} {pct:>5.1f}% {per_layer:>10.2f}", flush=True)
        print(f"{'TOTAL stages (CTA-0 clock)':<55} {sum_us:>12.1f}", flush=True)
        print(f"{'Wall-clock per step':<55} {1000*mean_wall:>12.1f} us", flush=True)
        print(f"{'Overhead unaccounted (kernel launch + post-stage)':<55} "
              f"{1000*mean_wall - sum_us:>12.1f} us", flush=True)
        if sum_us > 1000 * mean_wall:
            print(
                "  NOTE: bucket sum > wall ⇒ CTA-0 stall buckets overlap with "
                "other CTAs' work (double-count); treat single-bucket gaps as "
                "upper bounds, not critical-path contribution.",
                flush=True,
            )

        if args.output_json:
            out = {
                "tp": ws, "ep": ws, "L": L, "K_topk": K_topk,
                "sm_clock_hz": sm_clock_hz,
                "wall_ms_per_step": mean_wall,
                "stage_us_mean": {label: sum(samples)/len(samples) for label, samples in aggregate.items()},
                "stage_us_per_step": {label: samples for label, samples in aggregate.items()},
                "step_wall_ms": step_total_ms_list,
            }
            with open(args.output_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[rank 0] wrote {args.output_json}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

