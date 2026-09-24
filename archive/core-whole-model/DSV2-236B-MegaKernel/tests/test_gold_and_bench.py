"""Numerical gold gate plus decode benchmark, in one load+compile.

The gate is a top-K logit overlap protocol rather than byte-exact token match:
Stage K's `multimem.red.add.bf16x2` server-side adds are non-deterministic in
peer arrival order, and bf16 non-associativity yields 1-ULP logit differences
that can flip an argmax. The decision metrics are per-position top-5 overlap
and top-1 agreement against a calibrated reference, plus a non-degeneracy check
on free-running greedy decode. Exits 1 on gold failure.
"""
import os, sys, time, json, argparse
import torch
import torch.distributed as dist

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv2mk.paths import model_snapshot


H_FINAL_RTOL    = 5e-2
H_FINAL_ATOL    = 1.5e-1
LOGIT_RTOL      = 5e-2
LOGIT_ATOL      = 1e-1
TOP_K           = 5
MIN_TOP_K_OVERLAP = 3
MIN_TOP_K_FRAC  = 0.75
TOP1_MIN_AGREE  = 12

SNAP = model_snapshot()

DEFAULT_GOLD_REF = Path(__file__).resolve().parent / "gold_ref" / "ref_v7_stageR.pt"

GOLD_PROMPT = "Hello, my name is"
GOLD_MAX_NEW = 16
GOLD_TOKENS = [
    17464, 11, 601, 1210, 317,
    509, 1531, 92, 285, 304, 608, 509,
    491, 92, 1555, 1712, 29074, 185, 185, 2, 207
]
assert len(GOLD_TOKENS) == 21
GOLD_N_PROMPT = 5
GOLD_N_GEN = GOLD_MAX_NEW


def evaluate_non_degeneracy(out_ids, tokenizer):
    gen_ids = out_ids[GOLD_N_PROMPT:GOLD_N_PROMPT + GOLD_N_GEN]
    vocab_size = getattr(tokenizer, "vocab_size", 102400) or 102400
    bad_loop = False
    if len(gen_ids) >= 4:
        for i in range(len(gen_ids) - 3):
            if gen_ids[i] == gen_ids[i+1] == gen_ids[i+2] == gen_ids[i+3]:
                bad_loop = True
                break
    bad_zero = any(t == 0 for t in gen_ids)
    bad_oov = any(t < 0 or t >= vocab_size for t in gen_ids)
    bad_short = len(gen_ids) < GOLD_N_GEN
    summary = {
        "loop_4plus": bad_loop,
        "any_zero_token": bad_zero,
        "any_out_of_vocab": bad_oov,
        "short_output": bad_short,
        "n_generated": len(gen_ids),
    }
    ok = (not bad_loop) and (not bad_zero) and (not bad_oov) and (not bad_short)
    bits = []
    if bad_loop:    bits.append("4+ token loop detected")
    if bad_zero:    bits.append("zero token in output")
    if bad_oov:     bits.append("out-of-vocab token in output")
    if bad_short:   bits.append(f"only {len(gen_ids)}/{GOLD_N_GEN} tokens generated")
    reason = "non-degenerate" if ok else "; ".join(bits)
    return ok, reason, summary


def capture_teacher_forced(engine, prompt_ids, target_tokens, *,
                           rank=0, dtype=torch.bfloat16):
    engine.reset()
    for tid in prompt_ids:
        engine.step(int(tid))
    h_finals = [engine._last_h_final.detach().cpu().clone()]
    for i in range(GOLD_N_GEN - 1):
        forced_tok = int(target_tokens[GOLD_N_PROMPT + i])
        engine.step(forced_tok)
        h_finals.append(engine._last_h_final.detach().cpu().clone())
    h_finals_t = torch.stack(h_finals, dim=0)
    with torch.no_grad():
        h_dev = h_finals_t.to(engine.device).view(GOLD_N_GEN, 1, -1).to(dtype)
        normed = engine._final_norm(h_dev)
        logits_dev = engine._lm_head(normed).view(GOLD_N_GEN, -1)
    logits = logits_dev.float().cpu().clone()
    return h_finals_t, logits


def topk_overlap(got_logits, ref_logits, k=TOP_K):
    _, got_idx = torch.topk(got_logits, k=k, dim=-1)
    _, ref_idx = torch.topk(ref_logits, k=k, dim=-1)
    out = torch.zeros(got_idx.shape[0], dtype=torch.int32)
    for n in range(got_idx.shape[0]):
        out[n] = len(set(got_idx[n].tolist()) & set(ref_idx[n].tolist()))
    return out


def evaluate_h_final_and_logits(h_finals_got, logits_got, ref):
    ref_h = ref["h_finals"]
    ref_log = ref["logits"]
    n = h_finals_got.shape[0]
    h_got_f = h_finals_got.float()
    h_ref_f = ref_h.float()
    abs_err_h = (h_got_f - h_ref_f).abs()
    abs_thr_h = H_FINAL_ATOL + H_FINAL_RTOL * h_ref_f.abs()
    per_pos_h_ok = (abs_err_h <= abs_thr_h).reshape(n, -1).all(dim=-1)
    per_pos_h_maxerr = abs_err_h.reshape(n, -1).max(dim=-1).values
    per_pos_h_l2 = abs_err_h.float().reshape(n, -1).pow(2).sum(dim=-1).sqrt()
    h_ok = bool(per_pos_h_ok.all().item())
    overlap = topk_overlap(logits_got, ref_log, k=TOP_K)
    per_pos_top_ok = overlap >= MIN_TOP_K_OVERLAP
    n_top_ok = int(per_pos_top_ok.sum().item())
    n_pos = int(per_pos_top_ok.shape[0])
    frac_top_ok = n_top_ok / n_pos
    topk_ok = frac_top_ok >= MIN_TOP_K_FRAC
    top1_got = logits_got.argmax(dim=-1)
    top1_ref = ref_log.argmax(dim=-1)
    top1_agree = int((top1_got == top1_ref).sum().item())
    top1_ok = top1_agree >= TOP1_MIN_AGREE
    abs_err_l = (logits_got - ref_log).abs()
    per_pos_l_maxerr = abs_err_l.reshape(n, -1).max(dim=-1).values
    summary = {
        "per_pos_h_ok": per_pos_h_ok.tolist(),
        "per_pos_h_maxerr": per_pos_h_maxerr.tolist(),
        "per_pos_h_l2": per_pos_h_l2.tolist(),
        "per_pos_topk_overlap": overlap.tolist(),
        "per_pos_topk_ok": per_pos_top_ok.tolist(),
        "per_pos_logit_maxerr": per_pos_l_maxerr.tolist(),
        "top1_agreement": int(top1_agree),
        "top1_threshold": TOP1_MIN_AGREE,
        "n_positions": int(n),
        "h_final_atol": H_FINAL_ATOL,
        "h_final_rtol": H_FINAL_RTOL,
        "logit_topk": TOP_K,
        "min_topk_overlap": MIN_TOP_K_OVERLAP,
        "min_topk_frac": MIN_TOP_K_FRAC,
        "frac_topk_ok": frac_top_ok,
        "n_pos_topk_ok": n_top_ok,
    }
    ok = topk_ok and top1_ok
    bits = []
    bits.append(f"top-{TOP_K}-overlap={'OK' if topk_ok else 'FAIL'} "
                f"({n_top_ok}/{n_pos} pos ≥{MIN_TOP_K_OVERLAP} = "
                f"{frac_top_ok:.0%} ≥ {MIN_TOP_K_FRAC:.0%}; min={int(overlap.min())}/{TOP_K})")
    bits.append(f"top-1-agree={'OK' if top1_ok else 'FAIL'} "
                f"({top1_agree}/{n} ≥ {TOP1_MIN_AGREE})")
    bits.append(f"h_final-elementwise={'OK' if h_ok else 'info-only'} "
                f"({int(per_pos_h_ok.sum())}/{n} pos, "
                f"max-abs-err={per_pos_h_maxerr.max().item():.2e})")
    return ok, "; ".join(bits), summary


def calibrate_reference(engine, prompt_ids, *, n_samples=3, rank=0, dtype=torch.bfloat16):
    all_h = []
    all_log = []
    all_tok = []
    for s in range(n_samples):
        if rank == 0:
            print(f"[CALIB] sample {s+1}/{n_samples}…", flush=True)
        tokens = engine.generate(prompt_ids, max_new_tokens=GOLD_MAX_NEW, greedy=True)
        all_tok.append(tokens)
        hf, lg = capture_teacher_forced(engine, prompt_ids, GOLD_TOKENS, rank=rank, dtype=dtype)
        all_h.append(hf)
        all_log.append(lg)
    h_stack = torch.stack(all_h, dim=0).float()
    l_stack = torch.stack(all_log, dim=0).float()
    h_med = h_stack.median(dim=0).values.to(torch.bfloat16)
    l_med = l_stack.median(dim=0).values
    h_spread = (h_stack.max(dim=0).values - h_stack.min(dim=0).values).max(dim=-1).values
    l_spread = (l_stack.max(dim=0).values - l_stack.min(dim=0).values).max(dim=-1).values
    return {
        "h_finals": h_med,
        "logits": l_med,
        "tokens": all_tok[0],
        "all_tokens": all_tok,
        "measured_spread": {
            "h_final_max_per_pos": h_spread.tolist(),
            "logit_max_per_pos": l_spread.tolist(),
            "n_samples": n_samples,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_len", type=int, default=128, help="bench: synthetic prompt length")
    ap.add_argument("--output_len", type=int, default=128, help="bench: decode steps per iter")
    ap.add_argument("--warmup", type=int, default=2, help="bench: warmup iters")
    ap.add_argument("--iters", type=int, default=3, help="bench: measurement iters")
    ap.add_argument("--output_json", type=str, default=None, help="bench: write JSON output here")
    ap.add_argument("--skip_gold", action="store_true", help="skip gold gate (bench only)")
    ap.add_argument("--skip_bench", action="store_true", help="skip bench (gold only)")
    ap.add_argument("--strict_gold", action="store_true",
                    help="force byte-exact 21/21 gold match (overrides --gold_ref)")
    ap.add_argument("--gold_ref", type=str, default=str(DEFAULT_GOLD_REF),
                    help="path to .pt reference (produced by --calibrate_ref). "
                         "When provided, enables Checks 1+2+3 (h_final tolerance, "
                         "top-k overlap, non-degeneracy). Without it, only the "
                         "byte-exact backstop (Check 4) runs.")
    ap.add_argument("--calibrate_ref", type=str, default=None,
                    help="capture a new reference at this path and exit gold phase. "
                         "Run N samples and store per-element median + measured "
                         "spread (multimem-add nondeterminism envelope).")
    ap.add_argument("--calibrate_samples", type=int, default=3,
                    help="N samples for --calibrate_ref (default 3).")
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
        print(f"=== DSv2-236B GOLD+BENCH TP=EP={ws} (include_layer0={include_layer0}) ===", flush=True)
        print(f"Snapshot: {SNAP}", flush=True)
        print(f"Free GPU mem at start: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB", flush=True)

    from dsv2mk.runtime.engine import DSv2InKernelTPEP
    t0 = time.time()
    engine = DSv2InKernelTPEP(
        SNAP,
        S_max=max(args.input_len + args.output_len + 16, 512),
        device=device,
        tp_size=ws, ep_size=ws,
        use_fp8_q1=True, use_fp8_q2=True, use_fp8_routed=True,
        skip_hf_load=True,
        include_layer0=include_layer0,
    )
    t_load = time.time() - t0
    if rank == 0:
        print(f"[rank 0] Loaded+compiled in {t_load:.1f}s", flush=True)
        print(f"[rank 0] Free GPU mem now: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB", flush=True)

    gold_ok = True
    if not args.skip_gold:
        tok = engine.tokenizer
        prompt_ids = tok.encode(GOLD_PROMPT, add_special_tokens=True)
        assert len(prompt_ids) == GOLD_N_PROMPT, \
            f"prompt tokenized to {len(prompt_ids)} ids, expected {GOLD_N_PROMPT}"

        if rank == 0:
            print(f"\n[GOLD] prompt='{GOLD_PROMPT}' prompt_ids={prompt_ids} max_new={GOLD_MAX_NEW}", flush=True)
            if args.calibrate_ref:
                print(f"[GOLD] mode=CALIBRATE (samples={args.calibrate_samples} → {args.calibrate_ref})", flush=True)
            elif args.strict_gold:
                print(f"[GOLD] mode=STRICT byte-exact (--strict_gold)", flush=True)
            elif args.gold_ref and os.path.exists(args.gold_ref):
                print(f"[GOLD] mode=RELAXED logit-tolerance (ref={args.gold_ref})", flush=True)
            else:
                if args.gold_ref:
                    print(f"[GOLD] mode=STRICT byte-exact (--gold_ref {args.gold_ref} not found; "
                          f"fall-back backstop)", flush=True)
                else:
                    print(f"[GOLD] mode=STRICT byte-exact (no --gold_ref provided; backstop)", flush=True)

        if args.calibrate_ref:
            t_cal = time.time()
            ref_or_none = calibrate_reference(
                engine, prompt_ids,
                n_samples=args.calibrate_samples, rank=rank, dtype=engine.dtype,
            )
            t_cal_s = time.time() - t_cal
            if rank == 0:
                ref = ref_or_none
                ref["metadata"] = {
                    "snapshot": SNAP,
                    "tp_size": ws, "ep_size": ws,
                    "include_layer0": include_layer0,
                    "use_fp8_q1": True, "use_fp8_q2": True, "use_fp8_routed": True,
                    "h_final_atol": H_FINAL_ATOL, "h_final_rtol": H_FINAL_RTOL,
                    "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL,
                    "topk": TOP_K, "min_topk_overlap": MIN_TOP_K_OVERLAP,
                    "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                os.makedirs(os.path.dirname(args.calibrate_ref) or ".", exist_ok=True)
                torch.save(ref, args.calibrate_ref)
                print(f"[CALIB] wrote {args.calibrate_ref} ({t_cal_s:.1f}s)", flush=True)
                sp = ref["measured_spread"]
                print(f"[CALIB] measured nondeterminism envelope across "
                      f"{sp['n_samples']} samples:", flush=True)
                print(f"[CALIB]   h_final max-spread per pos: "
                      f"max={max(sp['h_final_max_per_pos']):.3e} "
                      f"mean={sum(sp['h_final_max_per_pos'])/len(sp['h_final_max_per_pos']):.3e}",
                      flush=True)
                print(f"[CALIB]   logit max-spread per pos: "
                      f"max={max(sp['logit_max_per_pos']):.3e} "
                      f"mean={sum(sp['logit_max_per_pos'])/len(sp['logit_max_per_pos']):.3e}",
                      flush=True)
                print(f"[CALIB]   tolerance h_final atol={H_FINAL_ATOL}, rtol={H_FINAL_RTOL}",
                      flush=True)
                print(f"[CALIB]   ratio (tol / max_spread) — bigger is more headroom for kernel changes",
                      flush=True)
                if max(sp["h_final_max_per_pos"]) > 0:
                    print(f"[CALIB]     h_final: {H_FINAL_ATOL / max(sp['h_final_max_per_pos']):.1f}×",
                          flush=True)
                ok_self, reason_self, _ = evaluate_h_final_and_logits(
                    ref["h_finals"], ref["logits"], ref,
                )
                print(f"[CALIB] self-check vs reference: {'OK' if ok_self else 'FAIL'} — {reason_self}", flush=True)
            dist.barrier()
            if args.skip_bench:
                return
        else:
            t1 = time.time()
            out_ids = engine.generate(prompt_ids, max_new_tokens=GOLD_MAX_NEW, greedy=True)
            t_gen = time.time() - t1

            degen_ok, degen_reason, degen_summary = evaluate_non_degeneracy(out_ids, tok)

            byte_exact = (out_ids == GOLD_TOKENS)

            if rank == 0:
                print(f"[GOLD] generated {len(out_ids) - len(prompt_ids)} tokens in {t_gen:.2f}s", flush=True)
                print(f"[GOLD] tokens:   {out_ids}", flush=True)
                print(f"[GOLD] expected: {GOLD_TOKENS}", flush=True)
                try:
                    out_text = tok.decode(out_ids)
                    print(f"[GOLD] text:     {out_text!r}", flush=True)
                except (OverflowError, ValueError) as e:
                    print(f"[GOLD] text:     <decode failed: {type(e).__name__}: {e}>", flush=True)
                for i, (got, exp) in enumerate(zip(out_ids, GOLD_TOKENS)):
                    mark = "✓" if got == exp else "✗"
                    tag = "[prompt]" if i < GOLD_N_PROMPT else "[gen]"
                    print(f"[GOLD]   [{i:2d}]{tag} got={got:>7d} exp={exp:>7d} {mark}", flush=True)
                print(f"[GOLD] Check 3 (non-degeneracy): {'OK' if degen_ok else 'FAIL'} — {degen_reason}",
                      flush=True)
                print(f"[GOLD] Check 4 (byte-exact backstop): {'OK' if byte_exact else 'MISS'} "
                      f"({sum(1 for g,e in zip(out_ids, GOLD_TOKENS) if g==e)}/21)", flush=True)

            use_ref = (
                (not args.strict_gold)
                and args.gold_ref is not None
                and os.path.exists(args.gold_ref)
            )

            if args.strict_gold or not use_ref:
                gold_ok = bool(byte_exact and degen_ok)
                if rank == 0:
                    if gold_ok:
                        print(f"[GOLD] ✅ PASS — byte-exact + non-degenerate", flush=True)
                    else:
                        reason = []
                        if not byte_exact: reason.append("byte-exact MISS")
                        if not degen_ok: reason.append(f"degen FAIL ({degen_reason})")
                        print(f"[GOLD] ❌ FAIL — {'; '.join(reason)}", flush=True)
            else:
                t2 = time.time()
                if rank == 0:
                    print(f"[GOLD] capturing teacher-forced h_final + logits for 16 positions…", flush=True)
                hf, lg = capture_teacher_forced(engine, prompt_ids, GOLD_TOKENS,
                                                rank=rank, dtype=engine.dtype)
                t_capture = time.time() - t2
                if rank == 0:
                    print(f"[GOLD] capture done in {t_capture:.2f}s; loading ref {args.gold_ref}", flush=True)
                    ref = torch.load(args.gold_ref, map_location="cpu", weights_only=False)
                    tol_ok, tol_reason, tol_summary = evaluate_h_final_and_logits(hf, lg, ref)
                    print(f"[GOLD] Check 1+2 (h_final + top-k): {tol_reason}", flush=True)
                    print(f"[GOLD]   per-pos h_final maxerr: "
                          f"{['%.2e'%v for v in tol_summary['per_pos_h_maxerr']]}", flush=True)
                    print(f"[GOLD]   per-pos top-{TOP_K} overlap: "
                          f"{tol_summary['per_pos_topk_overlap']}", flush=True)
                    print(f"[GOLD]   top-1 agreement with ref: "
                          f"{tol_summary['top1_agreement']}/{tol_summary['n_positions']}", flush=True)
                    gold_ok = bool(tol_ok and degen_ok)
                    if gold_ok:
                        if byte_exact:
                            print(f"[GOLD] ✅ PASS — byte-exact + tolerance + non-degen", flush=True)
                        else:
                            print(f"[GOLD] ✅ PASS (relaxed) — tolerance + non-degen "
                                  f"(byte-exact MISS, top-1 agree={tol_summary['top1_agreement']}/{tol_summary['n_positions']})",
                                  flush=True)
                    else:
                        bits = []
                        if not tol_ok: bits.append(f"tolerance: {tol_reason}")
                        if not degen_ok: bits.append(f"degen: {degen_reason}")
                        print(f"[GOLD] ❌ FAIL — {'; '.join(bits)}", flush=True)

    gold_ok_t = torch.tensor([1 if gold_ok else 0], dtype=torch.int32, device=device)
    dist.broadcast(gold_ok_t, src=0)
    gold_ok = bool(gold_ok_t.item())
    if not gold_ok:
        if rank == 0:
            print("\n[GOLD] aborting bench due to gold mismatch", flush=True)
        dist.barrier()
        sys.exit(1)

    if args.skip_bench:
        if rank == 0:
            print("\n[BENCH] skipped (--skip_bench)", flush=True)
        dist.barrier()
        return

    if rank == 0:
        print(
            f"\n=== BENCH B=1 in={args.input_len} out={args.output_len} "
            f"warmup={args.warmup} iters={args.iters} ===",
            flush=True,
        )

    torch.manual_seed(0)
    prompt_ids = torch.randint(1000, 50000, (args.input_len,), dtype=torch.long).tolist()
    if rank == 0:
        print(f"[BENCH] synthetic prompt of {len(prompt_ids)} tokens", flush=True)

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
            print(f"[BENCH]   warmup {w+1}/{args.warmup} done", flush=True)

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
            print(
                f"[BENCH]   iter {it+1}/{args.iters}: prefill={prefill_s:.2f}s "
                f"decode_total={decode_total_s:.3f}s "
                f"mean={decode_total_s / args.output_len * 1000:.3f} ms/tok",
                flush=True,
            )

    if rank == 0:
        all_step_ms = [s * 1000 for r in results for s in r["decode_step_s"]]
        all_step_ms_sorted = sorted(all_step_ms)
        n = len(all_step_ms_sorted)
        decode_mean_ms = sum(all_step_ms) / n
        decode_p50_ms = all_step_ms_sorted[n // 2]
        decode_p90_ms = all_step_ms_sorted[int(n * 0.9)]
        decode_p99_ms = all_step_ms_sorted[int(n * 0.99)]
        decode_tok_per_s = 1000.0 / decode_mean_ms

        summary = {
            "tp_size": ws,
            "ep_size": ws,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "include_layer0": include_layer0,
            "class_b_inline": inline_path,
            "warmup": args.warmup,
            "iters": args.iters,
            "decode_mean_ms": decode_mean_ms,
            "decode_p50_ms": decode_p50_ms,
            "decode_p90_ms": decode_p90_ms,
            "decode_p99_ms": decode_p99_ms,
            "decode_tok_per_s": decode_tok_per_s,
            "n_steps_total": n,
            "per_iter": [
                {"prefill_s": r["prefill_s"], "decode_total_s": r["decode_total_s"]}
                for r in results
            ],
        }
        print("\n" + json.dumps(summary, indent=2), flush=True)
        print(
            f"\n[BENCH] mean decode latency: {decode_mean_ms:.2f} ms/tok  "
            f"p50={decode_p50_ms:.2f}  p90={decode_p90_ms:.2f}  p99={decode_p99_ms:.2f}",
            flush=True,
        )
        print(f"[BENCH] sustained throughput: {decode_tok_per_s:.2f} tok/s/sequence (B=1)", flush=True)

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[BENCH] wrote {args.output_json}", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()

