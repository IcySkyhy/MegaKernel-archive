#!/usr/bin/env python
"""
run.py -- gate and benchmark a compiled megakernel.

  python harness/run.py OUT_DIR --model MODEL_DIR [--gate N] [--bench P,D] [--guard]

The gate feeds a *randomly drawn* token sequence, seeded from the clock, so no
amount of memorisation in the engine can pass it; the seed is printed so any
result can be reproduced exactly.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mkrun import Engine, reference_logits, dequantized_reference_logits, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate", type=int, default=16, help="teacher-forced steps (0 = skip)")
    ap.add_argument("--prefill", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ref-device", default="cuda:1")
    ap.add_argument("--ref-dequant", action="store_true",
                    help="reference from the same weights dequantised to bf16 "
                         "(weight-only formats: isolates reduction order from "
                         "the reference's activation quantisation)")
    ap.add_argument("--ref-npy", default="", help="precomputed reference logits (see ref_cpu.py)")
    ap.add_argument("--ref-first", action="store_true",
                    help="compute the reference BEFORE loading the engine and free it; "
                         "the only way to gate a model too large to hold both at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--weight-sample", type=int, default=0, help="0 = every weight tensor")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    seed = a.seed or (int(time.time() * 1e6) & 0x7fffffff)
    if a.ref_npy and not a.seed:
        raise SystemExit("--ref-npy requires --seed, so the token sequence matches")
    rng = np.random.default_rng(seed)
    cfg = json.load(open(os.path.join(a.out, "build.json")))
    print(f"== {cfg['name']}  ({cfg['arch']})  seed={seed}")

    # The reference and the engine both want the whole model in memory.  Below
    # about 8 B parameters they fit side by side; above that they do not, and the
    # only way to gate at all is to compute the reference FIRST and free it.  The
    # token ids come from build.json's vocab so they can be drawn before either
    # exists.
    ref = None
    if a.gate > 0 and a.ref_first:
        gate_ids = rng.integers(0, cfg["vocab"], size=a.gate).astype(np.int32)
        print("-- reference first (the engine and the reference do not fit together)")
        t0 = time.time()
        ref = (dequantized_reference_logits if a.ref_dequant else reference_logits)(
            a.model, gate_ids.tolist(), device=a.ref_device)
        print(f"   reference forward: {time.time()-t0:.1f}s on {a.ref_device}")

    eng = Engine(a.out, a.model, device=a.device, max_len=max(a.prefill + a.decode + 8, 64))
    print(f"   weights loaded in {eng.load_s:.1f}s   grid={eng.grid()} regs={eng.regs()} "
          f"smem={eng.smem()}B maxctx={eng.maxctx}")
    res = dict(build=cfg, seed=seed, grid=eng.grid(), regs=eng.regs(), smem=eng.smem())

    # ---------------------------------------------------------------- gate
    if a.gate > 0:
        n = a.gate
        ids = gate_ids if ref is not None else rng.integers(0, eng.vocab, size=n).astype(np.int32)
        np.save(os.path.join(a.out, "gate_ids.npy"), ids)
        eng.reset(); eng.set_tokens(ids)
        got = np.zeros((n, eng.vocab), dtype=np.float32)
        for t in range(n):
            eng.run(1, greedy=0)
            got[t] = eng.logits()
        print("-- gate: teacher-forced vs HuggingFace transformers")
        t0 = time.time()
        if ref is not None:
            pass
        elif a.ref_npy:
            ref = np.load(a.ref_npy)
            print(f"   reference logits from {a.ref_npy}")
        elif a.ref_dequant:
            ref = dequantized_reference_logits(a.model, ids.tolist(), device=a.ref_device)
            print(f"   dequantised reference: {time.time()-t0:.1f}s on {a.ref_device}")
        else:
            ref = reference_logits(a.model, ids.tolist(), device=a.ref_device)
            print(f"   reference forward: {time.time()-t0:.1f}s on {a.ref_device}")
        res["gate"] = gate(ref, got)
        np.save(os.path.join(a.out, "gate_ref.npy"), ref[:4])
        np.save(os.path.join(a.out, "gate_got.npy"), got[:4])

    # ---------------------------------------------------------------- metric
    if a.decode > 0:
        P, D = a.prefill, a.decode
        ids = rng.integers(0, eng.vocab, size=P + D + 1).astype(np.int32)
        eng.reset(); eng.set_tokens(ids)
        eng.run(P, greedy=0)                       # prefill, one token at a time
        # N-vs-1 differencing cancels everything that is not per-token work
        eng.set_pos(P); eng.run(1, greedy=0); t1 = eng.last_ms()
        eng.set_pos(P); eng.run(D, greedy=0); tN = eng.last_ms()
        w0 = time.time(); eng.set_pos(P); eng.run(D, greedy=0); wall = (time.time() - w0) * 1e3
        ms = (tN - t1) / (D - 1)
        bpt = cfg["bytes_per_token"]
        peak = cfg["stream_peak_gbs"]
        bw = (bpt / 1e6) / ms                       # GB/s
        print(f"-- decode, batch 1, ctx {P}, {D} steps")
        print(f"   decode_ms_per_tok    {ms:.4f}")
        print(f"   tok_per_s            {1000/ms:.1f}")
        print(f"   bw_util_pct          {100*bw/peak:.2f}   ({bw:.0f} of {peak:.0f} GB/s)")
        print(f"   roofline_ms          {cfg['roofline_ms']:.4f}   (x{ms/cfg['roofline_ms']:.2f})"
              "   bytes / stream peak")
        if cfg.get("floor_ms"):
            # the same bytes charged at each format's measured gemv ceiling: a
            # 4-bit matrix cannot stream at the dense rate, so this is the floor
            # a kernel can actually reach
            print(f"   floor_ms             {cfg['floor_ms']:.4f}   (x{ms/cfg['floor_ms']:.2f})"
                  "   bytes / per-format gemv ceiling")
            res["floor_ms"] = cfg["floor_ms"]
        print(f"   predicted_ms         {cfg['predicted_ms']:.4f}   "
              f"(planner error {100*(ms-cfg['predicted_ms'])/ms:+.1f}%)")
        print(f"   wall/event agreement {100*wall/tN:.1f}%")
        res["decode_ms_per_tok"] = ms
        res["bw_util_pct"] = 100 * bw / peak
        res["wall_ms"] = wall
        res["event_ms"] = tN

    # ---------------------------------------------------------------- guard
    if a.guard:
        from mkguard import run_guard
        res["guard"] = run_guard(eng, a.model, cfg, rng,
                                 measured_ms=res.get("decode_ms_per_tok"),
                                 weight_sample=a.weight_sample or None)

    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)
    eng.free()
    ok = res.get("gate", {}).get("passed", True) and all(
        g["passed"] for g in res.get("guard", {}).get("checks", []))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
