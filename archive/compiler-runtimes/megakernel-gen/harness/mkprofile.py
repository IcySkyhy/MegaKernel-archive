"""
profile.py -- the compiler's view of its own kernel.

The megakernel writes a nanosecond mark after every grid barrier, from block 0.
Reading them back gives the real per-stage schedule, which is compared here
against what the planner predicted.  This is not a search: it is the feedback
that tells the *compiler* where its cost model is wrong, once, for every model
it will ever compile.
"""
import ctypes, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mkrun import Engine


def profile(eng, cfg, ctx=256, warm=8):
    L = eng.lib
    for n, r, a in [("mk_prof", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]),
                    ("mk_prof_enable", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
                    ("mk_prof_slots", ctypes.c_int, [ctypes.c_void_p])]:
        f = getattr(L, n); f.restype = r; f.argtypes = a
    rng = np.random.default_rng(1)
    ids = rng.integers(0, eng.vocab, size=ctx + warm + 1).astype(np.int32)
    eng.reset(); eng.set_tokens(ids); eng.run(ctx, greedy=0)
    L.mk_prof_enable(eng.h, 1)
    eng.run(1, greedy=0)
    slots = L.mk_prof_slots(eng.h)
    buf = (ctypes.c_ulonglong * slots)()
    n = L.mk_prof(eng.h, buf, slots)
    L.mk_prof_enable(eng.h, 0)
    t = np.array(buf[:n], dtype=np.uint64)
    t = t[t > 0]
    d = np.diff(t.astype(np.float64)) / 1000.0          # us per interval

    names = cfg["stages"]
    ns = len(names)
    nl = cfg["layers"]
    # [start][embed][ layers*ns ][lm_head]
    emb = d[0]
    body = d[1:1 + nl * ns].reshape(nl, ns)
    lm = d[1 + nl * ns] if len(d) > 1 + nl * ns else 0.0

    pred = np.array(cfg["stage_pred_us"], dtype=float)
    byt = np.array(cfg["stage_bytes"], dtype=float)
    tot = float(d.sum())
    print(f"-- measured stage schedule ({tot:.3f} us/token, ctx={ctx})")
    print(f"   {'stage':<10} {'meas us':>9} {'pred us':>9} {'err':>7} {'MB':>8} {'GB/s':>8} {'share':>7}")
    print(f"   {'embed':<10} {emb:>9.2f} {'-':>9} {'-':>7} {'-':>8} {'-':>8} {100*emb/tot:>6.1f}%")
    rows = []
    for i, nm in enumerate(names):
        m = body[:, i].sum()
        p = pred[i] * nl
        gbs = (byt[i] * nl / 1e9) / (m / 1e6) if m > 0 else 0
        print(f"   {nm:<10} {m:>9.2f} {p:>9.2f} {100*(m-p)/max(m,1e-9):>6.0f}% "
              f"{byt[i]*nl/1e6:>8.1f} {gbs:>8.0f} {100*m/tot:>6.1f}%")
        rows.append(dict(stage=nm, meas_us=float(m), pred_us=float(p), gbs=float(gbs)))
    lmb = cfg["lm_head_bytes"]
    print(f"   {'lm_head':<10} {lm:>9.2f} {cfg['lm_head_pred_us']:>9.2f} "
          f"{100*(lm-cfg['lm_head_pred_us'])/max(lm,1e-9):>6.0f}% {lmb/1e6:>8.1f} "
          f"{(lmb/1e9)/(lm/1e6) if lm>0 else 0:>8.0f} {100*lm/tot:>6.1f}%")
    rows.append(dict(stage="lm_head", meas_us=float(lm), pred_us=cfg["lm_head_pred_us"],
                     gbs=float((lmb/1e9)/(lm/1e6)) if lm > 0 else 0.0))
    return dict(total_us=tot, embed_us=float(emb), stages=rows)


def stage_bench(eng, cfg, ctx=512, reps=200):
    """Each stage as an ordinary kernel launch, cold weights, cycling layers.

    Three numbers per stage: what the planner predicted, what the stage costs
    standalone, and what it costs inside the megakernel.  The first gap is a
    cost-model error; the second is what fusion actually costs.
    """
    rng = np.random.default_rng(2)
    ids = rng.integers(0, eng.vocab, size=ctx + 8).astype(np.int32)
    eng.reset(); eng.set_tokens(ids); eng.run(ctx, greedy=0)
    names = [eng.stage_name(i) for i in range(eng.nstages())]
    pred = dict(zip(cfg["stages"], cfg["stage_pred_us"]))
    pred["lm_head"] = cfg["lm_head_pred_us"]
    byt = dict(zip(cfg["stages"], cfg["stage_bytes"]))
    byt["lm_head"] = cfg["lm_head_bytes"]
    print(f"-- standalone stage kernels (ctx={ctx}, {reps} reps, layer cycled)")
    print(f"   {'stage':<10} {'alone us':>9} {'pred us':>9} {'MB':>8} {'GB/s':>8}")
    out = []
    for i, nm in enumerate(names):
        us = eng.bench_stage(i, reps)
        b = byt.get(nm, 0)
        gbs = (b / 1e9) / (us / 1e6) if us > 0 and b else 0
        print(f"   {nm:<10} {us:>9.2f} {pred.get(nm, float('nan')):>9.2f} {b/1e6:>8.1f} {gbs:>8.0f}")
        out.append(dict(stage=nm, alone_us=us, pred_us=pred.get(nm), gbs=gbs))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=256); ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.out, "build.json")))
    eng = Engine(a.out, a.model, device=a.device, max_len=a.ctx + 64)
    r = profile(eng, cfg, ctx=a.ctx)
    r["standalone"] = stage_bench(eng, cfg, ctx=a.ctx)
    if a.json: json.dump(r, open(a.json, "w"), indent=1)
    eng.free()
