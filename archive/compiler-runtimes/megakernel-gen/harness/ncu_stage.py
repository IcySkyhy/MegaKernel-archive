"""Run one stage kernel a few times so `ncu` has something to profile."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mkrun import Engine

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--model", required=True)
ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--stages", default="")
a = ap.parse_args()
eng = Engine(a.out, a.model, device=0, max_len=a.ctx + 64)
rng = np.random.default_rng(2)
ids = rng.integers(0, eng.vocab, size=a.ctx + 8).astype(np.int32)
eng.reset(); eng.set_tokens(ids); eng.run(a.ctx, greedy=0)
want = set(a.stages.split(",")) if a.stages else None
for i in range(eng.nstages()):
    nm = eng.stage_name(i)
    if want and nm not in want:
        continue
    eng.bench_stage(i, a.reps)
eng.free()
