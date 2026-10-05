"""
guard.py -- the adversarial checks.

The gate answers "does it match the reference".  These answer the question the
gate cannot: "could it have matched without doing the work?"  Latency is
trivially cheatable -- the fastest kernel is the one that computes nothing --
and a compiler that is allowed to see its own score will find any hole left in
the metric.  So each check below is an attack on a specific way of cheating,
and every one of them must fail to succeed.

  weight_sensitivity   corrupt one weight tensor at a time and require the
                       output to move.  A layer that is never read cannot be
                       detected by any end-to-end comparison; this detects it
                       directly, and it is run on the FIRST, a MIDDLE and the
                       LAST layer so that a truncated loop cannot hide.
  history_sensitivity  change an earlier token and require a later step's
                       logits to move: proves the KV cache is consulted rather
                       than the model attending to nothing.
  rope_applied         zero the rotary tables and require the output to move.
                       (The obvious "same token, two positions" test is wrong:
                       attention over identical values is genuinely position
                       invariant, and the reference agrees bit for bit.)
  input_sensitivity    different inputs, different outputs: no memoisation.
  determinism          the same input twice must give bit-identical logits:
                       a run-to-run difference means a race, and a race means
                       the correctness result was luck.
  timing_floor         measured ms/token cannot be below the roofline of the
                       bytes the model must read.  Physics as an assertion.
  no_reference_link    the shared object must not link torch or transformers,
                       and the generated sources must contain no captured
                       output.  The engine cannot consult the answer key.
"""
import ctypes, glob, os, re, subprocess
import numpy as np


def _logits_for(eng, ids, upto=None):
    """Run a fresh sequence from position 0 and return the last step's logits."""
    n = len(ids) if upto is None else upto
    eng.reset()
    eng.set_tokens(np.asarray(ids, dtype=np.int32))
    eng.run(n, greedy=0)
    return eng.logits().copy()


def _rel_change(a, b):
    d = np.abs(a - b).max()
    s = max(np.abs(a).max(), 1e-9)
    return float(d / s)


def check_weight_sensitivity(eng, rng, sample=None):
    """Zero one whole weight tensor at a time; the output must move.

    This is the check an end-to-end logit comparison cannot make.  A kernel that
    silently runs 20 of 24 layers, or drops the up-projection of every expert,
    still produces plausible logits -- and if the compiler is ever allowed to
    optimise against a score, that is precisely the shape of the shortcut it
    will find.  Zeroing the WHOLE tensor (rather than a slice) is what makes the
    question exact: after this, no read of it can affect anything.

    The bytes are put back by re-reading them from the checkpoint, so the model
    is bit-identical afterwards -- which is itself asserted.
    """
    ws = eng.weights()
    n = len(ws)
    idx = list(range(n))
    if sample and sample < n:
        idx = sorted(rng.choice(n, size=sample, replace=False).tolist())
    ids = rng.integers(0, eng.vocab, size=6).astype(np.int32)
    base = _logits_for(eng, ids)
    fails, worst = [], (1e9, "")
    for i in idx:
        role, off, nb = ws[i]
        eng.zero_weight(i)
        got = _logits_for(eng, ids)
        eng.reload_weight(i)
        ch = _rel_change(base, got)
        if ch < 1e-6:
            fails.append(role)
        if ch < worst[0]:
            worst = (ch, role)
    restored = _logits_for(eng, ids)
    if not np.array_equal(base, restored):
        fails.append("RESTORE-NOT-BIT-EXACT")
    return dict(name="weight_sensitivity", passed=len(fails) == 0,
                probed=len(idx), of=n, insensitive=fails[:12],
                least_sensitive=f"{worst[1]}@{worst[0]:.2e}")


def check_history_sensitivity(eng, rng):
    """Flipping an early token must move a later step's logits."""
    ids = rng.integers(0, eng.vocab, size=8).astype(np.int32)
    a = _logits_for(eng, ids)
    ids2 = ids.copy()
    ids2[1] = (ids2[1] + 12345) % eng.vocab
    b = _logits_for(eng, ids2)
    ch = _rel_change(a, b)
    return dict(name="history_sensitivity", passed=ch > 1e-3, change=ch)


def check_rope_applied(eng, rng):
    """Zero the rotary tables; the output must move.

    The obvious position test -- feed the same token at two positions and demand
    different logits -- is WRONG, and the reference proves it: attention over
    identical value vectors returns that value whatever the scores are, so a
    repeated token is genuinely position-invariant and `transformers` produces
    bit-identical logits too.  Testing the tables directly asks the question
    that was actually meant.
    """
    ids = rng.integers(0, eng.vocab, size=6).astype(np.int32)
    base = _logits_for(eng, ids)
    eng.rope_zero()
    got = _logits_for(eng, ids)
    eng.rope_rebuild()
    restored = _logits_for(eng, ids)
    ch = _rel_change(base, got)
    return dict(name="rope_applied", passed=ch > 1e-3 and np.array_equal(base, restored),
                change=ch, restored=bool(np.array_equal(base, restored)))


def check_input_sensitivity(eng, rng):
    a = _logits_for(eng, rng.integers(0, eng.vocab, size=5).astype(np.int32))
    b = _logits_for(eng, rng.integers(0, eng.vocab, size=5).astype(np.int32))
    ch = _rel_change(a, b)
    return dict(name="input_sensitivity", passed=ch > 1e-3, change=ch)


def check_determinism(eng, rng):
    ids = rng.integers(0, eng.vocab, size=6).astype(np.int32)
    a = _logits_for(eng, ids)
    b = _logits_for(eng, ids)
    same = bool(np.array_equal(a, b))
    return dict(name="determinism", passed=same,
                max_diff=float(np.abs(a - b).max()))


def check_timing_floor(cfg, ms):
    """No kernel reads its weights faster than the memory system allows."""
    if ms is None:
        return dict(name="timing_floor", passed=True, note="not measured")
    floor = cfg["roofline_ms"]
    return dict(name="timing_floor", passed=ms >= floor * 0.98,
                measured_ms=ms, roofline_ms=floor)


def check_no_reference_link(outdir):
    """The engine must not be able to consult the answer."""
    bad = []
    so = os.path.join(outdir, "libmk.so")
    try:
        out = subprocess.run(["ldd", so], capture_output=True, text=True).stdout
        for lib in ("torch", "transformers", "python"):
            if lib in out:
                bad.append(f"libmk.so links {lib}")
    except Exception as e:
        bad.append(f"ldd failed: {e}")
    for f in glob.glob(os.path.join(outdir, "*.cu")) + glob.glob(os.path.join(outdir, "*.h")):
        src = open(f, errors="ignore").read()
        for pat in ("gate_ref", "gate_got", "reference_logits", "expected_logits"):
            if pat in src:
                bad.append(f"{os.path.basename(f)} mentions {pat}")
    return dict(name="no_reference_link", passed=len(bad) == 0, problems=bad)


def run_guard(eng, model_dir, cfg, rng, measured_ms=None, weight_sample=None):
    print("-- guard: adversarial checks")
    checks = [
        check_no_reference_link(eng.dir),
        check_determinism(eng, rng),
        check_input_sensitivity(eng, rng),
        check_rope_applied(eng, rng),
        check_history_sensitivity(eng, rng),
        check_weight_sensitivity(eng, rng, sample=weight_sample),
        check_timing_floor(cfg, measured_ms),
    ]
    for c in checks:
        extra = {k: v for k, v in c.items() if k not in ("name", "passed")}
        print(f"   {'PASS' if c['passed'] else 'FAIL'}  {c['name']:<22} {extra}")
    return dict(checks=checks, passed=all(c["passed"] for c in checks))
