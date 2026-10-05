"""M75 probe (read-only): what does the frozen golden do under a *different* numeric stack?

Two independent readings against the committed `data/**.bin` -- nothing is written and no
criterion is evaluated here; the script only *measures*, so a reader can tell a real `.bin`
drift apart from a numpy/BLAS change:

  * bf16 code drift of the float golden recomputed from the loaded bins (per group, per
    tensor), as `differing/total` and the max code delta in bf16 ulp;
  * the q-aware `measured.max_abs` values the manifest stores vs. what this stack
    recomputes, i.e. the judgement `selfcheck.py` makes with a 1e-9 tolerance.

Usage:
    python3.12 evidence/numeric_env_probe.py [--data DIR] [--groups m1,m33] [--tolerance 1e-9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import moe_block_ref as ref  # noqa: E402
import qaware_ref as qr  # noqa: E402

MEASURED_KEYS = ("routed_output", "shared_output", "moe_output", "y_final", "res2")


def _packed(t: dict) -> tuple:
    experts = {k: t[k + ".bin"] for k in
               ("experts.gate_up_proj", "experts.gate_up_proj.weight_scale",
                "experts.down_proj", "experts.down_proj.weight_scale")}
    shared = {k: t[k + ".bin"] for k in
              ("shared_expert.gate_proj", "shared_expert.gate_proj.weight_scale",
               "shared_expert.up_proj", "shared_expert.up_proj.weight_scale",
               "shared_expert.down_proj", "shared_expert.down_proj.weight_scale")}
    return experts, shared


def report_environment() -> None:
    blas = np.__config__.CONFIG["Build Dependencies"]["blas"]
    print(f"interpreter {sys.executable}")
    print(f"numpy {np.__version__}")
    print(f"blas {blas['name']} {blas['version']}")


def code_drift(data: Path, name: str) -> None:
    ds = ref.load_dataset(data / name)
    t, man = ds["tensors"], ds["manifest"]
    experts, shared = _packed(t)
    recomputed = ref.reference_moe_block(
        t["x.bin"], t["router_weight.bin"], t["shared_expert_gate_weight.bin"],
        experts, shared, man["top_k"])
    for key in ("routed_output", "shared_output", "moe_output"):
        a = ref.f32_to_bf16_bits(t[key + ".bin"]).astype(np.int32)
        b = ref.f32_to_bf16_bits(recomputed[key]).astype(np.int32)
        delta = np.abs(a - b)
        print(f"  {name}/{key}: codes differing {int((delta != 0).sum())}/{delta.size} "
              f"max code delta {int(delta.max())}")


def max_abs_tolerance(data: Path, name: str, tol: float) -> int:
    outdir = data / name
    ds = ref.load_dataset(outdir)
    t, man = ds["tensors"], ds["manifest"]
    experts, shared = _packed(t)
    layer = {"x_res": t["x_res.bin"], "gamma1": t["gamma1.bin"].reshape(-1),
             "gamma2": t["gamma2.bin"].reshape(-1)}
    float_ref = dict(ref.reference_moe_block(
        t["x.bin"], t["router_weight.bin"], t["shared_expert_gate_weight.bin"],
        experts, shared, man["top_k"]))
    float_ref["res1"], float_ref["x_norm1"] = ref.rmsnorm_ref(layer["x_res"], layer["gamma1"])
    float_ref["res2"], float_ref["y_final"] = ref.rmsnorm_ref(
        ref.f32_to_bf16(float_ref["moe_output"]), layer["gamma2"], residual=float_ref["res1"])
    bad = 0
    for rule in sorted(man["quantization"]["qaware_golden"]):
        q = qr.qaware_moe_block(t["x.bin"], t["router_weight.bin"],
                                t["shared_expert_gate_weight.bin"], experts, shared,
                                man["top_k"], rule=rule, layer=layer)
        report = qr.deviation_report(float_ref, q, t["x.bin"], experts, shared,
                                     rule=rule, layer=layer)
        stored = qr.load_qaware(outdir / "qaware" / rule)["manifest"]["deviation_budget"]
        for tname in MEASURED_KEYS:
            if tname not in stored["measured"]:
                continue
            a = stored["measured"][tname]["max_abs"]
            b = report["measured"][tname]["max_abs"]
            over = abs(a - b) > tol
            bad += int(over)
            print(f"  {name}/qaware/{rule} {tname}: stored {a:.9g} recomputed {b:.9g} "
                  f"|d| {abs(a - b):.3e} {'FAIL(>%g)' % tol if over else 'ok'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--groups", type=str, default="m1,m33")
    ap.add_argument("--tolerance", type=float, default=1e-9,
                    help="the selfcheck.py tolerance for stored measured max_abs")
    args = ap.parse_args()
    report_environment()
    for name in [g for g in args.groups.split(",") if g]:
        code_drift(args.data, name)
    over = 0
    for name in [g for g in args.groups.split(",") if g]:
        over += max_abs_tolerance(args.data, name, args.tolerance)
    print(f"measured.max_abs over the {args.tolerance:g} tolerance: {over}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
