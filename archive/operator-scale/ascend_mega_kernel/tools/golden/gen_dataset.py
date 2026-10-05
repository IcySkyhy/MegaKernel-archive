"""Generate the deterministic MoE block test datasets (inputs + golden).

Usage:
    python3.12 gen_dataset.py [--outdir DIR] [--groups m1,m33,real]

Groups (name -> m, experts, top_k, seed):
    m1    1   4    2   20260926   decode-shaped slice, committed
    m33   33  4    2   20260927   prefill-shaped slice (odd m), committed
    real  8   512  10  20260928   real topology (E=512, topK=10); 1.25 GiB = 1280.8 MiB of
                                  weights, generated on demand and not committed (README)

All groups keep the real model K/N: hidden=2560, moe_intermediate=640,
shared_expert_intermediate=640. m1/m33 shrink only the expert count / top-k; the real
group keeps the true 512 experts and topK=10.

Besides the float golden, every group carries one quant-aware golden per activation
quantization rule:

  * ``qaware/floor`` — **default and authoritative**: the official ops-nn MXFP4 sequence
    (OCP, ``npu_dynamic_mx_quant(round_mode="round")``), transcribed line by line in
    ``moe_block_ref.quantize_ocp`` (user ruling 2026-09-26, docs/13);
  * ``qaware/ceil`` — **legacy**: the replaced m3 scalar quantizer's rule, kept only so
    pre-ruling goldens stay reproducible.

See ``qaware_ref.py`` and the README for the tensor set, the deviation budget and the
device cross-check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from moe_block_ref import generate_dataset

HERE = Path(__file__).resolve().parent

# name -> (m, num_experts, top_k, seed); seeds fixed for reproducibility.
# The real group needs 1.25 GiB of weights, so it is not committed, only generated.
SPECS = {
    "m1": (1, 4, 2, 20260926),
    "m33": (33, 4, 2, 20260927),
    "real": (8, 512, 10, 20260928),
}
COMMITTED = ("m1", "m33")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=HERE / "data",
                    help="output root (default: <script_dir>/data)")
    ap.add_argument("--groups", type=str, default=",".join(COMMITTED),
                    help="comma-separated group names to generate "
                         f"(available: {','.join(SPECS)}; default: {','.join(COMMITTED)})")
    ap.add_argument("--rules", type=str, default="floor,ceil",
                    help="activation quantization rules for the quant-aware golden "
                         "(comma-separated subset of floor,ceil; empty = skip; "
                         "floor = official ops-nn sequence, ceil = legacy)")
    args = ap.parse_args()

    rules = tuple(r for r in args.rules.split(",") if r)
    for name in args.groups.split(","):
        if not name:
            continue
        m, e, k, seed = SPECS[name]
        outdir = args.outdir / name
        manifest = generate_dataset(outdir, m, e, k, seed, rules=rules)
        n_files = len(manifest["files"])
        total = sum(f["size_bytes"] for f in manifest["files"])
        print(f"[gen] {outdir}: m={m} E={e} top_k={k} seed={seed} "
              f"({n_files} tensors, {total / 2**20:.1f} MiB)")
        for rule, info in manifest["quantization"]["qaware_golden"].items():
            print(f"[gen]   qaware/{rule}: {info['tensors']} tensors; "
                  f"routed/shared/moe measured-vs-float max_abs = "
                  f"{[round(v['measured_max_abs'], 4) for v in info['summary'].values()]}, "
                  f"ceiling = {[round(v['ceiling'], 2) for v in info['summary'].values()]}")
        if name == "real":
            print(f"[gen]   note: '{name}' weights are not committed — they are "
                  f"reproducible from seed {seed} via this script")
    print("[gen] done")


if __name__ == "__main__":
    main()
