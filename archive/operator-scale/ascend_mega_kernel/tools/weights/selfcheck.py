"""Self-checks for the extracted MoE weight slice (tools/weights).

Checks, against an extract_moe_slice.py output directory:
  1. Bin/json integrity: every manifest file loads back with matching
     shape/dtype and byte size.
  2. Byte-provenance (if the source checkpoint is reachable): each bin is a
     verbatim slice of the named checkpoint tensor — packed bins compare
     byte-for-byte, bf16 bins compare bit-for-bit.
  3. MXFP4 round trip (M4 contract): unpack -> pack -> unpack reproduces
     the same dequantized values for every routed-expert and shared-expert
     tensor: unpack(pack(unpack(p, s))) == unpack(p, s) exactly.
  4. Weight statistics: dequantized mean/std reported per tensor and
     checked against the M4 sanity baseline (synthetic expert weights
     std=0.03, router/shared-gate std=0.02): same order of magnitude,
     |mean| small vs std, all finite and nonzero.
  5. Output size under the 2 GB budget.
  6. tools/golden compatibility: the slice plugs into
     moe_block_ref.reference_moe_block — seeded random activations run the
     full router -> top-k -> dequant -> SwiGLU -> combine pipeline and
     produce finite, statistically sane outputs.

Usage:
    python3.12 selfcheck.py [--data DIR] [--model-dir DIR]

Defaults: --data <script_dir>/data/layer0_e0-8,
          --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4
(check 2 is skipped when the model dir is absent.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "golden"))
sys.path.insert(0, str(HERE))

import moe_block_ref as ref  # noqa: E402
from safetensors_reader import ShardReader  # noqa: E402

# M4 sanity baseline: synthetic dataset weights used std=0.03 for expert
# weights and std=0.02 for router / shared-expert-gate (moe_block_ref.py).
M4_BASELINE_STD = {"expert": 0.03, "router": 0.02}
# Real trained weights are allowed to sit within this multiplicative band
# around the baseline (i.e. same order of magnitude: 0.1x..10x).
STD_BAND = (0.1, 10.0)
MAX_MEAN_OVER_STD = 0.25
SIZE_BUDGET = 2 * 1024**3

# bin -> (kind) kind in {expert, shared, router, shared_gate}
PACKED_BINS = {
    "experts.gate_up_proj.bin": "expert",
    "experts.gate_up_proj.weight_scale.bin": "expert",
    "experts.down_proj.bin": "expert",
    "experts.down_proj.weight_scale.bin": "expert",
    "shared_expert.gate_proj.bin": "shared",
    "shared_expert.gate_proj.weight_scale.bin": "shared",
    "shared_expert.up_proj.bin": "shared",
    "shared_expert.up_proj.weight_scale.bin": "shared",
    "shared_expert.down_proj.bin": "shared",
    "shared_expert.down_proj.weight_scale.bin": "shared",
    "router_weight.bin": "router",
    "shared_expert_gate_weight.bin": "shared_gate",
}

# paired (packed, scales) bins for the round-trip check
PACKED_PAIRS = [
    ("experts.gate_up_proj.bin", "experts.gate_up_proj.weight_scale.bin",
     "experts"),
    ("experts.down_proj.bin", "experts.down_proj.weight_scale.bin",
     "experts"),
    ("shared_expert.gate_proj.bin", "shared_expert.gate_proj.weight_scale.bin",
     None),
    ("shared_expert.up_proj.bin", "shared_expert.up_proj.weight_scale.bin",
     None),
    ("shared_expert.down_proj.bin", "shared_expert.down_proj.weight_scale.bin",
     None),
]


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise SystemExit(f"selfcheck failed: {msg}")


def load_slice(datadir: Path) -> tuple[dict, dict]:
    with open(datadir / "manifest.json") as f:
        manifest = json.load(f)
    tensors = {}
    for d in manifest["files"]:
        tensors[d["tensor"]] = ref.read_bin(datadir / d["tensor"], d)
    return manifest, tensors


def check_bins(datadir: Path, manifest: dict, tensors: dict) -> None:
    print("check 1: bin/json integrity")
    for d in manifest["files"]:
        p = datadir / d["tensor"]
        check(p.exists() and p.stat().st_size == d["size_bytes"],
              f"{d['tensor']}: {p.stat().st_size if p.exists() else 'missing'}"
              f" == {d['size_bytes']} bytes")
        check(list(tensors[d["tensor"]].shape) == d["shape"],
              f"{d['tensor']}: shape {tensors[d['tensor']].shape} "
              f"== {d['shape']}")


def check_provenance(datadir: Path, manifest: dict, tensors: dict,
                     model_dir: Path) -> None:
    print("check 2: byte-provenance vs source checkpoint")
    if not model_dir.is_dir():
        print(f"  [SKIP] model dir {model_dir} not reachable")
        return
    layer = manifest["layer"]
    a, b = manifest["expert_offset"], manifest["expert_offset"] + manifest["num_experts"]
    reader = ShardReader(model_dir)
    try:
        prov = manifest["source"]["tensors"]
        for bin_name, kind in PACKED_BINS.items():
            src = prov[bin_name]["checkpoint_tensor"]
            if kind in ("expert", "router"):
                fresh = reader.load(src, slice(a, b))
            else:
                fresh = reader.load(src)
            got = np.fromfile(datadir / bin_name, dtype=np.uint8)
            want = np.ascontiguousarray(fresh).view(np.uint8).reshape(-1)
            check(np.array_equal(got, want),
                  f"{bin_name}: byte-verbatim slice of {src}")
    finally:
        reader.close()


def check_roundtrip(tensors: dict) -> None:
    print("check 3: MXFP4 round trip unpack(pack(unpack(p,s))) == unpack(p,s)")
    for packed_bin, scales_bin, axis in PACKED_PAIRS:
        packed = tensors[packed_bin]
        scales = tensors[scales_bin]
        if axis == "experts":
            for e in range(packed.shape[0]):
                w1 = ref.unpack_mxfp4(packed[e], scales[e])
                w2 = ref.unpack_mxfp4(*ref.pack_mxfp4(w1))
                check(np.array_equal(w2, w1),
                      f"{packed_bin}[{e}]: dequant fixed point")
        else:
            w1 = ref.unpack_mxfp4(packed, scales)
            w2 = ref.unpack_mxfp4(*ref.pack_mxfp4(w1))
            check(np.array_equal(w2, w1), f"{packed_bin}: dequant fixed point")


def check_stats(tensors: dict) -> None:
    print("check 4: dequantized weight statistics vs M4 sanity baseline")
    for packed_bin, scales_bin, axis in PACKED_PAIRS:
        packed, scales = tensors[packed_bin], tensors[scales_bin]
        per = [ref.unpack_mxfp4(packed[e], scales[e])
               for e in range(packed.shape[0])] if axis == "experts" \
            else [ref.unpack_mxfp4(packed, scales)]
        w = np.stack(per).ravel()
        mean, std = float(w.mean()), float(w.std())
        finite = np.isfinite(w).all()
        base = M4_BASELINE_STD["expert"] if axis == "experts" else \
            M4_BASELINE_STD["router"]
        ok = (finite and std > 0 and
              STD_BAND[0] * base <= std <= STD_BAND[1] * base and
              abs(mean) <= MAX_MEAN_OVER_STD * std)
        check(ok, f"{packed_bin}: mean={mean:+.5f} std={std:.5f} "
                  f"(M4 baseline std={base}, band {STD_BAND})")
    for bin_name, base in (("router_weight.bin", M4_BASELINE_STD["router"]),
                           ("shared_expert_gate_weight.bin",
                            M4_BASELINE_STD["router"])):
        w = tensors[bin_name].astype(np.float32).ravel()
        mean, std = float(w.mean()), float(w.std())
        ok = (np.isfinite(w).all() and std > 0 and
              STD_BAND[0] * base <= std <= STD_BAND[1] * base and
              abs(mean) <= MAX_MEAN_OVER_STD * std)
        check(ok, f"{bin_name}: mean={mean:+.5f} std={std:.5f} "
                  f"(M4 baseline std={base}, band {STD_BAND})")


def check_size(datadir: Path, manifest: dict) -> None:
    print("check 5: output size budget")
    total = sum(f["size_bytes"] for f in manifest["files"])
    check(total < SIZE_BUDGET,
          f"total {total} B ({total / 1e6:.1f} MB) < 2 GB budget")


def check_golden_compat(manifest: dict, tensors: dict) -> None:
    print("check 6: tools/golden compatibility (reference_moe_block run)")
    e = manifest["num_experts"]
    top_k = min(2, e)  # same top-k=2 shape as the golden datasets
    rng = np.random.default_rng(20260926)
    x = ref.f32_to_bf16(rng.standard_normal((5, ref.HIDDEN),
                                            dtype=np.float32))
    experts_packed = {
        "experts.gate_up_proj": tensors["experts.gate_up_proj.bin"],
        "experts.gate_up_proj.weight_scale":
            tensors["experts.gate_up_proj.weight_scale.bin"],
        "experts.down_proj": tensors["experts.down_proj.bin"],
        "experts.down_proj.weight_scale":
            tensors["experts.down_proj.weight_scale.bin"],
    }
    shared_packed = {
        "shared_expert.gate_proj": tensors["shared_expert.gate_proj.bin"],
        "shared_expert.gate_proj.weight_scale":
            tensors["shared_expert.gate_proj.weight_scale.bin"],
        "shared_expert.up_proj": tensors["shared_expert.up_proj.bin"],
        "shared_expert.up_proj.weight_scale":
            tensors["shared_expert.up_proj.weight_scale.bin"],
        "shared_expert.down_proj": tensors["shared_expert.down_proj.bin"],
        "shared_expert.down_proj.weight_scale":
            tensors["shared_expert.down_proj.weight_scale.bin"],
    }
    out = ref.reference_moe_block(
        x, tensors["router_weight.bin"],
        tensors["shared_expert_gate_weight.bin"],
        experts_packed, shared_packed, top_k,
    )
    moe = out["moe_output"]
    check(moe.shape == (5, ref.HIDDEN), f"moe_output shape {moe.shape}")
    check(np.isfinite(moe).all(), "moe_output finite")
    check(float(moe.std()) > 0, f"moe_output alive (std={float(moe.std()):.4f})")
    ids = out["topk_ids"]
    check(np.all(ids >= 0) and np.all(ids < e), "topk_ids within expert range")
    wsum = out["topk_weights"].sum(axis=1)
    check(np.allclose(wsum, 1.0, atol=1e-6), "topk_weights renormalized to 1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=HERE / "data" / "layer0_e0-8")
    ap.add_argument("--model-dir", type=Path,
                    default=Path("/workspace/Qwen3.8-Flash-Next-MXFP4"))
    args = ap.parse_args()

    manifest, tensors = load_slice(args.data)
    check_bins(args.data, manifest, tensors)
    check_provenance(args.data, manifest, tensors, args.model_dir)
    check_roundtrip(tensors)
    check_stats(tensors)
    check_size(args.data, manifest)
    check_golden_compat(manifest, tensors)
    print("all selfchecks passed")


if __name__ == "__main__":
    main()
