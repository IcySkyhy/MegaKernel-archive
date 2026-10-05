"""Extract a small MoE weight slice from the real Qwen3.8-Flash-Next-MXFP4
checkpoint into a tools/golden-compatible packed bin + json dataset.

Pulls, for one decoder layer (default 0) and a contiguous range of routed
experts (default 8 starting at expert 0):

    router gate weight              mlp.gate.weight                [E, 2560] bf16
    shared expert gate weight       mlp.shared_expert_gate.weight  [1, 2560] bf16
    routed expert gate_up (packed)  mlp.experts.gate_up_proj       [E,1280,1280] u8
    routed expert gate_up scales    mlp.experts.gate_up_proj.weight_scale
    routed expert down (packed)     mlp.experts.down_proj          [E,2560,320] u8
    routed expert down scales       mlp.experts.down_proj.weight_scale
    shared expert gate/up/down      mlp.shared_expert.{gate,up,down}_proj.weight(+scale)

The checkpoint tensors are already MXFP4-packed / bf16, so extraction is a
byte-verbatim slice — no requantization. Output files keep the exact
names/layout of the tools/golden dataset input bins (see
tools/golden/README.md), so the slice drops straight into
moe_block_ref.reference_moe_block and friends.

Default output: <script_dir>/data/layer{L}_e{a}-{b}/ (~22 MB for 8 experts,
far under the 2 GB budget). The extraction reads only the requested expert
rows through memmap views, i.e. ~8/512 of the two big shard tensors.

Usage:
    python3.12 extract_moe_slice.py \
        --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4 \
        --layer 0 --expert-offset 0 --num-experts 8 \
        --outdir data/layer0_e0-8

Pure numpy; reuses tools/golden/moe_block_ref.py for bin/json I/O.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "golden"))

import moe_block_ref as ref  # noqa: E402
from safetensors_reader import ShardReader  # noqa: E402

MODEL_PREFIX = "model.language_model"

# bin name -> checkpoint tensor suffix under layers.{L}.mlp (None = whole tensor)
ROUTED_EXPERT_TENSORS = {
    "experts.gate_up_proj": "experts.gate_up_proj",
    "experts.gate_up_proj.weight_scale": "experts.gate_up_proj.weight_scale",
    "experts.down_proj": "experts.down_proj",
    "experts.down_proj.weight_scale": "experts.down_proj.weight_scale",
}
WHOLE_TENSORS = {
    "router_weight": "gate.weight",
    "shared_expert_gate_weight": "shared_expert_gate.weight",
    "shared_expert.gate_proj": "shared_expert.gate_proj.weight",
    "shared_expert.gate_proj.weight_scale": "shared_expert.gate_proj.weight_scale",
    "shared_expert.up_proj": "shared_expert.up_proj.weight",
    "shared_expert.up_proj.weight_scale": "shared_expert.up_proj.weight_scale",
    "shared_expert.down_proj": "shared_expert.down_proj.weight",
    "shared_expert.down_proj.weight_scale": "shared_expert.down_proj.weight_scale",
}

BIN_OF_TENSOR = {
    "gate.weight": "router_weight.bin",
    "shared_expert_gate.weight": "shared_expert_gate_weight.bin",
}


def ckpt_name(layer: int, suffix: str) -> str:
    return f"{MODEL_PREFIX}.layers.{layer}.mlp.{suffix}"


def extract(
    model_dir: Path,
    layer: int,
    expert_offset: int,
    num_experts: int,
    outdir: Path,
) -> dict:
    a, b = expert_offset, expert_offset + num_experts
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reader = ShardReader(model_dir)
    mlp_prefix = f"{MODEL_PREFIX}.layers.{layer}.mlp."

    wanted = (
        [(ckpt_name(layer, s), "routed") for s in ROUTED_EXPERT_TENSORS.values()]
        + [(ckpt_name(layer, s), "whole") for s in WHOLE_TENSORS.values()]
    )
    missing = [n for n, _ in wanted if n not in reader.names()]
    incomplete = [n for n, _ in wanted
                  if n in reader.names() and not reader.is_complete(n)]
    if missing or incomplete:
        reader.close()
        raise SystemExit(
            "checkpoint slice not fully available yet "
            f"(model dir still downloading?):\n  missing: {missing}\n"
            f"  incomplete: {incomplete}\n"
            f"shards with incomplete tensors: {list(reader.incomplete)}"
        )

    files: list[dict] = []
    provenance: dict[str, dict] = {}

    def put(name: str, arr: np.ndarray, dtype_name: str, desc: str,
            src: str) -> None:
        d = ref.write_bin(outdir / name, arr, dtype_name)
        d.update({"kind": "input", "description": desc})
        with open(str(outdir / name) + ".json", "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")
        files.append(d)
        info = reader.info(src)
        provenance[name] = {
            "checkpoint_tensor": src,
            "shard": info.file.name,
            "source_dtype": info.st_dtype,
            "source_shape": list(info.shape),
            "slice": ([a, b] if src in
                      [ckpt_name(layer, s) for s in ROUTED_EXPERT_TENSORS.values()]
                      + [ckpt_name(layer, "gate.weight")] else None),
        }

    # --- routed experts: slice rows [a:b] of the first dimension ----------
    for bin_stem, suffix in ROUTED_EXPERT_TENSORS.items():
        src = ckpt_name(layer, suffix)
        arr = reader.load(src, slice(a, b))
        scale = suffix.endswith("weight_scale")
        put(bin_stem + ".bin", arr, "uint8",
            ("E8M0 group scales [E, N, K//32], bias 127, group 32 along K"
             if scale else
             "MXFP4 packed weights [E, N, K//2], lohi nibbles along K"),
            src)

    # --- router gate weight: rows [a:b], bf16 ------------------------------
    src = ckpt_name(layer, "gate.weight")
    put("router_weight.bin", reader.load_f32(src, slice(a, b)), "bf16",
        "router gate weight [num_experts, hidden], bf16", src)

    # --- shared expert + shared gate: whole tensors ------------------------
    for bin_stem, suffix in WHOLE_TENSORS.items():
        if bin_stem in ("router_weight",):
            continue
        src = ckpt_name(layer, suffix)
        if suffix == "shared_expert_gate.weight":
            bin_name, dt, desc = "shared_expert_gate_weight.bin", "bf16", \
                "shared expert gate weight [1, hidden], bf16"
        elif suffix.endswith("weight_scale"):
            bin_name, dt, desc = bin_stem + ".bin", "uint8", \
                "E8M0 group scales [N, K//32], bias 127, group 32 along K"
        else:
            bin_name, dt, desc = bin_stem + ".bin", "uint8", \
                "MXFP4 packed weights [N, K//2], lohi nibbles along K"
        put(bin_name, reader.load_f32(src) if dt == "bf16"
            else reader.load(src), dt, desc, src)

    manifest = {
        "format_version": ref.MANIFEST_VERSION,
        "name": outdir.name,
        "kind": "moe_weight_slice",
        "weights_only": True,
        "layer": layer,
        "expert_offset": expert_offset,
        "num_experts": num_experts,
        "hidden": ref.HIDDEN,
        "moe_intermediate": ref.MOE_INTERMEDIATE,
        "shared_expert_intermediate": ref.SHARED_INTERMEDIATE,
        "group_size": ref.GROUP_SIZE,
        "quantization": {
            "format": "mxfp4-pack-quantized-e8m0",
            "weight": "e2m1 nibbles, lohi byte order, group 32 along K",
            "scale": "e8m0 bias-127, scale = 2**ceil(log2(amax/6)) per group",
            "note": "byte-verbatim slice of the checkpoint; no requantization",
        },
        "source": {
            "model_dir": str(model_dir),
            "model_type": "qwen4_exp",
            "extracted_utc": datetime.now(timezone.utc).isoformat(),
            "tensors": provenance,
        },
        "files": files,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    reader.close()
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path,
                    default=Path("/workspace/Qwen3.8-Flash-Next-MXFP4"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert-offset", type=int, default=0)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default: <script_dir>/data/layer{L}_e{a}-{b}")
    args = ap.parse_args()

    a, b = args.expert_offset, args.expert_offset + args.num_experts
    outdir = args.outdir or HERE / "data" / f"layer{args.layer}_e{a}-{b}"
    manifest = extract(args.model_dir, args.layer, args.expert_offset,
                       args.num_experts, outdir)

    total = sum(f["size_bytes"] for f in manifest["files"])
    print(f"[extract] layer={args.layer} experts=[{a},{b}) -> {outdir}")
    for f in manifest["files"]:
        print(f"  {f['tensor']:<44} {str(f['shape']):<22} {f['size_bytes']:>10} B")
    print(f"[extract] total {total} B ({total / 1e6:.1f} MB), "
          f"{len(manifest['files'])} tensors")


if __name__ == "__main__":
    main()
