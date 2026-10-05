#!/usr/bin/env python3.12
"""从 Qwen3.8-Flash-Next-MXFP4 checkpoint 抽取一层 QSA indexer 权重。

输出（bf16 原始字节，行主序，与 checkpoint 逐字节一致）：
  data/layer{L}_index_qk_proj.bin  bf16 [640, 2560]   = self_attn.indexer.index_qk_proj.weight
  data/layer{L}_q_layernorm.bin    bf16 [128]         = self_attn.indexer.q_layernorm.weight
  data/layer{L}_k_layernorm.bin    bf16 [128]         = self_attn.indexer.k_layernorm.weight

用法：
  /usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/extract_indexer_weight.py [--layer 3]
"""

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "tools" / "weights"))

from safetensors_reader import ShardReader  # noqa: E402

MODEL_DIR = "/workspace/Qwen3.8-Flash-Next-MXFP4"

TENSORS = {
    "index_qk_proj": ("self_attn.indexer.index_qk_proj.weight", (640, 2560)),
    "q_layernorm": ("self_attn.indexer.q_layernorm.weight", (128,)),
    "k_layernorm": ("self_attn.indexer.k_layernorm.weight", (128,)),
}


def prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()

    rdr = ShardReader(args.model_dir)
    out_dir = HERE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"model_dir": args.model_dir, "layer": args.layer, "tensors": {}}

    for short, (suffix, shape) in TENSORS.items():
        name = f"{prefix(args.layer)}.{suffix}"
        info = rdr.info(name)
        if tuple(info.shape) != shape:
            raise SystemExit(f"{name}: shape {info.shape} != expected {shape}")
        if info.st_dtype != "BF16":
            raise SystemExit(f"{name}: dtype {info.st_dtype} != BF16")
        raw = rdr.load(name)  # uint16 bit view, zero copy
        blob = raw.tobytes()
        path = out_dir / f"layer{args.layer}_{short}.bin"
        path.write_bytes(blob)
        manifest["tensors"][short] = {
            "source": name,
            "shape": list(shape),
            "dtype": "bf16",
            "file": path.name,
            "nbytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        print(f"[extract] {name} {shape} -> {path.name} ({len(blob)} bytes)")

    (out_dir / f"layer{args.layer}_indexer_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"[extract] manifest -> data/layer{args.layer}_indexer_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
