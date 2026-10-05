"""Extract the full-layer-0 router gate weight [512, 2560] bf16 from the
Qwen3.8-Flash-Next-MXFP4 checkpoint into m7_router_topk/data/router_weight.bin
(golden-compatible bin + json, see tools/golden/README.md).

tools/weights/data/layer0_e0-8/router_weight.bin holds only the first 8 expert
rows (that slice was cut at 8 experts to keep the expert-weight bins small);
the routing semantics for M7 need all 512 expert rows, so this script re-extracts
the same checkpoint tensor without the row slice, byte-verbatim.

Usage:
    /usr/local/python3.12.13/bin/python3.12 m7_router_topk/extract_router_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools" / "weights"))
sys.path.insert(0, str(REPO / "tools" / "golden"))

from moe_block_ref import write_bin, read_bin, f32_to_bf16  # noqa: E402
from safetensors_reader import ShardReader  # noqa: E402

MODEL_DIR = "/workspace/Qwen3.8-Flash-Next-MXFP4"
TENSOR = "model.language_model.layers.0.mlp.gate.weight"
OUT = REPO / "m7_router_topk" / "data" / "router_weight.bin"


def main() -> None:
    rdr = ShardReader(MODEL_DIR)
    info = rdr.info(TENSOR)
    assert info.st_dtype == "BF16" and tuple(info.shape) == (512, 2560), info
    w = rdr.load_f32(TENSOR)  # exact bf16 -> f32 upcast, [512, 2560]
    desc = write_bin(OUT, w, "bf16")
    # round-trip check: stored bits must be byte-verbatim bf16
    back = read_bin(OUT, desc)
    assert (back == w).all(), "round-trip mismatch"
    print(f"wrote {OUT} shape={desc['shape']} dtype={desc['dtype']} "
          f"({desc['size_bytes']} bytes), round-trip ok")


if __name__ == "__main__":
    main()
