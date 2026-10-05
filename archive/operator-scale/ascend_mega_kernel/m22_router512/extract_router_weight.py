"""Extract the layer-0 router gate weight [512, 2560] bf16 from the
Qwen3.8-Flash-Next-MXFP4 checkpoint into m22_router512/data/router_weight.bin.

M42 任务 1 要求「保留真实 checkpoint `mlp.gate.weight` 切片与 moe_block_ref.py 对拍
能力」。本脚本与 m7_router_topk/extract_router_weight.py 抽同一张量、同一工具链
（tools/weights/safetensors_reader.py），并对 m7 的切片做 **sha256 逐字节互证** ——
两份切片必须完全相同，否则报错退出（证明「真实 checkpoint 逐字节」这条判据的取值
来源在我这个分支里可独立复现）。

Usage:
    /usr/local/python3.12.13/bin/python3.12 m22_router512/extract_router_weight.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools" / "weights"))
sys.path.insert(0, str(REPO / "tools" / "golden"))

from moe_block_ref import write_bin, read_bin  # noqa: E402
from safetensors_reader import ShardReader  # noqa: E402

MODEL_DIR = "/workspace/Qwen3.8-Flash-Next-MXFP4"
TENSOR = "model.language_model.layers.0.mlp.gate.weight"
OUT = REPO / "m22_router512" / "data" / "router_weight.bin"
M7_COPY = REPO / "m7_router_topk" / "data" / "router_weight.bin"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rdr = ShardReader(MODEL_DIR)
    info = rdr.info(TENSOR)
    assert info.st_dtype == "BF16" and tuple(info.shape) == (512, 2560), info
    w = rdr.load_f32(TENSOR)  # exact bf16 -> f32 upcast, [512, 2560]
    desc = write_bin(OUT, w, "bf16")
    # round-trip check: stored bits must be byte-verbatim bf16
    back = read_bin(OUT, desc)
    assert (back == w).all(), "round-trip mismatch"
    digest = sha256(OUT)
    print(f"wrote {OUT} shape={desc['shape']} dtype={desc['dtype']} "
          f"({desc['size_bytes']} bytes), round-trip ok")
    print(f"sha256(m22) = {digest}")
    if M7_COPY.exists():
        m7_digest = sha256(M7_COPY)
        print(f"sha256(m7 ) = {m7_digest}")
        assert digest == m7_digest, "m22 slice differs from m7 slice (byte-verbatim check)"
        print("m22 slice == m7 slice, byte-verbatim OK")


if __name__ == "__main__":
    main()
