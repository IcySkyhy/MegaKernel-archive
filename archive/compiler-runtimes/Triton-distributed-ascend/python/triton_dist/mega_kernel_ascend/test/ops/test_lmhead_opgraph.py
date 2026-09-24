# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# OpGraph docking sample: Ascend-shaped final_norm + lm_head (ranks==1).
#
# Mirrors Ascend ``buildLMHead(mlpOutput, residual)`` without AllGather:
#   add_rms_norm -> StaticOpResult<3>: y[M,H], x[M,H], rstd[M,1]
#   matmul_lmhead(y, lm_head.weight) -> logits
#
# Ascend tiling derives dimExtents from op.outputs -> Tensor.desc.shape; the bin
# must carry non-empty shapes on every add_rms_norm output (ports y/x/rstd).
#
# Weight layout follows Ascend: lm_head.weight is ``[H, V]`` (not ``[V, H]``).
# Default shapes are Qwen3-30B-A3B decode (B*S=1, H=2048, V=151936).
#
# Run (``-p`` required; ``--write-golden`` regenerates checked-in bins):
#   python -m triton_dist.mega_kernel_ascend.test.ops.test_lmhead_opgraph -p A3
#   python -m triton_dist.mega_kernel_ascend.test.ops.test_lmhead_opgraph -p 950
#   python -m triton_dist.mega_kernel_ascend.test.ops.test_lmhead_opgraph -p A3 --write-golden
from __future__ import annotations

import argparse
import os
import tempfile

import torch

from triton_dist.mega_kernel_ascend.models_blade.model_builder import ModelBuilder
from triton_dist.mega_triton_kernel.core.op_graph import (
    MemoryPool,
    build_opgraph_from_graph,
)

_OPS_DIR = os.path.dirname(__file__)
_EXPECTED_OPGRAPH = {
    "A3": os.path.join(_OPS_DIR, "lmhead_opgraph.expected.bin"),
    "950": os.path.join(_OPS_DIR, "lmhead_opgraph.950.expected.bin"),
}
_PLATFORMS = tuple(sorted(_EXPECTED_OPGRAPH))

_NORM_WEIGHT_EXTERNAL_ADDR = 0x12C0C0011000
_LM_HEAD_WEIGHT_EXTERNAL_ADDR = 0x12C0C0012000

_DEFAULT_M = 1
_DEFAULT_H = 2048
_DEFAULT_V = 151936


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform",
        required=True,
        choices=_PLATFORMS,
        help="Platform tag for golden path",
    )
    parser.add_argument("--m", type=int, default=_DEFAULT_M, help="token rows B*S")
    parser.add_argument("--hidden", type=int, default=_DEFAULT_H, help="hidden size H")
    parser.add_argument("--vocab", type=int, default=_DEFAULT_V, help="vocab size V")
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="overwrite checked-in golden for -p instead of comparing",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="tensor device (cpu is enough for OpGraph export)",
    )
    return parser.parse_args()


def _first_diff(actual: bytes, expected: bytes) -> str:
    n = min(len(actual), len(expected))
    for i in range(n):
        if actual[i] != expected[i]:
            return f"first diff at offset {i}: actual=0x{actual[i]:02x} expected=0x{expected[i]:02x}"
    if len(actual) != len(expected):
        return f"length mismatch: actual={len(actual)} expected={len(expected)}"
    return "identical"


def build_lmhead_subgraph(device: str, m: int, hidden: int, vocab: int) -> ModelBuilder:
    dtype = torch.bfloat16
    mlp_out = torch.empty((m, hidden), dtype=dtype, device=device)
    residual = torch.empty((m, hidden), dtype=dtype, device=device)
    norm_w = torch.empty((hidden,), dtype=dtype, device=device)
    # Ascend StaticOpResult<3>: y, x, rstd (names + shapes from ops::)
    y_out = torch.empty((m, hidden), dtype=dtype, device=device)
    x_out = torch.empty((m, hidden), dtype=dtype, device=device)
    rstd = torch.empty((m, 1), dtype=dtype, device=device)
    lm_w = torch.empty((hidden, vocab), dtype=dtype, device=device)
    logits = torch.empty((m, vocab), dtype=dtype, device=device)

    builder = ModelBuilder(auto_declare=False)
    # Declaration order == wire tensor ids (must stay stable for Ascend docking).
    builder.declare_tensor(mlp_out, "mlp_output",
                           pool_hint=int(MemoryPool.Recyclable), external_addr=0)
    builder.declare_tensor(residual, "residual",
                           pool_hint=int(MemoryPool.Recyclable), external_addr=0)
    builder.declare_tensor(norm_w, "norm.weight",
                           pool_hint=int(MemoryPool.External),
                           external_addr=_NORM_WEIGHT_EXTERNAL_ADDR)
    builder.declare_tensor(y_out, "y",
                           pool_hint=int(MemoryPool.Recyclable), external_addr=0)
    builder.declare_tensor(x_out, "x",
                           pool_hint=int(MemoryPool.Recyclable), external_addr=0)
    builder.declare_tensor(rstd, "rstd",
                           pool_hint=int(MemoryPool.Recyclable), external_addr=0)
    builder.declare_tensor(lm_w, "lm_head.weight",
                           pool_hint=int(MemoryPool.External),
                           external_addr=_LM_HEAD_WEIGHT_EXTERNAL_ADDR)
    builder.declare_tensor(logits, "lm_head.out",
                           pool_hint=int(MemoryPool.Recyclable), external_addr=0)

    id0, id1 = builder.build_lm_head(
        mlp_out, residual, norm_w, lm_w, y_out, x_out, rstd, logits, ranks=1
    )
    assert (id0, id1) == (0, 1), f"op_id should be 0/1, got {id0}, {id1}"
    return builder


def main():
    args = parse_args()
    builder = build_lmhead_subgraph(args.device, args.m, args.hidden, args.vocab)
    og = build_opgraph_from_graph(builder._graph)
    assert len(og.ops) == 2
    assert og.ops[0].type == "add_rms_norm"
    assert og.ops[1].type == "matmul_lmhead"
    assert len(og.ops[0].outputs) == 3, "add_rms_norm must export StaticOpResult<3>"
    assert og.ops[0].outputs[0] == og.ops[1].inputs[0], "matmul must consume y"
    assert [t.name for t in og.tensors] == [
        "mlp_output", "residual", "norm.weight", "y", "x", "rstd",
        "lm_head.weight", "lm_head.out",
    ]
    by_name = {t.name: t for t in og.tensors}
    assert by_name["mlp_output"].desc.shape == [args.m, args.hidden]
    assert by_name["y"].desc.shape == [args.m, args.hidden]
    assert by_name["x"].desc.shape == [args.m, args.hidden]
    assert by_name["rstd"].desc.shape == [args.m, 1]
    assert by_name["norm.weight"].desc.shape == [args.hidden]
    assert by_name["lm_head.weight"].desc.shape == [args.hidden, args.vocab]
    assert by_name["lm_head.out"].desc.shape == [args.m, args.vocab]
    for op in og.ops:
        for tid in op.outputs:
            assert og.tensors[tid].desc.shape, (
                f"empty desc.shape on {op.type} output tensor "
                f"'{og.tensors[tid].name}' (tiling dimExtents need shape)"
            )
    print(f"[info] shapes M={args.m} H={args.hidden} V={args.vocab} "
          f"(Qwen3-30B-A3B defaults H=2048 V=151936)")
    print("[info] add_rms_norm outs: y[M,H], x[M,H], rstd[M,1]")

    golden_path = _EXPECTED_OPGRAPH[args.platform]
    fd, actual_path = tempfile.mkstemp(suffix="_lmhead_opgraph.bin")
    os.close(fd)
    try:
        builder.save_opgraph(actual_path)
        with open(actual_path, "rb") as f:
            actual = f.read()
    finally:
        os.remove(actual_path)

    if args.write_golden:
        with open(golden_path, "wb") as f:
            f.write(actual)
        print(f"[OK] wrote golden {golden_path} ({len(actual)} bytes)")
    else:
        if not os.path.isfile(golden_path):
            raise SystemExit(
                f"missing golden {golden_path}; re-run with --write-golden first"
            )
        with open(golden_path, "rb") as f:
            expected = f.read()
        assert actual == expected, (
            f"OpGraph must match golden byte-for-byte ({args.platform}, "
            f"{golden_path}): {_first_diff(actual, expected)}"
        )
        print(f"[OK] OpGraph byte-identical to golden for -p {args.platform} "
              f"({len(actual)} bytes)")

    src = builder.compile()
    assert "if op_id == 0:" in src and "elif op_id == 1:" in src
    assert "add_rms_norm_compute(" in src
    assert "matmul_lmhead_compute(" in src
    print("[OK] blade codegen: op_id 0 -> add_rms_norm_compute, "
          "op_id 1 -> matmul_lmhead_compute")
    builder.run()
    builder.finalize()


if __name__ == "__main__":
    main()
