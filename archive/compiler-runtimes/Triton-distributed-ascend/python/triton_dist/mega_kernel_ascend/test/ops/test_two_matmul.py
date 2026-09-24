# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Blade megakernel test: two chained matmuls via make_matmul (no user dispatch).
# OpGraph must match the checked-in golden **byte-for-byte** (no canonicalize).
# op_id == Graph node index (0, 1); compile() emits if/elif -> matmul_compute.
#
# Numerical torch.matmul checks are intentionally not here yet: blade
# ``ModelBuilder.run()`` is still a launch stub (no customOp wiring). Restore
# assert_close against torch once run() executes the generated megakernel.
#
# Run (``-p`` is required):
#   python -m triton_dist.mega_kernel_ascend.test.ops.test_two_matmul -p A3
#   python -m triton_dist.mega_kernel_ascend.test.ops.test_two_matmul -p 950
import argparse
import os
import tempfile

import torch
from triton_dist.mega_kernel_ascend.models_blade.model_builder import ModelBuilder
from triton_dist.mega_triton_kernel.core.op_graph import MemoryPool

_OPS_DIR = os.path.dirname(__file__)
# Per-platform checked-in goldens; compare with ``actual == expected`` only.
_EXPECTED_OPGRAPH = {
    "A3": os.path.join(_OPS_DIR, "two_matmul_opgraph.expected.bin"),
    "950": os.path.join(_OPS_DIR, "two_matmul_opgraph.950.expected.bin"),
}
_PLATFORM_OP_NAME = {"A3": "Matmul", "950": "matmulA5"}
# Fixed External device address of `weight` in the Ascend reference OpGraph dump.
_WEIGHT_EXTERNAL_ADDR = 0x12C0C0013000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform",
        required=True,
        choices=sorted(_PLATFORM_OP_NAME),
        help="Ascend platform: A3 exports Matmul; 950 exports matmulA5",
    )
    return parser.parse_args()


def expected_opgraph_bytes(platform: str) -> bytes:
    """Load the checked-in golden for ``platform`` (no remapping)."""
    path = _EXPECTED_OPGRAPH[platform]
    with open(path, "rb") as f:
        return f.read()


def _first_diff(actual: bytes, expected: bytes) -> str:
    n = min(len(actual), len(expected))
    for i in range(n):
        if actual[i] != expected[i]:
            return f"first diff at offset {i}: actual=0x{actual[i]:02x} expected=0x{expected[i]:02x}"
    if len(actual) != len(expected):
        return f"length mismatch: actual={len(actual)} expected={len(expected)}"
    return "identical"


if __name__ == "__main__":
    args = parse_args()
    op_name = _PLATFORM_OP_NAME[args.platform]
    torch.npu.set_device(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)

    builder = ModelBuilder()
    M, K, N1, N2 = 128, 256, 128, 256
    dtype = torch.bfloat16

    # Out = (A @ B) @ weight
    a = torch.randn((M, K), dtype=dtype, device="npu") / 10
    b = torch.randn((K, N1), dtype=dtype, device="npu") / 10
    weight = torch.randn((N1, N2), dtype=dtype, device="npu") / 10
    mid = torch.zeros((M, N1), dtype=dtype, device="npu")
    out = torch.zeros((M, N2), dtype=dtype, device="npu")

    # Declaration order == Ascend wire tensor ids.
    builder.declare_tensor(a, "A", pool_hint=int(MemoryPool.Recyclable), external_addr=0)
    builder.declare_tensor(b, "B", pool_hint=int(MemoryPool.Persistent), external_addr=0)
    builder.declare_tensor(weight, "weight", pool_hint=int(MemoryPool.External),
                           external_addr=_WEIGHT_EXTERNAL_ADDR)
    builder.declare_tensor(mid, "Matmul0.out", pool_hint=int(MemoryPool.Recyclable),
                           external_addr=0)
    builder.declare_tensor(out, "Matmul1.out", pool_hint=int(MemoryPool.Recyclable),
                           external_addr=0)

    id0 = builder.make_matmul(a, b, mid, layer_id=0, op_name=op_name)
    id1 = builder.make_matmul(mid, weight, out, layer_id=1, op_name=op_name)
    assert (id0, id1) == (0, 1), f"op_id should be Graph node indices, got {id0}, {id1}"

    expected = expected_opgraph_bytes(args.platform)
    fd, actual_path = tempfile.mkstemp(suffix="_two_matmul_opgraph.bin")
    os.close(fd)
    try:
        builder.save_opgraph(actual_path)
        with open(actual_path, "rb") as f:
            actual = f.read()
    finally:
        os.remove(actual_path)
    assert actual == expected, (
        f"OpGraph must match golden byte-for-byte ({args.platform}/{op_name}, "
        f"{_EXPECTED_OPGRAPH[args.platform]}): {_first_diff(actual, expected)}")
    print(f"[OK] OpGraph byte-identical to golden for -p {args.platform} "
          f"(op={op_name}, {len(actual)} bytes)")

    src = builder.compile()
    assert "from triton_dist.mega_kernel_ascend.kernels import *" in src
    assert "if op_id == 0:" in src
    assert "elif op_id == 1:" in src
    assert src.count("matmul_compute(") == 2
    print(f"[OK] blade codegen: op_id 0/1 (Graph nodes) -> matmul_compute "
          f"(-p {args.platform})")

    builder.run()
    builder.finalize()
