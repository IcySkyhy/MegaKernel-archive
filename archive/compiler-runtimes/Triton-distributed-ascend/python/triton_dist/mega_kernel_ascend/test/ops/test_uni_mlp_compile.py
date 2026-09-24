# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Unified blade API smoke: same make_fc1 -> silu_mul_up -> fc2 sequence as
# mega_triton_kernel test_mlp_layer, no declare_tensor, compile-only.
#
# Run: python -m triton_dist.mega_kernel_ascend.test.ops.test_uni_mlp_compile
from __future__ import annotations

import os

import torch

from triton_dist.mega_kernel_ascend import ModelBuilder


def main():
    os.environ["TRITON_MEGAKERNEL"] = "1"
    # Device placement only; builder API matches TD make_* usage.
    if hasattr(torch, "npu"):
        torch.npu.set_device(0)
        device = "npu"
    else:
        device = "cpu"

    batch, seq_len = 1, 1
    hidden_size = 512
    intermediate_size = 1024
    dtype = torch.bfloat16

    fc1_weight = torch.randn(
        (intermediate_size * 2, hidden_size), dtype=dtype, device=device
    ) / 10
    fc2_weight = torch.randn(
        (hidden_size, intermediate_size), dtype=dtype, device=device
    ) / 10
    mlp_in = torch.randn(
        (batch * seq_len, hidden_size), dtype=dtype, device=device
    )
    fc1_out = torch.zeros(
        (batch * seq_len, intermediate_size * 2), dtype=dtype, device=device
    )
    act_out = torch.zeros(
        (batch * seq_len, intermediate_size), dtype=dtype, device=device
    )
    fc2_out = torch.zeros(
        (batch * seq_len, hidden_size), dtype=dtype, device=device
    )

    builder = ModelBuilder()  # default backend=blade
    id0 = builder.make_fc1(mlp_in, fc1_weight, fc1_out)
    id1 = builder.make_silu_mul_up(fc1_out, act_out)
    id2 = builder.make_fc2(act_out, fc2_weight, fc2_out)
    assert (id0, id1, id2) == (0, 1, 2)

    src = builder.compile()
    assert "from triton_dist.mega_kernel_ascend.kernels import *" in src
    assert "if op_id == 0:" in src
    assert "elif op_id == 1:" in src
    assert "elif op_id == 2:" in src
    assert "matmul_compute(" in src
    assert "silu_mul_up_compute(" in src
    builder.run()
    builder.finalize()
    print("[OK] uni API mlp compile: fc1 -> silu_mul_up -> fc2 (op_id 0/1/2)")


if __name__ == "__main__":
    main()
