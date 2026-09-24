# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Blade megakernel smoke: make_add -> codegen (op_id if/elif -> add_compute).
# Tensors are auto-declared by ModelBuilder; no declare_tensor needed.
#
# Run: python -m triton_dist.mega_kernel_ascend.test.ops.test_add_blade

import os

import torch

from triton_dist.mega_kernel_ascend.models_blade.model_builder import ModelBuilder


def main():
    os.environ["TRITON_MEGAKERNEL"] = "1"
    torch.npu.set_device(0)

    M, N = 128, 256

    lhs = torch.randn(M, N, dtype=torch.bfloat16, device="npu")
    rhs = torch.randn(M, N, dtype=torch.bfloat16, device="npu")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="npu")

    builder = ModelBuilder()
    builder.make_add(lhs, rhs, out, block=256)
    builder.compile()
    assert "from triton_dist.mega_kernel_ascend.kernels import *" in builder._dsl_src
    assert "add_compute(ios, args, tileid, tilecnt)" in builder._dsl_src
    assert "if op_id == 0:" in builder._dsl_src

    # Numerical check requires blade launch in run(); keep shape wiring smoke for now.
    print(f"[OK] megakernel add codegen registered add_compute ({M}x{N}, bf16)")
    print(builder._dsl_src)

    add_out = builder.run()

    # torch reference
    add_out_ref = lhs + rhs
    torch.testing.assert_close(add_out_ref, add_out, atol=0, rtol=0)



if __name__ == "__main__":
    main()
