# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# End-to-end NPU test: ModelBuilder.make_add -> compile -> run, assert vs torch.
# Run: python -m triton_dist.mega_kernel_ascend.test.ops.test_add
import argparse

import torch
from triton_dist.mega_kernel_ascend import ScoreboardModelBuilder as ModelBuilder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true",
                        help="enable profiling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.npu.set_device(0)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)

    builder = ModelBuilder()
    batch = 1
    seq_len = 1
    hidden_size = 5120
    dtype = torch.bfloat16

    lhs = torch.randn((batch, seq_len, hidden_size), dtype=dtype, device="npu")
    rhs = torch.randn((batch, seq_len, hidden_size), dtype=dtype, device="npu")
    add_out = torch.empty((batch, seq_len, hidden_size), dtype=dtype, device="npu")

    builder.make_add(lhs, rhs, add_out)
    builder.compile()

    for i in range(30):
        tmp_input_0 = torch.randn(lhs.shape, dtype=dtype, device="npu")
        tmp_input_1 = torch.randn(rhs.shape, dtype=dtype, device="npu")

        lhs.copy_(tmp_input_0)
        rhs.copy_(tmp_input_1)
        builder.run()

        # torch reference
        add_out_ref = lhs + rhs
        torch.testing.assert_close(add_out_ref, add_out, atol=0, rtol=0)

    print("[OK] test_add passed: mega-kernel add matches torch on NPU "
          f"({builder.device_prop.NUM_SMS} SMs, 30 iters).")
    builder.finalize()
