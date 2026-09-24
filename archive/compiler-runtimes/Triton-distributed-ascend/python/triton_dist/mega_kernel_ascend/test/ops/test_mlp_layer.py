# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# NPU test: 3-op chain fc1 -> silu_mul_up -> fc2, assert vs torch. Exercises the
# scoreboard cross-op dependency path (silu waits on fc1, fc2 waits on silu).
# Single-process, no allreduce.
# Run: python -m triton_dist.mega_kernel_ascend.test.ops.test_mlp_layer
import argparse
import sys
import types

# torch_impl_utils.py does `import flashinfer` at top-level; absent on NPU.
class _Mock:
    def __getattr__(self, _): return _Mock()
    def __call__(self, *a, **k): return _Mock()
sys.modules.setdefault("flashinfer", _Mock())

import torch
from triton_dist.mega_kernel_ascend import ScoreboardModelBuilder as ModelBuilder
from triton_dist.mega_triton_kernel.test.torch_impl_utils import torch_gate_silu_mul_up


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--intra_kernel_profile", default=False, action="store_true",
                        help="enable intra kernel profiling")
    parser.add_argument("--enable_runtime_scheduler", default=False, action="store_true",
                        help="enable runtime scheduler")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.npu.set_device(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)

    builder = ModelBuilder(enable_profiling=args.intra_kernel_profile,
                           enable_runtime_scheduler=args.enable_runtime_scheduler)
    batch = 1
    seq_len = 1
    hidden_size = 5120
    tp_size = 1
    intermediate_size = 25600 // tp_size
    dtype = torch.bfloat16

    fc1_weight = torch.randn((intermediate_size * 2, hidden_size), dtype=dtype, device="npu") / 10
    fc2_weight = torch.randn((hidden_size, intermediate_size), dtype=dtype, device="npu") / 10
    mlp_layer_input = torch.randn((batch * seq_len, hidden_size), dtype=dtype, device="npu")
    fc1_output = torch.zeros((batch * seq_len, intermediate_size * 2), dtype=dtype, device="npu")
    act_out = torch.zeros((batch * seq_len, intermediate_size), dtype=dtype, device="npu")
    fc2_out = torch.zeros((batch * seq_len, hidden_size), dtype=dtype, device="npu")

    builder.make_fc1(mlp_layer_input, fc1_weight, fc1_output)
    builder.make_silu_mul_up(fc1_output, act_out)
    builder.make_fc2(act_out, fc2_weight, fc2_out)
    builder.compile()

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="npu", dtype=torch.int8)
    import triton
    triton.set_allocator(alloc_fn)

    for i in range(30):
        tmp_input = torch.randn(mlp_layer_input.shape, dtype=dtype, device="npu")
        mlp_layer_input.copy_(tmp_input)
        builder.run()

        # torch reference
        fc1_output_ref = torch.nn.functional.linear(mlp_layer_input, fc1_weight)
        act_out_ref = torch_gate_silu_mul_up(fc1_output_ref)
        fc2_output_ref = torch.nn.functional.linear(act_out_ref, fc2_weight)
        torch.testing.assert_close(fc1_output_ref, fc1_output, atol=0, rtol=0)
        torch.testing.assert_close(act_out_ref, act_out, atol=0, rtol=0)
        torch.testing.assert_close(fc2_output_ref, fc2_out, atol=0, rtol=0)

    print("[OK] test_mlp_layer passed: mega-kernel fc1->silu_mul_up->fc2 matches "
          f"torch on NPU ({builder.device_prop.NUM_SMS} SMs, 30 iters).")
    builder.finalize()
