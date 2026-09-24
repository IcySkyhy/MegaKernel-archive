# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Named elementwise compute kernels for Blade megakernel (no scheduling).

import triton
import triton.language as tl


@triton.jit
def add_compute(ios, args, tileid, tilecnt):
    """Clean add compute. No wait_deps, no notify, no set_cond.

    Call site is generated from this function's own signature (any subset of
    megakernel locals such as ios/args/tileid/tilecnt/op_id/...).
    ios: pointer to IO tensor pointer array (x1, x2, out)
    args: pointer to tiling data (n_elements)
    """
    BLOCK: tl.constexpr = 256
    x1_ptr = tl.load(ios.to(tl.uint64).to(tl.pointer_type(tl.int64))).to(tl.pointer_type(tl.bfloat16))
    x2_ptr = tl.load((ios.to(tl.uint64) + 8).to(tl.pointer_type(tl.int64))).to(tl.pointer_type(tl.bfloat16))
    out_ptr = tl.load((ios.to(tl.uint64) + 16).to(tl.pointer_type(tl.int64))).to(tl.pointer_type(tl.bfloat16))
    n_elements = tl.load(args.to(tl.uint64).to(tl.pointer_type(tl.int32)))
    offs = tileid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(out_ptr + offs, tl.load(x1_ptr + offs, mask=mask) + tl.load(x2_ptr + offs, mask=mask), mask=mask)


@triton.jit
def silu_mul_up_compute(ios, args, tileid, tilecnt):
    """Blade silu*up tile compute. Body placeholder until tiling is wired."""
    _ = tileid + tilecnt


@triton.jit
def add_rms_norm_compute(ios, args, tileid, tilecnt):
    """Blade fused add+rms_norm tile compute. Body placeholder until tiling is wired."""
    _ = tileid + tilecnt
