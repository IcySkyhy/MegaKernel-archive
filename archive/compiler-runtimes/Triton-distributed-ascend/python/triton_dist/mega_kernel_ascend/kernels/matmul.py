# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Named matmul compute for blade megakernel (one Graph node -> one op_id branch).

import triton


@triton.jit
def matmul_compute(ios, args, tileid, tilecnt):
    """Blade matmul tile compute. Call site follows this signature.

    Body is a placeholder until blade tiling buffers are wired; codegen only
    needs the named entry for op_id if/elif dispatch.
    """
    # Keep the JIT happy with a trivial use of tile indices.
    _ = tileid + tilecnt


@triton.jit
def matmul_lmhead_compute(ios, args, tileid, tilecnt):
    """Blade lm_head matmul tile compute (small-M path). Body placeholder."""
    _ = tileid + tilecnt
