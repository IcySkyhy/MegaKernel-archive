# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Blade compute kernels re-exported for megakernel template
# ``from triton_dist.mega_kernel_ascend.kernels import *``.

from .elementwise import add_compute, add_rms_norm_compute, silu_mul_up_compute
from .matmul import matmul_compute, matmul_lmhead_compute

__all__ = [
    "add_compute",
    "add_rms_norm_compute",
    "matmul_compute",
    "matmul_lmhead_compute",
    "silu_mul_up_compute",
]
