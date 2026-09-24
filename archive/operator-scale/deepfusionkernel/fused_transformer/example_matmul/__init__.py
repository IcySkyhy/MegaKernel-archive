"""
Example matrix multiplication kernels and benchmarking scripts.
"""

import torch
from .kernels import *
from ..utils import flatten


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
NAME = {
    "matmul_silu": matmul_silu,
    "matmul_silu_vec": matmul_silu_vec,
}
KERNELS = {
    "matmul_silu": matmul_silu_kernel,
    "matmul_silu_vec": matmul_silu_vec_kernel,
}
KERNEL_LIST = flatten(list(KERNELS.values()))
