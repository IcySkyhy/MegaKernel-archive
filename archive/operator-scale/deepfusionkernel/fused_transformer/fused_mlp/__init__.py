"""
Fused MLP implementations and benchmarking scripts for transformer models.
This module includes various fused MLP kernels, autotuning scripts to find the best configurations,
and utilities to load and apply autotuning results.
This module invokes triton kernels directly through PyTorch bindings.
"""

from ..utils import flatten
from .fused_gmlp import *


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
NAME = {
    "m_tkn": gatedmlp_m_tkn,
    "m_tkn_vec": gatedmlp_m_tkn_vec,
    "mt_kn": gatedmlp_mt_kn,
    "mt_kn_vec": gatedmlp_mt_kn_vec,
    "mt_kn_keepA2": gatedmlp_mt_kn_keepA2,
    "mt_kn_keepA2_vec": gatedmlp_mt_kn_keepA2_vec,
    "t_mkn": gatedmlp_t_mkn,
    "t_mkn_vec": gatedmlp_t_mkn_vec,
    "sep_knls": gatedmlp_separated,
    "sep_knls_vec": gatedmlp_separated_vec,
}
KERNELS = {
    "m_tkn": gatedmlp_m_tkn_kernel,
    "m_tkn_vec": gatedmlp_m_tkn_vec_kernel,
    "mt_kn": gatedmlp_mt_kn_kernel,
    "mt_kn_vec": gatedmlp_mt_kn_vec_kernel,
    "mt_kn_keepA2": gatedmlp_mt_kn_keepA2_kernel,
    "mt_kn_keepA2_vec": gatedmlp_mt_kn_keepA2_vec_kernel,
    "t_mkn": gatedmlp_t_mkn_kernel,
    "t_mkn_vec": gatedmlp_t_mkn_vec_kernel,
    "sep_knls": (gatedmlp_separated_a2_kernel, gatedmlp_separated_y_kernel),
    "sep_knls_vec": (gatedmlp_separated_a2_vec_kernel, gatedmlp_separated_y_vec_kernel),
}
KERNEL_LIST = flatten(list(KERNELS.values()))
