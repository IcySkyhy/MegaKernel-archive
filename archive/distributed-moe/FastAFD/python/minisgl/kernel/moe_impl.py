from typing import Any, Dict

import torch


def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
) -> None:
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        **config,
    )


import functools


@functools.cache
def _get_moe_sum_cuda_module():
    from .utils import load_jit
    return load_jit(
        "moe_sum",
        cuda_files=["moe_sum.cu"],
        cuda_wrappers=[("launch", "MoeSumKernel::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def moe_sum_reduce_cuda(input: torch.Tensor, output: torch.Tensor) -> None:
    """CUDA fused split-K reduce for MoE: out[m, h] = sum_k input[m, k, h].

    Block-per-token, 256 thr/block, uint4 (8 bf16) vec loads + fp32 acc.
    Templated on TOPK ∈ {2, 4, 6, 8, 16} (compile-time unroll), generic
    fallback otherwise. Requires h % 8 == 0 and bf16/fp16 input.
    """
    assert input.is_contiguous() and output.is_contiguous()
    assert input.dim() == 3 and output.dim() == 2
    m, topk, h = input.shape
    assert output.shape == (m, h)
    assert input.dtype == output.dtype
    assert h % 8 == 0, f"hidden={h} must be divisible by 8 (uint4 vec)"
    if m == 0 or topk == 0 or h == 0:
        return
    _get_moe_sum_cuda_module().launch(output, input)
