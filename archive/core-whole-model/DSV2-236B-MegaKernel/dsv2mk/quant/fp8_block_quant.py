"""Block-scaled FP8 E4M3 quantisation with UE8M0 scale factors.

Matches DeepSeek-V2/V3 per-128 quantisation granularity. CuTe-DSL block-scaled
MMA supports sf_vec_size in {16, 32}, so per-128 scales are replicated 4x along
K at sf_vec_size=32 and reordered into the hardware atom-major SFA/SFB layout.
"""
from __future__ import annotations

from typing import Tuple

import torch

FP8_E4M3_MAX: float = 448.0
UE8M0_BIAS: int = 127


def _compute_block_amax(x: torch.Tensor, block_k: int) -> torch.Tensor:
    assert x.shape[-1] % block_k == 0, (
        f"K-dim {x.shape[-1]} not divisible by block_k {block_k}"
    )
    n_blocks_k = x.shape[-1] // block_k
    x_f32 = x.detach().to(torch.float32)
    reshaped = x_f32.reshape(*x_f32.shape[:-1], n_blocks_k, block_k)
    amax = reshaped.abs().amax(dim=-1)
    return amax


def _amax_to_ue8m0_byte(amax: torch.Tensor) -> torch.Tensor:
    assert amax.dtype == torch.float32
    safe = amax.clone()
    safe[safe == 0] = 1.0
    scale_pow2 = safe / FP8_E4M3_MAX
    log2_scale = torch.ceil(torch.log2(scale_pow2))
    byte = (log2_scale + UE8M0_BIAS).clamp(0, 255).to(torch.uint8)
    byte[amax == 0] = 0
    return byte


def _ue8m0_byte_to_scale_fp32(byte: torch.Tensor) -> torch.Tensor:
    assert byte.dtype == torch.uint8
    exp = byte.to(torch.int32) - UE8M0_BIAS
    return torch.pow(torch.tensor(2.0, dtype=torch.float32, device=byte.device), exp.to(torch.float32))


def quantize_fp8_e4m3_block(
    W: torch.Tensor,
    block_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert W.shape[-1] % block_k == 0, (
        f"K-dim {W.shape[-1]} not divisible by block_k {block_k}"
    )
    device = W.device
    n_blocks_k = W.shape[-1] // block_k

    amax = _compute_block_amax(W, block_k).to(device)
    scale_byte = _amax_to_ue8m0_byte(amax)
    scale_fp32 = _ue8m0_byte_to_scale_fp32(scale_byte)

    W_f32 = W.detach().to(torch.float32)
    scale_per_elem = scale_fp32.unsqueeze(-1).expand(*amax.shape, block_k).reshape(*W.shape)
    safe_scale = torch.where(scale_per_elem == 0, torch.ones_like(scale_per_elem), scale_per_elem)
    W_scaled = W_f32 / safe_scale
    W_clamped = W_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    W_fp8 = W_clamped.to(torch.float8_e4m3fn)
    return W_fp8, scale_byte


def dequantize_fp8_e4m3_block(
    W_fp8: torch.Tensor,
    scales_ue8m0: torch.Tensor,
    block_k: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert W_fp8.dtype == torch.float8_e4m3fn
    assert scales_ue8m0.dtype == torch.uint8
    assert W_fp8.shape[-1] // block_k == scales_ue8m0.shape[-1], (
        f"W K {W_fp8.shape[-1]} // {block_k} != scales K {scales_ue8m0.shape[-1]}"
    )
    scale_fp32 = _ue8m0_byte_to_scale_fp32(scales_ue8m0)
    n_blocks_k = scales_ue8m0.shape[-1]
    scale_per_elem = scale_fp32.unsqueeze(-1).expand(*scales_ue8m0.shape, block_k).reshape(*W_fp8.shape)
    W_f32 = W_fp8.to(torch.float32) * scale_per_elem
    return W_f32.to(out_dtype)


def pack_ue8m0_to_int32(scales_ue8m0: torch.Tensor) -> torch.Tensor:
    assert scales_ue8m0.dtype == torch.uint8
    assert scales_ue8m0.shape[-1] % 4 == 0, (
        f"Last dim {scales_ue8m0.shape[-1]} not divisible by 4 for UE8M0 packing"
    )
    leading = scales_ue8m0.shape[:-1]
    last = scales_ue8m0.shape[-1]
    grouped = scales_ue8m0.reshape(*leading, last // 4, 4).to(torch.int32)
    packed = (
        grouped[..., 0]
        | (grouped[..., 1] << 8)
        | (grouped[..., 2] << 16)
        | (grouped[..., 3] << 24)
    )
    return packed.contiguous()


def unpack_int32_to_ue8m0(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.int32
    leading = packed.shape[:-1]
    last = packed.shape[-1]
    b0 = (packed & 0xFF).to(torch.uint8)
    b1 = ((packed >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed >> 16) & 0xFF).to(torch.uint8)
    b3 = ((packed >> 24) & 0xFF).to(torch.uint8)
    out = torch.stack([b0, b1, b2, b3], dim=-1).reshape(*leading, last * 4)
    return out.contiguous()


def replicate_ue8m0_to_sf_vec_size(
    scales_ue8m0: torch.Tensor,
    logical_block_k: int = 128,
    sf_vec_size: int = 32,
) -> torch.Tensor:
    assert scales_ue8m0.dtype == torch.uint8
    assert logical_block_k % sf_vec_size == 0, (
        f"logical_block_k {logical_block_k} not divisible by sf_vec_size {sf_vec_size}"
    )
    rep = logical_block_k // sf_vec_size
    leading = scales_ue8m0.shape[:-1]
    kb = scales_ue8m0.shape[-1]
    out = scales_ue8m0.unsqueeze(-1).expand(*leading, kb, rep).reshape(*leading, kb * rep)
    return out.contiguous()


def reorder_sf_to_hw_layout(
    sf_per_vec: torch.Tensor,
    sf_vec_size: int = 32,
) -> torch.Tensor:
    assert sf_per_vec.dtype == torch.uint8
    assert sf_vec_size == 32, (
        f"Only sf_vec_size=32 is supported (got {sf_vec_size})"
    )
    mn, k_sf = sf_per_vec.shape
    assert mn % 128 == 0, f"MN {mn} not divisible by 128 (HW SF atom MN)"
    assert k_sf % 4 == 0, f"K_sf {k_sf} not divisible by 4 (HW SF atom K)"
    rest_m = mn // 128
    rest_k = k_sf // 4
    sf5 = sf_per_vec.reshape(rest_m, 4, 32, rest_k, 4)
    out = sf5.permute(0, 3, 2, 1, 4).contiguous()
    return out.reshape(-1)


def quantize_to_fp8_with_hw_sf(
    W: torch.Tensor,
    logical_block_k: int = 128,
    sf_vec_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert sf_vec_size == 32
    W_fp8, sf_logical = quantize_fp8_e4m3_block(W, block_k=logical_block_k)
    sf_per_vec = replicate_ue8m0_to_sf_vec_size(
        sf_logical, logical_block_k=logical_block_k, sf_vec_size=sf_vec_size
    )
    if sf_per_vec.dim() == 2:
        sf_hw = reorder_sf_to_hw_layout(sf_per_vec, sf_vec_size=sf_vec_size)
    elif sf_per_vec.dim() == 3:
        L = sf_per_vec.shape[0]
        per_layer = [
            reorder_sf_to_hw_layout(sf_per_vec[l].contiguous(), sf_vec_size=sf_vec_size)
            for l in range(L)
        ]
        sf_hw = torch.stack(per_layer, dim=0).contiguous()
    else:
        raise AssertionError(f"Expected 2-D or 3-D W, got {W.dim()}-D")
    return W_fp8, sf_hw


def fp8_gemm_block_scaled_ref(
    x_fp8: torch.Tensor,
    W_fp8: torch.Tensor,
    x_scales_ue8m0: torch.Tensor,
    W_scales_ue8m0: torch.Tensor,
    block_k: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert W_fp8.dtype == torch.float8_e4m3fn
    assert x_fp8.shape[-1] == W_fp8.shape[-1]
    M, K = x_fp8.shape
    N, _ = W_fp8.shape
    assert K % block_k == 0
    n_blocks_k = K // block_k
    assert x_scales_ue8m0.shape == (M, n_blocks_k), (
        f"bad x_scales {x_scales_ue8m0.shape}, expected {(M, n_blocks_k)}"
    )
    assert W_scales_ue8m0.shape == (N, n_blocks_k), (
        f"bad W_scales {W_scales_ue8m0.shape}, expected {(N, n_blocks_k)}"
    )

    x_sf = _ue8m0_byte_to_scale_fp32(x_scales_ue8m0)
    W_sf = _ue8m0_byte_to_scale_fp32(W_scales_ue8m0)

    x_f32 = x_fp8.to(torch.float32).reshape(M, n_blocks_k, block_k)
    W_f32 = W_fp8.to(torch.float32).reshape(N, n_blocks_k, block_k)
    inner = torch.einsum("mkb,nkb->mnk", x_f32, W_f32)
    sf_outer = x_sf.unsqueeze(1) * W_sf.unsqueeze(0)
    acc = (inner * sf_outer).sum(dim=-1)
    return acc.to(out_dtype)

