# LlamaGen GPT decode megakernel (single token, batch=1)
# Architecture: RMSNorm, GQA, 2D interleaved RoPE, SwiGLU MLP, no QK-norm
# Synchronisation: 5 flags per layer [qkv_done, attn_done, out_done, w13_done, w2_done]
# This implementation is heavily inspired from @gau-nerst's learn-cuda megakernel - https://github.com/gau-nernst/learn-cuda/blob/main/12_megakernel/model_triton.py
# You might find most of the ideas either directly adapted or re implemented in a similar way as in his codebase
# The major changes being the interleaved ROPE pairs, instead of fused mlp w13 we load seperate, Non-power-of-2 dims

import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional
from torch import Tensor


@triton.jit
def _rms_norm(x_ptr, norm_ptr, dim: tl.constexpr, DIM_BLOCK: tl.constexpr):
    # DIM_BLOCK is next power-of-2 >= dim; padded slots load as 0 and are harmless
    offs = tl.arange(0, DIM_BLOCK)
    mask = offs < dim
    x    = tl.load(x_ptr    + offs, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) * (1.0 / dim) + 1e-5
    return x * norm * tl.rsqrt(mean_sq)


@triton.jit
def _grid_sync(ptr, num_pids):
    tl.atomic_add(ptr, 1, sem="release", scope="gpu")
    while tl.atomic_add(ptr, 0, sem="acquire", scope="gpu") != num_pids:
        pass


@triton.jit
def llamagen_megakernel(
    input_ids_ptr,       # ()               – current token id
    x_ptr,               # (dim,)           – hidden state scratch
    tok_embeddings_ptr,  # (vocab, dim)
    num_layers,
    attn_norm_ptr,       # (layers, dim)
    kv_cache_ptr,        # (layers, 2, max_ctx, n_kv_head, head_dim)
    wqkv_ptr,            # (layers, total_kv_dim, dim)
    wo_ptr,              # (layers, dim, n_head*head_dim)
    attn_tmp_ptr,        # (total_kv_dim,)  – QKV results; Q overwritten with attn out
    rope_ptr,            # (head_dim,)      – flattened (head_dim//2, 2) for current pos
    position,
    max_context,
    ffn_norm_ptr,        # (layers, dim)
    w1_ptr,              # (layers, hidden_dim, dim)
    w3_ptr,              # (layers, hidden_dim, dim)
    w2_ptr,              # (layers, dim, hidden_dim)
    mlp_tmp_ptr,         # (hidden_dim,)    – SwiGLU intermediate
    norm_ptr,            # (dim,)
    flag_ptr,
    dim: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    hidden_dim: tl.constexpr,
    ATTN_BLOCK_N: tl.constexpr,
    ATTN_BLOCK_K: tl.constexpr,
    MLP_BLOCK_N: tl.constexpr,
    MLP_BLOCK_K: tl.constexpr,
    head_dim: tl.constexpr,
    DIM_BLOCK: tl.constexpr,  # triton.next_power_of_2(dim)
):
    raw_pid  = tl.program_id(0)
    num_pids = tl.num_programs(0)

    offs_dim = tl.arange(0, DIM_BLOCK)
    dim_mask = offs_dim < dim
    input_id = tl.load(input_ids_ptr)
    tl.store(x_ptr + offs_dim,
             tl.load(tok_embeddings_ptr + input_id * dim + offs_dim, mask=dim_mask, other=0.0),
             mask=dim_mask)

    for layer_id in range(num_layers):

        # QKV Projection
        x_normed = _rms_norm(x_ptr, attn_norm_ptr + layer_id * dim, dim, DIM_BLOCK)

        total_kv_dim: tl.constexpr = (num_heads + num_kv_heads * 2) * head_dim
        offs_db = tl.arange(0, DIM_BLOCK)
        db_mask = offs_db < dim
        for pid_n in range(raw_pid, total_kv_dim // ATTN_BLOCK_N, num_pids):
            offs_n    = pid_n * ATTN_BLOCK_N + tl.arange(0, ATTN_BLOCK_N)
            wqkv_ptrs = (wqkv_ptr
                         + layer_id * total_kv_dim * dim
                         + offs_n[:, None] * dim
                         + offs_db[None, :])
            wqkv = tl.load(wqkv_ptrs, mask=db_mask[None, :], other=0.0)
            tl.store(attn_tmp_ptr + offs_n, tl.sum(x_normed * wqkv, axis=1))

        _grid_sync(flag_ptr + layer_id * 5 + 0, num_pids)

        q_ptr = attn_tmp_ptr
        k_ptr = q_ptr + num_heads    * head_dim
        v_ptr = k_ptr + num_kv_heads * head_dim

        # Attention (one threadblock per Q head
        if raw_pid < num_heads:
            head_id    = raw_pid
            kv_head_id = head_id // (num_heads // num_kv_heads)

            offs_hdim = tl.arange(0, head_dim)
            offs_half = tl.arange(0, head_dim // 2)

            v = tl.load(v_ptr + kv_head_id * head_dim + offs_hdim)

            # 2D RoPE: interleaved [cos0,sin0,cos1,sin1,...] at current position
            cos = tl.load(rope_ptr + offs_half * 2    ).to(tl.float32)
            sin = tl.load(rope_ptr + offs_half * 2 + 1).to(tl.float32)

            # Rotate Q, pre-bake exp2 softmax scale
            scale   = 1.4426950408889634 * (head_dim ** -0.5)
            q_h_ptr = q_ptr + head_id * head_dim
            q_re = tl.load(q_h_ptr + offs_half * 2    ).to(tl.float32)
            q_im = tl.load(q_h_ptr + offs_half * 2 + 1).to(tl.float32)
            tl.store(q_h_ptr + offs_half * 2,     (q_re * cos - q_im * sin) * scale)
            tl.store(q_h_ptr + offs_half * 2 + 1, (q_im * cos + q_re * sin) * scale)

            # Rotate K, write K/V to cache
            k_h_ptr = k_ptr + kv_head_id * head_dim
            k_re = tl.load(k_h_ptr + offs_half * 2    ).to(tl.float32)
            k_im = tl.load(k_h_ptr + offs_half * 2 + 1).to(tl.float32)

            k_cache_ptr = kv_cache_ptr + layer_id * 2 * max_context * num_kv_heads * head_dim
            v_cache_ptr = k_cache_ptr  + max_context * num_kv_heads * head_dim
            kv_base     = (position * num_kv_heads + kv_head_id) * head_dim
            tl.store(k_cache_ptr + kv_base + offs_half * 2,     k_re * cos - k_im * sin)
            tl.store(k_cache_ptr + kv_base + offs_half * 2 + 1, k_im * cos + k_re * sin)
            tl.store(v_cache_ptr + kv_base + offs_hdim, v)

            q_new = tl.load(q_h_ptr + offs_hdim)  # [head_dim], scaled+rotated

            k_cache_ptrs = k_cache_ptr + (tl.arange(0, ATTN_BLOCK_K)[:, None] * num_kv_heads * head_dim
                                          + kv_head_id * head_dim + offs_hdim[None, :])
            v_cache_ptrs = v_cache_ptr + (tl.arange(0, ATTN_BLOCK_K)[None, :] * num_kv_heads * head_dim
                                          + kv_head_id * head_dim + offs_hdim[:, None])

            max_s   = tl.full((1,), float("-inf"), dtype=tl.float32)
            sum_exp = tl.zeros((ATTN_BLOCK_K,), dtype=tl.float32)
            o       = tl.zeros((head_dim, ATTN_BLOCK_K), dtype=tl.float32)

            num_iters = tl.cdiv(position + 1, ATTN_BLOCK_K) - 1
            for _ in range(num_iters):
                k_block = tl.load(k_cache_ptrs)
                v_block = tl.load(v_cache_ptrs)
                s = tl.sum(q_new * k_block, axis=1)
                new_max_s = tl.maximum(max_s, tl.max(s, axis=0))
                rescale   = tl.exp2(max_s - new_max_s)
                p         = tl.exp2(s - new_max_s)
                sum_exp   = sum_exp * rescale + p
                o         = o * rescale + p * v_block
                max_s     = new_max_s
                k_cache_ptrs += ATTN_BLOCK_K * num_kv_heads * head_dim
                v_cache_ptrs += ATTN_BLOCK_K * num_kv_heads * head_dim

            mask    = tl.arange(0, ATTN_BLOCK_K) < position + 1 - num_iters * ATTN_BLOCK_K
            k_block = tl.load(k_cache_ptrs)
            v_block = tl.load(v_cache_ptrs, mask=mask[None, :], other=0.0)
            s       = tl.where(mask, tl.sum(q_new * k_block, axis=1), float("-inf"))
            new_max_s = tl.maximum(max_s, tl.max(s, axis=0))
            rescale   = tl.exp2(max_s - new_max_s)
            p         = tl.exp2(s - new_max_s)
            sum_exp   = sum_exp * rescale + p
            o         = o * rescale + p * v_block

            tl.store(q_h_ptr + offs_hdim, tl.sum(o, axis=1) / tl.sum(sum_exp))
            tl.atomic_add(flag_ptr + layer_id * 5 + 1, 1, sem="release", scope="gpu")

        while tl.atomic_add(flag_ptr + layer_id * 5 + 1, 0, sem="acquire", scope="gpu") != num_heads:
            pass

        # Output projection + residual 
        q_dim: tl.constexpr = num_heads * head_dim
        for pid_n in range(raw_pid, dim // ATTN_BLOCK_N, num_pids):
            offs_n  = pid_n * ATTN_BLOCK_N + tl.arange(0, ATTN_BLOCK_N)
            offs_k  = tl.arange(0, ATTN_BLOCK_K)
            o_ptrs  = q_ptr + offs_k
            wo_ptrs = wo_ptr + layer_id * dim * q_dim + offs_n[:, None] * q_dim + offs_k[None, :]
            acc     = tl.zeros((ATTN_BLOCK_N, ATTN_BLOCK_K), dtype=tl.float32)
            for _ in range(q_dim // ATTN_BLOCK_K):
                acc    += tl.load(o_ptrs) * tl.load(wo_ptrs)
                o_ptrs  += ATTN_BLOCK_K
                wo_ptrs += ATTN_BLOCK_K
            offs_n = pid_n * ATTN_BLOCK_N + tl.arange(0, ATTN_BLOCK_N)
            acc    = tl.sum(acc, axis=1) + tl.load(x_ptr + offs_n)
            tl.store(x_ptr + offs_n, acc)

        _grid_sync(flag_ptr + layer_id * 5 + 2, num_pids)

        # ── SwiGLU W1/W3 projection ─────────────────────────────────────────
        x_normed = _rms_norm(x_ptr, ffn_norm_ptr + layer_id * dim, dim, DIM_BLOCK)
        offs_db  = tl.arange(0, DIM_BLOCK)
        db_mask  = offs_db < dim
        for pid_n in range(raw_pid, hidden_dim // MLP_BLOCK_N, num_pids):
            offs_n  = pid_n * MLP_BLOCK_N + tl.arange(0, MLP_BLOCK_N)
            w1_ptrs = w1_ptr + layer_id * hidden_dim * dim + offs_n[:, None] * dim + offs_db[None, :]
            w3_ptrs = w3_ptr + layer_id * hidden_dim * dim + offs_n[:, None] * dim + offs_db[None, :]
            a1 = tl.sum(x_normed * tl.load(w1_ptrs, mask=db_mask[None, :], other=0.0), axis=1)
            a3 = tl.sum(x_normed * tl.load(w3_ptrs, mask=db_mask[None, :], other=0.0), axis=1)
            tl.store(mlp_tmp_ptr + offs_n, a1 * tl.sigmoid(a1) * a3)  # silu(a1) * a3

        _grid_sync(flag_ptr + layer_id * 5 + 3, num_pids)

        # W2 projection + residual
        for pid_n in range(raw_pid, dim // MLP_BLOCK_N, num_pids):
            offs_n   = pid_n * MLP_BLOCK_N + tl.arange(0, MLP_BLOCK_N)
            offs_k   = tl.arange(0, MLP_BLOCK_K)
            tmp_ptrs = mlp_tmp_ptr + offs_k
            w2_ptrs  = w2_ptr + layer_id * dim * hidden_dim + offs_n[:, None] * hidden_dim + offs_k[None, :]
            acc      = tl.zeros((MLP_BLOCK_N, MLP_BLOCK_K), dtype=tl.float32)
            for _ in range(hidden_dim // MLP_BLOCK_K):
                acc     += tl.load(tmp_ptrs) * tl.load(w2_ptrs)
                tmp_ptrs += MLP_BLOCK_K
                w2_ptrs  += MLP_BLOCK_K
            offs_n = pid_n * MLP_BLOCK_N + tl.arange(0, MLP_BLOCK_N)
            acc    = tl.sum(acc, axis=1) + tl.load(x_ptr + offs_n)
            tl.store(x_ptr + offs_n, acc)

        _grid_sync(flag_ptr + layer_id * 5 + 4, num_pids)

    # Final norm; store only valid dim elements
    x_normed = _rms_norm(x_ptr, norm_ptr, dim, DIM_BLOCK)
    offs_dim  = tl.arange(0, DIM_BLOCK)
    tl.store(x_ptr + offs_dim, x_normed, mask=offs_dim < dim)

    if raw_pid == 0:  # reset flags for next call
        for i in range(num_layers * 5):
            tl.store(flag_ptr + i, 0)


@dataclass
class LlamaGenParams:
    tok_embeddings: Tensor   # (vocab_size, dim)
    l_attn_norm: Tensor      # (num_layers, dim)
    l_wqkv: Tensor           # (num_layers, total_kv_dim, dim)
    l_wo: Tensor             # (num_layers, dim, num_heads * head_dim)
    l_ffn_norm: Tensor       # (num_layers, dim)
    l_w1: Tensor             # (num_layers, hidden_dim, dim)
    l_w3: Tensor             # (num_layers, hidden_dim, dim)
    l_w2: Tensor             # (num_layers, dim, hidden_dim)
    norm: Tensor             # (dim,)
    output: Tensor           # (vocab_size, dim)
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass
class LlamaGenBuffers:
    kv_cache: Tensor   # (num_layers, 2, max_context, num_kv_heads, head_dim)
    rope: Tensor       # (max_seq, head_dim) = flattened freqs_cis (max_seq, head_dim//2, 2)
    position: int


def setup_from_transformer(model, max_context: int, dtype=None) -> tuple["LlamaGenParams", "LlamaGenBuffers"]:
    """Stack per-layer weights and precompute rope from a LlamaGen Transformer."""
    if dtype is None:
        dtype = model.tok_embeddings.weight.dtype
    device = model.tok_embeddings.weight.device

    def w(t):
        return t.to(dtype=dtype, device=device).contiguous()

    config     = model.config
    num_heads  = config.n_head
    num_kv_heads = config.n_kv_head if config.n_kv_head is not None else config.n_head
    head_dim   = config.dim // num_heads

    params = LlamaGenParams(
        tok_embeddings=w(model.tok_embeddings.weight),
        l_attn_norm=torch.stack([w(l.attention_norm.weight)    for l in model.layers]),
        l_wqkv     =torch.stack([w(l.attention.wqkv.weight)    for l in model.layers]),
        l_wo       =torch.stack([w(l.attention.wo.weight)      for l in model.layers]),
        l_ffn_norm =torch.stack([w(l.ffn_norm.weight)          for l in model.layers]),
        l_w1       =torch.stack([w(l.feed_forward.w1.weight)   for l in model.layers]),
        l_w3       =torch.stack([w(l.feed_forward.w3.weight)   for l in model.layers]),
        l_w2       =torch.stack([w(l.feed_forward.w2.weight)   for l in model.layers]),
        norm  =w(model.norm.weight),
        output=w(model.output.weight),
        num_layers  =model.n_layer,
        num_heads   =num_heads,
        num_kv_heads=num_kv_heads,
        head_dim    =head_dim,
    )

    # freqs_cis: (max_seq, head_dim//2, 2) → flatten → (max_seq, head_dim)
    rope = model.freqs_cis.to(device=device)[:max_context].reshape(max_context, -1).to(torch.float32)

    kv_cache = torch.zeros(model.n_layer, 2, max_context, num_kv_heads, head_dim,
                           dtype=dtype, device=device)
    return params, LlamaGenBuffers(kv_cache=kv_cache, rope=rope, position=0)


_FLAG: Optional[Tensor] = None


def llamagen_decode(input_ids: Tensor, params: LlamaGenParams, buffers: LlamaGenBuffers) -> Tensor:
    """Single decode step. Returns logits (1, vocab_size)."""
    global _FLAG
    device = input_ids.device
    if _FLAG is None:
        _FLAG = torch.zeros(1000, dtype=torch.int32, device=device)

    assert input_ids.shape[0] == 1

    _, total_kv_dim, dim = params.l_wqkv.shape
    _, hidden_dim, _     = params.l_w1.shape
    max_context          = buffers.kv_cache.shape[2]

    x       = params.tok_embeddings.new_empty(dim)
    attn_tmp = params.tok_embeddings.new_empty(total_kv_dim)
    mlp_tmp  = params.tok_embeddings.new_empty(hidden_dim)

    num_sms   = torch.cuda.get_device_properties(device).multi_processor_count
    DIM_BLOCK = triton.next_power_of_2(dim)

    llamagen_megakernel[(num_sms,)](
        input_ids, x, params.tok_embeddings, params.num_layers,
        params.l_attn_norm, buffers.kv_cache, params.l_wqkv, params.l_wo,
        attn_tmp, buffers.rope[buffers.position], buffers.position, max_context,
        params.l_ffn_norm, params.l_w1, params.l_w3, params.l_w2, mlp_tmp,
        params.norm, _FLAG,
        dim=dim, num_heads=params.num_heads, num_kv_heads=params.num_kv_heads,
        hidden_dim=hidden_dim,
        ATTN_BLOCK_N=8, ATTN_BLOCK_K=128, MLP_BLOCK_N=4, MLP_BLOCK_K=512,
        head_dim=params.head_dim, DIM_BLOCK=DIM_BLOCK,
    )

    buffers.position += 1
    return x.unsqueeze(0) @ params.output.T  # LM head kept outside kernel
