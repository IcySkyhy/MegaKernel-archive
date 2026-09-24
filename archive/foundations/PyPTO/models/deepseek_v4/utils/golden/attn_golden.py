#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math

import torch
import torch_npu


def gen_block_table(actual_seq_len, block_size, block_table_shape, cmp_r=1, enable_win=False, s1=0):
    block_num_per_batch = []
    block_num = 0

    if enable_win:
        actual_seq_len_tmp = torch.clamp(actual_seq_len, max=block_size + s1 - 1).to(actual_seq_len.device)
    else:
        actual_seq_len_tmp = actual_seq_len + 0
    # 处理 torch tensor 类型的 actual_seq_len
    for actual_seq in actual_seq_len_tmp:
        block_num_per_batch.append(math.ceil(actual_seq.item() // cmp_r / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机排列

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32, device=actual_seq_len.device)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kv_cache_out, block_table, actual_seqs):
    b = actual_seqs.shape[0]
    n2 = kv_cache_out.shape[2]
    d = kv_cache_out.shape[3]
    block_size = kv_cache_out.shape[1]
    dtype = kv_cache_out.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    kv_max = (torch.max(actual_seqs).item() + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    kv_cache = torch.zeros([b, kv_max, n2, d], dtype=dtype).to(kv_cache_out.device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, d], dtype=dtype)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        kv_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor

    return kv_cache


def softmax(x, attn_sink, is_fp16=False, is_new_sink=False):
    # 使用 torch 的 softmax 实现
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    if attn_sink is not None:
        if not is_new_sink:
            x_sum += attn_sink.unsqueeze(-1)
        else:
            x_sum += torch.exp(attn_sink.unsqueeze(-1) - x_max)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum


def ifa_golden(
    q,
    kv,
    attn_sink,
    blk_cfa,
    start_pos,
    out,
    enable_flash=True,
    cmp_r=1,
    is_new_sink=False,
    kv_win=None,
    blk_win=None,
    is_prefill=False,
):
    if not enable_flash:
        fp64 = torch.float64
        q = q.to(fp64)
        kv = kv.to(fp64)
        b = start_pos.shape[0]
        blk_size = kv.shape[1]
        bs = q.shape[0]
        s1 = bs // b
        nkv = kv.shape[2]
        d = kv.shape[3]
        softmax_scale = d**-0.5
        ori_act_seqs = start_pos + s1
        compress_actual_seqs = ori_act_seqs // cmp_r
        kv_bsnd = kv_cache_concat_bsnd(kv, blk_cfa, compress_actual_seqs)
        win_seq_len = 0
        if kv_win is not None and blk_win is not None:
            k_cfa_bsnd = kv_cache_concat_bsnd(kv, blk_cfa, compress_actual_seqs)
            k_win_bsnd = kv_cache_concat_bsnd(kv_win, blk_win, ori_act_seqs * 0 + 128 + s1 - 1)
            kv_bsnd = torch.cat([k_win_bsnd, k_cfa_bsnd], dim=1)
            win_seq_len = blk_size
        for i in range(b):
            for j in range(s1):
                for n2_idx in range(nkv):
                    seq_len = min(win_seq_len, ori_act_seqs[i]) + (ori_act_seqs[i] - s1 + 1 + j) // cmp_r
                    q_bs = q[i * s1 + j]
                    kv_bs = kv_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    qk_bmm_res = torch.matmul(q_bs, kv_bs.transpose(1, 0))
                    qk_ele_res = qk_bmm_res * softmax_scale
                    softmax_res, _, _ = softmax(qk_ele_res, attn_sink, True, is_new_sink=is_new_sink)
                    bmm2_res = torch.matmul(softmax_res, kv_bs)
                    out[i * s1 + j] = bmm2_res
    else:
        ifa_flash_torch(
            q=q,
            kv=kv,
            attn_sink=attn_sink,
            block_table=blk_cfa,
            start_pos=start_pos,
            out=out,
            cmp_r=cmp_r,
            is_new_sink=is_new_sink,
            kv_win=kv_win,
            blk_win=blk_win,
            is_prefill=is_prefill,
        )


def matmul_proxy(left, right):
    fp32 = torch.float32
    return torch.matmul(left.to(fp32), right.to(fp32)).to(fp32)


def get_block_kv(kv_2d, block_table, b_idx, s2_idx, block_size, cur_seq):
    block_idx = block_table[b_idx][s2_idx]
    block_idx_valid = max(block_idx, 0)
    actual_s2_tile = min(block_size, cur_seq - s2_idx * block_size)
    kj_start = block_idx_valid * block_size
    kj_end = kj_start + actual_s2_tile
    kvj = kv_2d[kj_start:kj_end, :]
    return kvj


def flash_end(out, attn_sink, li_upd, mi_upd, oi_upd, n2g_ofs, g_tile, bs_ofs, dtype, is_new_sink=False):
    li = li_upd.unsqueeze(-1)
    if attn_sink is not None:
        if not is_new_sink:
            li += attn_sink.unsqueeze(-1)
        else:
            li += torch.exp(attn_sink - mi_upd).unsqueeze(-1)
    oi_final = oi_upd / li
    oi_upd_3d = oi_final.unsqueeze(0)
    attn_out_start = n2g_ofs
    attn_out_end = n2g_ofs + g_tile
    if attn_out_end > out.shape[1]:
        attn_out_end = out.shape[1]
        attn_out_start = attn_out_end - g_tile
    out[bs_ofs:bs_ofs + 1, attn_out_start:attn_out_end, :] = oi_upd_3d.to(dtype)


def ifa_flash_torch(
    q,
    kv,
    attn_sink,
    block_table,
    start_pos,
    out,
    cmp_r=1,
    is_new_sink=False,
    kv_win=None,
    blk_win=None,
    is_prefill=False,
):
    """
    Args:
        q: Query [batch_size * s1, num_head, head_size]
        k: Key cache [num_blocks, block_size, kv_head_num, head_size]
        v: Value cache [num_blocks, block_size, kv_head_num, head_size]
        block_table: Block mapping table for compress kv cache [batch_size, max_num_blocks_per_query]
        start_pos: Actual start position [batch_size], satisify start_pos + s1 = original actual seq
        out: Output [batch_size * s1, num_head, head_size]
    """
    fp32 = torch.float32
    q_shape = q.shape
    device = q.device
    dtype = q.dtype
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = start_pos.shape[0]
    s1 = bs1 // b
    original_actual_seqs = start_pos + s1
    k_shape = kv.shape
    _, block_size, n2, _ = k_shape
    g = n1 // n2
    g_tile = g
    kv_2d = kv.reshape(-1, d)
    q_2d = q.reshape(-1, d)
    scale = d**-0.5

    for b_idx in range(b):
        for s1_idx in range(s1):
            cur_seq = (original_actual_seqs[b_idx] - (s1 - 1 - s1_idx)) // cmp_r
            cur_seq = max(cur_seq, 0)
            s2_loop = math.ceil(cur_seq / block_size)
            for g_idx in range(g // g_tile):
                oi_upd = torch.zeros((g_tile, d), device=device, dtype=fp32)
                li_upd = torch.zeros(g_tile, device=device, dtype=fp32)
                mi_upd = torch.zeros(g_tile, device=device, dtype=fp32)
                bs_ofs = b_idx * s1 + s1_idx
                n2g_ofs = g_idx * g_tile
                qi_start = bs_ofs * n1 + n2g_ofs
                qi_end = qi_start + g_tile
                qi = q_2d[qi_start:qi_end, :]
                if kv_win is not None and blk_win is not None:
                    kv_win_2d = kv_win.reshape(-1, d)
                    cur_seq_win = min(block_size, original_actual_seqs[b_idx] - (s1 - 1 - s1_idx))
                    kv_win_tmp = get_block_kv(kv_win_2d, blk_win, b_idx, 0, block_size, cur_seq_win)
                    mm1 = matmul_proxy(qi, kv_win_tmp.t())
                    muls_res = mm1 * scale
                    tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
                    tsub = muls_res - tilda_mij
                    tilda_pij = torch.exp(tsub)
                    tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                    oi_tmp = matmul_proxy(tilda_pij.to(dtype), kv_win_tmp)
                    oi_upd = oi_tmp
                    li_upd = tilda_lij.squeeze(-1)
                    mi_upd = tilda_mij.squeeze(-1)
                    if s2_loop == 0:
                        flash_end(
                            out,
                            attn_sink,
                            li_upd,
                            mi_upd,
                            oi_upd,
                            n2g_ofs,
                            g_tile,
                            bs_ofs,
                            dtype,
                            is_new_sink=is_new_sink,
                        )
                for s2_idx in range(s2_loop):
                    kvj = get_block_kv(kv_2d, block_table, b_idx, s2_idx, block_size, cur_seq)
                    mm1 = matmul_proxy(qi, kvj.t())
                    muls_res = mm1 * scale
                    tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
                    if s2_idx == 0 and kv_win is None:
                        tsub = muls_res - tilda_mij
                        tilda_pij = torch.exp(tsub)
                        tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                        oi_tmp = matmul_proxy(tilda_pij.to(dtype), kvj)
                        oi_upd = oi_tmp
                        li_upd = tilda_lij.squeeze(-1)
                        mi_upd = tilda_mij.squeeze(-1)
                    else:
                        mi = mi_upd.unsqueeze(-1)
                        max_new, _ = torch.max(torch.cat([mi, tilda_mij], dim=-1), dim=-1, keepdim=True)
                        tsub = muls_res - max_new
                        tilda_pij = torch.exp(tsub)
                        tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                        tsub2 = torch.sub(mi, max_new)
                        mi_upd = max_new.squeeze(-1)
                        update_mul = torch.exp(tsub2)
                        li = li_upd.unsqueeze(-1)
                        sum_new = li * update_mul + tilda_lij
                        li_upd = sum_new.squeeze(-1)
                        q1 = matmul_proxy(tilda_pij.to(dtype), kvj)
                        oi_upd = oi_upd * update_mul + q1
                    if s2_idx == s2_loop - 1:
                        flash_end(
                            out,
                            attn_sink,
                            li_upd,
                            mi_upd,
                            oi_upd,
                            n2g_ofs,
                            g_tile,
                            bs_ofs,
                            dtype,
                            is_new_sink=is_new_sink,
                        )
    return out


def add_rms_norm_npu_golden(residual_input, x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    residual_input_fp32 = residual_input.to(torch.float32)
    x_fp32 = x_fp32 + residual_input_fp32
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16), x_fp32.to(torch.bfloat16)


def rms_norm_npu_golden(x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16)


def _apply_rotary_emb_neuron(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin):
    x_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    q_embed = _apply_rotary_emb_neuron(q, cos, sin)
    k_embed = _apply_rotary_emb_neuron(k, cos, sin)

    if x_dtype != torch.float32:
        q_embed = q_embed.to(x_dtype)
        k_embed = k_embed.to(x_dtype)
    return q_embed, k_embed


def attention_pre_golden(
    hidden_states,
    residual,
    input_layernorm_weight,
    input_layernorm_bias,
    qkv_proj_scale,
    qkv_proj_offset,
    qkv_proj_weight,
    qkv_proj_quant_bias,
    qkv_proj_deq_scale,
    q_norm_weight,
    q_norm_bias,
    k_norm_weight,
    k_norm_bias,
    cos,
    sin,
    eps,
):
    bs = hidden_states.shape[0]
    d = q_norm_weight.shape[0]
    rotary_dim = d // 2
    q_size = qkv_proj_weight.shape[1] - 2 * d
    x_g, residual_g = add_rms_norm_npu_golden(
        hidden_states, residual, input_layernorm_weight, input_layernorm_bias, eps
    )

    # matmul
    x_quant = torch_npu.npu_quantize(x_g, qkv_proj_scale, qkv_proj_offset, torch.qint8, -1, False)
    mm_golden = torch_npu.npu_quant_matmul(
        x_quant,
        qkv_proj_weight,
        qkv_proj_deq_scale,
        bias=qkv_proj_quant_bias,
        output_dtype=torch.bfloat16,
    )

    # split
    q_g, k_g, v_g = mm_golden.split([q_size, d, d], dim=-1)
    # nms norm
    q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // d, d)
    q_by_head = rms_norm_npu_golden(q_by_head, q_norm_weight, q_norm_bias, eps)

    k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // d, d)
    k_by_head = rms_norm_npu_golden(k_by_head, k_norm_weight, k_norm_bias, eps)

    # apply rope
    q_rot = q_by_head[..., :rotary_dim]
    q_pass = q_by_head[..., rotary_dim:]
    k_rot = k_by_head[..., :rotary_dim]
    k_pass = k_by_head[..., rotary_dim:]
    q_r, k_r = apply_rotary_pos_emb_v2(q_rot, k_rot, cos, sin)
    q_cat = torch.cat((q_r, q_pass), dim=-1)
    k_cat = torch.cat((k_r, k_pass), dim=-1)
    # post process
    q_r = q_cat.view(bs, q_size)
    k_r = k_cat.view(bs, d)
    return q_r, k_r, v_g, residual_g


def _scatter_update(value, cache, slots):
    bs = value.shape[0]
    block_number, block_size, n2, head_size = cache.shape
    for bs_idx in range(bs):
        index = slots[bs_idx]
        dim_0 = index // block_size
        dim_1 = index % block_size
        cache[dim_0, dim_1, :] = value[bs_idx][:]


def scatter_golden(k_r, v_g, key_cache, value_cache, slot_mapping):
    n2, d = key_cache.shape[-2], key_cache.shape[-1]
    b = k_r.shape[0]  # 这里严格意义上是bs，而不是b， 但当前s1为1
    key_scatter = k_r.view(b, n2, d)
    value_scatter = v_g.view(b, n2, d)
    _scatter_update(key_scatter, key_cache, slot_mapping)
    _scatter_update(value_scatter, value_cache, slot_mapping)


def attention_golden(
    hidden_states,
    residual,
    input_layernorm_weight,
    input_layernorm_bias,
    qkv_proj_scale,
    qkv_proj_offset,
    qkv_proj_weight,
    qkv_proj_quant_bias,
    qkv_proj_deq_scale,
    q_norm_weight,
    q_norm_bias,
    k_norm_weight,
    k_norm_bias,
    cos,
    sin,
    key_cache,
    value_cache,
    block_tables,
    actual_seq_lens,
    slot_mapping,
    eps,
    enable_residual,
    num_decode_tokens,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_r, k_r, v_g, residual = attention_pre_golden(
        hidden_states=hidden_states,
        residual=residual,
        input_layernorm_weight=input_layernorm_weight,
        input_layernorm_bias=input_layernorm_bias,
        qkv_proj_scale=qkv_proj_scale,
        qkv_proj_offset=qkv_proj_offset,
        qkv_proj_weight=qkv_proj_weight,
        qkv_proj_quant_bias=qkv_proj_quant_bias,
        qkv_proj_deq_scale=qkv_proj_deq_scale,
        q_norm_weight=q_norm_weight,
        q_norm_bias=q_norm_bias,
        k_norm_weight=k_norm_weight,
        k_norm_bias=k_norm_bias,
        cos=cos,
        sin=sin,
        eps=eps,
    )
    scatter_golden(k_r, v_g, key_cache, value_cache, slot_mapping)
    bs = q_r.shape[0]
    d = key_cache.shape[-1]
    q_r = q_r.view(bs, -1, d)
    attn_out = torch.zeros(q_r.shape, dtype=q_r.dtype).to(device=q_r.device)
    attn_out1 = torch.zeros(q_r.shape, dtype=q_r.dtype).to(device=q_r.device)
    ifa_golden(
        q_r,
        key_cache,
        value_cache,
        block_tables,
        actual_seq_lens,
        attn_out,
        enable_flash=False,
    )
    ifa_golden(
        q_r,
        key_cache,
        value_cache,
        block_tables,
        actual_seq_lens,
        attn_out1,
        enable_flash=True,
    )
    return attn_out, q_r, k_r, v_g, residual, key_cache, value_cache, attn_out1
