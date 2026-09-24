#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR a PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import os

import torch
import torch.nn.functional as functional

import pypto


def l2norm(query: pypto.Tensor, key: pypto.Tensor, eps: float = 1e-6) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    L2 normalization.

    Parameters
    ---------
    query: [l, d]
    key: [l, d]
    eps=1e-6

    Return
    ---------
    query_after_l2norm: [l, d]
    key_after_l2norm: [l, d]
    """

    pypto.set_vec_tile_shapes(128, 128)
    # L2
    query_after_l2norm = query / pypto.sqrt((query * query).sum(-1, keepdim=True) + eps)
    key_after_l2norm = key / pypto.sqrt((key * key).sum(-1, keepdim=True) + eps)

    return query_after_l2norm, key_after_l2norm


def pre_attn(
    gate_view: pypto.Tensor, key_view_2d: pypto.Tensor, beta_view: pypto.Tensor, tril: pypto.Tensor, mask: pypto.Tensor
) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Calculate gate_cumsum、decay_mask、beta_k、kkt

    Parameters
    ---------
    gate: [l, 1]
    key: [l, d]
    beta: [l, 1]
    tril: [l, l]
    mask: [l, l]

    Return
    ---------
    gate_cum: [l, 1]
    decay_mask: [l, l]
    a: [l, l]
    key_beta: [l, d]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # cal_cumsum
    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)  # [l,1]
    # cal_decay_mask
    decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp()  # [l,l]
    # beta_k
    key_beta = key_view_2d * beta_view  # [l,d]
    # kkt计算
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True)  # [l,l]
    a = kkt * decay_mask * mask  # [l,l]

    return gate_cum, decay_mask, a, key_beta


def inverse_pto(attn: pypto.Tensor, eye: pypto.Tensor, size: int, zeros_16, zeros_32, zeros_64) -> pypto.Tensor:
    min_length = size // 8
    pypto.set_vec_tile_shapes(128, 128)

    attn_8_8_list = []
    for i in range(8):
        attn_8_8_list.append(attn.view([min_length, min_length], [min_length * i, min_length * i]) + 0.0)
    attn_tmp_dim0 = pypto.concat(attn_8_8_list, dim=0)
    attn_tmp_dim1 = pypto.concat(attn_8_8_list, dim=1)

    attn_tmp_dim1_inv = inverse_pto_min_length(attn_tmp_dim0, attn_tmp_dim1, eye, min_length, min_length * 8)

    attn_8_8_inv_list = []
    for i in range(8):
        attn_8_8_inv_list.append(attn_tmp_dim1_inv[:, min_length * i:min_length * (i + 1)] + 0.0)

    attn_4_inv_list = []
    for i in range(4):
        attn_4_inv_list.append(
            inverse_matmul(
                attn=attn,
                attn_1_1_inv=attn_8_8_inv_list[i * 2],
                attn_2_2_inv=attn_8_8_inv_list[i * 2 + 1],
                x_ofs=min_length * i * 2,
                y_ofs=min_length * i * 2,
                m_len=min_length,
                zero_tensor=zeros_16,
            )
        )

    attn_2_inv_list = []
    for i in range(2):
        attn_2_inv_list.append(
            inverse_matmul(
                attn=attn,
                attn_1_1_inv=attn_4_inv_list[i * 2],
                attn_2_2_inv=attn_4_inv_list[i * 2 + 1],
                x_ofs=min_length * i * 4,
                y_ofs=min_length * i * 4,
                m_len=min_length * 2,
                zero_tensor=zeros_32,
            )
        )
    attn_inv = inverse_matmul(
        attn=attn,
        attn_1_1_inv=attn_2_inv_list[0],
        attn_2_2_inv=attn_2_inv_list[1],
        x_ofs=0,
        y_ofs=0,
        m_len=min_length * 4,
        zero_tensor=zeros_64,
    )
    return attn_inv


def inverse_pto_min_length(
    attn_dim0: pypto.Tensor,
    attn_dim1: pypto.Tensor,
    eye: pypto.Tensor,
    row_num: int,
    col_num: int,
) -> pypto.Tensor:
    size = col_num // row_num  # 8

    attn_inv_list = {}
    attn_inv_list[1] = attn_dim1[:2, :]
    pypto.set_vec_tile_shapes(128, 128)

    for i in range(2, row_num, 1):
        # Add 0.0 to enable attn_inv_cur to enter the UB in advance
        attn_inv_cur = attn_inv_list[i - 1] + 0.0
        row = attn_dim1.view([1, col_num], [i, 0])
        # 使能合轴时，则采用该方法
        row_expand = row.reshape([size, row_num]).view([size, i], [0, 0]).transpose(1, 0).reshape([size * i, 1])
        # row_expand = attn_dim0_trans.view([size * i, 1], [0, i])
        attn_inv_cur_reshape = attn_inv_cur.reshape([size * i, row_num])
        prod_mul = (row_expand * attn_inv_cur_reshape).reshape([i, col_num])

        prod = prod_mul.sum(0, keepdim=True)
        attn_update = row + prod

        attn_inv_list[i] = pypto.concat([attn_inv_cur, attn_update], dim=0)

    res = attn_inv_list[row_num - 1] + eye

    return res


def inverse_matmul(
    attn: pypto.Tensor,
    attn_1_1_inv: pypto.Tensor,
    attn_2_2_inv: pypto.Tensor,
    x_ofs: int,
    y_ofs: int,
    m_len: int,
    zero_tensor: pypto.Tensor,
) -> pypto.Tensor:
    """
    Calculate inverse of small matrix.

    Parameters
    ---------
    attn: [l, l]
    attn_1_1_inv: attn upper left matrix
    attn_2_2_inv: attn bottom right matrix
    x_ofs: row offset
    y_ofs: column offset
    len: matrix length

    Return
    ---------
    attn_inv: [len * 2, len * 2]
    """
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    attn_2_1 = attn.view([m_len, m_len], [x_ofs + m_len, y_ofs])

    attn_2_1_inv = (attn_2_2_inv @ attn_2_1) @ attn_1_1_inv

    attn_inv = pypto.tensor([m_len * 2, m_len * 2], dtype=attn_1_1_inv.dtype)
    attn_inv[0:m_len, 0:m_len] = attn_1_1_inv
    attn_inv[0:m_len, m_len:m_len * 2] = zero_tensor
    attn_inv[m_len:m_len * 2, 0:m_len] = attn_2_1_inv
    attn_inv[m_len:m_len * 2, m_len:m_len * 2] = attn_2_2_inv

    return attn_inv


def cal_value_and_key_cumdecay(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate value and k cumdecay

    Parameters:
    -------------
    attn: [l, l]
    value_view: [l, d]
    beta_view: [l, d]
    key_beta: [l, d]
    gate_cum: [l, 1]

    Return:
    -------------
    value_out: [l, d]
    key_cum_out: [l, d]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # value_out
    value_beta_view = value_view * beta_view  # [l, d]
    value_out = pypto.matmul(attn, value_beta_view, pypto.DT_FP32)  # [l, d]
    # k_cumdecay_out
    g_exp = pypto.exp(gate_cum)  # [l, 1]
    weighted_k_beta_view = key_beta * g_exp  # [l, d]
    key_cum_out = pypto.matmul(attn, weighted_k_beta_view, pypto.DT_FP32)  # [l, d]

    return value_out, key_cum_out


def recurrent_state_attn_all(
    query: pypto.Tensor,
    key: pypto.Tensor,
    value: pypto.Tensor,
    k_cumdecay: pypto.Tensor,
    gate: pypto.Tensor,
    state: pypto.Tensor,
    decay_mask: pypto.Tensor,
    tril: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
    dv = value.shape[-1]
    l = gate.valid_shape[0]  # noqa: E741
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(64, 128)
    _last_gate_1 = gate[l - 1:l, :]
    kgexp = key * (_last_gate_1 - gate).exp()  # [l, Dk]
    qgexp = query * gate_exp
    pypto.set_vec_tile_shapes(64, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True)  # [l, Dk] @ [Dk, Dv] = [l, Dv]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    attn_inter = pypto.matmul(qgexp, state, pypto.DT_FP32, b_trans=True)  # [l, Dk] @ [Dk, Dv] = [l, Dv]
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    temp_matmul_vprime = pypto.matmul(v_prime, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, l] @ [l, Dk] = [Dv, Dk]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    temp_matmul_value = pypto.matmul(value, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, l] @ [l, Dk] = [l, Dk]
    attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True)  # [l, Dk] @ [Dk, l] = [l, l]
    _last_gate_2 = pypto.expand_clone(gate_exp[l - 1:l, :], (dv, 1))  # [Dv, 1]
    final_state_1 = state * _last_gate_2
    state_new = final_state_1 + temp_matmul_value - temp_matmul_vprime
    pypto.set_vec_tile_shapes(128, 128)
    attn_tmp = attn * decay_mask * tril  # [l, l]
    pypto.set_vec_tile_shapes(64, 128)
    chunk_attn_value = pypto.matmul(attn_tmp, value, pypto.DT_FP32)  # [l, l] @ [l, Dv] = [l, Dv]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    chunk_attn_vprime = pypto.matmul(attn_tmp, v_prime, pypto.DT_FP32)  # [l, l] @ [l, Dv] = [l, Dv]
    chunk_attn_out = attn_inter + chunk_attn_value - chunk_attn_vprime
    return chunk_attn_out, state_new


def chunk_gated_delta_rule(b, nqk, nv, d, l):  # noqa: E741
    t = pypto.DYNAMIC
    b1 = b + 1
    b1 = pypto.DYNAMIC
    b = pypto.DYNAMIC
    query_shape = [t, nqk, d]
    key_shape = [t, nqk, d]
    value_shape = [t, nv, d]
    beta_shape = [t, nv]
    gate_shape = [t, nv]
    states_shape = [b, nv, d, d]
    mask_shape = [l, l]
    tril_mask_shape = [l, l]
    eye_shape = [16, l]
    act_seq_len_shape = [b1]
    core_attn_out_shape = [t, nv, d]
    last_state_data_shape = [b, nv, d, d]

    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 1, "device_sched_parallelism": 8})
    def kernel(
        query: pypto.Tensor(query_shape, pypto.DT_FP32),
        key: pypto.Tensor(key_shape, pypto.DT_FP32),
        value: pypto.Tensor(value_shape, pypto.DT_FP32),
        beta: pypto.Tensor(beta_shape, pypto.DT_FP32),
        gate: pypto.Tensor(gate_shape, pypto.DT_FP32),
        states: pypto.Tensor(states_shape, pypto.DT_FP32),
        mask: pypto.Tensor(mask_shape, pypto.DT_FP32),
        tril_mask: pypto.Tensor(tril_mask_shape, pypto.DT_FP32),
        eye: pypto.Tensor(eye_shape, pypto.DT_FP32),
        act_seq_len: pypto.Tensor(act_seq_len_shape, pypto.DT_INT32),
        core_attn_out: pypto.Tensor(core_attn_out_shape, pypto.DT_FP32),
        last_state_data: pypto.Tensor(last_state_data_shape, pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)

        _, nqk, d = query.shape
        _, nv, d = value.shape
        b = states.shape[0]
        l, l = mask.shape  # noqa: E741
        group = nv // nqk
        for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]
            for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx", parallel=True):  #
                nqk_idx = nv_idx // group
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                last_state = states[b_idx, nv_idx]
                for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16, 1]):  #
                    bs_ofs = b_ofs + s_idx
                    actual_l = (s - s_idx).min(l)
                    ## view
                    query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                    key_view = pypto.view(key, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                    value_view = pypto.view(value, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                    beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])
                    gate_view = pypto.view(gate, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])

                    pypto.set_vec_tile_shapes(128, 128, 128)
                    query_view_2d = pypto.reshape(query_view, [l, d], valid_shape=[actual_l, d])
                    key_view_2d = pypto.reshape(key_view, [l, d], valid_shape=[actual_l, d])
                    value_view_2d = pypto.reshape(value_view, [l, d], valid_shape=[actual_l, d])

                    zeros_16 = pypto.full(size=[16, 16], fill_value=0.0, dtype=pypto.DT_FP32)
                    zeros_32 = pypto.full(size=[32, 32], fill_value=0.0, dtype=pypto.DT_FP32)
                    zeros_64 = pypto.full(size=[64, 64], fill_value=0.0, dtype=pypto.DT_FP32)
                    # compute
                    # qk_l2norm
                    query_norm, key_norm = l2norm(query_view_2d, key_view_2d)
                    scale = 1 / d**0.5
                    query_scale = query_norm * scale

                    gate_cum, decay_mask, a_block, key_beta = pre_attn(gate_view, key_norm, beta_view, tril_mask, mask)
                    # inverse
                    a_block_inverse = inverse_pto(a_block, eye, 128, zeros_16, zeros_32, zeros_64)

                    # cal_value_and_keycumdecay
                    value_out, key_cum_out = cal_value_and_key_cumdecay(
                        a_block_inverse, value_view_2d, beta_view, key_beta, gate_cum
                    )
                    chunk_attn_out, cur_state = recurrent_state_attn_all(
                        query_scale, key_norm, value_out, key_cum_out, gate_cum, last_state, decay_mask, tril_mask
                    )
                    # assemble
                    # pypto.set_vec_tile_shapes(16, 16, 128, 128)
                    last_state[:] = cur_state
                    core_attn_out[bs_ofs:bs_ofs + l, nv_idx] = chunk_attn_out
                    last_state_data[b_idx, nv_idx] = last_state

    return kernel


def pypto_chunk_gated_delta_rule(query_data, key_data, value_data, beta_data, gate_data, state_data, act_seq_len):
    """
    PyPTO calculate chunk Gated Delta Rule.

    Parameters
    ---------
    query_data: [t, nqk, d]
    key_data: [t, nqk, d]
    value_data: [t, nv, d]
    beta_data: [t, nv]
    gate_data: [t, nv]
    state_data: [b, nv, d, d]
    act_seq_len: [b,]

    Return
     ---------
    core_attn_out: [t, nv, d]
    state_data: [b, nv, d, d]
    """
    t, nv, d = value_data.shape
    nqk = query_data.shape[1]
    l = 128  # noqa: E741
    b = state_data.shape[0]

    if not query_data.is_contiguous():
        query_data = query_data.contiguous()
    if not key_data.is_contiguous():
        key_data = key_data.contiguous()
    if not value_data.is_contiguous():
        value_data = value_data.contiguous()
    if not beta_data.is_contiguous():
        beta_data = beta_data.contiguous()
    if not gate_data.is_contiguous():
        gate_data = gate_data.contiguous()
    if not state_data.is_contiguous():
        state_data = state_data.contiguous()

    # output
    core_attn_out = torch.ones([t, nv, d], dtype=torch.float32, device=query_data.device)
    last_state_data = torch.zeros([b, nv, d, d], dtype=torch.float32, device=query_data.device)
    # helper data
    mask_data = torch.tril(-torch.ones([l, l], dtype=torch.float32, device=query_data.device), diagonal=-1)
    tril_mask_data = torch.ones([l, l], device=query_data.device).float().tril()  # lower triangular
    eye_data = torch.eye(16, device=query_data.device).repeat(1, 8).float()

    inputs = [
        query_data,
        key_data,
        value_data,
        beta_data,
        gate_data,
        state_data,
        mask_data,
        tril_mask_data,
        eye_data,
        act_seq_len,
    ]

    outputs = [core_attn_out, last_state_data]
    chunk_gated_delta_rule(b, nqk, nv, d, l)(*inputs, *outputs)
    # torch.npu.synchronize()
    return core_attn_out, last_state_data


def segs_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    act_seq_len,
    chunk_size=128,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    t, n1, d = query.shape
    t, n, d = value.shape
    batch = act_seq_len.shape[0] - 1

    query = query.repeat_interleave(n // n1, dim=1)
    key = key.repeat_interleave(n // n1, dim=1)

    final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query.device)

    query, key, value, beta, g = [
        x.transpose(0, 1).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query.device)

    for b_idx in range(batch):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        _l = 64
        result_list = []
        recurrent_state = initial_state[b_idx:b_idx + 1, ...]
        # for s_idx in range(0, pad_seq_length, seg_s):
        chunk_query = query[:, b_ofs:b_ofs + s, :].reshape(1, n, s, d)
        chunk_key = key[:, b_ofs:b_ofs + s, :].reshape(1, n, s, d)
        chunk_value = value[:, b_ofs:b_ofs + s, :].reshape(1, n, s, d)
        chunk_gate = g[:, b_ofs:b_ofs + s].reshape(1, n, s)
        chunk_beta = beta[:, b_ofs:b_ofs + s].reshape(1, n, s)
        cur_attn, cur_state = torch_chunk_gated_delta_rule(
            chunk_query,
            chunk_key,
            chunk_value,
            chunk_gate,
            chunk_beta,
            chunk_size,
            recurrent_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
        )
        result_list.append(cur_attn.squeeze(0))
        recurrent_state = cur_state

        batch_attn = torch.cat(result_list, dim=0)[:s]
        final_attn[b_ofs:b_ofs + s] = batch_attn
        final_state[b_idx:b_idx + 1, ...] = recurrent_state
    return final_attn, final_state


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=128,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    b, n, s, d = value.shape
    l = chunk_size  # noqa: E741
    _c = max(1, s // l)

    initial_state = initial_state.transpose(3, 2)
    _initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = functional.pad(query, (0, 0, 0, pad_size))
    key = functional.pad(key, (0, 0, 0, pad_size))
    value = functional.pad(value, (0, 0, 0, pad_size))
    beta = functional.pad(beta, (0, pad_size))
    g = functional.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    b, h, n, d = query.shape
    query, key, value, k_beta, v_beta = [
        x.reshape(b, h, -1, chunk_size, d) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(b, h, -1, chunk_size)
    causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)  # cal_cumsum
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()  # cal_decay_mask
    attn = -(k_beta @ key.transpose(-1, -2) * decay_mask).masked_fill(causal, 0)  # cal_pre_attn

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)  # cal_inverse

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))  # cal_value_and_kcumdecay

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query.device).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value).to(query.device)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(causal, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    bs, hd = core_attn_out.shape[:2]
    dim = core_attn_out.shape[-1]
    core_attn_out = core_attn_out.reshape(bs, hd, -1, dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    last_recurrent_state = last_recurrent_state.transpose(-1, -2)
    return core_attn_out, last_recurrent_state


def detailed_tensor_compare(tensor1, tensor2, rtol=1e-3, atol=1e-3, verbose=True, max_outliers_display=20):
    """
    详细的张量比较，分析不在容差范围内的元素比例，并显示超出容差的具体信息

    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        rtol: 相对容差
        atol: 绝对容差
        verbose: 是否打印详细信息
        max_outliers_display: 最大显示的超出容差的元素数量

    Returns:
        dict: 包含比较结果的字典
    """
    # 确保张量可以比较
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()

    # 计算差异
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)  # 避免除零

    # 容差检查
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    # 统计信息
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements

    # 差异统计
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    # 超出容差的差异统计
    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()

        # 获取超出容差的索引和值
        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]

        # 按差异大小排序（从大到小）
        sorted_indices = torch.argsort(outlier_diffs, descending=True)
        sorted_outlier_indices = tuple(ind[sorted_indices] for ind in outlier_indices)
        sorted_outlier_values1 = outlier_values1[sorted_indices]
        sorted_outlier_values2 = outlier_values2[sorted_indices]
        sorted_outlier_diffs = outlier_diffs[sorted_indices]
        sorted_outlier_relative_diffs = outlier_relative_diffs[sorted_indices]

    else:
        max_out_diff = 0.0
        mean_out_diff = 0.0
        sorted_outlier_indices = None
        sorted_outlier_values1 = None
        sorted_outlier_values2 = None
        sorted_outlier_diffs = None
        sorted_outlier_relative_diffs = None

    result = {
        'total_elements': total_elements,
        'out_of_tolerance_count': out_of_tolerance_count,
        'out_of_tolerance_ratio': out_of_tolerance_ratio,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_out_of_tolerance_diff': max_out_diff,
        'mean_out_of_tolerance_diff': mean_out_diff,
        'all_close': out_of_tolerance_count == 0,
        'tolerance_mask': tolerance_mask,
        'diff_tensor': diff,
        'outlier_indices': sorted_outlier_indices,
        'outlier_values1': sorted_outlier_values1,
        'outlier_values2': sorted_outlier_values2,
        'outlier_diffs': sorted_outlier_diffs,
        'outlier_relative_diffs': sorted_outlier_relative_diffs,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("📊 张量详细比较报告")
        print("=" * 60)
        print(f"总元素数量: {total_elements:,}")
        print(f"超出容差元素数量: {out_of_tolerance_count:,}")
        print(f"超出容差比例: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio * 100:.4f}%)")
        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")
        print(f"差异标准差: {std_diff:.6f}")
        print(f"容差设置: rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            print(f"超出容差的最大差异: {max_out_diff:.6f}")
            print(f"超出容差的平均差异: {mean_out_diff:.6f}")

            # 显示超出容差的详细信息
            print(f"\n🔍 超出容差的元素详情 (显示前{min(max_outliers_display, out_of_tolerance_count)}个):")
            print("-" * 80)
            print(f"{'索引':<20} {'Tensor1值':<15} {'Tensor2值':<15} {'绝对差异':<12} {'相对差异':<12}")
            print("-" * 80)

            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                idx_str = str(tuple(sorted_outlier_indices[j][i].item() for j in range(len(sorted_outlier_indices))))
                print(
                    f"{idx_str:<20} {sorted_outlier_values1[i].item():<15.6f} "
                    f"{sorted_outlier_values2[i].item():<15.6f} "
                    f"{sorted_outlier_diffs[i].item():<12.6f} {sorted_outlier_relative_diffs[i].item():<12.6f}"
                )

            if out_of_tolerance_count > max_outliers_display:
                print(f"... 还有 {out_of_tolerance_count - max_outliers_display} 个超出容差的元素未显示")

        print(f"\n✅ 张量匹配: {result['all_close']}")
        print("=" * 60)

    return result


@pypto.options(pass_options={"enable_slice": True})
def test_chunk_gated_delta_rule():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()
    s = 1024 * 32
    t = s * 2
    nqk = 16
    nv = 16
    d = 128
    act_seq_len = [0, s, t]
    b = len(act_seq_len) - 1

    # # prepare inputs data
    torch.manual_seed(12)
    query_data = torch.rand([t, nqk, d], dtype=torch.float32, device=f'npu:{device_id}') * (1.3655 + 0.2785) - (
        1.3655 + 0.2785
    )
    key_data = torch.rand([t, nqk, d], dtype=torch.float32, device=f'npu:{device_id}') * (1.4664 + 0.2785) - (
        1.4664 + 0.2785
    )
    value_data = torch.rand([t, nv, d], dtype=torch.float32, device=f'npu:{device_id}') * (1.6488 + 0.2785) - (
        1.6488 + 0.2785
    )
    beta_data = torch.rand([t, nv], dtype=torch.float32, device=f'npu:{device_id}') * (0.8927 - 0.0889) - (
        0.8927 - 0.0889
    )
    gate_data = torch.rand([t, nv], dtype=torch.float32, device=f'npu:{device_id}') * (-0.1343 + 37.5452) - (
        -0.1343 + 37.5452
    )
    states_data = torch.zeros([b, nv, d, d], dtype=torch.float32, device=f'npu:{device_id}')
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32, device=f'npu:{device_id}')

    # calculate torch result
    core_attn_out_torch, final_state_torch = segs_chunk_gated_delta_rule(
        query_data.clone(),
        key_data.clone(),
        value_data.clone(),
        gate_data.clone(),
        beta_data.clone(),
        initial_state=states_data.clone(),
        act_seq_len=act_seq_len.clone(),
    )
    print("finish torch")
    # calculate pypto result
    inputs = [query_data, key_data, value_data, beta_data, gate_data, states_data, act_seq_len]
    core_attn_out_pypto, final_state_pypto = pypto_chunk_gated_delta_rule(*inputs)
    print("================pto vs torch==================")
    detailed_tensor_compare(core_attn_out_pypto, core_attn_out_torch)
    detailed_tensor_compare(final_state_pypto, final_state_torch)
    pypto.runtime._device_fini()


if __name__ == "__main__":
    test_chunk_gated_delta_rule()
