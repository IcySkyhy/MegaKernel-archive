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
"""
AllGather MatMul Module

This module implements a fused all-gather and matmul operation for distributed models.
It efficiently combines communication and computation, reducing memory overhead.

Main Functions:
    - allgather_matmul: Main function for fused all-gather and matmul computation
    - test_allgather_matmul: Functional test function
"""

import multiprocessing as mp
import os
import traceback

from distributed_config import DistributedConfig, collect_process_errors
import numpy as np
import pytest
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses import fake_tensor

import pypto


@pypto.frontend.jit(new_ir=False)
def allgather_matmul_kernel(
    in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    matmul_weight: pypto.Tensor(),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    group_name,
    world_size,
    attn_dim_per_tp_val,
):
    batch_size_per_rank = in_tensor.shape[0]

    view_row_shape = 8
    bs_loop = (batch_size_per_rank + view_row_shape - 1) // view_row_shape

    pypto.set_vec_tile_shapes(attn_dim_per_tp_val)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_AG_MM", idx_name="bs_idx"):
        shmem_shape = [view_row_shape * world_size, attn_dim_per_tp_val]
        shmem_tensor = pypto.distributed.create_shmem_tensor(group_name, world_size, pypto.DT_BF16, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)

        in_tensor_tile = pypto.view(
            in_tensor,
            (view_row_shape, in_tensor.shape[1]),
            [bs_idx * view_row_shape, 0],
            valid_shape=[(batch_size_per_rank - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]],
        )

        pypto.set_vec_tile_shapes(view_row_shape, attn_dim_per_tp_val)
        data_clear_out = pypto.distributed.shmem_clear_data(shmem_tensor, shmem_shape, [0, 0], pred=[in_tensor_tile])
        barrier_out = pypto.distributed.shmem_barrier_all(shmem_barrier_signal, [data_clear_out])

        gathered_tensor = pypto.tensor([view_row_shape * world_size, attn_dim_per_tp_val], pypto.DT_BF16, "gathered")

        pypto.set_vec_tile_shapes(view_row_shape, attn_dim_per_tp_val)
        for dyn_idx in range(world_size):
            put_out = pypto.distributed.shmem_put(
                in_tensor_tile,
                [my_pe * view_row_shape, 0],
                shmem_tensor,
                dyn_idx,
                put_op=pypto.AtomicType.SET,
                pred=[barrier_out],
            )
            pypto.distributed.shmem_signal(
                shmem_tensor,
                dyn_idx,
                1,
                [view_row_shape, attn_dim_per_tp_val],
                [my_pe * view_row_shape, 0],
                target_pe=dyn_idx,
                sig_op=pypto.AtomicType.ADD,
                pred=[put_out],
            )

            valid_row = (batch_size_per_rank - bs_idx * view_row_shape).min(view_row_shape)
            wait_until_out = pypto.distributed.shmem_wait_until(
                shmem_tensor,
                my_pe,
                1,
                [view_row_shape, attn_dim_per_tp_val],
                [dyn_idx * view_row_shape, 0],
                cmp=pypto.OpType.EQ,
                clear_signal=True,
                pred=[in_tensor_tile],
            )
            pypto.set_pass_options(sg_set_scope=(1, True, False))
            shmem_get_out = pypto.experimental.shmem_load(
                shmem_tensor,
                my_pe,
                [view_row_shape, attn_dim_per_tp_val],
                [dyn_idx * view_row_shape, 0],
                pred=[wait_until_out],
                valid_shape=[valid_row, attn_dim_per_tp_val],
            )

            gathered_tensor[dyn_idx * view_row_shape:] = shmem_get_out
            pypto.set_pass_options(sg_set_scope=-1)

        pypto.set_pass_options(sg_set_scope=(1, True, False))
        pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
        matmul_result = pypto.matmul(gathered_tensor, matmul_weight, pypto.DT_BF16, b_trans=True)
        out_tensor[bs_idx * pypto.symbolic_scalar(view_row_shape):] = matmul_result
        pypto.set_pass_options(sg_set_scope=-1)


_KERNEL_MAP = {
    'normal': allgather_matmul_kernel,
}


def generate_golden_data(config: DistributedConfig):
    batch_size_per_rank = 8
    attn_dim_per_tp = 1536
    hidden_size = 5120

    torch.manual_seed(42)

    input_datas = []
    for _ in range(config.world_size):
        in_tensor = torch.randn((batch_size_per_rank, attn_dim_per_tp), dtype=torch.bfloat16)
        matmul_weight = torch.randn((hidden_size, attn_dim_per_tp), dtype=torch.bfloat16)
        input_data = [in_tensor, matmul_weight]
        input_datas.append(input_data)

    output_datas = allgather_matmul_result_golden(batch_size_per_rank, input_datas)
    return input_datas, output_datas, attn_dim_per_tp


def allgather_matmul_result_golden(batch_size_per_rank, input_datas):
    output_datas = []
    full_batch_size = batch_size_per_rank * len(input_datas)

    for inputs in input_datas:
        allgather_result = torch.zeros((full_batch_size, input_datas[0][0].shape[1]), dtype=torch.bfloat16)
        for dyn_idx, input_data in enumerate(input_datas):
            in_tensor = input_data[0]
            allgather_result[dyn_idx * batch_size_per_rank:(dyn_idx + 1) * batch_size_per_rank, :] = in_tensor

        matmul_weight = inputs[1]
        matmul_result = torch.matmul(allgather_result, matmul_weight.T)
        output_datas.append([matmul_result])

    return output_datas


def get_check_threshold(world_size: int, golden_tensor: torch.Tensor):
    scale = np.mean(np.abs(golden_tensor.cpu().flatten().tolist())) + 1e-12
    eps = 2**-7
    tolerance_coeff = min((1 + 0.25 * np.log2(world_size)) * 1.5, 3.0)
    rtol = tolerance_coeff * eps * np.sqrt(world_size)
    atol = rtol * scale
    return rtol, atol


def allgather_matmul_worker(
    config: DistributedConfig,
    input_data: list,
    output_data: list,
    logical_rank_id: int,
    error_queue: mp.Queue = None,
    kernel_name: str = 'normal',
    attn_dim_per_tp: int = 1536,
):
    try:
        kernel = _KERNEL_MAP[kernel_name]
        groups = config.init_hccl_comm(logical_rank_id)
        device = f'npu:{config.get_physical_device_id(logical_rank_id)}'

        in_tensor, matmul_weight = input_data
        golden_out_tensor = output_data[0]

        out_tensor = torch.empty(
            (in_tensor.shape[0] * config.world_size, matmul_weight.shape[0]), dtype=torch.bfloat16, device=device
        )

        inputs = [in_tensor.to(device), matmul_weight.to(device), out_tensor]

        kernel(*inputs, groups[0], config.world_size, attn_dim_per_tp)

        out_tensor_rtol, out_tensor_atol = get_check_threshold(config.world_size, golden_out_tensor)
        np.testing.assert_allclose(
            np.array(out_tensor.cpu().flatten().tolist()),
            np.array(golden_out_tensor.cpu().flatten().tolist()),
            rtol=out_tensor_rtol,
            atol=out_tensor_atol,
        )
    except Exception as e:
        if error_queue is not None:
            error_queue.put((logical_rank_id, str(e), traceback.format_exc()))
        raise


@allow_in_graph
def allgather_matmul(
    in_tensor: torch.Tensor,
    matmul_weight: torch.Tensor,
    group_name: str,
    world_size: int,
):
    if isinstance(in_tensor, fake_tensor.FakeTensor):
        return None

    attn_dim_per_tp = in_tensor.shape[1]
    out_tensor = torch.empty(
        (in_tensor.shape[0] * world_size, matmul_weight.shape[0]), dtype=torch.bfloat16, device=in_tensor.device
    )

    allgather_matmul_kernel(in_tensor, matmul_weight, out_tensor, group_name, world_size, attn_dim_per_tp)

    return out_tensor


@pytest.mark.world_size(2)
@pypto.options(pass_options={"enable_slice": True})
def test_allgather_matmul():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=2)
    input_datas, output_datas, attn_dim_per_tp = generate_golden_data(config)

    error_queue = mp.Queue()

    processes = []
    for i in range(config.world_size):
        p = mp.Process(
            target=allgather_matmul_worker,
            args=(config, input_datas[i], output_datas[i], i, error_queue, 'normal', attn_dim_per_tp),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    collect_process_errors(processes, error_queue)


def main():
    test_allgather_matmul()


if __name__ == '__main__':
    main()
