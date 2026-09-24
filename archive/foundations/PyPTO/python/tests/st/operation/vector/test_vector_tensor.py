#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import copy
import math
import os

import numpy as np
from numpy.testing import assert_allclose
import torch

import pypto


@pypto.options(pass_options={"enable_slice": True})
def test_exp_tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    a = pypto.tensor(shape, dtype, "SQRT_TENSOR_a")
    b = pypto.tensor(shape, dtype, "SQRT_TENSOR_b")

    with pypto.function("EXP", a, b):
        for b_idx in pypto.loop(int(math.ceil(n / view_shape[0])), name="LOOP_SQRT_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(math.ceil(m / view_shape[1])), name="LOOP_SQRT_L1", idx_name="s_idx"):
                tile_a = pypto.view(
                    a,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(tile_a.exp())
                pypto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)

    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor)

    expected = torch.exp(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_scatterupdate_tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 1
    s = 1
    n = 1
    d = 8
    block_num = 1
    block_size = 1
    src_shape = (b, s, n, d)
    index_shape = (b, s)
    dst_shape = (block_num, block_size, n, d)

    view_shape = (b, s, n, d)
    tile_shape = (b, s, n, d)
    pypto.runtime._device_init()

    src_tensor = pypto.tensor(src_shape, pypto.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pypto.tensor(index_shape, pypto.DT_INT32, "PTO_TENSOR_INDEX")
    update_tensor = pypto.tensor(dst_shape, pypto.DT_INT32, "PTO_TENSOR_DST")
    dst_tensor = pypto.tensor(dst_shape, pypto.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(src_shape[0] / view_shape[0])
    s_loop_num = math.ceil(src_shape[1] / view_shape[1])
    with pypto.function("MAIN", src_tensor, index_tensor, update_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pypto.tensor(dst_shape, pypto.DT_INT32, "PTO_TENSOR_TMP")
                view_tensor_src = pypto.view(
                    src_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1], 0, 0],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                        n,
                        d,
                    ],
                )
                view_tensor_index = pypto.view(
                    index_tensor,
                    [view_shape[0], view_shape[1]],
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(index_shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(index_shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                view_tensor_dst = pypto.view(update_tensor, dst_shape, [0, 0, 0, 0])
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1], tile_shape[2], tile_shape[3])
                tmp_dst_tensor.move(view_tensor_dst.scatter_update(-2, view_tensor_index, view_tensor_src))
                pypto.set_vec_tile_shapes(1, 64, n, d)
                pypto.assemble(tmp_dst_tensor, [0, 0, 0, 0], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = np.random.uniform(2, 3, src_shape).astype(np.int32)
    input1_tensor = np.random.choice(range(0, dst_shape[0] * dst_shape[1]), index_shape, replace=False).astype(np.int32)
    input2_tensor = np.random.uniform(1, 2, dst_shape).astype(np.int32)
    result = copy.copy(input2_tensor)
    d_data = np.zeros(dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[3]).astype(np.int32)

    a_tensor = torch.from_numpy(input0_tensor)
    b_tensor = torch.from_numpy(input1_tensor)
    c_tensor = torch.from_numpy(input2_tensor)
    d_tensor = torch.from_numpy(d_data)

    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")
    pto_d_tensor = pypto.from_torch(d_tensor, "d_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor, pto_d_tensor)

    for _b in range(b):
        for _s in range(s):
            result[input1_tensor[_b][_s] // block_size][input1_tensor[_b][_s] % block_size][:] = input0_tensor[_b][_s][
                :
            ]
    result_t = torch.from_numpy(result).to(d_tensor.device).to(d_tensor.dtype)
    result_t = result_t.reshape_as(d_tensor)
    assert (d_tensor == result_t).all().item()
    pypto.runtime._device_fini()


class ScatterParamInfo:
    def __init__(self, sdata: float, axis: int, b, s, idx0, idx1):
        self.src_shape = (b, s)
        self.indices_shape = (idx0, idx1)
        self.view_shape = (b, s)
        self.tile_shape = (b, 16)
        self.sdata = sdata
        self.axis = axis


def scatter_2dim_proc(scatter_para, is_inplace):
    pypto.runtime._device_init()
    src_shape = scatter_para.src_shape
    indices_shape = scatter_para.indices_shape
    view_shape = scatter_para.view_shape
    tile_shape = scatter_para.tile_shape

    self_tensor = pypto.tensor(src_shape, pypto.DT_FP32, "PTO_TENSOR_SRC")
    indices_tensor = pypto.tensor(indices_shape, pypto.DT_INT64, "PTO_TENSOR_INDEX")
    dst_tensor = pypto.tensor(src_shape, pypto.DT_FP32, "PTO_TENSOR_DST")
    src = scatter_para.sdata

    b_loop_num = math.ceil(indices_shape[0] / view_shape[0])
    s_loop_num = math.ceil(indices_shape[1] / view_shape[1])
    with pypto.function("MAIN", self_tensor, indices_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pypto.tensor(view_shape, pypto.DT_FP32, "PTO_TENSOR_TMP")
                view_tensor_src = pypto.view(
                    self_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                view_tensor_index = pypto.view(
                    indices_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(indices_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(indices_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                if is_inplace:
                    view_tensor_src.scatter_(scatter_para.axis, view_tensor_index, src)
                    tmp_dst_tensor.move(view_tensor_src)
                else:
                    tmp_dst_tensor.move(view_tensor_src.scatter(scatter_para.axis, view_tensor_index, src))
                pypto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.rand(*src_shape, dtype=torch.float32) * 2 - 1
    input1_tensor = torch.randint(0, src_shape[scatter_para.axis], indices_shape, dtype=torch.int64)
    c_tensor = torch.zeros_like(input0_tensor)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_input1_tensor = pypto.from_torch(input1_tensor, "input1_tensor")
    pto_result_tensor = pypto.from_torch(c_tensor, "c_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_input1_tensor, pto_result_tensor)

    result = input0_tensor.clone()
    for i in range(indices_shape[0]):
        for j in range(indices_shape[1]):
            if scatter_para.axis == 0:
                result[input1_tensor[i, j], j] = scatter_para.sdata
            else:
                result[i, input1_tensor[i, j]] = scatter_para.sdata

    assert torch.equal(c_tensor, result)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_scatter__onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 5
    idx0 = 2
    idx1 = 5
    scatter_para = ScatterParamInfo(2.0, 0, b, s, idx0, idx1)

    scatter_2dim_proc(scatter_para, True)


@pypto.options(pass_options={"enable_slice": True})
def test_scatter_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 4
    idx0 = 3
    idx1 = 4
    scatter_para = ScatterParamInfo(2.0, 1, b, s, idx0, idx1)

    scatter_2dim_proc(scatter_para, False)


@pypto.options(pass_options={"enable_slice": True})
def test_scatter_add_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 5
    idx0 = 2
    idx1 = 5
    pypto.runtime._device_init()
    src_shape = (b, s)
    indices_shape = (idx0, idx1)
    view_shape = (b, s)
    tile_shape = (b, 16)

    self_tensor = pypto.tensor(src_shape, pypto.DT_FP32, "PTO_TENSOR_SRC")
    indices_tensor = pypto.tensor(indices_shape, pypto.DT_INT64, "PTO_TENSOR_INDEX")
    dst_tensor = pypto.tensor(src_shape, pypto.DT_FP32, "PTO_TENSOR_DST")
    axis = 0
    reduce = 'add'
    src = 2.0

    b_loop_num = math.ceil(indices_shape[0] / view_shape[0])
    s_loop_num = math.ceil(indices_shape[1] / view_shape[1])
    with pypto.function("MAIN", self_tensor, indices_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pypto.tensor(view_shape, pypto.DT_FP32, "PTO_TENSOR_TMP")
                view_tensor_self = pypto.view(
                    self_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                view_tensor_index = pypto.view(
                    indices_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(indices_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(indices_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_dst_tensor.move(view_tensor_self.scatter(axis, view_tensor_index, src, reduce=reduce))
                pypto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.rand(*src_shape, dtype=torch.float32) * 2 - 1
    input1_tensor = torch.randint(0, src_shape[axis], indices_shape, dtype=torch.int64)
    c_tensor = torch.zeros_like(input0_tensor)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_input1_tensor = pypto.from_torch(input1_tensor, "input1_tensor")
    pto_result_tensor = pypto.from_torch(c_tensor, "c_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_input1_tensor, pto_result_tensor)

    result = input0_tensor.clone()
    for i in range(indices_shape[0]):
        for j in range(indices_shape[1]):
            if axis == 0:
                result[input1_tensor[i, j], j] += src
            else:
                result[i, input1_tensor[i, j]] += src

    assert torch.equal(c_tensor, result)
    pypto.runtime._device_fini()


def scatter_2dim_tensor_proc(scatter_para, is_inplace):
    pypto.runtime._device_init()
    self_shape = scatter_para.src_shape
    indices_shape = scatter_para.indices_shape
    view_shape = scatter_para.view_shape
    tile_shape = scatter_para.tile_shape

    self_tensor = pypto.tensor(self_shape, pypto.DT_FP32, "PTO_TENSOR_SELF")
    indices_tensor = pypto.tensor(indices_shape, pypto.DT_INT64, "PTO_TENSOR_INDEX")
    src_tensor = pypto.tensor(self_shape, pypto.DT_FP32, "PTO_TENSOR_SRC")
    dst_tensor = pypto.tensor(self_shape, pypto.DT_FP32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(indices_shape[0] / view_shape[0])
    s_loop_num = math.ceil(indices_shape[1] / view_shape[1])
    with pypto.function("MAIN", self_tensor, indices_tensor, src_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pypto.tensor(view_shape, pypto.DT_FP32, "PTO_TENSOR_TMP")
                view_tensor_self = pypto.view(
                    self_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(self_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(self_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                view_tensor_index = pypto.view(
                    indices_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(indices_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(indices_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                view_tensor_src = pypto.view(
                    src_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(self_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(self_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                if is_inplace:
                    view_tensor_self.scatter_(scatter_para.axis, view_tensor_index, view_tensor_src)
                    tmp_dst_tensor.move(view_tensor_self)
                else:
                    tmp_dst_tensor.move(view_tensor_self.scatter(scatter_para.axis, view_tensor_index, view_tensor_src))
                pypto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.rand(*self_shape, dtype=torch.float32) * 2 - 1
    input1_tensor = torch.randint(0, self_shape[scatter_para.axis], indices_shape, dtype=torch.int64)
    input2_tensor = torch.rand(*self_shape, dtype=torch.float32) * 10 - 1
    c_tensor = torch.zeros_like(input0_tensor)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_input1_tensor = pypto.from_torch(input1_tensor, "input1_tensor")
    pto_input2_tensor = pypto.from_torch(input2_tensor, "input2_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_input1_tensor, pto_input2_tensor, pto_c_tensor)

    result = input0_tensor.clone()
    for i in range(indices_shape[0]):
        for j in range(indices_shape[1]):
            if scatter_para.axis == 0:
                result[input1_tensor[i, j], j] = input2_tensor[i, j]
            else:
                result[i, input1_tensor[i, j]] = input2_tensor[i, j]

    assert torch.equal(c_tensor, result)
    pypto.runtime._device_fini()


@pypto.options(pass_options={"enable_slice": True})
def test_scatter__tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 5
    idx0 = 2
    idx1 = 5
    scatter_para = ScatterParamInfo(0, 0, b, s, idx0, idx1)

    scatter_2dim_tensor_proc(scatter_para, True)


@pypto.options(pass_options={"enable_slice": True})
def test_scatter_tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 4
    idx0 = 3
    idx1 = 4
    scatter_para = ScatterParamInfo(0, 1, b, s, idx0, idx1)

    scatter_2dim_tensor_proc(scatter_para, False)


@pypto.options(pass_options={"enable_slice": True})
def test_scatter_tensor_add_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 5
    idx0 = 2
    idx1 = 5
    pypto.runtime._device_init()
    self_shape = (b, s)
    indices_shape = (idx0, idx1)
    view_shape = (b, s)
    tile_shape = (b, 16)

    self_tensor = pypto.tensor(self_shape, pypto.DT_FP32, "PTO_TENSOR_SELF")
    indices_tensor = pypto.tensor(indices_shape, pypto.DT_INT32, "PTO_TENSOR_INDEX")
    src_tensor = pypto.tensor(self_shape, pypto.DT_FP32, "PTO_TENSOR_SRC")
    dst_tensor = pypto.tensor(self_shape, pypto.DT_FP32, "PTO_TENSOR_DST")
    reduce = 'add'
    axis = 0

    b_loop_num = math.ceil(indices_shape[0] / view_shape[0])
    s_loop_num = math.ceil(indices_shape[1] / view_shape[1])
    with pypto.function("MAIN", self_tensor, indices_tensor, src_tensor, dst_tensor):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_tensor = pypto.tensor(view_shape, pypto.DT_FP32, "PTO_TENSOR_TMP")
                view_tensor_self = pypto.view(
                    self_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(self_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(self_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                view_tensor_index = pypto.view(
                    indices_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(indices_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(indices_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                view_tensor_src = pypto.view(
                    src_tensor,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(self_shape[0]) - b_idx * view_shape[0]).min(
                            pypto.symbolic_scalar(view_shape[0])
                        ),
                        (pypto.symbolic_scalar(self_shape[1]) - s_idx * view_shape[1]).min(
                            pypto.symbolic_scalar(view_shape[1])
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_tensor.move(view_tensor_self.scatter(axis, view_tensor_index, view_tensor_src, reduce=reduce))
                pypto.assemble(tmp_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)

    assert isinstance(dst_tensor, pypto.tensor)

    input0_tensor = torch.rand(*self_shape, dtype=torch.float32)
    input1_tensor = torch.randint(0, self_shape[axis], indices_shape, dtype=torch.int32)
    input2_tensor = torch.rand(*self_shape, dtype=torch.float32) * 10 - 1
    c_tensor = torch.zeros_like(input0_tensor)

    pto_input0_tensor = pypto.from_torch(input0_tensor, "input0_tensor")
    pto_input1_tensor = pypto.from_torch(input1_tensor, "input1_tensor")
    pto_input2_tensor = pypto.from_torch(input2_tensor, "input2_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_input0_tensor, pto_input1_tensor, pto_input2_tensor, pto_c_tensor)

    result = input0_tensor.clone()
    for i in range(indices_shape[0]):
        for j in range(indices_shape[1]):
            if axis == 0:
                result[input1_tensor[i, j], j] += input2_tensor[i, j]
            else:
                result[i, input1_tensor[i, j]] += input2_tensor[i, j]

    assert torch.equal(c_tensor, result)
    pypto.runtime._device_fini()
