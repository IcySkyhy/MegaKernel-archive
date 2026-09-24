#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
FillPad Operator System Test

FillPad 功能说明:
- 输入 tensor 有两个关键属性:
  - shape: tensor 的总大小 (pad后总大小)
  - valid_shape: 有效数据的大小
- fillpad 的作用: 将 valid_shape 之外的区域填充为指定值

示例:
  输入 tensor shape = [32, 32], valid_shape = [16, 16]
  fillpad 后: [0:16, 0:16] 保持原数据, [16:32, :] 和 [:, 16:32] 被填充为 pad_val
"""

from st.pypto_test import TestBuilder
import torch

import pypto


def op_fillpad(params, a, b):
    """
    FillPad 算子实现

    参数说明:
    - view_shape: 每个 tile 的大小 (等于 pad 后的总大小)
    - tile_shape: NPU 计算单元的 tile 形状
    - valid_shape: 有效数据的大小 (小于 view_shape 时会产生 padding 区域)
    - pad_val: 填充值
    """
    view_shape, tile_shape, valid_shape, pad_val = params
    valid_h, valid_w = valid_shape

    # 单次循环处理整个 tensor (view_shape 等于 tensor shape)
    for _ in pypto.loop(1, name="LOOP_FILLPAD_L0", idx_name="b_idx"):
        for _ in pypto.loop(1, name="LOOP_FILLPAD_L1", idx_name="s_idx"):
            offset_x = 0
            offset_y = 0

            # 创建带 valid_shape 的 view
            # shape = view_shape (pad后总大小)
            # valid_shape = [valid_h, valid_w] (有效数据大小)
            tile_a = pypto.view(a, view_shape, [offset_x, offset_y], valid_shape=[valid_h, valid_w])

            pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])

            # fillpad: 填充 valid_shape 之外的区域
            tile_res = tile_a.fillpad(mode="constant", value=pad_val)

            pypto.assemble(tile_res, [offset_x, offset_y], b)


def op_fillpad_golden(params, a, b):
    """
    FillPad Golden 实现

    将 valid_shape 之外的区域填充为 pad_val
    """
    view_shape, tile_shape, valid_shape, pad_val = params
    valid_h, valid_w = valid_shape

    result = a.clone()
    h, w = a.shape

    # 填充 valid_shape 之外的区域
    # 行方向: valid_h 之后的行
    if valid_h < h:
        result[valid_h:, :] = pad_val
    # 列方向: valid_w 之后的列
    if valid_w < w:
        result[:, valid_w:] = pad_val

    return result


class FillPadTest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)

    def get_input_from_param(self):
        view_shape = self.params[0]
        n_in, m_in = view_shape
        a_tensor = torch.rand(n_in, m_in, dtype=torch.float32) * 10
        self.setup_inputs(a_tensor)
        self.set_tol(rtol=1e-3, atol=1e-3)
        return (a_tensor,)


def test():
    """
    测试用例说明:
    - view_shape = (32, 32): pad 后的总大小
    - tile_shape = (16, 16): NPU 计算单元 tile 形状
    - valid_shape = (16, 16): 有效数据大小 (只有前 16x16 有数据)
    - pad_val = 0: 填充值

    预期结果:
    - 输出 tensor 的 [0:16, 0:16] 保持原数据
    - 输出 tensor的 [16:32, :] 和 [:, 16:32] 被填充为 0
    """
    params = ((32, 32), (16, 16), (16, 16), 0.0)
    st = FillPadTest(params, op_fillpad, op_fillpad_golden, tiling=32)
    st()


@pypto.options(pass_options={"enable_slice": True})
def test_partial_valid():
    """
    测试部分有效数据的场景
    - view_shape = (32, 32)
    - valid_shape = (20, 24): 有效数据只有 20x24
    """
    params = ((32, 32), (16, 16), (20, 24), 0.0)
    st = FillPadTest(params, op_fillpad, op_fillpad_golden, tiling=32)
    st()


@pypto.options(pass_options={"enable_slice": True})
def test_small_valid():
    """
    测试有效数据较小的场景
    - view_shape = (16, 16)
    - valid_shape = (8, 8): 有效数据只有 8x8
    """
    params = ((16, 16), (8, 8), (8, 8), 0.0)
    st = FillPadTest(params, op_fillpad, op_fillpad_golden, tiling=16)
    st()


if __name__ == "__main__":
    test()
    test_partial_valid()
    test_small_valid()
