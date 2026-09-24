# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ternary type-promotion control_flow 前端测试。

测试覆盖场景:
  1. 覆盖同整数类别三元表达式类型提升：DT_INT8/DT_INT16/DT_INT32/DT_INT64 与 INDEX。
  2. 覆盖浮点类别三元表达式类型提升：DT_FP16 与 DT_FP32。
  3. 覆盖 seqused_q/seqused_k 标量读取与 shape[1] INDEX 混用的真实 used-size 场景。
  4. 覆盖深层嵌套三元表达式，确保内层 if 表达式位于正确控制流上下文。
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# =============================================================================
# Test 1: 5 层嵌套三元 - 整数同类别类型提升
#         5-layer nested ternary - same-category integer type promotion
# =============================================================================
@pl.jit(auto_mutex=True)
def ternary_promote_nested_int_kernel(
    int8_data: pl.Tensor[[pl.DYNAMIC], pl.DT_INT8],
    int16_data: pl.Tensor[[pl.DYNAMIC], pl.DT_INT16],
    int32_data: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    int64_data: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[8], pl.DT_INT64],
):
    data_len = int32_data.shape[0]
    with pl.section_vector():
        flag0 = pl.getval(int32_data, 0) > 0
        flag1 = pl.getval(int32_data, 1) > 0
        flag2 = pl.getval(int32_data, 2) > 0
        flag3 = pl.getval(int32_data, 3) > 0
        flag4 = pl.getval(int32_data, 4) > 0

        nested_value = (
            (pl.getval(int16_data, 0) if flag1 else pl.getval(int32_data, 5))
            if flag0
            else (
                (data_len if flag2 else pl.getval(int32_data, 6))
                if flag1
                else (
                    (pl.getval(int32_data, 7) if flag3 else pl.getval(int64_data, 0))
                    if flag2
                    else (
                        (pl.getval(int8_data, 0) if flag4 else pl.getval(int32_data, 8))
                        if flag3
                        else (pl.getval(int16_data, 1) if flag4 else pl.getval(int64_data, 1))
                    )
                )
            )
        )

        int32_index_value = pl.getval(int32_data, 5) if flag0 else data_len
        index_int32_value = data_len if flag1 else pl.getval(int32_data, 6)
        int32_int64_value = pl.getval(int32_data, 7) if flag2 else pl.getval(int64_data, 0)
        int16_int32_value = pl.getval(int16_data, 0) if flag0 else pl.getval(int32_data, 9)
        int16_int64_value = pl.getval(int16_data, 1) if flag1 else pl.getval(int64_data, 1)
        int8_int32_value = pl.getval(int8_data, 0) if flag2 else pl.getval(int32_data, 9)

        pl.setval(out, 0, nested_value)
        pl.setval(out, 1, int32_index_value)
        pl.setval(out, 2, index_int32_value)
        pl.setval(out, 3, int32_int64_value)
        pl.setval(out, 4, data_len)
        pl.setval(out, 5, int16_int32_value)
        pl.setval(out, 6, int16_int64_value)
        pl.setval(out, 7, int8_int32_value)


# =============================================================================
# Test 2: 5 层嵌套三元 - 浮点同类别类型提升
#         5-layer nested ternary - same-category floating type promotion
# =============================================================================
@pl.jit(auto_mutex=True)
def ternary_promote_nested_float_kernel(
    fp16_data: pl.Tensor[[pl.DYNAMIC], pl.DT_FP16],
    fp32_data: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    guard: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[5], pl.DT_FP32],
):
    with pl.section_vector():
        flag0 = pl.getval(guard, 0) > 0
        flag1 = pl.getval(guard, 1) > 0
        flag2 = pl.getval(guard, 2) > 0
        flag3 = pl.getval(guard, 3) > 0
        flag4 = pl.getval(guard, 4) > 0

        nested_value = (
            (pl.getval(fp16_data, 0) if flag1 else pl.getval(fp32_data, 0))
            if flag0
            else (
                (pl.getval(fp32_data, 1) if flag2 else pl.getval(fp16_data, 1))
                if flag1
                else (
                    (pl.getval(fp16_data, 2) if flag3 else pl.getval(fp32_data, 2))
                    if flag2
                    else (
                        (pl.getval(fp32_data, 3) if flag4 else pl.getval(fp16_data, 3))
                        if flag3
                        else (pl.getval(fp16_data, 4) if flag4 else pl.getval(fp32_data, 4))
                    )
                )
            )
        )

        fp16_fp32_value = pl.getval(fp16_data, 0) if flag0 else pl.getval(fp32_data, 0)
        fp32_fp16_value = pl.getval(fp32_data, 1) if flag1 else pl.getval(fp16_data, 1)

        pl.setval(out, 0, nested_value)
        pl.setval(out, 1, fp16_fp32_value)
        pl.setval(out, 2, fp32_fp16_value)
        pl.setval(out, 3, pl.getval(fp16_data, 2) if flag2 else pl.getval(fp32_data, 2))
        pl.setval(out, 4, pl.getval(fp32_data, 3) if flag3 else pl.getval(fp16_data, 3))


# =============================================================================
# Test 3: used-size 真实场景 - Tensor getval 与 shape[1] INDEX 混用
#         Real used-size pattern - tensor getval mixed with shape[1] INDEX
# =============================================================================
@pl.jit(auto_mutex=True)
def ternary_promote_used_size_kernel(
    query_index: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    key_index: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    seqused_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    seqused_k: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    used_flags: pl.Tensor[[2], pl.DT_INT32],
    out: pl.Tensor[[2], pl.DT_INT64],
):
    s1_dim = query_index.shape[1]
    s2_dim = key_index.shape[1]
    b_idx = 0
    q_used = pl.getval(used_flags, 0) > 0
    k_used = pl.getval(used_flags, 1) > 0

    with pl.section_vector():
        q_used_size = pl.getval(seqused_q, b_idx) if q_used else s1_dim
        k_used_size = pl.getval(seqused_k, b_idx) if k_used else s2_dim
        pl.setval(out, 0, q_used_size)
        pl.setval(out, 1, k_used_size)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_promote_nested_int():
    device = ST_DEVICE
    torch.npu.set_device(device)
    int8_data = torch.tensor([7, 11], device=device, dtype=torch.int8)
    int16_data = torch.tensor([31, 37], device=device, dtype=torch.int16)
    int32_data = torch.tensor([1, 1, -1, 1, -1, 13, 17, 19, 23, 29], device=device, dtype=torch.int32)
    int64_data = torch.tensor([101, 202], device=device, dtype=torch.int64)
    out = torch.zeros(8, device=device, dtype=torch.int64)

    ternary_promote_nested_int_kernel(int8_data, int16_data, int32_data, int64_data, out)
    torch.npu.synchronize()

    expected = torch.tensor([31, 13, 10, 101, 10, 31, 37, 29], device=device, dtype=torch.int64)
    assert torch.equal(out, expected), f"got {out.cpu().tolist()}, expected {expected.cpu().tolist()}"
    logging.info("test_ternary_promote_nested_int passed")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_promote_nested_float():
    device = ST_DEVICE
    torch.npu.set_device(device)
    fp16_data = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], device=device, dtype=torch.float16)
    fp32_data = torch.tensor([10.25, 20.25, 30.25, 40.25, 50.25], device=device, dtype=torch.float32)
    guard = torch.tensor([-1, -1, 1, 1, -1], device=device, dtype=torch.int32)
    out = torch.zeros(5, device=device, dtype=torch.float32)

    ternary_promote_nested_float_kernel(fp16_data, fp32_data, guard, out)
    torch.npu.synchronize()

    expected = torch.tensor([3.5, 10.25, 2.5, 3.5, 40.25], device=device, dtype=torch.float32)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
    logging.info("test_ternary_promote_nested_float passed")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_promote_used_size():
    device = ST_DEVICE
    torch.npu.set_device(device)
    query_index = torch.zeros((2, 4), device=device, dtype=torch.int32)
    key_index = torch.zeros((2, 6), device=device, dtype=torch.int32)
    seqused_q = torch.tensor([7, 8], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([11, 12], device=device, dtype=torch.int32)
    used_flags = torch.tensor([1, 0], device=device, dtype=torch.int32)
    out = torch.zeros(2, device=device, dtype=torch.int64)

    ternary_promote_used_size_kernel(query_index, key_index, seqused_q, seqused_k, used_flags, out)
    torch.npu.synchronize()

    expected = torch.tensor([7, 6], device=device, dtype=torch.int64)
    assert torch.equal(out, expected), f"got {out.cpu().tolist()}, expected {expected.cpu().tolist()}"
    logging.info("test_ternary_promote_used_size passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_ternary_promote_nested_int()
    test_ternary_promote_nested_float()
    test_ternary_promote_used_size()
