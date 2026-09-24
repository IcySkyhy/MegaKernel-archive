# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 NPU coverage for Python-compatible scalar numeric semantics."""

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 — registers npu backend

import pypto


@pl.jit()
def scalar_numeric_semantics_kernel(
    lhs: pl.DT_INT32,
    rhs: pl.DT_INT32,
    float_out: pl.Tensor[[1], pl.DT_FP32],
    int_out: pl.Tensor[[6], pl.DT_INT32],
):
    with pl.section_vector():
        flag = lhs > rhs
        float_out[0] = lhs / rhs
        int_out[0] = flag + 7
        int_out[1] = flag & 3
        int_out[2] = flag | 2
        int_out[3] = flag ^ 3
        int_out[4] = flag == 1
        int_out[5] = flag < 2


def _run_scalar_case(lhs, rhs, expected_div, expected_int):
    device = "npu:0"
    torch.npu.set_device(device)
    float_out = torch.zeros(1, device=device, dtype=torch.float32)
    int_out = torch.zeros(6, device=device, dtype=torch.int32)

    scalar_numeric_semantics_kernel(lhs, rhs, float_out, int_out)
    torch.npu.synchronize()

    expected_float = torch.tensor([expected_div], device=device, dtype=torch.float32)
    expected_numeric = torch.tensor(expected_int, device=device, dtype=torch.int32)
    torch.testing.assert_close(float_out, expected_float, rtol=0, atol=0)
    assert torch.equal(int_out, expected_numeric), (
        f"got {int_out.cpu().tolist()}, expected {expected_numeric.cpu().tolist()}"
    )


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_runtime_int_truediv_and_bool_numeric_semantics():
    _run_scalar_case(7, 2, 3.5, [8, 1, 3, 2, 1, 1])
    _run_scalar_case(1, 2, 0.5, [7, 0, 2, 3, 0, 1])


if __name__ == "__main__":
    test_runtime_int_truediv_and_bool_numeric_semantics()
