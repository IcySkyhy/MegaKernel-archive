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

# UT for pto_assert.md documentation examples — covers all 4 call patterns.
# Run on NPU and check device log for actual output.

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


@pl.jit()
def pto_assert_doc_examples_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    flag: pl.DT_BOOL,
    offset: pl.DT_INT32,
    value: pl.DT_INT32,
):
    with pl.section_vector():
        pl.pto_assert(flag)

        pl.pto_assert(flag, "flag is false")

        # Example 3: condition + formatted message (offset != 2 is True, assert passes)
        pl.pto_assert(offset != 2, "offset=%d", offset)

        pl.pto_assert(flag, "unexpected state", loc=True)

        out[0] = 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pto_assert_doc_examples():
    """Verify all pto_assert.md documentation examples with True conditions.

    Inputs: flag=True, offset=32, value=42
    All assertions pass (conditions are True), kernel completes normally.
    Expected device log (no output since all conditions are True):
        (no assertion failure output)

    Note: To see the assertion failure output format, run test_assert_fail separately.
    """
    _check_npu()
    logging.info("------------test_pto_assert_doc_examples--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    pto_assert_doc_examples_kernel(out, True, 32, 42)
    torch.npu.synchronize()
    assert out.tolist()[0] == 1, f"got {out.tolist()}"
    logging.info("pto_assert_doc_examples passed!")


@pl.jit()
def pto_assert_fail_example_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    with pl.section_vector():
        # Trigger assertion failure to capture output format
        pl.pto_assert(False, "this should fail with offset=%d", 99)
        out[0] = 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pto_assert_fail_example():
    """Trigger assertion failure to capture device log output format.

    Expected device log:
        Assertion failed: False
        this should fail with offset=99

    Note: NPU assertion failure only logs on device, does not raise host-side exception.
    """
    _check_npu()
    logging.info("------------test_pto_assert_fail_example--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    pto_assert_fail_example_kernel(out)
    torch.npu.synchronize()
    logging.info("pto_assert_fail_example completed (check device log for output)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_pto_assert_doc_examples()
    test_pto_assert_fail_example()
    logging.info("\nAll pto_assert doc example tests passed!")
