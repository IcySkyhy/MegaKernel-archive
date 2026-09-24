# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared fixtures for A5 SIMT scalar operation system tests."""

import os

import pytest
import torch


@pytest.fixture(scope="session")
def a5_device():
    """Select and return one available Ascend 950 device."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")
    return device


@pytest.fixture
def assert_simt_close():
    """Compare one SIMT result using a dtype-appropriate tolerance."""

    def _assert(actual, expected, *, rtol=None, atol=None):
        actual = actual.cpu()
        if not actual.is_floating_point():
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return
        default_tolerance = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }
        default_rtol, default_atol = default_tolerance[actual.dtype]
        torch.testing.assert_close(
            actual,
            expected,
            rtol=default_rtol if rtol is None else rtol,
            atol=default_atol if atol is None else atol,
            equal_nan=True,
        )

    return _assert


@pytest.fixture
def run_float_unary(a5_device, assert_simt_close):
    """Run a unary interface for FP16, BF16, and FP32."""

    def _run(kernel, values, golden, **tolerances):
        fp32 = values.to(torch.float32).reshape(1, -1)
        sources = (fp32.to(torch.float16), fp32.to(torch.bfloat16), fp32)
        outputs = tuple(torch.empty_like(source, device=a5_device) for source in sources)
        kernel(*(source.to(a5_device) for source in sources), *outputs)
        torch.npu.synchronize()
        for source, output in zip(sources, outputs):
            expected = golden(source.to(torch.float32)).to(source.dtype)
            assert_simt_close(output, expected, **tolerances)

    return _run


@pytest.fixture
def run_float_binary(a5_device, assert_simt_close):
    """Run a binary interface for FP16, BF16, and FP32."""

    def _run(kernel, lhs_values, rhs_values, golden, **tolerances):
        lhs_fp32 = lhs_values.to(torch.float32).reshape(1, -1)
        rhs_fp32 = rhs_values.to(torch.float32).reshape(1, -1)
        operand_pairs = tuple(
            (lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        outputs = tuple(torch.empty_like(lhs, device=a5_device) for lhs, _ in operand_pairs)
        arguments = [operand.to(a5_device) for pair in operand_pairs for operand in pair]
        kernel(*arguments, *outputs)
        torch.npu.synchronize()
        for (lhs, rhs), output in zip(operand_pairs, outputs):
            expected = golden(lhs.to(torch.float32), rhs.to(torch.float32)).to(lhs.dtype)
            assert_simt_close(output, expected, **tolerances)

    return _run


@pytest.fixture
def run_float_predicate(a5_device, assert_simt_close):
    """Run a floating-point classification interface for all supported dtypes."""

    def _run(kernel, values, golden):
        fp32 = values.to(torch.float32).reshape(1, -1)
        sources = (fp32.to(torch.float16), fp32.to(torch.bfloat16), fp32)
        outputs = tuple(torch.empty(source.shape, dtype=torch.bool, device=a5_device) for source in sources)
        kernel(*(source.to(a5_device) for source in sources), *outputs)
        torch.npu.synchronize()
        for source, output in zip(sources, outputs):
            assert_simt_close(output, golden(source))

    return _run


@pytest.fixture
def run_fp32_unary(a5_device, assert_simt_close):
    """Run an FP32-only unary interface."""

    def _run(kernel, values, golden, **tolerances):
        source = values.to(torch.float32).reshape(1, -1)
        output = torch.empty_like(source, device=a5_device)
        kernel(source.to(a5_device), output)
        torch.npu.synchronize()
        assert_simt_close(output, golden(source), **tolerances)

    return _run


@pytest.fixture
def run_float_ternary(a5_device, assert_simt_close):
    """Run a ternary interface for FP16, BF16, and FP32."""

    def _run(kernel, lhs_values, rhs_values, addend_values, golden, **tolerances):
        fp32_operands = tuple(
            values.to(torch.float32).reshape(1, -1) for values in (lhs_values, rhs_values, addend_values)
        )
        operand_triples = tuple(
            tuple(operand.to(dtype) for operand in fp32_operands)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        outputs = tuple(torch.empty_like(lhs, device=a5_device) for lhs, _, _ in operand_triples)
        arguments = [operand.to(a5_device) for triple in operand_triples for operand in triple]
        kernel(*arguments, *outputs)
        torch.npu.synchronize()
        for (lhs, rhs, addend), output in zip(operand_triples, outputs):
            expected = golden(
                lhs.to(torch.float32),
                rhs.to(torch.float32),
                addend.to(torch.float32),
            ).to(lhs.dtype)
            assert_simt_close(output, expected, **tolerances)

    return _run
