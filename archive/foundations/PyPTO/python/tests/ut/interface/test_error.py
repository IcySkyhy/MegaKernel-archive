#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Unit tests for pypto._error module.

This test suite verifies that error handling works correctly in PyPTO by
testing error scenarios that trigger different error types through the frontend.
"""

import pytest
import torch

import pypto
from pypto.error import FeError, ParserError, PyptoError, PyptoGeneralError, _catch_and_wrap_error


def test_python_error_codes_are_bound_from_cpp():
    """Test that Python wrapper error codes are exported from the C++ binding."""
    assert int(pypto.pypto_impl.ExternalError.RUNTIME_ERROR) == 0x00003
    assert int(pypto.pypto_impl.ExternalError.NAME_ERROR) == 0x00004
    assert int(pypto.pypto_impl.ExternalError.NOT_IMPLEMENTED_ERROR) == 0x00005
    assert int(pypto.pypto_impl.ExternalError.KEY_ERROR) == 0x00006
    assert int(pypto.pypto_impl.ExternalError.INVALID_OPERATION) == 0x00007
    assert int(pypto.pypto_impl.ExternalError.INVALID_TYPE) == 0x00001
    assert int(pypto.pypto_impl.ExternalError.INVALID_VAL) == 0x00002
    assert int(pypto.pypto_impl.ExternalError.OUT_OF_RANGE) == 0x00008
    assert int(pypto.pypto_impl.ExternalError.UNKNOWN) == 0x0FFFF


def test_varargs_error():
    """Test that variable-length arguments trigger proper error handling."""

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM}, new_ir=False)
    def varargs_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), *args):
        out[:] = x

    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)

    with pytest.raises(ParserError, match="Variable-length arguments"):
        varargs_kernel(x, out)


def test_kwargs_error():
    """Test that keyword arguments trigger proper error handling."""

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM}, new_ir=False)
    def kwargs_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), **kwargs):
        out[:] = x

    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)

    with pytest.raises(ParserError, match="Keyword argument packing"):
        kwargs_kernel(x, out)


def test_error_message_contains_error_code():
    """Test that error messages contain error codes."""

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM}, new_ir=False)
    def varargs_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), *args):
        out[:] = x

    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)

    try:
        varargs_kernel(x, out)
        assert False, "Should have raised an error"
    except Exception as e:
        error_str = str(e)
        assert "ErrCode: F00005" in error_str
        assert len(error_str) > 0


def test_pypto_error_init():
    err = PyptoError(0xF00001, "test error")
    assert "ErrCode: F00001" in str(err)
    assert "test error" in str(err)


def test_pypto_error_no_duplicate_errcode():
    err = PyptoError(0xF00001, "ErrCode: F00003, original error")
    assert "ErrCode: F00003" in str(err)
    assert str(err).count("ErrCode:") == 1


def test_catch_and_wrap_error_normal():
    @_catch_and_wrap_error("test operation")
    def normal_func(x):
        return x * 2

    assert normal_func(5) == 10


def test_catch_and_wrap_error_wraps_exception():
    @_catch_and_wrap_error("test operation")
    def failing_func():
        raise ValueError("test error")

    with pytest.raises(PyptoGeneralError) as exc_info:
        failing_func()

    assert "Failed to test operation" in str(exc_info.value)
    assert "test error" in str(exc_info.value)


def test_catch_and_wrap_error_preserves_errcode():
    @_catch_and_wrap_error("test operation")
    def failing_func():
        raise RuntimeError("ErrCode: F00003, some error")

    with pytest.raises(PyptoGeneralError) as exc_info:
        failing_func()

    assert "ErrCode: F00003" in str(exc_info.value)


def test_error_on_input_tensor_reassign():
    """Test that reassigning input tensor triggers ParserError."""

    @pypto.frontend.jit(
        new_ir=False,
        runtime_options={"run_mode": pypto.RunMode.SIM}, host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
    )
    def error_assign_input(
        a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(32, 32)
        c = a + b  # noqa: F841

    a = torch.rand((32, 32), dtype=torch.float16)
    b = torch.rand((32, 32), dtype=torch.float16)
    c = torch.zeros((32, 32), dtype=torch.float16)

    with pytest.raises(ParserError, match="Input tensor 'c' cannot be reassigned"):
        error_assign_input(a, b, c)


def test_error_on_non_contiguous_input_tensor():
    """Test that passing a non-contiguous input tensor triggers FeError."""

    @pypto.frontend.jit(
        new_ir=False,
        runtime_options={"run_mode": pypto.RunMode.SIM}, host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
    )
    def add(
        a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(32, 32)
        c = a + b  # noqa: F841

    a = torch.rand((32, 32), dtype=torch.float16)
    b = torch.rand((32, 32), dtype=torch.float16).transpose(0, 1)
    c = torch.zeros((32, 32), dtype=torch.float16)

    with pytest.raises(FeError, match="not all tensors are contiguous"):
        add(a, b, c)


def test_error_on_first_input_tensor_reassign():
    """Test that reassigning the first input tensor triggers ParserError."""

    @pypto.frontend.jit(
        new_ir=False,
        runtime_options={"run_mode": pypto.RunMode.SIM}, host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
    )
    def error_assign_first_input(
        a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(32, 32)
        a = b + c  # noqa: F841

    a = torch.rand((32, 32), dtype=torch.float16)
    b = torch.rand((32, 32), dtype=torch.float16)
    c = torch.zeros((32, 32), dtype=torch.float16)

    with pytest.raises(ParserError, match="Input tensor 'a' cannot be reassigned"):
        error_assign_first_input(a, b, c)


def test_error_location_on_reshape_dynamic_shape(capsys):
    """Test that reshape errors report the exact kernel source location."""

    @pypto.frontend.jit(
        new_ir=False,
        runtime_options={"run_mode": pypto.RunMode.SIM}, host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
    )
    def reshape_dynamic_shape_error(
        a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
        b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(32, 32)
        pypto.reshape(a, [a.shape[0] * a.shape[1]])

    a = torch.rand((32, 32), dtype=torch.float16)
    b = torch.rand((32, 32), dtype=torch.float16)
    c = torch.zeros((32, 32), dtype=torch.float16)

    with pytest.raises(ParserError) as exc_info:
        reshape_dynamic_shape_error(a, b, c)

    captured = capsys.readouterr()
    diagnostic = captured.out + captured.err
    expected_lineno = exc_info.value.node.lineno
    assert "reshape() requires integer shape" in str(exc_info.value)
    assert f"test_error.py:{expected_lineno}" in diagnostic
    assert "pypto.reshape(a, [a.shape[0] * a.shape[1]])" in diagnostic
    assert "^" in diagnostic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
