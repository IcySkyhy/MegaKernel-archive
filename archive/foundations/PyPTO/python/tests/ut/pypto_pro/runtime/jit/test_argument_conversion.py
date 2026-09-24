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
"""Pure-Python unit tests for _validate_args() and _args_to_ctypes().

No hardware or external tools are required.
"""

import ctypes
import importlib
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock torch before any pypto_pro imports so jit.py's top-level 'import torch'
# resolves to our MagicMock.  torch.Tensor is replaced with a concrete class
# so that isinstance() checks inside _validate_args() work correctly.
# ---------------------------------------------------------------------------
_torch_mock = MagicMock()


class _MockTensor:
    """Minimal stand-in for torch.Tensor used in validation tests."""

    def __init__(self, shape: list[int], dtype):
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def data_ptr() -> int:
        return 0


# Assign the concrete class so isinstance(x, torch.Tensor) works in jit code.
_torch_mock.Tensor = _MockTensor

DataType = importlib.import_module("pypto_pro").DataType
_jit = importlib.import_module("pypto_pro.runtime.jit")
ParamSpec = _jit.ParamSpec
ParamKind = _jit.ParamKind
_args_to_ctypes = getattr(_jit, "_args_to_ctypes")
_collect_dyn_vars = getattr(_jit, "_collect_dyn_vars")
_validate_args = getattr(_jit, "_validate_args")
torch = importlib.import_module("torch")


@pytest.fixture(autouse=True)
def _mock_torch():
    """Inject mock torch into sys.modules for the duration of each test only."""
    _jit_mod = importlib.import_module("pypto_pro.runtime.jit")

    _saved_torch = sys.modules.get("torch")
    _saved_npu = sys.modules.get("torch.npu")
    _saved_jit_torch = _jit_mod.torch
    sys.modules["torch"] = _torch_mock
    sys.modules["torch.npu"] = MagicMock()
    _jit_mod.torch = _torch_mock
    yield
    if _saved_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _saved_torch
    if _saved_npu is None:
        sys.modules.pop("torch.npu", None)
    else:
        sys.modules["torch.npu"] = _saved_npu
    _jit_mod.torch = _saved_jit_torch


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------


def _tensor_spec(name, dtype, shape) -> ParamSpec:
    return ParamSpec(name, ParamKind.TENSOR, dtype, shape)


def _ptr_spec(name, dtype) -> ParamSpec:
    return ParamSpec(name, ParamKind.PTR, dtype, None)


def _scalar_spec(name, dtype) -> ParamSpec:
    return ParamSpec(name, ParamKind.SCALAR, dtype, None)


def test_validate_args_count_mismatch():
    """Wrong number of args raises TypeError."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    with pytest.raises(TypeError, match="Expected 1 args, got 2"):
        _validate_args(
            (_MockTensor([64], torch.float32), _MockTensor([64], torch.float32)), specs
        )


@pytest.mark.parametrize(
    "specs, args",
    [
        ([_tensor_spec("x", DataType.FP32, [64])], (_MockTensor([64], torch.float32),)),
        ([_tensor_spec("x", DataType.FP32, [-1, 128])], (_MockTensor([999, 128], torch.float32),)),
        ([_ptr_spec("p", DataType.FP32)], (_MockTensor([1], torch.float32),)),
        ([_scalar_spec("n", DataType.INT32)], (42,)),
        ([_scalar_spec("scale", DataType.FP32)], (1.5,)),
        ([_scalar_spec("flag", DataType.INT32)], (True,)),
        ([_scalar_spec("flag", DataType.BOOL)], (False,)),
        ([_tensor_spec("x", DataType.FP32, [-1, 64])], (_MockTensor([999, 64], torch.float32),)),
    ],
)
def test_validate_args_accepts_supported_args(specs, args):
    """Supported tensor, ptr, scalar, bool and dynamic-dim args pass validation."""
    _validate_args(args, specs)


@pytest.mark.parametrize(
    "specs, args, error",
    [
        ([_tensor_spec("x", DataType.FP32, [64])], (3.14,), "expected torch.Tensor"),
        ([_tensor_spec("x", DataType.INT32, [64])], (_MockTensor([64], torch.float32),), "dtype mismatch"),
        ([_tensor_spec("x", DataType.FP32, [64, 128])], (_MockTensor([64], torch.float32),), "rank mismatch"),
        (
            [_tensor_spec("x", DataType.FP32, [64, 128])],
            (_MockTensor([64, 256], torch.float32),),
            "dim\\[1\\] mismatch",
        ),
        ([_scalar_spec("n", DataType.INT32)], (_MockTensor([1], torch.float32),), "expected Python scalar"),
        ([_scalar_spec("n", DataType.INT32)], (1.5,), "float value passed for non-float dtype"),
        ([_scalar_spec("scale", DataType.FP32)], (42,), "int value passed for non-integer dtype"),
        (
            [_scalar_spec("scale", DataType.FP32)],
            (True,),
            "bool value passed for non-boolean/non-integer dtype",
        ),
    ],
)
def test_validate_args_rejects_invalid_args(specs, args, error):
    """Invalid tensor and scalar args fail with the expected validation family."""
    with pytest.raises(TypeError, match=error):
        _validate_args(args, specs)


def test_validate_args_mixed_params():
    """Mixed tensor + scalar params all validated correctly."""
    specs = [
        _tensor_spec("input", DataType.FP32, [64]),
        _tensor_spec("output", DataType.FP32, [64]),
        _scalar_spec("n", DataType.INT32),
    ]
    args = (
        _MockTensor([64], torch.float32),
        _MockTensor([64], torch.float32),
        128,
    )
    _validate_args(args, specs)  # must not raise


# ---------------------------------------------------------------------------
# _args_to_ctypes
# ---------------------------------------------------------------------------


def test_args_to_ctypes_tensor():
    """torch.Tensor → c_void_p wrapping data_ptr()."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    tensor = _MockTensor([64], torch.float32)
    result = _args_to_ctypes((tensor,), specs)
    assert len(result) == 1
    assert isinstance(result[0], ctypes.c_void_p)
    assert (
        result[0].value == tensor.data_ptr() or result[0].value is None
    )  # 0 maps to None


@pytest.mark.parametrize(
    "spec, arg, ctype, expected",
    [
        (_ptr_spec("p", DataType.FP32), _MockTensor([1], torch.float32), ctypes.c_void_p, None),
        (_scalar_spec("scale", DataType.FP32), 1.5, ctypes.c_float, 1.5),
        (_scalar_spec("n", DataType.INT32), 42, ctypes.c_int32, 42),
    ],
)
def test_args_to_ctypes_single_arg(spec, arg, ctype, expected):
    """ptr and scalar params are converted to the expected ctypes wrappers."""
    result = _args_to_ctypes((arg,), [spec])
    assert len(result) == 1
    assert isinstance(result[0], ctype)
    if expected is not None:
        assert result[0].value == pytest.approx(expected)


def test_args_to_ctypes_mixed():
    """Mixed tensor + scalar → correct ctypes sequence."""
    specs = [
        _tensor_spec("input", DataType.FP32, [64]),
        _scalar_spec("n", DataType.INT32),
    ]
    tensor = _MockTensor([64], torch.float32)
    result = _args_to_ctypes((tensor, 7), specs)
    assert isinstance(result[0], ctypes.c_void_p)
    assert isinstance(result[1], ctypes.c_int32)
    assert result[1].value == 7


# ---------------------------------------------------------------------------
# _collect_dyn_vars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "specs, expected",
    [
        (
            [
                _tensor_spec("a", DataType.FP32, ["M", "N"]),
                _tensor_spec("b", DataType.FP32, ["M", "N"]),
            ],
            ["M", "N"],
        ),
        (
            [
                _tensor_spec("a", DataType.FP32, ["M", "N"]),
                _tensor_spec("b", DataType.FP32, ["N", "K"]),
            ],
            ["M", "N", "K"],
        ),
        (
            [
                _tensor_spec("a", DataType.FP32, [64, 128]),
                _tensor_spec("b", DataType.FP32, [128]),
            ],
            [],
        ),
    ],
)
def test_collect_dyn_vars(specs, expected):
    """Dynamic vars are deduped in appearance order."""
    assert _collect_dyn_vars(specs) == expected


# ---------------------------------------------------------------------------
# _validate_args — dynamic variable tests
# ---------------------------------------------------------------------------


def test_validate_args_dyn_consistent():
    """M=64 in both tensors — consistent, must not raise."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", 128]),
        _tensor_spec("b", DataType.FP32, ["M", 64]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([64, 64], torch.float32),
    )
    _validate_args(args, specs)  # must not raise


def test_validate_args_dyn_inconsistent():
    """M=64 in tensor a but M=128 in tensor b — must raise TypeError mentioning 'M'."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", 128]),
        _tensor_spec("b", DataType.FP32, ["M", 64]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([128, 64], torch.float32),
    )
    with pytest.raises(TypeError, match="M"):
        _validate_args(args, specs)


# ---------------------------------------------------------------------------
# _args_to_ctypes — dynamic variable appending tests
# ---------------------------------------------------------------------------


def test_args_to_ctypes_dyn_appended():
    """Tensor with shape ["M", 64]: result should have c_void_p then c_int64(M_value)."""
    m_var = 32
    specs = [_tensor_spec("a", DataType.FP32, ["M", 64])]
    tensor = _MockTensor([m_var, 64], torch.float32)
    result = _args_to_ctypes((tensor,), specs)
    assert len(result) == 2
    assert isinstance(result[0], ctypes.c_void_p)
    assert isinstance(result[1], ctypes.c_int64)
    assert result[1].value == m_var


def test_args_to_ctypes_dyn_order():
    """Tensor a(M,N) and tensor b(N,K): appended order must be M, N, K."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["N", "K"]),
    ]
    args = (
        _MockTensor([10, 20], torch.float32),
        _MockTensor([20, 30], torch.float32),
    )
    result = _args_to_ctypes(args, specs)
    assert len(result) == 5
    assert isinstance(result[2], ctypes.c_int64) and result[2].value == 10  # M
    assert isinstance(result[3], ctypes.c_int64) and result[3].value == 20  # N
    assert isinstance(result[4], ctypes.c_int64) and result[4].value == 30  # K


def test_args_to_ctypes_dyn_dedup():
    """Two tensors sharing M,N: each var appended exactly once."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["M", "N"]),
    ]
    args = (
        _MockTensor([8, 16], torch.float32),
        _MockTensor([8, 16], torch.float32),
    )
    result = _args_to_ctypes(args, specs)
    assert len(result) == 4
    assert result[2].value == 8  # M
    assert result[3].value == 16  # N


def test_args_to_ctypes_no_dyn():
    """No dynamic dims: result length equals number of args."""
    specs = [
        _tensor_spec("a", DataType.FP32, [64, 128]),
        _tensor_spec("b", DataType.FP32, [128]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([128], torch.float32),
    )
    result = _args_to_ctypes(args, specs)
    assert len(result) == len(args)
