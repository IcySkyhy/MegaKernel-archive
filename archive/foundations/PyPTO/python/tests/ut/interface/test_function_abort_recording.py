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
"""Unit tests for function recording abort / cleanup paths."""

from unittest import mock

import pytest

import pypto
import pypto._controller as controller
from pypto._controller import Controller, _finalize_function_recording, function
from pypto.error import FeError


def test_finalize_function_recording_none_func():
    assert _finalize_function_recording(None, None) is None
    user_err = ValueError("user")
    assert _finalize_function_recording(None, user_err) is user_err


def test_finalize_end_function_success():
    func = mock.Mock()
    assert _finalize_function_recording(func, None) is None
    func.EndFunction.assert_called_once()
    func.AbortRecording.assert_not_called()


def test_finalize_abort_on_user_error():
    func = mock.Mock()
    user_err = RuntimeError("user")
    assert _finalize_function_recording(func, user_err) is user_err
    func.AbortRecording.assert_called_once()
    func.EndFunction.assert_not_called()


def test_finalize_abort_failure_preserves_user_error():
    func = mock.Mock()
    func.AbortRecording.side_effect = RuntimeError("abort fail")
    user_err = ValueError("user")
    assert _finalize_function_recording(func, user_err) is user_err


def test_finalize_end_function_failure_returns_end_error():
    func = mock.Mock()
    func.EndFunction.side_effect = RuntimeError("end fail")
    result = _finalize_function_recording(func, None)
    assert isinstance(result, RuntimeError)
    assert str(result) == "end fail"


def test_function_record_func_construct_failure_resets_in_function():
    controller.reset()
    Controller.in_function = False
    with mock.patch(
        "pypto._controller.pypto_impl.RecordFunc",
        side_effect=RuntimeError("construct fail"),
    ):
        with pytest.raises(FeError, match="construct fail"):
            with function("main"):
                pass
    assert Controller.in_function is False


def test_function_user_error_resets_in_function():
    controller.reset()
    a = pypto.tensor((8, 8), pypto.DT_FP32, "a")
    b = pypto.tensor((8, 8), pypto.DT_FP32, "b")

    with pytest.raises(FeError, match="boom"):
        with pypto.function("MAIN", a, b):
            raise ValueError("boom")
    assert Controller.in_function is False

    with pypto.function("MAIN2", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        b.move(pypto.add(a, b))
    assert Controller.in_function is False


def test_function_abort_recording_failure_still_resets_in_function():
    controller.reset()
    a = pypto.tensor((8, 8), pypto.DT_FP32, "a")
    b = pypto.tensor((8, 8), pypto.DT_FP32, "b")

    with mock.patch(
        "pypto._controller.pypto_impl.RecordFunc.AbortRecording",
        side_effect=RuntimeError("abort fail"),
    ):
        with pytest.raises(FeError, match="user"):
            with pypto.function("MAIN", a, b):
                raise ValueError("user")
    assert Controller.in_function is False
