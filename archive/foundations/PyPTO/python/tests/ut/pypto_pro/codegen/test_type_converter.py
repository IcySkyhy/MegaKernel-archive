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
"""Unit tests for TypeConverter class."""

from pypto_pro import DataType, codegen
from pypto_pro.ir import PipeType
import pytest


@pytest.mark.parametrize("dtype, expected", [
    (DataType.FP16, "half"),
    (DataType.FP32, "float"),
    (DataType.INT8, "int8_t"),
    (DataType.INT32, "int32_t"),
    (DataType.BF16, "bfloat16_t"),
    (DataType.UINT8, "uint8_t"),
])
def test_convert_data_type(dtype, expected):
    """Test DataType to C++ type string conversion."""
    assert dtype.to_c_type_string() == expected


@pytest.mark.parametrize("pipe_type, expected", [
    (PipeType.MTE1, "PIPE_MTE1"),
    (PipeType.MTE2, "PIPE_MTE2"),
    (PipeType.MTE3, "PIPE_MTE3"),
    (PipeType.M, "PIPE_M"),
    (PipeType.V, "PIPE_V"),
    (PipeType.S, "PIPE_S"),
    (PipeType.FIX, "PIPE_FIX"),
    (PipeType.ALL, "PIPE_ALL"),
])
def test_convert_pipe_type(pipe_type, expected):
    """Test PipeType to C++ string conversion."""
    converter = codegen.TypeConverter()
    assert converter.ConvertPipeType(pipe_type) == expected


@pytest.mark.parametrize("event_id, expected", [
    (0, "EVENT_ID0"),
    (1, "EVENT_ID1"),
    (3, "EVENT_ID3"),
    (7, "EVENT_ID7"),
])
def test_convert_event_id(event_id, expected):
    """Test valid event ID conversion."""
    converter = codegen.TypeConverter()
    assert converter.ConvertEventId(event_id) == expected


@pytest.mark.parametrize("invalid_id", [-1, 8, 100, -100])
def test_convert_event_id_invalid(invalid_id):
    """Test event ID with invalid value raises error."""
    converter = codegen.TypeConverter()
    with pytest.raises(RuntimeError, match=rf"Event ID must be in range \[0, 7\].*got {invalid_id}"):
        converter.ConvertEventId(invalid_id)
