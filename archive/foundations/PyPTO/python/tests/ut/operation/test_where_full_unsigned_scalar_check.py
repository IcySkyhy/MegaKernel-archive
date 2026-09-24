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
"""Scalar dtype interception tests for where/full.

Covers two scenarios that must be rejected at the Python entry:
1. Unsigned integer dtype + negative scalar (mirrors gcd scalar range check in
   binary.cpp GetIntegerScalarRange), so downstream torch tensor conversion does
   not fail ambiguously.
2. float scalar paired with a non-fp32 tensor (e.g. fp16): input and other have
   inconsistent dtype. Previously this slipped through to the C++ calc layer and
   surfaced as a misleading "input/output type inconsistent" error; it should
   instead be reported as an input/other dtype mismatch.
"""

import pytest

import pypto
from pypto.error import PyptoError


def test_where_negative_scalar_uint8_other_side():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_UINT8, "x")
    with pytest.raises(PyptoError, match="negative"):
        pypto.where(cond, x, -100)


def test_where_negative_scalar_uint8_input_side():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    y = pypto.tensor([4], pypto.DT_UINT8, "y")
    with pytest.raises(PyptoError, match="negative"):
        pypto.where(cond, -100, y)


def test_where_negative_scalar_uint16():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_UINT16, "x")
    with pytest.raises(PyptoError, match="negative"):
        pypto.where(cond, x, -1)


def test_where_negative_scalar_uint32():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_UINT32, "x")
    with pytest.raises(PyptoError, match="negative"):
        pypto.where(cond, x, -1)


def test_where_float_scalar_fp16_tensor_other_side():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_FP16, "x")
    with pytest.raises(PyptoError, match="data type inconsistent"):
        pypto.where(cond, x, 1.0)


def test_where_float_scalar_fp16_tensor_input_side():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    y = pypto.tensor([4], pypto.DT_FP16, "y")
    with pytest.raises(PyptoError, match="data type inconsistent"):
        pypto.where(cond, 1.0, y)


def test_where_float_scalar_bf16_tensor():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_BF16, "x")
    with pytest.raises(PyptoError, match="data type inconsistent"):
        pypto.where(cond, x, 0.0)


def test_where_element_tensor_dtype_mismatch():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_FP16, "x")
    with pytest.raises(PyptoError, match="data type inconsistent"):
        pypto.where(cond, x, pypto.Element(pypto.DT_FP32, 1.0))


def test_where_tensor_element_dtype_mismatch():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    y = pypto.tensor([4], pypto.DT_FP32, "y")
    with pytest.raises(PyptoError, match="data type inconsistent"):
        pypto.where(cond, pypto.Element(pypto.DT_FP16, 1.0), y)


def test_where_element_element_dtype_mismatch():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    with pytest.raises(PyptoError, match="data type inconsistent"):
        pypto.where(cond, pypto.Element(pypto.DT_FP16, 1.0), pypto.Element(pypto.DT_FP32, 0.0))


def test_where_tensor_tensor_dtype_mismatch():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_FP16, "x")
    y = pypto.tensor([4], pypto.DT_FP32, "y")
    with pytest.raises(PyptoError, match="data type inconsistent"):
        pypto.where(cond, x, y)


def test_full_negative_fill_value_uint8():
    with pytest.raises(PyptoError, match="negative"):
        pypto.full([2, 2], -100, pypto.DT_UINT8)


def test_full_negative_fill_value_uint16():
    with pytest.raises(PyptoError, match="negative"):
        pypto.full([2, 2], -1, pypto.DT_UINT16)


def test_full_negative_fill_value_uint32():
    with pytest.raises(PyptoError, match="negative"):
        pypto.full([2, 2], -1, pypto.DT_UINT32)


def test_full_negative_float_fill_value_uint8():
    with pytest.raises(PyptoError, match="negative"):
        pypto.full([2, 2], -1.0, pypto.DT_UINT8)


def test_full_negative_element_uint8():
    with pytest.raises(PyptoError, match="negative"):
        pypto.full([2, 2], pypto.Element(pypto.DT_UINT8, -100), pypto.DT_UINT8)


def test_full_negative_element_uint16():
    with pytest.raises(PyptoError, match="negative"):
        pypto.full([2, 2], pypto.Element(pypto.DT_UINT16, -1), pypto.DT_UINT16)


def test_where_negative_element_uint8_other_side():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    x = pypto.tensor([4], pypto.DT_UINT8, "x")
    with pytest.raises(PyptoError, match="negative"):
        pypto.where(cond, x, pypto.Element(pypto.DT_UINT8, -100))


def test_where_negative_element_uint8_input_side():
    cond = pypto.tensor([4], pypto.DT_BOOL, "cond")
    y = pypto.tensor([4], pypto.DT_UINT8, "y")
    with pytest.raises(PyptoError, match="negative"):
        pypto.where(cond, pypto.Element(pypto.DT_UINT8, -100), y)
