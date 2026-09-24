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
"""Block-facing smoke tests for DataType exports.

Full DataType behavior is covered by the C++ IR tests and the generic Python IR
binding tests. The block suite only checks the pypto_pro entry points that the
Block DSL depends on.
"""

import pypto_pro
from pypto_pro import DataType, ir
import pypto_pro.language as pl


def test_pypto_pro_reexports_common_block_dtypes():
    assert pypto_pro.DT_FP16 == DataType.FP16
    assert pypto_pro.DT_FP32 == DataType.FP32
    assert pypto_pro.DT_INT32 == DataType.INT32
    assert pypto_pro.DT_BOOL == DataType.BOOL


def test_language_dtype_aliases_are_usable_in_block_annotations():
    tensor_type = pl.Tensor[[16, 32], pl.DT_FP16]

    assert isinstance(tensor_type, pl.Tensor)
    assert tensor_type.dtype == DataType.FP16
    assert tensor_type.shape == [16, 32]


def test_index_dtype_is_available_for_block_shape_symbols():
    shape_var = ir.Var("m", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    tensor_type = ir.TensorType([shape_var], DataType.FP16)

    assert tensor_type.shape[0] is shape_var
    assert shape_var.type.dtype == DataType.INDEX
    assert DataType.INDEX.to_string() == "index"
