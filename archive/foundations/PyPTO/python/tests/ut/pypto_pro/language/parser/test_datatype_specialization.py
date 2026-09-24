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
import pypto_pro.language as pl
from pypto_pro.runtime.jit import _validate_datatype_key
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest

from pypto.pypto_impl import ir


def _parse_launch_key(kernel, key):
    return getattr(kernel, "_parse_launch_key")(key)


def test_datatype_launch_without_tilingkey_uses_third_bracket_item():
    @pl.jit(datatype={"x": "x_dtype"})
    def kernel(x: pl.Ptr[pl.DT_UINT8]):
        if x_dtype == pl.DT_FP16:  # noqa: F821
            tmp: pl.DT_INT64 = 1  # noqa: F841
        else:
            tmp: pl.DT_INT64 = 2  # noqa: F841
        return

    stream, block_dim, tiling_key, dtype_key = _parse_launch_key(kernel, (None, 1, {"x": pl.DT_FP16}))

    assert stream is None
    assert block_dim == 1
    assert tiling_key is None
    assert dtype_key == {"x": pl.DT_FP16}
    kernel.to_kernel_def(datatype_consts={"x_dtype": pl.DT_FP16}).parse_target_program(
        ir.SectionKind.Vector
    )


def test_datatype_launch_with_tilingkey_requires_fourth_bracket_item():
    class TK:
        Mode = TilingKeyField(bits=1, values=[0, 1])

    @pl.jit(tiling_key=TK, datatype={"x": "x_dtype"})
    def kernel(x: pl.Ptr[pl.DT_UINT8]):
        if x_dtype == pl.DT_FP16:  # noqa: F821
            tmp: pl.DT_INT64 = Mode  # noqa: F821, F841
        else:
            tmp: pl.DT_INT64 = 0  # noqa: F841
        return

    with pytest.raises(ValueError, match="tilingkey, datatype"):
        _parse_launch_key(kernel, (None, 1, {"Mode": 0}))

    _, _, tiling_key, dtype_key = _parse_launch_key(
        kernel,
        (None, 1, {"Mode": 0}, {"x": pl.DT_BF16})
    )
    assert tiling_key == {"Mode": 0}
    assert dtype_key == {"x": pl.DT_BF16}


def test_datatype_shared_variable_requires_matching_dtypes():
    @pl.jit(datatype={"x": "io_dtype", "y": "io_dtype"})
    def kernel(x: pl.Ptr[pl.DT_UINT8], y: pl.Ptr[pl.DT_UINT8]):
        if io_dtype == pl.DT_FP16:  # noqa: F821
            tmp: pl.DT_INT64 = 1  # noqa: F841
        else:
            tmp: pl.DT_INT64 = 2  # noqa: F841
        return

    with pytest.raises(ValueError, match="disagree"):
        _validate_datatype_key(
            kernel.datatype_schema,
            {"x": pl.DT_FP16, "y": pl.DT_BF16},
        )
