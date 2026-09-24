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

"""
PyPTO Runtime module - High-level kernel programming API.

This module provides:
- @kernel decorator: combines @pl.program + @pl.function for single-kernel use cases
- Extended Tile/Tensor types with manual MemRef specification for address control
- Re-exports of common pypto_pro.language symbols for convenience

Typical usage:
    import pypto_pro.language as pl

    @pl.jit
    def my_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        tile = pl.load(x, [0, 0], [64, 64])
        result = pl.add(tile, tile)
        return pl.store(result, [0, 0], [64, 64], x)

    # my_kernel is an ir.Program with a single function
"""

__all__ = [
    "kernel",
    "KernelDef",
    "jit",
]


from pypto.pypto_impl.ir import MemorySpace  # noqa: F401

# Preload pipeline transform — single-point feature, accessed via pl.pipeline.*
# (e.g. pl.pipeline.PipelineConfig), not exposed as top-level framework API.
from . import pipeline  # noqa: F401
from .jit import jit
from .kernel import KernelDef, kernel
