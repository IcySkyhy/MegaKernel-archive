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

"""Type wrappers for PyPTO Language DSL.

This module provides type annotation classes for PyPTO's language DSL:
- Scalar: Scalar values with specific data types
- Tensor: Multi-dimensional arrays in global memory
"""

__all__ = ["DYNAMIC", "STATIC", "Ptr", "Scalar", "Tensor"]

from pypto_pro.language.typing.ptr import Ptr
from pypto_pro.language.typing.scalar import Scalar
from pypto_pro.language.typing.shape import DYNAMIC, STATIC
from pypto_pro.language.typing.tensor import Tensor
