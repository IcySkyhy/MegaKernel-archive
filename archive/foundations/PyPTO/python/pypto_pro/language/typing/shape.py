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

"""PyPTO Pro-local tensor dimension policies.

These markers intentionally do not reuse ``python/pypto`` status enums.  PyPTO
Pro only shares the public annotation spelling with that frontend; its parser,
binding rules, and runtime remain independent.
"""

from __future__ import annotations

__all__ = ["DYNAMIC", "STATIC"]


class _ShapePolicy:
    """Identity-based singleton marker used in Tensor shape annotations."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


DYNAMIC = _ShapePolicy("DYNAMIC")
STATIC = _ShapePolicy("STATIC")
