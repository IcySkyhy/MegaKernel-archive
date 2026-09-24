# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small shared parser utilities."""

from __future__ import annotations

__all__ = ["_const_int_value", "_is_const_int"]


from pypto.pypto_impl import ir


def _is_const_int(value: object) -> bool:
    """Check if a value is a compile-time constant integer."""
    if isinstance(value, (int, ir.ConstInt)):
        return True
    return isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt)


def _const_int_value(value: object) -> int | None:
    """Extract integer value from a compile-time constant, or None."""
    if isinstance(value, int):
        return value
    if isinstance(value, ir.ConstInt):
        return value.value
    if isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt):
        return -value.operand.value
    return None
