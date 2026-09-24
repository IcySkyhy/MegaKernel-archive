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

"""PyPTO error codes used by parser diagnostics."""

from __future__ import annotations

__all__ = ["ErrorCode", "get_error_code"]


from enum import IntEnum

from pypto.pypto_impl import ExternalError

_PYPTO_ERROR_CODE_PREFIX = 0xF00000


def _to_pypto_error_code(error: ExternalError) -> int:
    """Convert a centrally allocated external error to its PyPTO error number."""
    return _PYPTO_ERROR_CODE_PREFIX | int(error)


class ErrorCode(IntEnum):
    """Parser error codes backed by the centrally allocated PyPTO codes."""

    INVALID_TYPE = _to_pypto_error_code(ExternalError.INVALID_TYPE)
    INVALID_VAL = _to_pypto_error_code(ExternalError.INVALID_VAL)
    NAME_ERROR = _to_pypto_error_code(ExternalError.NAME_ERROR)
    NOT_IMPLEMENTED_ERROR = _to_pypto_error_code(ExternalError.NOT_IMPLEMENTED_ERROR)
    INVALID_OPERATION = _to_pypto_error_code(ExternalError.INVALID_OPERATION)
    UNKNOWN = _to_pypto_error_code(ExternalError.UNKNOWN)


def get_error_code(exception_type: type) -> ErrorCode | None:
    """Get error code for exception type.

    Args:
        exception_type: Exception class

    Returns:
        Corresponding error code or None
    """

    error_code = getattr(exception_type, "error_code", None)
    return error_code if isinstance(error_code, ErrorCode) else None
