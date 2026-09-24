#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Error classes and exception handling for PyPTO.

This module defines custom exception classes and error handling utilities
for the PyPTO framework. It provides a consistent error interface with
error codes for different error categories.

Classes:
    PyptoError: Base exception class for all PyPTO errors.
    ParserError: Parser exception with AST node association and error codes.
    RenderedParserError: A special error class indicating that the error.

Decorators:
    _catch_and_wrap_error: Decorator to catch exceptions and wrap them with context.
"""

import ast
from functools import wraps
from typing import Callable, Optional, Union

from . import pypto_impl


def _to_py_error_code(err_code) -> int:
    return 0xF00000 | int(err_code)


_ERROR_CODE_UNKNOWN = _to_py_error_code(pypto_impl.ExternalError.UNKNOWN)

_error_mapping = {
    TypeError: _to_py_error_code(pypto_impl.ExternalError.INVALID_TYPE),
    ValueError: _to_py_error_code(pypto_impl.ExternalError.INVALID_VAL),
    RuntimeError: _to_py_error_code(pypto_impl.ExternalError.RUNTIME_ERROR),
    NameError: _to_py_error_code(pypto_impl.ExternalError.NAME_ERROR),
    NotImplementedError: _to_py_error_code(pypto_impl.ExternalError.NOT_IMPLEMENTED_ERROR),
    KeyError: _to_py_error_code(pypto_impl.ExternalError.KEY_ERROR),
    IndexError: _to_py_error_code(pypto_impl.ExternalError.OUT_OF_RANGE),
}


def _get_err_code(msg: Exception) -> int:
    """Get error code based on exception type."""
    return _error_mapping.get(type(msg), _ERROR_CODE_UNKNOWN)


class PyptoError(Exception):
    """Base exception class for all PyPTO errors.

    Args:
        msg: Exception object or error message string.
        err_code: Error code (int).
    """

    def __init__(self, err_code: int, msg: Union[str, Exception]):
        if isinstance(msg, Exception):
            err_str = str(msg)
            if "ErrCode" in err_str:
                msg = err_str
            else:
                msg = f"ErrCode: {err_code:X}, {type(msg).__name__}: {err_str}"
        elif "ErrCode" not in msg:
            msg = f"ErrCode: {err_code:X}, {msg}"
        super().__init__(msg)
        self.node: Optional[ast.AST] = None


class PyptoGeneralError(PyptoError):
    """General exception class with automatic error code derivation.

    Args:
        msg: Exception object or error message string.
    """

    def __init__(self, msg: Union[str, Exception]):
        err_code = _get_err_code(msg) if isinstance(msg, Exception) else _ERROR_CODE_UNKNOWN
        super().__init__(err_code, msg)


class ParserError(PyptoError):
    """ParserError class for diagnostics.

    Args:
        node: AST node where the error occurred, used for source location
              reporting and error message formatting.
        msg: Exception object or error message string.

    Attributes:
        node: The AST node where the error occurred.
    """

    def __init__(self, node: ast.AST, msg: Union[str, Exception]):
        err_code = _get_err_code(msg) if isinstance(msg, Exception) else _ERROR_CODE_UNKNOWN
        super().__init__(err_code, msg)
        self.node = node


class RenderedParserError(ParserError):
    """Error class for diagnostics with rendered message."""


class FeError(PyptoGeneralError):
    """Error class for frontend errors."""


class PyptoRtError(PyptoGeneralError):
    """Runtime error class for all runtime errors."""


class PassError(PyptoGeneralError):
    """Error class for Pass management and configuration errors."""


class VerifyError(PyptoGeneralError):
    """Error class for errors when enabling pass verify."""


def _catch_and_wrap_error(operation_name: str) -> Callable:
    """Decorator to catch exceptions and wrap them into PyptoGeneralError with context.

    This decorator wraps methods to provide consistent error handling:
    - If the exception is already a PyptoError (or its subclass), it's raised directly (avoid double-wrapping)
    - Otherwise, the exception is wrapped into PyptoGeneralError with context information

    Parameters
    ----------
    operation_name : str
        Description of the operation being performed, used in error message.

    Returns
    -------
    Callable
        Decorated function with unified exception handling.

    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, PyptoError):
                    raise
                func_name = getattr(args[0], '__name__', func.__name__) if args else func.__name__
                err_msg = str(e)
                if "ErrCode" in err_msg:
                    err = PyptoGeneralError(f"Failed to {operation_name} '{func_name}'.\n{err_msg}")
                else:
                    err = PyptoGeneralError(e.__class__(f"Failed to {operation_name} '{func_name}'.\n{err_msg}"))
                raise err.with_traceback(e.__traceback__) from None

        return wrapper

    return decorator
