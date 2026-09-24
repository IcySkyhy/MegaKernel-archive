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
__all__ = [
    "log_debug",
    "log_info",
    "log_warn",
    "log_error",
    "log_fatal",
    "log_event",
    "check",
    "internal_check",
    "raise_error",
    "set_log_level",
    "get_log_level",
    "InternalError",
    "LogLevel",
]

from typing import NoReturn, TypeVar

from . import pypto_impl
from .pypto_impl import InternalError, LogLevel

T = TypeVar("T")

_log_level = pypto_impl.get_log_level()


def log_debug(message: str) -> None:
    """Log a debug message."""
    if _log_level <= LogLevel.DEBUG:
        pypto_impl.log(LogLevel.DEBUG, message)


def log_info(message: str) -> None:
    """Log an info message."""
    if _log_level <= LogLevel.INFO:
        pypto_impl.log(LogLevel.INFO, message)


def log_warn(message: str) -> None:
    """Log a warning message."""
    if _log_level <= LogLevel.WARN:
        pypto_impl.log(LogLevel.WARN, message)


def log_error(message: str) -> None:
    """Log an error message."""
    if _log_level <= LogLevel.ERROR:
        pypto_impl.log(LogLevel.ERROR, message)


def log_fatal(message: str) -> None:
    """Log a fatal message."""
    if _log_level <= LogLevel.FATAL:
        pypto_impl.log(LogLevel.FATAL, message)


def log_event(message: str) -> None:
    """Log an event message."""
    if _log_level <= LogLevel.EVENT:
        pypto_impl.log(LogLevel.EVENT, message)


def check(condition: bool, message: str) -> None:
    """Check a condition and throw ValueError if it fails."""
    if not condition:
        pypto_impl.raise_error("ValueError", message)


def internal_check(condition: bool, message: str) -> None:
    """Check a condition and throw ValueError if it fails."""
    if not condition:
        pypto_impl.raise_error("InternalError", message)


def raise_error(error_type: str, message: str) -> NoReturn:
    """Raise an error from C++ for testing error handling."""
    pypto_impl.raise_error(error_type, message)


def value_error(message: str) -> NoReturn:
    """Raise a ValueError from C++ for testing error handling."""
    raise_error("ValueError", message)


def internal_error(message: str) -> NoReturn:
    """Raise an InternalError from C++ for testing error handling."""
    raise_error("InternalError", message)


def set_log_level(level: LogLevel) -> None:
    """Set the log level for PyPTO IR."""
    global _log_level
    _log_level = level
    pypto_impl.set_log_level(level)


def get_log_level() -> LogLevel:
    """Get the current log level for PyPTO IR."""
    return _log_level
