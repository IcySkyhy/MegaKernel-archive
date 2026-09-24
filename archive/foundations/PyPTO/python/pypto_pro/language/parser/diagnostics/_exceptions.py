# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser error exceptions with rich diagnostic information."""

from __future__ import annotations

__all__ = [
    "ParserError",
    "ParserSyntaxError",
    "ParserTypeError",
    "UndefinedVariableError",
    "SSAViolationError",
    "UnsupportedFeatureError",
    "InvalidOperationError",
    "ScopeIsolationError",
]


from pypto.error import PyptoError
from pypto.pypto_impl import ir

from ._error_codes import ErrorCode


def _normalize_span(span: ir.Span | None) -> dict[str, str | int | None] | None:
    """Convert an IR span to a plain dictionary with standard location fields."""
    if span is None:
        return None
    if isinstance(span, dict):
        return span

    filename = getattr(span, "filename", None)
    begin_line = getattr(span, "begin_line", 0)
    begin_column = getattr(span, "begin_column", 0)
    end_line = getattr(span, "end_line", 0)
    end_column = getattr(span, "end_column", 0)
    return {
        "filename": filename,
        "line": begin_line,
        "column": begin_column,
        "file": filename,
        "begin_line": begin_line,
        "begin_column": begin_column,
        "end_line": end_line,
        "end_column": end_column,
    }


class ParserError(PyptoError):
    """Base class for all parser errors with diagnostic information.

    This exception captures detailed context about parsing errors including
    source location, error message, and optional hints for fixing the error.
    """

    error_code = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str,
        span: ir.Span | None = None,
        hint: str | None = None,
        note: str | None = None,
        source_lines: list[str] | None = None,
    ):
        """Initialize parser error.

        Args:
            message: Error message describing what went wrong
            span: Source location where error occurred
            hint: Optional hint for how to fix the error
            note: Optional additional note about the error
            source_lines: Optional source code lines for context
        """
        super().__init__(int(self.error_code), message)
        self.message = message

        # Extract span information to avoid keeping C++ objects alive
        # This prevents memory leaks when exceptions are caught and held
        self.span = _normalize_span(span)

        self.hint = hint
        self.note = note
        self.source_lines = source_lines

    def __str__(self) -> str:
        parts = [super().__str__()]
        self._append_location(parts)
        if self.hint:
            parts.append(f"  hint: {self.hint}")
        if self.note:
            parts.append(f"  note: {self.note}")
        return "\n".join(parts)

    def _append_location(self, parts: list) -> None:
        if not self.span:
            return
        filename = self.span.get("filename") or "<unknown>"
        line = self.span.get("line") or self.span.get("begin_line")
        if not line:
            return
        column = self.span.get("column") or self.span.get("begin_column")
        parts.append(f"  --> {filename}:{line}:{column}")
        self._append_source_preview(parts, line, column)

    def _append_source_preview(self, parts: list, line: int, column: int) -> None:
        source_line = self._get_source_line(line)
        if not source_line:
            return
        end_column = self.span.get("end_column") or 0
        parts.append("   |")
        parts.append(f"{line:3} | {source_line}")
        if column and end_column and end_column > column:
            width = end_column - column
            parts.append(f"   | {' ' * (column - 1)}{'^' * width}")
        elif column:
            parts.append(f"   | {' ' * (column - 1)}^")

    def _get_source_line(self, line: int) -> str | None:
        if not self.source_lines:
            return None
        idx = line - 1
        if 0 <= idx < len(self.source_lines):
            return self.source_lines[idx].rstrip("\n")
        return None


class ParserSyntaxError(ParserError):
    """Raised when DSL syntax is violated."""

    error_code = ErrorCode.INVALID_VAL


class ParserTypeError(ParserError):
    """Raised when type annotation is incorrect or missing."""

    error_code = ErrorCode.INVALID_TYPE


class FinalRejectionError(ParserTypeError):
    """Raised when a rejection must not be retried as a Python expression.

    ``ExpressionParserMixin.parse_expression`` recovers from a failed IR build by
    re-evaluating the expression in Python, so a position that deliberately
    rejects an otherwise-valid value has to say the rejection is final. Raising a
    plain ``ParserTypeError`` there would let the retry succeed and the rejection
    disappear; raising this instead propagates it unchanged.

    Subclasses ``ParserTypeError`` so callers that catch the general type still
    catch it — only the parser's retry treats it specially.
    """

    error_code = ErrorCode.INVALID_TYPE


class UndefinedVariableError(ParserError):
    """Raised when referencing an undefined variable."""

    error_code = ErrorCode.NAME_ERROR


class SSAViolationError(ParserError):
    """Raised when SSA property is violated (variable redefinition)."""

    error_code = ErrorCode.INVALID_OPERATION

    def __init__(
        self,
        message: str,
        span: ir.Span | None = None,
        hint: str | None = None,
        note: str | None = None,
        source_lines: list[str] | None = None,
        previous_span: ir.Span | None = None,
    ):
        """Initialize SSA violation error.

        Args:
            message: Error message describing what went wrong
            span: Source location where error occurred
            hint: Optional hint for how to fix the error
            note: Optional additional note about the error
            source_lines: Optional source code lines for context
            previous_span: Optional previous definition location
        """
        super().__init__(message, span, hint, note, source_lines)

        self.previous_span = _normalize_span(previous_span)


class UnsupportedFeatureError(ParserError):
    """Raised when using an unsupported Python feature in DSL."""

    error_code = ErrorCode.NOT_IMPLEMENTED_ERROR


class InvalidOperationError(ParserError):
    """Raised when an operation is invalid or unknown."""

    error_code = ErrorCode.INVALID_OPERATION


class ScopeIsolationError(ParserError):
    """Raised when scope isolation is violated."""

    error_code = ErrorCode.INVALID_OPERATION
