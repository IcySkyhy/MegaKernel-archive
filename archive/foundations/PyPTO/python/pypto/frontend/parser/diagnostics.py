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

"""PTO Script Parser Source and Diagnostics.

This module provides source code management and diagnostic reporting utilities
for the PTO Script Parser. It handles:

1. Source Code Management:
   - Reading and caching source code from functions, strings, or files
   - Tracking line numbers and source locations

2. Diagnostic Reporting:
   - Error, warning, and info message formatting
   - Pretty-printing of errors with source context
   - Color-coded terminal output for better readability

Key Classes:
    - Source: Represents source code with location information
    - Diagnostics: Manages diagnostic messages and error reporting
    - DiagnosticLevel: Enumeration of diagnostic severity levels

The diagnostic system provides rich error messages with source context,
making it easier for users to identify and fix issues in their PTO scripts.
"""

import ast
import enum
import inspect
import typing
from typing import Callable, NoReturn, Union

from pypto.error import RenderedParserError

PRIOR_CONTEXT_LINES = 2
SUBSEQUENT_CONTEXT_LINES = 4


class DiagnosticLevel(enum.IntEnum):
    """Severity levels for diagnostic messages, corresponding to diagnostic.h definitions."""

    BUG = 10
    ERROR = 20
    WARNING = 30
    INFO = 40
    DEBUG = 50


class Source:
    def __init__(self, program: Callable):
        self.source_name = inspect.getfile(program)  # type: ignore
        source_lines, start_line = inspect.getsourcelines(program)  # type: ignore
        indent_lines = "if True:\n " + " ".join(source_lines)
        mod = ast.parse(indent_lines)
        if_stmt = typing.cast(ast.If, mod.body[0])
        self.tree = typing.cast(ast.FunctionDef, if_stmt.body[0])
        self._fix_line_numbers(self.tree, start_line)

    @staticmethod
    def _fix_line_numbers(tree: ast.AST, lineno: int) -> int:
        for node in ast.walk(tree):
            if hasattr(node, "lineno"):
                node.lineno += lineno - 2
                node.end_lineno += lineno - 2
                node.col_offset -= 1
                node.end_col_offset -= 1

    def as_ast(self) -> ast.FunctionDef:
        return self.tree


class Span:
    """Represents a source code location span for diagnostic messages."""

    source_name: str
    line: int
    end_line: int
    column: int
    end_column: int

    def __init__(
        self,
        source_name: str,
        lineno: int,
        end_lineno: int,
        col_offset: int,
        end_col_offset: int,
    ):
        self.source_name = source_name
        self.line = lineno
        self.end_line = end_lineno
        self.column = col_offset
        self.end_column = end_col_offset


class DiagnosticItem:
    """Represents a single diagnostic message item.

    Parameters
    ----------
    level : DiagnosticLevel
        The severity level of this diagnostic message.
    """

    level: DiagnosticLevel
    span: Span
    message: str

    def __init__(self, level: DiagnosticLevel, span: Span, message: str):
        self.level = level
        self.span = span
        self.message = message

    def render_to_console(self) -> None:
        """Output the diagnostic item to the console with formatting."""
        # ANSI escape sequences for terminal color coding by severity
        color_map = {
            DiagnosticLevel.BUG: "\033[95m",  # Magenta
            DiagnosticLevel.ERROR: "\033[91m",  # Red
            DiagnosticLevel.WARNING: "\033[93m",  # Yellow
            DiagnosticLevel.INFO: "\033[94m",  # Blue
            DiagnosticLevel.DEBUG: "\033[92m",  # Green
        }
        level_labels = {
            DiagnosticLevel.BUG: "INTERNAL BUG",
            DiagnosticLevel.ERROR: "ERROR",
            DiagnosticLevel.WARNING: "WARNING",
            DiagnosticLevel.INFO: "INFO",
            DiagnosticLevel.DEBUG: "DEBUG",
        }
        color_reset = "\033[0m"
        bold_style = "\033[1m"

        selected_color = color_map.get(self.level, "")
        severity_label = level_labels.get(self.level, "")

        # Output formatted header with color coding
        header_parts = [
            bold_style,
            selected_color,
            severity_label,
            color_reset,
            " ",
            bold_style,
            self.span.source_name,
            ":",
            str(self.span.line),
            ":",
            str(self.span.column),
            ":",
            color_reset,
            " ",
            self.message,
        ]
        print("".join(header_parts))

        # Load and present source code context
        with open(self.span.source_name, "r", encoding="utf-8") as source_file:
            file_lines = source_file.readlines()

        preceding_context = PRIOR_CONTEXT_LINES
        following_context = SUBSEQUENT_CONTEXT_LINES
        context_start = max(0, self.span.line - 1 - preceding_context)
        context_end = min(len(file_lines), self.span.end_line + following_context)

        # Determine padding width for line number display
        max_line_digits = len(str(context_end))

        # Display surrounding source lines
        for line_idx in range(context_start, context_end):
            current_line_num = line_idx + 1
            line_text = file_lines[line_idx].rstrip("\n")

            # Determine if this line is part of the error span
            is_span_line = self.span.line <= current_line_num <= self.span.end_line

            if is_span_line:
                formatted_line = (
                    selected_color + str(current_line_num).rjust(max_line_digits) + " |" + color_reset + " " + line_text
                )
                print(formatted_line)
            else:
                line_display = str(current_line_num).rjust(max_line_digits) + " | " + line_text
                print(line_display)

            # Add visual indicator for error span lines
            if is_span_line:
                # Compute indentation for caret positioning
                if current_line_num == self.span.line:
                    caret_start = self.span.column - 1
                else:
                    caret_start = 0

                if current_line_num == self.span.end_line:
                    caret_end = self.span.end_column - 1
                else:
                    caret_end = len(line_text)

                # Output caret indicator line
                padding_spaces = " " * (max_line_digits + 3 + caret_start)
                caret_chars = "^" * max(1, caret_end - caret_start)
                caret_line = selected_color + padding_spaces + caret_chars + color_reset
                print(caret_line)

        print()  # Blank line for visual separation


class DiagnosticContext:
    """Container for managing diagnostic messages within a source context.

    Parameters
    ----------
    source : Source
        The source code object associated with these diagnostics.
    """

    source: Source
    diagnostics: list[DiagnosticItem]

    def __init__(self, source: Source):
        self.source = source
        self.diagnostics = []

    def emit(self, diagnostic: DiagnosticItem) -> None:
        """Add a diagnostic message to the collection.

        Parameters
        ----------
        diagnostic : DiagnosticItem
            The diagnostic message to add.
        """
        self.diagnostics.append(diagnostic)

    def render(self) -> None:
        """Display all collected diagnostics to the console and clear the list."""
        for diagnostic_item in self.diagnostics:
            diagnostic_item.render_to_console()
        self.diagnostics.clear()


class Diagnostics:
    """Manages diagnostic message generation and reporting for the parser.

    Parameters
    ----------
    source : Source
        The source code being analyzed.

    ctx : DiagnosticContext
        The diagnostic context container for managing messages.
    """

    source: Source
    context: DiagnosticContext

    def __init__(self, source: Source):
        self.source = source
        self.context = DiagnosticContext(source)

    def __del__(self) -> None:
        """Output all diagnostics and perform cleanup."""
        self._render()

    def emit(self, node: ast.AST, message: str, level: DiagnosticLevel = DiagnosticLevel.INFO) -> None:
        """Generate and record a diagnostic message.

        Parameters
        ----------
        node : ast.AST
            The AST node containing location information for the diagnostic.

        message : str
            The diagnostic message text.

        level : DiagnosticLevel
            The severity level of the diagnostic.
        """
        line_number = getattr(node, "lineno", 1)
        column_position = getattr(node, "col_offset", 0)
        ending_line = getattr(node, "end_lineno", line_number)
        ending_column = getattr(node, "end_col_offset", column_position)
        self.context.emit(
            DiagnosticItem(
                level=level,
                span=Span(
                    source_name=self.source.source_name,
                    lineno=line_number,
                    end_lineno=ending_line,
                    col_offset=column_position,
                    end_col_offset=ending_column,
                ),
                message=str(message),
            )
        )

    def bug(self, node: ast.AST, message: Union[str, Exception]) -> NoReturn:
        """Generate a bug-level diagnostic and raise an exception.

        Parameters
        ----------
        node : ast.AST
            The AST node associated with the bug.

        message : str
            The bug description message.

        Raises
        ------
        RenderedParserError
            Always raises RenderedParserError after displaying the diagnostic.
        """
        self.emit(node, str(message), DiagnosticLevel.BUG)
        self._render()
        raise RenderedParserError(node, message) from None

    def error(self, node: ast.AST, message: Union[str, Exception]) -> NoReturn:
        """Generate an error-level diagnostic and raise an exception.

        Parameters
        ----------
        node : ast.AST
            The AST node associated with the error.

        message : str
            The error description message.

        Raises
        ------
        RenderedParserError
            Always raises RenderedParserError after displaying the diagnostic.
        """
        self.emit(node, message, DiagnosticLevel.ERROR)
        self._render()
        raise RenderedParserError(node, message) from None

    def warning(self, node: ast.AST, message: str) -> None:
        """Generate a warning-level diagnostic message.

        Parameters
        ----------
        node : ast.AST
            The AST node associated with the warning.
        """
        self.emit(node, message, DiagnosticLevel.WARNING)

    def info(self, node: ast.AST, message: str) -> None:
        """Generate an info-level diagnostic message.

        Parameters
        ----------
        node : ast.AST
            The AST node associated with the informational message.

        message : str
            The informational message text.
        """
        self.emit(node, message, DiagnosticLevel.INFO)

    def debug(self, node: ast.AST, message: str) -> None:
        """Generate a debug-level diagnostic message.

        Parameters
        ----------
        node : ast.AST
            The AST node associated with the debug message.

        message : str
            The debug message text.
        """
        self.emit(node, message, DiagnosticLevel.DEBUG)

    def _render(self) -> None:
        """Output all diagnostics to the console."""
        self.context.render()
