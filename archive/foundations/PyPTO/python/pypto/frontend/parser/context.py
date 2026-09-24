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

"""Variable table and context management for PTO Script Parser.

This module provides context management and variable scoping utilities for
parsing PTO scripts. It implements a stack-based variable table that tracks
variables across different scopes and blocks during parsing.

The main components are:
- VarTableFrame: Represents a single scope/block of variables
- VarTable: Stack of frames managing variable lifetime and shadowing
- Context managers for automatic frame cleanup
"""

import ast
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Optional

from pypto.error import FeError


class ContextFrame:
    """Represents a single scope frame for variable storage.

    Each frame maintains a collection of variable names that were introduced
    within a specific block or scope during parsing.
    """

    vars: set[str]

    def __init__(self):
        self.vars = set()

    def add(self, var: str, node: Optional[ast.AST] = None) -> None:
        """Register a new variable name in this context frame.

        Parameters
        ----------
        var : str
            The identifier of the variable to register.
        node : Optional[ast.AST]
            The AST node associated with this variable, used for error reporting.
        """
        if var in self.vars:
            if node is None:
                raise FeError(NameError(f"Variable '{var}' already exists in the current scope"))
            else:
                raise FeError(NameError(f"Variable '{var}' already exists in the current scope"))
        self.vars.add(var)


class Context:
    """Manages variable scoping and context during PTO script parsing.

    Maintains a stack of context frames, each representing a different scope,
    and tracks variable values across these scopes.

    Parameters
    ----------
    frames : list[ContextFrame]
        The stack of context frames representing nested scopes.
    """

    frames: list[ContextFrame]
    name2value: dict[str, list[Any]]
    marked_for_deletion: set[str]

    def __init__(self):
        self.frames = []
        self.name2value = defaultdict(list)
        self.marked_for_deletion = set()

    @contextmanager
    def with_frame(self):
        """Establish a new context frame that can be used with a with statement.

        Returns
        -------
        result : Iterator[None]
            A context manager that creates a new frame and automatically
            cleans it up when the context exits.
        """
        try:
            self.frames.append(ContextFrame())
            yield
        finally:
            frame = self.frames.pop()
            for var in frame.vars:
                self.name2value[var].pop()

    def add(
        self,
        var: str,
        value: Any,
        node: Optional[ast.AST] = None,
    ) -> None:
        """Register or update a variable in the current context frame.
        --------
        >>> ctx = Context()
        >>> with ctx.with_frame():
        ...     ctx.add("x", 1)  # Frame 0: x = 1
        ...     with ctx.with_frame():
        ...         ctx.add("x", 2)  # Updates Frame 0: x = 2
        ...     # After inner frame exits, x is still 2 in Frame 0
        """
        values = self.name2value[var]
        if values:
            values[-1] = value
        else:
            self.frames[-1].add(var, node)
            values.append(value)

    def get(self) -> dict[str, Any]:
        """Retrieve a dictionary containing the most recent value for each variable."""
        result = {}
        for var, values in self.name2value.items():
            if values:
                result[var] = values[-1]
        return result

    def delete(self, var: str) -> None:
        """Remove a variable from the topmost context frame."""
        frame = self.frames[-1]
        if var not in frame.vars:
            raise FeError(NameError(f"Variable '{var}' is not defined in the current scope"))

        frame.vars.discard(var)
        self.name2value[var].pop()

    def mark_for_deletion(self, var_names: set[str]) -> None:
        self.marked_for_deletion.update(var_names)

    def cleanup_marked(self) -> None:
        for var in self.marked_for_deletion:
            for frame in reversed(self.frames):
                if var in frame.vars:
                    frame.vars.discard(var)
                    self.name2value[var].pop()
                    break
        # Reset the marked set
        self.marked_for_deletion.clear()
