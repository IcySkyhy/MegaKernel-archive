# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scope management and SSA verification for IR parsing."""

from __future__ import annotations

__all__ = ["ScopeManager", "SSAViolationError", "ScopeIsolationError"]


from typing import Any

from pypto.pypto_impl.ir import Expr
from pypto_pro.ir import Span

from .diagnostics import ParserTypeError, ScopeIsolationError, SSAViolationError


class ScopeManager:
    """Manages variable scopes and optionally enforces SSA properties."""

    def __init__(
        self,
        strict_ssa: bool = False,
        tile_mutex_meta: dict[Any, tuple[Any, Any]] | None = None,
    ):
        """Initialize scope manager.

        Args:
            strict_ssa: If True, enforce SSA (single assignment per variable).
                       If False (default), allow variable reassignment.
            tile_mutex_meta: Shared parser metadata mapping tile expressions to
                             their runtime mutex expression and candidate ids.
        """
        self.strict_ssa = strict_ssa
        self._tile_mutex_meta = tile_mutex_meta
        self.scopes: list[dict[str, Any]] = [{}]  # Stack of scope dictionaries
        self.assignments: dict[str, int] = {}  # Track assignment count per variable
        self.scope_types: list[str] = ["global"]  # Track type of each scope
        self.var_spans: list[dict[str, Span]] = [{}]  # Track span for each variable definition
        self.mask_reg_vars: set[str] = set()  # Variable names declared as MaskReg

    def enter_scope(self, scope_type: str) -> None:
        """Enter a new scope.

        Args:
            scope_type: Type of scope ('function', 'for', 'if')
        """
        self.scopes.append({})
        self.var_spans.append({})
        self.scope_types.append(scope_type)

    def exit_scope(self, leak_vars: bool = False) -> dict[str, Any]:
        """Exit current scope and return defined variables.

        Args:
            leak_vars: If True, copy all variables from exiting scope to parent scope.
                       Used for plain syntax where variables should be visible after for/if.

        Returns:
            Dictionary of variables defined in the exited scope
        """
        if len(self.scopes) <= 1:
            raise RuntimeError("Cannot exit global scope")

        scope_vars = self.scopes.pop()
        self.var_spans.pop()
        self.scope_types.pop()

        self._merge_tile_mutex_meta_to_outer(scope_vars, leak_vars)

        # Leak variables to parent scope if requested
        if leak_vars and self.scopes:
            parent_scope = self.scopes[-1]
            for name, value in scope_vars.items():
                parent_scope[name] = value

        return scope_vars

    def _merge_tile_mutex_meta_to_outer(self, scope_vars: dict[str, Any], leak_vars: bool) -> None:
        """Merge final inner mutex candidates into the nearest outer binding."""
        if self._tile_mutex_meta is None:
            return

        for name, inner_var in scope_vars.items():
            if not isinstance(inner_var, Expr):
                continue
            inner_meta = self._tile_mutex_meta.get(inner_var)
            if inner_meta is None:
                continue

            outer_var = self.lookup_var(name)
            if outer_var is None:
                continue

            outer_meta = self._tile_mutex_meta.get(outer_var)
            if outer_meta is None:
                continue

            inner_mutex_ids, inner_ids = inner_meta
            outer_mutex_ids, outer_ids = outer_meta
            outer_mutex_id_count = len(outer_mutex_ids)
            inner_mutex_id_count = len(inner_mutex_ids)
            if outer_mutex_id_count != inner_mutex_id_count:
                raise ParserTypeError(
                    f"cannot merge tile mutex metadata with different ID counts: "
                    f"{outer_mutex_id_count} and {inner_mutex_id_count}"
                )
            merged_ids = list(dict.fromkeys(list(outer_ids or ()) + list(inner_ids or ())))

            if leak_vars:
                self._tile_mutex_meta[inner_var] = (inner_mutex_ids, merged_ids)
            else:
                self._tile_mutex_meta[outer_var] = (outer_mutex_ids, merged_ids)

    def register_mask_reg_var(self, name: str) -> None:
        """Mark a variable as a MaskReg (mask register).

        Called when a variable is declared via vf.mask_reg, vf.create_mask,
        vf.update_mask, or any op whose dst is inferred to be MaskReg.
        Enables input-type-based dst inference for unified ops.
        """
        self.mask_reg_vars.add(name)

    def is_mask_reg_var(self, name: str) -> bool:
        """Return True if the variable was declared as a MaskReg."""
        return name in self.mask_reg_vars

    def define_var(self, name: str, value: Any, allow_redef: bool = False, span: Any | None = None) -> None:
        """Define a variable in the current scope.

        Args:
            name: Variable name
            value: Variable value/node
            allow_redef: If True, allow redefinition (for iter_args and parameters)
            span: Optional source location span for error reporting

        Raises:
            SSAViolationError: If strict_ssa=True and variable is already defined in current scope
        """
        current_scope = self.scopes[-1]
        current_spans = self.var_spans[-1]

        # Check SSA: variable should not already be defined in current scope
        if name in current_scope and not allow_redef:
            if self.strict_ssa:
                # Get the previous definition span
                previous_span = current_spans.get(name)

                raise SSAViolationError(
                    f"Variable '{name}' is already defined",
                    span=span,
                    previous_span=previous_span,
                    hint="Use a different variable name for each assignment (SSA form requires unique names)",
                    note="Each variable can only be assigned once per scope",
                )
            # In non-SSA mode: just update the variable (no error)

        current_scope[name] = value
        if span is not None:
            current_spans[name] = span

        # Track assignment count globally
        if name not in self.assignments:
            self.assignments[name] = 0
        self.assignments[name] += 1

    def lookup_var(self, name: str) -> Any | None:
        """Lookup variable in scope chain.

        Args:
            name: Variable name to look up

        Returns:
            Variable value/node if found, None otherwise
        """
        # Search from innermost to outermost scope
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_var_bounded(self, name: str, barrier: str = "inline") -> Any | None:
        """Lookup variable, stopping at the nearest scope of type ``barrier``.

        Used to prevent inline functions from capturing IR variables from the
        caller's scope. Variables defined within the inline scope (params,
        locals) are found; variables in outer scopes beyond the barrier are not.
        """
        for i in range(len(self.scopes) - 1, -1, -1):
            if name in self.scopes[i]:
                return self.scopes[i][name]
            if self.scope_types[i] == barrier:
                return None
        return None

    def is_defined(self, name: str) -> bool:
        """Check if variable is defined in any accessible scope.

        Args:
            name: Variable name

        Returns:
            True if variable is defined, False otherwise
        """
        return self.lookup_var(name) is not None

    def current_scope_type(self) -> str:
        """Get the type of the current scope.

        Returns:
            Scope type string ('global', 'function', 'for', 'if')
        """
        return self.scope_types[-1]

    def in_scope_type(self, scope_type: str) -> bool:
        """Check if currently in a specific scope type.

        Args:
            scope_type: Type to check for

        Returns:
            True if in the specified scope type
        """
        return scope_type in self.scope_types
