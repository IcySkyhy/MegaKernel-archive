#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scope-based liveness analysis for automatic variable deletion.

This module implements scope-aware liveness analysis for the PTO Script Parser.
It analyzes the abstract syntax tree (AST) to determine when variables are last
used and can be safely deleted, with proper handling of scope boundaries
and scope lifting for cross-scope variable usage.

Unified Deletion Rules:
    Rule A: Variable deletion point should be at the same scope level as definition.
            Delete after last use within that scope.
    Rule B: If use scope and definition scope have no nesting relationship,
            lift definition scope to common ancestor, then apply Rule A.

Note:
    This analyzer records ALL variables regardless of type. The actual tensor
    type filtering is handled by parser.py's _filter_tensor_vars() method
    during execution, which uses runtime type information (isinstance checks).
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pypto.error import FeError


@dataclass
class Scope:
    """Represents a code block scope in the AST.

    A scope corresponds to a code block such as:
    - Function body (root scope)
    - For loop body
    - If body / Else body

    Each scope maintains information about:
    - Variables defined within it
    - Entry and exit statement IDs for deletion timing
    - Parent-child relationships forming a scope tree
    """

    scope_id: int
    scope_type: str
    parent: Optional['Scope'] = None
    children: List['Scope'] = field(default_factory=list)
    tensor_defs: Set[str] = field(default_factory=set)
    exit_stmt_id: Optional[int] = None
    entry_stmt_id: Optional[int] = None
    stmt_ids: List[int] = field(default_factory=list)

    @staticmethod
    def find_common_ancestor_of_multiple(scopes: List['Scope']) -> Optional['Scope']:
        """Find the common ancestor of multiple scopes.

        This represents the outermost definition scope level across all
        branches (if/else) or nested redefinitions.
        """
        if not scopes:
            return None

        if len(scopes) == 1:
            return scopes[0]

        ancestor = scopes[0]
        for scope in scopes[1:]:
            ancestor = ancestor.find_common_ancestor(scope)
            if not ancestor:
                return None

        return ancestor

    def is_nested_in(self, other: 'Scope') -> bool:
        """Check if this scope is nested within another scope."""
        current = self.parent
        while current:
            if current.scope_id == other.scope_id:
                return True
            current = current.parent
        return False

    def find_common_ancestor(self, other: 'Scope') -> Optional['Scope']:
        """Find the common ancestor scope of two scopes."""
        ancestors: Dict[int, Scope] = {}
        current = self
        while current:
            ancestors[current.scope_id] = current
            current = current.parent

        current = other
        while current:
            if current.scope_id in ancestors:
                return ancestors[current.scope_id]
            current = current.parent
        return None

    def depth(self) -> int:
        """Calculate the depth of this scope in the scope tree."""
        d = 0
        current = self.parent
        while current:
            d += 1
            current = current.parent
        return d


@dataclass
class VarInfo:
    """Information about a variable for liveness analysis."""

    var_name: str
    def_scope_id: int
    def_stmt_id: int
    def_scope_ids: Set[int] = field(default_factory=set)
    use_points: List[Tuple[int, int]] = field(default_factory=list)
    needs_scope_lift: bool = False
    lift_target_scope_id: Optional[int] = None
    delete_after_stmt_id: Optional[int] = None
    delete_after_scope_exit: bool = False
    delete_scope_id: Optional[int] = None


@dataclass
class LivenessResult:
    """Structured result from scope-based liveness analysis."""

    var_info: Dict[str, VarInfo] = field(default_factory=dict)
    scope_lift_suggestions: List[Tuple[str, int, int]] = field(default_factory=list)
    delete_points: Dict[int, Set[str]] = field(default_factory=dict)
    scope_tree: Optional[Scope] = None
    scope_map: Dict[int, Scope] = field(default_factory=dict)


class ScopeLivenessAnalyzer(ast.NodeVisitor):
    """Scope-based liveness analyzer for variables.

    This analyzer builds a scope tree from the AST, collects variable
    definitions and uses, and applies unified deletion rules.

    Type filtering is handled by parser.py during execution using
    _filter_tensor_vars() which checks isinstance(value, pypto.Tensor).
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.result: LivenessResult = LivenessResult()
        self.exempt_vars: Set[str] = set()
        self.scope_stack: List[Scope] = []
        self.current_scope: Optional[Scope] = None
        self.scope_counter: int = 0
        self.current_stmt_id: Optional[int] = None

    def analyze(self, node: ast.AST, exempt_vars: Optional[Set[str]] = None) -> LivenessResult:
        """Analyze the AST and return structured liveness result.

        Parameters
        ----------
        node : ast.AST
            The AST node to analyze (typically FunctionDef).
        exempt_vars : Optional[Set[str]]
            Variables that should not be auto-deleted (e.g., function arguments).

        Returns
        -------
        LivenessResult
            Structured analysis result containing deletion points,
            scope lifting suggestions, and detailed variable info.
        """
        if exempt_vars:
            self.exempt_vars = set(exempt_vars)

        self.visit(node)
        self._apply_deletion_rules()

        return self.result

    # -------------------------------------------------------------------------
    # Statement Visitors
    # -------------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition - creates root scope."""
        for arg in node.args.args:
            self.exempt_vars.add(arg.arg)

        root_scope = self._create_scope(scope_type='root', entry_stmt_id=self._get_node_id(node))
        self.result.scope_tree = root_scope
        self._push_scope(root_scope)

        for stmt in node.body:
            self.visit(stmt)

        root_scope.exit_stmt_id = self._get_node_id(node)
        self._pop_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        self.visit(node.value)

        for target in node.targets:
            self._visit_assign_target(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignment."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        if node.value:
            self.visit(node.value)

        self._visit_assign_target(node.target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        self.visit(node.value)
        self.visit(node.target)

    def visit_For(self, node: ast.For) -> None:
        """Visit for loop - creates loop scope."""
        stmt_id = self._get_node_id(node)
        self.current_stmt_id = stmt_id

        if self.current_scope:
            self.current_scope.stmt_ids.append(stmt_id)

        self.visit(node.iter)

        loop_scope = self._create_scope(scope_type='for', entry_stmt_id=stmt_id, exit_stmt_id=stmt_id)
        self._push_scope(loop_scope)

        if isinstance(node.target, ast.Name):
            loop_var = node.target.id
            self._record_var_def(loop_var, stmt_id)
            self.exempt_vars.add(loop_var)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self._record_var_def(elt.id, stmt_id)
                    self.exempt_vars.add(elt.id)

        for stmt in node.body:
            self.visit(stmt)

        self._pop_scope()

    def visit_While(self, node: ast.While) -> None:
        """Visit while loop - not supported."""
        raise FeError(NotImplementedError("While loop is not supported yet."))

    def visit_If(self, node: ast.If) -> None:
        """Visit if statement - creates if scope with body/else sub-scopes."""
        stmt_id = self._get_node_id(node)
        self.current_stmt_id = stmt_id

        if self.current_scope:
            self.current_scope.stmt_ids.append(stmt_id)

        self.visit(node.test)

        if_scope = self._create_scope(scope_type='if', entry_stmt_id=stmt_id, exit_stmt_id=stmt_id)
        self._push_scope(if_scope)

        if_body_scope = self._create_scope(scope_type='if_body')
        self._push_scope(if_body_scope)

        for stmt in node.body:
            self.visit(stmt)

        self._pop_scope()

        if node.orelse:
            else_body_scope = self._create_scope(scope_type='else_body')
            self._push_scope(else_body_scope)

            for stmt in node.orelse:
                self.visit(stmt)

            self._pop_scope()

        self._pop_scope()

    def visit_Return(self, node: ast.Return) -> None:
        """Visit return statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        if node.value:
            self.visit(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Visit explicit delete statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.exempt_vars.add(target.id)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit expression statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        self.visit(node.value)

    def visit_Pass(self, node: ast.Pass) -> None:
        """Visit pass statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

    # -------------------------------------------------------------------------
    # Expression Visitors
    # -------------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        """Visit name node - record variable uses."""
        if isinstance(node.ctx, ast.Load):
            self._record_var_use(node.id, self.current_stmt_id)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access."""
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit subscript access."""
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit binary operation."""
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Visit unary operation."""
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Visit comparison."""
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """Visit tuple."""
        for elt in node.elts:
            self.visit(elt)

    def visit_List(self, node: ast.List) -> None:
        """Visit list."""
        for elt in node.elts:
            self.visit(elt)

    def visit_Slice(self, node: ast.Slice) -> None:
        """Visit slice."""
        if node.lower:
            self.visit(node.lower)
        if node.upper:
            self.visit(node.upper)
        if node.step:
            self.visit(node.step)

    # -------------------------------------------------------------------------
    # Deletion Rule Application (Unified)
    # -------------------------------------------------------------------------

    def _apply_deletion_rules(self) -> None:
        """Apply unified deletion rules to compute deletion points.

        Algorithm:
        1. Check Rule B - if non-nested usage exists, perform scope lifting
        2. Map all use points to def_scope's stmt_ids:
           - Use in def_scope → use stmt_id directly
           - Use in nested scope → climb to direct child scope, use its entry_stmt_id
        3. Find last stmt_id with use in def_scope.stmt_ids list (execution order)
        4. Set deletion point after that stmt_id
        """
        for var_name, var_info in self.result.var_info.items():
            if var_name in self.exempt_vars:
                continue

            if not var_info.use_points:
                self.result.delete_points.setdefault(var_info.def_stmt_id, set()).add(var_name)
                continue

            self._compute_delete_point_unified(var_info)

        self._build_delete_points_from_var_info()

    def _compute_effective_def_scope_from_all_scopes(self, var_info: VarInfo) -> Optional[Scope]:
        """Compute effective def_scope as common ancestor of all def and use scopes.

        This approach naturally finds the outermost boundary where the variable
        is active (defined or used), ensuring deletion happens at the correct level.
        """
        # Collect all scope IDs (definition + use)
        all_scope_ids = set(var_info.def_scope_ids)
        for scope_id, _ in var_info.use_points:
            all_scope_ids.add(scope_id)

        # Get scope objects
        scopes = [self.result.scope_map.get(sid) for sid in all_scope_ids]
        scopes = [s for s in scopes if s]

        # Find common ancestor of all scopes
        return Scope.find_common_ancestor_of_multiple(scopes)

    def _check_independent_definitions(self, var_info: VarInfo) -> bool:
        """Check if all definitions are in independent (mutually exclusive) scopes.

        Independent means:
        1. Multiple definition scopes (defined in different scopes)
        2. Each def scope has corresponding use in the same scope
        3. No uses outside the def scopes (uses must be contained within def scopes or their nested scopes)
        4. All def scopes are in different execution paths (no nesting, potentially brother or cousin scopes)

        Example: o_final defined in scope 6 and scope 10, which are in different
        if/else branches (mutually exclusive execution paths).
        """
        if len(var_info.def_scope_ids) <= 1:
            return False

        # Condition 2: Each def scope has use in the same scope
        for def_scope_id in var_info.def_scope_ids:
            uses_in_this_scope = [stmt_id for sid, stmt_id in var_info.use_points if sid == def_scope_id]
            if not uses_in_this_scope:
                return False

        # Condition 3: Check for uses outside def scopes
        if self._has_uses_outside_def_scopes(var_info):
            return False

        # Condition 4: Check if def scopes have nesting relationship
        if self._has_def_scopes_nesting(var_info):
            return False

        return True

    def _has_uses_outside_def_scopes(self, var_info: VarInfo) -> bool:
        """Check if there are uses outside def scopes (not nested in any def scope).

        Returns True if has outside uses, False otherwise.
        """
        for scope_id, _ in var_info.use_points:
            if scope_id in var_info.def_scope_ids:
                continue

            use_scope = self.result.scope_map.get(scope_id)
            if not use_scope:
                continue

            # Check if this use_scope is nested in ANY def_scope
            if not self._is_use_nested_in_any_def_scope(use_scope, var_info.def_scope_ids):
                # Found outside use
                return True

        return False

    def _is_use_nested_in_any_def_scope(self, use_scope: Scope, def_scope_ids: Set[int]) -> bool:
        """Check if use_scope is nested in any def scope.

        Returns True if nested, False otherwise.
        """
        for def_id in def_scope_ids:
            def_scope = self.result.scope_map.get(def_id)
            if def_scope and use_scope.is_nested_in(def_scope):
                return True
        return False

    def _has_def_scopes_nesting(self, var_info: VarInfo) -> bool:
        """Check if def scopes have nesting relationship.

        Returns True if has nesting, False otherwise.
        """
        def_scopes = [self.result.scope_map.get(sid) for sid in var_info.def_scope_ids]
        def_scopes = [s for s in def_scopes if s]

        for i, scope_i in enumerate(def_scopes):
            for scope_j in def_scopes[i + 1:]:
                if scope_i.is_nested_in(scope_j) or scope_j.is_nested_in(scope_i):
                    return True
        return False

    def _find_last_use_in_scope(self, def_scope: Scope, uses_in_scope: list) -> Optional[int]:
        """Find last use statement ID within a scope."""
        if not def_scope.stmt_ids:
            return max(uses_in_scope) if uses_in_scope else None

        last_use_stmt_id = None
        for stmt_id in def_scope.stmt_ids:
            if stmt_id in uses_in_scope:
                last_use_stmt_id = stmt_id

        return last_use_stmt_id if last_use_stmt_id else max(uses_in_scope)

    def _select_deepest_delete_point(self, delete_points_list: list) -> Optional[tuple]:
        """Select delete point from deepest scope."""
        max_depth = -1
        selected_point = None

        for stmt_id, scope_id, scope_exit in delete_points_list:
            scope = self.result.scope_map.get(scope_id)
            if scope and scope.depth() > max_depth:
                max_depth = scope.depth()
                selected_point = (stmt_id, scope_id, scope_exit)

        return selected_point

    def _set_delete_point_for_independent_defs(self, var_info: VarInfo) -> None:
        """Set deletion point for independent definitions.

        For independent definitions in mutually exclusive scopes (e.g., if/else branches),
        set a single unified deletion point at the common ancestor scope exit.

        For for loop scopes, use conservative deletion strategy with no explicit deletion.
        """
        if self._all_def_scopes_are_for_loops(var_info):
            return

        def_scopes = self._get_valid_def_scopes(var_info)
        if not def_scopes:
            return

        if self._handle_brother_scope_deletion(var_info, def_scopes):
            return

        delete_points_list = self._build_delete_points_for_each_def_scope(var_info)
        if not delete_points_list:
            return

        self._finalize_delete_points_selection(var_info, delete_points_list)

    def _all_def_scopes_are_for_loops(self, var_info: VarInfo) -> bool:
        """Check if all definition scopes are for loop scopes."""
        return all(
            (scope := self.result.scope_map.get(sid)) and scope.scope_type == 'for' for sid in var_info.def_scope_ids
        )

    def _get_valid_def_scopes(self, var_info: VarInfo) -> List[Scope]:
        """Get list of valid scope objects from def_scope_ids."""
        def_scopes = [self.result.scope_map.get(sid) for sid in var_info.def_scope_ids]
        return [s for s in def_scopes if s]

    def _handle_brother_scope_deletion(self, var_info: VarInfo, def_scopes: List[Scope]) -> bool:
        """Handle deletion for brother scopes (mutually exclusive execution paths).

        Returns True if handled, False otherwise.
        """
        if not self._are_scopes_brothers(def_scopes):
            return False

        common_ancestor = Scope.find_common_ancestor_of_multiple(def_scopes)
        if not common_ancestor:
            return False

        var_info.delete_after_stmt_id = common_ancestor.exit_stmt_id
        var_info.delete_scope_id = common_ancestor.scope_id
        var_info.delete_after_scope_exit = True

        if common_ancestor.exit_stmt_id:
            self.result.delete_points.setdefault(common_ancestor.exit_stmt_id, set()).add(var_info.var_name)

        return True

    def _build_delete_points_for_each_def_scope(self, var_info: VarInfo) -> list:
        """Build delete point candidates for each definition scope."""
        delete_points_list = []

        for def_scope_id in var_info.def_scope_ids:
            def_scope = self.result.scope_map.get(def_scope_id)
            if not def_scope:
                continue

            if def_scope.scope_type == 'for':
                delete_point = self._get_for_loop_delete_point(def_scope, def_scope_id)
                if delete_point:
                    delete_points_list.append(delete_point)
            else:
                delete_point = self._get_normal_scope_delete_point(var_info, def_scope, def_scope_id)
                delete_points_list.append(delete_point)

        return delete_points_list

    def _get_for_loop_delete_point(self, def_scope: Scope, def_scope_id: int) -> Optional[tuple]:
        """Get delete point for for loop scope."""
        if def_scope.exit_stmt_id:
            return (def_scope.exit_stmt_id, def_scope_id, True)
        elif def_scope.entry_stmt_id:
            return (def_scope.entry_stmt_id, def_scope_id, True)
        return None

    def _get_normal_scope_delete_point(self, var_info: VarInfo, def_scope: Scope, def_scope_id: int) -> tuple:
        """Get delete point for normal scope (non-for-loop)."""
        uses_in_scope = [stmt_id for sid, stmt_id in var_info.use_points if sid == def_scope_id]

        if uses_in_scope:
            last_use = self._find_last_use_in_scope(def_scope, uses_in_scope)
            stmt_id = last_use if last_use else var_info.def_stmt_id
        else:
            stmt_id = var_info.def_stmt_id

        return (stmt_id, def_scope_id, False)

    def _finalize_delete_points_selection(self, var_info: VarInfo, delete_points_list: list) -> None:
        """Select and apply the deepest delete point."""
        selected_point = self._select_deepest_delete_point(delete_points_list)
        if selected_point:
            var_info.delete_after_stmt_id = selected_point[0]
            var_info.delete_scope_id = selected_point[1]
            var_info.delete_after_scope_exit = selected_point[2]

        for stmt_id, _, _ in delete_points_list:
            self.result.delete_points.setdefault(stmt_id, set()).add(var_info.var_name)

    def _are_scopes_brothers(self, scopes: List[Scope]) -> bool:
        """Check if all scopes are brothers (mutually exclusive, same parent).

        Returns True if all scopes share the same parent and are not nested
        within each other, False otherwise.
        """
        if len(scopes) <= 1:
            return False

        parents = [s.parent for s in scopes if s.parent]
        if not parents or len(parents) != len(scopes):
            return False

        first_parent = parents[0]
        if not all(p.scope_id == first_parent.scope_id for p in parents):
            return False

        for i, scope_i in enumerate(scopes):
            for scope_j in scopes[i + 1:]:
                if scope_i.is_nested_in(scope_j) or scope_j.is_nested_in(scope_i):
                    return False

        return True

    def _compute_delete_point_unified(self, var_info: VarInfo) -> None:
        """Compute deletion point using unified rules."""
        if self._check_independent_definitions(var_info):
            self._set_delete_point_for_independent_defs(var_info)
            return

        effective_def_scope = self._compute_effective_def_scope_from_all_scopes(var_info)

        if not effective_def_scope:
            effective_def_scope = self.result.scope_map.get(var_info.def_scope_id)

        if not effective_def_scope:
            return

        self._set_delete_point(var_info, effective_def_scope)

    def _set_delete_point_in_parent_scope(self, var_info: VarInfo, def_scope: Scope) -> None:
        """Set deletion point in parent scope when def_scope has empty stmt_ids.

        This handles cases where def_scope is an 'if' scope (which only contains
        control flow, not actual statements). We use the parent scope to find
        the correct deletion point after the if statement.
        """
        if not def_scope.parent:
            var_info.delete_after_stmt_id = var_info.def_stmt_id
            var_info.delete_scope_id = def_scope.scope_id
            var_info.delete_after_scope_exit = False
            return

        parent_scope = def_scope.parent

        if def_scope.entry_stmt_id and def_scope.entry_stmt_id in parent_scope.stmt_ids:
            if_stmt_pos = parent_scope.stmt_ids.index(def_scope.entry_stmt_id)

            if if_stmt_pos < len(parent_scope.stmt_ids) - 1:
                last_stmt_id = parent_scope.stmt_ids[if_stmt_pos + 1]
                var_info.delete_after_stmt_id = last_stmt_id
                var_info.delete_scope_id = parent_scope.scope_id
                var_info.delete_after_scope_exit = False
            else:
                var_info.delete_after_stmt_id = def_scope.entry_stmt_id
                var_info.delete_scope_id = parent_scope.scope_id
                var_info.delete_after_scope_exit = True
        else:
            var_info.delete_after_stmt_id = var_info.def_stmt_id
            var_info.delete_scope_id = parent_scope.scope_id
            var_info.delete_after_scope_exit = False

    def _set_delete_point(self, var_info: VarInfo, def_scope: Scope) -> None:
        """Set deletion point based on representative scope.

        Unified logic for both same-scope and cross-scope usage.
        Maps all use points to def_scope's stmt_ids and finds the last one
        in execution order to ensure deletion happens after all uses complete.
        """
        if not def_scope.stmt_ids:
            self._set_delete_point_in_parent_scope(var_info, def_scope)
            return

        stmt_ids_info = {}

        for scope_id, stmt_id in var_info.use_points:
            use_scope = self.result.scope_map.get(scope_id)
            if not use_scope:
                continue

            if scope_id == def_scope.scope_id:
                stmt_ids_info[stmt_id] = False
            elif use_scope.is_nested_in(def_scope):
                current = use_scope
                while current.parent and current.parent.scope_id != def_scope.scope_id:
                    current = current.parent

                if current.entry_stmt_id:
                    stmt_ids_info[current.entry_stmt_id] = True
            elif def_scope.find_common_ancestor(use_scope):
                # Brother scope: use container stmt from def_scope's parent
                # (e.g., if_body's parent is if scope, which has entry_stmt_id)
                if def_scope.parent and def_scope.parent.entry_stmt_id:
                    stmt_ids_info[def_scope.parent.entry_stmt_id] = True

        last_stmt_id = None
        last_from_nested = False
        for sid in def_scope.stmt_ids:
            if sid in stmt_ids_info:
                last_stmt_id = sid
                last_from_nested = stmt_ids_info[sid]

        if last_stmt_id:
            var_info.delete_after_stmt_id = last_stmt_id
            var_info.delete_scope_id = def_scope.scope_id
            var_info.delete_after_scope_exit = last_from_nested
        else:
            var_info.delete_after_stmt_id = var_info.def_stmt_id
            var_info.delete_scope_id = def_scope.scope_id
            var_info.delete_after_scope_exit = False

    def _build_delete_points_from_var_info(self) -> None:
        """Build delete_points dict from VarInfo."""
        for var_name, var_info in self.result.var_info.items():
            if var_name in self.exempt_vars:
                continue

            if var_info.delete_after_stmt_id:
                self.result.delete_points.setdefault(var_info.delete_after_stmt_id, set()).add(var_name)

    def _next_scope_id(self) -> int:
        """Generate next scope ID."""
        self.scope_counter += 1
        return self.scope_counter

    def _get_node_id(self, node: ast.AST) -> int:
        """Get unique ID for an AST node."""
        return id(node)

    def _create_scope(
        self, scope_type: str, entry_stmt_id: Optional[int] = None, exit_stmt_id: Optional[int] = None
    ) -> Scope:
        """Create a new scope and add to tree."""
        scope = Scope(
            scope_id=self._next_scope_id(),
            scope_type=scope_type,
            parent=self.current_scope,
            entry_stmt_id=entry_stmt_id,
            exit_stmt_id=exit_stmt_id,
        )

        if self.current_scope:
            self.current_scope.children.append(scope)

        self.result.scope_map[scope.scope_id] = scope

        return scope

    def _push_scope(self, scope: Scope) -> None:
        """Push a scope onto the stack and set as current."""
        self.scope_stack.append(scope)
        self.current_scope = scope

    def _pop_scope(self) -> Optional[Scope]:
        """Pop the current scope from stack."""
        if self.scope_stack:
            popped = self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1] if self.scope_stack else None
            return popped
        return None

    def _record_var_def(self, var_name: str, stmt_id: Optional[int]) -> None:
        """Record a variable definition.

        Collects all definition scopes in def_scope_ids set.
        The effective definition scope will be computed as the common ancestor
        of all definition scopes during deletion point calculation.
        """
        if not self.current_scope:
            return

        if stmt_id is None:
            return

        self.current_scope.tensor_defs.add(var_name)

        if var_name not in self.result.var_info:
            self.result.var_info[var_name] = VarInfo(
                var_name=var_name,
                def_scope_id=self.current_scope.scope_id,
                def_stmt_id=stmt_id,
                def_scope_ids={self.current_scope.scope_id},
            )
        else:
            info = self.result.var_info[var_name]
            info.def_stmt_id = stmt_id
            info.def_scope_ids.add(self.current_scope.scope_id)

    def _record_var_use(self, var_name: str, stmt_id: Optional[int]) -> None:
        """Record a variable use."""
        if stmt_id is None:
            return

        if var_name in self.exempt_vars:
            return

        if var_name not in self.result.var_info:
            return

        scope_id = self.current_scope.scope_id if self.current_scope else 0
        self.result.var_info[var_name].use_points.append((scope_id, stmt_id))

    def _visit_assign_target(
        self, target: ast.expr, value_node: Optional[ast.AST] = None, ann_type: Optional[str] = None
    ) -> None:
        """Visit assignment target and record variable definitions."""
        if isinstance(target, ast.Name):
            var_name = target.id

            if var_name in self.exempt_vars:
                return

            self._record_var_def(var_name, self.current_stmt_id)

        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_assign_target(elt, value_node, ann_type)

        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)
