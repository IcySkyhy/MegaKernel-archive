# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Centralized expression evaluator for resolving Python expressions against closure variables."""

from __future__ import annotations

__all__ = ["ExprEvaluator"]


import ast
from typing import TYPE_CHECKING, Any, Callable

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType

from .diagnostics import ParserTypeError

if TYPE_CHECKING:
    from ._span_tracker import SpanTracker

# Safe subset of builtins allowed during expression evaluation
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "max": max,
    "min": min,
    "abs": abs,
    "sum": sum,
    "round": round,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "range": range,
    "isinstance": isinstance,
    "type": type,
    "True": True,
    "False": False,
    "None": None,
}


class ExprEvaluator:
    """Evaluates Python AST expressions against closure variables.

    Uses Python's eval() with a restricted builtins whitelist for safety.
    Can return raw Python values or convert them to IR expressions.

    Names resolve the way Python resolves them: kernel-local variables the parser
    has already folded to compile-time constants shadow same-named closure
    variables. This lets every compile-time position (``make_tile_group(addrs=)``,
    shapes, axes, ``x in <list>``, ...) accept a local ``addr = 0x10000`` exactly
    as it accepts the literal.
    """

    def __init__(
        self,
        closure_vars: dict[str, Any],
        span_tracker: "SpanTracker | None" = None,
        local_binding: "Callable[[str], tuple[str, Any] | None] | None" = None,
    ):
        """Initialize expression evaluator.

        Args:
            closure_vars: Variables from the enclosing scope
            span_tracker: Optional span tracker for source locations in errors
            local_binding: Optional lookup of how a name is bound inside the
                kernel, as ``(kind, value)``; see ``ASTParser.local_binding``.
                Without it only closure variables resolve.
        """
        self.closure_vars = closure_vars
        self.span_tracker = span_tracker
        self.local_binding = local_binding

    def eval_expr(self, node: ast.expr) -> Any:
        """Evaluate an AST expression node against closure variables.

        Args:
            node: AST expression node to evaluate

        Returns:
            The Python value resulting from evaluation

        Raises:
            ParserTypeError: If expression cannot be evaluated
        """
        span = self._get_span(node)
        expr_str = ast.unparse(node)

        try:
            code = compile(ast.Expression(body=node), "<pypto-eval>", "eval")
            # Security note: closure_vars come from the user's own enclosing Python scope.
            # The DSL parser is not a sandbox — users already have full control of the
            # Python process. The builtins whitelist prevents accidental access to dangerous
            # builtins (open, __import__, exec) but does not prevent calling methods on
            # objects the user placed in scope, which is by design.
            env = {**self.closure_vars, **self._local_const_values(node)}
            return eval(code, {"__builtins__": _SAFE_BUILTINS}, env)
        except NameError as e:
            raise ParserTypeError(
                f"Cannot resolve expression '{expr_str}': {e}",
                span=span,
                hint=self._unresolved_hint(node),
            ) from e
        except Exception as e:
            raise ParserTypeError(
                f"Failed to evaluate expression '{expr_str}': {e}",
                span=span,
            ) from e

    def try_eval_expr(self, node: ast.expr) -> tuple[bool, Any]:
        """Try to evaluate an AST expression, returning success status.

        Non-throwing variant of eval_expr for cases where evaluation failure
        should fall through to other resolution strategies.

        Args:
            node: AST expression node to evaluate

        Returns:
            Tuple of (success, value). On failure, value is None.
        """
        try:
            return (True, self.eval_expr(node))
        except ParserTypeError:
            return (False, None)

    def _local_const_values(self, node: ast.expr) -> dict[str, Any]:
        """Parse-time values of the kernel-local names *node* references.

        A ``runtime`` binding aborts the whole evaluation rather than being left
        out: omitting the name would let it resolve to a same-named closure
        variable and fold a wrong constant.

        Only the names the expression actually reads are resolved, so the cost
        stays proportional to the expression rather than to the size of the
        parser's scopes.
        """
        values: dict[str, Any] = {}
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Name) or sub.id in values or self.local_binding is None:
                continue
            binding = self.local_binding(sub.id)
            if binding is None:
                continue
            kind, value = binding
            if kind == "runtime":
                raise ParserTypeError(
                    f"'{sub.id}' is a runtime value, not known while parsing",
                    span=self._get_span(node),
                )
            found, value = (True, value) if kind == "parse_time" else self.ir_to_python_value(value)
            if found:
                values[sub.id] = value
        return values

    @classmethod
    def ir_to_python_value(cls, expr: Any) -> tuple[bool, Any]:
        """Parse-time Python value of *expr*.

        Inverse of :meth:`python_value_to_ir`, and the single place the parser
        asks "does this have a value known while parsing?". IR constants unwrap
        to their Python value; a value that is not IR at all (an enum such as
        ``DataType``, a ``TileType``) already *is* a parse-time value; an IR
        expression that is not constant, and a bare AST node, have none.

        Returns:
            Tuple of (converted, value). ``converted`` is False when the
            expression has no parse-time value; ``value`` is then None.
        """
        if isinstance(expr, ir.ConstBool):
            return (True, bool(expr.value))
        if isinstance(expr, ir.ConstInt):
            # A bool may be carried as a BOOL-typed ConstInt; keep it a bool so it
            # round-trips through python_value_to_ir unchanged.
            if expr.type.dtype == DataType.BOOL:
                return (True, bool(expr.value))
            return (True, int(expr.value))
        if isinstance(expr, ir.ConstFloat):
            return (True, float(expr.value))
        if isinstance(expr, ir.Cast):
            # A cast of a constant is still constant; apply the cast to the value.
            found, value = cls.ir_to_python_value(expr.operand)
            if not found or not isinstance(value, (bool, int, float)):
                return (False, None)
            dtype = expr.type.dtype
            if dtype == DataType.BOOL:
                return (True, bool(value))
            if dtype.is_int():
                return (True, int(value))
            if dtype.is_float():
                return (True, float(value))
            return (False, None)
        if isinstance(expr, ir.MakeTuple):
            elements = []
            for element in expr.elements:
                found, value = cls.ir_to_python_value(element)
                if not found:
                    return (False, None)
                elements.append(value)
            return (True, tuple(elements))
        if isinstance(expr, (ir.Expr, ast.AST)):
            return (False, None)
        return (True, expr)

    def _unresolved_hint(self, node: ast.expr) -> str:
        """Hint for a name that eval() could not resolve.

        A name bound in the kernel gets a targeted message: it exists, but as a
        runtime value, and this position needs a value known while parsing.
        """
        if self.local_binding is not None:
            runtime_names = sorted(
                {
                    sub.id
                    for sub in ast.walk(node)
                    if isinstance(sub, ast.Name) and (self.local_binding(sub.id) or (None,))[0] == "runtime"
                }
            )
            if runtime_names:
                names = ", ".join(f"'{name}'" for name in runtime_names)
                return (
                    f"{names} is a runtime value; this position requires a compile-time constant. "
                    "Assign it from literals or other constants instead of from a tensor/loop value"
                )
        return "Make sure the variable is defined in the enclosing scope or is a compile-time constant"

    def python_value_to_ir(self, value: Any, span: ir.Span) -> ir.Expr:
        """Convert a Python value to an IR expression.

        Args:
            value: Python value (bool, int, float, ir.Expr, list, or tuple)
            span: Source span for the expression

        Returns:
            IR expression representing the value
        """
        # bool before int because isinstance(True, int) is True
        if isinstance(value, bool):
            return ir.ConstBool(value, span)
        if isinstance(value, int):
            return ir.ConstInt(value, DataType.INDEX, span)
        if isinstance(value, float):
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        if isinstance(value, ir.Expr):
            return value
        # Tile wrapper — unwrap to the underlying IR Call expression.
        if hasattr(value, "unwrap") and callable(value.unwrap):
            inner = value.unwrap()
            if isinstance(inner, ir.Expr):
                return inner
        if isinstance(value, (list, tuple)):
            return ir.MakeTuple([self.python_value_to_ir(elt, span) for elt in value], span)
        raise ParserTypeError(
            f"Unsupported closure variable type: {type(value).__name__}",
            span=span,
            hint="Closure variables must be int, float, bool, list, tuple, or IR expressions",
        )

    def _get_span(self, node: ast.AST) -> "ir.Span":
        """Get span for an AST node, falling back to unknown."""
        if self.span_tracker is not None:
            return self.span_tracker.get_span(node)

        return ir.Span.unknown()
