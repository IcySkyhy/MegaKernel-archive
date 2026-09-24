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

"""PTO Script Parser."""

import ast
from collections.abc import Iterator
import functools
import inspect
import re
from typing import Any, Callable, Optional, Union

import pypto
from pypto._utils import clear_source_location, set_source_location
from pypto.error import FeError, ParserError, RenderedParserError
from pypto.symbolic_scalar import SymbolicScalar, SymInt

from .context import Context
from .diagnostics import DiagnosticLevel, Diagnostics, Source
from .evaluator import ExprEvaluator
from .liveness import ScopeLivenessAnalyzer

ParamSpec = tuple[str, bool, Any]


class NestedFunctionMarker:
    """Marker used to identify functions intended for nested inline execution."""

    def __init__(self) -> None:
        self._original_func: Optional[Callable] = None
        self._func_name: str = ""

    def _check_input_defs_match(self, call_args: list, param_specs: list) -> None:
        """Check if input tensor definitions match with call arguments.

        This method validates that the tensor arguments passed to a nested function
        match the tensor definitions in the function signature, similar to the
        validation performed in JIT functions.

        Parameters
        ----------
        call_args : list
            List of actual arguments passed to the function.
        param_specs : list
            List of parameter specifications (name, is_tensor, annotation).

        Raises
        ------
        ValueError
            If tensor shapes, dtypes, or other properties don't match.
        """
        # Check the number of input tensors and input tensor definitions
        if len(param_specs) != len(call_args):
            raise FeError(
                RuntimeError(
                    f"There are {len(param_specs)} input param(s), \
                but {len(call_args)} input arg(s)."
                )
            )

        def ordinal(n):
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
            if 11 <= n % 100 <= 13:
                suffix = 'th'
            return f"{n}{suffix}"

        idx = 0
        for (param_name, is_tensor, annotation), arg_value in zip(param_specs, call_args):
            idx += 1
            if not is_tensor:
                continue

            if not isinstance(arg_value, pypto.Tensor):
                continue

            input_tensor_def = annotation
            if not isinstance(input_tensor_def, pypto.Tensor):
                continue

            # Skip checking if the input tensor definition is None or（shape len is 0 && shape object is not list）
            if len(input_tensor_def.shape) != 0 or input_tensor_def.status_shape is not None:
                # 根据属性input_tensor_def.status_shape做判断, def的shape len 小于等于 tensor的shape len
                is_diff_shape = (
                    len(arg_value.shape) != len(input_tensor_def.shape)
                    if input_tensor_def.status_shape is None
                    else len(arg_value.shape) < len(input_tensor_def.shape)
                )

                # Check the shape of input tensors and input tensor definitions
                if is_diff_shape:
                    raise FeError(
                        ValueError(
                            f"In nested function '{self._func_name}': "
                            f"The number of dimensions of {ordinal(idx)} parameter '{param_name}' "
                            f"({len(arg_value.shape)}) does not match "
                            f"number of dimensions of parameter definition ({len(input_tensor_def.shape)})."
                        )
                    )
                for i, dim in enumerate(input_tensor_def.shape):
                    if isinstance(dim, int) and arg_value.shape[i] != dim:
                        raise FeError(
                            ValueError(
                                f"In nested function '{self._func_name}': "
                                f"The shape of {ordinal(idx)} parameter '{param_name}' {arg_value.shape} "
                                f"does not match the shape of parameter definition {input_tensor_def.shape}."
                            )
                        )

            # Check the dtype of input tensors and input tensor definitions
            if input_tensor_def.explicit_dtype is not None and arg_value.dtype != input_tensor_def.dtype:
                raise FeError(
                    ValueError(
                        f"In nested function '{self._func_name}': "
                        f"The dtype of {ordinal(idx)} parameter '{param_name}' ({arg_value.dtype}) "
                        f"does not match the dtype of parameter definition ({input_tensor_def.dtype})."
                    )
                )


DEFAULT_VISIT = {
    "Interactive",
    "Module",
    "Expression",
    "Pass",
}

_NESTED_CALL_UNHANDLED = object()


def _catch_parser_errors(func):
    """Decorator to normalize parser error handling for public APIs."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RenderedParserError:
            # Flush any pending messages (like context info) in the current diagnostics
            if hasattr(self, "diag"):
                self.diag._render()
            # Already rendered; just re-raise.
            raise
        except ParserError as err:
            # User-triggered parser error with location info.
            self.diag.error(err.node, str(err))
        except Exception as err:  # pylint: disable=broad-except
            # Unexpected native error, surface as bug; try to find a node.
            node = kwargs.get("node")
            if node is None:
                for arg in args:
                    if isinstance(arg, ast.AST):
                        node = arg
                        break
            if node is None:
                raise
            else:
                self.diag.error(node, err)

    return wrapper


class Parser(ast.NodeVisitor):
    """Main parser for PTO Script that converts Python AST to PTO IR.

    The Parser class implements the visitor pattern to traverse the AST and
    generate PTO intermediate representation. It manages variable scoping through
    the Context system, evaluates expressions using ExprEvaluator, and reports
    errors through the Diagnostics system.

    Key Features
    ------------
    - Lazy parsing: AST preparation is separated from IR generation
    - Dynamic dimension binding: Symbolic dimensions resolved at runtime
    - Automatic memory management: Variables deleted based on liveness analysis
    - Nested function inlining: Functions marked with @function are inlined
    - Rich error reporting: Source-aware error messages with context

    Parsing Workflow
    ----------------
    1. parse(): Prepare AST and run liveness analysis
    2. bind_dynamic_dims_to_input_tensors(): Optionally bind symbolic dimensions
    3. execute(): Traverse AST and generate PTO IR
    4. Result: pypto.Function ready for compilation and execution

    Attributes
    ----------
    diag : Diagnostics
        Diagnostics instance for error reporting with source locations.
    context : Context
        Context instance managing variable scopes and lifetime.
    delete_after : dict[int, set[str]]
        Mapping from statement IDs to variables to delete after that statement.
        Generated by liveness analysis to enable automatic memory management.
    scope_lift_suggestions : list[tuple[str, int, int]]
        Suggestions for scope lifting to handle cross-scope variable usage.
        Each tuple contains (var_name, current_scope_id, target_scope_id).
        This is stored for reference; actual scope lifting requires manual code changes.
    _parsed_node : Optional[ast.AST]
        The prepared AST node, stored after parse() for lazy execution.
    _parsed_extra_vars : dict[str, Any]
        Extra variables (globals, nonlocals) captured from the original function.
    _result : Optional[Any]
        The result of parsing, typically a pypto.Function object.
    _signature_cache : Optional[tuple[list[pypto.Tensor], list[pypto.Tensor]]]
        Cached function signature (inputs, outputs) to avoid re-parsing.
    _lowered_signature_cache: Optional[tuple[list[pypto.Tensor], list[pypto.Tensor]]]
        Cached function signature (inputs, outputs) with symbolic dimensions lowered to concrete values.
    _bound_dim_values : Optional[dict[str, SymInt]]
        Mapping from symbolic dimension names to bound values. The bound values can be
        concrete integers (static specialization) or runtime symbolic expressions
        (e.g., input shape-derived SymbolicScalar).

    Examples
    --------
    >>> source = Source(my_function)
    >>> parser = Parser(source, captured_vars)
    >>> parser.parse()
    >>> parser.bind_dynamic_dims_to_input_tensors([[1024, 1024]])
    >>> pto_func = parser.execute()
    """

    diag: Diagnostics
    context: Context
    delete_after: dict[int, set[str]]
    scope_lift_suggestions: list[tuple[str, int, int]]
    _parsed_node: Optional[ast.AST]
    _parsed_extra_vars: dict[str, Any]
    _result: Optional[Any]
    _signature_cache: Optional[tuple[list[pypto.Tensor], list[pypto.Tensor]]]
    _lowered_signature_cache: Optional[tuple[list[pypto.Tensor], list[pypto.Tensor]]]

    # ==========================================================================================
    # Public API
    # ==========================================================================================

    def __init__(self, source: Source, extra_vars: Optional[dict[str, Any]] = None) -> None:
        self.diag = Diagnostics(source)
        self.context = Context()
        self.delete_after = {}
        self.scope_lift_suggestions = []
        self._parsed_node = None
        self._parsed_extra_vars = extra_vars or {}
        self._result = None
        self._signature_cache = None
        self._lowered_signature_cache = None
        self._bound_dim_values: Optional[dict[str, SymInt]] = None
        self.input_pto_tensor: Optional[list[pypto.Tensor]] = None

    @staticmethod
    def match_input_shapes(
        input_shapes: list[list[int]],
        input_tensor_defs: Optional[list[pypto.Tensor]] = None,
    ) -> dict[str, int]:
        """Match input tensors to symbolic dimensions.

        Creates a mapping from SymbolicScalar objects (found in tensor shapes or
        as symbolic parameters) to their concrete values based on actual input data.

        Parameters
        ----------
        input_shapes : list[list[int]]
            List of input shapes.

        input_tensor_defs : Optional[list[pypto.Tensor]]
            List of input tensor definitions.

        Returns
        -------
        dict[str, int]
            Mapping from SymbolicScalar objects to their concrete values.
        """
        dim_value_map = {}

        # Get the signature to know which inputs have symbolic dimensions

        def _assign_dim_value(dim: pypto.SymbolicScalar, actual_value: int) -> None:
            if dim_value_map.get(str(dim), actual_value) != actual_value:
                raise FeError(
                    ValueError(
                        f"Symbolic scalar {dim} has multiple concrete values: {dim_value_map[dim]} and {actual_value}"
                    )
                )
            dim_value_map[str(dim)] = actual_value

        # Iterate through both the actual inputs and their definitions
        for actual_input_shape, tensor_def in zip(input_shapes, input_tensor_defs):
            if isinstance(actual_input_shape, list):
                # For Tensor inputs, map each symbolic dimension to its concrete shape value
                for axis, dim in enumerate(tensor_def.shape):
                    if isinstance(dim, pypto.SymbolicScalar):
                        # Extract the actual shape value from the input tensor
                        actual_value = actual_input_shape[axis]
                        if isinstance(actual_value, int):
                            _assign_dim_value(dim, actual_value)
            else:
                raise FeError(TypeError(f"Invalid input shape type: {type(actual_input_shape)}, expected list"))
        return dim_value_map

    def source_name(self):
        return self.diag.source.source_name

    @_catch_parser_errors
    def parse(self) -> "Parser":
        """The main parse method for parser (lazy mode).

        Prepares the AST but defers actual parsing until execute() is called.

        Returns
        -------
        res : Parser
            Returns self for chaining.
        """
        node = self.diag.source.as_ast()
        analyzer = ScopeLivenessAnalyzer()
        exempt_vars = set(self._parsed_extra_vars.keys())
        result = analyzer.analyze(node, exempt_vars)
        self.delete_after = result.delete_points
        self.scope_lift_suggestions = result.scope_lift_suggestions

        # Store for later execution (lazy mode)
        self._parsed_node = node
        return self

    @_catch_parser_errors
    def bind_dynamic_dims_to_input_tensors(
        self,
        input_tensor_defs: Optional[list[pypto.Tensor]] = None,
    ) -> None:
        """Bind symbolic dimensions to runtime input shape expressions.

        Instead of specializing symbolic dimensions to concrete integers, this method
        binds each symbolic dimension to the corresponding runtime shape of the input
        tensor. This keeps dynamic axes truly dynamic across invocations.

        Parameters
        ----------
        input_tensor_defs : Optional[list[pypto.Tensor]]
            Tensor definitions from the signature. If None, the signature is parsed
            from the function annotations.
        """
        if input_tensor_defs is None:
            input_tensor_defs, _ = self.get_signature()

        lowered_input_defs, _ = self.get_signature(lower_symbolic_dims=True)

        dim_value_map: dict[str, SymInt] = {}

        def _assign_dim_value(dim: pypto.SymbolicScalar, actual_value: SymInt) -> None:
            key = str(dim)
            if key in dim_value_map:
                # Allow a dynamic symbol to appear in multiple input tensors.
                # Keep the first binding so the symbol is usable in expressions.
                # Concrete shape equality is still enforced at call time via
                # match_input_shapes() when outputs are allocated.
                if str(dim_value_map[key]) != str(actual_value):
                    return
            dim_value_map[key] = actual_value

        for tensor_def, lowered_tensor in zip(input_tensor_defs, lowered_input_defs):
            for axis, dim in enumerate(tensor_def.shape):
                if isinstance(dim, pypto.SymbolicScalar):
                    runtime_dim = lowered_tensor.shape[axis]
                    _assign_dim_value(dim, runtime_dim)

        self._bound_dim_values = dim_value_map

    @_catch_parser_errors
    def get_signature(
        self,
        lower_symbolic_dims: bool = False,
    ) -> tuple[list[pypto.Tensor], list[pypto.Tensor]]:
        """Extract function signature (inputs and outputs) without full parsing.

        Returns
        -------
        res : tuple[list[pypto.Tensor], list[pypto.Tensor]]
            A tuple of (input_tensors, output_tensors).

        Raises
        ------
        RuntimeError
            If parse() was not called before get_signature().
        """

        if self._signature_cache is not None and not lower_symbolic_dims:
            return self._signature_cache

        elif self._lowered_signature_cache is not None and lower_symbolic_dims:
            return self._lowered_signature_cache

        function_node = self.diag.source.as_ast()

        # Temporarily set up context to parse signature
        with self.context.with_frame():
            for k, v in self._parsed_extra_vars.items():
                self.context.add(k, v)
            # If sample inputs were provided, use them to concretize symbolic dims.
            self._apply_bound_dim_values_to_context_frame()

            # Get input arguments (only tensors allowed)
            tensor_input_args_def = self._visit_arguments(function_node.args)

            tensor_input_args = self.input_pto_tensor[:len(tensor_input_args_def)]  # ensure len equal

            for in_obj, def_obj in zip(tensor_input_args, tensor_input_args_def):
                in_obj.name = def_obj.name

            # Return annotation is not allowed; use out parameter and out.move() instead.
            if function_node.returns is not None:
                raise ParserError(
                    function_node.returns,
                    NotImplementedError(
                        "Return annotation is not allowed. Use an out parameter and "
                        "out.move(...) inside the kernel instead."
                    ),
                )
            output_tensors = []

            self._signature_cache = (
                tensor_input_args,
                output_tensors,
            )

            lowered_input_tensors, lowered_output_tensors = [], []
            for input_tensor in tensor_input_args:
                shapes = [-1 if isinstance(dim, pypto.SymbolicScalar) else dim for dim in input_tensor.shape]
                lowered_input_tensors.append(
                    pypto.Tensor(
                        shapes,
                        input_tensor.dtype,
                        input_tensor.name,
                        input_tensor.format,
                        input_tensor.data_ptr,
                        input_tensor.device,
                        input_tensor.ori_shape,
                    )
                )
            for output_tensor in output_tensors:
                shapes = [-1 if isinstance(dim, pypto.SymbolicScalar) else dim for dim in output_tensor.shape]
                lowered_output_tensors.append(
                    pypto.Tensor(
                        shapes,
                        output_tensor.dtype,
                        output_tensor.name,
                        output_tensor.format,
                        output_tensor.data_ptr,
                        output_tensor.device,
                        output_tensor.ori_shape,
                    )
                )

            self._lowered_signature_cache = (
                lowered_input_tensors,
                lowered_output_tensors,
            )
            return self._signature_cache if not lower_symbolic_dims else self._lowered_signature_cache

    @_catch_parser_errors
    def execute(self) -> Any:
        """Execute the deferred parsing.

        Returns
        -------
        res : Any
            The AST node visiting result.

        Raises
        ------
        RuntimeError
            If parse() was not called before execute().
        """
        if self._parsed_node is None:
            raise FeError(RuntimeError("parse() must be called before execute()"))

        # Return cached result if already executed
        if self._result is not None:
            return self._result

        # Execute the deferred parsing
        with self.context.with_frame():
            for k, v in self._parsed_extra_vars.items():
                self.context.add(k, v)
            # Apply any concrete bindings for symbolic dimensions
            self._apply_bound_dim_values_to_context_frame()
            self._result = self.visit(self._parsed_node)
        return self._result

    def report(self, node: ast.AST, msg: str, level: DiagnosticLevel) -> None:
        """Report a diagnostic."""
        self.diag.emit(node, msg, level)

    def visit(self, node: ast.AST) -> Any:
        """The general visiting method.

        Parameters
        ----------
        node : ast.AST
            The AST node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        if isinstance(node, (list, tuple)):
            result = None
            for item in node:
                res = self.visit(item)
                if res is not None:
                    result = res
            return result
        if not isinstance(node, ast.AST):
            raise ParserError(
                node,
                TypeError(f"Expected ast.AST, got {type(node)}."),
            )
        name = node.__class__.__name__.split(".")[-1]

        if name in DEFAULT_VISIT:
            func = self.generic_visit
        else:
            # Convert CamelCase to snake_case for function names
            snake_case_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            func = getattr(self, f"_visit_{snake_case_name}", None)
        if func is None:
            raise ParserError(
                node,
                NotImplementedError(
                    f"{name} is not supported by the PTO parser yet. "
                    f"Please check the documentation for supported Python features."
                ),
            )
        return func(node)

    # ==========================================================================================
    # Private APIs (implementation details)
    # ==========================================================================================
    def _get_function_def_from_func(self, func: Any) -> ast.FunctionDef:
        """Get FunctionDef AST node from a function object.

        Parameters
        ----------
        func : Any
            The function object (can be a callable or NestedFunctionMarker).

        Returns
        -------
        Optional[ast.FunctionDef]
            The FunctionDef AST node if found, None otherwise.
        """
        # If it's a NestedFunctionMarker, get the original function
        # Check for _original_func attribute to identify NestedFunctionMarker instances
        if hasattr(func, "_original_func"):
            func = func._original_func
        return Source(func).as_ast()

    def _collect_function_environment(self, func: Any) -> dict[str, Any]:
        """Extract globals and nonlocals referenced by the function."""
        env: dict[str, Any] = {}
        if func is None:
            return env
        try:
            closure_vars = inspect.getclosurevars(func)
        except Exception:  # pylint: disable=broad-except
            return env
        env.update(closure_vars.globals)
        env.update(closure_vars.nonlocals)
        return env

    def _is_nested_function(self, decorator_list: list[ast.expr]) -> bool:
        """Check if a function is marked for nested calling by examining its decorators.

        Parameters
        ----------
        decorator_list : list[ast.expr]
            List of decorator expressions from the function definition.

        Returns
        -------
        bool
            True if the function is marked for nested calling, False otherwise.
        """
        for decorator in decorator_list:
            # Check if decorator is pto.frontend.function or evaluates to NestedFunctionMarker
            try:
                # Try to evaluate the decorator expression
                decorator_value = self._visit_expr(decorator)
                # Check for _original_func attribute to identify NestedFunctionMarker instances
                if hasattr(decorator_value, "_original_func"):
                    return True
            except Exception:  # pylint: disable=broad-except
                # If evaluation fails, try to check if it's a direct reference to pto.frontend.function
                # Check if it's an Attribute node like pto.frontend.function
                if isinstance(decorator, ast.Attribute):
                    # Check if it's pto.frontend.function
                    attr_chain = []
                    current = decorator
                    while isinstance(current, ast.Attribute):
                        attr_chain.insert(0, current.attr)
                        if isinstance(current.value, ast.Name):
                            attr_chain.insert(0, current.value.id)
                            break
                        elif isinstance(current.value, ast.Attribute):
                            current = current.value
                        else:
                            break
                    # Check if the chain matches pto.frontend.function
                    if (
                        len(attr_chain) >= 3
                        and attr_chain[0] == "pto"
                        and attr_chain[1] == "frontend"
                        and attr_chain[2] == "function"
                    ):
                        return True
                # Also check if it's a simple Name node that refers to function
                elif isinstance(decorator, ast.Name):
                    # Check if the name refers to pto.frontend.function in the context
                    var_values = self.context.get()
                    if decorator.id in var_values:
                        value = var_values[decorator.id]
                        # Check for _original_func attribute to identify NestedFunctionMarker instances
                        if hasattr(value, "_original_func"):
                            return True
        return False

    def _eval_expr(
        self,
        node: Union[ast.Expression, ast.expr],
        extra_vars: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Evaluate an expression node using the current context.

        This method evaluates expressions during parsing, handling both regular
        expressions and special cases like nested function calls. It merges the
        current context with any extra variables provided.

        Parameters
        ----------
        node : Union[ast.Expression, ast.expr]
            The expression node to evaluate.
        extra_vars : Optional[dict[str, Any]], optional
            Additional variables to make available during evaluation.

        Returns
        -------
        Any
            The evaluated result of the expression.
        """
        if isinstance(node, ast.Expr) and hasattr(node, "value"):
            node = node.value

        if isinstance(node, ast.Call):
            nested_result = self._try_nested_call(node, extra_vars)
            if nested_result is not _NESTED_CALL_UNHANDLED:
                return nested_result

        var_values = self.context.get()
        if extra_vars is not None:
            for k, v in extra_vars.items():
                var_values[k] = v
        return ExprEvaluator.eval(node, var_values, self.diag)

    def _apply_bound_dim_values_to_context_frame(self) -> None:
        """Replace symbolic scalars in the current frame with bound values.
        This method is called after bind_dynamic_dims_to_input_tensors() or
        bind_dynamic_dims_to_input_tensors(). It iterates through all variables in
        the current frame and replaces SymbolicScalar instances with their bound
        values when available. The bound values can be concrete integers or runtime
        symbolic expressions.
        """
        if not self._bound_dim_values:
            return
        if not self.context.frames:
            return

        def get_tuple_runtime_value(current_value: tuple):
            current_value_list = list(current_value)
            for axis, dim in enumerate(current_value):
                if isinstance(dim, pypto.SymbolicScalar):
                    dim = self._bound_dim_values[str(dim)]
                    current_value_list[axis] = dim
            return current_value_list

        current_frame = self.context.frames[-1]
        for var_name in list(current_frame.vars):
            values_stack = self.context.name2value.get(var_name, [])
            if not values_stack:
                continue
            current_value = values_stack[-1]
            if isinstance(current_value, tuple):
                self.context.add(var_name, tuple(get_tuple_runtime_value(current_value)))
            elif isinstance(current_value, SymbolicScalar) and str(current_value) in self._bound_dim_values:
                bound_value = self._bound_dim_values[str(current_value)]
                self.context.add(var_name, bound_value)

    def _visit_body(self, node: list[ast.stmt]) -> Any:
        """The general body visiting method.

        Parameters
        ----------
        node : list[ast.stmt]
            The list of statements in body.

        Returns
        -------
        res : Any
            The visiting result.
        """
        for stmt in node:
            self.visit(stmt)
            self._auto_cleanup_after_stmt(stmt)

    def _validate_return_statements(self, node: ast.FunctionDef) -> None:
        """Forbid return statements; use out parameter and out.move() instead.

        Parameters
        ----------
        node : ast.FunctionDef
            The function definition node to validate.

        Raises
        ------
        ParserError
            If any return statement is found.
        """

        def find_returns(stmts: list[ast.stmt]) -> list[tuple[int, ast.Return]]:
            """Find all return statements with their positions."""
            returns = []
            for idx, stmt in enumerate(stmts):
                if isinstance(stmt, ast.Return):
                    returns.append((idx, stmt))
                elif isinstance(stmt, ast.If):
                    nested_returns = find_returns(stmt.body)
                    if nested_returns:
                        returns.extend(nested_returns)
                    if stmt.orelse:
                        returns.extend(find_returns(stmt.orelse))
                elif isinstance(stmt, ast.For):
                    nested_returns = find_returns(stmt.body)
                    if nested_returns:
                        returns.extend(nested_returns)
                    if stmt.orelse:
                        returns.extend(find_returns(stmt.orelse))
            return returns

        returns = find_returns(node.body)
        if returns:
            raise ParserError(
                returns[0][1],
                ValueError(
                    "Return statements are not allowed. Use an out parameter and "
                    "out.move(...) to write results instead."
                ),
            )

    def _mark_dynamic_dimensions(self, tensors: list[pypto.Tensor]) -> list[pypto.Tensor]:
        """Mark dynamic dimensions for tensors.

        Dynamic dimensions are those specified with pypto.dynamic() and represented
        as SymbolicScalar objects. This method marks them in the PTO IR.

        Parameters
        ----------
        tensors : list[pypto.Tensor]
            List of tensors to process.

        Returns
        -------
        list[pypto.Tensor]
            List of tensors with dynamic dimensions marked.
        """
        result = []
        for tensor in tensors:
            if isinstance(tensor, pypto.Tensor):
                shape = [-1 if isinstance(dim, pypto.SymbolicScalar) else dim for dim in tensor.shape]
                result.append(pypto.Tensor(shape, tensor.dtype, tensor.name))
            else:
                raise ParserError(
                    tensor,
                    TypeError(f"Tensor must be a pypto.Tensor, but got {type(tensor)}."),
                )
        return result

    def _add_tensor_args_to_context(self, tensor_args: list[pypto.Tensor]) -> None:
        """Add tensor arguments to the parsing context.

        Parameters
        ----------
        tensor_args : list[pypto.Tensor]
            List of tensor arguments to add to context.
        """
        for arg in tensor_args:
            if isinstance(arg, pypto.Tensor):
                arg.is_input = True
                self.context.add(arg.name, arg)

    def _add_metadata_to_context(self, func_name: str) -> None:
        """Add function metadata to context for use in other visit methods.

        Parameters
        ----------
        func_name : str
            The function name (e.g. for error reporting).
        """
        self.context.add("__func_name__", func_name)

    def _visit_function_def(self, node: ast.FunctionDef) -> pypto.Function:
        """The general function definition visit method.

        Parameters
        ----------
        node : ast.FunctionDef
            The FunctionDef node.

        Note
        ----
        FunctionDef node structure:
            name: str
            args: arguments
            body: list[stmt]
            decorator_list: list[expr]
            returns: Optional[expr]
        """
        # Validate return statements in function body (forbid return; use out.move() instead)
        self._validate_return_statements(node)

        # Check if function is marked for nested calling (before with block so it can be reused later)
        is_nested = self._is_nested_function(node.decorator_list)

        with self.context.with_frame():
            # Step 1: Extract function signature (no return annotation; output_args always [])
            tensor_input_args, output_args = self.get_signature(lower_symbolic_dims=True)

            # Step 2: Add arguments to parsing context
            self._add_tensor_args_to_context(tensor_input_args)

            # Step 3: Add metadata to context
            self._add_metadata_to_context(node.name)

            # Step 4: Create PTO function and parse body
            if is_nested:
                # For nested functions, we don't create a pypto.Function; body will be inlined on call.
                return None
            else:
                set_source_location(filename=self.source_name(), lineno=node.lineno)
                with pypto.function(node.name, *tensor_input_args, *output_args):
                    clear_source_location()
                    for _ in pypto.loop(1, name="default_loop_1"):
                        self._visit_body(node.body)

        return pypto.functions.get_last_function()

    def _visit_arg(self, node: ast.arg, default_value: Any = None) -> Union[pypto.Tensor, tuple[str, Any]]:
        """The general arg visiting method.

        Parameters
        ----------
        node : ast.arg
            The AST arg node.
        default_value : Any, optional
            Default value for non-tensor parameters (from node.defaults).

        Returns
        -------
        res : pypto.Tensor or tuple[str, Any]
            The tensor argument, or (param_name, default_value) for non-tensor params.

        Note
        ----
        Non-tensor parameters must come after all tensor parameters.
        """
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self._visit_arg(arg) for arg in node.elts]
        name = node.arg
        if node.annotation is None:
            # Non-tensor parameter (no annotation)
            return (name, default_value)
        anno = self._visit_expr(node.annotation)
        if hasattr(anno, 'to_tensor'):
            tensor = anno.to_tensor(name)
            return tensor
        else:
            # Non-tensor parameter (annotation is not pypto.Tensor)
            return (name, default_value)

    def _validate_arguments_node(self, node: ast.arguments) -> None:
        """Validate that arguments node has no unsupported features."""
        if node.vararg is not None:
            raise ParserError(
                node,
                NotImplementedError(
                    "Variable-length arguments (*args) are not supported. Please use a fixed number of arguments."
                ),
            )
        if len(node.kwonlyargs) > 0:
            raise ParserError(
                node,
                NotImplementedError(
                    "Keyword-only arguments are not supported. Please use regular positional or keyword arguments."
                ),
            )
        if len(node.kw_defaults) > 0:
            raise ParserError(
                node,
                NotImplementedError(
                    "Keyword argument defaults are not supported. All arguments must be explicitly provided."
                ),
            )
        if node.kwarg is not None:
            raise ParserError(
                node,
                NotImplementedError(
                    "Keyword argument packing (**kwargs) is not supported. Please use explicit keyword arguments."
                ),
            )
        n_defaults = len(node.defaults)
        if n_defaults > len(node.args):
            raise ParserError(
                node,
                ValueError("Number of default values exceeds number of arguments."),
            )
        if len(node.posonlyargs) > 0:
            raise ParserError(
                node,
                NotImplementedError("Position-only arguments are not supported. Please use regular arguments."),
            )

    def _build_arg_default_values(self, node: ast.arguments) -> list[Any]:
        """Build default value list for each argument. args[-n_defaults:] get defaults."""
        n_defaults = len(node.defaults)
        default_values = [None] * len(node.args)
        if n_defaults > 0:
            for i, default in enumerate(node.defaults):
                default_val = self._visit_expr(default)
                default_values[len(node.args) - n_defaults + i] = default_val
        return default_values

    def _raise_if_tensor_after_non_tensor(self, arg: ast.arg) -> None:
        """Raise if tensor param appears after non-tensor param."""
        raise ParserError(
            arg,
            ValueError(
                "Non-tensor parameters must come after all tensor "
                "parameters. "
                f"Found tensor parameter '{arg.arg}' after "
                "non-tensor parameter(s)."
            ),
        )

    def _raise_if_tensor_has_default(self, arg: ast.arg) -> None:
        """Raise if tensor param has default value."""
        raise ParserError(
            arg,
            ValueError(
                f"Default values are only allowed for non-tensor parameters. Parameter '{arg.arg}' is a tensor."
            ),
        )

    def _parse_arguments_with_specs(self, node: ast.arguments) -> tuple[list[pypto.Tensor], list[ParamSpec]]:
        """Parse function arguments into tensor args and param specs.

        Returns
        -------
        tensor_args : list[pypto.Tensor]
            List of tensor arguments.
        param_specs : list[ParamSpec]
            List of (name, is_tensor, value) for all parameters.
        """
        self._validate_arguments_node(node)
        default_values = self._build_arg_default_values(node)
        n_defaults = len(node.defaults)
        first_default_idx = len(node.args) - n_defaults if n_defaults > 0 else len(node.args)

        tensor_args: list[pypto.Tensor] = []
        param_specs: list[ParamSpec] = []
        seen_non_tensor = False

        for idx, arg in enumerate(node.args):
            result = self._visit_arg(arg, default_values[idx])
            if isinstance(result, pypto.Tensor):
                if idx >= first_default_idx:
                    self._raise_if_tensor_has_default(arg)
                if seen_non_tensor:
                    self._raise_if_tensor_after_non_tensor(arg)
                tensor_args.append(result)
                param_specs.append((arg.arg, True, result))
            elif isinstance(result, list):
                if seen_non_tensor:
                    self._raise_if_tensor_after_non_tensor(arg)
                for item in result:
                    if isinstance(item, pypto.Tensor):
                        tensor_args.append(item)
                        param_specs.append((arg.arg, True, item))
            else:
                name, default = result
                seen_non_tensor = True
                param_specs.append((name, False, default))

        return tensor_args, param_specs

    def _visit_arguments(self, node: ast.arguments) -> list[pypto.Tensor]:
        """The general arguments visiting method.

        Parameters
        ----------
        node : ast.arguments
            The AST arguments node.

        Returns
        -------
        res : tuple[list[pypto.Tensor], list[tuple[str, type]]]
            A tuple of (tensor_args, non_tensor_args) where:
            - tensor_args: list of Tensor arguments
            - non_tensor_args: list of (name, type) tuples for non-tensor arguments
        """
        tensor_args, _ = self._parse_arguments_with_specs(node)
        return tensor_args

    def _try_nested_call(self, node: ast.Call, extra_vars: Optional[dict[str, Any]] = None) -> Optional[Any]:
        """Attempt to inline a nested function call marked with @function decorator.

        This method handles inline expansion of functions decorated with
        @pypto.frontend.function. When such a function is called, its body is
        inlined directly into the caller's IR instead of generating a separate
        function call.

        The inlining process:
        1. Identify if the call target is a nested function
        2. Extract the function's AST and environment (closures, globals)
        3. Map call arguments to function parameters
        4. Parse the function body in a new scope with mapped parameters
        5. Return the result value

        Parameters
        ----------
        node : ast.Call
            The function call AST node.
        extra_vars : Optional[dict[str, Any]], optional
            Additional variables to include in the evaluation context.

        Returns
        -------
        Any
            The result of the inlined function, or _NESTED_CALL_UNHANDLED if inlining
            is not applicable.

        Raises
        ------
        ParserError
            If the function AST cannot be obtained or parameter mapping fails.
        """
        # Only simple name calls (no attributes/methods) are considered for inlining.
        if not isinstance(node.func, ast.Name):
            return _NESTED_CALL_UNHANDLED

        # Collect current context variables and any extra_vars provided by eval_expr.
        func_name = node.func.id
        var_values = self.context.get()
        if extra_vars:
            var_values = {**var_values, **extra_vars}

        if func_name not in var_values:
            return _NESTED_CALL_UNHANDLED

        # Resolve the callee function object; if it's a NestedFunctionMarker, unwrap to the original function.
        func_value = var_values[func_name]
        if isinstance(func_value, NestedFunctionMarker):
            func_obj = func_value._original_func
        else:
            func_obj = func_value

        # Check if the function is a builtin function or a function without source code.
        if inspect.isbuiltin(func_obj) or inspect.isbuiltin(func_value):
            return _NESTED_CALL_UNHANDLED

        # Also check if it's a builtin function by checking the module
        if hasattr(func_obj, '__module__') and func_obj.__module__ == 'builtins':
            return _NESTED_CALL_UNHANDLED

        # Merge closure/global variables of the target function to allow resolving
        # free variables used inside the nested function body.
        env_vars = self._collect_function_environment(func_obj)
        if env_vars:
            var_values = {**env_vars, **var_values}

        # Dynamically obtain the FunctionDef AST of the callee; fail fast if unavailable.
        func_def_node = self._get_function_def_from_func(func_obj)
        if func_def_node is None:
            # If we cannot get the AST for non-builtin functions, it might be a C extension
            # or other callable that doesn't have Python source code. Let the evaluator handle it.
            return _NESTED_CALL_UNHANDLED

        # If callee is a normal function, ensure it is decorated as nested; otherwise, bail out.
        if not isinstance(func_value, NestedFunctionMarker):
            if not self._is_nested_function(func_def_node.decorator_list):
                return _NESTED_CALL_UNHANDLED

        # Parse parameters/return annotations to get tensor/non-tensor lists and ordered specs.
        # Seed the temp frame with the callee's env so that annotations depending on globals
        # (e.g., pto, helper constants) resolve correctly.
        with self.context.with_frame():
            for name, value in env_vars.items():
                self.context.add(name, value)

            tensor_input_args, param_specs = self._parse_arguments_with_specs(func_def_node.args)

            # Return annotation not allowed; nested functions use out param and out.move().
            if func_def_node.returns is not None:
                raise ParserError(
                    func_def_node.returns,
                    ValueError(
                        "Return annotation is not allowed in nested functions. "
                        "Use an out parameter and out.move(...) instead."
                    ),
                )
            _output_args = []

        # Evaluate call-site arguments; keyword arguments are not supported yet.
        # Use merged env (locals + globals of callee + caller extras) so symbols referenced
        # in the callsite expressions are visible.
        call_args = [self._eval_expr(arg, var_values) for arg in node.args]
        if node.keywords:
            raise ParserError(
                node,
                NotImplementedError("Keyword arguments in nested function calls are not supported yet."),
            )

        # Validate argument count matches the signature.
        expected_arg_count = len(param_specs)
        if len(call_args) != expected_arg_count:
            raise ParserError(
                node,
                ValueError(f"Function {func_name} expects {expected_arg_count} arguments, but got {len(call_args)}"),
            )

        # If callee is a normal function, ensure it is decorated as nested; otherwise, bail out.
        if isinstance(func_value, NestedFunctionMarker):
            try:
                func_value._check_input_defs_match(call_args, param_specs)
            except ValueError as e:
                raise ParserError(node, e) from e

        body_nodes = func_def_node.body
        with self.context.with_frame():
            # Make callee globals/nonlocals available to the inlined body.
            for name, value in env_vars.items():
                self.context.add(name, value)

            # Bind parameters in declared order and validate types.
            for (param_name, is_tensor, annotation), arg_value in zip(param_specs, call_args):
                if is_tensor:
                    if not isinstance(arg_value, pypto.Tensor):
                        raise ParserError(
                            node,
                            TypeError(f"Expected tensor argument for {param_name}, got {type(arg_value)}"),
                        )
                    self.context.add(param_name, arg_value)
                else:
                    if annotation is bool and not isinstance(arg_value, bool):
                        raise ParserError(
                            node,
                            TypeError(f"Expected bool argument for {param_name}, got {type(arg_value)}"),
                        )
                    if annotation is int and not isinstance(arg_value, (int, pypto.SymbolicScalar)):
                        raise ParserError(
                            node,
                            TypeError(f"Expected int argument for {param_name}, got {type(arg_value)}"),
                        )
                    self.context.add(param_name, arg_value)

            self.context.add("__func_name__", func_name)

            # Inline-execute the callee body.
            old_diag = self.diag
            try:
                try:
                    self.diag = Diagnostics(Source(func_obj))
                except Exception:
                    # Fallback to existing diagnostics if source extraction fails
                    pass

                try:
                    self._visit_body(body_nodes)
                except ParserError as e:
                    if not isinstance(e, RenderedParserError):
                        self.diag.error(e.node, str(e))
                    raise
            except RenderedParserError:
                self.diag = old_diag
                self.diag.info(node, f"In call to '{func_name}'")
                raise
            finally:
                self.diag = old_diag

            # No return value; nested functions use out parameter and out.move().
            return None

    def _visit_for(self, node: ast.For) -> Any:
        """The general for visiting method.

        Parameters
        ----------
        node : ast.For
            The AST for node.

        Returns
        -------
        res : Any
            The visiting result.

        Note
        ----
        For node structure:
            target: expr (loop variable)
            iter: expr (iterator expression, e.g., range(10))
            body: list[stmt] (loop body)
            orelse: list[stmt] (else clause, not supported)
        """
        # Check for unsupported else clause
        if node.orelse:
            raise ParserError(
                node,
                NotImplementedError(
                    "For-else clauses are not supported. Consider using a separate if statement after the loop."
                ),
            )

        # Extract loop variable information
        loop_vars = self._extract_loop_variables(node.target)

        # Evaluate iterator expression
        iter_expr = self._eval_expr(node.iter)

        # Handle different iterator types
        if isinstance(iter_expr, range):
            iter_expr = self._convert_range_iterator(node, iter_expr)

        if isinstance(iter_expr, (list, tuple)):
            self._handle_list_tuple_iterator(node, iter_expr, loop_vars)
        elif isinstance(iter_expr, Iterator):
            self._handle_pto_iterator(node, iter_expr, loop_vars)
        else:
            raise ParserError(
                node.iter,
                TypeError(f"Loop iterator must be range  Iterator, or list/tuple, but got {type(iter_expr).__name__}."),
            )

    def _extract_loop_variables(self, target: ast.expr) -> tuple[bool, str, list[str]]:
        """Extract loop variable information from target expression.

        Returns
        -------
        tuple[bool, str, list[str]]
            (is_tuple_unpack, loop_var_name, target_names)
        """
        is_tuple_unpack = isinstance(target, (ast.Tuple, ast.List))

        if isinstance(target, ast.Name):
            loop_var_name = target.id
            target_names = [loop_var_name]
        elif is_tuple_unpack:
            target_names = []
            for elt in target.elts:
                if not isinstance(elt, ast.Name):
                    raise ParserError(
                        elt,
                        TypeError(
                            f"Tuple unpacking in for loop only supports simple names, but got {type(elt).__name__}."
                        ),
                    )
                target_names.append(elt.id)
            loop_var_name = None  # Not used for tuple unpacking
        else:
            raise ParserError(
                target,
                TypeError(
                    f"Loop variable must be a simple name or tuple/list for unpacking, but got {type(target).__name__}."
                ),
            )

        return is_tuple_unpack, loop_var_name, target_names

    def _convert_range_iterator(self, node: ast.For, range_expr: range) -> list:
        """Convert range object to list for unified processing."""
        if (
            isinstance(range_expr.start, SymbolicScalar)
            or isinstance(range_expr.stop, SymbolicScalar)
            or isinstance(range_expr.step, SymbolicScalar)
        ):
            raise ParserError(
                node,
                TypeError("range() not support symbolic scalar yet, try use pypto.loop"),
            )

        return list(range(range_expr.start, range_expr.stop, range_expr.step))

    def _handle_list_tuple_iterator(
        self, node: ast.For, iter_expr: Union[list, tuple], loop_vars: tuple[bool, str, list[str]]
    ) -> None:
        """Handle list/tuple iterators by unrolling at compile time."""
        is_tuple_unpack, loop_var_name, target_names = loop_vars

        if len(iter_expr) == 0:
            raise ParserError(
                node.iter,
                ValueError("Empty list/tuple cannot be used as loop iterator."),
            )

        # Validate tuple unpacking compatibility
        if is_tuple_unpack and len(iter_expr) != len(target_names):
            raise ParserError(
                node.target,
                ValueError(f"Cannot unpack {len(iter_expr)} values into {len(target_names)} targets."),
            )

        # Unroll the loop at compile time
        # Use Python function-level scoping semantics, do not create a new frame
        # Variable lifetime is controlled by liveness analysis
        for item in iter_expr:
            self._assign_loop_variable(node.target, item, is_tuple_unpack, loop_var_name, target_names)
            self._visit_body(node.body)

        # Clean up variables after loop based on liveness analysis
        self._auto_cleanup_after_stmt(node)

    def _handle_pto_iterator(self, node: ast.For, iterator: Iterator, loop_vars: tuple[bool, str, list[str]]) -> None:
        """Handle PTO iterators with traditional loop processing."""
        is_tuple_unpack, loop_var_name, target_names = loop_vars

        # Use Python function-level scoping semantics, do not create a new frame
        # Variable lifetime is controlled by liveness analysis
        for loop_var in iterator:
            self._assign_loop_variable(node.target, loop_var, is_tuple_unpack, loop_var_name, target_names)
            self._visit_body(node.body)

        # Clean up variables after loop based on liveness analysis
        self._auto_cleanup_after_stmt(node)

    def _assign_loop_variable(
        self, target: ast.expr, value: Any, is_tuple_unpack: bool, loop_var_name: str, target_names: list[str]
    ) -> None:
        """Assign loop variable value to context."""
        if is_tuple_unpack:
            # Tuple unpacking validation and assignment
            if not isinstance(value, (tuple, list)):
                raise ParserError(
                    target,
                    TypeError(f"Expected tuple/list for unpacking, got {type(value).__name__}."),
                )
            if len(value) != len(target_names):
                raise ParserError(
                    target,
                    ValueError(f"Cannot unpack {len(value)} values into {len(target_names)} targets."),
                )
            self._assign_target(target, value)
        else:
            # Single variable assignment
            self.context.add(loop_var_name, value)

    def _assign_target(self, target: ast.expr, expr: Any) -> None:
        """Helper method to assign an expression to a target.

        Parameters
        ----------
        target : ast.expr
            The assignment target (Name, Tuple, List, or Subscript).
        expr : Any
            The value to assign.
        """
        if isinstance(target, ast.Name):
            # 检查是否试图重新赋值输入 tensor
            values_stack = self.context.name2value.get(target.id, [])
            if values_stack:
                existing_value = values_stack[-1]
                if isinstance(existing_value, pypto.Tensor) and getattr(existing_value, 'is_input', False):
                    raise ParserError(
                        target,
                        ValueError(
                            f"Input tensor '{target.id}' cannot be reassigned. "
                            f"Use subscript assignment ('{target.id}[:] = ...') "
                            f"or '{target.id}.move(...)' instead."
                        ),
                    )

            # Set the tensor name if expr is a tensor
            if isinstance(expr, pypto.Tensor):
                expr.name = target.id
            self.context.add(target.id, expr)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(expr, (list, tuple)):
                raise ParserError(
                    target,
                    TypeError(f"Cannot unpack non-sequence type {type(expr).__name__}"),
                )
            if len(target.elts) != len(expr):
                raise ParserError(
                    target,
                    ValueError(f"Cannot unpack {len(expr)} values into {len(target.elts)} targets"),
                )
            for t, e in zip(target.elts, expr):
                self._assign_target(t, e)
        elif isinstance(target, ast.Subscript):
            set_source_location(filename=self.source_name(), lineno=target.lineno)
            # This handles in-place tensor updates using Python's subscript syntax.
            # Evaluate the value (e.g., b) to get the tensor being assigned to
            tensor = self._eval_expr(target.value)
            # Evaluate the slice (e.g., : or 0) to get the slice/index object
            slice_obj = self._eval_expr(target.slice)
            # Perform the assignment using __setitem__, which translates to
            # the appropriate PTO IR operation for tensor element/slice updates
            tensor[slice_obj] = expr
            clear_source_location()
        else:
            raise ParserError(
                target,
                TypeError(f"Assignment target must be a name, tuple, or subscript, but got {type(target).__name__}."),
            )

    def _visit_assign(self, node: ast.Assign) -> None:
        """The general assign visiting method.

        Parameters
        ----------
        node : ast.Assign
            The AST assign node.

        Returns
        -------
        res : None
            The visiting result. None.

        Note
        ----
        Assign node structure:
            targets: list[expr]
            value: expr
        """
        expr = self._visit_expr(node.value)

        for target in node.targets:
            self._assign_target(target, expr)

    def _visit_ann_assign(self, node: ast.AnnAssign) -> Any:
        """The general annotated assign visiting method.

        Parameters
        ----------
        node : ast.Assign
            The AST annotated assign node.

        Returns
        -------
        res : Any
            The visiting result.

        Note
        ----
        AnnAssign node structure:
            target: expr
            annotation: expr
            value: Optional[expr]
        """
        # Reuse the assign visiting method to visit the annotated assign node.
        return self._visit_assign(node)

    def _visit_aug_assign(self, node: ast.AugAssign) -> None:
        """The general augmented assign visiting method.

        This method handles compound assignment statements like +=, -=, *=, etc.
        It converts them to equivalent binary operations and assignments.

        Parameters
        ----------
        node : ast.AugAssign
            The AST augmented assign node.

        Returns
        -------
        res : None
            The visiting result. None.

        Note
        ----
        AugAssign node structure:
            target: expr
            op: operator (Add, Sub, Mult, Div, etc.)
            value: expr
        """
        # Evaluate the value expression first
        value_expr = self._eval_expr(node.value)

        # Handle different target types
        if isinstance(node.target, ast.Name):
            # For Name targets (e.g., out += y), get the value directly from context
            # since node.target has Store context and cannot be evaluated with _eval_expr
            var_values = self.context.get()
            if node.target.id not in var_values:
                raise ParserError(
                    node.target,
                    NameError(f"name '{node.target.id}' is not defined"),
                )
            target_value = var_values[node.target.id]
        elif isinstance(node.target, ast.Subscript):
            set_source_location(filename=self.source_name(), lineno=node.target.lineno)
            # For Subscript targets (e.g., a[i] += y), evaluate the tensor and slice/index
            tensor = self._eval_expr(node.target.value)
            slice_obj = self._eval_expr(node.target.slice)

            # Get the current value from the subscript
            target_value = tensor[slice_obj]
            clear_source_location()
        else:
            raise ParserError(
                node.target,
                NotImplementedError(f"Augmented assignment target type {type(node.target).__name__} is not supported."),
            )

        # Perform the binary operation based on the operator type
        if isinstance(node.op, ast.Add):
            result = target_value + value_expr
        elif isinstance(node.op, ast.Sub):
            result = target_value - value_expr
        elif isinstance(node.op, ast.Mult):
            result = target_value * value_expr
        elif isinstance(node.op, ast.Div):
            result = target_value / value_expr
        elif isinstance(node.op, ast.FloorDiv):
            result = target_value // value_expr
        elif isinstance(node.op, ast.Mod):
            result = target_value % value_expr
        elif isinstance(node.op, ast.Pow):
            result = target_value**value_expr
        elif isinstance(node.op, ast.LShift):
            result = target_value << value_expr
        elif isinstance(node.op, ast.RShift):
            result = target_value >> value_expr
        elif isinstance(node.op, ast.BitAnd):
            result = target_value & value_expr
        elif isinstance(node.op, ast.BitOr):
            result = target_value | value_expr
        elif isinstance(node.op, ast.BitXor):
            result = target_value ^ value_expr
        elif isinstance(node.op, ast.MatMult):
            result = target_value @ value_expr
        else:
            raise ParserError(
                node,
                NotImplementedError(f"Augmented assignment operator {type(node.op).__name__} is not supported."),
            )

        # Assign the result back to the target
        self._assign_target(node.target, result)

    def _visit_expr(self, node: ast.Expr) -> Any:
        """The general expression visiting method.

        Parameters
        ----------
        node : ast.Expr
            The AST expression node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        if isinstance(node, ast.Expr):
            return self._eval_expr(node.value)
        return self._eval_expr(node)

    def _visit_if(self, node: ast.If) -> Any:
        """The general if visiting method.

        Parameters
        ----------
        node : ast.If
            The AST if node.

        Returns
        -------
        res : Any
            The visiting result.

        Note
        ----
        If node structure:
            test: expr (condition expression)
            body: list[stmt] (if body)
            orelse: list[stmt] (else/elif body)
        """
        # Evaluate the test condition
        test_expr = self._eval_expr(node.test)

        if isinstance(test_expr, pypto.SymbolicScalar):
            cond = pypto.cond(test_expr, file=self.source_name(), lineno=node.lineno)
        elif isinstance(test_expr, bool):
            cond = test_expr
        else:
            raise ParserError(
                node.test,
                TypeError(f"Test condition must be a symbolic scalar or boolean, but got {type(test_expr).__name__}."),
            )

        # Execute the if statement using the condition as a context manager
        if cond:
            # Visit the if body
            self._visit_body(node.body)
        else:
            # Visit the else body (if it exists)
            if node.orelse:
                self._visit_body(node.orelse)

    def _visit_return(self, node: ast.Return) -> Any:
        """Forbid return statements; use out parameter and out.move() instead."""
        raise ParserError(
            node,
            ValueError(
                "Return statements are not allowed. Use an out parameter and out.move(...) to write results instead."
            ),
        )

    def _visit_delete(self, node: ast.Delete) -> None:
        """The general delete visiting method.

        Parameters
        ----------
        node : ast.Delete
            The AST delete node.

        Returns
        -------
        res : None
            The visiting result. None.

        Note
        ----
        Delete node structure:
            targets: list[expr]
        """
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Delete a simple variable
                try:
                    self.context.delete(target.id)
                except NameError as e:
                    raise ParserError(target, e) from e
                except ValueError as e:
                    raise ParserError(target, e) from e
            else:
                raise ParserError(
                    target,
                    TypeError(f"Delete target must be a name, but got {type(target).__name__}."),
                )

    def _is_tensor_var(self, var_name: str) -> bool:
        """Check if a variable is a pypto.Tensor type.

        Parameters
        ----------
        var_name : str
            Variable name to check.

        Returns
        -------
        bool
            True if the variable is a pypto.Tensor, False otherwise.
        """
        try:
            values = self.context.name2value.get(var_name, [])
            return bool(values and isinstance(values[-1], pypto.Tensor))
        except Exception:
            return False

    def _filter_tensor_vars(self, var_names: set[str]) -> set[str]:
        """Filter variables that are pypto.Tensor type.

        Only pypto.Tensor variables need memory cleanup through liveness analysis.
        Other Python objects (int, str, list, dict, etc.) are managed by
        Python's garbage collector and do not require explicit cleanup.

        Parameters
        ----------
        var_names : set[str]
            Set of variable names to filter.

        Returns
        -------
        tensor_vars : set[str]
            Set of variable names that are pypto.Tensor type.
        """
        tensor_vars = set()
        for var_name in var_names:
            if self._is_tensor_var(var_name):
                tensor_vars.add(var_name)
        return tensor_vars

    def _auto_cleanup_after_stmt(self, stmt: ast.stmt) -> None:
        """Automatically cleanup variables after statement if enabled.

        Parameters
        ----------
        stmt : ast.stmt
            The statement that was just visited.
        """

        stmt_id = id(stmt)
        if stmt_id in self.delete_after:
            vars_to_delete = self.delete_after[stmt_id]
            # Filter: only delete pypto.Tensor type variables
            tensor_vars = self._filter_tensor_vars(vars_to_delete)
            if tensor_vars:
                self.context.mark_for_deletion(tensor_vars)
                self.context.cleanup_marked()

    def _visit_assert(self, node: ast.Assert) -> None:
        """The general assert visiting method.

        Parameters
        ----------
        node : ast.Assert
            The AST assert node.

        Returns
        -------
        res : None
            The visiting result. None.

        Raises
        ------
        ParserError
            If the assert condition can be statically evaluated to False.

        Note
        ----
        Assert node structure:
            test: expr
            msg: Optional[expr]

        This implementation provides true assertion capability:
        - For statically evaluable conditions, checks at compile time
        - For dynamic conditions, generates runtime assertion code
        """
        # Evaluate the assert condition
        test_result = self._visit_expr(node.test)

        # Prepare the error message
        if node.msg:
            msg_result = self._visit_expr(node.msg)
            error_msg = str(msg_result) if msg_result is not None else "Assertion failed"
        else:
            error_msg = "Assertion failed"

        try:
            if not bool(test_result):
                raise ParserError(node, f"AssertionError: {error_msg}")
        except (TypeError, ValueError) as e:
            raise ParserError(
                TypeError(node, f"Cannot convert assert condition of type {type(test_result).__name__} to boolean."),
            ) from e
