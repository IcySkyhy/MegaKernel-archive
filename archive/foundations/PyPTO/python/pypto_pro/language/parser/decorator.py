# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Decorator for parsing DSL functions to IR."""

from __future__ import annotations

__all__ = ["function", "inline", "program", "vector_function"]


import ast
from collections.abc import Callable
import inspect
import linecache
import logging
import sys
import textwrap
from typing import TypeVar

from pypto.pypto_impl import ir

from ._ast_parser import ASTParser
from .diagnostics import ParserError, ParserSyntaxError

logger = logging.getLogger(__name__)


def _strip_self_parameter(func_def: ast.FunctionDef) -> ast.FunctionDef:
    """Return a copy of func_def with the leading 'self' parameter removed.

    If the first parameter is not 'self', returns the original node unchanged.

    Args:
        func_def: AST FunctionDef node

    Returns:
        FunctionDef with 'self' stripped, or the original if no 'self' found
    """
    if not func_def.args.args or func_def.args.args[0].arg != "self":
        return func_def

    new_args = ast.arguments(
        posonlyargs=func_def.args.posonlyargs,
        args=func_def.args.args[1:],
        vararg=func_def.args.vararg,
        kwonlyargs=func_def.args.kwonlyargs,
        kw_defaults=func_def.args.kw_defaults,
        kwarg=func_def.args.kwarg,
        defaults=func_def.args.defaults,
    )
    new_func_def = ast.FunctionDef(
        name=func_def.name,
        args=new_args,
        body=func_def.body,
        decorator_list=func_def.decorator_list,
        returns=func_def.returns,
        type_comment=func_def.type_comment,
        lineno=func_def.lineno,
        col_offset=func_def.col_offset,
    )
    if hasattr(func_def, "end_lineno"):
        new_func_def.end_lineno = func_def.end_lineno
    if hasattr(func_def, "end_col_offset"):
        new_func_def.end_col_offset = func_def.end_col_offset
    return new_func_def


def _calculate_col_offset(source_lines: list[str]) -> int:
    """Calculate the column offset (indentation) of the first non-empty line.

    This is needed because ast.parse() requires code starting at column 0,
    but we need to report errors at the correct column in the original file.

    Args:
        source_lines: List of source code lines

    Returns:
        Column offset (number of leading spaces/tabs in first non-empty line)
    """
    for line in source_lines:
        if line.strip():  # Skip empty lines
            return len(line) - len(line.lstrip())
    return 0


def _parse_ast_tree(source_code: str, entity_type: str) -> ast.AST:
    """Parse source code into an AST tree with proper error handling.

    Args:
        source_code: Python source code to parse
        entity_type: Type of entity being parsed ("function" or "class") for error messages

    Returns:
        Parsed AST tree

    Raises:
        ParserSyntaxError: If the source code has syntax errors
    """
    try:
        return ast.parse(source_code)
    except SyntaxError as e:
        raise ParserSyntaxError(
            f"Failed to parse {entity_type} source: {e.msg}",
            hint=f"Check for Python syntax errors in your {entity_type}",
        ) from e


TypeASTNode = TypeVar("TypeASTNode", ast.FunctionDef, ast.ClassDef)


def _find_ast_node(tree: ast.AST, node_type: type[TypeASTNode], name: str, entity_type: str) -> TypeASTNode:
    """Find a specific AST node by type and name.

    Args:
        tree: AST tree to search
        node_type: Type of AST node to find (ast.FunctionDef or ast.ClassDef)
        name: Name of the node to find
        entity_type: Type of entity for error messages ("function" or "class")

    Returns:
        Found AST node

    Raises:
        ParserSyntaxError: If the node cannot be found
    """
    for node in ast.walk(tree):
        if isinstance(node, node_type) and node.name == name:
            return node

    raise ParserSyntaxError(
        f"Could not find {entity_type} definition for {name}",
        hint=f"Ensure the {entity_type} is properly defined",
    )


def _attach_source_lines_to_error(error: ParserError, source_file: str, source_lines_raw: list[str]) -> None:
    """Attach source lines to a ParserError if not already present.

    Args:
        error: ParserError to attach source lines to
        source_file: Path to the source file
        source_lines_raw: Raw source lines as fallback
    """
    if error.source_lines is None:
        # Use the span's filename if it differs (e.g., error in an inline function)
        target_file = source_file
        if error.span and isinstance(error.span, dict):
            span_file = error.span.get("filename")
            if span_file and span_file != source_file:
                target_file = span_file
        try:
            with open(target_file, encoding="utf-8") as f:
                error.source_lines = f.read().split("\n")
        except Exception:
            # Fallback to the raw source lines if we can't read the file
            error.source_lines = source_lines_raw


def _is_pl_function_decorator(decorator: ast.AST) -> bool:
    """Return whether one decorator node denotes pl.function/function."""
    if isinstance(decorator, ast.Attribute):
        return decorator.attr == "function"
    if isinstance(decorator, ast.Name):
        return decorator.id == "function"
    if isinstance(decorator, ast.Call):
        return _is_function_decorator_call(decorator)
    return False


def _has_pl_function_decorator(node: ast.FunctionDef) -> bool:
    """Check if a function node has @pl.function decorator.

    Args:
        node: AST FunctionDef node to check

    Returns:
        True if the node has @pl.function decorator
    """
    for decorator in node.decorator_list:
        if _is_pl_function_decorator(decorator):
            return True
    return False


def _extract_function_type_from_decorator(node: ast.FunctionDef) -> ir.FunctionType:
    """Extract function type from @pl.function(type=...) decorator.

    Searches through the function's decorators to find @pl.function(type=...)
    and extracts the FunctionType value. If no type parameter is found,
    returns FunctionType.Opaque as the default.

    Args:
        node: AST FunctionDef node to extract function type from

    Returns:
        FunctionType extracted from decorator, or FunctionType.Opaque if not specified
    """
    for decorator in node.decorator_list:
        # Look for @pl.function(...) or @function(...) with call syntax
        if not _is_function_decorator_call(decorator):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "type":
                return _function_type_from_ast_value(keyword.value)

    # Default to Opaque if no type specified
    return ir.FunctionType.Opaque


def _is_function_decorator_call(decorator: ast.AST) -> bool:
    """Return whether a decorator is @pl.function(...) or @function(...)."""
    if not isinstance(decorator, ast.Call):
        return False
    decorator_func = decorator.func
    if isinstance(decorator_func, ast.Attribute):
        return decorator_func.attr == "function"
    return isinstance(decorator_func, ast.Name) and decorator_func.id == "function"


def _function_type_from_ast_value(value: ast.AST) -> ir.FunctionType:
    """Resolve FunctionType from a decorator keyword AST value."""
    if not isinstance(value, ast.Attribute):
        return ir.FunctionType.Opaque
    mapping = {
        "InCore": ir.FunctionType.InCore,
        "Opaque": ir.FunctionType.Opaque,
    }
    return mapping.get(value.attr, ir.FunctionType.Opaque)


def _is_class_method(fn: Callable) -> bool:
    """Check if a function is a method inside a class (not a standalone function).

    This performs strict validation to determine if a function with 'self' as the first
    parameter is actually defined inside a class, rather than a standalone function that
    just happens to have 'self' as a parameter name.

    Args:
        fn: Function to check

    Returns:
        True if the function is a method inside a class
    """
    # Check if first parameter is 'self'
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        if not (params and params[0] == "self"):
            return False
    except (ValueError, TypeError):
        return False

    # Check if __qualname__ indicates this is a method (contains a dot)
    qualname = fn.__qualname__
    if "." not in qualname:
        # No dot in qualname means it's a standalone function, not a method
        return False

    # Verify it has indentation (defined inside a class, not at module level)
    try:
        source_lines_raw, _ = inspect.getsourcelines(fn)
        col_offset = _calculate_col_offset(source_lines_raw)
        if col_offset > 0:
            # This is an indented method inside a class
            return True
    except (OSError, TypeError):
        # If we can't get source lines, assume it's a method based on qualname
        # (This can happen with dynamically generated code)
        return True

    return False


def _get_source_file(entity: Callable | type) -> str:
    """Get source filename for an entity, with fallback to code object attributes.

    Args:
        entity: Function or class to get source file for

    Returns:
        Source filename string
    """
    try:
        return inspect.getfile(entity)
    except (OSError, TypeError):
        pass

    # Fallback: extract from code object
    if callable(entity) and hasattr(entity, "__code__"):
        return entity.__code__.co_filename

    # For classes, find a method with a code object
    if isinstance(entity, type):
        for attr in entity.__dict__.values():
            if callable(attr) and hasattr(attr, "__code__"):
                return attr.__code__.co_filename

    return "<unknown>"


def _find_entity_in_source(
    all_lines: list[str], name: str, entity_type: str, start_line_hint: int | None = None
) -> tuple[list[str], int] | None:
    """Find an entity definition in source lines using AST parsing.

    Args:
        all_lines: All source lines from the file
        name: Name of the entity to find
        entity_type: "function" or "class"
        start_line_hint: Optional line number to disambiguate entities with the same name

    Returns:
        Tuple of (source_lines, starting_line_1based) or None if not found
    """
    source_text = "".join(all_lines)
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None

    node_type = ast.FunctionDef if entity_type == "function" else ast.ClassDef
    candidates = [node for node in ast.walk(tree) if isinstance(node, node_type) and node.name == name]

    if not candidates:
        return None

    if len(candidates) == 1:
        node = candidates[0]
    elif start_line_hint is not None:
        # Disambiguate using the code object's line number
        node = min(candidates, key=lambda n: abs(n.lineno - start_line_hint))
    else:
        node = candidates[0]

    # Start from the first decorator line if present
    start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    end_line = node.end_lineno or node.lineno
    # Lines are 1-based in AST
    source_lines = all_lines[start_line - 1:end_line]
    return source_lines, start_line


def _source_from_inspect(entity: Callable | type) -> tuple[str, list[str], int] | None:
    """Get source information via inspect, returning None when unavailable."""
    try:
        source_file = inspect.getfile(entity)
        source_lines_raw, starting_line = inspect.getsourcelines(entity)
        return source_file, source_lines_raw, starting_line
    except (OSError, TypeError):
        return None


def _source_from_linecache(
    source_file: str,
    name: str,
    entity_type: str,
    start_line_hint: int | None,
) -> tuple[str, list[str], int] | None:
    """Get source information from linecache."""
    all_lines = linecache.getlines(source_file)
    if not all_lines:
        return None
    result = _find_entity_in_source(all_lines, name, entity_type, start_line_hint)
    return None if result is None else (source_file, result[0], result[1])


def _restore_linecache_entry(prev_entry) -> None:
    """Restore the synthetic <string> linecache entry used for python -c parsing."""
    if prev_entry is not None:
        linecache.cache["<string>"] = prev_entry
    else:
        linecache.cache.pop("<string>", None)


def _source_from_python_c(
    source_file: str,
    name: str,
    entity_type: str,
    start_line_hint: int | None,
) -> tuple[str, list[str], int] | None:
    """Get source information from sys.orig_argv for python -c invocations."""
    if source_file != "<string>" or not hasattr(sys, "orig_argv"):
        return None
    try:
        c_index = sys.orig_argv.index("-c")
    except ValueError:
        return None
    if c_index + 1 >= len(sys.orig_argv):
        return None

    code_str = sys.orig_argv[c_index + 1]
    code_lines = code_str.splitlines(keepends=True)
    prev_entry = linecache.cache.get("<string>")
    linecache.cache["<string>"] = (len(code_str), None, code_lines, "<string>")
    try:
        result = _find_entity_in_source(code_lines, name, entity_type, start_line_hint)
        return None if result is None else (source_file, result[0], result[1])
    finally:
        _restore_linecache_entry(prev_entry)


def _get_source_info(entity: Callable | type, entity_type: str) -> tuple[str, list[str], int]:
    """Get source file, source lines, and starting line for an entity.

    Tries multiple strategies:
    1. Standard inspect.getsourcelines()
    2. linecache fallback (handles IPython, pre-populated cache)
    3. sys.orig_argv for `python -c` invocations
    4. Clear error with actionable hint

    Args:
        entity: Function or class to get source for
        entity_type: "function" or "class"

    Returns:
        Tuple of (source_file, source_lines_raw, starting_line)

    Raises:
        ParserSyntaxError: If source cannot be retrieved by any strategy
    """
    name = entity.__name__ if hasattr(entity, "__name__") else str(entity)

    # Get a line number hint from the code object to disambiguate same-name entities
    start_line_hint: int | None = None
    if callable(entity) and hasattr(entity, "__code__"):
        start_line_hint = entity.__code__.co_firstlineno

    # Strategy 1: Standard inspect
    inspect_result = _source_from_inspect(entity)
    if inspect_result is not None:
        return inspect_result

    # Get source file via fallback for strategies 2-3
    source_file = _get_source_file(entity)

    # Strategy 2: linecache fallback
    linecache_result = _source_from_linecache(source_file, name, entity_type, start_line_hint)
    if linecache_result is not None:
        return linecache_result

    # Strategy 3: sys.orig_argv for `python -c`
    python_c_result = _source_from_python_c(source_file, name, entity_type, start_line_hint)
    if python_c_result is not None:
        return python_c_result

    # Strategy 4: Clear error
    raise ParserSyntaxError(
        f"Cannot retrieve source code for {entity_type} '{name}'",
        hint="Save your code to a .py file, or use pl.parse() / pl.parse_program() to parse from a string",
    )


def function(
    fn: Callable | None = None,
    *,
    type: ir.FunctionType = ir.FunctionType.Opaque,
    strict_ssa: bool = False,
) -> ir.Function:
    """Decorator that parses a DSL function and returns IR Function.

    This decorator analyzes the decorated function's AST, parses the DSL
    constructs (type annotations, pl.range, etc.), and builds
    an IR Function object.

    Args:
        fn: Python function decorated with @pl.function
        type: Function type (Opaque, InCore, or Helper)
        strict_ssa: If True, enforce SSA (single assignment per variable).
                   If False (default), allow variable reassignment (non-SSA mode).

    Returns:
        IR Function object (or decorator if used with parameters)

    Example:
        >>> @pl.function
        ... def my_func(x: pl.Tensor[[64, 128], pl.DT_FP16]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        ...     result = pl.tensor.create_tensor([64, 128], dtype=pl.DT_FP32)
        ...     return result
    """
    # Capture the caller's scope for variable resolution in type annotations
    caller_frame = inspect.currentframe().f_back
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    def _decorator(f: Callable) -> ir.Function:
        # Check if this is a method inside a class decorated with @pl.program
        # If so, return the original function - it will be parsed by @pl.program decorator
        if _is_class_method(f):
            # Don't parse now - let @pl.program handle it with proper global_vars context
            return f

        # Get source code and file information
        source_file, source_lines_raw, starting_line = _get_source_info(f, "function")
        source_code = "".join(source_lines_raw)

        # Calculate indentation offset before dedenting
        col_offset = _calculate_col_offset(source_lines_raw)

        # Remove leading indentation so ast.parse() can parse it
        source_code = textwrap.dedent(source_code)

        # Use dedented source lines so column offsets align with AST
        source_lines = source_code.split("\n")

        # Calculate line offset (AST line numbers are 1-based, but we want to map to original file)
        line_offset = starting_line - 1

        try:
            tree = _parse_ast_tree(source_code, "function")
            func_def = _find_ast_node(tree, ast.FunctionDef, f.__name__, "function")

            # Create parser and parse the function. Field names need a side table during
            # parse (for attribute-access resolution); create one for this standalone parse.
            parser = ASTParser(
                source_file,
                source_lines,
                ir.SectionKind.Vector,
                line_offset,
                col_offset,
                strict_ssa=strict_ssa,
                closure_vars=closure_vars,
                debug_info=ir.IRDebugInfo(),
            )

            try:
                ir_func = parser.parse_function(func_def, func_type=type)
            except ParserError:
                # Re-raise ParserError as-is, it already has source lines
                logger.debug("Re-raising ParserError in _decorator", exc_info=True)
                raise
            except Exception as e:
                # Wrap unexpected exceptions as ParserError
                raise ParserSyntaxError(
                    f"Failed to parse function '{f.__name__}': {e}",
                    hint="Check your function definition for errors",
                ) from e

            return ir_func

        except ParserError as e:
            # Attach source lines if not already present
            _attach_source_lines_to_error(e, source_file, source_lines_raw)
            # Always raise the exception to the caller
            raise

    if fn is None:
        return _decorator
    else:
        # Called without parameters: @pl.function
        return _decorator(fn)


_SIMT_FUNCTION_MARKER = "_pypto_simt_function"
_SIMT_MAX_THREADS_ATTR = "_pypto_simt_max_threads"


def simt_function(fn: Callable | None = None, *, max_threads: int | None = None) -> Callable:
    """Mark a callable for delayed SIMT parsing at its call site."""
    if fn is None and max_threads is None:
        raise TypeError("@pl.simt.function requires max_threads or a directly decorated function")

    def decorate(func: Callable) -> Callable:
        setattr(func, _SIMT_FUNCTION_MARKER, True)
        setattr(func, _SIMT_MAX_THREADS_ATTR, max_threads)
        return func

    return decorate(fn) if fn is not None else decorate


def is_simt_function(fn: Callable) -> bool:
    """Return whether *fn* is marked as a SIMT function template."""
    return bool(getattr(fn, _SIMT_FUNCTION_MARKER, False))


def get_simt_max_threads(fn: Callable) -> int | None:
    """Return the launch bound recorded by @pl.simt.function."""
    return getattr(fn, _SIMT_MAX_THREADS_ATTR, None)


def inline(fn: Callable) -> Callable:
    """Deprecated: @pl.inline has been removed. Use @pl.fn or pass as an annotated callable."""
    import warnings

    warnings.warn(
        "@pl.inline is deprecated and will be removed. "
        "Use @pl.fn with type annotations, or pass the function as an annotated callable.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fn


def vector_function(fn: Callable) -> Callable:
    """Decorator marking a function as a VF (Vector Function) body.

    The entire function body is implicitly wrapped in a VF section scope
    (equivalent to the former ``with pl.section_vf():`` block).  Only
    ``vf.*`` operations are allowed inside.

    Tile parameters are inferred from the call site (like implicit helpers);
    no scalar type annotations are required for tile-valued parameters.

    Args:
        fn: Python function whose body contains only ``vf.*`` operations.

    Returns:
        The original function with ``_pypto_vector_function`` attribute set.
        At a call site inside a ``@pl.kernel`` the parser detects the marker
        and compiles the body inside an implicit ``ir.SectionKind.VF`` scope.

    Example:
        >>> @pl.vector_function
        ... def vf_body(in_tile, out_tile):
        ...     preg = vf.create_mask(pattern="ALL", dtype=pl.FP32)
        ...     reg = vf.load_align(in_tile, 0)
        ...     reg = vf.add(reg, reg, preg)
        ...     vf.store_align(out_tile, reg, preg)
    """
    _mark_vector_function(fn)
    return fn


def _mark_vector_function(fn: Callable) -> None:
    """Mark *fn* as a VF (Vector Function) body.

    Sets the internal attribute that the parser checks at call sites to
    decide whether to compile the function body inside a VF section scope.
    """
    setattr(fn, "_pypto_vector_function", True)


def is_vector_function(fn: Callable) -> bool:
    """Return True if *fn* was decorated with ``@pl.vector_function``."""
    return getattr(fn, "_pypto_vector_function", False)


def _collect_program_func_defs(class_def: ast.ClassDef, class_name: str) -> tuple[set[str], list[ast.FunctionDef]]:
    """Collect @pl.function method definitions from a program class."""
    global_vars = set()
    func_defs = []
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and _has_pl_function_decorator(node):
            global_vars.add(node.name)
            func_defs.append(node)
    if not func_defs:
        raise ParserSyntaxError(
            f"Class '{class_name}' contains no @pl.function decorated methods",
            hint="Add at least one method decorated with @pl.function",
        )
    return global_vars, func_defs


def _merge_program_external_funcs(
    external_functions: dict[str, ir.Function],
    parser_external_funcs: dict[str, ir.Function],
) -> None:
    """Merge external functions discovered while parsing one program method."""
    for ext_name, ext_func in parser_external_funcs.items():
        if ext_name in external_functions and external_functions[ext_name] is not ext_func:
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{ext_name}'",
                hint="External functions must have unique names; rename one of the functions",
            )
        external_functions[ext_name] = ext_func


def _parse_one_program_function(
    func_def: ast.FunctionDef,
    source_file: str,
    source_lines: list[str],
    line_offset: int,
    col_offset: int,
    global_vars: set[str],
    gvar_to_func: dict[str, ir.Function],
    strict_ssa: bool,
    closure_vars: dict,
    debug_info: ir.IRDebugInfo,
) -> tuple[ir.Function, dict[str, ir.Function]]:
    """Parse one @pl.function method from a @pl.program class."""
    func_type = _extract_function_type_from_decorator(func_def)
    func_def_to_parse = _strip_self_parameter(func_def)
    parser = ASTParser(
        source_file,
        source_lines,
        ir.SectionKind.Vector,
        line_offset,
        col_offset,
        global_vars=global_vars,
        gvar_to_func=gvar_to_func,
        strict_ssa=strict_ssa,
        closure_vars=closure_vars,
        debug_info=debug_info,
    )
    try:
        ir_func = parser.parse_function(func_def_to_parse, func_type=func_type)
    except SyntaxError as e:
        logger.debug("SyntaxError while parsing function '%s'", func_def_to_parse.name)
        raise ParserSyntaxError(
            f"Failed to parse function '{func_def_to_parse.name}': {e.msg}",
            hint="Check for Python syntax errors in your function definition",
        ) from e
    return ir_func, parser.external_funcs


def _parse_program_functions(
    func_defs: list[ast.FunctionDef],
    source_file: str,
    source_lines: list[str],
    line_offset: int,
    col_offset: int,
    global_vars: set[str],
    strict_ssa: bool,
    closure_vars: dict,
    debug_info: ir.IRDebugInfo,
) -> tuple[list[ir.Function], dict[str, ir.Function]]:
    """Parse all @pl.function methods and merge discovered external functions."""
    functions = []
    gvar_to_func: dict[str, ir.Function] = {}
    external_functions: dict[str, ir.Function] = {}
    for func_def in func_defs:
        ir_func, parser_external_funcs = _parse_one_program_function(
            func_def,
            source_file,
            source_lines,
            line_offset,
            col_offset,
            global_vars,
            gvar_to_func,
            strict_ssa,
            closure_vars,
            debug_info,
        )
        functions.append(ir_func)
        gvar_to_func[ir_func.name] = ir_func
        _merge_program_external_funcs(external_functions, parser_external_funcs)
    return functions, external_functions


def program(cls: type | None = None, *, strict_ssa: bool = False) -> ir.Program:
    """Decorator that parses a class with @pl.function methods into a Program.

    The class should contain one or more methods decorated with @pl.function.
    Each method is parsed as a separate function and added to the program.
    Methods must have 'self' as the first parameter (standard Python syntax),
    which is automatically stripped from the IR.

    Args:
        cls: Class with @pl.function decorated methods
        strict_ssa: If True, enforce SSA (single assignment per variable).
                   If False (default), allow variable reassignment (non-SSA mode).

    Returns:
        IR Program object (or decorator if used with parameters)

    Example:
        >>> @pl.program
        ... class MyProgram:
        ...     @pl.function
        ...     def add(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        ...         result: pl.Tensor[[64], pl.DT_FP32] = pl.add(x, 1.0)
        ...         return result
        ...
        ...     @pl.function
        ...     def mul(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        ...         result: pl.Tensor[[64], pl.DT_FP32] = pl.mul(x, 2.0)
        ...         return result
        >>> # MyProgram is now an ir.Program object
    """

    # Capture the caller's scope for variable resolution in type annotations
    caller_frame = inspect.currentframe().f_back
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    def _decorator(c: type) -> ir.Program:
        source_file, source_lines_raw, starting_line = _get_source_info(c, "class")
        source_code = textwrap.dedent("".join(source_lines_raw))
        source_lines = source_code.split("\n")
        line_offset = starting_line - 1
        col_offset = _calculate_col_offset(source_lines_raw)

        try:
            tree = _parse_ast_tree(source_code, "class")
            class_def = _find_ast_node(tree, ast.ClassDef, c.__name__, "class")

            # Pass 1: Collect all @pl.function methods
            global_vars, func_defs = _collect_program_func_defs(class_def, c.__name__)

            # Pass 2: Parse each function body with function name set for cross-function calls
            # Build a map from function name to parsed functions as we go, so later functions
            # can use return type information from earlier functions
            debug_info = ir.IRDebugInfo()
            functions, external_functions = _parse_program_functions(
                func_defs,
                source_file,
                source_lines,
                line_offset,
                col_offset,
                global_vars,
                strict_ssa,
                closure_vars,
                debug_info,
            )

            # Combine internal and external functions
            all_functions = functions + list(external_functions.values())

            # Create Program with class name and span
            program_span = ir.Span(source_file, starting_line, col_offset)
            prog = ir.Program(all_functions, c.__name__, program_span, debug_info)

            return prog

        except ParserError as e:
            # Attach source lines if not already present
            _attach_source_lines_to_error(e, source_file, source_lines_raw)
            raise

    if cls is None:
        return _decorator
    else:
        # Called without parameters: @pl.program
        return _decorator(cls)
