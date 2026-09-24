# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Debug operations for PyPTO IR."""

from __future__ import annotations

__all__ = ["pto_assert", "dump_data", "printf", "trap"]

import ast
from collections.abc import Sequence
from typing import Any

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call, ConstBool, ConstInt, DataType, Expr, ScalarType, Span, TensorType, TileType

from .._utils import _get_span_or_capture, _normalize_expr, _to_make_tuple
from ._op_registry import OpSpec, op_impl, register_table

_PRINTF_FLAGS = set("-+ #0")
_INTEGER_CONVERSIONS = {"d", "i", "u", "x"}
_FLOAT_CONVERSIONS = {"f"}
_POINTER_CONVERSIONS = {"p"}
_SUPPORTED_CONVERSIONS = _INTEGER_CONVERSIONS | _FLOAT_CONVERSIONS | _POINTER_CONVERSIONS
_INTEGER_DTYPE_NAMES = ("INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64")
_INTEGER_DTYPES = tuple(getattr(DataType, name) for name in _INTEGER_DTYPE_NAMES if hasattr(DataType, name))

_SIGNED_INTEGER_DTYPES = tuple(
    getattr(DataType, dtype_name) for dtype_name in ("INT8", "INT16", "INT32", "INT64") if hasattr(DataType, dtype_name)
)

_UNSIGNED_INTEGER_DTYPES = tuple(
    getattr(DataType, dtype_name)
    for dtype_name in ("UINT8", "UINT16", "UINT32", "UINT64")
    if hasattr(DataType, dtype_name)
)


def _is_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsInt"):
        return dtype.IsInt()
    return dtype in _INTEGER_DTYPES


def _is_signed_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsSignedInt"):
        return dtype.IsSignedInt()
    return dtype in _SIGNED_INTEGER_DTYPES


def _is_unsigned_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsUnsignedInt"):
        return dtype.IsUnsignedInt()
    return dtype in _UNSIGNED_INTEGER_DTYPES


def _is_index_dtype(dtype: DataType) -> bool:
    return dtype == DataType.INDEX


def _is_bool_dtype(dtype: DataType) -> bool:
    return dtype == DataType.BOOL


def _normalize_location_flag(loc: bool, op_name: str) -> bool:
    if not isinstance(loc, bool):
        raise TypeError(f"debug.{op_name} requires bool loc flag, but got {type(loc).__name__}")
    return loc


def _scan_printf_format(format_str: str) -> list[str]:
    specs: list[str] = []
    i = 0
    while i < len(format_str):
        if format_str[i] != "%":
            i += 1
            continue
        if i + 1 < len(format_str) and format_str[i + 1] == "%":
            raise ValueError("printf does not support literal '%%'")

        j = i + 1
        while j < len(format_str) and format_str[j] in _PRINTF_FLAGS:
            j += 1
        while j < len(format_str) and format_str[j].isdigit():
            j += 1
        if j < len(format_str) and format_str[j] == ".":
            j += 1
            if j >= len(format_str) or not format_str[j].isdigit():
                raise ValueError("printf precision must be followed by digits")
            while j < len(format_str) and format_str[j].isdigit():
                j += 1
        if j >= len(format_str):
            raise ValueError("printf format string ends with an incomplete conversion")

        conversion = format_str[j]
        if conversion not in _SUPPORTED_CONVERSIONS:
            raise ValueError(f"printf does not support conversion '%{conversion}'")

        specs.append(format_str[i:j + 1])
        i = j + 1

    return specs


def _normalize_printf_args(
    args: Sequence[int | float | Expr | bool], actual_span: Span
) -> tuple[list[Expr], list[bool]]:
    normalized_args: list[Expr] = []
    raw_bool_args: list[bool] = []
    for arg in args:
        if isinstance(arg, Expr):
            normalized_args.append(arg)
            raw_bool_args.append(False)
        elif isinstance(arg, bool):
            normalized_args.append(ConstBool(arg, actual_span))
            raw_bool_args.append(True)
        else:
            normalized_args.append(
                _normalize_expr(arg, actual_span, int_dtype=DataType.INT64, float_dtype=DataType.FP32)
            )
            raw_bool_args.append(False)

    return normalized_args, raw_bool_args


def _validate_printf_arguments(
    format_str: str, normalized_args: Sequence[Expr], raw_bool_args: Sequence[bool], *, op_name: str
) -> None:
    specs = _scan_printf_format(format_str)
    if len(specs) != len(normalized_args):
        raise ValueError(f"{op_name} format expects {len(specs)} scalar arguments, but got {len(normalized_args)}")

    for idx, (spec, arg, is_raw_bool) in enumerate(zip(specs, normalized_args, raw_bool_args)):
        arg_type = arg.type
        conversion = spec[-1]

        # %p accepts pointer types (PtrType)
        if conversion == "p":
            if not isinstance(arg_type, _ir_core.PtrType):
                raise TypeError(
                    f"debug.{op_name} conversion '{spec}' requires pointer, but got {type(arg_type).__name__}"
                )
            continue

        if not isinstance(arg_type, ScalarType):
            raise TypeError(
                f"debug.{op_name} argument {idx} requires ScalarType input, but got {type(arg_type).__name__}"
            )
        scalar_type = arg_type

        if conversion in {"d", "i"} and not (
            _is_signed_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
        ):
            raise TypeError(
                f"debug.{op_name} conversion '{spec}' requires signed integer, bool, or index scalar, "
                f"but got {scalar_type.dtype}"
            )
        if conversion == "u" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
            or is_raw_bool
        ):
            raise TypeError(
                f"debug.{op_name} conversion '{spec}' requires unsigned integer, bool, or index scalar, "
                f"but got {scalar_type.dtype}"
            )
        if conversion == "x" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype) or _is_index_dtype(scalar_type.dtype)
        ):
            raise TypeError(
                f"debug.{op_name} conversion '{spec}' requires unsigned integer or index scalar, "
                f"but got {scalar_type.dtype}"
            )
        if conversion == "x" and is_raw_bool:
            raise TypeError(f"debug.{op_name} conversion '{spec}' does not support bool scalars")
        if conversion in _FLOAT_CONVERSIONS and scalar_type.dtype != DataType.FP32:
            raise TypeError(f"debug.{op_name} conversion '{spec}' requires FP32 scalar, but got {scalar_type.dtype}")


def dump_tensor(
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    loc: bool = False,
    span: Span | None = None,
    flag: str | None = None,
) -> Call:
    """Print a tensor or tensor window for debugging.

    Supported forms:
    - Full tensor dump: ``dump_tensor(tensor)``
    - Full dumps with dynamic tensor shapes
    - Window dump with dynamic ``offsets``
    - Window dump with dynamic ``shapes``

    Backend support:
    - full dump and window dump support dynamic ``offsets``/``shapes``

    Current limitation:
    - Windowed dumps still require the innermost tensor stride to be statically 1
    """
    actual_span = _get_span_or_capture(span)
    show_location = _normalize_location_flag(loc, "dump_tensor")
    if flag is not None and not isinstance(flag, str):
        raise TypeError(f"debug.dump_tensor requires str flag, but got {type(flag).__name__}")
    tensor_type = tensor.type
    if not isinstance(tensor_type, TensorType):
        raise TypeError(f"debug.dump_tensor requires TensorType input, but got {type(tensor_type).__name__}")

    if (offsets is None) != (shapes is None):
        raise ValueError("debug.dump_tensor offsets and shapes must be provided together")

    rank = len(tensor_type.shape)
    if offsets is None and shapes is None:
        offsets_tuple = _to_make_tuple([0] * rank, actual_span)
        shapes_tuple = _to_make_tuple(tensor_type.shape, actual_span)
    else:
        if tensor_type.tensor_view is not None and tensor_type.tensor_view.stride:
            last_stride = tensor_type.tensor_view.stride[-1]
            if not isinstance(last_stride, ConstInt):
                raise NotImplementedError(
                    "debug.dump_tensor windowed mode requires the innermost stride to be statically 1"
                )
            if last_stride.value != 1:
                raise ValueError(
                    f"debug.dump_tensor windowed mode requires innermost stride == 1, got {last_stride.value}"
                )
        offsets_tuple = _to_make_tuple(offsets, actual_span)
        shapes_tuple = _to_make_tuple(shapes, actual_span)

    if len(offsets_tuple.elements) != rank or len(shapes_tuple.elements) != rank:
        raise ValueError(
            f"debug.dump_tensor offsets/shapes must match tensor rank {rank}, got "
            f"{len(offsets_tuple.elements)} offsets and {len(shapes_tuple.elements)} shapes"
        )

    for idx, shape_expr in enumerate(shapes_tuple.elements):
        if isinstance(shape_expr, ConstInt) and shape_expr.value <= 0:
            raise ValueError(f"debug.dump_tensor shape at axis {idx} must be positive, got {shape_expr.value}")

    dump_kwargs: dict[str, Any] = {"show_location": show_location}
    if flag:
        dump_kwargs["dump_flag"] = flag
    return _ir_core.create_op_call(
        "debug.dump_tensor",
        [tensor, offsets_tuple, shapes_tuple],
        dump_kwargs,
        actual_span,
    )


def _validate_dump_offsets_shapes(
    offsets_tuple: _ir_core.MakeTuple,
    shapes_tuple: _ir_core.MakeTuple,
    rank: int,
    op_name: str,
) -> None:
    if len(offsets_tuple.elements) != rank or len(shapes_tuple.elements) != rank:
        raise ValueError(
            f"debug.{op_name} offsets/shapes must match tile rank {rank}, got "
            f"{len(offsets_tuple.elements)} offsets and {len(shapes_tuple.elements)} shapes"
        )
    for idx, shape_expr in enumerate(shapes_tuple.elements):
        if isinstance(shape_expr, ConstInt) and shape_expr.value <= 0:
            raise ValueError(f"debug.{op_name} shape at axis {idx} must be positive, got {shape_expr.value}")


def dump_tile(
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    workspace: Expr | None = None,
    loc: bool = False,
    span: Span | None = None,
    section: _ir_core.SectionKind | None = None,
    flag: str | None = None,
) -> Call:
    """Print a tile or tile window for debugging.

    Supported forms:
    - Full tile dump: ``dump_tile(tile)``
    - Full dump of tiles with dynamic valid-shape
    - Window dump with dynamic ``offsets``
    - Window dump with dynamic ``shapes``
    - Acc tile dump with ``workspace`` (GM temporary space)

    Backend support:
    - window ``shapes`` may be dynamic

    Current limitations:
    - Tile windows are currently 2D-only
    - Tile printing requires a printable ``vec`` tile; CCE window dumps handle
      this by copying the requested window into a temporary vec tile before
      printing
    - Acc tile window dump is not yet supported (full dump only)
    """
    actual_span = _get_span_or_capture(span)
    show_location = _normalize_location_flag(loc, "dump_tile")
    if flag is not None and not isinstance(flag, str):
        raise TypeError(f"debug.dump_tile requires str flag, but got {type(flag).__name__}")
    tile_type = tile.type
    if not isinstance(tile_type, TileType):
        raise TypeError(f"debug.dump_tile requires TileType input, but got {type(tile_type).__name__}")
    if section is not None:
        _validate_dump_memory_section(tile_type, section, actual_span)
    if (offsets is None) != (shapes is None):
        raise ValueError("debug.dump_tile offsets and shapes must be provided together")

    if workspace is not None:
        ws_type = workspace.type
        if not isinstance(ws_type, TensorType):
            raise TypeError(f"debug.dump_tile workspace must be TensorType, but got {type(ws_type).__name__}")

    rank = len(tile_type.shape)
    dump_kwargs: dict[str, Any] = {"show_location": show_location}
    if flag:
        dump_kwargs["dump_flag"] = flag
    if workspace is not None:
        if offsets is None and shapes is None:
            offsets_tuple = _to_make_tuple([0] * rank, actual_span)
            shapes_tuple = _to_make_tuple(tile_type.shape, actual_span)
        else:
            offsets_tuple = _to_make_tuple(offsets, actual_span)
            shapes_tuple = _to_make_tuple(shapes, actual_span)
        _validate_dump_offsets_shapes(offsets_tuple, shapes_tuple, rank, "dump_tile")
        return _ir_core.create_op_call(
            "debug.dump_tile",
            [tile, offsets_tuple, shapes_tuple, workspace],
            dump_kwargs,
            actual_span,
        )

    if offsets is None and shapes is None:
        return _ir_core.create_op_call("debug.dump_tile", [tile], dict(dump_kwargs), actual_span)

    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    _validate_dump_offsets_shapes(offsets_tuple, shapes_tuple, rank, "dump_tile")

    return _ir_core.create_op_call(
        "debug.dump_tile",
        [tile, offsets_tuple, shapes_tuple],
        dict(dump_kwargs),
        actual_span,
    )


def _validate_dump_memory_section(data_type, section, span: Span | None = None) -> None:
    """Validate that dumping ``data_type`` is legal in ``section``.

    GM tensors may be dumped from either section; on-chip tiles must live in
    Vec (UB) inside a Vector section and Acc (L0C) inside a Cube section.
    """
    if isinstance(data_type, TensorType):
        return
    if not isinstance(data_type, TileType):
        return
    mem = getattr(data_type, "target_memory", None)
    if mem is None and data_type.memref is not None:
        mem = data_type.memref.memory_space_
    if mem is None:
        return
    if section == _ir_core.SectionKind.Vector:
        if mem == _ir_core.MemorySpace.Vec:
            return
        from pypto_pro.language.parser.diagnostics import ParserSyntaxError

        raise ParserSyntaxError(
            f"pl.dump_data of a {mem.name} tile is not allowed in a Vector section; "
            "only Vec (UB) tiles (and GM tensors) can be dumped there",
            span=span,
            hint="Move pl.dump_data into a 'with pl.section_cube():' block, or dump a Vec/UB tile.",
        )
    if section == _ir_core.SectionKind.Cube:
        if mem == _ir_core.MemorySpace.Acc:
            return
        from pypto_pro.language.parser.diagnostics import ParserSyntaxError

        raise ParserSyntaxError(
            f"pl.dump_data of a {mem.name} tile is not allowed in a Cube section; "
            "only Acc (L0C) tiles (and GM tensors) can be dumped there",
            span=span,
            hint="Move pl.dump_data into a 'with pl.section_vector():' block, or dump an Acc/L0C tile.",
        )


def dump_data(
    data: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    workspace: Expr | None = None,
    loc: bool = False,
    span: Span | None = None,
    section: _ir_core.SectionKind | None = None,
    flag: str | None = None,
) -> Call:
    """Unified debug dump entry — dispatches to dump_tensor or dump_tile based on input type.

    If *data* is a TensorType (GM tensor), delegates to :func:`dump_tensor`.
    If *data* is a TileType (on-chip tile), delegates to :func:`dump_tile`.

    All parameters are forwarded to the underlying function unchanged.
    """
    data_type = data.type
    if flag is not None and not isinstance(flag, str):
        raise TypeError(f"debug.dump_data requires str flag, but got {type(flag).__name__}")
    if isinstance(data_type, TensorType):
        if workspace is not None:
            raise ValueError("debug.dump_data: workspace is only valid for Tile inputs, not Tensor")
        return dump_tensor(data, offsets, shapes, loc=loc, span=span, flag=flag)
    elif isinstance(data_type, TileType):
        return dump_tile(data, offsets, shapes, workspace=workspace, loc=loc, span=span, section=section, flag=flag)
    else:
        raise TypeError(f"debug.dump_data requires Tensor or Tile input, but got {type(data_type).__name__}")


def printf(format_str: str, *args: int | float | Expr, loc: bool = False, span: Span | None = None) -> Call:
    """Print scalar values using a compile-time format string."""
    actual_span = _get_span_or_capture(span)
    show_location = _normalize_location_flag(loc, "printf")
    if not isinstance(format_str, str):
        raise TypeError(f"debug.printf requires string format literal, but got {type(format_str).__name__}")

    normalized_args, raw_bool_args = _normalize_printf_args(args, actual_span)
    _validate_printf_arguments(format_str, normalized_args, raw_bool_args, op_name="printf")

    kwargs: dict[str, Any] = {"format": format_str, "show_location": show_location}
    return _ir_core.create_op_call("debug.printf", normalized_args, kwargs, actual_span)


def pto_assert(
    condition: bool | Expr,
    format_str: str | None = None,
    *args: int | float | Expr | bool,
    condition_text: str | None = None,
    loc: bool = False,
    span: Span | None = None,
) -> Call:
    """Abort execution when a scalar boolean condition is false."""
    actual_span = _get_span_or_capture(span)
    show_location = _normalize_location_flag(loc, "pto_assert")
    if isinstance(condition, Expr):
        condition_expr = condition
    elif isinstance(condition, bool):
        condition_expr = ConstBool(condition, actual_span)
    else:
        raise TypeError(f"debug.pto_assert requires a scalar bool condition, but got {type(condition).__name__}")

    condition_type = condition_expr.type
    if not isinstance(condition_type, ScalarType) or condition_type.dtype != DataType.BOOL:
        raise TypeError(
            "debug.pto_assert requires a scalar bool condition, "
            f"but got {type(condition_type).__name__}({getattr(condition_type, 'dtype', condition_type)})"
        )

    if condition_text is None:
        condition_text = "condition"
    if not isinstance(condition_text, str):
        raise TypeError(
            f"debug.pto_assert requires string condition_text metadata, but got {type(condition_text).__name__}"
        )

    normalized_args: list[Expr] = []
    if format_str is None:
        format_value = ""
    else:
        if not isinstance(format_str, str):
            raise TypeError(f"debug.pto_assert requires string literal format, but got {type(format_str).__name__}")
        normalized_args, raw_bool_args = _normalize_printf_args(args, actual_span)
        _validate_printf_arguments(format_str, normalized_args, raw_bool_args, op_name="pto_assert")
        format_value = format_str

    kwargs: dict[str, Any] = {
        "condition_text": condition_text,
        "format": format_value,
        "show_location": show_location,
    }
    return _ir_core.create_op_call("debug.assert", [condition_expr, *normalized_args], kwargs, actual_span)


def trap(*, span: Span | None = None) -> Call:
    """Abort execution by inserting a trap."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("debug.trap", [], {}, actual_span)


# ---------------------------------------------------------------------------
# Declarative op registration + special handlers
# ---------------------------------------------------------------------------

register_table(
    {
        "printf": OpSpec(builder=printf),
        "trap": OpSpec(builder=trap),
    }
)


@op_impl("dump_data")
def _parse_dump_data(self, call: ast.Call):
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(a) for a in call.args]
    kwargs = self.parse_op_kwargs(call)
    return dump_data(*args, section=self.target, **kwargs, span=span)


@op_impl("pto_assert")
def _parse_pto_assert(self, call: ast.Call):
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(a) for a in call.args]
    kwargs = self.parse_op_kwargs(call)
    condition_text = self.span_tracker.get_source_text(call.args[0])
    return pto_assert(*args, condition_text=condition_text, **kwargs, span=span)
