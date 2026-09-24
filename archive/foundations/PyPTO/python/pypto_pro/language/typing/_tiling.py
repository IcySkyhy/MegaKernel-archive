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

"""Tiling class utilities for PyPTO Language DSL."""

from __future__ import annotations

__all__ = [
    "is_tiling_class",
    "get_tiling_fields",
    "ScalarFieldInfo",
    "ArrayFieldInfo",
    "FieldInfo",
    "get_tiling_ctype_struct",
    "tiling_instance_to_bytes",
]


import ast
import ctypes
from dataclasses import dataclass, is_dataclass
import inspect
import textwrap
from typing import Union

from pypto.pypto_impl.ir import DataType

_PYTHON_TYPE_TO_DTYPE: dict[type, DataType] = {
    int: DataType.INDEX,
    float: DataType.FP32,
    bool: DataType.BOOL,
}

_PYTHON_TYPE_NAME_TO_TYPE: dict[str, type] = {type_.__name__: type_ for type_ in _PYTHON_TYPE_TO_DTYPE}


@dataclass(frozen=True)
class ScalarFieldInfo:
    """Field info for a scalar tiling field (int, float, or bool)."""

    dtype: DataType


@dataclass(frozen=True)
class ArrayFieldInfo:
    """Field info for a fixed-length array tiling field (T[N])."""

    dtype: DataType
    size: int


FieldInfo = Union[ScalarFieldInfo, ArrayFieldInfo]


def _parse_field_annotation(annotation: object) -> FieldInfo:
    """Normalize a scalar or fixed-array tiling annotation without evaluating it."""
    if isinstance(annotation, type) and annotation in _PYTHON_TYPE_TO_DTYPE:
        return ScalarFieldInfo(dtype=_PYTHON_TYPE_TO_DTYPE[annotation])
    if not isinstance(annotation, str):
        raise TypeError("expected int, float, bool, or T[N]")

    try:
        expression = ast.parse(annotation, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"invalid annotation syntax {annotation!r}") from exc

    if isinstance(expression, ast.Name):
        dtype = _PYTHON_TYPE_NAME_TO_TYPE.get(expression.id)
        if dtype is None:
            raise TypeError(f"unsupported scalar type {expression.id!r}")
        return ScalarFieldInfo(dtype=_PYTHON_TYPE_TO_DTYPE[dtype])

    if not isinstance(expression, ast.Subscript) or not isinstance(expression.value, ast.Name):
        raise TypeError("expected a scalar type or fixed-array annotation T[N]")
    dtype = _PYTHON_TYPE_NAME_TO_TYPE.get(expression.value.id)
    if dtype is None:
        raise TypeError(f"array element type must be int, float, or bool, got {expression.value.id!r}")
    size_node = expression.slice
    if not isinstance(size_node, ast.Constant):
        raise ValueError("array size must be a positive integer literal")
    if not isinstance(size_node.value, int) or isinstance(size_node.value, bool) or size_node.value <= 0:
        raise ValueError("array size must be a positive integer literal")
    size = size_node.value
    if size > 2048:
        raise ValueError(f"Array size must not exceed 2048, got {size}")
    return ArrayFieldInfo(dtype=_PYTHON_TYPE_TO_DTYPE[dtype], size=size)


def is_tiling_class(cls: object) -> bool:
    """Return True if cls is a user-defined tiling class.

    A tiling class is a dataclass with at least one field,
    all annotated as int, float, bool, or T[N].

    Args:
        cls: Object to check

    Returns:
        True if cls is a valid tiling class
    """
    if not isinstance(cls, type):
        return False
    if not is_dataclass(cls):
        return False
    annotations = getattr(cls, "__annotations__", {})
    if not annotations:
        return False

    _check_duplicate_annotations(cls)

    try:
        for annotation in annotations.values():
            _parse_field_annotation(annotation)
    except (TypeError, ValueError):
        return False
    return True


def _check_duplicate_annotations(cls: type) -> None:
    """Parse class source AST to detect duplicate annotated field names.

    Python deduplicates __annotations__ at class creation, so runtime
    introspection cannot see duplicates. Use AST parsing as a best-effort check.
    """
    try:
        src = textwrap.dedent(inspect.getsource(cls))
    except (TypeError, OSError):
        return
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            ann_names = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    ann_names.append(stmt.target.id)
            if len(ann_names) != len(set(ann_names)):
                seen: set[str] = set()
                for name in ann_names:
                    if name in seen:
                        raise ValueError(
                            f"Tiling class '{cls.__name__}' has duplicate field '{name}'. "
                            f"All field names must be unique."
                        )
                    seen.add(name)
            return


def get_tiling_fields(cls: type) -> dict[str, FieldInfo]:
    """Return ordered {field_name: FieldInfo} for a validated tiling class.

    Args:
        cls: A tiling class (validated by is_tiling_class)

    Returns:
        Ordered dict mapping field names to their FieldInfo (ScalarFieldInfo or ArrayFieldInfo)

    Raises:
        ValueError: If cls is not a valid tiling class
    """
    if not isinstance(cls, type) or not is_dataclass(cls):
        raise ValueError(f"Not a valid tiling class: {cls!r}. Expected a dataclass.")
    annotations = getattr(cls, "__annotations__", {})
    if not annotations:
        raise ValueError(f"Not a valid tiling class: {cls!r}. At least one field is required.")
    result: dict[str, FieldInfo] = {}
    for name, annotation in annotations.items():
        try:
            result[name] = _parse_field_annotation(annotation)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid tiling field {name!r} annotation {annotation!r}: {exc}") from exc
    return result


def _field_ctype(dtype: DataType) -> type:
    # Reuse the single DataType→ctype map maintained in the runtime (lazy import to avoid a
    # module-level cycle; jit imports this module inside functions). Must stay in sync with
    # the C struct member types emitted by CCE codegen's CppTypeForField (e.g. INDEX → int64_t).
    from pypto_pro.runtime.jit import _PL_DTYPE_TO_CTYPE

    ctype = _PL_DTYPE_TO_CTYPE.get(str(dtype))
    if ctype is None:
        raise TypeError(f"No ctypes mapping for tiling field dtype {dtype}")
    return ctype


# Cache the generated ctypes.Structure subclass per tiling class.
_CTYPE_STRUCT_CACHE: dict[type, type] = {}


def get_tiling_ctype_struct(cls: type) -> type:
    """Return a ctypes.Structure subclass whose layout matches the C ``struct Tiling``.

    Scalar field → one ctypes member; T[N] → a length-N ctypes array member.
    The native ctypes alignment rules match the C struct emitted by codegen, so the
    byte layout (sizeof + field offsets) is identical on both sides.
    """
    cached = _CTYPE_STRUCT_CACHE.get(cls)
    if cached is not None:
        return cached
    fields: list[tuple[str, type]] = []
    for name, info in get_tiling_fields(cls).items():
        base = _field_ctype(info.dtype)
        if isinstance(info, ArrayFieldInfo):
            fields.append((name, base * info.size))
        else:
            fields.append((name, base))
    struct_cls = type(f"{cls.__name__}_CStruct", (ctypes.Structure,), {"_fields_": fields})
    _CTYPE_STRUCT_CACHE[cls] = struct_cls
    return struct_cls


def tiling_instance_to_bytes(instance: object) -> bytes:
    """Serialize a tiling-class instance to its C-struct byte layout."""
    cls = type(instance)
    struct_cls = get_tiling_ctype_struct(cls)
    values: list = []
    for name, info in get_tiling_fields(cls).items():
        val = getattr(instance, name)
        if isinstance(info, ArrayFieldInfo):
            if (
                isinstance(val, (str, bytes, bytearray))
                or not hasattr(val, "__getitem__")
                or not hasattr(val, "__len__")
            ):
                raise TypeError(
                    f"Tiling field '{name}' is T[{info.size}]: expected an indexable "
                    f"sequence, got {type(val).__name__!r}"
                )
            if len(val) != info.size:
                raise ValueError(f"Tiling field '{name}' expected {info.size} elements, got {len(val)}")
            base = _field_ctype(info.dtype)
            values.append((base * info.size)(*[val[i] for i in range(info.size)]))
        else:
            values.append(val)
    return bytes(struct_cls(*values))
