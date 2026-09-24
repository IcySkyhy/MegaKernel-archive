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

"""Normalize and bind tensor shape policies used by PyPTO Pro frontends."""

from __future__ import annotations

__all__ = [
    "BoundDimension",
    "BoundKernelSignature",
    "BoundTensorShape",
    "DynamicDim",
    "FixedDim",
    "KernelSignatureSpec",
    "StaticDim",
    "StaticTail",
    "TensorShapeSpec",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import inspect
import re
from typing import Any, Union

from pypto_pro.language.typing.shape import DYNAMIC, STATIC, _ShapePolicy


def _get_annotations(func, namespace):
    get_annotations = getattr(inspect, "get_annotations", None)
    if get_annotations is not None:
        return get_annotations(func, globals=namespace, locals=namespace, eval_str=True)

    annotations = dict(getattr(func, "__annotations__", {}))
    for name, annotation in annotations.items():
        if isinstance(annotation, str):
            annotations[name] = eval(annotation, namespace, namespace)
    return annotations


def _dynamic_name(parameter_name: str, axis: int) -> str:
    safe_name = re.sub(r"[^0-9A-Za-z_]", "_", parameter_name)
    return f"__pypto_dyn_{safe_name}_{axis}"


def _validate_positive_int(value: Any, parameter_name: str, axis: int, *, source: str) -> int:
    if type(value) is not int:
        raise TypeError(
            f"Tensor parameter '{parameter_name}' axis {axis} {source} must be a positive integer, "
            f"got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"Tensor parameter '{parameter_name}' axis {axis} {source} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class FixedDim:
    """A dimension fixed by the source annotation."""

    value: int


@dataclass(frozen=True)
class DynamicDim:
    """A runtime dimension that does not specialize compilation."""

    parameter_name: str
    axis: int

    @property
    def name(self) -> str:
        return _dynamic_name(self.parameter_name, self.axis)


@dataclass(frozen=True)
class StaticDim:
    """A runtime dimension embedded into one compiled variant."""

    parameter_name: str
    axis: int


@dataclass(frozen=True)
class StaticTail:
    """A final ellipsis expanded into zero or more StaticDim records."""

    parameter_name: str
    start_axis: int


DimensionSpec = Union[FixedDim, DynamicDim, StaticDim, StaticTail]


@dataclass(frozen=True)
class BoundDimension:
    """One lowered dimension: either a constant value or a dynamic ABI name."""

    axis: int
    value: int | None = None
    dynamic_name: str | None = None
    is_static: bool = False

    def __post_init__(self) -> None:
        if (self.value is None) == (self.dynamic_name is None):
            raise ValueError("A bound dimension must contain exactly one of value or dynamic_name")


@dataclass(frozen=True)
class BoundTensorShape:
    """Concrete lowering information for one tensor parameter."""

    parameter_name: str
    parameter_index: int
    dimensions: tuple[BoundDimension, ...]

    @property
    def static_signature(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (self.parameter_index, dim.axis, dim.value)
            for dim in self.dimensions
            if dim.is_static and dim.value is not None
        )

    def to_ir_shape(self, span):
        """Build the canonical IR dimensions stored in the parameter TensorType."""
        from pypto.pypto_impl import ir
        from pypto.pypto_impl.ir import DataType

        result = []
        for dim in self.dimensions:
            if dim.dynamic_name is not None:
                result.append(ir.Var(dim.dynamic_name, ir.ScalarType(DataType.INDEX), span))
            else:
                if dim.value is None:  # Guard the BoundDimension invariant for type checkers.
                    raise ValueError("A constant bound dimension must contain a value")
                result.append(ir.ConstInt(dim.value, DataType.INDEX, span))
        return result


@dataclass(frozen=True)
class TensorShapeSpec:
    """Validated, unbound shape policy for one tensor parameter."""

    parameter_name: str
    parameter_index: int
    dimensions: tuple[DimensionSpec, ...]
    dtype: Any = field(default=None, compare=False, repr=False)

    @property
    def requires_binding(self) -> bool:
        return any(isinstance(dim, (StaticDim, StaticTail)) for dim in self.dimensions)

    @property
    def has_ellipsis(self) -> bool:
        return bool(self.dimensions) and isinstance(self.dimensions[-1], StaticTail)

    @classmethod
    def from_annotation(
        cls,
        parameter_name: str,
        parameter_index: int,
        shape: Sequence[Any],
        *,
        dtype: Any = None,
    ) -> "TensorShapeSpec":
        if not isinstance(shape, (list, tuple)):
            raise TypeError(
                f"Tensor parameter '{parameter_name}' shape must be a list or tuple, got {type(shape).__name__}"
            )

        dimensions: list[DimensionSpec] = []
        ellipsis_seen = False
        for axis, raw_dim in enumerate(shape):
            if raw_dim is Ellipsis:
                if ellipsis_seen or axis != len(shape) - 1:
                    raise TypeError(
                        f"Tensor parameter '{parameter_name}' ellipsis must appear once "
                        "and only as the final shape item"
                    )
                dimensions.append(StaticTail(parameter_name, axis))
                ellipsis_seen = True
                continue
            if type(raw_dim) is int:
                try:
                    value = _validate_positive_int(raw_dim, parameter_name, axis, source="annotation")
                except ValueError as exc:
                    raise TypeError(str(exc)) from exc
                dimensions.append(FixedDim(value))
                continue
            if raw_dim is DYNAMIC:
                dimensions.append(DynamicDim(parameter_name, axis))
                continue
            if raw_dim is STATIC:
                dimensions.append(StaticDim(parameter_name, axis))
                continue
            policy_name = raw_dim.name if isinstance(raw_dim, _ShapePolicy) else type(raw_dim).__name__
            raise TypeError(
                f"Tensor parameter '{parameter_name}' axis {axis} annotation must be DYNAMIC, STATIC, "
                f"a positive integer, or final ellipsis; got {policy_name}"
            )
        return cls(parameter_name, parameter_index, tuple(dimensions), dtype=dtype)

    def bind(self, actual_shape: Sequence[int] | None) -> BoundTensorShape:
        if actual_shape is None:
            if self.requires_binding:
                raise ValueError(
                    f"static_shapes must provide tensor parameter '{self.parameter_name}' because it contains "
                    "STATIC or ellipsis dimensions"
                )
            actual: tuple[int, ...] | None = None
        else:
            if not isinstance(actual_shape, Sequence) or isinstance(actual_shape, (str, bytes)):
                raise TypeError(
                    f"Shape for tensor parameter '{self.parameter_name}' must be a sequence of positive integers"
                )
            actual = tuple(
                _validate_positive_int(value, self.parameter_name, axis, source="runtime dimension")
                for axis, value in enumerate(actual_shape)
            )

        explicit_rank = len(self.dimensions) - (1 if self.has_ellipsis else 0)
        if actual is not None:
            rank_matches = len(actual) >= explicit_rank if self.has_ellipsis else len(actual) == explicit_rank
            if not rank_matches:
                expected = f"at least {explicit_rank}" if self.has_ellipsis else str(explicit_rank)
                raise ValueError(
                    f"Tensor parameter '{self.parameter_name}' rank mismatch: expected {expected}, got {len(actual)}"
                )

        bound: list[BoundDimension] = []
        for axis, spec in enumerate(self.dimensions):
            if isinstance(spec, StaticTail):
                if actual is None:
                    raise ValueError(
                        f"static_shapes must provide tensor parameter '{self.parameter_name}' to expand ellipsis"
                    )
                for tail_axis in range(axis, len(actual)):
                    bound.append(BoundDimension(tail_axis, value=actual[tail_axis], is_static=True))
                break
            if isinstance(spec, FixedDim):
                if actual is not None and actual[axis] != spec.value:
                    raise ValueError(
                        f"Tensor parameter '{self.parameter_name}' axis {axis} mismatch: "
                        f"expected {spec.value}, got {actual[axis]}"
                    )
                bound.append(BoundDimension(axis, value=spec.value))
                continue
            if isinstance(spec, DynamicDim):
                bound.append(BoundDimension(axis, dynamic_name=spec.name))
                continue
            if actual is None:
                raise ValueError(f"static_shapes must provide tensor parameter '{self.parameter_name}' axis {axis}")
            bound.append(BoundDimension(axis, value=actual[axis], is_static=True))
        return BoundTensorShape(self.parameter_name, self.parameter_index, tuple(bound))


@dataclass(frozen=True)
class BoundKernelSignature:
    """All bound tensor shapes for one parsed/compiled kernel variant."""

    tensors: tuple[BoundTensorShape, ...]

    @property
    def static_signature(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(entry for tensor in self.tensors for entry in tensor.static_signature)

    def tensor(self, parameter_name: str) -> BoundTensorShape:
        for tensor in self.tensors:
            if tensor.parameter_name == parameter_name:
                return tensor
        raise KeyError(parameter_name)

    def get_tensor(self, parameter_name: str) -> BoundTensorShape | None:
        try:
            return self.tensor(parameter_name)
        except KeyError:
            return None


@dataclass(frozen=True)
class KernelSignatureSpec:
    """Normalized tensor policy declarations for one Python kernel signature."""

    tensors: tuple[TensorShapeSpec, ...]
    python_signature: inspect.Signature | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_callable(
        cls,
        func,
        closure_vars: Mapping[str, Any] | None = None,
    ) -> "KernelSignatureSpec":
        from pypto_pro.language.typing.tensor import Tensor

        namespace = dict(getattr(func, "__globals__", {}))
        namespace.update(closure_vars or {})
        try:
            annotations = _get_annotations(func, namespace)
        except (NameError, TypeError, ValueError) as exc:
            raise TypeError(f"Failed to evaluate annotations for kernel '{func.__name__}': {exc}") from exc

        signature = inspect.signature(func)
        tensor_specs: list[TensorShapeSpec] = []
        for parameter_index, (name, parameter) in enumerate(signature.parameters.items()):
            annotation = annotations.get(name, parameter.annotation)
            if isinstance(annotation, Tensor):
                tensor_specs.append(
                    TensorShapeSpec.from_annotation(
                        name,
                        parameter_index,
                        annotation.shape,
                        dtype=annotation.dtype,
                    )
                )
        return cls(tuple(tensor_specs), python_signature=signature)

    def bind_runtime_args(self, args: tuple, kwargs: Mapping[str, Any] | None = None) -> BoundKernelSignature:
        if self.python_signature is None:
            raise TypeError("bind_runtime_args requires a signature created with from_callable()")
        bound_args = self.python_signature.bind(*args, **dict(kwargs or {}))
        bound_args.apply_defaults()
        shapes: dict[str, Sequence[int]] = {}
        for tensor in self.tensors:
            value = bound_args.arguments[tensor.parameter_name]
            if value is None:
                continue
            shape = getattr(value, "shape", None)
            if shape is None:
                raise TypeError(
                    f"Tensor parameter '{tensor.parameter_name}' must provide a shape, got {type(value).__name__}"
                )
            shapes[tensor.parameter_name] = shape
        return self.bind_static_shapes(shapes)

    def bind_static_shapes(
        self,
        static_shapes: Mapping[str, Sequence[int]] | None,
    ) -> BoundKernelSignature:
        if static_shapes is not None and not isinstance(static_shapes, Mapping):
            raise TypeError(
                f"static_shapes must be a mapping from parameter name to shape, got {type(static_shapes).__name__}"
            )
        provided = dict(static_shapes or {})
        known_names = {tensor.parameter_name for tensor in self.tensors}
        unknown = sorted(set(provided) - known_names)
        if unknown:
            raise ValueError(f"static_shapes contains unknown tensor parameters: {', '.join(unknown)}")

        bound_tensors: list[BoundTensorShape] = []
        for tensor in self.tensors:
            actual_shape = provided.get(tensor.parameter_name)
            bound_tensors.append(tensor.bind(actual_shape))
        return BoundKernelSignature(tuple(bound_tensors))
