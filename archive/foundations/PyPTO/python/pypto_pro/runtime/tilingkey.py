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

"""TilingKey schema for compile-time specialization of JIT kernels.

A user declares a plain Python class whose fields are ``TilingKeyField`` descriptors.
A concrete key (a dict) selects field values at launch time; the parser folds each field
reference into the IR as a constant, so each key compiles to its own specialized kernel
(no C++ template parameters). The class may optionally define ``is_valid(self, key)`` to
reject illegal field combinations; ``key`` is a tuple of the concrete field values in
field-definition order (the same order as :meth:`TilingKeySchema.field_names`)::

    class FaTilingKey:
        NeedAttnMask = TilingKeyField(bits=1, values=[0, 1])
        BlockM = TilingKeyField(bits=2, values=[0, 1, 2])

        def is_valid(self, key):
            # key == (NeedAttnMask, BlockM); a plain predicate over the concrete values
            need_attn_mask, block_m = key
            return not (need_attn_mask == 1 and block_m == 0)

    @pl.jit(tiling_key=FaTilingKey)
    def kernel(...): ...

    kernel[stream, block_dim, {"NeedAttnMask": 1, "BlockM": 2}](...)

``is_valid`` serves two purposes: JIT rejects an illegal concrete key at launch, and the
binary-delivery path enumerates every field combination and keeps only the valid ones.
"""

from __future__ import annotations

__all__ = ["TilingKeyField", "TilingKeySchema"]

import itertools

from pypto.pypto_impl import ValueError as PyptoValueError

_MAX_TOTAL_BITS = 64


class TilingKeyField:
    """Descriptor declaring a single tilingkey field.

    Args:
        bits: Bit width reserved for the field in the packed 64-bit key (> 0).
        values: The allowed concrete values for the field (non-empty).
    """

    def __init__(self, *, bits: int, values) -> None:
        self.bits = bits
        self.values = tuple(values)
        self.name: str | None = None
        self.offset: int | None = None

    def __set_name__(self, owner, name: str) -> None:
        self.name = name


class TilingKeySchema:
    """Validated tilingkey schema collected from a user class.

    Fields are collected in class-definition order. Each field is assigned a bit
    offset; the packed 64-bit value places each field value's index in ``values`` at
    its offset. This matches AscendC template-tiling keys, where the field bits select
    an entry from the field's value list.
    """

    def __init__(self, cls: type) -> None:
        if not isinstance(cls, type):
            raise PyptoValueError(f"tiling_key must be a class, got {type(cls).__name__}")
        self._cls = cls
        self._fields: list[TilingKeyField] = self._collect_fields(cls)
        self._valid_combos: list[tuple] | None = None
        if not self._fields:
            raise PyptoValueError(f"tiling_key class '{cls.__name__}' has no TilingKeyField members")
        self.is_valid = None
        is_valid = getattr(cls, "is_valid", None)
        if is_valid is not None:
            if not callable(is_valid):
                raise PyptoValueError(f"tiling_key class '{cls.__name__}' has a non-callable 'is_valid' member")
            inst = object.__new__(cls)
            self.is_valid = is_valid.__get__(inst, cls)
        self._validate_and_assign_offsets()

    @property
    def cls_name(self) -> str:
        return self._cls.__name__

    @staticmethod
    def _collect_fields(cls: type) -> list[TilingKeyField]:
        fields = []
        for name, value in vars(cls).items():
            if isinstance(value, TilingKeyField):
                if value.name is None:
                    value.name = name
                fields.append(value)
        return fields

    def enumerate_valid(self) -> list[tuple]:
        """All valid field combinations (cartesian product filtered by is_valid).

        Returns list of tuples, each tuple is field values in definition order.
        """
        if self._valid_combos is not None:
            return self._valid_combos

        if self.is_valid is None:
            self._valid_combos = list(itertools.product(*(f.values for f in self._fields)))
            return self._valid_combos

        is_valid_func = self.is_valid
        self._valid_combos = [
            combo for combo in itertools.product(*(f.values for f in self._fields)) if is_valid_func(combo)
        ]
        return self._valid_combos

    def field_names(self) -> list[str]:
        """Field names in definition order (the template-param order)."""
        return [f.name for f in self._fields]

    def pack(self, key: dict) -> int:
        """Pack a concrete key into its unique 64-bit identifier."""
        packed = 0
        for field in self._fields:
            value_index = field.values.index(key[field.name])
            packed |= value_index << field.offset
        return packed

    def validate_concrete(self, key: dict) -> None:
        """Validate a concrete key dict against the schema."""
        if not isinstance(key, dict):
            raise PyptoValueError(f"tilingkey must be specified as a dict, got {type(key).__name__}")
        if sorted(key) != sorted(self.field_names()):
            raise PyptoValueError(
                f"tilingkey keys {sorted(key)} do not match schema fields {sorted(self.field_names())}"
            )
        for field in self._fields:
            v = key[field.name]
            if v not in field.values:
                raise PyptoValueError(
                    f"tilingkey field '{field.name}' value {v!r} is not a candidate; allowed: {list(field.values)}"
                )
        key_tuple = tuple(key[f.name] for f in self._fields)
        if self.is_valid and not self.is_valid(key_tuple):
            raise PyptoValueError(f"tilingkey {key} did not pass {self.cls_name}.is_valid")

    def _validate_and_assign_offsets(self) -> None:
        offset = 0
        for field in self._fields:
            if field.bits <= 0:
                raise PyptoValueError(f"tilingkey field '{field.name}' must have bits > 0, got {field.bits}")
            if not field.values:
                raise PyptoValueError(f"tilingkey field '{field.name}' must have a non-empty 'values' list")
            for v in field.values:
                if not isinstance(v, int) or isinstance(v, bool):
                    raise PyptoValueError(f"tilingkey field '{field.name}' values must be ints, got {v!r}")
            if len(set(field.values)) != len(field.values):
                raise PyptoValueError(f"tilingkey field '{field.name}' values must be unique")
            max_candidates = 1 << field.bits
            if len(field.values) > max_candidates:
                raise PyptoValueError(
                    f"tilingkey field '{field.name}' has {len(field.values)} candidates, "
                    f"exceeding the {max_candidates} representable by {field.bits} bits"
                )
            field.offset = offset
            offset += field.bits
        if offset > _MAX_TOTAL_BITS:
            raise PyptoValueError(
                f"tiling_key class '{self._cls.__name__}' uses {offset} bits, exceeding the {_MAX_TOTAL_BITS}-bit limit"
            )
