#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verification helpers: conditional print and tensor dump (ToFile)."""

import re
from typing import List, Tuple, Union

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from .._utils import to_sym
from ..error import VerifyError
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


@op_wrapper
def pass_verify_print(*values, cond: Union[int, SymbolicScalar] = 1) -> None:
    """
    Conditionally print tensors / symbolic scalars during pass verify.

    Parameters
    ----------
    *values :
        Mixed list of tensors, symbolic scalars and normal Python objects.
        - Tensor / pypto.Tensor: printed by backend in a compact tensor format
        - int / SymbolicScalar: treated as symbolic scalars and printed as scalar values
        - other python objects: converted to string and printed directly
    cond : int or SymbolicScalar, optional
        Print condition. When `cond` evaluates to 0, nothing will be printed.
        Default is 1 (always print).

    Notes
    -----
    This API is intended for **pass verification** and has no effect
    on numerical results. It only affects log output.

    Examples
    --------
    >>> # Print a tensor and a scalar unconditionally
    >>> pass_verify_print(tensor_a, " step=", 10)
    >>>
    >>> # Only print when idx > 0
    >>> for idx in pypto.loop(10):
    ...     pass_verify_print("idx=", idx, cond=(idx > 0))
    """
    from .. import pypto_impl as _impl

    fmt_parts: List[str] = []
    tensors: List[_impl.Tensor] = []
    scalars: List[_impl.SymbolicScalar] = []

    for v in values:
        if isinstance(v, _impl.Tensor):
            tensors.append(v)
            fmt_parts.append("{T}")
        elif isinstance(v, Tensor):
            tensors.append(v.base())
            fmt_parts.append("{T}")
        elif isinstance(v, (int, _impl.SymbolicScalar)):
            scalars.append(to_sym(v))
            fmt_parts.append("{S}")
        else:
            fmt_parts.append(str(v))

    fmt = "".join(fmt_parts)
    cond_base = to_sym(cond)

    _impl.PrintIf(cond_base, fmt, tensors, scalars)


def _parse_format_string(
    format_str: str, **kwargs: Union[int, SymbolicScalar, pypto_impl.SymbolicScalar]
) -> Tuple[str, List[pypto_impl.SymbolicScalar]]:
    from .. import pypto_impl as _impl
    from .._utils import to_sym as _to_sym

    pattern = re.compile(r'\$([a-zA-Z0-9_]+)')

    scalars: List[_impl.SymbolicScalar] = []
    parts: List[str] = []
    last_pos = 0

    for match in pattern.finditer(format_str):
        if match.start() > last_pos:
            parts.append(format_str[last_pos:match.start()])

        placeholder_name = match.group(1)

        if placeholder_name not in kwargs:
            raise VerifyError(KeyError(f"Placeholder '${placeholder_name}' not found in keyword arguments"))

        value = kwargs[placeholder_name]
        scalars.append(_to_sym(value))
        parts.append("{S}")

        last_pos = match.end()

    if last_pos < len(format_str):
        parts.append(format_str[last_pos:])

    return "".join(parts), scalars


@op_wrapper
def pass_verify_save(
    tensor: Tensor,
    fname: Union[str, SymbolicScalar, int],
    cond: Union[int, SymbolicScalar] = 1,
    **kwargs: Union[int, SymbolicScalar, pypto_impl.SymbolicScalar],
) -> None:
    """
    Conditionally dump a tensor to file during pass verify.

    Parameters
    ----------
    tensor : Tensor
        Tensor to be dumped.
    fname : str or SymbolicScalar or int
        File name template. Dynamic placeholders can be written as ``$name``,
        and will be substituted by the corresponding values from ``kwargs``.
        The actual save path is determined by backend verify configuration.
    cond : int or SymbolicScalar, optional
        Save condition. When `cond` evaluates to 0, this call is ignored.
        Default is 1 (always save).
    **kwargs :
        Mapping from placeholder name to scalar value, for example
        ``pass_verify_save(t, "tensor_$idx", idx=loop_idx)``.

    Notes
    -----
    This API is intended for **pass verification** and has no effect
    on computation results. It is mainly used to locate and compare
    intermediate tensors of specific passes / iterations.

    Examples
    --------
    >>> # Save tensor with fixed name
    >>> pass_verify_save(tensor_a, "tensor_a")
    >>>
    >>> # Save tensor of different loop iterations as tensor_out_0, tensor_out_1, ...
    >>> for idx in pypto.loop(10):
    ...     pass_verify_save(tensor_out, "tensor_out_$idx", idx=idx)
    """
    from .. import pypto_impl as _impl  # ensure base types

    fname_str, scalars = _parse_format_string(fname, **kwargs)
    cond_base = to_sym(cond)

    _impl.ToFile(tensor, fname_str, scalars, cond_base)
