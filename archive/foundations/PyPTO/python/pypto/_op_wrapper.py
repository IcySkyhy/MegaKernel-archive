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
"""PyPTO"""

import functools

from . import pypto_impl
from ._element import Element
from ._utils import clear_source_location, set_source_location
from .error import FeError
from .tensor import ShmemTensor, Tensor


def _to_base(arg):
    if isinstance(arg, (Tensor, Element, ShmemTensor)):
        return arg.base()
    elif isinstance(arg, (list, tuple)):
        return [_to_base(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: _to_base(v) for k, v in arg.items()}
    else:
        return arg


def _from_base(out):
    if isinstance(out, pypto_impl.Tensor):
        return Tensor.from_base(out)
    elif isinstance(out, pypto_impl.ShmemTensor):
        return ShmemTensor.from_base(out)
    elif isinstance(out, (list, tuple)):
        return [_from_base(a) for a in out]
    elif isinstance(out, dict):
        return {k: _from_base(v) for k, v in out.items()}
    else:
        return out


def op_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = _to_base(args)
        kwargs = _to_base(kwargs)
        if not isinstance(args, (list, tuple)):
            raise FeError(TypeError(f"args must be list or tuple, but got {type(args)}."))
        set_source_location()
        out = func(*args, **kwargs)
        clear_source_location()
        if out is None:
            return None
        else:
            return _from_base(out)

    return wrapper
