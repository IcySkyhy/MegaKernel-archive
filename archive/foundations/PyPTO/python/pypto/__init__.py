#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""

# torch/torch_npu may use cxxabi=0 or cxxabi=1, while pypto only support cxxabi=0
# if pypto load first, torch/torch_cpu may crash, force load torch first
try:
    import torch
except ImportError:
    pass

import pydoc as _pydoc

# shared lib should be loaded first
from . import _loader

from . import experimental
from .config import *
from ._controller import *
from .converter import from_torch
from .enum import *
from .op import *
from .operation import *
from .operator import *
from .pass_config import *
from .cost_model import *
from ._utils import ceildiv, bytes_of
from .platform import platform
from .runtime import verify, set_verify_golden_data, RunMode
from .symbolic_scalar import SymbolicScalar, SatStatus
from .tensor import Tensor
from .functions import Function, get_last_function, get_current_function
from ._element import Element
from .logging import *

# Import frontend after all other imports to avoid circular imports
from . import frontend

jit = frontend.jit
tensor = Tensor
element = Element
symbolic_scalar = SymbolicScalar


def _patch_getowndoc(obj):
    try:
        doc = obj.__doc__
        if doc is None:
            return None
        if obj is not type:
            typedoc = type(obj).__doc__
            if isinstance(typedoc, str) and typedoc == doc:
                return None
        return doc
    except AttributeError:
        return None
setattr(_pydoc, '_getowndoc', _patch_getowndoc)
del _patch_getowndoc
