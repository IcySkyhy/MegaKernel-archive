# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
__all__ = [
    "ast2pil",
    "compile",
    "function",
    "Source",
]

from .parser import Source, ast2pil
from .pil2ir import compile


def function(pyfunc):
    """Lower a Python function into a PIL Function at decoration time.

    A ``@pil.function``-decorated function is defined outside the compiled entry
    kernel but can be called from within one (or from another ``@pil.function``);
    the call is inlined via ``call_function`` exactly like a nested ``def``.

    Unlike the entry kernel (a side-effect kernel that may not ``return``), a
    ``@pil.function`` is a helper and may return a value, so it is parsed with
    ``entry_point=False``.
    """
    return ast2pil(pyfunc, entry_point=False)
