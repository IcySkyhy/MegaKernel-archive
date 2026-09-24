# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import functools


class OpRegistry:
    def __init__(self):
        self.ops = {}
        self.partials = {}

    def impl(self, stub, partial=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self.ops[stub] = wrapper
            self.partials[stub] = partial
            return func

        return decorator

    def dispatch(self, stub, ctx, *args, **kwargs):
        if stub not in self.ops:
            return stub(*args, **kwargs)

        if self.partials[stub]:
            return self.ops[stub](ctx, stub, *args, **kwargs)
        return self.ops[stub](ctx, *args, **kwargs)


_op_registry = OpRegistry()
impl = _op_registry.impl
dispatch = _op_registry.dispatch
