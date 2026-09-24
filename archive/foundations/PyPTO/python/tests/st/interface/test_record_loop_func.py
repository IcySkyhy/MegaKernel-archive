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
""" """

import pypto


def init_tensors():
    dtype = pypto.DT_FP32
    shape = (128, 128)
    a = pypto.tensor(shape, dtype, "a")
    b = pypto.tensor(shape, dtype, "b")
    c = pypto.tensor(shape, dtype, "c")
    return a, b, c


@pypto.options(pass_options={"enable_slice": True})
def test_dynamic_loop_nomacro():
    a, b, c = init_tensors()
    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)
        for k in pypto.loop(10, name="LOOP", idx_name="k"):
            b.move(pypto.add(a, a))

            if pypto.cond(k < 2):
                b.move(pypto.add(b, a))
            else:
                b.move(pypto.sub(b, a))

            if pypto.cond(k < 5):
                b.move(pypto.mul(b, a))
            else:
                b.move(pypto.div(b, a))
            c.move(pypto.sub(b, a))

    assert isinstance(b, pypto.tensor)
