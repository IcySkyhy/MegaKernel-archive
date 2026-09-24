# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

import pypto

from ..test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent

IR = (_GOLDEN_DIR / "test_merge_pass7.pypto").read_text()


def test_merge_pass7():
    """A nested if that exists only under one branch stays nested only under that branch's leaves."""

    def foo(x, y, z):
        pypto.set_vec_tile_shapes(16, 16)
        xv = pypto.view(x, [16, 16], [0, 0])
        _yv = pypto.view(y, [16, 16], [0, 0])
        for i in pypto.loop(x.shape[0], unroll_list=[4]):
            if i == 0:
                t = xv + 1
            else:
                t = xv - 1
            t = t + 2
            if i == x.shape[0] - 1:
                t = xv + 1
            else:
                t = xv - 1
            pypto.assemble(t, [0, 0], z)

    x = pypto.Tensor([-1, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([-1, 32], pypto.DT_FP32, 'y')
    z = pypto.Tensor([-1, 32], pypto.DT_FP32, 'z')
    func = run_merge_pass(foo, x, y, z)
    check_snapshot(func, IR)
