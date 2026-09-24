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

IR = (_GOLDEN_DIR / "test_oi_update.pypto").read_text()


def test_oi_update():

    def foo(a, b):
        n = pypto.symbolic_scalar("n")
        for x in pypto.loop(10):
            for y in pypto.loop(10):
                oi_update = pypto.tensor([32, 32], pypto.DT_FP32, "oi_update")
                for i in pypto.loop(n, unroll_list=[2]):
                    if i == 0:
                        oi_update[:] = a + 1
                    else:
                        oi_update[:] = oi_update - 1
                    if i + 1 >= n:
                        b[:] = oi_update // 2

    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = run_merge_pass(foo, y, z)
    check_snapshot(func, IR)


def test_oi_update_unroll():

    def foo(a, b):
        n = pypto.symbolic_scalar("n")
        for x in pypto.loop(10):
            for y in pypto.loop(10):
                oi_update = pypto.tensor([32, 32], pypto.DT_FP32, "oi_update")
                for i in pypto.loop(n, unroll_list=[32, 16, 8, 4, 1]):
                    if i == 0:
                        oi_update[:] = a + 1
                    else:
                        oi_update[:] = oi_update - 1
                    if i + 1 >= n:
                        b[:] = oi_update // 2

    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    run_merge_pass(foo, y, z)
