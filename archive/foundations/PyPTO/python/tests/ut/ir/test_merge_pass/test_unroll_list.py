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

IR = (_GOLDEN_DIR / "test_unroll_list.pypto").read_text()


def test_unroll_list():

    def foo(a, b, c):
        m, n = a.shape[0], a.shape[1]
        for x in pypto.loop(10):
            for y in pypto.loop(10):
                for z in pypto.loop(n):
                    cur_seq = m - (n - 1 - z)
                    k = (cur_seq + 32 - 1) // 32
                    for i in pypto.loop(k, unroll_list=[32]):
                        t = c + 1
                        if i == 0:
                            t = t + 1
                        else:
                            t = t + 2
                        if i == k - 1:
                            t = t + 3
                        else:
                            t = t + 4
                        b[:] = t
    x = pypto.Tensor((-1, -1), pypto.DT_FP32)
    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = run_merge_pass(foo, x, y, z)
    check_snapshot(func, IR)
