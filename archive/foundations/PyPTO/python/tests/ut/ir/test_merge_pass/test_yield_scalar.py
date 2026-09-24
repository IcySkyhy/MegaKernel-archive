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

IR = (_GOLDEN_DIR / "test_yield_scalar.pypto").read_text()

def test_yield_symbolic_scalar():
    def foo(x, y):
        for i in pypto.loop(10):
            if i == 0:
                off = 0
            else:
                off = 32
            y[off + 32:, off + 32:] = x

    a = pypto.Tensor((64, 32), pypto.DT_FP32, 'a')
    out = pypto.Tensor((32, 32), pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, a, out)
    check_snapshot(func, IR)
