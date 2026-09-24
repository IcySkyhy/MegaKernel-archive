# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from pathlib import Path

import pytest

import pypto

from ..test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent

IR = (_GOLDEN_DIR / "test_type_conflict.pypto").read_text()


def test_type_conflict():
    """yield var `a` has two types, but not used later"""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            if i == 0:
                a = pypto.full([32, 32], 1.0, pypto.DT_FP32)
                x[:] = a + 1
            else:
                a = pypto.full([64, 64], 2.0, pypto.DT_FP32)
                y[:] = a + 2

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([64, 64], pypto.DT_FP32, 'y')
    func = run_merge_pass(foo, x, y)
    check_snapshot(func, IR)


def test_type_conflict1():
    """yield var `a` has two types, but not used later"""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            if i == 0:
                a = pypto.full([32, 32], 1.0, pypto.DT_FP32)
            else:
                a = pypto.full([64, 64], 2.0, pypto.DT_FP32)
            c = a + 1
            x[:] = c

    x = pypto.Tensor([32, 32], pypto.DT_FP32, 'x')
    y = pypto.Tensor([64, 64], pypto.DT_FP32, 'y')
    with pytest.raises(Exception, match="Conflicted var a"):
        run_merge_pass(foo, x, y)
