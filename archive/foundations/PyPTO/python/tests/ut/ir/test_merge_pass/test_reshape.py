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

IR = (_GOLDEN_DIR / "test_reshape.pypto").read_text()


def reshape_kernel(
    out: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]],
):
    for b in pypto.loop(10, unroll_list=[1, 2]):
        for s in pypto.loop(10):
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
            pypto.set_vec_tile_shapes(16, 16, 64)

            tmp0 = pypto.full([4, 128, 128], 1.0, pypto.DT_FP32)
            tmp1 = pypto.full([4, 128, 128], 2.0, pypto.DT_FP32)
            tmp2 = pypto.matmul(tmp0, tmp1, out_dtype=pypto.DT_FP32)
            out[4 * b:, 128 * s:, :] = tmp2


def test_reshape():
    out = pypto.Tensor([-1, -1, 64])
    func = run_merge_pass(reshape_kernel, out)
    check_snapshot(func, IR)
