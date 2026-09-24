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

IR = (_GOLDEN_DIR / "test_oi_update1.pypto").read_text()


def test_oi_update1():

    def kernel(a, b):
        pypto.set_vec_tile_shapes(32, 32)
        oi_update_0 = pypto.tensor([32, 32], pypto.DT_FP32, "oi_update")
        oi_update_1 = pypto.tensor([32, 32], pypto.DT_FP32, "oi_update1")

        for k_tile_idx in pypto.loop(10, name="k_tile_loop"):
            for h_s_idx in range(2):
                if h_s_idx == 0:
                    oi_update = oi_update_0
                else:
                    oi_update = oi_update_1

                if pypto.is_loop_begin(k_tile_idx):
                    if pypto.is_loop_end(k_tile_idx):
                        pass
                    else:
                        oij = a + 1
                        oi_update[:] = oij
                else:
                    oi_tmp = a + oi_update
                    if pypto.is_loop_end(k_tile_idx):
                        pypto.assemble(oi_tmp, [0, 0], b)
                        pass
                    else:
                        oi_update[:] = oi_tmp


    y = pypto.Tensor((32, 32), pypto.DT_FP32)
    z = pypto.Tensor((32, 32), pypto.DT_FP32)
    func = run_merge_pass(kernel, y, z)
    check_snapshot(func, IR)
