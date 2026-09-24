# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TilingKey 参数校验测试（不需要 NPU 执行）。

正例验证正确传递 key 值不报错；反例验证各种非法输入抛出 ValueError。
这些测试仅验证 Python 侧的参数检查逻辑，不涉及 NPU 硬件。
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ---- 单字段 TilingKey ---------------------------------------------------
class TkSingle:
    OpType = TilingKeyField(bits=2, values=[0, 1, 2])


# ---- 多字段 TilingKey ---------------------------------------------------
# ---- 带有 is_valid 的 TilingKey -----------------------------------------
# ---- 公共 kernel 模板 ----------------------------------------------------


def _make_kernel(tiling_key_cls):
    @pl.jit(auto_mutex=True, tiling_key=tiling_key_cls)
    def _kernel(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
        z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ):
        m = x.shape[0]
        n = x.shape[1]
        tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile_a = pl.make_tile(tile_type, addr=0x0000, size=16384)
        tile_b = pl.make_tile(tile_type, addr=0x4000, size=16384)
        tile_c = pl.make_tile(tile_type, addr=0x8000, size=16384)

        with pl.section_vector():
            for i in pl.range(0, m, 64):
                for j in pl.range(0, n, 128):
                    pl.system.bar_all()
                    pl.load(tile_a, x, [i, j])
                    pl.load(tile_b, y, [i, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                    if OpType == 0:  # noqa: F821
                        pl.add(tile_c, tile_a, tile_b)  # noqa: F821
                    elif OpType == 1:  # noqa: F821
                        pl.sub(tile_c, tile_a, tile_b)  # noqa: F821
                    else:
                        pl.mul(tile_c, tile_a, tile_b)

                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.store(z, tile_c, [i, j])

    return _kernel


kernel_single = _make_kernel(TkSingle)
# ---- 正例 ----------------------------------------------------------------
# ---- 反例: 值不在候选集内 -------------------------------------------------
# ---- 反例: 缺少必填字段 --------------------------------------------------
# ---- 反例: 空 dict -------------------------------------------------------
# ---- 反例: 多余字段 ------------------------------------------------------
# ---- 反例: None 值 -------------------------------------------------------
# ---- 反例: 直接调用 (不使用 [...] 语法) ----------------------------------


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_direct_call_no_brackets():
    with pytest.raises(ValueError):
        import torch

        device = ST_DEVICE
        torch.npu.set_device(device)
        dummy = torch.zeros((128, 256), device=device, dtype=torch.float16)
        kernel_single(dummy)  # 有 tiling_key 的 kernel 不允许直接调用


# ---- 反例: dict 类型错误 -------------------------------------------------


# ---- 反例: is_valid 拒绝的组合 -------------------------------------------
