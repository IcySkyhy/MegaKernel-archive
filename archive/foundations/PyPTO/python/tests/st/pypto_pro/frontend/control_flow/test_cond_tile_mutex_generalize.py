# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""条件选 tile 的 auto_mutex 同步 —— 泛化场景集。

覆盖：连续两个 if/else、两个独立变量各自 if/else、if/elif/else 三分支、
只有 if 无 else、嵌套 if/else、if/else 与三元混用。

mutex_id 采用混合配置(单缓冲 / 多缓冲 / 一支单一支多)，压测 union_ids 合并与
select 在不对称缓冲下的正确性。每行 load 到条件选出的 tile，vf 读出到 dst；
src 每行数据不同，拷贝语义下期望 = src，若 mutex 锁错则数据错。
不使用 set_validshape(单独另测)。make_tile_group 直接内联(参数须编译期字面量)。
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf
import pytest
import torch

import pypto

vf = Vf

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

ROWS = 16
N = 64


@pl.vector_function
def vf_copy(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.load_align(src_tile, 0)
    vreg = vf.muls(vreg, 1.0, preg)
    vf.store_align(dst_tile, vreg, preg)


# ---------------------------------------------------------------------------
# 场景 1: 连续两个 if/else 对同一变量(第二段覆盖第一段的 select)
#   a0 单缓冲[4]，a1 双缓冲[5,6]  -> 一支单 / 一支多
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_two_ifelse(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[7])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0
            if r % 2 == 0:
                ld = a0
            else:
                ld = a1
            if r % 2 == 0:
                ld = a1
            else:
                ld = a0
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 2: 两个独立变量各自 if/else，互不干扰
#   la: a0[4] / a1[5,6]   lb: b0[7,8] / b1[9]
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_two_vars(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    g2 = pl.make_tile_group(type=tile_type, addrs=[0x2000, 0x2400], mutex_ids=[7, 8])
    g3 = pl.make_tile_group(type=tile_type, addrs=0x3000, mutex_ids=[9])
    og = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[10])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            b0 = g2.next()
            b1 = g3.next()
            la = a0
            if r % 2 == 0:
                la = a0
            else:
                la = a1
            lb = b0
            if r % 3 == 0:
                lb = b0
            else:
                lb = b1
            pl.load(la, src, [r, 0])
            pl.load(lb, src, [r, 0])
            vf_copy(la, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 3: if/elif/else 三分支(Python elif = 嵌套 if)
#   a0[4] 单 / a1[5,6] 双 / a2[7,8,9] 三
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_elif(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    g2 = pl.make_tile_group(type=tile_type, addrs=[0x2000, 0x2400, 0x2800], mutex_ids=[7, 8, 9])
    og = pl.make_tile_group(type=tile_type, addrs=0x3000, mutex_ids=[10])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            a2 = g2.next()
            ld = a0
            if r % 3 == 0:
                ld = a0
            elif r % 3 == 1:
                ld = a1
            else:
                ld = a2
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 4: 只有 if 无 else(else 隐式保留 if 前的值)
#   a0[4] 单 / a1[5,6] 双
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_if_only(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[7])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0
            if r % 2 == 1:
                ld = a1
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 5: 嵌套 if/else 选 tile
#   a0[4] / a1[5,6] / a2[7,8] / a3[9]
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_nested(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    g2 = pl.make_tile_group(type=tile_type, addrs=[0x2000, 0x2400], mutex_ids=[7, 8])
    g3 = pl.make_tile_group(type=tile_type, addrs=0x3000, mutex_ids=[9])
    og = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[10])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            a2 = g2.next()
            a3 = g3.next()
            ld = a0
            if r % 2 == 0:
                if r % 4 == 0:
                    ld = a0
                else:
                    ld = a1
            else:
                if r % 4 == 1:
                    ld = a2
                else:
                    ld = a3
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 6: if/else 后又用三元(两种机制混用)
#   a0[4] 单 / a1[5,6] 双
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ifelse_then_ternary(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[7])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0
            if r % 2 == 0:
                ld = a0
            else:
                ld = a1
            ld = a1 if r % 2 == 0 else a0
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 7: 单层三元，两个不同 tile_group(一支单缓冲 / 一支双缓冲)
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ternary(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[7])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0 if r % 2 == 0 else a1
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 8: 单层三元，两支都是 double buffer(两支 buf_id 各自动态)
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ternary_db(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=[0x0000, 0x0400], mutex_ids=[4, 5])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[6, 7])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[8])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0 if r % 2 == 0 else a1
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 9: 两层嵌套三元  a0 if c0 else (a1 if c1 else a2)
#   a0[4] 单 / a1[5,6] 双 / a2[7,8,9] 三
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ternary_nested(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    g2 = pl.make_tile_group(type=tile_type, addrs=[0x2000, 0x2400, 0x2800], mutex_ids=[7, 8, 9])
    og = pl.make_tile_group(type=tile_type, addrs=0x3000, mutex_ids=[10])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            a2 = g2.next()
            ld = a0 if r % 3 == 0 else (a1 if r % 3 == 1 else a2)
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 10: 三层嵌套三元  a0 if c0 else (a1 if c1 else (a2 if c2 else a3))
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ternary_nested3(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    g2 = pl.make_tile_group(type=tile_type, addrs=[0x2000, 0x2400], mutex_ids=[7, 8])
    g3 = pl.make_tile_group(type=tile_type, addrs=0x3000, mutex_ids=[9])
    og = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[10])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            a2 = g2.next()
            a3 = g3.next()
            ld = a0 if r % 4 == 0 else (a1 if r % 4 == 1 else (a2 if r % 4 == 2 else a3))
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 11: 单层 if/else，两个不同 tile_group(单 / 双)
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ifelse(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[4])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[5, 6])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[7])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0
            if r % 2 == 0:
                ld = a0
            else:
                ld = a1
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 12: if/else，两支都 double buffer
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ifelse_db(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=[0x0000, 0x0400], mutex_ids=[4, 5])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400], mutex_ids=[6, 7])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[8])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0
            if r % 2 == 0:
                ld = a0
            else:
                ld = a1
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 13: if/else，两支 triple buffer(每组 3 个 mutex_id)
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ifelse_multi(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g0 = pl.make_tile_group(type=tile_type, addrs=[0x0000, 0x0400, 0x0800], mutex_ids=[4, 5, 6])
    g1 = pl.make_tile_group(type=tile_type, addrs=[0x1000, 0x1400, 0x1800], mutex_ids=[7, 8, 9])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[10])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            a0 = g0.next()
            a1 = g1.next()
            ld = a0
            if r % 2 == 0:
                ld = a0
            else:
                ld = a1
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 14(已知主线 bug，未修): 同一 tile_group，if=next() / else=current()
#   next()/current() 选的 slot idx 在分支内定义，老 auto_mutex 用它作 buf_id 会作用域逃逸。
#   与本次 conditional-tile mutex 修复无关，标记 skip 保留复现。
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_next_current(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g = pl.make_tile_group(type=tile_type, addrs=[0x0000, 0x0400], mutex_ids=[4, 5])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[6])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            ld = g.current()
            if r % 2 == 0:
                ld = g.next()
            else:
                ld = g.current()
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


# ---------------------------------------------------------------------------
# 场景 15: 同一 tile_group，三元 next()/current() 条件选 slot
#   ld = g.next() if c else g.current() -- 三元路径(多 yield)版的 next/current
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def k_ternary_next_current(src: pl.Tensor[[ROWS, N], pl.DT_FP32], dst: pl.Tensor[[ROWS, N], pl.DT_FP32]):
    tile_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    g = pl.make_tile_group(type=tile_type, addrs=[0x0000, 0x0400], mutex_ids=[4, 5])
    og = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[6])
    with pl.section_vector():
        out = og.next()
        for r in pl.range(0, ROWS):
            g.current()  # 建立 cursor 基准
            ld = g.next() if r % 2 == 0 else g.current()
            pl.load(ld, src, [r, 0])
            vf_copy(ld, out)
            pl.store(dst, out, [r, 0])


def _run(kernel, name):
    device = ST_DEVICE
    torch.npu.set_device(device)
    src_cpu = torch.zeros((ROWS, N), dtype=torch.float32)
    for r in range(ROWS):
        src_cpu[r, :] = float(r + 1)
    src = src_cpu.to(device)
    expected = src_cpu.clone()  # 拷贝语义：无论选哪个 tile，数据都是 src[r]

    runs = 30
    for run in range(runs):
        dst = torch.zeros((ROWS, N), device=device, dtype=torch.float32)
        kernel[None, 1](src, dst)
        torch.npu.synchronize()
        got = dst.cpu()
        if not torch.allclose(got, expected, rtol=1e-3, atol=1e-3):
            diff = (got - expected).abs().sum(dim=1)
            bad = (diff > 1e-3).nonzero().flatten().tolist()
            details = [f"dst行{b}: got={got[b, 0].item()} 期望={expected[b, 0].item()}" for b in bad]
            raise AssertionError(f"[{name}] run {run}: 数据错误\n" + "\n".join(details))
    logging.info("[%s] PASSED (%d runs)", name, runs)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_two_ifelse():
    _run(k_two_ifelse, "two_ifelse")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_two_vars():
    _run(k_two_vars, "two_vars")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_elif():
    _run(k_elif, "elif")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_if_only():
    _run(k_if_only, "if_only")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_nested():
    _run(k_nested, "nested")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ifelse_then_ternary():
    _run(k_ifelse_then_ternary, "ifelse_then_ternary")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary():
    _run(k_ternary, "ternary")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_db():
    _run(k_ternary_db, "ternary_db")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_nested():
    _run(k_ternary_nested, "ternary_nested")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_nested3():
    _run(k_ternary_nested3, "ternary_nested3")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ifelse():
    _run(k_ifelse, "ifelse")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ifelse_db():
    _run(k_ifelse_db, "ifelse_db")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ifelse_multi():
    _run(k_ifelse_multi, "ifelse_multi")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_next_current():
    _run(k_next_current, "next_current")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_ternary_next_current():
    _run(k_ternary_next_current, "ternary_next_current")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for fn in (
        test_two_ifelse,
        test_two_vars,
        test_elif,
        test_if_only,
        test_nested,
        test_ifelse_then_ternary,
        test_ternary,
        test_ternary_db,
        test_ternary_nested,
        test_ternary_nested3,
        test_ifelse,
        test_ifelse_db,
        test_ifelse_multi,
        test_next_current,
        test_ternary_next_current,
    ):
        fn()
