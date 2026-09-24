#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Softmax DN VF API test — ProcessVec1DnNoUpdateVF (fp16 branch, full version).

Full version with Cast + DeInterleave + StoreAlign(DATA_BLOCK_COPY).
Constants: M=64, UB_N=128, D_SCALE=1.0

UB layout (DN mode / transposed):
  input_x_local_UB: fp32 [UB_N, M] = [128, 64]
  x_exp:            fp16 output (softmax result, NZ layout)
  new_global_max:   fp32 [64] row max
  new_global_sum:   fp32 [64] row sum
"""

import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ================================================================
# Constants
# ================================================================
M = 64  # elements per register (row stride)
UB_N = 128  # columns
D_SCALE = 1.0  # scale factor
minValue = -1e9  # noqa: N816
blockStride = UB_N >> 1  # noqa: N816  # = 64 (不加1，先不管bank冲突)
REPEAT_STRIDE = 1

# UB address layout
INPUT_SIZE = UB_N * M * 4  # fp32 [128, 64] = 32KB
XEXP_SIZE = UB_N * M * 2  # fp16 [128, 64] = 16KB
MAX_SIZE = M * 4  # fp32 [64] = 256B
SUM_SIZE = M * 4  # fp32 [64] = 256B

VA_INPUT = 0
VA_XEXP = VA_INPUT + INPUT_SIZE  # 0x8000
VA_MAX = VA_XEXP + XEXP_SIZE  # 0xC000
VA_SUM = VA_MAX + MAX_SIZE  # 0xC100


@pl.vector_function
def process_vec1_dn_no_update_vf(input_tile, x_exp_tile, max_tile, sum_tile):
    """Softmax DN VF kernel — full version with Cast + DeInterleave + Store."""
    # ---- fp32 registers ----
    # ---- fp16 registers (for Cast output) ----
    # ---- Masks ----
    preg_108 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)  # b32, for fp32 ops
    preg_134 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)  # b32, for fp32 exp/add
    preg_135 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)  # b32, for Cast
    preg_136 = vf.update_mask(128, dtype=pl.DT_FP16)  # for DATA_BLOCK_COPY store (fp16 data)

    # ---- Pointer offsets ----
    src_ub0 = input_tile
    src_ub1 = input_tile + M
    src_ub2 = input_tile + M * 2
    src_ub3 = input_tile + M * 3
    x_exp_1 = x_exp_tile + UB_N * 4  # second half of x_exp output

    # ============================================================
    # Phase 1: ReduceMax (4-way parallel)
    # ============================================================
    max0 = vf.full(minValue)
    max1 = vf.full(minValue)
    max2 = vf.full(minValue)
    max3 = vf.full(minValue)

    for iter_m in pl.range(0, UB_N // 4):
        src0 = vf.load_align(src_ub0, iter_m * M * 4)
        src1 = vf.load_align(src_ub1, iter_m * M * 4)
        src2 = vf.load_align(src_ub2, iter_m * M * 4)
        src3 = vf.load_align(src_ub3, iter_m * M * 4)
        max0 = vf.max(max0, src0, preg_108)
        max1 = vf.max(max1, src1, preg_108)
        max2 = vf.max(max2, src2, preg_108)
        max3 = vf.max(max3, src3, preg_108)

    max0 = vf.max(max0, max2, preg_108)
    max1 = vf.max(max1, max3, preg_108)
    max0 = vf.max(max0, max1, preg_108)
    max0 = vf.muls(max0, D_SCALE, preg_108)

    vf.store_align(max_tile, max0, preg_108)

    # ============================================================
    # Phase 2: ExpSub + Cast + DeInterleave + Store + Sum
    # ============================================================
    vreg_x_sum_0 = vf.full(0.0, preg_134)
    vreg_x_sum_1 = vf.full(0.0, preg_134)
    vreg_x_sum_2 = vf.full(0.0, preg_134)
    vreg_x_sum_3 = vf.full(0.0, preg_134)

    for i0 in pl.range(0, UB_N // 4):
        # Load from 4 quadrants
        vreg_x_f32_0 = vf.load_align(input_tile, i0 * M)
        vreg_x_f32_1 = vf.load_align(input_tile, UB_N * M // 4 + i0 * M)
        vreg_x_f32_2 = vf.load_align(input_tile, UB_N * M // 2 + i0 * M)
        vreg_x_f32_3 = vf.load_align(input_tile, UB_N * M // 2 + UB_N * M // 4 + i0 * M)

        # Scale
        vreg_x_f32_0 = vf.muls(vreg_x_f32_0, D_SCALE, preg_108)
        vreg_x_f32_1 = vf.muls(vreg_x_f32_1, D_SCALE, preg_108)
        vreg_x_f32_2 = vf.muls(vreg_x_f32_2, D_SCALE, preg_108)
        vreg_x_f32_3 = vf.muls(vreg_x_f32_3, D_SCALE, preg_108)

        vreg_x_exp_0 = vf.exp_sub(vreg_x_f32_0, max0, preg_134)
        vreg_x_exp_1 = vf.exp_sub(vreg_x_f32_1, max0, preg_134)
        vreg_x_exp_2 = vf.exp_sub(vreg_x_f32_2, max0, preg_134)
        vreg_x_exp_3 = vf.exp_sub(vreg_x_f32_3, max0, preg_134)

        vreg_x_exp_even_f16 = vf.astype(vreg_x_exp_0, preg_135, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16 = vf.astype(vreg_x_exp_2, preg_135, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_pack, vreg_x_exp_f16_packa = vf.de_interleave(vreg_x_exp_even_f16, vreg_x_exp_odd_f16)

        vreg_x_exp_even_f16_1 = vf.astype(vreg_x_exp_1, preg_135, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_odd_f16_1 = vf.astype(vreg_x_exp_3, preg_135, layout=pl.CastLayout.ZERO, dtype=pl.DT_FP16)
        vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa = vf.de_interleave(vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1)

        # Store fp16 result (DATA_BLOCK_COPY + POST_UPDATE)
        vf.store_align(
            x_exp_tile,
            vreg_x_exp_f16_pack,
            preg_136,
            block_stride=blockStride,
            repeat_stride=REPEAT_STRIDE,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_0 = vf.add(vreg_x_exp_0, vreg_x_sum_0, preg_134)
        vreg_x_sum_2 = vf.add(vreg_x_exp_2, vreg_x_sum_2, preg_134)

        vf.store_align(
            x_exp_1,
            vreg_x_exp_f16_1_pack,
            preg_136,
            block_stride=blockStride,
            repeat_stride=REPEAT_STRIDE,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
        vreg_x_sum_1 = vf.add(vreg_x_exp_1, vreg_x_sum_1, preg_134)
        vreg_x_sum_3 = vf.add(vreg_x_exp_3, vreg_x_sum_3, preg_134)

    # ============================================================
    # Phase 3: Sum merge + store
    # ============================================================
    vreg_x_sum0 = vf.add(vreg_x_sum_2, vreg_x_sum_0, preg_134)
    vreg_x_sum1 = vf.add(vreg_x_sum_3, vreg_x_sum_1, preg_134)
    vreg_x_sum0 = vf.add(vreg_x_sum0, vreg_x_sum1, preg_134)
    vf.store_align(sum_tile, vreg_x_sum0, preg_134)


@pl.jit()
def softmax_dn_vf_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out_xexp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out_max: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out_sum: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    # UB tiles
    input_tile_type = pl.TileType(
        shape=[UB_N, M],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    xexp_tile_type = pl.TileType(
        shape=[UB_N, M],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    max_tile_type = pl.TileType(
        shape=[1, M],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    sum_tile_type = pl.TileType(
        shape=[1, M],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )

    input_tile = pl.make_tile(input_tile_type, addr=VA_INPUT, size=INPUT_SIZE)
    x_exp_tile = pl.make_tile(xexp_tile_type, addr=VA_XEXP, size=XEXP_SIZE)
    max_tile = pl.make_tile(max_tile_type, addr=VA_MAX, size=MAX_SIZE)
    sum_tile = pl.make_tile(sum_tile_type, addr=VA_SUM, size=SUM_SIZE)

    with pl.section_vector():
        pl.load(input_tile, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        process_vec1_dn_no_update_vf(input_tile, x_exp_tile, max_tile, sum_tile)

        # Sync and store results
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out_xexp, x_exp_tile, [0, 0])
        pl.store(out_max, max_tile, [0, 0])
        pl.store(out_sum, sum_tile, [0, 0])


def compute_golden(x_dn, device):
    """Compute golden results for softmax DN kernel.

    Returns:
        golden_max: fp32 [64]
        golden_sum: fp32 [64]
        golden_xexp_store: fp16 [128*64] — what pl.store outputs from x_exp_tile
    """
    golden_max = x_dn.max(dim=0).values * D_SCALE
    x_scaled = x_dn * D_SCALE
    x_exp_fp32 = torch.exp(x_scaled - golden_max.unsqueeze(0))
    golden_sum = x_exp_fp32.sum(dim=0)

    # Simulate vsstb layout in UB
    x_exp_fp16 = x_exp_fp32.to(torch.float16)
    golden_xexp_store = simulate_vsstb_layout(x_exp_fp16, device)

    return golden_max, golden_sum, golden_xexp_store


def _simulate_vsstb_layout_legacy(x_exp_fp16, device):
    """Simulate vsstb writes into UB, then return what pl.store would output.

    UB x_exp region: XEXP_SIZE = 16640 bytes = 8320 fp16 elements
    pl.store outputs: x_exp_tile shape [128, 64] = 8192 fp16 elements (first 16384 bytes)

    vsstb writes 8 blocks of 16 fp16 per call:
      block[i] at ptr + i * blockStride * 16 fp16
      POST_UPDATE: ptr += REPEAT_STRIDE * 16 fp16

    Two streams:
      x_exp:   ptr0 starts at offset 0
      x_exp_1: ptr1 starts at offset UB_N*4 = 512 fp16 elements
    """
    block_size = 16  # fp16 elements per block
    num_blocks = 8
    bs_elem = blockStride * block_size  # 65 * 16 = 1040 fp16
    rs_elem = REPEAT_STRIDE * block_size  # 1 * 16 = 16 fp16
    xexp1_offset = UB_N * 4  # 512 fp16

    # UB buffer (in fp16 elements) — use full XEXP_SIZE
    buf_size = XEXP_SIZE // 2  # 8320 fp16 elements
    ub_buf = torch.zeros(buf_size, dtype=torch.float16, device=device)

    x_flat = x_exp_fp16.reshape(-1)  # [8192]

    ptr0 = 0  # x_exp stream
    ptr1 = xexp1_offset  # x_exp_1 stream

    num_iters = UB_N // 4  # 32
    for i0 in range(num_iters):
        # 4 quadrants
        quad0 = x_flat[i0 * M:i0 * M + M]
        quad1 = x_flat[UB_N * M // 4 + i0 * M:UB_N * M // 4 + i0 * M + M]
        quad2 = x_flat[UB_N * M // 2 + i0 * M:UB_N * M // 2 + i0 * M + M]
        quad3 = x_flat[UB_N * M // 2 + UB_N * M // 4 + i0 * M:UB_N * M // 2 + UB_N * M // 4 + i0 * M + M]

        pack0 = torch.cat([quad0, quad2])  # [128]
        pack1 = torch.cat([quad1, quad3])  # [128]

        # vsstb pack0 → x_exp stream
        for blk in range(num_blocks):
            src_s = blk * block_size
            dst_s = ptr0 + blk * bs_elem
            if dst_s + block_size <= buf_size:
                ub_buf[dst_s:dst_s + block_size] = pack0[src_s:src_s + block_size]
        ptr0 += rs_elem

        # vsstb pack1 → x_exp_1 stream
        for blk in range(num_blocks):
            src_s = blk * block_size
            dst_s = ptr1 + blk * bs_elem
            if dst_s + block_size <= buf_size:
                ub_buf[dst_s:dst_s + block_size] = pack1[src_s:src_s + block_size]
        ptr1 += rs_elem

    # pl.store outputs 128*64 = 8192 fp16 elements from UB
    store_size = UB_N * M  # 8192
    return ub_buf[:store_size]


def simulate_vsstb_layout(x_exp_fp16, device):
    """Simulate Cast + DeInterleave + vsstb(DATA_BLOCK_COPY, POST_UPDATE).

    The VF kernel loop (32 iterations) does:
      1. Load 4 quadrants → Muls → ExpSub → 4 fp32 exp regs (each 64 elements)
      2. Cast fp32→fp16 (PART_EVEN): exp_0→even_f16[64], exp_2→odd_f16[64]
      3. DeInterleave(even_f16, odd_f16) → pack[128], packa[128]
         DeInterleave takes odd-indexed elements from src0+src1 into dst0,
         even-indexed elements into dst1:
           src0=[e0,e1,e2,...], src1=[o0,o1,o2,...]
           dst0(pack) =[e0,e2,e4,...,o0,o2,o4,...]  (stored)
           dst1(packa)=[e1,e3,e5,...,o1,o3,o5,...]  (discarded)
      4. vsstb(pack, ptr, (65<<16)|1, mask, POST_UPDATE)
         128 fp16 = 8 blocks x 16 elements
         block[i] at ptr + i * blockStride * 16 fp16
         POST_UPDATE: ptr += REPEAT_STRIDE * 16 fp16

    Args:
        x_exp_fp16: fp16 tensor [UB_N, M] = [128, 64]
        device: torch device

    Returns:
        flat fp16 tensor [8192] matching UB x_exp layout
    """
    block_size = 16
    num_blocks = 8
    bs_elem = blockStride * block_size  # 65 * 16 = 1040 fp16 elements
    rs_elem = REPEAT_STRIDE * block_size  # 1 * 16 = 16 fp16 elements
    xexp1_offset = UB_N * 4  # 512 fp16 elements

    buf_size = XEXP_SIZE // 2  # 8192 fp16 elements
    out_buf = torch.zeros(buf_size, dtype=torch.float16, device=device)

    x_flat = x_exp_fp16.reshape(-1)  # [8192]

    ptr0 = 0  # x_exp stream
    ptr1 = xexp1_offset  # x_exp_1 stream

    num_iters = UB_N // 4  # 32
    for i0 in range(num_iters):
        # 4 quadrants (each 64 fp16 elements)
        quad0 = x_flat[i0 * M:i0 * M + M]
        quad1 = x_flat[UB_N * M // 4 + i0 * M:UB_N * M // 4 + i0 * M + M]
        quad2 = x_flat[UB_N * M // 2 + i0 * M:UB_N * M // 2 + i0 * M + M]
        quad3 = x_flat[UB_N * M // 2 + UB_N * M // 4 + i0 * M:UB_N * M // 2 + UB_N * M // 4 + i0 * M + M]

        # Cast PART_EVEN: fp32[64] → fp16, values placed at even indices of 128-element reg
        #
        # So pack = simple concatenation of the valid fp16 values
        pack0 = torch.cat([quad0, quad2])  # [128]
        pack1 = torch.cat([quad1, quad3])  # [128]

        # vsstb pack0 → x_exp stream
        for blk in range(num_blocks):
            src_s = blk * block_size
            dst_s = ptr0 + blk * bs_elem
            if dst_s + block_size <= buf_size:
                out_buf[dst_s:dst_s + block_size] = pack0[src_s:src_s + block_size]
        ptr0 += rs_elem

        # vsstb pack1 → x_exp_1 stream
        for blk in range(num_blocks):
            src_s = blk * block_size
            dst_s = ptr1 + blk * bs_elem
            if dst_s + block_size <= buf_size:
                out_buf[dst_s:dst_s + block_size] = pack1[src_s:src_s + block_size]
        ptr1 += rs_elem

    return out_buf


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_softmax_dn():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x_dn = torch.rand([UB_N, M], device=device, dtype=torch.float32)
    out_xexp = torch.empty([UB_N, M], device=device, dtype=torch.float16)
    out_max = torch.empty([1, M], device=device, dtype=torch.float32)
    out_sum = torch.empty([1, M], device=device, dtype=torch.float32)

    softmax_dn_vf_kernel(x_dn, out_xexp, out_max, out_sum)
    torch.npu.synchronize()

    golden_max, golden_sum, golden_xexp = compute_golden(x_dn, device)

    npu_max = out_max.squeeze(0)
    npu_sum = out_sum.squeeze(0)
    npu_xexp = out_xexp.reshape(-1)

    # ---- Max check ----
    logging.info("***********npu max***********")
    logging.info("%s", npu_max[:8])
    logging.info("***********golden max***********")
    logging.info("%s", golden_max[:8])
    torch.testing.assert_close(npu_max, golden_max, rtol=1e-3, atol=1e-3)
    logging.info("Max check PASSED")

    # ---- Sum check ----
    logging.info("***********npu sum***********")
    logging.info("%s", npu_sum[:8])
    logging.info("***********golden sum***********")
    logging.info("%s", golden_sum[:8])
    torch.testing.assert_close(npu_sum, golden_sum, rtol=1e-2, atol=1e-2)
    logging.info("Sum check PASSED")

    # ---- x_exp layout check ----
    logging.info("***********npu x_exp (first 32)***********")
    logging.info("%s", npu_xexp[:32])
    logging.info("***********golden x_exp (first 32)***********")
    logging.info("%s", golden_xexp[:32])
    torch.testing.assert_close(npu_xexp, golden_xexp, rtol=1e-2, atol=1e-2)
    logging.info("x_exp layout check PASSED")

    logging.info("Softmax DN VF API (full version) test PASSED!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_softmax_dn()
    logging.info("\nAll tests passed!")
