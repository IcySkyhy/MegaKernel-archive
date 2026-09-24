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

"""Frontend tests for pl.histogram function.

Tests histogram accumulation for radix sort preprocessing.
- UINT16 mode: isMSB=True (bits 15-8) or isMSB=False (bits 7-0 with idx filter)
- UINT32 mode: BYTE_3/2/1/0 for radix sort passes

THISTOGRAM Constraints (from PTOAS docs):
  1. src, idx, dst must all be tile_buf in loc=vec
  2. src/dst must use row_major + none_box layout
  3. idx must use col_major + none_box layout (DN-style)
  4. src dtype = ui16, idx dtype = ui8, dst dtype = ui32
  5. All must be rank-2 tiles
  6. idx rows must match src rows
  7. dst rows must match src rows
  8. idx must have exactly 1 column
  9. dst shape[1] >= 256

Note: THISTOGRAM is only supported on A5, not A2/A3.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


# ============================================================================
# Helper functions
# ============================================================================


def verify_histogram_output(
    tag: str,
    histogram_out: torch.Tensor,
    expected_histogram: torch.Tensor,
    test_name: str,
):
    """Verify histogram output matches expected reference.

    Args:
        tag: Backend mode (CCE or PTO)
        histogram_out: NPU output histogram
        expected_histogram: Reference histogram computed in Python
        test_name: Test case name for error messages
    """
    logging.info("\n--- %s mode ---", tag)

    # Print sample outputs
    logging.info("  [%s] histogram (row 0, bins 0-9): %s", tag, histogram_out[0, :10].tolist())
    logging.info("  [ref] histogram (row 0, bins 0-9): %s", expected_histogram[0, :10].tolist())
    logging.info("  [%s] total counts per row: %s", tag, histogram_out.sum(dim=1).tolist())
    logging.info("  [ref] total counts per row: %s", expected_histogram.sum(dim=1).tolist())

    # Compute max diff
    diff = torch.abs(histogram_out - expected_histogram).max().item()
    logging.info("  Max diff (%s vs ref): %s", tag, diff)

    # Verify
    assert diff == 0, f"{tag} {test_name} failed: max diff {diff}"
    logging.info("%s %s test passed!", tag, test_name)


# ============================================================================
# UINT16 histogram tests (BYTE_1/BYTE_0)
# ============================================================================

ROWS = 32  # ColMajor UINT8 alignment requires Rows * sizeof(T) % 32 == 0 → Rows % 32 == 0
COLS = 128
IDX_COLS_DN = 1  # DN (ColMajor) idx has 1 logical column; alignment satisfied by ROWS=32

# Buffer sizes (static constants for make_tile)
SRC_SIZE = ROWS * COLS * 2  # UINT16: 32 * 128 * 2 = 8192 bytes
IDX_SIZE = ROWS * IDX_COLS_DN  # UINT8 DN layout: 32 * 1 = 32 bytes
DST_SIZE = ROWS * 256 * 4  # UINT32: 32 * 256 * 4 = 32768 bytes


@pl.jit()
def histogram_uint16_msb_kernel_cce(
    src: pl.Tensor[[ROWS, COLS], pl.DT_UINT16],
    idx: pl.Tensor[[ROWS, IDX_COLS_DN], pl.DT_UINT8],
    histogram_out: pl.Tensor[[ROWS, 256], pl.DT_UINT32],
):
    """Build 256-bin histogram for uint16 data (MSB mode, bits 15-8).

    idx tile has shape (rows, IDX_COLS_DN) with DN layout (ColMajor), but is unused in MSB mode.
    Output: 256 bins per row counting frequency of each byte value.

    Note: DN layout requires Cols aligned to 32 bytes, so tensor/tile shape=[ROWS, IDX_COLS_DN]
          but we only use column 0 (logical shape=[ROWS, 1]).
    """
    pl.system.bar_all()

    # src: ND format (row_major + none_box)
    tile_src = pl.make_tile(
        pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addr=0x0000,
        size=SRC_SIZE,
    )

    # idx: DN format (col_major + none_box)
    # ColMajor alignment: Rows * sizeof(T) % 32 == 0 → ROWS=32 satisfies this for uint8
    tile_idx = pl.make_tile(
        pl.TileType(
            shape=[ROWS, IDX_COLS_DN],  # [32, 1]: 1 logical column, alignment from ROWS=32
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.DN,
        ),
        addr=0x2000,  # After tile_src (0x0000 + 8192 = 0x2000)
        size=IDX_SIZE,
    )

    # dst: ND format (row_major + none_box)
    tile_dst = pl.make_tile(
        pl.TileType(shape=[ROWS, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addr=0x2020,  # After tile_idx (0x2000 + 32 = 0x2020)
        size=DST_SIZE,
    )

    with pl.section_vector():
        pl.load(tile_src, src, [0, 0])
        pl.load(tile_idx, idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.histogram(tile_dst, tile_src, tile_idx, is_msb=True)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(histogram_out, tile_dst, [0, 0])


@pl.jit()
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_histogram_uint16_msb():
    """Test histogram with uint16 input, MSB mode (bits 15-8)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    torch.manual_seed(42)

    rows = ROWS
    cols = COLS
    idx_cols = IDX_COLS_DN  # DN layout requires physical cols aligned to 32

    logging.info("\n=== histogram uint16 MSB mode test ===")
    logging.info("Input shape: (%s, %s), dtype=UINT16", rows, cols)
    logging.info("Index shape: (%s, %s), dtype=UINT8 (DN layout)", rows, idx_cols)
    logging.info("Mode: isMSB=True (bits 15-8)")

    src = torch.randint(0, 65536, (rows, cols), device=device, dtype=torch.int32)
    src_uint16 = src.to(torch.uint16)

    # idx tensor must match tile physical shape [ROWS, IDX_COLS_DN]
    # Only column 0 is used (logical shape=[ROWS, 1])
    idx = torch.zeros((rows, idx_cols), device=device, dtype=torch.uint8)

    logging.info("Sample input (row 0, first 10 values): %s", src[0, :10].tolist())

    expected_histogram = torch.zeros((rows, 256), device=device, dtype=torch.int32)
    # Step 1: 统计每个值的次数
    for r in range(rows):
        for c in range(cols):
            val = src[r, c].item()
            msb_byte = (val >> 8) & 0xFF
            expected_histogram[r, msb_byte] += 1

    # Step 2: 计算前缀和（按bin顺序累加）
    # THISTOGRAM输出是递推前缀和：dst[bin] = dst[bin-1] + count[bin]
    for r in range(rows):
        acc = 0
        for bin_val in range(256):
            acc += expected_histogram[r, bin_val].item()
            expected_histogram[r, bin_val] = acc

    logging.info("\nExpected histogram (row 0, first 10 bins): %s", expected_histogram[0, :10].tolist())
    logging.info("Expected histogram total counts per row: %s", expected_histogram.sum(dim=1).tolist())

    # --- CCE ---
    histogram_out = torch.zeros((rows, 256), device=device, dtype=torch.int32)
    histogram_uint16_msb_kernel_cce(src_uint16, idx, histogram_out)
    torch.npu.synchronize()
    verify_histogram_output("CCE", histogram_out, expected_histogram, "histogram_uint16_msb")


# ============================================================================
# UINT16 histogram LSB mode (filtered)
# ============================================================================


@pl.jit()
def histogram_uint16_lsb_kernel_cce(
    src: pl.Tensor[[ROWS, COLS], pl.DT_UINT16],
    idx: pl.Tensor[[ROWS, IDX_COLS_DN], pl.DT_UINT8],
    histogram_out: pl.Tensor[[ROWS, 256], pl.DT_UINT32],
):
    """Build 256-bin histogram for uint16 data (LSB mode, bits 7-0, filtered).

    idx tile has shape (rows, IDX_COLS_DN) with DN layout (ColMajor), dtype=UINT8.
    LSB mode filters: only count elements where MSB == idx[row, 0].

    Note: DN layout requires Cols aligned to 32 bytes, so tensor/tile shape=[ROWS, IDX_COLS_DN]
          but we only use column 0 (logical shape=[ROWS, 1]).
    """
    pl.system.bar_all()

    tile_src = pl.make_tile(
        pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addr=0x0000,
        size=SRC_SIZE,
    )

    # idx: col_major + none_box (DN layout)
    # ColMajor alignment: Rows * sizeof(T) % 32 == 0 → ROWS=32 satisfies this for uint8
    tile_idx = pl.make_tile(
        pl.TileType(
            shape=[ROWS, IDX_COLS_DN],  # [32, 1]: 1 logical column, alignment from ROWS=32
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.DN,
        ),
        addr=0x2000,  # After tile_src (0x0000 + 8192 = 0x2000)
        size=IDX_SIZE,
    )

    tile_dst = pl.make_tile(
        pl.TileType(shape=[ROWS, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addr=0x2020,  # After tile_idx (0x2000 + 32 = 0x2020)
        size=DST_SIZE,
    )

    with pl.section_vector():
        pl.load(tile_src, src, [0, 0])
        pl.load(tile_idx, idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.histogram(tile_dst, tile_src, tile_idx, is_msb=False)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(histogram_out, tile_dst, [0, 0])


@pl.jit()
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_histogram_uint16_lsb():
    """Test histogram with uint16 input, LSB mode (bits 7-0, filtered by MSB)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    torch.manual_seed(100)

    rows = ROWS
    cols = COLS
    idx_cols = IDX_COLS_DN  # DN layout requires physical cols aligned to 32

    filter_values = [
        0xAB,
        0xCD,
        0x12,
        0x34,
        0x56,
        0x78,
        0x9A,
        0xBC,
        0xDE,
        0xF0,
        0x11,
        0x22,
        0x33,
        0x44,
        0x55,
        0x66,
        0x77,
        0x88,
        0x99,
        0xAA,
        0xBB,
        0xCC,
        0xDD,
        0xEE,
        0xFF,
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x07,
    ]

    logging.info("\n=== histogram uint16 LSB mode test ===")
    logging.info("Input shape: (%s, %s), dtype=UINT16", rows, cols)
    logging.info("Index shape: (%s, %s), dtype=UINT8 (DN layout)", rows, idx_cols)
    logging.info("Mode: isMSB=False (bits 7-0, filtered by MSB==idx)")
    logging.info("Filter MSB values per row: %s", filter_values)

    src = torch.randint(0, 65536, (rows, cols), device=device, dtype=torch.int32)

    logging.info("Sample input (row 0, first 10 values): %s", src[0, :10].tolist())

    # idx tensor must match tile physical shape [ROWS, IDX_COLS_DN]
    # Column 0 stores the filter values, other columns are padding
    idx = torch.zeros((rows, idx_cols), device=device, dtype=torch.uint8)
    for r, val in enumerate(filter_values):
        idx[r, 0] = val  # Only set column 0

    expected_histogram = torch.zeros((rows, 256), device=device, dtype=torch.int32)
    # Step 1: 统计每个值的次数
    for r in range(rows):
        filter_msb = filter_values[r]
        for c in range(cols):
            val = src[r, c].item()
            msb_byte = (val >> 8) & 0xFF
            if msb_byte == filter_msb:
                lsb_byte = val & 0xFF
                expected_histogram[r, lsb_byte] += 1

    # Step 2: 计算前缀和（按bin顺序累加）
    for r in range(rows):
        acc = 0
        for bin_val in range(256):
            acc += expected_histogram[r, bin_val].item()
            expected_histogram[r, bin_val] = acc

    logging.info("\nExpected histogram (row 0, first 10 bins): %s", expected_histogram[0, :10].tolist())
    logging.info("Expected histogram total counts per row: %s", expected_histogram.sum(dim=1).tolist())

    src_uint16 = src.to(torch.uint16)

    # --- CCE ---
    histogram_out = torch.zeros((rows, 256), device=device, dtype=torch.int32)
    histogram_uint16_lsb_kernel_cce(src_uint16, idx, histogram_out)
    torch.npu.synchronize()
    verify_histogram_output("CCE", histogram_out, expected_histogram, "histogram_uint16_lsb")


# ============================================================================
# UINT32 histogram BYTE_3 (MSB, first radix pass)
# ============================================================================

ROWS_U32 = 32  # ColMajor UINT8 alignment: Rows * sizeof(T) % 32 == 0 → Rows % 32 == 0
COLS_U32 = 128

SRC_SIZE_U32 = ROWS_U32 * COLS_U32 * 2  # UINT16: 32 * 128 * 2 = 8192 bytes
IDX_SIZE_U32 = ROWS_U32 * 1  # UINT8 DN layout: 32 * 1 = 32 bytes
DST_SIZE_U32 = ROWS_U32 * 256 * 4  # UINT32: 32 * 256 * 4 = 32768 bytes


@pl.jit()
def histogram_uint32_byte3_kernel_cce(
    src: pl.Tensor[[ROWS_U32, COLS_U32], pl.DT_UINT16],
    idx_dummy: pl.Tensor[[ROWS_U32, 1], pl.DT_UINT8],
    histogram_out: pl.Tensor[[ROWS_U32, 256], pl.DT_UINT32],
):
    """Build 256-bin histogram for uint16 data (MSB mode, bits 15-8).

    THISTOGRAM src must be UINT16 (hardware constraint).
    idx tile is unused for filtering but must use DN layout (col_major + none_box)
    as required by THISTOGRAM for all modes.
    ColMajor alignment: Rows * sizeof(UINT8) % 32 == 0 → ROWS_U32=32 satisfies this.
    """
    pl.system.bar_all()

    tile_src = pl.make_tile(
        pl.TileType(shape=[ROWS_U32, COLS_U32], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addr=0x0000,
        size=SRC_SIZE_U32,
    )

    # idx: col_major + none_box (DN layout) — required by THISTOGRAM regardless of mode
    tile_idx = pl.make_tile(
        pl.TileType(shape=[ROWS_U32, 1], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addr=0x2000,  # After tile_src (0x0000 + 8192 = 0x2000)
        size=IDX_SIZE_U32,
    )

    tile_dst = pl.make_tile(
        pl.TileType(shape=[ROWS_U32, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec, layout=pl.ND),
        addr=0x2020,  # After tile_idx (0x2000 + 32 = 0x2020)
        size=DST_SIZE_U32,
    )

    with pl.section_vector():
        pl.load(tile_src, src, [0, 0])
        pl.load(tile_idx, idx_dummy, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.histogram(tile_dst, tile_src, tile_idx, is_msb=True)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(histogram_out, tile_dst, [0, 0])


@pl.jit()
@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_histogram_uint32_byte3():
    """Test histogram with uint16 input, MSB mode (bits 15-8), ROWS_U32 rows.

    Note: THISTOGRAM src must be UINT16 (hardware constraint). This test uses
    UINT16 data with is_msb=True and a different row count than the UINT16 tests.
    """
    device = ST_DEVICE
    torch.npu.set_device(device)

    torch.manual_seed(200)

    rows = ROWS_U32
    cols = COLS_U32

    logging.info("\n=== histogram uint16 MSB (ROWS_U32 variant) test ===")
    logging.info("Input shape: (%s, %s), dtype=UINT16", rows, cols)
    logging.info("Mode: isMSB=True (bits 15-8)")

    src = torch.randint(0, 65536, (rows, cols), device=device, dtype=torch.int32)
    src_uint16 = src.to(torch.uint16)

    idx_dummy = torch.zeros((rows, 1), device=device, dtype=torch.uint8)

    logging.info("Sample input (row 0, first 10 values): %s", src[0, :10].tolist())

    expected_histogram = torch.zeros((rows, 256), device=device, dtype=torch.int32)
    # Step 1: 统计每个值的次数
    for r in range(rows):
        for c in range(cols):
            val = src[r, c].item()
            msb_byte = (val >> 8) & 0xFF
            expected_histogram[r, msb_byte] += 1

    # Step 2: 计算前缀和（按bin顺序累加）
    for r in range(rows):
        acc = 0
        for bin_val in range(256):
            acc += expected_histogram[r, bin_val].item()
            expected_histogram[r, bin_val] = acc

    logging.info("\nExpected histogram (row 0, first 10 bins): %s", expected_histogram[0, :10].tolist())
    logging.info("Expected histogram total counts per row: %s", expected_histogram.sum(dim=1).tolist())

    # --- CCE ---
    histogram_out = torch.zeros((rows, 256), device=device, dtype=torch.int32)
    histogram_uint32_byte3_kernel_cce(src_uint16, idx_dummy, histogram_out)
    torch.npu.synchronize()
    verify_histogram_output("CCE", histogram_out, expected_histogram, "histogram_uint32_byte3")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("%s", "=" * 60)
    logging.info("histogram tests (radix sort preprocessing)")
    logging.info("%s", "=" * 60)
    test_histogram_uint16_msb()
    test_histogram_uint16_lsb()
    test_histogram_uint32_byte3()
    logging.info("\n%s", "=" * 60)
    logging.info("All histogram tests passed!")
    logging.info("%s", "=" * 60)
