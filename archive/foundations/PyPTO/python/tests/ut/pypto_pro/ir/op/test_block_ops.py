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
"""Unit tests for explicit-output block operations."""

import inspect

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserTypeError
import pytest

from pypto.pypto_impl import ir as _ir


def _program_ir(program) -> str:
    return str(program)


def test_coordinate_apis_use_sequence_parameters():
    assert list(inspect.signature(pl.insert).parameters) == ["dst_tile", "src_tile", "offset"]
    assert list(inspect.signature(pl.set_validshape).parameters) == ["tile", "shape"]


def test_manual_add():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            b: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x0000, size=4096)
            tile_b = pl.make_tile(tile_type, addr=0x1000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x2000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.load(tile_b, b, [0, 0])
            pl.add(tile_out, tile_a, tile_b)
            return pl.store(output, tile_out, [0, 0])

    assert "block.add" in _program_ir(Program)


def test_manual_mul_scalar():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x3000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x4000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.mul(tile_out, tile_a, 2.0)
            return pl.store(output, tile_out, [0, 0])

    assert "block.muls" in _program_ir(Program)


def test_manual_add_scalar():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x3000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x4000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.add(tile_out, tile_a, 1.0)
            return pl.store(output, tile_out, [0, 0])

    assert "block.adds" in _program_ir(Program)


def test_manual_maximum_scalar():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x3000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x4000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.maximum(tile_out, tile_a, 0.0)
            return pl.store(output, tile_out, [0, 0])

    assert "block.maxs" in _program_ir(Program)


def test_manual_relu():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x5000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x6000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.relu(tile_out, tile_a)
            return pl.store(output, tile_out, [0, 0])

    assert "block.relu" in _program_ir(Program)


def test_manual_cmp():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            b: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x7000, size=4096)
            tile_b = pl.make_tile(tile_type, addr=0x8000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x9000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.load(tile_b, b, [0, 0])
            pl.eq(tile_out, tile_a, tile_b)
            return pl.store(output, tile_out, [0, 0])

    assert "block.cmp" in _program_ir(Program)


def test_manual_row_max():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            inp: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 1], pl.DT_FP32],
        ) -> pl.Tensor[[128, 1], pl.DT_FP32]:
            in_type = pl.TileType(shape=[32, 128], dtype=pl.DT_FP32)
            out_type = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32)
            tile_in = pl.make_tile(in_type, addr=0xA000, size=16384)
            tmp = pl.make_tile(out_type, addr=0xE000, size=128)
            tile_out = pl.make_tile(out_type, addr=0x12000, size=128)
            pl.load(tile_in, inp, [0, 0])
            pl.maximum(tile_out, tile_in, tmp, dim=0)
            return pl.store(output, tile_out, [0, 0])

    assert "block.row_max" in _program_ir(Program)


def test_manual_col_expand_mul():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            col: pl.Tensor[[128, 128], pl.DT_FP32],
            tile: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            col_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x13000, size=4096)
            tile_col = pl.make_tile(col_type, addr=0x14000, size=128)
            tile_out = pl.make_tile(tile_type, addr=0x15000, size=4096)
            pl.load(tile_a, tile, [0, 0])
            pl.load(tile_col, col, [0, 0])
            pl.expand_mul(tile_out, tile_a, tile_col, dim=1)
            return pl.store(output, tile_out, [0, 0])

    assert "block.col_expand_mul" in _program_ir(Program)


def test_manual_col_expand_div():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            col: pl.Tensor[[128, 128], pl.DT_FP32],
            tile: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            col_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x13000, size=4096)
            tile_col = pl.make_tile(col_type, addr=0x14000, size=128)
            tile_out = pl.make_tile(tile_type, addr=0x15000, size=4096)
            pl.load(tile_a, tile, [0, 0])
            pl.load(tile_col, col, [0, 0])
            pl.expand_div(tile_out, tile_a, tile_col, dim=1)
            return pl.store(output, tile_out, [0, 0])

    assert "block.col_expand_div" in _program_ir(Program)


def test_manual_and_scalar():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_INT32],
            output: pl.Tensor[[128, 128], pl.DT_INT32],
        ) -> pl.Tensor[[128, 128], pl.DT_INT32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32)
            tile_a = pl.make_tile(tile_type, addr=0x16000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x17000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.and_(tile_out, tile_a, 7)
            return pl.store(output, tile_out, [0, 0])

    assert "block.ands" in _program_ir(Program)


def test_manual_xor():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_INT32],
            b: pl.Tensor[[128, 128], pl.DT_INT32],
            output: pl.Tensor[[128, 128], pl.DT_INT32],
        ) -> pl.Tensor[[128, 128], pl.DT_INT32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32)
            tile_a = pl.make_tile(tile_type, addr=0x18000, size=4096)
            tile_b = pl.make_tile(tile_type, addr=0x19000, size=4096)
            tmp = pl.make_tile(tile_type, addr=0x1A000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x1B000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.load(tile_b, b, [0, 0])
            pl.xor(tile_out, tile_a, tile_b, tmp)
            return pl.store(output, tile_out, [0, 0])

    assert "block.xor" in _program_ir(Program)


def test_manual_addc():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            b: pl.Tensor[[128, 128], pl.DT_FP32],
            c: pl.Tensor[[128, 128], pl.DT_FP32],
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            tile_a = pl.make_tile(tile_type, addr=0x1C000, size=4096)
            tile_b = pl.make_tile(tile_type, addr=0x1D000, size=4096)
            tile_c = pl.make_tile(tile_type, addr=0x1E000, size=4096)
            tile_out = pl.make_tile(tile_type, addr=0x1F000, size=4096)
            pl.load(tile_a, a, [0, 0])
            pl.load(tile_b, b, [0, 0])
            pl.load(tile_c, c, [0, 0])
            pl.addc(tile_out, tile_a, tile_b, tile_c)
            return pl.store(output, tile_out, [0, 0])

    assert "block.addc" in _program_ir(Program)


def test_load_with_dynamic_valid_shape():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP32],
            rows: pl.DT_INT64,
            cols: pl.DT_INT64,
            output: pl.Tensor[[128, 128], pl.DT_FP32],
        ) -> pl.Tensor[[128, 128], pl.DT_FP32]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, valid_shape=[-1, -1])
            tile = pl.make_tile(tile_type, addr=0x20000, size=65536)
            pl.load(tile, a, [0, 0])
            pl.set_validshape(tile, [rows, cols])
            return pl.store(output, tile, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.load" in ir_str
    assert "block.set_validshape" in ir_str


def test_insert_with_offset_sequence():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            output: pl.Tensor[[32, 32], pl.DT_FP32],
        ) -> pl.Tensor[[32, 32], pl.DT_FP32]:
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
            dst = pl.make_tile(dst_type, addr=0x30200, size=4096)
            src = pl.make_tile(src_type, addr=0x31200, size=1024)
            pl.insert(dst, src, [8, 16])
            return pl.store(output, dst, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.insert" in ir_str


def test_manual_transpose():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            src_type = pl.TileType(shape=[8, 16], dtype=pl.DT_FP16)
            dst_type = pl.TileType(shape=[16, 8], dtype=pl.DT_FP16)
            src = pl.make_tile(src_type, addr=0x30000, size=256)
            dst = pl.make_tile(dst_type, addr=0x30100, size=256)
            pl.load(src, a, [0, 0])
            pl.transpose(dst, src)
            return pl.store(output, dst, [0, 0])

    assert "block.transpose" in _program_ir(Program)


def test_get_block_idx_without_prefix():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, output: pl.Tensor[[1], pl.DT_INT64]) -> pl.Tensor[[1], pl.DT_INT64]:
            _ = pl.get_block_idx()
            return output

    assert "get_block_idx" in _program_ir(Program)


def test_set_validshape_tile_group_4buf():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            rows: pl.DT_INT64,
            cols: pl.DT_INT64,
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
            a_db = pl.make_tile_group(
                type=tile_type, addrs=0x0000, mutex_ids=[0, 1, 2, 3])
            pl.set_validshape(a_db, [rows, cols])
            tile_a = a_db.next()
            pl.load(tile_a, a, [0, 0])
            return pl.store(output, tile_a, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.set_validshape" in ir_str
    vs_pos = ir_str.index("block.set_validshape")
    load_pos = ir_str.index("block.load")
    assert vs_pos < load_pos


def test_set_validshape_tile_group_single_tile():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            cols: pl.DT_INT64,
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
            a_db = pl.make_tile_group(
                type=tile_type, addrs=0x0000, mutex_ids=[0])
            pl.set_validshape(a_db, [128, cols])
            tile_a = a_db.next()
            pl.load(tile_a, a, [0, 0])
            return pl.store(output, tile_a, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.set_validshape" in ir_str
    vs_pos = ir_str.index("block.set_validshape")
    load_pos = ir_str.index("block.load")
    assert vs_pos < load_pos


def test_move_to_insert_subblock():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            output: pl.Tensor[[32, 32], pl.DT_FP32],
        ) -> pl.Tensor[[32, 32], pl.DT_FP32]:
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
            dst = pl.make_tile(dst_type, addr=0x30200, size=4096)
            src = pl.make_tile(src_type, addr=0x31200, size=1024)
            pl.move(dst, src, [8, 8])
            return pl.store(output, dst, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.insert" in ir_str
    assert "block.move" not in ir_str


def test_move_to_insert_transpose_layout():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            output: pl.Tensor[[32, 32], pl.DT_FP32],
        ) -> pl.Tensor[[32, 32], pl.DT_FP32]:
            dst_type = pl.TileType(
                shape=[32, 64], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec, layout=pl.ZN)
            src_type = pl.TileType(
                shape=[64, 16], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
            dst = pl.make_tile(dst_type, addr=0x30200, size=16384)
            src = pl.make_tile(src_type, addr=0x31200, size=4096)
            pl.move(dst, src, [0, 0])
            return pl.store(output, dst, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.insert" in ir_str
    assert "block.move" not in ir_str


def test_move_equal_shape_stays_move():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            output: pl.Tensor[[32, 32], pl.DT_FP32],
        ) -> pl.Tensor[[32, 32], pl.DT_FP32]:
            dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            dst = pl.make_tile(dst_type, addr=0x30200, size=4096)
            src = pl.make_tile(src_type, addr=0x31200, size=4096)
            pl.move(dst, src, [0, 0])
            return pl.store(output, dst, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.move" in ir_str


def test_move_src_larger_stays_move():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            output: pl.Tensor[[32, 32], pl.DT_FP32],
        ) -> pl.Tensor[[32, 32], pl.DT_FP32]:
            dst_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
            src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
            dst = pl.make_tile(dst_type, addr=0x30200, size=1024)
            src = pl.make_tile(src_type, addr=0x31200, size=4096)
            pl.move(dst, src, [0, 0])
            return pl.store(output, dst, [0, 0])

    ir_str = _program_ir(Program)
    assert "block.move" in ir_str


def test_make_tile_missing_addr_rejected():
    with pytest.raises(ParserTypeError, match="missing required keyword 'addr'"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, size=32768)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])


def test_make_tile_size_defaults_to_tile_footprint():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x0000)
            pl.load(tile_a, a, [0, 0])
            return pl.store(output, tile_a, [0, 0])

    # 128 * 128 elements * 2 bytes (FP16)
    assert 'memref_size=32768' in _program_ir(Program)


def test_make_tile_explicit_size_overrides_derived():
    """An NZ tile rounded up to whole fractals reserves more than shape * dtype."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x0000, size=40960)
            pl.load(tile_a, a, [0, 0])
            return pl.store(output, tile_a, [0, 0])

    ir_str = _program_ir(Program)
    assert 'memref_size=40960' in ir_str
    assert 'memref_size=32768' not in ir_str


def test_make_tile_runtime_size_rejected_with_compile_time_hint():
    with pytest.raises(ParserTypeError) as excinfo:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, addr=0x0000, size=a.shape[0] * 32)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])

    assert "compile-time integer" in str(excinfo.value)


def test_make_tile_runtime_addr_rejected_with_compile_time_hint():
    with pytest.raises(ParserTypeError) as excinfo:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, addr=a.shape[0] * 32, size=32768)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])

    assert "compile-time integer" in str(excinfo.value)
    assert "runtime value" in str(excinfo.value)


def test_make_tile_non_positive_size_rejected():
    with pytest.raises(ParserTypeError, match="positive byte count"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, addr=0x0000, size=0)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])


def test_make_tile_addr_after_tile_type():
    """The documented prototype: pl.make_tile(tile_type, *, addr, size=None)."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x1000)
            pl.load(tile_a, a, [0, 0])
            return pl.store(output, tile_a, [0, 0])

    ir_str = _program_ir(Program)
    assert 'memref_addr=4096' in ir_str
    assert 'memref_size=32768' in ir_str


def test_make_tile_addr_and_size_after_tile_type():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x1000, size=40960)
            pl.load(tile_a, a, [0, 0])
            return pl.store(output, tile_a, [0, 0])

    ir_str = _program_ir(Program)
    assert 'memref_addr=4096' in ir_str
    assert 'memref_size=40960' in ir_str


def test_make_tile_builder_form_rejected():
    """make_tile(shape, dtype, target_memory, ...) is the IR builder, not the DSL op."""

    with pytest.raises(ParserTypeError, match="takes a pl.TileType as its first argument"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_a = pl.make_tile([128, 128], pl.DT_FP16, pl.MemorySpace.Vec, 0x2000, 32768)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])


def test_make_tile_misaligned_addr_rejected():
    """addr reaches the alignment check as a plain int, whatever expression wrote it."""

    with pytest.raises(ValueError, match="32-byte aligned"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, addr=0x1)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])


def test_make_tile_runtime_addr_rejected_quoting_source():
    with pytest.raises(ParserTypeError) as excinfo:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, addr=a.shape[0] * 32)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])

    assert "compile-time integer" in str(excinfo.value)
    # the offending expression, not a repr of the parsed IR object
    assert "a.shape[0] * 32" in str(excinfo.value)


def test_make_tile_positional_addr_rejected():
    """The tile type is the only positional argument; addr/size are keywords."""

    with pytest.raises(ParserTypeError, match="takes 1 positional argument .* but 2 were given"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, 0x1000)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])


def test_make_tile_positional_addr_and_size_rejected_with_keyword_hint():
    """The rejection carries the fix, so the call site does not have to guess."""

    with pytest.raises(ParserTypeError) as excinfo:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.DT_FP16],
                output: pl.Tensor[[128, 128], pl.DT_FP16],
            ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
                tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
                tile_a = pl.make_tile(tile_type, 0x0000, 32768)
                pl.load(tile_a, a, [0, 0])
                return pl.store(output, tile_a, [0, 0])

    assert "takes 1 positional argument (the tile type) but 3 were given" in str(excinfo.value)
    # the hint spells out the accepted spelling
    assert "addr=" in str(excinfo.value)


VEC_BASE = 0x1000


def test_make_tile_addr_from_constant_expression():
    """addr goes through the full compile-time path, not just literals."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ) -> pl.Tensor[[128, 128], pl.DT_FP16]:
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=VEC_BASE + 0x20)
            pl.load(tile_a, a, [0, 0])
            return pl.store(output, tile_a, [0, 0])

    assert 'memref_addr=4128' in _program_ir(Program)


@pytest.mark.parametrize("memory", [pl.MemorySpace.Left, pl.MemorySpace.Right, pl.MemorySpace.Acc])
@pytest.mark.parametrize("layout", [pl.ZZ, pl.NN])
def test_a5_rejects_zz_nn_for_cube_buffers(monkeypatch, memory, layout):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    with pytest.raises(pl.parser.ParserError, match="do not support"):

        @pl.function
        def create_tile():
            tt = pl.TileType(
                shape=[128, 128],
                dtype=pl.DT_FP16,
                target_memory=memory,
                layout=layout,
            )
            tile = pl.make_tile(tt, addr=0x00000, size=32768)  # noqa: F841


def test_a5_allows_regular_cube_buffer_layouts(monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    @pl.function
    def create_tile():
        tt = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
        )
        tile = pl.make_tile(tt, addr=0x00000, size=32768)  # noqa: F841

    stmt = create_tile.body.stmts[0] if hasattr(create_tile.body, "stmts") else create_tile.body
    assert isinstance(stmt.var.type.hardware_info, _ir.HardwareInfo)
