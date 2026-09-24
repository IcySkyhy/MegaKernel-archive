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

"""Parser tests for the Tile-first A5 SIMT frontend."""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError, UnsupportedFeatureError
import pytest

from pypto.pypto_impl import ir


@pl.simt.function(max_threads=256)
def _tile_add(
    dst,
    src,
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.simt.function(max_threads=256)
def _gm_add(
    dst: pl.Tensor[[1, 256], pl.DT_FP32],
    src: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.simt.function
def _callee_add(value: pl.DT_INT32, delta: pl.DT_INT32) -> pl.DT_INT32:
    return value + delta


@pl.simt.function
def _callee_store(
    dst,
    index: pl.DT_UINT32,
    value: pl.DT_INT32,
):
    dst[0, index] = value


@pl.simt.function
def _callee_apply(
    dst,
    src,
    index: pl.DT_UINT32,
    delta: pl.DT_INT32,
):
    value = _callee_add(src[0, index], delta)
    _callee_store(dst, index, value)


@pl.simt.function(max_threads=32)
def _callee_entry(
    dst,
    src,
    delta: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    _callee_apply(dst, src, tid, delta)


@pl.simt.function(max_threads=256)
def _context_probe(dst):
    thread = pl.simt.thread_idx()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    grid = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()
    value = (
        thread.x
        + thread.y
        + thread.z
        + block.x
        + block.y
        + block.z
        + block_id.x
        + block_id.y
        + block_id.z
        + grid.x
        + grid.y
        + grid.z
        + pl.simt.warp_size()
    )
    dst[0, tid] = value


@pl.simt.function(max_threads=256)
def _tile_valid_shape_access(
    dst,
    src,
):
    tid = pl.simt.linear_thread_idx()
    rows = src.valid_shape[0]
    cols = src.valid_shape[1]
    row = tid // cols
    col = tid % cols
    if row < rows:
        dst[row, col] = src[row, col]


@pl.kernel
def _simt_tile_kernel(n: pl.DT_UINT32, delta: pl.DT_FP32):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tile_type, addr=0x0000, size=1024)
    dst = pl.make_tile(tile_type, addr=0x0400, size=1024)
    with pl.section_vector():
        pl.simt.launch(_tile_add, threads=256, args=(dst, src, n, delta))


@pl.kernel
def _simt_gm_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    with pl.section_vector():
        pl.simt.launch(_gm_add, threads=256, args=(out, x, n, delta))


@pl.kernel
def _simt_context_kernel():
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000, size=1024)
    with pl.section_vector():
        pl.simt.launch(_context_probe, threads=(8, 4, 8), args=(dst,))


@pl.kernel
def _simt_callee_kernel(delta: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000, size=128)
    src = pl.make_tile(tile_type, addr=0x0080, size=128)
    with pl.section_vector():
        pl.simt.launch(_callee_entry, threads=32, args=(dst, src, delta))


def _make_tile_launch_kernel(shape, dtype, target_memory, layout):
    @pl.kernel
    def kernel(n: pl.DT_UINT32, delta: pl.DT_FP32):
        tile_type = pl.TileType(
            shape=shape,
            dtype=dtype,
            target_memory=target_memory,
            layout=layout,
        )
        src = pl.make_tile(tile_type, addr=0x0000, size=1024)
        dst = pl.make_tile(tile_type, addr=0x0400, size=1024)
        with pl.section_vector():
            pl.simt.launch(_tile_add, threads=256, args=(dst, src, n, delta))

    return kernel


def test_simt_function_requires_max_threads_or_direct_decoration():
    with pytest.raises(TypeError, match="requires max_threads or a directly decorated function"):
        pl.simt.function()


def test_legacy_simt_decorators_are_not_exported():
    assert not hasattr(pl, "simt_function")
    assert not hasattr(pl, "simt_callee")


def test_simt_tile_function_and_launch_build_vector_program():
    program, matched = _simt_tile_kernel.parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("_tile_add")

    assert matched
    assert callable(_tile_add)
    assert not isinstance(_tile_add, ir.Function)
    assert function.func_type == ir.FunctionType.SimtVF
    assert function.has_attr("max_threads")
    assert function.get_attr("max_threads") == 256
    assert "#type(SimtVF)" in str(function)
    assert "simt.linear_thread_idx" in str(function)
    assert "block.getval" in str(function)
    assert "block.setval" in str(function)
    assert set(program.functions) == {"_tile_add", "_simt_tile_kernel"}
    assert "simt.launch" in str(program)


def test_simt_callee_records_type_calls_and_reachable_dependencies():
    program, matched = _simt_callee_kernel.parse_target_program(ir.SectionKind.Vector)
    callee = program.get_function("_callee_add")

    assert matched
    assert callee.func_type == ir.FunctionType.SimtCallee
    assert not callee.has_attr("max_threads")
    assert "#type(SimtCallee)" in str(callee)
    assert len(callee.return_types) == 1
    assert set(program.functions) == {
        "_callee_add",
        "_callee_store",
        "_callee_apply",
        "_callee_entry",
        "_simt_callee_kernel",
    }
    assert program.get_function("_callee_apply").func_type == ir.FunctionType.SimtCallee
    assert "_callee_add" in str(program.get_function("_callee_apply"))
    assert "_callee_store" in str(program.get_function("_callee_apply"))


def test_simt_gm_tensor_function_and_launch_reuse_scalar_tensor_access():
    program, matched = _simt_gm_kernel.parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("_gm_add")
    function_ir = str(function)

    assert matched
    assert function.func_type == ir.FunctionType.SimtVF
    assert function_ir.count("block.getval") == 1
    assert function_ir.count("block.setval") == 1
    assert set(program.functions) == {"_gm_add", "_simt_gm_kernel"}
    assert "simt.launch" in str(program)


def test_simt_function_rejects_nested_launch():
    @pl.simt.function(max_threads=32)
    def nested_launch():
        pl.simt.launch(_tile_add, threads=32, args=())

    @pl.kernel
    def kernel():
        with pl.section_vector():
            pl.simt.launch(nested_launch, threads=32, args=())

    with pytest.raises(ParserSyntaxError, match="Nested pl.simt.launch"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_context_exposes_xyz_components_and_three_dimensional_launch():
    program, matched = _simt_context_kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_context_probe"))

    assert matched
    assert function_ir.count("simt.thread_idx") >= 3
    assert function_ir.count("simt.block_dim") >= 3
    assert function_ir.count("simt.block_idx") >= 3
    assert function_ir.count("simt.grid_dim") >= 3
    assert "simt.linear_thread_idx" in function_ir
    assert "simt.warp_size" in function_ir
    assert "simt.launch" in str(program)


def test_simt_context_direct_call_uses_named_tuple_field_lowering():
    @pl.simt.function(max_threads=32)
    def direct_context(dst):
        value = (
            pl.simt.thread_idx().x
            + pl.simt.block_dim().y
            + pl.simt.block_idx().z
            + pl.simt.grid_dim().x
        )
        dst[0, 0] = value

    @pl.kernel
    def kernel():
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0, size=128)
        with pl.section_vector():
            pl.simt.launch(direct_context, threads=32, args=(dst,))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("direct_context"))

    assert function_ir.count("simt.thread_idx") >= 1
    assert function_ir.count("simt.block_dim") >= 1
    assert function_ir.count("simt.block_idx") >= 1
    assert function_ir.count("simt.grid_dim") >= 1


def test_simt_context_rejects_unknown_named_tuple_field():
    @pl.simt.function(max_threads=32)
    def invalid_context_field(dst):
        dst[0, 0] = pl.simt.thread_idx().w

    @pl.kernel
    def kernel():
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0, size=128)
        with pl.section_vector():
            pl.simt.launch(invalid_context_field, threads=32, args=(dst,))

    with pytest.raises(UnsupportedFeatureError, match="Standalone attribute access not supported"):
        kernel.parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize(
    ("shape", "dtype", "target_memory", "layout", "message"),
    [
        pytest.param(
            [1, 256],
            pl.DT_FP32,
            pl.MemorySpace.Mat,
            pl.NZ,
            "Vec-memory Tile",
            id="memory",
        ),
        pytest.param(
            [1, 256],
            pl.DT_FP32,
            pl.MemorySpace.Vec,
            pl.DN,
            "ND Vec Tile",
            id="layout",
        ),
    ],
)
def test_simt_launch_requires_compatible_tile(shape, dtype, target_memory, layout, message):
    kernel = _make_tile_launch_kernel(shape, dtype, target_memory, layout)

    with pytest.raises(ParserTypeError, match=message):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_function_rejects_block_operation_before_default_dispatch():
    @pl.simt.function(max_threads=32)
    def block_add(
        dst,
        lhs,
        rhs,
    ):
        pl.add(dst, lhs, rhs)

    @pl.kernel
    def kernel():
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0, size=128)
        lhs = pl.make_tile(tile_type, addr=128, size=128)
        rhs = pl.make_tile(tile_type, addr=256, size=128)
        with pl.section_vector():
            pl.simt.launch(block_add, threads=32, args=(dst, lhs, rhs))

    with pytest.raises(UnsupportedFeatureError, match="not supported inside a SIMT function"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_function_rejects_tile_subview():
    @pl.simt.function(max_threads=32)
    def tile_subview(src):
        _ = src[0:4, 0:32]

    @pl.kernel
    def kernel():
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0, size=2048)
        with pl.section_vector():
            pl.simt.launch(tile_subview, threads=32, args=(src,))

    with pytest.raises(UnsupportedFeatureError, match="Tile subview"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_tile_parameter_exposes_runtime_valid_shape():
    @pl.kernel
    def kernel():
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0, size=2048)
        src = pl.make_tile(tile_type, addr=2048, size=2048)
        with pl.section_vector():
            pl.simt.launch(_tile_valid_shape_access, threads=32, args=(dst, src))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_tile_valid_shape_access"))

    assert function_ir.count("block.tile_valid_shape") == 2


def test_thread_idx_rejected_outside_simt_function():
    with pytest.raises(ParserSyntaxError, match="only be used inside"):

        @pl.function
        def bad_thread_idx() -> pl.DT_UINT32:
            return pl.simt.thread_idx().x


def test_simt_launch_rejects_threads_above_bound():
    @pl.kernel
    def too_many_threads(n: pl.DT_UINT32, delta: pl.DT_FP32):
        tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0x0000, size=1024)
        dst = pl.make_tile(tile_type, addr=0x0400, size=1024)
        with pl.section_vector():
            pl.simt.launch(_tile_add, threads=288, args=(dst, src, n, delta))

    with pytest.raises(ParserTypeError, match="exceed"):
        too_many_threads.parse_target_program(ir.SectionKind.Vector)


def test_simt_launch_rejects_runtime_tuple_component():
    @pl.kernel
    def runtime_dimension(n: pl.DT_UINT32, delta: pl.DT_FP32):
        tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0x0000, size=1024)
        dst = pl.make_tile(tile_type, addr=0x0400, size=1024)
        with pl.section_vector():
            pl.simt.launch(_tile_add, threads=(8, 4, n), args=(dst, src, n, delta))

    with pytest.raises(ParserTypeError, match=r"compile-time integers.*\[1, 2048\]"):
        runtime_dimension.parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("max_threads", [0, 2049])
def test_simt_function_rejects_invalid_launch_bound(max_threads):
    @pl.simt.function(max_threads=max_threads)
    def invalid_bound(n: pl.DT_UINT32):
        return

    @pl.kernel
    def kernel(n: pl.DT_UINT32):
        with pl.section_vector():
            pl.simt.launch(invalid_bound, threads=1, args=(n,))

    from pypto_pro.language.parser.diagnostics import ParserSyntaxError

    with pytest.raises(ParserSyntaxError, match=r"\[1, 2048\]"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_launch_rejects_thread_count_above_hardware_limit():
    @pl.simt.function(max_threads=2048)
    def wide_function():
        return

    @pl.kernel
    def too_wide():
        with pl.section_vector():
            pl.simt.launch(wide_function, threads=(2048, 2), args=())

    with pytest.raises(ParserTypeError, match="must not exceed 2048"):
        too_wide.parse_target_program(ir.SectionKind.Vector)


def test_simt_function_infers_parameter_and_callee_return_types_at_call_site():
    @pl.simt.function
    def inferred_callee(value):
        return value

    @pl.simt.function(max_threads=32)
    def inferred_entry(dst, value):
        dst[0, 0] = inferred_callee(value)

    @pl.kernel
    def kernel(value: pl.DT_INT32):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0, size=128)
        with pl.section_vector():
            pl.simt.launch(inferred_entry, threads=32, args=(dst, value))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    callee = program.get_function("inferred_callee")
    entry = program.get_function("inferred_entry")

    assert isinstance(callee.params[0].type, ir.ScalarType)
    assert callee.params[0].type.dtype == pl.DT_INT32
    assert len(callee.return_types) == 1
    assert callee.return_types[0].dtype == pl.DT_INT32
    assert isinstance(entry.params[0].type, ir.TileType)
    assert entry.params[1].type.dtype == pl.DT_INT32


def test_simt_function_annotations_do_not_override_callsite_types():
    @pl.simt.function
    def annotated_callee(value: pl.DT_INT32) -> pl.DT_INT32:
        return value

    @pl.simt.function(max_threads=32)
    def annotated_entry(value: pl.DT_INT32):
        annotated_callee(value)

    @pl.kernel
    def kernel(value: pl.DT_INT64):
        with pl.section_vector():
            pl.simt.launch(annotated_entry, threads=32, args=(value,))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    callee = program.get_function("annotated_callee")
    entry = program.get_function("annotated_entry")

    assert entry.params[0].type.dtype == pl.DT_INT64
    assert callee.params[0].type.dtype == pl.DT_INT64
    assert callee.return_types[0].dtype == pl.DT_INT64


def test_simt_callee_rejects_return_incompatible_with_annotation():
    @pl.simt.function
    def bad_return(value: pl.DT_FP32) -> pl.DT_INT32:
        return value

    @pl.simt.function(max_threads=32)
    def entry(value: pl.DT_FP32):
        bad_return(value)

    @pl.kernel
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            pl.simt.launch(entry, threads=32, args=(value,))

    with pytest.raises(ParserTypeError, match="Return 'bad_return' annotated as"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_function_rejects_argument_incompatible_with_annotation():
    @pl.simt.function(max_threads=32)
    def entry(value: pl.DT_INT32):
        return

    @pl.kernel
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            pl.simt.launch(entry, threads=32, args=(value,))

    with pytest.raises(ParserTypeError, match="SIMT parameter 'value' annotated as"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_cached_simt_function_rejects_incompatible_argument_type():
    @pl.simt.function(max_threads=32)
    def entry(value):
        return

    @pl.kernel
    def kernel(integer: pl.DT_INT32, floating: pl.DT_FP32):
        with pl.section_vector():
            pl.simt.launch(entry, threads=32, args=(integer,))
            pl.simt.launch(entry, threads=32, args=(floating,))

    with pytest.raises(ParserTypeError, match="SIMT parameter 'value' annotated as"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_entry_rejects_value_return():
    @pl.simt.function(max_threads=32)
    def entry(value: pl.DT_INT32):
        return value

    @pl.kernel
    def kernel(value: pl.DT_INT32):
        with pl.section_vector():
            pl.simt.launch(entry, threads=32, args=(value,))

    with pytest.raises(ParserSyntaxError, match="only supports bare return or return None"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_scalar_simt_callee_accepts_early_return_when_all_paths_return():
    @pl.simt.function
    def absolute(value: pl.DT_INT32) -> pl.DT_INT32:
        if value < 0:
            return -value
        return value

    @pl.simt.function(max_threads=32)
    def entry(value: pl.DT_INT32):
        absolute(value)

    @pl.kernel
    def kernel(value: pl.DT_INT32):
        with pl.section_vector():
            pl.simt.launch(entry, threads=32, args=(value,))

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("absolute")
    assert function.func_type == ir.FunctionType.SimtCallee
    assert str(function).count("return") == 2


def test_simt_callee_cannot_be_launched_directly():
    @pl.kernel
    def invalid_launch(value: pl.DT_INT32):
        with pl.section_vector():
            pl.simt.launch(_callee_add, threads=32, args=(value, value))

    with pytest.raises(ParserTypeError, match="not a launchable @pl.simt.function"):
        invalid_launch.parse_target_program(ir.SectionKind.Vector)


def test_recursive_simt_callee_is_rejected_during_instantiation():
    @pl.simt.function
    def recursive(value: pl.DT_INT32) -> pl.DT_INT32:
        return recursive(value)

    @pl.simt.function(max_threads=32)
    def entry(value: pl.DT_INT32):
        recursive(value)

    @pl.kernel
    def kernel(value: pl.DT_INT32):
        with pl.section_vector():
            pl.simt.launch(entry, threads=32, args=(value,))

    with pytest.raises(ParserSyntaxError, match="Recursive helper"):
        kernel.parse_target_program(ir.SectionKind.Vector)


def test_simt_launch_auto_mutex_inserts_pipe_v_lock_unlock():
    @pl.simt.function(max_threads=256)
    def inplace_add(data, delta: pl.DT_FP32):
        tid = pl.simt.linear_thread_idx()
        data[0, tid] = data[0, tid] + delta

    @pl.kernel
    def kernel(
        x: pl.Tensor[[1, 256], pl.DT_FP32],
        out: pl.Tensor[[1, 256], pl.DT_FP32],
        delta: pl.DT_FP32,
    ):
        tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        data = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
        with pl.section_vector():
            pl.load(data.current(), x, [0, 0])
            pl.simt.launch(inplace_add, threads=256, args=(data.current(), delta))
            pl.store(out, data.current(), [0, 0])

    program, _ = kernel.parse_target_program(ir.SectionKind.Vector)
    ir_str = str(program)

    assert "system.mutex_lock_dyn" in ir_str
    assert "system.mutex_unlock_dyn" in ir_str
    assert "simt.launch" in ir_str

    kernel_ir = str(program.get_function("kernel"))
    lock_pos = kernel_ir.index("system.mutex_lock_dyn")
    launch_pos = kernel_ir.index("simt.launch")
    unlock_pos = kernel_ir.index("system.mutex_unlock_dyn", launch_pos)
    assert lock_pos < launch_pos < unlock_pos
