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
"""CCE codegen smoke tests for frontend while/for break and continue.

The tests verify the generated C++ uses native loop and jump constructs for the
block DSL patterns that codegen owns. They intentionally avoid asserting older
temporary lowering details such as synthetic while guards or SSA slot names.
"""

import logging
import re

import pypto_pro.language as pl


def _compile_to_cce(kernel_def) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel_def, "a5", "")
    return _assemble_cv_source(cube, vector).content


def _assert_lowered_while_cpp(cpp: str, condition_fragment: str) -> tuple[int, int]:
    """Assert condition evaluation is emitted in a native C++ while loop."""
    loop_index = cpp.index("while (")
    condition_index = cpp.index(condition_fragment, loop_index)
    assert loop_index <= condition_index
    return loop_index, condition_index


def _assert_materialized_jump_guard(cpp: str, condition_fragment: str, jump: str) -> tuple[int, int, int]:
    """Assert a source condition is emitted in an if containing the requested native jump."""
    condition_index = cpp.index(condition_fragment)
    jump_index = cpp.index(jump, condition_index)
    inline_guard = cpp.rfind("if (", 0, condition_index)
    is_inline = inline_guard >= 0 and "\n" not in cpp[inline_guard:condition_index]
    guard_index = inline_guard if is_inline else cpp.index("if (", condition_index)
    if is_inline:
        assert guard_index < condition_index < jump_index
    else:
        assert condition_index < guard_index < jump_index
    return condition_index, guard_index, jump_index


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@pl.kernel
def _while_basic_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000, size=16384)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        pl.add(tile_c, tile_a, tile_b)
        i = i + 1


@pl.kernel
def _while_continue_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000, size=16384)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        if i == 2:
            i = i + 1
            continue
        pl.add(tile_c, tile_a, tile_b)
        i = i + 1


@pl.kernel
def _while_break_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000, size=16384)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        if i == 2:
            break
        pl.add(tile_c, tile_a, tile_b)
        i = i + 1


@pl.kernel
def _while_break_carry_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000, size=16384)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    acc = 0
    while i < 5:
        i = i + 1
        if i == 3:
            break
        acc = acc + i


@pl.kernel
def _while_accumulate_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type_a = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = pl.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = pl.make_tile(tile_type_c, addr=0x8000, size=16384)
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    i = 0
    acc = 0
    while i < 5:
        pl.add(tile_c, tile_a, tile_b)
        acc = acc + i
        i = i + 1


@pl.kernel(auto_mutex=True)
def _while_getval_condition_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_group = pl.make_tile_group(type=tile_type, addrs=[0x0000], mutex_ids=[20])
    with pl.section_vector():
        tile = tile_group.next()
        pl.load(tile, a, [0, 0])
        i = 0
        while tile[0, i] > 0:
            i = i + 1
            if i >= 2:
                break


@pl.kernel
def _for_continue_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    acc = 0
    for i in pl.range(5):
        if i == 2:
            continue
        acc = acc + i


@pl.kernel
def _for_break_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
):
    acc = 0
    for i in pl.range(5):
        if i == 3:
            break
        acc = acc + i


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cce_while_basic():
    cpp = _compile_to_cce(_while_basic_kernel)
    logging.info("\n=== test_cce_while_basic ===\n%s", cpp)

    _assert_lowered_while_cpp(cpp, "< 5")
    assert "_can_continue" not in cpp, "Basic while needs no break flag"
    assert "TADD(" in cpp, "Loop body op should be emitted"
    assert "+ 1" in cpp, "Expected the i = i + 1 update"


def test_cce_while_continue():
    cpp = _compile_to_cce(_while_continue_kernel)
    logging.info("\n=== test_cce_while_continue ===\n%s", cpp)

    _assert_lowered_while_cpp(cpp, "< 5")
    assert "_can_continue" not in cpp, "Native continue needs no break flag"
    assert "continue;" in cpp, "Expected a native C++ continue"
    _, guard_idx, cont_idx = _assert_materialized_jump_guard(cpp, "== 2", "continue;")
    assert guard_idx < cont_idx, "continue must be inside its guard"
    assert re.search(
        r"\bi(?:_iter)?_\d+\s*=\s*i(?:_iter)?_\d+;",
        cpp[guard_idx:cont_idx],
    ), "induction var update must precede continue"
    assert "TADD(" in cpp, "Body op should remain on the fall-through path"


def test_cce_while_break():
    cpp = _compile_to_cce(_while_break_kernel)
    logging.info("\n=== test_cce_while_break ===\n%s", cpp)

    assert "_can_continue" not in cpp, "Native break needs no _can_continue flag"
    _assert_lowered_while_cpp(cpp, "< 5")
    _assert_materialized_jump_guard(cpp, "== 2", "break;")
    assert cpp.count("break;") >= 1, "Expected source C++ break"
    assert "TADD(" in cpp, "Body op present on the no-break path"


def test_cce_while_break_carry():
    cpp = _compile_to_cce(_while_break_carry_kernel)
    logging.info("\n=== test_cce_while_break_carry ===\n%s", cpp)

    assert "break;" in cpp, "Expected a native C++ break"
    assert "_can_continue" not in cpp, "Native break needs no flag"
    _, guard_idx, break_idx = _assert_materialized_jump_guard(cpp, "== 3", "break;")
    assert guard_idx < break_idx, "break must be inside its guard"


def test_cce_while_accumulate():
    cpp = _compile_to_cce(_while_accumulate_kernel)
    logging.info("\n=== test_cce_while_accumulate ===\n%s", cpp)

    assert "while (" in cpp, "Expected a C++ while loop"
    assert "_can_continue" not in cpp, "No break/continue, so no flag"
    assert "TADD(" in cpp, "Body op should be emitted"
    assert "+ 1" in cpp, "i update present"


def test_cce_while_getval_condition():
    cpp = _compile_to_cce(_while_getval_condition_kernel)
    logging.info("\n=== test_cce_while_getval_condition ===\n%s", cpp)

    lock_index = cpp.index("get_buf(PIPE_S")
    getval_index = cpp.index(".GetValue(", lock_index)
    unlock_index = cpp.index("rls_buf(PIPE_S", getval_index)
    assert lock_index < getval_index < unlock_index


def test_cce_for_continue():
    cpp = _compile_to_cce(_for_continue_kernel)
    logging.info("\n=== test_cce_for_continue ===\n%s", cpp)

    assert "for (" in cpp, "Expected a C++ for loop"
    assert "_can_continue" not in cpp, "Native continue needs no flag"
    assert "continue;" in cpp, "Expected a native C++ continue"
    _assert_materialized_jump_guard(cpp, "== 2", "continue;")
    assert "acc_0" in cpp, "Loop-carried acc threaded through the for loop"


def test_cce_for_break():
    cpp = _compile_to_cce(_for_break_kernel)
    logging.info("\n=== test_cce_for_break ===\n%s", cpp)

    assert "for (" in cpp, "Expected a C++ for loop"
    assert "_can_continue" not in cpp, "Native break needs no flag"
    assert "break;" in cpp, "Expected a native C++ break"
    _assert_materialized_jump_guard(cpp, "== 3", "break;")
