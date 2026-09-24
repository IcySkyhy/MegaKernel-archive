# -*- coding: utf-8 -*-
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for ir.Pass.infer_token_pass."""

from pathlib import Path
import re

import pypto
from pypto import ir
from pypto.pil.compile_pipeline import compile_new_ir

from ..test_common import check_snapshot

_GOLDEN_DIR = Path(__file__).parent


def _run_infer_token_pass(func, *args):
    pipeline = [("infer_token_pass", ir.Pass.infer_token_pass())]
    return compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)


def _tensor(name="x"):
    return pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name=name)


def _condition():
    return pypto.Tensor(shape=(-1,), dtype=pypto.DT_INT32, name="condition")


class _NormalizedSnapshot:
    def __init__(self, func):
        self._func = func
        self.name = func.name

    def __str__(self):
        return _normalize_token_names(str(self._func))


def _normalize_token_names(text):
    token_names = {}
    pattern = re.compile(r"(_[A-Za-z0-9]+_token_|__token_|_if_token_|_for_token_)\d+")

    def replace(match):
        name = match.group(0)
        if name not in token_names:
            token_names[name] = match.group(1) + str(len(token_names) + 1)
        return token_names[name]

    return re.sub(r"loop_idx_\d+", "loop_idx_N", pattern.sub(replace, text))


def _check_snapshot(func, golden):
    check_snapshot(_NormalizedSnapshot(func), _normalize_token_names(golden))


IR_0 = (_GOLDEN_DIR / "test_infer_pass_test_repeated_input.pypto").read_text()
def test_repeated_input():
    def foo(x):
        pypto.add(x, x)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_0)


IR_1 = (_GOLDEN_DIR / "test_infer_pass_test_multiple_outputs.pypto").read_text()
def test_multiple_outputs():
    def foo(x):
        pypto.topk(x, 2)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_1)


IR_2 = (_GOLDEN_DIR / "test_infer_pass_test_multiple_reads.pypto").read_text()
def test_multiple_reads():
    def foo(x):
        lhs = x + 1.0
        rhs = x - 1.0
        pypto.add(lhs, rhs)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_2)


IR_3 = (_GOLDEN_DIR / "test_infer_pass_test_read_after_write.pypto").read_text()
def test_read_after_write():
    def foo(a, b):
        value = pypto.exp(b)
        pypto.assemble(value, [0, 0], a)
        pypto.exp(a)
        pypto.exp(a)

    func = _run_infer_token_pass(foo, _tensor("a"), _tensor("b"))
    check_snapshot(func, IR_3)


IR_4 = (_GOLDEN_DIR / "test_infer_pass_test_write_after_read.pypto").read_text()
def test_write_after_read():
    def foo(x):
        pypto.set_vec_tile_shapes(16, 16)
        aux = pypto.tensor([32, 16], pypto.DT_FP32, name="aux")
        read = aux + 1.0
        src = pypto.full([16, 16], 2.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        pypto.add(read, x)

    func = _run_infer_token_pass(foo, _tensor())
    check_snapshot(func, IR_4)


IR_5 = (_GOLDEN_DIR / "test_infer_pass_test_read_read_write.pypto").read_text()
def test_read_read_write():
    def foo(a):
        value = pypto.exp(a)
        pypto.exp(a)
        pypto.assemble(value, [0, 0], a)

    func = _run_infer_token_pass(foo, _tensor("a"))
    check_snapshot(func, IR_5)


IR_6 = (_GOLDEN_DIR / "test_infer_pass_test_write_write_read.pypto").read_text()
def test_write_write_read():
    def foo(a, b, c):
        first = pypto.exp(b)
        pypto.assemble(first, [0, 0], a)
        second = pypto.exp(c)
        pypto.assemble(second, [0, 0], a)
        pypto.exp(a)

    func = _run_infer_token_pass(foo, _tensor("a"), _tensor("b"), _tensor("c"))
    check_snapshot(func, IR_6)


IR_7 = (_GOLDEN_DIR / "test_infer_pass_test_if_input.pypto").read_text()
def test_if_input():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        if condition.shape[0] > 0:
            result = pypto.add(aux, x)
        else:
            result = pypto.sub(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_7)


IR_8 = (_GOLDEN_DIR / "test_infer_pass_test_if_output.pypto").read_text()
def test_if_output():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        if condition.shape[0] > 0:
            src_then = pypto.full([16, 16], 1.0, pypto.DT_FP32)
            pypto.assemble(src_then, [0, 0], aux)
        else:
            src_else = pypto.full([16, 16], 2.0, pypto.DT_FP32)
            pypto.assemble(src_else, [0, 0], aux)
        result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_8)


IR_9 = (_GOLDEN_DIR / "test_infer_pass_test_if_passthrough.pypto").read_text()
def test_if_passthrough():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        before = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(before, [0, 0], aux)
        if condition.shape[0] > 0:
            inside = pypto.full([16, 16], 2.0, pypto.DT_FP32)
            pypto.assemble(inside, [0, 0], aux)
        result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_9)


IR_10 = (_GOLDEN_DIR / "test_infer_pass_test_if_multiple_reads.pypto").read_text()
def test_if_multiple_reads():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        if condition.shape[0] > 0:
            read0 = aux + 1.0
            read1 = aux - 1.0
            result = pypto.add(read0, read1)
        else:
            result = x + 1.0
        src = pypto.full([16, 16], 2.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_10)


IR_11 = (_GOLDEN_DIR / "test_infer_pass_test_for_input.pypto").read_text()
def test_for_input():
    def foo(x, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        for _ in pypto.loop(2):
            result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"))
    _check_snapshot(func, IR_11)


IR_12 = (_GOLDEN_DIR / "test_infer_pass_test_for_output.pypto").read_text()
def test_for_output():
    def foo(x, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
            pypto.assemble(src, [i * 8, 0], aux)
        result = pypto.add(x, aux)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"))
    _check_snapshot(func, IR_12)


IR_13 = (_GOLDEN_DIR / "test_infer_pass_test_if_none.pypto").read_text()
def test_if_none():
    def foo(x, aux, out, condition):
        pypto.set_vec_tile_shapes(16, 16)
        if condition.shape[0] > 0:
            inside = pypto.full([16, 16], 1.0, pypto.DT_FP32)
            pypto.assemble(inside, [0, 0], aux)
        result = pypto.add(aux, x)
        pypto.assemble(result, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"), _condition())
    _check_snapshot(func, IR_13)


IR_14 = (_GOLDEN_DIR / "test_infer_pass_test_for_read_output.pypto").read_text()
def test_for_read_output():
    def foo(x, aux, out):
        pypto.set_vec_tile_shapes(16, 16)
        for _ in pypto.loop(2):
            read = pypto.add(aux, x)
        src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(read, [0, 0], out)

    func = _run_infer_token_pass(foo, _tensor(), _tensor("aux"), _tensor("out"))
    _check_snapshot(func, IR_14)
