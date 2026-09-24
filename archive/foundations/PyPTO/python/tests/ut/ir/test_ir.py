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
import pytest

import pypto
from pypto import ir


def test_error_types():
    with pytest.raises(ValueError):
        pypto.raise_error("ValueError", "test value error")
    with pytest.raises(TypeError):
        pypto.raise_error("TypeError", "test type error")
    with pytest.raises(RuntimeError):
        pypto.raise_error("RuntimeError", "test runtime error")
    with pytest.raises(NotImplementedError):
        pypto.raise_error("NotImplementedError", "test not implemented error")
    with pytest.raises(IndexError):
        pypto.raise_error("IndexError", "test index error")
    with pytest.raises(AssertionError):
        pypto.raise_error("AssertionError", "test assertion error")
    with pytest.raises(pypto.InternalError):
        pypto.raise_error("InternalError", "test internal error")
    # Error -> Python Exception (base)
    with pytest.raises(Exception, match="base error"):
        pypto.raise_error("Error", "base error")
    # Unknown type -> falls back to Error -> Python Exception
    with pytest.raises(Exception, match="Unknown error type"):
        pypto.raise_error("UnknownType", "should fail")


def test_assert_structural_equal_raises_runtime_error_with_fe_code():
    from pypto import pypto_impl

    span = ir.Span("test", 1, 1)
    lhs = ir.ConstInt(1, ir.INT32, span)
    rhs = ir.ConstInt(2, ir.INT32, span)

    with pytest.raises(RuntimeError) as exc_info:
        pypto_impl.ir.assert_structural_equal(lhs, rhs)

    msg = str(exc_info.value)
    assert "ASSERT FAILED" in msg
    assert "INVALID_VAL" in msg
    assert "Structural equality assertion failed" in msg


def test_dtypes():
    dtypes = [
        # bits, is_signed, is_unsigned, is_float, name, c_type_string
        (ir.BOOL, 8, False, False, False, "bool", "bool"),
        (ir.INT4, 4, True, False, False, "int4", "unknown"),
        (ir.INT8, 8, True, False, False, "int8", "int8_t"),
        (ir.INT16, 16, True, False, False, "int16", "int16_t"),
        (ir.INT32, 32, True, False, False, "int32", "int32_t"),
        (ir.INT64, 64, True, False, False, "int64", "int64_t"),
        (ir.INDEX, 64, True, False, False, "index", "int64_t"),
        (ir.UINT4, 4, False, True, False, "uint4", "unknown"),
        (ir.UINT8, 8, False, True, False, "uint8", "uint8_t"),
        (ir.UINT16, 16, False, True, False, "uint16", "uint16_t"),
        (ir.UINT32, 32, False, True, False, "uint32", "uint32_t"),
        (ir.UINT64, 64, False, True, False, "uint64", "uint64_t"),
        (ir.FP4, 4, False, False, True, "fp4", "unknown"),
        (ir.FP8E4M3FN, 8, False, False, True, "fp8e4m3fn", "float8_e4m3_t"),
        (ir.FP8E5M2, 8, False, False, True, "fp8e5m2", "float8_e5m2_t"),
        (ir.FP16, 16, False, False, True, "fp16", "half"),
        (ir.FP32, 32, False, False, True, "fp32", "float"),
        (ir.FP64, 64, False, False, True, "fp64", "double"),
        (ir.BF16, 16, False, False, True, "bfloat16", "bfloat16_t"),
        (ir.HF4, 4, False, False, True, "hf4", "unknown"),
        (ir.HF8, 8, False, False, True, "hf8", "hifloat8_t"),
    ]
    for dtype, *info in dtypes:
        assert info == [
            dtype.bits(),
            dtype.is_signed(),
            dtype.is_unsigned(),
            dtype.is_float(),
            str(dtype),
            dtype.to_c_type_string(),
        ]


def test_span():
    span = ir.Span("span", 1, 2, 3, 4)
    assert span.filename == "span"
    assert span.begin_line == 1
    assert span.begin_column == 2
    assert span.end_line == 3
    assert span.end_column == 4
    assert not span.is_unknown()

    span = ir.Span("span", 1, 2)
    assert span.filename == "span"
    assert span.begin_line == 1
    assert span.begin_column == 2
    assert span.end_line == -1
    assert span.end_column == -1
    assert not span.is_unknown()

    span = ir.Span.unknown()
    assert span.is_unknown()


def test_logging(capfd):
    pypto.set_log_level(pypto.LogLevel.INFO)
    assert pypto.get_log_level() == pypto.LogLevel.INFO

    pypto.log_debug("test debug message")
    pypto.log_info("test info message")
    pypto.log_warn("test warn message")
    pypto.log_error("test error message")
    pypto.log_event("test event message")
    pypto.log_fatal("test fatal message")

    captured = capfd.readouterr()
    assert "event" in captured.err
    assert "fatal" in captured.err
    assert "error" in captured.err
    assert "warn" in captured.err
    assert "info" in captured.err
    assert "debug" not in captured.err

    # Test C++ log functions directly (pypto_impl.log_debug etc.)
    from pypto import pypto_impl

    pypto_impl.log_debug("cpp debug msg")
    pypto_impl.log_info("cpp info msg")
    pypto_impl.log_warn("cpp warn msg")
    pypto_impl.log_error("cpp error msg")
    pypto_impl.log_event("cpp event msg")
    pypto_impl.log_fatal("cpp fatal msg")
    captured = capfd.readouterr()
    # Note: C++ logger may filter DEBUG level, so we only assert INFO and above
    assert "cpp info msg" in captured.err
    assert "cpp warn msg" in captured.err
    assert "cpp error msg" in captured.err
    assert "cpp event msg" in captured.err
    assert "cpp fatal msg" in captured.err


def test_check():
    with pytest.raises(ValueError, match="test check message"):
        pypto.check(False, "test check message")
    pypto.check(True, "test check message")

    with pytest.raises(pypto.InternalError, match="test internal check message"):
        pypto.internal_check(False, "test internal check message")
    pypto.internal_check(True, "test internal check message")


def test_basic_types():
    # UnknownType
    ut = ir.UnknownType()
    assert str(ut) == "unknown"

    # ScalarType
    st = ir.ScalarType(ir.INT32)
    assert st.dtype == ir.INT32
    assert str(st) == "int32_t"

    # TensorType with int64_t shape
    tt = ir.TensorType([16, 32], ir.FP32)
    assert tt.dtype == ir.FP32
    assert len(tt.shape) == 2
    assert tt.memref is None
    assert str(tt) == "tensor<16 x 32, float, tensor_view<>>"

    # TensorType with memref
    offset = ir.ConstInt(0, ir.INT64, ir.Span("test", 1, 1))
    memref = ir.MemRef(ir.MemorySpace.DDR, offset, 1024)
    tt2 = ir.TensorType([16, 32], ir.FP16, memref)
    assert tt2.memref is not None
    assert tt2.memref.size == 1024
    assert str(tt2) == "tensor<16 x 32, half, tensor_view<>>"

    # TupleType
    tup = ir.TupleType([ir.ScalarType(ir.INT32), ir.ScalarType(ir.FP32)])
    assert len(tup.types) == 2
    assert str(tup) == "tuple<int32_t, float>"

    pt = ir.PtrType()
    assert str(pt) == "ptr<int8_t>"

    # TokenType
    tt3 = ir.TokenType()
    assert str(tt3) == "token"

    # NoneType
    nt = ir.NoneType()
    assert str(nt) == "none"

    # LogicalTensorType
    lt = ir.LogicalTensorType()
    assert str(lt) == "v0_logical_tensor"

    # Struct debug info
    from pypto import pypto_impl

    span = ir.Span("test", 1, 1)
    struct_type = ir.TupleType([ir.ScalarType(ir.INT64), ir.ScalarType(ir.INT32)])
    debug_info = pypto_impl.ir.IRDebugInfo()
    debug_info.register_tuple_fields(struct_type, ["cursor", "limit"])
    debug_info.register_tuple_name(struct_type, "BufferState")
    assert debug_info.get_tuple_fields(struct_type) == ["cursor", "limit"]
    assert debug_info.get_tuple_name(struct_type) == "BufferState"
    assert debug_info.get_tuple_fields(ir.TupleType([ir.ScalarType(ir.INT32)])) is None

    prog = ir.Program([], "test_prog", span, debug_info)
    assert prog.debug_info.get_tuple_fields(struct_type) == ["cursor", "limit"]
    assert prog.debug_info.get_tuple_name(struct_type) == "BufferState"
    assert ir.Program([], "empty_prog", span).debug_info is None


def test_basic_expr():
    span = ir.Span("test", 1, 1)
    st = ir.ScalarType(ir.INT32)

    # ConstInt
    ci = ir.ConstInt(42, ir.INT32, span)
    assert str(ci) == "42"

    # ConstFloat
    cf = ir.ConstFloat(3.14, ir.FP32, span)
    assert str(cf) == "3.14"

    # ConstBool
    cb = ir.ConstBool(True, span)
    assert str(cb) == "true"

    # Var
    var = ir.Var("x", st, span)
    assert str(var) == "%x"
    assert var.name == "x"

    # IterArg
    init_val = ir.ConstInt(0, ir.INT32, span)
    iter_arg = ir.IterArg("acc", st, init_val, span)
    assert str(iter_arg) == "acc"
    assert iter_arg.name == "acc"
    assert isinstance(iter_arg.initValue, ir.ConstInt)
    assert iter_arg.initValue.value == 0

    # Binary expressions
    bop_names = {
        ir.Add: "add", ir.Sub: "sub", ir.Mul: "mul", ir.FloorDiv: "div",
        ir.FloatDiv: "fdiv", ir.FloorMod: "mod", ir.Pow: "pow",
        ir.Eq: "eq", ir.Ne: "ne", ir.Lt: "lt", ir.Le: "le",
        ir.Gt: "gt", ir.Ge: "ge", ir.And: "land", ir.Or: "lor",
        ir.Xor: "lxor", ir.BitAnd: "and", ir.BitOr: "or",
        ir.BitXor: "xor", ir.BitShiftLeft: "shl", ir.BitShiftRight: "shr",
    }
    for bop_cls, op_name in bop_names.items():
        a = ir.ConstInt(1, ir.INT32, span)
        b = ir.ConstInt(2, ir.INT32, span)
        expr = bop_cls(a, b, ir.INT32, span)
        assert str(expr) == f"(1 {op_name} 2)"

    a = ir.ConstInt(1, ir.INT32, span)
    b = ir.ConstInt(2, ir.INT32, span)
    expr = ir.Min(a, b, ir.INT32, span)
    assert str(expr) == "(1 min 2)"

    a = ir.ConstInt(1, ir.INT32, span)
    b = ir.ConstInt(2, ir.INT32, span)
    expr = ir.Max(a, b, ir.INT32, span)
    assert str(expr) == "(1 max 2)"

    # Unary expressions
    neg = ir.Neg(a, ir.INT32, span)
    assert str(neg) == "(neg 1)"

    not_expr = ir.Not(cb, ir.BOOL, span)
    assert str(not_expr) == "(not true)"

    bit_not = ir.BitNot(a, ir.INT32, span)
    assert str(bit_not) == "(inv 1)"

    abs_expr = ir.Abs(a, ir.INT32, span)
    assert str(abs_expr) == "(abs 1)"

    cast = ir.Cast(a, ir.FP32, span)
    assert str(cast) == "(cast float 1)"

    # MakeTuple and TupleGetItemExpr
    mt = ir.MakeTuple([a, b], span)
    assert str(mt) == "tuple(1, 2)"
    tgi = ir.TupleGetItem(mt, 0, span)
    assert str(tgi) == "getitem(tuple(1, 2), 0)"
    get_item = ir.GetItemExpr(mt, ir.ConstInt(1, ir.INDEX, span), span)
    assert str(get_item) == "getitem(tuple(1, 2), 1)"

    # Call with Op
    call = ir.Call("my_op", [a, b], span)
    assert str(call) == "my_op(1, 2)"

    # MemRef
    offset = ir.ConstInt(0, ir.INT64, span)
    memref = ir.MemRef(ir.MemorySpace.Vec, offset, 2048, span)
    assert str(memref) == "memref(Vec, 0, 2048)"


def test_basic_stmt():
    from pypto import pypto_impl

    span = ir.Span("test", 1, 1)
    st = ir.ScalarType(ir.INT32)

    # Helper variables and expressions
    x = ir.Var("x", st, span)
    y = ir.Var("y", st, span)
    val42 = ir.ConstInt(42, ir.INT32, span)
    val0 = ir.ConstInt(0, ir.INT32, span)
    val1 = ir.ConstInt(1, ir.INT32, span)
    val10 = ir.ConstInt(10, ir.INT32, span)

    # AssignStmt
    assign = ir.AssignStmt(x, val42, span)
    assert str(assign) == "int32_t %x = 42;"

    # SeqStmts
    assign_x = ir.AssignStmt(x, val42, span)
    assign_y = ir.AssignStmt(y, val0, span)
    seq = ir.SeqStmts([assign_x, assign_y], span)
    assert str(seq) == "\n".join(["{", "    int32_t %x = 42;", "    int32_t %y = 0;", "}"])

    # IfStmt
    cond = ir.ConstBool(True, span)
    if_stmt = ir.IfStmt(cond, assign_x, None, [], span)
    assert str(if_stmt) == "\n".join(["if true then {", "    int32_t %x = 42;", "} else {}"])

    if_else = ir.IfStmt(cond, assign_x, assign_y, [], span)
    assert str(if_else) == "\n".join(
        ["if true then {", "    int32_t %x = 42;", "} else {", "    int32_t %y = 0;", "}"]
    )

    # ForStmt
    i = ir.Var("i", st, span)
    init = ir.ConstInt(0, ir.INT32, span)
    iter_arg = ir.IterArg("sum", st, init, span)
    ret_var = ir.Var("sum_out", st, span)
    body = ir.YieldStmt([val1], span)
    attrs: dict = {"a": True, "parallel": True}
    for_stmt = ir.ForStmt(i, val0, val10, val1, [iter_arg], body, [ret_var], span, attrs)
    assert str(for_stmt) == "\n".join(
        [
            "int32_t %sum_out = for %i inrange 0, 10, 1 iter { int32_t %sum = 0; } #a(true) #parallel(true) {",
            "    yield 1;",
            "}",
        ]
    )

    # WhileStmt
    while_stmt = ir.WhileStmt(cond, [], assign_x, [], span)
    assert str(while_stmt) == "\n".join(["while true iter { } {", "    int32_t %x = 42;", "}"])
    # SectionStmt
    for section_kind, section_name in [
        (pypto_impl.ir.SectionKind.Vector, "Vector"),
        (pypto_impl.ir.SectionKind.Cube, "Cube"),
        (pypto_impl.ir.SectionKind.VF, "VF"),
    ]:
        section = pypto_impl.ir.SectionStmt(section_kind, assign_x, span)
        assert str(section) == "\n".join([f"section {section_name} {{", "    int32_t %x = 42;", "}"])

    # YieldStmt and ReturnStmt
    yield_stmt = ir.YieldStmt([val42], span)
    assert str(yield_stmt) == "yield 42;"
    empty_yield = ir.YieldStmt(span)
    assert str(empty_yield) == "yield;"

    return_stmt = ir.ReturnStmt([val42], span)
    assert str(return_stmt) == "return 42;"
    empty_return = ir.ReturnStmt(span)
    assert str(empty_return) == "return;"

    # BreakStmt and ContinueStmt
    break_stmt = ir.BreakStmt(span)
    assert str(break_stmt) == "break;"
    break_with_value = ir.BreakStmt([val1], span)
    assert str(break_with_value) == "break 1;"
    continue_stmt = ir.ContinueStmt(span)
    assert str(continue_stmt) == "continue;"
    continue_with_value = ir.ContinueStmt([val1], span)
    assert str(continue_with_value) == "continue 1;"

    # EvalStmt
    call = ir.Call("some_op", [val42], span)
    eval_stmt = ir.EvalStmt(call, span)
    assert str(eval_stmt) == "some_op(42);"

    # ScalarOpStmt and TensorOpStmt
    token = ir.Var("tok", ir.TokenType(), span)
    scalar_op = pypto_impl.ir.ScalarOpStmt(x, token, "add", [val1], span)
    assert str(scalar_op) == "int32_t %x, token %tok = add(1);"
    tensor_op = ir.TensorOpStmt([x], [token], "matmul", [val1], [], {}, span)
    assert str(tensor_op) == "int32_t %x, token %tok = matmul(1) token();"

    # Function
    func = ir.Function("test_func", [x], [st], assign_x, span)
    assert str(func) == "\n".join(
        [
            "function test_func incast(int32_t %x) outcast(int32_t) #type(Opaque) #entry(false) {",
            "    int32_t %x = 42;",
            "}",
        ]
    )
    helper_func = ir.Function("helper", [x], [st], ir.YieldStmt([x], span), span, ir.FunctionType.Helper)
    assert str(helper_func) == "\n".join(
        [
            "function helper incast(int32_t %x) outcast(int32_t) #type(Helper) #entry(false) {",
            "    yield %x;",
            "}",
        ]
    )

    # Program
    func2 = ir.Function("test_func2", [x], [st], assign_x, span)
    prog = ir.Program([func, func2], "test_prog", span)
    assert str(prog) == "\n".join(
        [
            "program #entry() {",
            "    function test_func incast(int32_t %x) outcast(int32_t) #type(Opaque) #entry(false) {",
            "        int32_t %x = 42;",
            "    }",
            "    function test_func2 incast(int32_t %x) outcast(int32_t) #type(Opaque) #entry(false) {",
            "        int32_t %x = 42;",
            "    }",
            "}",
        ]
    )
    assert prog["test_func"] is not None


def test_call_kwargs_conversion():
    """
    Test ConvertKwargsDict via ir.Call kwargs parameter.
    Covers: DataType, bool, int, string, float, error case.
    Note: MemorySpace/PipeType/CoreType/list types are converted but not readable via kwargs getter.
    """
    span = ir.Span("test", 1, 1)
    a = ir.ConstInt(1, ir.INT32, span)

    # Normal cases: types that work end-to-end (convert + read back)
    kwargs = {
        "dtype": ir.FP16,  # DataType
        "flag": True,  # bool
        "axis": 0,  # int
        "mode": "fast",  # string
        "scale": 0.5,  # float
    }
    call = ir.Call("test_op", [a], kwargs, span)
    assert call.kwargs["dtype"] == ir.FP16
    assert call.kwargs["flag"] is True
    assert call.kwargs["axis"] == 0
    assert call.kwargs["mode"] == "fast"
    assert call.kwargs["scale"] == 0.5

    # Error case: unsupported type
    with pytest.raises(TypeError):
        ir.Call("test_op", [a], {"bad": object()}, span)
