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
"""Unit tests for @pl.function, @pl.inline, and @pl.program decorators."""

import linecache
import sys
import textwrap

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserTypeError
from pypto_pro.language.parser.diagnostics._exceptions import (
    ParserSyntaxError,
    UndefinedVariableError,
)
import pytest


def test_simple_function():
    """Test parsing simple function with no control flow."""

    @pl.function
    def add_tensors(
        x: pl.Tensor[[64, 128], pl.DT_FP16],
        y: pl.Tensor[[64, 128], pl.DT_FP16],
    ) -> pl.Tensor[[64, 128], pl.DT_FP16]:
        result: pl.Tensor[[64, 128], pl.DT_FP16] = pl.tensor.add(x, y)
        return result

    assert isinstance(add_tensors, ir.Function)
    assert add_tensors.name == "add_tensors"
    assert len(add_tensors.params) == 2
    assert len(add_tensors.return_types) == 1


def test_function_with_multiple_statements():
    """Test function with multiple statements."""

    @pl.function
    def multi_op(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        a: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.mul(x, 2.0)
        b: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(a, 1.0)
        c: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.sub(b, 0.5)
        return c

    assert isinstance(multi_op, ir.Function)
    assert multi_op.name == "multi_op"


def test_function_with_multiple_params():
    """Test function with multiple parameters."""

    @pl.function
    def three_param(
        x: pl.Tensor[[64], pl.DT_FP32],
        y: pl.Tensor[[64], pl.DT_FP32],
        z: pl.Tensor[[64], pl.DT_FP32],
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        temp: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, y)
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(temp, z)
        return result

    assert len(three_param.params) == 3


def test_function_with_tensor_create():
    """Test function that creates tensors."""

    @pl.function
    def create_tensor(n: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[64, 128], pl.DT_FP32]:
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.tensor.create_tensor([64, 128], dtype=pl.DT_FP32)
        return result

    assert isinstance(create_tensor, ir.Function)


def test_function_with_binary_ops():
    """Test function with binary operations."""

    @pl.function
    def binary_ops(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        # Using operator overloading
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(
            pl.tensor.mul(x, 2.0),
            pl.tensor.create_tensor([64], dtype=pl.DT_FP32),
        )
        return result

    assert isinstance(binary_ops, ir.Function)


def test_function_with_list_arguments():
    """Test function that uses list arguments."""

    @pl.function
    def with_lists(x: pl.Tensor[[64, 128], pl.DT_FP32]) -> pl.Tensor[[32, 64], pl.DT_FP32]:
        # view takes list arguments
        result: pl.Tensor[[32, 64], pl.DT_FP32] = pl.tensor.view(x, [32, 64], [0, 0])
        return result

    assert isinstance(with_lists, ir.Function)


def test_function_with_eval_stmt():
    """Test parsing evaluation statements into EvalStmt."""

    @pl.function
    def with_eval_stmt(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        # Standalone evaluation statements should become EvalStmt
        pl.tensor.create_tensor([32], dtype=pl.DT_FP32)
        pl.tensor.create_tensor([64], dtype=pl.DT_FP32)

        # Regular assignment
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return result

    body = with_eval_stmt.body
    assert isinstance(body, ir.SeqStmts)
    assert sum(isinstance(stmt, ir.EvalStmt) for stmt in body.stmts) == 2
    assert isinstance(body.stmts[-1], ir.ReturnStmt)


def test_function_with_different_dtypes():
    """Test function with various data types."""

    @pl.function
    def dtypes(
        fp16: pl.Tensor[[64], pl.DT_FP16],
        fp32: pl.Tensor[[64], pl.DT_FP32],
        int32: pl.Tensor[[64], pl.DT_INT32],
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(pl.tensor.cast(fp16, target_type=pl.DT_FP32), fp32)
        return result

    assert len(dtypes.params) == 3


def test_invalid_function_no_annotations():
    """Test that function without annotations raises error."""

    with pytest.raises(ParserTypeError, match="missing type annotation"):

        @pl.function
        def no_annotations(x):
            return x


def test_function_preserves_name():
    """Test that function name is preserved."""

    @pl.function
    def my_custom_function_name(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        return x

    assert my_custom_function_name.name == "my_custom_function_name"


def test_function_with_negative_numbers():
    """Test function with negative number literals."""

    @pl.function
    def with_negatives(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, -1.5)
        return result

    assert isinstance(with_negatives, ir.Function)


def test_function_with_scalar_param():
    """Test function with scalar parameter - subscript notation."""

    @pl.function
    def add_scalar(
        x: pl.Tensor[[64], pl.DT_FP32],
        scalar: pl.DT_FP32,
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, scalar)
        return result

    assert isinstance(add_scalar, ir.Function)
    assert add_scalar.name == "add_scalar"
    assert len(add_scalar.params) == 2

    # Check that second parameter is ScalarType
    scalar_param = add_scalar.params[1]
    assert isinstance(scalar_param.type, ir.ScalarType)
    assert scalar_param.type.dtype == pl.DT_FP32


def test_function_with_multiple_scalar_params():
    """Test function with multiple scalar parameters."""

    @pl.function
    def scale_and_offset(
        x: pl.Tensor[[64], pl.DT_FP32],
        scale: pl.DT_FP32,
        offset: pl.DT_FP32,
    ) -> pl.Tensor[[64], pl.DT_FP32]:
        scaled: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.mul(x, scale)
        result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(scaled, offset)
        return result

    assert len(scale_and_offset.params) == 3
    assert isinstance(scale_and_offset.params[1].type, ir.ScalarType)
    assert isinstance(scale_and_offset.params[2].type, ir.ScalarType)


def test_function_with_different_scalar_types():
    """Test function with scalars of different types."""

    @pl.function
    def mixed_scalars(
        fp_scalar: pl.DT_FP32,
        int_scalar: pl.DT_INT32,
    ) -> pl.DT_FP32:
        return fp_scalar

    assert isinstance(mixed_scalars.params[0].type, ir.ScalarType)
    assert mixed_scalars.params[0].type.dtype == pl.DT_FP32
    assert isinstance(mixed_scalars.params[1].type, ir.ScalarType)
    assert mixed_scalars.params[1].type.dtype == pl.DT_INT32


def test_function_returning_scalar():
    """Test function that returns a scalar."""

    @pl.function
    def return_scalar(x: pl.DT_INT64) -> pl.DT_INT64:
        return x

    assert isinstance(return_scalar, ir.Function)
    assert len(return_scalar.return_types) == 1
    assert isinstance(return_scalar.return_types[0], ir.ScalarType)


def test_scalar_annotation_uses_dt_dtype():
    """DT_* constants are the public scalar annotation syntax."""

    @pl.function
    def scalar_identity(x: pl.DT_FP32) -> pl.DT_FP32:
        return x

    assert isinstance(scalar_identity.params[0].type, ir.ScalarType)
    assert scalar_identity.params[0].type.dtype == pl.DT_FP32
    assert pl.DT_FP32 == pl.DT_FP32


def test_manual_ops_with_scalar():
    """Test block operations with scalar parameter."""

    @pl.function(type=pl.FunctionType.InCore)
    def block_add_scalar(
        input_tile: pl.Tensor[[64, 64], pl.DT_FP32],
        scalar: pl.DT_FP32,
        output: pl.Tensor[[64, 64], pl.DT_FP32],
    ) -> pl.Tensor[[64, 64], pl.DT_FP32]:
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32)
        tile = pl.make_tile(tile_type, addr=0x0000, size=16384)
        result = pl.make_tile(tile_type, addr=0x4000, size=16384)
        pl.load(tile, input_tile, [0, 0])
        pl.add(result, tile, scalar)
        output_new: pl.Tensor[[64, 64], pl.DT_FP32] = pl.store(output, result, [0, 0])
        return output_new

    assert isinstance(block_add_scalar, ir.Function)
    assert block_add_scalar.func_type == pl.FunctionType.InCore
    assert isinstance(block_add_scalar.params[1].type, ir.ScalarType)



def test_tuple_return_two_tensors():
    """Test function with tuple[Tensor, Tensor] return type."""

    @pl.function
    def two_outputs(
        x: pl.Tensor[[64], pl.DT_FP32],
    ) -> tuple[pl.Tensor[[64], pl.DT_FP32], pl.Tensor[[64], pl.DT_FP32]]:
        a: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        b: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.mul(x, 2.0)
        return a, b

    assert isinstance(two_outputs, ir.Function)
    # A tuple return is a single TupleType (not flattened into multiple return types).
    assert len(two_outputs.return_types) == 1
    ret = two_outputs.return_types[0]
    assert isinstance(ret, ir.TupleType)
    assert len(ret.types) == 2
    assert isinstance(ret.types[0], ir.TensorType)
    assert isinstance(ret.types[1], ir.TensorType)


def test_tuple_return_mixed_types():
    """Test function with tuple[Tensor, Scalar] return type."""

    @pl.function
    def mixed_return(
        x: pl.Tensor[[64], pl.DT_FP32],
        idx: pl.DT_INT64,
    ) -> tuple[pl.Tensor[[64], pl.DT_FP32], pl.DT_INT64]:
        a: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
        return a, idx

    assert isinstance(mixed_return, ir.Function)
    # A tuple return is a single TupleType (not flattened into multiple return types).
    assert len(mixed_return.return_types) == 1
    ret = mixed_return.return_types[0]
    assert isinstance(ret, ir.TupleType)
    assert len(ret.types) == 2
    assert isinstance(ret.types[0], ir.TensorType)
    assert isinstance(ret.types[1], ir.ScalarType)


class TestProgramDecorator:
    """Tests for @pl.program decorator."""

    @staticmethod
    def test_single_function_program():
        """Test @pl.program with a single function."""

        @pl.program
        class SimpleProgram:
            @pl.function
            def add_one(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                return result

        assert isinstance(SimpleProgram, ir.Program)
        assert SimpleProgram.name == "SimpleProgram"
        assert len(SimpleProgram.functions) == 1

        # Verify the function is accessible
        add_func = SimpleProgram.get_function("add_one")
        assert add_func is not None
        assert add_func.name == "add_one"
        # self parameter should be stripped
        assert len(add_func.params) == 1
        assert add_func.params[0].name == "x"

    @staticmethod
    def test_multiple_functions_program():
        """Test @pl.program with multiple functions."""

        @pl.program
        class MathOps:
            @pl.function
            def square(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.mul(x, x)
                return result

            @pl.function
            def double(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                two: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.create_tensor([1], dtype=pl.DT_INT32)
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.mul(x, two)
                return result

        assert isinstance(MathOps, ir.Program)
        assert MathOps.name == "MathOps"
        assert len(MathOps.functions) == 2

        # Verify both functions exist
        square_func = MathOps.get_function("square")
        double_func = MathOps.get_function("double")
        assert square_func is not None
        assert double_func is not None

    @staticmethod
    def test_recursive_call():
        """Test function calling itself recursively via self.method_name()."""

        @pl.program
        class RecursiveTest:
            @pl.function
            def factorial(self, n: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                _zero: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.create_tensor([1], dtype=pl.DT_INT32)
                one: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.create_tensor([1], dtype=pl.DT_INT32)
                # Note: This is just for testing IR structure, not a real factorial implementation
                # In real DSL, we'd need if statements for base case
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.add(n, one)
                return result

        assert isinstance(RecursiveTest, ir.Program)

    @staticmethod
    def test_self_parameter_stripped():
        """Test that self parameter is properly stripped from IR."""

        @pl.program
        class SelfTest:
            @pl.function
            def test_func(
                self, x: pl.Tensor[[1], pl.DT_INT32], y: pl.Tensor[[1], pl.DT_INT32]
            ) -> pl.Tensor[[1], pl.DT_INT32]:
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.add(x, y)
                return result

        func = SelfTest.get_function("test_func")
        assert func is not None
        # Should only have x and y parameters (self stripped)
        assert len(func.params) == 2
        assert func.params[0].name == "x"
        assert func.params[1].name == "y"

    @staticmethod
    def test_program_name_from_class():
        """Test that program name is extracted from class name."""

        @pl.program
        class MyCustomProgram:
            @pl.function
            def dummy(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                return x

        assert MyCustomProgram.name == "MyCustomProgram"

    @staticmethod
    def test_empty_class_error():
        """Test that empty class raises error."""
        with pytest.raises(ParserSyntaxError):  # Should raise ParserSyntaxError

            @pl.program
            class EmptyProgram:
                pass

    def test_cross_function_calls(self):
        """Test cross-function calls using self.method() syntax."""

        @pl.program
        class CallTest:
            @pl.function
            def square(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.mul(x, x)
                return result

            @pl.function
            def sum_of_squares(
                self, a: pl.Tensor[[1], pl.DT_INT32], b: pl.Tensor[[1], pl.DT_INT32]
            ) -> pl.Tensor[[1], pl.DT_INT32]:
                # Call square method using self
                a_squared: pl.Tensor[[1], pl.DT_INT32] = self.square(a)
                b_squared: pl.Tensor[[1], pl.DT_INT32] = self.square(b)
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.add(a_squared, b_squared)
                return result

        assert isinstance(CallTest, ir.Program)
        assert len(CallTest.functions) == 2

        # Verify sum_of_squares function exists and has proper parameters
        sum_func = CallTest.get_function("sum_of_squares")
        assert sum_func is not None
        # Should have 2 params (a, b) - self is stripped
        assert len(sum_func.params) == 2

    def test_forward_reference(self):
        """Test calling a function defined later in the class."""

        @pl.program
        class ForwardRef:
            @pl.function
            def caller(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                # Call helper which is defined below
                result: pl.Tensor[[1], pl.DT_INT32] = self.helper(x)
                return result

            @pl.function
            def helper(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.mul(x, 2)
                return result

        assert isinstance(ForwardRef, ir.Program)
        assert len(ForwardRef.functions) == 2

    def test_transitive_calls(self):
        """Test transitive calls where A calls B calls C."""

        @pl.program
        class TransitiveCalls:
            @pl.function
            def a(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                result: pl.Tensor[[1], pl.DT_INT32] = self.b(x)
                return result

            @pl.function
            def b(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                result: pl.Tensor[[1], pl.DT_INT32] = self.c(x)
                return result

            @pl.function
            def c(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                result: pl.Tensor[[1], pl.DT_INT32] = pl.tensor.mul(x, 3)
                return result

        assert isinstance(TransitiveCalls, ir.Program)
        assert len(TransitiveCalls.functions) == 3

    def test_undefined_method_call_error(self):
        """Test that calling undefined method raises error."""
        with pytest.raises(UndefinedVariableError):  # Should raise UndefinedVariableError

            @pl.program
            class UndefinedCall:
                @pl.function
                def caller(self, x: pl.Tensor[[1], pl.DT_INT32]) -> pl.Tensor[[1], pl.DT_INT32]:
                    # Try to call a method that doesn't exist
                    result: pl.Tensor[[1], pl.DT_INT32] = self.nonexistent(x)  # type: ignore
                    return result

    def test_tuple_unpacking_from_cross_function_call(self):
        """Test tuple unpacking from self.func() returning multiple values."""

        @pl.program
        class TupleUnpack:
            @pl.function
            def split(
                self, x: pl.Tensor[[64], pl.DT_FP32]
            ) -> tuple[pl.Tensor[[64], pl.DT_FP32], pl.Tensor[[64], pl.DT_FP32]]:
                a: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                b: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.mul(x, 2.0)
                return a, b

            @pl.function
            def caller(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                a, b = self.split(x)
                result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(a, b)
                return result

        assert isinstance(TupleUnpack, ir.Program)
        assert len(TupleUnpack.functions) == 2

        caller_func = TupleUnpack.get_function("caller")
        assert caller_func is not None


def test_function_with_linecache_source():
    """Test that @pl.function works via linecache when inspect fails (e.g., exec)."""
    code = textwrap.dedent("""\
        import pypto_pro.language as pl

        @pl.function
        def add_one(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
            return result
    """)
    filename = "<test_linecache_function>"
    code_lines = code.splitlines(keepends=True)
    # Pre-populate linecache so the fallback strategy can find the source
    linecache.cache[filename] = (len(code), None, code_lines, filename)
    try:
        compiled = compile(code, filename, "exec")
        namespace: dict = {}
        exec(compiled, namespace)  # noqa: S102
        result = namespace["add_one"]
        assert isinstance(result, ir.Function)
        assert result.name == "add_one"
        assert len(result.params) == 1
    finally:
        linecache.cache.pop(filename, None)


def test_function_with_orig_argv_source(monkeypatch):
    """Test that @pl.function works via sys.orig_argv for python -c scenarios."""
    code = textwrap.dedent("""\
        import pypto_pro.language as pl

        @pl.function
        def add_one(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
            return result
    """)
    # Simulate python -c by using <string> filename and setting sys.orig_argv
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-c", code], raising=False)
    filename = "<string>"
    compiled = compile(code, filename, "exec")
    namespace: dict = {}
    exec(compiled, namespace)  # noqa: S102
    result = namespace["add_one"]
    assert isinstance(result, ir.Function)
    assert result.name == "add_one"
    assert len(result.params) == 1


def test_function_without_source_gives_clear_error():
    """Test that @pl.function gives a clear ParserSyntaxError when no source is available."""
    code = textwrap.dedent("""\
        import pypto_pro.language as pl
        from pypto_pro.language.parser.diagnostics._exceptions import ParserSyntaxError

        try:
            @pl.function
            def add_one(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                return result
            assert False, "Should have raised ParserSyntaxError"
        except ParserSyntaxError as e:
            assert "Cannot retrieve source code" in str(e)
            assert "pl.parse()" in e.hint
    """)
    # Use a filename that won't be in linecache or on disk
    filename = "<no_source_available>"
    compiled = compile(code, filename, "exec")
    namespace: dict = {}
    exec(compiled, namespace)  # noqa: S102


def test_program_with_linecache_source():
    """Test that @pl.program works via linecache when inspect fails (e.g., exec)."""
    code = textwrap.dedent("""\
        import pypto_pro.language as pl

        @pl.program
        class MyProgram:
            @pl.function
            def add_one(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                return result
    """)
    filename = "<test_linecache_program>"
    code_lines = code.splitlines(keepends=True)
    # Pre-populate linecache so the fallback strategy can find the source
    linecache.cache[filename] = (len(code), None, code_lines, filename)
    try:
        compiled = compile(code, filename, "exec")
        namespace: dict = {}
        exec(compiled, namespace)  # noqa: S102
        result = namespace["MyProgram"]
        assert isinstance(result, ir.Program)
        assert result.name == "MyProgram"
        assert len(result.functions) == 1
    finally:
        linecache.cache.pop(filename, None)


def test_program_with_orig_argv_source(monkeypatch):
    """Test that @pl.program works via sys.orig_argv for python -c scenarios."""
    code = textwrap.dedent("""\
        import pypto_pro.language as pl

        @pl.program
        class MyProgram:
            @pl.function
            def add_one(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                return result
    """)
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-c", code], raising=False)
    filename = "<string>"
    compiled = compile(code, filename, "exec")
    namespace: dict = {}
    exec(compiled, namespace)  # noqa: S102
    result = namespace["MyProgram"]
    assert isinstance(result, ir.Program)
    assert result.name == "MyProgram"
    assert len(result.functions) == 1


def test_program_without_source_gives_clear_error():
    """Test that @pl.program gives a clear ParserSyntaxError when no source is available."""
    code = textwrap.dedent("""\
        import pypto_pro.language as pl
        from pypto_pro.language.parser.diagnostics._exceptions import ParserSyntaxError

        try:
            @pl.program
            class MyProgram:
                @pl.function
                def add_one(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                    result: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, 1.0)
                    return result
            assert False, "Should have raised ParserSyntaxError"
        except ParserSyntaxError as e:
            assert "Cannot retrieve source code" in str(e)
            assert "pl.parse()" in e.hint
    """)
    # Use a filename that won't be in linecache or on disk
    filename = "<no_source_available_program>"
    compiled = compile(code, filename, "exec")
    namespace: dict = {}
    exec(compiled, namespace)  # noqa: S102


def test_plain_helper_cannot_capture_caller_ir_variable():
    """A directly inlined helper must receive caller IR values as arguments."""

    def helper(x):
        return pl.tensor.add(x, caller_local)  # noqa: F821

    with pytest.raises(UndefinedVariableError, match="Undefined variable 'caller_local'"):

        @pl.function
        def invalid_capture(
            x: pl.Tensor[[64], pl.DT_FP32],
        ) -> pl.Tensor[[64], pl.DT_FP32]:
            caller_local: pl.Tensor[[64], pl.DT_FP32] = pl.tensor.add(x, x)  # noqa: F841
            return helper(x)


def test_inline_deprecated_warning():
    """@pl.inline is deprecated and should warn at decoration time."""

    with pytest.warns(DeprecationWarning, match="@pl.inline is deprecated"):

        @pl.inline
        def deprecated_inline(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            return x

    assert deprecated_inline.__name__ == "deprecated_inline"
