#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from .test_pil_builder_utils import Expr, PILTestCase


class FunctionDefTest(PILTestCase):

    def test_three_level_nesting(self):
        def outer():
            def middle():
                def inner():
                    Expr.str(0)

                inner()

            middle()

        outer()

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_three_level_nesting_with_return_values(self):
        def outer():
            def middle():
                def inner():
                    Expr.str(0)
                    Expr.str(1)

                inner()
                Expr.str(2)

            middle()
            Expr.str(3)

        outer()

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1), ('str', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_function_with_decorator(self):
        var_e = Expr(0)

        @var_e.decorate(1)
        def func():
            Expr.str(2)

        func()

        self.assertEqual(Expr.trace, [('init', 0), ('decorate', 1), ('decorate.wrapper', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_e': Expr(0, {}, {})})

    def test_function_with_multiple_decorators(self):
        var_e = Expr(0)

        @var_e.decorate(1)
        @var_e.decorate(2)
        def func():
            Expr.str(3)

        func()

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('decorate', 1),
                ('decorate', 2),
                ('decorate.wrapper', 2),
                ('decorate.wrapper', 1),
                ('str', 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_e': Expr(0, {}, {})})

    def test_function_with_default_arg(self):
        def func(x=Expr.int(0)):
            Expr.str(x)

        func()
        func(Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_function_with_multiple_defaults(self):
        def func(x=Expr.int(0), y=Expr.int(1)):
            Expr.str(x)
            Expr.str(y)

        func()

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_three_level_nesting_with_decorator_and_default(self):
        var_e = Expr(0)

        @var_e.decorate(1)
        def outer(x=Expr.int(2)):
            @var_e.decorate(3)
            def middle(y=Expr.int(4)):
                def inner():
                    Expr.str(x)
                    Expr.str(y)

                inner()

            middle()

        outer()

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('decorate', 1),
                ('int', 2),
                ('decorate.wrapper', 1),
                ('decorate', 3),
                ('int', 4),
                ('decorate.wrapper', 3),
                ('str', 2),
                ('str', 4),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_e': Expr(0, {}, {})})


class LambdaTest(PILTestCase):

    def test_lambda_no_args(self):
        def f():
            return Expr.str(0)

        _var_r = f()

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_single_arg(self):
        def f(x):
            return Expr.str(x)

        _var_r = f(Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_multiple_args(self):
        def f(x, y):
            return Expr.str(x)

        _var_r = f(Expr.int(0), Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_default_not_overridden(self):
        def f(x=Expr.int(0)):
            return Expr.str(x)

        _var_r = f()

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_default_overridden(self):
        def f(x=Expr.int(0)):
            return Expr.str(x)

        _var_r = f(Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_vararg(self):
        def f(*args):
            return Expr.str(args[0])

        _var_r = f(Expr.int(0), Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_kwonly(self):
        def f(*, key):
            return Expr.str(key)

        _var_r = f(key=Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_kwonly_default_not_overridden(self):
        def f(*, key=0):
            return Expr.str(key)

        _var_r = f()

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_kwargs(self):
        def f(**kw):
            return Expr.str(kw['x'])

        _var_r = f(x=Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_body_const(self):
        def f():
            return 42

        var_x = f()
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('str', 42)])
        self.assertEqual(self.get_var(locals()), {'var_x': 42})

    def test_lambda_body_binop(self):
        def f(x):
            return x + Expr.int(1)

        var_x = f(Expr.int(0))
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_x': 1})

    def test_lambda_body_ifexp(self):
        def f(x):
            return Expr.str(0) if Expr.true(x) else Expr.str(1)

        _var_r = f(Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('true', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_body_nested_call(self):
        def f(x):
            return Expr.str(Expr.int(x))

        _var_r = f(0)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_nested(self):
        def outer(x):
            return lambda y: Expr.str(x)

        inner = outer(Expr.int(0))
        _var_r = inner(Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_lambda_as_argument(self):
        def apply(fn, val):
            return fn(val)

        var_r = apply(lambda x: Expr.int(x), Expr.int(0))
        Expr.str(var_r)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_r': 0})

    def test_lambda_in_listcomp(self):
        fs = [lambda x=Expr.int(i):Expr.str(x) for i in range(Expr.int(3))]
        for f in fs:
            _var_r = f()

        self.assertEqual(
            Expr.trace,
            [('int', 3), ('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)],
        )
        self.assertEqual(self.get_var(locals()), {})


class ReturnTest(PILTestCase):

    def test_return_bare(self):
        def func():
            Expr.str(0)
            return

        func()

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_return_name(self):
        def func():
            var_x = Expr.int(0)
            return var_x

        var_r = func()
        Expr.str(var_r)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_r': 0})

    def test_return_call_expr(self):
        def func():
            return Expr.int(0)

        var_r = func()
        Expr.str(var_r)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_r': 0})

    def test_return_constant(self):
        def func():
            return 42

        var_r = func()
        Expr.str(var_r)

        self.assertEqual(Expr.trace, [('str', 42)])
        self.assertEqual(self.get_var(locals()), {'var_r': 42})

    def test_return_binop(self):
        def func():
            return Expr.int(0) + Expr.int(1)

        var_r = func()
        Expr.str(var_r)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_r': 1})

    def test_return_tuple(self):
        def func():
            return (Expr.int(0), Expr.int(1))

        var_r = func()
        Expr.str(var_r[0])
        Expr.str(var_r[1])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_r': (0, 1)})

    def test_return_tuple_starred(self):
        def make():
            return [Expr.int(1), Expr.int(2)]

        def func():
            return (Expr.int(0), *make())

        var_r = func()
        Expr.str(var_r[0])
        Expr.str(var_r[1])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_r': (0, 1, 2)})

    def test_return_const_tuple(self):
        def func():
            return (0, 1, 2)

        var_r = func()
        Expr.str(var_r[0])

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_r': (0, 1, 2)})

    def test_return_early(self):
        def func(flag):
            if flag:
                return Expr.int(0)
            return Expr.int(1)

        var_a = func(Expr.true(0))
        Expr.str(var_a)
        var_b = func(Expr.false(1))
        Expr.str(var_b)

        self.assertEqual(Expr.trace, [('true', 0), ('int', 0), ('str', 0), ('false', 1), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1})

    def test_return_nested(self):
        def outer():
            def inner():
                return Expr.int(0)

            var_x = inner()
            Expr.str(var_x)
            return Expr.int(1)

        var_r = outer()
        Expr.str(var_r)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_r': 1})


class CallTest(PILTestCase):

    def test_call_pos_int_const(self):
        Expr.str(0)

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_str_const(self):
        Expr.str('hello')

        self.assertEqual(Expr.trace, [('str', 'hello')])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_none_const(self):
        def func(x):
            Expr.str(x)

        func(None)

        self.assertEqual(Expr.trace, [('str', None)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_bool_const(self):
        def func(x):
            Expr.str(x)

        func(True)

        self.assertEqual(Expr.trace, [('str', True)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_multiple_consts(self):
        def func(x, y):
            Expr.str(x)
            Expr.str(y)

        func(0, 1)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_mixed_const_and_expr(self):
        def func(x, y):
            Expr.str(x)
            Expr.str(y)

        func(0, Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_named_arg_int_const(self):
        def func(x):
            Expr.str(x)

        func(x=0)

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_named_arg_str_const(self):
        def func(x):
            Expr.str(x)

        func(x='hello')

        self.assertEqual(Expr.trace, [('str', 'hello')])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_named_arg_none_const(self):
        def func(x):
            Expr.str(x)

        func(x=None)

        self.assertEqual(Expr.trace, [('str', None)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_named_arg_multiple_consts(self):
        def func(x, y):
            Expr.str(x)
            Expr.str(y)

        func(x=0, y=1)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_named_arg_mixed_const_and_expr(self):
        def func(x, y):
            Expr.str(x)
            Expr.str(y)

        func(x=0, y=Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_keyword_override_one_default(self):
        def func(x=0, y=1):
            Expr.str(x)
            Expr.str(y)

        func(x=Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_keyword_override_all_defaults(self):
        def func(x=0, y=0):
            Expr.str(x)
            Expr.str(y)

        func(x=Expr.int(0), y=Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_keyword_arg_expr_value(self):
        def func(x):
            Expr.str(x)

        func(x=Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_vararg_empty(self):
        def func(*args):
            for var_a in args:
                Expr.str(var_a)

        func()

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_vararg_one(self):
        def func(*args):
            for var_a in args:
                Expr.str(var_a)

        func(Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_vararg_many(self):
        def func(*args):
            for var_a in args:
                Expr.str(var_a)

        func(Expr.int(0), Expr.int(1), Expr.int(2))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_and_vararg(self):
        def func(x, *args):
            Expr.str(x)
            for var_a in args:
                Expr.str(var_a)

        func(Expr.int(0), Expr.int(1), Expr.int(2))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_kwonly_required(self):
        def func(*, key):
            Expr.str(key)

        func(key=Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_kwonly_default_not_overridden(self):
        def func(*, key=0):
            Expr.str(key)

        func()

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_kwonly_default_overridden(self):
        def func(*, key=0):
            Expr.str(key)

        func(key=Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_and_kwonly(self):
        def func(x, *, y):
            Expr.str(x)
            Expr.str(y)

        func(Expr.int(0), y=Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_vararg_and_kwonly(self):
        def func(*args, key):
            for var_a in args:
                Expr.str(var_a)
            Expr.str(key)

        func(Expr.int(0), Expr.int(1), key=Expr.int(2))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_double_star_expand(self):
        def func(x, y):
            Expr.str(x)
            Expr.str(y)

        var_d = {'x': Expr.int(0), 'y': Expr.int(1)}
        func(**var_d)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_d': {'x': 0, 'y': 1}})

    def test_call_double_star_from_func(self):
        def make():
            return {'x': Expr.int(0), 'y': Expr.int(1)}

        def func(x, y):
            Expr.str(x)
            Expr.str(y)

        func(**make())

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_pos_and_double_star(self):
        def func(x, y, z):
            Expr.str(x)
            Expr.str(y)
            Expr.str(z)

        var_d = {'y': Expr.int(1), 'z': Expr.int(2)}
        func(Expr.int(0), **var_d)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('int', 0), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_d': {'y': 1, 'z': 2}})

    def test_call_keyword_and_double_star(self):
        def func(x, y, z):
            Expr.str(x)
            Expr.str(y)
            Expr.str(z)

        var_d = {'z': Expr.int(2)}
        func(Expr.int(0), y=Expr.int(1), **var_d)

        self.assertEqual(Expr.trace, [('int', 2), ('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_d': {'z': 2}})

    def test_call_nested_pos_arg(self):
        Expr.str(Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_nested_multiple_pos_args(self):
        def func(x, y):
            Expr.str(x)
            Expr.str(y)

        func(Expr.int(0), Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_nested_keyword_arg(self):
        def func(key):
            Expr.str(key)

        func(key=Expr.int(0))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_nested_two_deep(self):
        def inner():
            return Expr.int(0)

        Expr.str(inner())

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_nested_three_deep(self):
        def inner():
            return Expr.int(0)

        def middle(x):
            return x

        Expr.str(middle(inner()))

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_call_star(self):
        def inner():
            return [Expr.int(0), Expr.int(1)]

        def middle(a, b):
            return a, b, Expr.int(3)

        middle(*inner())

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})
