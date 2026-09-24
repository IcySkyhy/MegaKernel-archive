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


class RaiseTest(PILTestCase):

    def test_raise_bare(self):
        try:
            raise Expr.TypeA(0)
        except Expr.TypeA:
            try:
                raise
            except Expr.TypeA:
                Expr.str(1)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_raise_bare_simple(self):
        try:
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                raise
        except Expr.TypeA as e:
            Expr.int(e.value)

        self.assertEqual(Expr.trace, [('error', 0), ('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_raise_name(self):
        try:
            exc = Expr.TypeA(0)
            raise exc
        except Expr.TypeA as e:
            Expr.int(e.value)

        self.assertEqual(Expr.trace, [('error', 0), ('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_raise_attribute(self):
        var_obj = Expr(0)
        var_obj.val = Expr.TypeB(1)
        try:
            raise var_obj.val
        except Expr.TypeB as e:
            Expr.int(e.value)

        self.assertEqual(
            list(Expr.trace),
            [('init', 0), ('error', 1), ('setattr', 0, 'val', Expr.TypeB(1)), ('getattr', 0, 'val'), ('int', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': Expr.TypeB(1)})})

    def test_raise_subscript(self):
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.TypeC(2)
        try:
            raise var_obj[Expr.str(1)]
        except Expr.TypeC as e:
            Expr.int(e.value)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('error', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', Expr.TypeC(2)),
                ('str', 1),
                ('getitem', 0, 'str(1)'),
                ('int', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {'str(1)': Expr.TypeC(2)}, {})})

    def test_raise_ifexp_true(self):
        try:
            raise Expr.TypeA(0) if Expr.true(1) else Expr.TypeB(2)
        except Expr.TypeA as e:
            Expr.int(e.value)

        self.assertEqual(Expr.trace, [('true', 1), ('error', 0), ('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_raise_ifexp_false(self):
        try:
            raise Expr.TypeA(0) if Expr.false(1) else Expr.TypeB(2)
        except Expr.TypeB as e:
            Expr.int(e.value)

        self.assertEqual(Expr.trace, [('false', 1), ('error', 2), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_raise_from_name(self):
        try:
            cause = Expr.TypeA(0)
            raise Expr.TypeB(1) from cause
        except Expr.TypeB as e:
            Expr.int(e.value)

        self.assertEqual(Expr.trace, [('error', 0), ('error', 1), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_raise_from_attribute(self):
        var_obj = Expr(0)
        var_obj.val = Expr.TypeA(1)
        try:
            raise Expr.TypeB(2) from var_obj.val
        except Expr.TypeB as e:
            Expr.int(e.value)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('error', 1),
                ('setattr', 0, 'val', Expr.TypeA(1)),
                ('error', 2),
                ('getattr', 0, 'val'),
                ('int', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': Expr.TypeA(1)})})

    def test_raise_attribute_from_subscript(self):
        var_obj = Expr(0)
        var_obj.val = Expr.TypeB(1)
        var_obj[Expr.str(2)] = Expr.TypeA(3)
        try:
            raise var_obj.val from var_obj[Expr.str(2)]
        except Expr.TypeB as e:
            Expr.int(e.value)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('error', 1),
                ('setattr', 0, 'val', Expr.TypeB(1)),
                ('error', 3),
                ('str', 2),
                ('setitem', 0, 'str(2)', Expr.TypeA(3)),
                ('getattr', 0, 'val'),
                ('str', 2),
                ('getitem', 0, 'str(2)'),
                ('int', 1),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(0, {'str(2)': Expr.TypeA(3)}, {'val': Expr.TypeB(1)})},
        )


class TryTest(PILTestCase):

    def test_try_single_typed_no_binding(self):
        try:
            raise Expr.TypeA(0)
        except Expr.TypeA:
            Expr.str(1)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_single_typed_with_binding(self):
        try:
            raise Expr.TypeA(0)
        except Expr.TypeA as e:
            Expr.str(e.value)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_multi_handler_typea(self):
        try:
            raise Expr.TypeA(0)
        except Expr.TypeA:
            Expr.str(10)
        except Expr.TypeB:
            Expr.str(20)
        except Expr.TypeC:
            Expr.str(30)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 10)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_multi_handler_typeb(self):
        try:
            raise Expr.TypeB(0)
        except Expr.TypeA:
            Expr.str(10)
        except Expr.TypeB:
            Expr.str(20)
        except Expr.TypeC:
            Expr.str(30)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 20)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_multi_handler_typec(self):
        try:
            raise Expr.TypeC(0)
        except Expr.TypeA:
            Expr.str(10)
        except Expr.TypeB:
            Expr.str(20)
        except Expr.TypeC:
            Expr.str(30)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 30)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_bare_except(self):
        try:
            raise Expr.TypeA(0)
        except Expr.TypeB:
            Expr.str(10)
        except Exception:
            Expr.str(20)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 20)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_else_no_exception(self):
        try:
            Expr.str(0)
        except Expr.TypeA:
            Expr.str(10)
        else:
            Expr.str(20)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 20)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_else_with_exception(self):
        try:
            if Expr.true(0):
                raise Expr.TypeA(1)
        except Expr.TypeA:
            Expr.str(10)
        else:
            Expr.str(20)

        self.assertEqual(Expr.trace, [('true', 0), ('error', 1), ('str', 10)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_finally_no_exception(self):
        try:
            Expr.str(0)
        finally:
            Expr.str(1)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_finally_with_exception(self):
        try:
            raise Expr.TypeA(0)
        except Expr.TypeA:
            Expr.str(10)
        finally:
            Expr.str(20)

        self.assertEqual(Expr.trace, [('error', 0), ('str', 10), ('str', 20)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_combined_no_exception(self):
        try:
            Expr.str(0)
        except Expr.TypeA as e:
            Expr.int(e.value)
        except Expr.TypeB as e:
            Expr.str(e.value)
        else:
            Expr.str(30)
        finally:
            Expr.str(40)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 30), ('str', 40)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_combined_typea(self):
        try:
            if Expr.true(0):
                raise Expr.TypeA(1)
        except Expr.TypeA as e:
            Expr.int(e.value)
        except Expr.TypeB as e:
            Expr.str(e.value)
        else:
            Expr.str(30)
        finally:
            Expr.str(40)

        self.assertEqual(Expr.trace, [('true', 0), ('error', 1), ('int', 1), ('str', 40)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_combined_typeb(self):
        try:
            if Expr.true(0):
                raise Expr.TypeB(1)
        except Expr.TypeA as e:
            Expr.int(e.value)
        except Expr.TypeB as e:
            Expr.str(e.value)
        else:
            Expr.str(30)
        finally:
            Expr.str(40)

        self.assertEqual(Expr.trace, [('true', 0), ('error', 1), ('str', 1), ('str', 40)])
        self.assertEqual(self.get_var(locals()), {})

    def test_try_nested(self):
        try:
            try:
                raise Expr.TypeA(0)
            except Expr.TypeB:
                Expr.str(10)
        except Expr.TypeA as e:
            Expr.int(e.value)

        self.assertEqual(Expr.trace, [('error', 0), ('int', 0)])
        self.assertEqual(self.get_var(locals()), {})


class AssertTest(PILTestCase):

    def test_assert_pass_subscript(self):
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.true(2)
        assert var_obj[Expr.str(1)]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('true', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', True),
                ('str', 1),
                ('getitem', 0, 'str(1)'),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {'str(1)': True}, {})})

    def test_assert_pass_subscript_msg_call(self):
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.true(2)
        assert var_obj[Expr.str(1)], Expr.str(3)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('true', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', True),
                ('str', 1),
                ('getitem', 0, 'str(1)'),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {'str(1)': True}, {})})

    def test_assert_pass_subscript_msg_name(self):
        var_msg = 'assertion failed'
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.true(2)
        assert var_obj[Expr.str(1)], var_msg

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('true', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', True),
                ('str', 1),
                ('getitem', 0, 'str(1)'),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_msg': 'assertion failed', 'var_obj': Expr(0, {'str(1)': True}, {})},
        )

    def test_assert_fail_subscript_no_msg(self):
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.false(2)
        try:
            assert var_obj[Expr.str(1)]
        except AssertionError:
            Expr.str(3)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('false', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', False),
                ('str', 1),
                ('getitem', 0, 'str(1)'),
                ('str', 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {'str(1)': False}, {})})

    def test_assert_fail_subscript_msg_call(self):
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.false(2)
        try:
            assert var_obj[Expr.str(1)], Expr.str(3)
        except AssertionError:
            Expr.str(4)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('false', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', False),
                ('str', 1),
                ('getitem', 0, 'str(1)'),
                ('str', 3),
                ('str', 4),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {'str(1)': False}, {})})

    def test_assert_fail_subscript_msg_name(self):
        var_msg = 'assertion failed'
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.false(2)
        try:
            assert var_obj[Expr.str(1)], var_msg
        except AssertionError:
            Expr.str(3)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('false', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', False),
                ('str', 1),
                ('getitem', 0, 'str(1)'),
                ('str', 3),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_msg': 'assertion failed', 'var_obj': Expr(0, {'str(1)': False}, {})},
        )
