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


class ForTest(PILTestCase):

    def test_for_simple_name(self):
        for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
            Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_x': 2})

    def test_for_with_orelse(self):
        for var_x in [Expr.int(0), Expr.int(1)]:
            Expr.str(var_x)
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_x': 1})

    def test_for_tuple_unpack_target(self):
        for var_x, var_y in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]:
            Expr.str(var_x)
            Expr.str(var_y)

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 0), ('str', 1), ('str', 2), ('str', 3)],
        )
        self.assertEqual(self.get_var(locals()), {'var_x': 2, 'var_y': 3})

    def test_for_tuple_unpack_target_with_orelse(self):
        for var_x, var_y in [(Expr.int(0), Expr.int(1))]:
            Expr.str(var_x)
            Expr.str(var_y)
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_x': 0, 'var_y': 1})

    def test_for_attribute_target(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(-1)
        for var_obj.val in [Expr.int(0), Expr.int(1)]:
            Expr.str(var_obj.val)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', -1),
                ('setattr', 0, 'val', -1),
                ('int', 0),
                ('int', 1),
                ('setattr', 0, 'val', 0),
                ('getattr', 0, 'val'),
                ('str', 0),
                ('setattr', 0, 'val', 1),
                ('getattr', 0, 'val'),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_for_subscript_target(self):
        var_obj = Expr(0)
        var_obj[0] = Expr.int(-1)
        for var_obj[0] in [Expr.int(0), Expr.int(1)]:
            Expr.str(var_obj[0])

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', -1),
                ('setitem', 0, 0, -1),
                ('int', 0),
                ('int', 1),
                ('setitem', 0, 0, 0),
                ('getitem', 0, 0),
                ('str', 0),
                ('setitem', 0, 0, 1),
                ('getitem', 0, 0),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {})})

    def test_for_call_iter(self):
        for var_x in range(Expr.int(3)):
            Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 3), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_x': 2})

    def test_for_call_iter_with_orelse(self):
        for var_x in range(Expr.int(2)):
            Expr.str(var_x)
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('int', 2), ('str', 0), ('str', 1), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_x': 1})

    def test_for_nested(self):
        for var_i in [Expr.int(0), Expr.int(1)]:
            for var_j in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_i)
                Expr.str(var_j)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('str', 0),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 0),
                ('str', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': 1, 'var_j': 1})

    def test_for_nested_outer_orelse(self):
        for var_i in [Expr.int(0), Expr.int(1)]:
            for var_j in [Expr.int(0)]:
                Expr.str(var_i)
                Expr.str(var_j)
        else:
            Expr.str(99)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('str', 1),
                ('str', 0),
                ('str', 99),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': 1, 'var_j': 0})

    def test_for_nested_both_orelse(self):
        for var_i in [Expr.int(0), Expr.int(1)]:
            for var_j in [Expr.int(0)]:
                Expr.str(var_i)
                Expr.str(var_j)
            else:
                Expr.str(88)
        else:
            Expr.str(99)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('str', 0),
                ('str', 0),
                ('str', 88),
                ('int', 0),
                ('str', 1),
                ('str', 0),
                ('str', 88),
                ('str', 99),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': 1, 'var_j': 0})

    def test_for_three_level_nesting(self):
        for var_i in range(Expr.int(2)):
            for var_j in [Expr.int(0), Expr.int(1)]:
                for var_x, var_y in [(Expr.int(0), Expr.int(1))]:
                    Expr.str(var_i)
                    Expr.str(var_j)
                    Expr.str(var_x)
                    Expr.str(var_y)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 2),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('str', 0),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 1),
                ('str', 0),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 0),
                ('str', 0),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 1),
                ('str', 0),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': 1, 'var_j': 1, 'var_x': 0, 'var_y': 1})


class WhileTest(PILTestCase):

    def test_while_false_no_body(self):
        while Expr.false(0):
            Expr.str(1)

        self.assertEqual(Expr.trace, [('false', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_false_with_orelse(self):
        # orelse fires on natural exit (condition was False from the start)
        while Expr.false(0):
            Expr.str(1)
        else:
            Expr.str(2)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_true_break(self):
        while Expr.true(0):
            Expr.str(1)
            break

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_true_break_orelse_not_fired(self):
        # break suppresses orelse
        while Expr.true(0):
            Expr.str(1)
            break
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_true_body_then_break(self):
        while Expr.true(0):
            Expr.str(1)
            Expr.str(2)
            break

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_const_var_break(self):
        while 1:
            Expr.str(0)
            break

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_const_var_break_orelse_not_fired(self):
        while 1:
            Expr.str(0)
            break
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_const_var_multi_body_break(self):
        while 1:
            Expr.str(0)
            Expr.str(1)
            Expr.str(2)
            break

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_nested_func_func(self):
        while Expr.true(0):
            while Expr.true(1):
                Expr.str(2)
                break
            Expr.str(3)
            break

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('str', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_nested_func_func_inner_orelse(self):
        while Expr.true(0):
            while Expr.false(1):
                Expr.str(2)
            else:
                Expr.str(3)
            break

        self.assertEqual(Expr.trace, [('true', 0), ('false', 1), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_nested_const_outer_func_inner(self):
        while 1:
            while Expr.true(0):
                Expr.str(1)
                break
            break

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_nested_const_outer_false_inner_orelse(self):
        while True:
            while Expr.false(0):
                Expr.str(1)
            else:
                Expr.str(2)
            break

        self.assertEqual(Expr.trace, [('false', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_three_level_nesting(self):
        while 1:
            while Expr.true(0):
                while Expr.true(1):
                    Expr.str(2)
                    break
                Expr.str(3)
                break
            break

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('str', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_while_three_level_mixed_orelse(self):
        var_outer = True
        while var_outer:
            while Expr.true(0):
                while Expr.false(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
                break
            break

        self.assertEqual(Expr.trace, [('true', 0), ('false', 1), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {'var_outer': True})


class IfTest(PILTestCase):

    def test_simple_if_true(self):
        if Expr.true(0):
            Expr.str(1)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_simple_if_false(self):
        if Expr.false(0):
            Expr.str(1)

        self.assertEqual(Expr.trace, [('false', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_else_true(self):
        if Expr.true(0):
            Expr.str(1)
        else:
            Expr.str(2)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_else_false(self):
        if Expr.false(0):
            Expr.str(1)
        else:
            Expr.str(2)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_elif_else_first(self):
        if Expr.true(0):
            Expr.str(1)
        elif Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_elif_else_second(self):
        if Expr.false(0):
            Expr.str(1)
        elif Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('false', 0), ('true', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_elif_else_third(self):
        if Expr.false(0):
            Expr.str(1)
        elif Expr.false(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('false', 0), ('false', 2), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {})

    def test_nested_if_true_true(self):
        if Expr.true(0):
            if Expr.true(1):
                Expr.str(2)
            else:
                Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_nested_if_true_false(self):
        if Expr.true(0):
            if Expr.false(1):
                Expr.str(2)
            else:
                Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('false', 1), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_nested_if_false(self):
        if Expr.false(0):
            if Expr.true(1):
                Expr.str(2)
            else:
                Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_and_condition_both_true(self):
        if Expr.true(0) and Expr.true(1):
            Expr.str(2)
        else:
            Expr.str(3)

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_and_condition_first_false(self):
        if Expr.false(0) and True:
            Expr.str(2)
        else:
            Expr.str(3)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_and_condition_second_false(self):
        if Expr.true(0) and False:
            Expr.str(2)
        else:
            Expr.str(3)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_or_condition_both_false(self):
        if False or Expr.false(1):
            Expr.str(2)
        else:
            Expr.str(3)

        self.assertEqual(Expr.trace, [('false', 1), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_or_condition_first_true(self):
        if Expr.true(0) or Expr.false(1):
            Expr.str(2)
        else:
            Expr.str(3)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_or_condition_second_true(self):
        if Expr.false(0) or Expr.true(1):
            Expr.str(2)
        else:
            Expr.str(3)

        self.assertEqual(Expr.trace, [('false', 0), ('true', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_if_and_or_combined(self):
        if Expr.true(0) and Expr.false(1) or Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('false', 1), ('true', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_sequential_ifs(self):
        if Expr.true(0):
            Expr.str(1)
        if Expr.false(2):
            Expr.str(3)
        if Expr.true(4):
            Expr.str(5)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1), ('false', 2), ('true', 4), ('str', 5)])
        self.assertEqual(self.get_var(locals()), {})


class BreakTest(PILTestCase):

    def test_break_for_first(self):
        for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
            Expr.str(var_x)
            break

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_x': 0})

    def test_break_for_conditional(self):
        for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
            Expr.str(var_x)
            if Expr.true(var_x):
                break

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('true', 0)])
        self.assertEqual(self.get_var(locals()), {'var_x': 0})

    def test_break_for_suppresses_orelse(self):
        for var_x in [Expr.int(0), Expr.int(1)]:
            Expr.str(var_x)
            break
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_x': 0})

    def test_break_for_nested_inner(self):
        for var_i in [Expr.int(0), Expr.int(1)]:
            for var_j in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                Expr.str(var_j)
                break
            Expr.str(var_i)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('str', 0),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': 1, 'var_j': 0})

    def test_break_for_nested_outer(self):
        for var_i in [Expr.int(0), Expr.int(1)]:
            for var_j in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_j)
            Expr.str(var_i)
            break

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 0)],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': 0, 'var_j': 1})

    def test_break_while_const(self):
        while True:
            Expr.str(0)
            break

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_break_while_func_cond(self):
        while Expr.true(0):
            Expr.str(1)
            break

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_break_while_suppresses_orelse(self):
        while Expr.true(0):
            Expr.str(1)
            break
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_break_while_named_expr_cond(self):
        var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
        var_i = [0]
        while var_n := var_i[0] < len(var_items):
            Expr.str(var_items[var_i[0]])
            var_i[0] = var_i[0] + 1
            if var_i[0] == 2:
                break
        Expr.str(var_n)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_items': [0, 1, 2], 'var_i': [2], 'var_n': True})

    def test_break_while_named_expr_cond_suppresses_orelse(self):
        var_items = [Expr.int(0), Expr.int(1)]
        var_i = [0]
        while var_n := var_i[0] < len(var_items):
            Expr.str(var_items[var_i[0]])
            break
        else:
            Expr.str(99)
        Expr.str(var_n)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_items': [0, 1], 'var_i': [0], 'var_n': True})

    def test_break_while_nested(self):
        while Expr.true(0):
            while Expr.true(1):
                Expr.str(2)
                break
            Expr.str(3)
            break

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('str', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})


class ContinueTest(PILTestCase):

    def test_continue_for_basic(self):
        for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
            continue
            Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {'var_x': 2})

    def test_continue_for_conditional(self):
        for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
            if Expr.true(var_x):
                continue
            Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('true', 0), ('true', 1), ('true', 2)])
        self.assertEqual(self.get_var(locals()), {'var_x': 2})

    def test_continue_for_orelse_fires(self):
        for var_x in [Expr.int(0), Expr.int(1)]:
            Expr.str(var_x)
            continue
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_x': 1})

    def test_continue_for_nested_inner(self):
        for var_i in [Expr.int(0), Expr.int(1)]:
            Expr.str(var_i)
            for var_j in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                if Expr.true(var_j):
                    continue
                Expr.str(var_j)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('true', 0),
                ('true', 1),
                ('true', 2),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('true', 0),
                ('true', 1),
                ('true', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': 1, 'var_j': 2})

    def test_continue_for_nested_outer(self):
        for var_i in [Expr.int(0), Expr.int(1), Expr.int(2)]:
            if Expr.true(var_i):
                continue
            for var_j in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_j)
            Expr.str(var_i)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('true', 0), ('true', 1), ('true', 2)])
        self.assertEqual(self.get_var(locals()), {'var_i': 2})

    def test_continue_while_const(self):
        var_n = [0]
        while var_n[0] < 3:
            var_n[0] = var_n[0] + 1
            if var_n[0] == 2:
                continue
            Expr.str(var_n[0])

        self.assertEqual(Expr.trace, [('str', 1), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {'var_n': [3]})

    def test_continue_while_func_cond(self):
        var_i = [0]
        while Expr.true(var_i[0]):
            var_i[0] = var_i[0] + 1
            if var_i[0] < 2:
                continue
            Expr.str(var_i[0])
            break

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_i': [2]})

    def test_continue_while_orelse_fires(self):
        var_n = [0]
        while var_n[0] < 2:
            var_n[0] = var_n[0] + 1
            continue
        else:
            Expr.str(99)

        self.assertEqual(Expr.trace, [('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_n': [2]})

    def test_continue_while_named_expr_cond(self):
        var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
        var_i = [0]
        while var_n := var_i[0] < len(var_items):
            var_cur = var_i[0]
            var_i[0] = var_i[0] + 1
            if var_cur == 1:
                continue
            Expr.str(var_items[var_cur])
        Expr.str(var_n)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 2), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_items': [0, 1, 2], 'var_i': [3], 'var_n': False, 'var_cur': 2})

    def test_continue_while_nested(self):
        var_i = [0]
        while Expr.true(var_i[0]):
            var_j = [0]
            while Expr.true(var_j[0]):
                var_j[0] = var_j[0] + 1
                if var_j[0] < 2:
                    continue
                Expr.str(var_j[0])
                break
            var_i[0] = var_i[0] + 1
            if var_i[0] < 2:
                continue
            Expr.str(var_i[0])
            break

        self.assertEqual(
            Expr.trace,
            [
                ('true', 0),
                ('true', 0),
                ('true', 1),
                ('str', 2),
                ('true', 1),
                ('true', 0),
                ('true', 1),
                ('str', 2),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_i': [2], 'var_j': [2]})


class WithTest(PILTestCase):

    def test_with_single_no_as(self):
        with Expr.ContextManager(enter_n=0, exit_n=1):
            Expr.str(2)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 2), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_with_single_as_name(self):
        with Expr.ContextManager(enter_n=0, exit_n=1) as var_ctx:
            Expr.str(2)
            Expr.str(var_ctx._enter_n)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 2), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_ctx': Expr.ContextManager(enter_n=0, exit_n=1)})

    def test_with_ctx_from_call(self):
        def make_cm():
            return Expr.ContextManager(init_n=Expr.int(0))

        with make_cm():
            Expr.str(1)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_with_multiple_no_as(self):
        with Expr.ContextManager(enter_n=0, exit_n=10), Expr.ContextManager(enter_n=1, exit_n=11):
            Expr.str(2)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1), ('str', 2), ('str', 11), ('str', 10)])
        self.assertEqual(self.get_var(locals()), {})

    def test_with_multiple_as_names(self):
        with (
            Expr.ContextManager(enter_n=0, exit_n=10) as var_a,
            Expr.ContextManager(enter_n=1, exit_n=11) as var_b,
        ):
            Expr.str(2)
            Expr.str(var_a._enter_n)
            Expr.str(var_b._enter_n)

        self.assertEqual(
            Expr.trace,
            [('str', 0), ('str', 1), ('str', 2), ('str', 0), ('str', 1), ('str', 11), ('str', 10)],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_a': Expr.ContextManager(enter_n=0, exit_n=10), 'var_b': Expr.ContextManager(enter_n=1, exit_n=11)},
        )

    def test_with_multiple_mixed_as(self):
        with (
            Expr.ContextManager(enter_n=0, exit_n=10) as var_a,
            Expr.ContextManager(enter_n=1, exit_n=11),
        ):
            Expr.str(2)
            Expr.str(var_a._enter_n)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1), ('str', 2), ('str', 0), ('str', 11), ('str', 10)])
        self.assertEqual(self.get_var(locals()), {'var_a': Expr.ContextManager(enter_n=0, exit_n=10)})

    def test_with_three_items(self):
        with (
            Expr.ContextManager(enter_n=0, exit_n=10),
            Expr.ContextManager(enter_n=1, exit_n=11),
            Expr.ContextManager(enter_n=2, exit_n=12),
        ):
            Expr.str(3)

        self.assertEqual(
            Expr.trace,
            [('str', 0), ('str', 1), ('str', 2), ('str', 3), ('str', 12), ('str', 11), ('str', 10)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_with_as_attr_target(self):
        var_obj = Expr(2)
        with Expr.ContextManager(enter_n=0, exit_n=1) as var_obj.val:
            Expr.str(3)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 2),
                ('str', 0),
                ('setattr', 2, 'val', Expr.ContextManager(enter_n=0, exit_n=1)),
                ('str', 3),
                ('str', 1),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(2, {}, {'val': Expr.ContextManager(enter_n=0, exit_n=1)})},
        )

    def test_with_as_subscript_target(self):
        var_obj = Expr(2)
        var_obj[0] = None
        with Expr.ContextManager(enter_n=0, exit_n=1) as var_obj[0]:
            Expr.str(3)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 2),
                ('setitem', 2, 0, None),
                ('str', 0),
                ('setitem', 2, 0, Expr.ContextManager(enter_n=0, exit_n=1)),
                ('str', 3),
                ('str', 1),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(2, {0: Expr.ContextManager(enter_n=0, exit_n=1)}, {})},
        )
