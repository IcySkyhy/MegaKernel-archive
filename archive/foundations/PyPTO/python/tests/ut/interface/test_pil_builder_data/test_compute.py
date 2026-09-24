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


class BoolopTest(PILTestCase):

    def test_true_and_true_and_true(self):
        if Expr.true(0) and Expr.true(1) and Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('true', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_true_and_false_and_true(self):
        if Expr.true(0) and Expr.false(1) and Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('false', 1), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {})

    def test_false_and_true_and_true(self):
        if Expr.false(0) and Expr.true(1) and Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {})

    def test_false_or_false_or_false(self):
        if Expr.true(0) and Expr.true(1) and Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('true', 1), ('true', 2), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_false_or_true_or_false(self):
        if Expr.true(0) and Expr.false(1) and Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 0), ('false', 1), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {})

    def test_true_or_false_or_false(self):
        if Expr.false(0) and Expr.true(1) and Expr.true(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {})


class IfexpTest(PILTestCase):

    def test_true_test_body_selected(self):
        _var_x = Expr.str(0) if Expr.true(1) else Expr.str(2)

        self.assertEqual(Expr.trace, [('true', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_false_test_orelse_selected(self):
        _var_x = Expr.str(0) if Expr.false(1) else Expr.str(2)

        self.assertEqual(Expr.trace, [('false', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_nested_ifexp_in_body(self):
        _var_x = (Expr.str(0) if Expr.true(1) else Expr.str(2)) if Expr.true(3) else Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 3), ('true', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_nested_ifexp_in_orelse(self):
        _var_x = Expr.str(0) if Expr.false(1) else (Expr.str(2) if Expr.true(3) else Expr.str(4))

        self.assertEqual(Expr.trace, [('false', 1), ('true', 3), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_nested_ifexp_in_orelse_false(self):
        _var_x = Expr.str(0) if Expr.false(1) else (Expr.str(2) if Expr.false(3) else Expr.str(4))

        self.assertEqual(Expr.trace, [('false', 1), ('false', 3), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {})

    def test_ifexp_as_if_test(self):
        if Expr.str(0) if Expr.true(1) else Expr.str(2):
            Expr.str(3)
        else:
            Expr.str(4)

        self.assertEqual(Expr.trace, [('true', 1), ('str', 0), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {})


class BinOpTest(PILTestCase):

    def test_add(self):
        _var_x = Expr.int(0) + Expr.int(1) * Expr.int(2)
        _var_y = Expr.int(0) - Expr.int(1) // Expr.int(2)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})


class UnaryOpTest(PILTestCase):

    def test_add(self):
        _var_x = -Expr.int(0) + Expr.int(1) * Expr.int(2)
        _var_y = -Expr.int(0) - Expr.int(1) // Expr.int(2)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})


class CompareTest(PILTestCase):

    def test_cmp_lt_true(self):
        var_x = Expr.int(1) < Expr.int(2)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_lt_false(self):
        var_x = Expr.int(2) < Expr.int(1)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 2), ('int', 1), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_x': False})

    def test_cmp_lte(self):
        var_x = Expr.int(1) <= Expr.int(1)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_gt(self):
        var_x = Expr.int(2) > Expr.int(1)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 2), ('int', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_gte(self):
        var_x = Expr.int(2) >= Expr.int(2)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 2), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_eq_true(self):
        var_x = Expr.int(1) == Expr.int(1)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_eq_false(self):
        var_x = Expr.int(1) == Expr.int(2)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_x': False})

    def test_cmp_neq(self):
        var_x = Expr.int(1) != Expr.int(2)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_is(self):
        var_a = Expr.int(0)
        var_x = var_a is var_a
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_x': True})

    def test_cmp_is_not(self):
        var_a = Expr.int(0)
        var_b = Expr.int(1)
        var_x = var_a is not var_b
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_x': True})

    def test_cmp_in(self):
        var_l = [Expr.int(0), Expr.int(1)]
        var_x = Expr.int(0) in var_l
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 0), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1], 'var_x': True})

    def test_cmp_not_in(self):
        var_l = [Expr.int(0), Expr.int(1)]
        var_x = Expr.int(2) not in var_l
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1], 'var_x': True})

    def test_cmp_chain_lt_lt_all_true(self):
        # e.g. 1 < 2 < 3 — both sub-comparisons true, b evaluated once
        var_x = Expr.int(1) < Expr.int(2) < Expr.int(3)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('int', 3), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_chain_lt_lt_first_false(self):
        # e.g. 3 < 2 < 4 — first false, third operand not evaluated
        var_x = Expr.int(3) < Expr.int(2) < Expr.int(4)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 3), ('int', 2), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_x': False})

    def test_cmp_chain_lt_eq(self):
        # e.g. 1 < 2 == 2
        var_x = Expr.int(1) < Expr.int(2) == Expr.int(2)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_chain_three_ops(self):
        # e.g. 1 < 2 <= 3 < 4
        var_x = Expr.int(1) < Expr.int(2) <= Expr.int(3) < Expr.int(4)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('int', 3), ('int', 4), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_in_binop(self):
        var_x = (Expr.int(1) < Expr.int(2)) + 0
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_x': 1})

    def test_cmp_in_unary(self):
        var_x = not (Expr.int(1) == Expr.int(2))
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_cmp_if_true(self):
        if Expr.int(1) < Expr.int(2):
            Expr.str(0)
        else:
            Expr.str(1)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_cmp_if_false(self):
        if Expr.int(2) < Expr.int(1):
            Expr.str(0)
        else:
            Expr.str(1)

        self.assertEqual(Expr.trace, [('int', 2), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_cmp_for_iter(self):
        for var_x in [Expr.int(0) < Expr.int(1), Expr.int(2) < Expr.int(1)]:
            Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 1), ('str', True), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_x': False})

    def test_cmp_while_test(self):
        var_n = [0]
        while var_n[0] < 3:
            Expr.str(var_n[0])
            var_n[0] = var_n[0] + 1

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_n': [3]})

    def test_cmp_call_pos_arg(self):
        Expr.str(Expr.int(1) < Expr.int(2))

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {})

    def test_cmp_call_kw_arg(self):
        def func(x):
            Expr.str(x)

        func(x=Expr.int(1) == Expr.int(1))

        self.assertEqual(Expr.trace, [('int', 1), ('int', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {})

    def test_cmp_in_tuple(self):
        var_t = (Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4))
        Expr.str(var_t[0])
        Expr.str(var_t[1])

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('int', 3), ('int', 4), ('str', True), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_t': (True, False)})

    def test_cmp_in_list(self):
        var_l = [Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4)]
        Expr.str(var_l[0])
        Expr.str(var_l[1])

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('int', 3), ('int', 4), ('str', True), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_l': [True, False]})

    def test_cmp_dict_value(self):
        var_d = {0: Expr.int(1) < Expr.int(2)}
        Expr.str(var_d[0])

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_d': {0: True}})

    def test_cmp_dict_key(self):
        var_d = {Expr.int(1) == Expr.int(1): Expr.int(0)}
        Expr.str(var_d[True])

        self.assertEqual(Expr.trace, [('int', 1), ('int', 1), ('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_d': {True: 0}})

    def test_cmp_in_set(self):
        var_s = {Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4)}
        Expr.str(True in var_s)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('int', 3), ('int', 4), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_s': {False, True}})

    def test_cmp_as_subscript_index(self):
        var_arr = Expr(0)
        var_arr[True] = Expr.int(99)
        var_x = var_arr[Expr.int(1) == Expr.int(1)]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 99),
                ('setitem', 0, True, 99),
                ('int', 1),
                ('int', 1),
                ('getitem', 0, True),
                ('str', 99),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {True: 99}, {}), 'var_x': 99})

    def test_cmp_as_slice_bound(self):
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
        # True == 1, so slice [True:3] == [1:3]
        var_s = var_l[Expr.int(1) == Expr.int(1):]
        Expr.str(var_s[0])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 1), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2], 'var_s': [1, 2]})

    def test_cmp_annotation_with_value(self):
        var_x: bool = Expr.int(1) < Expr.int(2)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})
