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


class ListCompTest(PILTestCase):

    def test_listcomp_simple(self):
        _var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_one_if(self):
        _var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x)]

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('true', 0),
                ('str', 0),
                ('true', 1),
                ('str', 1),
                ('true', 2),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_two_ifs(self):
        _var_l = [
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]
            if Expr.true(var_x)
            if var_x != 2
        ]

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('true', 0), ('str', 0), ('true', 1), ('str', 1), ('true', 2)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_if_boolop(self):
        _var_l = [
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]
            if Expr.true(0) and Expr.true(var_x)
        ]

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('true', 0),
                ('true', 0),
                ('str', 0),
                ('true', 0),
                ('true', 1),
                ('str', 1),
                ('true', 0),
                ('true', 2),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_if_ifexp(self):
        _var_l = [
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))
        ]

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('true', 0), ('true', 0), ('str', 0), ('true', 0), ('true', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_target_tuple(self):
        _var_l = [Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_target_list(self):
        _var_l = [Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_target_attr(self):
        var_obj = Expr(0)
        var_obj.val = None
        _var_l = [Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setattr', 0, 'val', None),
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

    def test_listcomp_target_subscript(self):
        var_arr = Expr(0)
        var_arr[0] = None
        _var_l = [Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setitem', 0, 0, None),
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
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {0: 1}, {})})

    def test_listcomp_two_fors(self):
        _var_l = [Expr.str(var_x)
                  for var_x in [Expr.int(0), Expr.int(1)]
                  for var_y in [Expr.int(2), Expr.int(3)]]

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('str', 0),
                ('str', 0),
                ('int', 2),
                ('int', 3),
                ('str', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_two_fors_with_if(self):
        _var_l = [
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            if Expr.true(var_x)
            for var_y in [Expr.int(2), Expr.int(3)]
            if Expr.true(var_y)
        ]

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('true', 0),
                ('int', 2),
                ('int', 3),
                ('true', 2),
                ('str', 0),
                ('true', 3),
                ('str', 0),
                ('true', 1),
                ('int', 2),
                ('int', 3),
                ('true', 2),
                ('str', 1),
                ('true', 3),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_listcomp_three_fors(self):
        _var_l = [
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            for var_y in [Expr.int(0), Expr.int(1)]
            for var_z in [Expr.int(0), Expr.int(1)]
        ]

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})


class SetCompTest(PILTestCase):

    def test_setcomp_simple(self):
        _var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_setcomp_one_if(self):
        _var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x)}

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('true', 0),
                ('str', 0),
                ('true', 1),
                ('str', 1),
                ('true', 2),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_setcomp_if_boolop(self):
        _var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)] if Expr.true(0) and Expr.true(var_x)}

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('true', 0), ('true', 0), ('str', 0), ('true', 0), ('true', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_setcomp_if_ifexp(self):
        _var_s = {
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))
        }

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('true', 0), ('true', 0), ('str', 0), ('true', 0), ('true', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_setcomp_target_tuple(self):
        _var_s = {Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_setcomp_target_list(self):
        _var_s = {Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_setcomp_target_attr(self):
        var_obj = Expr(0)
        var_obj.val = None
        _var_s = {Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]}

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setattr', 0, 'val', None),
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

    def test_setcomp_target_subscript(self):
        var_arr = Expr(0)
        var_arr[0] = None
        _var_s = {Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]}

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setitem', 0, 0, None),
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
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {0: 1}, {})})

    def test_setcomp_two_fors(self):
        _var_s = {Expr.str(var_x)
                  for var_x in [Expr.int(0), Expr.int(1)]
                  for var_y in [Expr.int(2), Expr.int(3)]}

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('str', 0),
                ('str', 0),
                ('int', 2),
                ('int', 3),
                ('str', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_setcomp_three_fors(self):
        _var_s = {
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            for var_y in [Expr.int(0), Expr.int(1)]
            for var_z in [Expr.int(0), Expr.int(1)]
        }

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})


class DictCompTest(PILTestCase):

    def test_dictcomp_simple(self):
        _var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 0), ('str', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dictcomp_one_if(self):
        _var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)] if Expr.true(var_x)}

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('true', 0), ('int', 0), ('str', 0), ('true', 1), ('int', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_dictcomp_if_boolop(self):
        _var_d = {
            Expr.int(var_x): Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            if Expr.true(0) and Expr.true(var_x)
        }

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('true', 0),
                ('true', 0),
                ('int', 0),
                ('str', 0),
                ('true', 0),
                ('true', 1),
                ('int', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_dictcomp_if_ifexp(self):
        _var_d = {
            Expr.int(var_x): Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))
        }

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('true', 0),
                ('true', 0),
                ('int', 0),
                ('str', 0),
                ('true', 0),
                ('true', 1),
                ('int', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_dictcomp_target_tuple(self):
        _var_d = {
            Expr.int(var_a): Expr.str(var_b)
            for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]
        }

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('int', 0), ('str', 1), ('int', 2), ('str', 3)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_dictcomp_target_list(self):
        _var_d = {
            Expr.int(var_a): Expr.str(var_b)
            for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]
        }

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('int', 0), ('str', 1), ('int', 2), ('str', 3)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_dictcomp_target_attr(self):
        var_obj = Expr(0)
        var_obj.val = None
        _var_d = {Expr.int(var_obj.val): Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]}

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setattr', 0, 'val', None),
                ('int', 0),
                ('int', 1),
                ('setattr', 0, 'val', 0),
                ('getattr', 0, 'val'),
                ('int', 0),
                ('getattr', 0, 'val'),
                ('str', 0),
                ('setattr', 0, 'val', 1),
                ('getattr', 0, 'val'),
                ('int', 1),
                ('getattr', 0, 'val'),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_dictcomp_target_subscript(self):
        var_arr = Expr(0)
        var_arr[0] = None
        _var_d = {Expr.int(var_arr[0]): Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]}

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setitem', 0, 0, None),
                ('int', 0),
                ('int', 1),
                ('setitem', 0, 0, 0),
                ('getitem', 0, 0),
                ('int', 0),
                ('getitem', 0, 0),
                ('str', 0),
                ('setitem', 0, 0, 1),
                ('getitem', 0, 0),
                ('int', 1),
                ('getitem', 0, 0),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {0: 1}, {})})

    def test_dictcomp_two_fors(self):
        _var_d = {
            Expr.int(var_x): Expr.str(var_y)
            for var_x in [Expr.int(0), Expr.int(1)]
            for var_y in [Expr.int(2), Expr.int(3)]
        }

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('int', 0),
                ('str', 2),
                ('int', 0),
                ('str', 3),
                ('int', 2),
                ('int', 3),
                ('int', 1),
                ('str', 2),
                ('int', 1),
                ('str', 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_dictcomp_three_fors(self):
        _var_d = {
            Expr.int(var_x): Expr.str(var_z)
            for var_x in [Expr.int(0), Expr.int(1)]
            for var_y in [Expr.int(0), Expr.int(1)]
            for var_z in [Expr.int(0), Expr.int(1)]
        }

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('str', 0),
                ('int', 0),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('str', 0),
                ('int', 0),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('int', 1),
                ('str', 0),
                ('int', 1),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('int', 1),
                ('str', 0),
                ('int', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})


class GeneratorExpTest(PILTestCase):

    def test_genexp_simple(self):
        g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)])
        _var_l = list(g)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_genexp_one_if(self):
        g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x))
        _var_l = list(g)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('true', 0),
                ('str', 0),
                ('true', 1),
                ('str', 1),
                ('true', 2),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_genexp_if_boolop(self):
        g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)] if Expr.true(0) and Expr.true(var_x))
        _var_l = list(g)

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('true', 0), ('true', 0), ('str', 0), ('true', 0), ('true', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_genexp_if_ifexp(self):
        g = (
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))
        )
        _var_l = list(g)

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('true', 0), ('true', 0), ('str', 0), ('true', 0), ('true', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_genexp_target_tuple(self):
        g = (Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))])
        _var_l = list(g)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_genexp_target_list(self):
        g = (Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]])
        _var_l = list(g)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_genexp_target_attr(self):
        var_obj = Expr(0)
        var_obj.val = None
        g = (Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)])
        _var_l = list(g)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setattr', 0, 'val', None),
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

    def test_genexp_target_subscript(self):
        var_arr = Expr(0)
        var_arr[0] = None
        g = (Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)])
        _var_l = list(g)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('setitem', 0, 0, None),
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
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {0: 1}, {})})

    def test_genexp_two_fors(self):
        g = (Expr.str(var_x)
             for var_x in [Expr.int(0), Expr.int(1)]
             for var_y in [Expr.int(2), Expr.int(3)])
        _var_l = list(g)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('str', 0),
                ('str', 0),
                ('int', 2),
                ('int', 3),
                ('str', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_genexp_three_fors(self):
        g = (
            Expr.str(var_x)
            for var_x in [Expr.int(0), Expr.int(1)]
            for var_y in [Expr.int(0), Expr.int(1)]
            for var_z in [Expr.int(0), Expr.int(1)]
        )
        _var_l = list(g)

        self.assertEqual(
            Expr.trace,
            [
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('str', 0),
                ('str', 0),
                ('int', 0),
                ('int', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 1),
                ('int', 0),
                ('int', 1),
                ('str', 1),
                ('str', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {})


class YieldTest(PILTestCase):

    def test_yield_bare(self):
        def gen():
            yield

        _var_l = list(gen())

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_yield_name(self):
        def gen():
            var_x = Expr.int(0)
            yield var_x

        var_l = list(gen())
        Expr.str(var_l[0])

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0]})

    def test_yield_call_expr(self):
        def gen():
            yield Expr.int(0)

        var_l = list(gen())
        Expr.str(var_l[0])

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0]})

    def test_yield_constant(self):
        def gen():
            yield 42

        var_l = list(gen())
        Expr.str(var_l[0])

        self.assertEqual(Expr.trace, [('str', 42)])
        self.assertEqual(self.get_var(locals()), {'var_l': [42]})

    def test_yield_binop(self):
        def gen():
            yield Expr.int(0) + Expr.int(1)

        var_l = list(gen())
        Expr.str(var_l[0])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_l': [1]})

    def test_yield_attr(self):
        def gen():
            var_obj = Expr(0)
            var_obj.val = Expr.int(99)
            yield var_obj.val

        var_l = list(gen())
        Expr.str(var_l[0])

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setattr', 0, 'val', 99), ('getattr', 0, 'val'), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_l': [99]})

    def test_yield_subscript(self):
        def gen():
            var_arr = [Expr.int(0), Expr.int(1)]
            yield var_arr[0]

        var_l = list(gen())
        Expr.str(var_l[0])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0]})

    def test_yield_multiple(self):
        def gen():
            yield Expr.int(0)
            yield Expr.int(1)
            yield Expr.int(2)

        var_l = list(gen())
        Expr.str(var_l[0])
        Expr.str(var_l[1])
        Expr.str(var_l[2])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2]})

    def test_yield_in_if(self):
        def gen(flag):
            if flag:
                yield Expr.int(0)
            else:
                yield Expr.int(1)

        var_l0 = list(gen(Expr.true(0)))
        Expr.str(var_l0[0])
        var_l1 = list(gen(Expr.false(1)))
        Expr.str(var_l1[0])

        self.assertEqual(Expr.trace, [('true', 0), ('int', 0), ('str', 0), ('false', 1), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_l0': [0], 'var_l1': [1]})

    def test_yield_in_for(self):
        def gen():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                yield var_x

        var_l = list(gen())
        Expr.str(var_l[0])
        Expr.str(var_l[1])
        Expr.str(var_l[2])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2]})

    def test_yield_from_name(self):
        def inner():
            yield Expr.int(0)
            yield Expr.int(1)

        def gen():
            var_g = inner()
            yield from var_g

        var_l = list(gen())
        Expr.str(var_l[0])
        Expr.str(var_l[1])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1]})

    def test_yield_from_call(self):
        def inner():
            yield Expr.int(0)
            yield Expr.int(1)

        def gen():
            yield from inner()

        var_l = list(gen())
        Expr.str(var_l[0])
        Expr.str(var_l[1])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1]})

    def test_yield_send_value(self):
        def gen():
            var_sent = yield Expr.int(0)
            Expr.str(var_sent)

        g = gen()
        next(g)
        try:
            g.send(Expr.int(1))
        except StopIteration:
            pass

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})
