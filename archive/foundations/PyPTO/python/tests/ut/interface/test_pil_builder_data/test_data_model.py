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


class ConstantTest(PILTestCase):

    def test_const_binop_left(self):
        var_x = 2 + Expr.int(0)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_x': 2})

    def test_const_binop_right(self):
        var_x = Expr.int(0) + 3
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {'var_x': 3})

    def test_const_binop_both(self):
        var_x = 2 + 3
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('str', 5)])
        self.assertEqual(self.get_var(locals()), {'var_x': 5})

    def test_const_unary_neg(self):
        var_x = -1
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('str', -1)])
        self.assertEqual(self.get_var(locals()), {'var_x': -1})

    def test_const_unary_not(self):
        var_x = not False
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_x': True})

    def test_const_if_true(self):
        if 1:
            Expr.str(0)
        else:
            Expr.str(1)

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_const_if_false(self):
        if 0:
            Expr.str(0)
        else:
            Expr.str(1)

        self.assertEqual(Expr.trace, [('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_const_for_iter(self):
        for var_x in (0, 1, 2):
            Expr.str(var_x)

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_x': 2})

    def test_const_while_test(self):
        while 1:
            Expr.str(0)
            break

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_const_call_pos_arg(self):
        def func(x):
            Expr.str(x)

        func(42)

        self.assertEqual(Expr.trace, [('str', 42)])
        self.assertEqual(self.get_var(locals()), {})

    def test_const_call_kw_arg(self):
        def func(x):
            Expr.str(x)

        func(x=42)

        self.assertEqual(Expr.trace, [('str', 42)])
        self.assertEqual(self.get_var(locals()), {})

    def test_const_in_tuple(self):
        var_t = (0, Expr.int(1), 2)
        Expr.str(var_t[0])
        Expr.str(var_t[1])
        Expr.str(var_t[2])

        self.assertEqual(Expr.trace, [('int', 1), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_t': (0, 1, 2)})

    def test_const_in_list(self):
        var_l = [0, Expr.int(1), 2]
        Expr.str(var_l[0])
        Expr.str(var_l[1])
        Expr.str(var_l[2])

        self.assertEqual(Expr.trace, [('int', 1), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2]})

    def test_const_dict_key(self):
        var_d = {0: Expr.int(1)}
        Expr.str(var_d[0])

        self.assertEqual(Expr.trace, [('int', 1), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_d': {0: 1}})

    def test_const_dict_value(self):
        var_d = {Expr.int(0): 99}
        Expr.str(var_d[0])

        self.assertEqual(Expr.trace, [('int', 0), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_d': {0: 99}})

    def test_const_dict_both(self):
        var_d = {0: 99}
        Expr.str(var_d[0])

        self.assertEqual(Expr.trace, [('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_d': {0: 99}})

    def test_const_in_set(self):
        var_s = {0, Expr.int(1)}
        Expr.str(0 in var_s)

        self.assertEqual(Expr.trace, [('int', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_s': {0, 1}})

    def test_const_subscript_index(self):
        var_obj = Expr(0)
        var_obj[0] = Expr.int(1)
        var_x = var_obj[0]
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 1), ('setitem', 0, 0, 1), ('getitem', 0, 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {}), 'var_x': 1})

    def test_const_slice_bounds(self):
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_s = var_l[1:3]
        Expr.str(var_s[0])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2, 3], 'var_s': [1, 2]})

    def test_const_annotation_no_value(self):
        _var_x: int

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_const_annotation_with_value(self):
        var_x: int = Expr.int(0)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_x': 0})


class JoinedStrTest(PILTestCase):

    def test_fstr_single_expr(self):
        _var_x = f"{Expr.str(0)}"

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_prefix_and_expr(self):
        _var_x = f"prefix_{Expr.str(0)}"

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_expr_and_suffix(self):
        _var_x = f"{Expr.str(0)}_suffix"

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_prefix_expr_suffix(self):
        _var_x = f"prefix_{Expr.str(0)}_suffix"

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_two_exprs(self):
        _var_x = f"{Expr.str(0)}{Expr.str(1)}"

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_expr_sep_expr(self):
        _var_x = f"{Expr.str(0)}_{Expr.str(1)}"

        self.assertEqual(Expr.trace, [('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_conversion_s(self):
        _var_x = f"{Expr.int(0)!s}"

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_conversion_r(self):
        _var_x = f"{Expr.int(0)!r}"

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_conversion_a(self):
        _var_x = f"{Expr.int(0)!a}"

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_format_spec_const(self):
        _var_x = f"{Expr.str(0):>10}"

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_format_spec_expr(self):
        var_fmt = '>10'
        _var_x = f"{Expr.str(0):{var_fmt}}"

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_fmt': '>10'})

    def test_fstr_conversion_and_format_spec(self):
        _var_x = f"{Expr.int(0)!r:>10}"

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_fstr_nested_call_expr(self):
        _var_x = f"{Expr.str(Expr.int(0))}"

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {})


class DictTest(PILTestCase):

    def test_dict_empty(self):
        _var_d = {}

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_const_key_call_value(self):
        _var_d = {0: Expr.int(1)}

        self.assertEqual(Expr.trace, [('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_call_key_call_value(self):
        _var_d = {Expr.int(0): Expr.int(1)}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_multiple_pairs(self):
        _var_d = {Expr.int(0): Expr.int(1), Expr.int(2): Expr.int(3)}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_three_pairs(self):
        _var_d = {Expr.int(0): Expr.int(1), Expr.int(2): Expr.int(3), Expr.int(4): Expr.int(5)}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('int', 4), ('int', 5)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_spread_only(self):
        var_other = {0: Expr.int(0)}
        _var_d = {**var_other}

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {'var_other': {0: 0}})

    def test_dict_normal_then_spread(self):
        var_other = {2: Expr.int(2)}
        _var_d = {Expr.int(0): Expr.int(1), **var_other}

        self.assertEqual(Expr.trace, [('int', 2), ('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_other': {2: 2}})

    def test_dict_spread_in_middle(self):
        var_other = {1: Expr.int(2)}
        _var_d = {Expr.int(0): Expr.int(0), **var_other, Expr.int(3): Expr.int(4)}

        self.assertEqual(Expr.trace, [('int', 2), ('int', 0), ('int', 0), ('int', 3), ('int', 4)])
        self.assertEqual(self.get_var(locals()), {'var_other': {1: 2}})

    def test_dict_multiple_spreads(self):
        var_a = {0: Expr.int(0)}
        var_b = {1: Expr.int(1)}
        _var_d = {**var_a, **var_b}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_a': {0: 0}, 'var_b': {1: 1}})

    def test_dict_spread_from_func_call(self):
        def make():
            return {Expr.int(0): Expr.int(1)}

        _var_d = {**make()}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_normal_then_spread_from_call(self):
        def make():
            return {Expr.int(2): Expr.int(3)}

        _var_d = {Expr.int(0): Expr.int(1), **make()}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_spread_from_call_then_normal(self):
        def make():
            return {Expr.int(0): Expr.int(1)}

        _var_d = {**make(), Expr.int(2): Expr.int(3)}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_multiple_spreads_from_calls(self):
        def make_a():
            return {Expr.int(0): Expr.int(1)}

        def make_b():
            return {Expr.int(2): Expr.int(3)}

        _var_d = {**make_a(), **make_b()}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_dict_nested_value(self):
        _var_d = {Expr.int(0): {Expr.int(1): Expr.int(2)}}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})


class SetTest(PILTestCase):

    def test_set_single_const(self):
        _var_s = {0}

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_set_single_call(self):
        _var_s = {Expr.int(0)}

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_set_multiple_calls(self):
        _var_s = {Expr.int(0), Expr.int(1), Expr.int(2)}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_set_spread_only(self):
        var_other = [Expr.int(0), Expr.int(1)]
        _var_s = {*var_other}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_other': [0, 1]})

    def test_set_spread_from_func_call(self):
        def make():
            return [Expr.int(0), Expr.int(1)]

        _var_s = {*make()}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_set_normal_then_spread(self):
        def make():
            return [Expr.int(2), Expr.int(3)]

        _var_s = {Expr.int(0), Expr.int(1), *make()}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_set_spread_then_normal(self):
        def make():
            return [Expr.int(0), Expr.int(1)]

        _var_s = {*make(), Expr.int(2), Expr.int(3)}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_set_spread_in_middle(self):
        def make():
            return [Expr.int(1), Expr.int(2)]

        _var_s = {Expr.int(0), *make(), Expr.int(3)}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_set_multiple_spreads_from_calls(self):
        def make_a():
            return [Expr.int(0), Expr.int(1)]

        def make_b():
            return [Expr.int(2), Expr.int(3)]

        _var_s = {*make_a(), *make_b()}

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})


class ListTest(PILTestCase):

    def test_list_empty(self):
        _var_l = []

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_single_const(self):
        _var_l = [0]

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_single_call(self):
        _var_l = [Expr.int(0)]

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_multiple_calls(self):
        _var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_starred_only(self):
        var_other = [Expr.int(0), Expr.int(1)]
        _var_l = [*var_other]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_other': [0, 1]})

    def test_list_starred_from_call(self):
        def make():
            return [Expr.int(0), Expr.int(1)]

        _var_l = [*make()]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_normal_then_starred(self):
        def make():
            return [Expr.int(2), Expr.int(3)]

        _var_l = [Expr.int(0), Expr.int(1), *make()]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_starred_then_normal(self):
        def make():
            return [Expr.int(0), Expr.int(1)]

        _var_l = [*make(), Expr.int(2), Expr.int(3)]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_starred_in_middle(self):
        def make():
            return [Expr.int(1), Expr.int(2)]

        _var_l = [Expr.int(0), *make(), Expr.int(3)]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_multiple_starred_from_calls(self):
        def make_a():
            return [Expr.int(0), Expr.int(1)]

        def make_b():
            return [Expr.int(2), Expr.int(3)]

        _var_l = [*make_a(), *make_b()]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_list_nested(self):
        _var_l = [Expr.int(0), [Expr.int(1), Expr.int(2)]]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})


class TupleTest(PILTestCase):

    def test_tuple_single_const(self):
        _var_t = (0,)

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_single_call(self):
        _var_t = (Expr.int(0),)

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_multiple_calls(self):
        _var_t = (Expr.int(0), Expr.int(1), Expr.int(2))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_starred_only(self):
        var_other = [Expr.int(0), Expr.int(1)]
        _var_t = (*var_other,)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_other': [0, 1]})

    def test_tuple_starred_from_call(self):
        def make():
            return [Expr.int(0), Expr.int(1)]

        _var_t = (*make(),)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_normal_then_starred(self):
        def make():
            return [Expr.int(2), Expr.int(3)]

        _var_t = (Expr.int(0), Expr.int(1), *make())

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_starred_then_normal(self):
        def make():
            return [Expr.int(0), Expr.int(1)]

        _var_t = (*make(), Expr.int(2), Expr.int(3))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_starred_in_middle(self):
        def make():
            return [Expr.int(1), Expr.int(2)]

        _var_t = (Expr.int(0), *make(), Expr.int(3))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_multiple_starred_from_calls(self):
        def make_a():
            return [Expr.int(0), Expr.int(1)]

        def make_b():
            return [Expr.int(2), Expr.int(3)]

        _var_t = (*make_a(), *make_b())

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_tuple_nested(self):
        _var_t = (Expr.int(0), (Expr.int(1), Expr.int(2)))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})


class AttributeTest(PILTestCase):

    def test_attr_binop_left(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(2)
        var_x = var_obj.val + Expr.int(1)
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 2), ('setattr', 0, 'val', 2), ('getattr', 0, 'val'), ('int', 1), ('str', 3)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 2}), 'var_x': 3})

    def test_attr_binop_right(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(3)
        var_x = Expr.int(1) + var_obj.val
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 3), ('setattr', 0, 'val', 3), ('int', 1), ('getattr', 0, 'val'), ('str', 4)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 3}), 'var_x': 4})

    def test_attr_binop_both(self):
        var_a = Expr(0)
        var_a.val = Expr.int(2)
        var_b = Expr(1)
        var_b.val = Expr.int(3)
        var_x = var_a.val + var_b.val
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 2),
                ('setattr', 0, 'val', 2),
                ('init', 1),
                ('int', 3),
                ('setattr', 1, 'val', 3),
                ('getattr', 0, 'val'),
                ('getattr', 1, 'val'),
                ('str', 5),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_a': Expr(0, {}, {'val': 2}), 'var_b': Expr(1, {}, {'val': 3}), 'var_x': 5},
        )

    def test_attr_unary_neg(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(5)
        var_x = -var_obj.val
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 5), ('setattr', 0, 'val', 5), ('getattr', 0, 'val'), ('str', -5)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 5}), 'var_x': -5})

    def test_attr_unary_not(self):
        var_obj = Expr(0)
        var_obj.val = Expr.true(0)
        var_x = not var_obj.val
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('true', 0), ('setattr', 0, 'val', True), ('getattr', 0, 'val'), ('str', False)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': True}), 'var_x': False})

    def test_attr_if_true(self):
        var_obj = Expr(0)
        var_obj.val = Expr.true(0)
        if var_obj.val:
            Expr.str(1)
        else:
            Expr.str(2)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('true', 0), ('setattr', 0, 'val', True), ('getattr', 0, 'val'), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': True})})

    def test_attr_if_false(self):
        var_obj = Expr(0)
        var_obj.val = Expr.false(0)
        if var_obj.val:
            Expr.str(1)
        else:
            Expr.str(2)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('false', 0), ('setattr', 0, 'val', False), ('getattr', 0, 'val'), ('str', 2)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': False})})

    def test_attr_for_iter(self):
        var_obj = Expr(0)
        var_obj.val = [Expr.int(0), Expr.int(1), Expr.int(2)]
        for var_x in var_obj.val:
            Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('setattr', 0, 'val', [0, 1, 2]),
                ('getattr', 0, 'val'),
                ('str', 0),
                ('str', 1),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': [0, 1, 2]}), 'var_x': 2})

    def test_attr_while_test(self):
        var_obj = Expr(0)
        var_obj.val = Expr.true(0)
        while var_obj.val:
            Expr.str(1)
            var_obj.val = False

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('true', 0),
                ('setattr', 0, 'val', True),
                ('getattr', 0, 'val'),
                ('str', 1),
                ('setattr', 0, 'val', False),
                ('getattr', 0, 'val'),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': False})})

    def test_attr_call_pos_arg(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(1)
        Expr.str(var_obj.val)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 1), ('setattr', 0, 'val', 1), ('getattr', 0, 'val'), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_attr_call_kw_arg(self):
        def func(x):
            Expr.str(x)

        var_obj = Expr(0)
        var_obj.val = Expr.int(1)
        func(x=var_obj.val)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 1), ('setattr', 0, 'val', 1), ('getattr', 0, 'val'), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_attr_in_tuple(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(1)
        var_t = (var_obj.val, Expr.int(2))
        Expr.str(var_t[0])
        Expr.str(var_t[1])

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('getattr', 0, 'val'),
                ('int', 2),
                ('str', 1),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1}), 'var_t': (1, 2)})

    def test_attr_in_list(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(1)
        var_l = [var_obj.val, Expr.int(2)]
        Expr.str(var_l[0])
        Expr.str(var_l[1])

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('getattr', 0, 'val'),
                ('int', 2),
                ('str', 1),
                ('str', 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1}), 'var_l': [1, 2]})

    def test_attr_dict_key(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        var_d = {var_obj.val: Expr.int(1)}
        Expr.str(var_d[0])

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 0), ('setattr', 0, 'val', 0), ('getattr', 0, 'val'), ('int', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 0}), 'var_d': {0: 1}})

    def test_attr_dict_value(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(99)
        var_d = {Expr.int(0): var_obj.val}
        Expr.str(var_d[0])

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setattr', 0, 'val', 99), ('int', 0), ('getattr', 0, 'val'), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 99}), 'var_d': {0: 99}})

    def test_attr_in_set(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(1)
        var_s = {var_obj.val, Expr.int(2)}
        Expr.str(1 in var_s)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 1), ('setattr', 0, 'val', 1), ('getattr', 0, 'val'), ('int', 2), ('str', True)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1}), 'var_s': {1, 2}})

    def test_attr_subscript_index(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        var_arr = Expr(1)
        var_arr[0] = Expr.int(99)
        var_x = var_arr[var_obj.val]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 0),
                ('setattr', 0, 'val', 0),
                ('init', 1),
                ('int', 99),
                ('setitem', 1, 0, 99),
                ('getattr', 0, 'val'),
                ('getitem', 1, 0),
                ('str', 99),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(0, {}, {'val': 0}), 'var_arr': Expr(1, {0: 99}, {}), 'var_x': 99},
        )

    def test_attr_slice_lower(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(1)
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
        var_s = var_l[var_obj.val:3]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('getattr', 0, 'val'),
                ('str', 1),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(0, {}, {'val': 1}), 'var_l': [0, 1, 2], 'var_s': [1, 2]},
        )

    def test_attr_slice_upper(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(2)
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
        var_s = var_l[0:var_obj.val]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 2),
                ('setattr', 0, 'val', 2),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('getattr', 0, 'val'),
                ('str', 0),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(0, {}, {'val': 2}), 'var_l': [0, 1, 2], 'var_s': [0, 1]},
        )

    def test_attr_slice_both(self):
        var_lo = Expr(0)
        var_lo.val = Expr.int(1)
        var_hi = Expr(1)
        var_hi.val = Expr.int(3)
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_s = var_l[var_lo.val:var_hi.val]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('init', 1),
                ('int', 3),
                ('setattr', 1, 'val', 3),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('getattr', 0, 'val'),
                ('getattr', 1, 'val'),
                ('str', 1),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {
                'var_lo': Expr(0, {}, {'val': 1}),
                'var_hi': Expr(1, {}, {'val': 3}),
                'var_l': [0, 1, 2, 3],
                'var_s': [1, 2],
            },
        )

    def test_attr_annotation_no_value(self):
        _var_x: Expr.ContextManager

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_attr_annotation_with_value(self):
        var_x: Expr.ContextManager = Expr.int(0)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_x': 0})


class SubscriptTest(PILTestCase):

    def test_subscript_binop_left(self):
        var_l = [Expr.int(2), Expr.int(3)]
        var_x = var_l[0] + Expr.int(1)
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 2), ('int', 3), ('int', 1), ('str', 3)])
        self.assertEqual(self.get_var(locals()), {'var_l': [2, 3], 'var_x': 3})

    def test_subscript_binop_right(self):
        var_l = [Expr.int(3), Expr.int(4)]
        var_x = Expr.int(1) + var_l[0]
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 3), ('int', 4), ('int', 1), ('str', 4)])
        self.assertEqual(self.get_var(locals()), {'var_l': [3, 4], 'var_x': 4})

    def test_subscript_binop_both(self):
        var_l = [Expr.int(2), Expr.int(3)]
        var_x = var_l[0] + var_l[1]
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 2), ('int', 3), ('str', 5)])
        self.assertEqual(self.get_var(locals()), {'var_l': [2, 3], 'var_x': 5})

    def test_subscript_unary_neg(self):
        var_l = [Expr.int(5)]
        var_x = -var_l[0]
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 5), ('str', -5)])
        self.assertEqual(self.get_var(locals()), {'var_l': [5], 'var_x': -5})

    def test_subscript_unary_not(self):
        var_l = [Expr.true(0)]
        var_x = not var_l[0]
        Expr.str(var_x)

        self.assertEqual(Expr.trace, [('true', 0), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_l': [True], 'var_x': False})

    def test_subscript_if_true(self):
        var_l = [Expr.true(0)]
        if var_l[0]:
            Expr.str(1)
        else:
            Expr.str(2)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_l': [True]})

    def test_subscript_if_false(self):
        var_l = [Expr.false(0)]
        if var_l[0]:
            Expr.str(1)
        else:
            Expr.str(2)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_l': [False]})

    def test_subscript_for_iter(self):
        var_l = [[Expr.int(0), Expr.int(1), Expr.int(2)]]
        for var_x in var_l[0]:
            Expr.str(var_x)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_l': [[0, 1, 2]], 'var_x': 2})

    def test_subscript_while_test(self):
        var_l = [True]
        while var_l[0]:
            Expr.str(0)
            var_l[0] = False

        self.assertEqual(Expr.trace, [('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_l': [False]})

    def test_subscript_call_pos_arg(self):
        var_l = [Expr.int(0)]
        Expr.str(var_l[0])

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0]})

    def test_subscript_call_kw_arg(self):
        def func(x):
            Expr.str(x)

        var_l = [Expr.int(0)]
        func(x=var_l[0])

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_l': [0]})

    def test_subscript_in_tuple(self):
        var_l = [Expr.int(1), Expr.int(2)]
        var_t = (var_l[0], var_l[1])
        Expr.str(var_t[0])
        Expr.str(var_t[1])

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_l': [1, 2], 'var_t': (1, 2)})

    def test_subscript_in_list(self):
        var_l = [Expr.int(1), Expr.int(2)]
        var_r = [var_l[0], var_l[1]]
        Expr.str(var_r[0])
        Expr.str(var_r[1])

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_l': [1, 2], 'var_r': [1, 2]})

    def test_subscript_dict_key(self):
        var_keys = [Expr.int(0)]
        var_d = {var_keys[0]: Expr.int(99)}
        Expr.str(var_d[0])

        self.assertEqual(Expr.trace, [('int', 0), ('int', 99), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_keys': [0], 'var_d': {0: 99}})

    def test_subscript_dict_value(self):
        var_vals = [Expr.int(99)]
        var_d = {Expr.int(0): var_vals[0]}
        Expr.str(var_d[0])

        self.assertEqual(Expr.trace, [('int', 99), ('int', 0), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_vals': [99], 'var_d': {0: 99}})

    def test_subscript_in_set(self):
        var_l = [Expr.int(1), Expr.int(2)]
        var_s = {var_l[0], var_l[1]}
        Expr.str(1 in var_s)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_l': [1, 2], 'var_s': {1, 2}})

    def test_subscript_as_index(self):
        var_idx = [Expr.int(0)]
        var_arr = Expr(0)
        var_arr[0] = Expr.int(99)
        var_x = var_arr[var_idx[0]]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('init', 0), ('int', 99), ('setitem', 0, 0, 99), ('getitem', 0, 0), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_idx': [0], 'var_arr': Expr(0, {0: 99}, {}), 'var_x': 99})

    def test_subscript_annotation_no_value(self):
        _var_x: list[int]

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_subscript_annotation_with_value(self):
        var_x: list[int] = [Expr.int(0)]
        Expr.str(var_x[0])

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_x': [0]})

    def test_slice_index_attr(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(1)
        var_arr = Expr(1)
        var_arr[1] = Expr.int(99)
        var_x = var_arr[var_obj.val]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('init', 1),
                ('int', 99),
                ('setitem', 1, 1, 99),
                ('getattr', 0, 'val'),
                ('getitem', 1, 1),
                ('str', 99),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(0, {}, {'val': 1}), 'var_arr': Expr(1, {1: 99}, {}), 'var_x': 99},
        )

    def test_slice_index_call(self):
        def idx():
            return Expr.int(0)

        var_arr = Expr(0)
        var_arr[0] = Expr.int(99)
        var_x = var_arr[idx()]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setitem', 0, 0, 99), ('int', 0), ('getitem', 0, 0), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {0: 99}, {}), 'var_x': 99})

    def test_slice_index_binop(self):
        var_arr = Expr(0)
        var_arr[2] = Expr.int(99)
        var_x = var_arr[Expr.int(1) + Expr.int(1)]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setitem', 0, 2, 99), ('int', 1), ('int', 1), ('getitem', 0, 2), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {2: 99}, {}), 'var_x': 99})

    def test_slice_index_unary(self):
        var_arr = Expr(0)
        var_arr[-1] = Expr.int(99)
        var_x = var_arr[-Expr.int(1)]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setitem', 0, -1, 99), ('int', 1), ('getitem', 0, -1), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {-1: 99}, {}), 'var_x': 99})

    def test_slice_index_subscript(self):
        var_idxs = [Expr.int(0)]
        var_arr = Expr(0)
        var_arr[0] = Expr.int(99)
        var_x = var_arr[var_idxs[0]]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('init', 0), ('int', 99), ('setitem', 0, 0, 99), ('getitem', 0, 0), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_idxs': [0], 'var_arr': Expr(0, {0: 99}, {}), 'var_x': 99})

    def test_slice_index_dict(self):
        var_d = {0: Expr.int(1)}
        var_arr = Expr(0)
        var_arr[1] = Expr.int(99)
        var_x = var_arr[var_d[0]]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('int', 1), ('init', 0), ('int', 99), ('setitem', 0, 1, 99), ('getitem', 0, 1), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_d': {0: 1}, 'var_arr': Expr(0, {1: 99}, {}), 'var_x': 99})

    def test_slice_index_set(self):
        var_s = {0}
        var_arr = Expr(0)
        var_arr[True] = Expr.int(99)
        var_x = var_arr[0 in var_s]
        Expr.str(var_x)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setitem', 0, True, 99), ('getitem', 0, True), ('str', 99)],
        )
        self.assertEqual(self.get_var(locals()), {'var_s': {0}, 'var_arr': Expr(0, {True: 99}, {}), 'var_x': 99})

    def test_slice_index_named_expr(self):
        var_arr = Expr(0)
        var_arr[0] = Expr.int(99)
        var_x = var_arr[(var_k := Expr.int(0))]
        Expr.str(var_x)
        Expr.str(var_k)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setitem', 0, 0, 99), ('int', 0), ('getitem', 0, 0), ('str', 99), ('str', 0)],
        )
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {0: 99}, {}), 'var_k': 0, 'var_x': 99})

    def test_slice_range_attr_bounds(self):
        var_lo = Expr(0)
        var_lo.val = Expr.int(1)
        var_hi = Expr(1)
        var_hi.val = Expr.int(3)
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_s = var_l[var_lo.val:var_hi.val]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('init', 1),
                ('int', 3),
                ('setattr', 1, 'val', 3),
                ('int', 0),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('getattr', 0, 'val'),
                ('getattr', 1, 'val'),
                ('str', 1),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {
                'var_lo': Expr(0, {}, {'val': 1}),
                'var_hi': Expr(1, {}, {'val': 3}),
                'var_l': [0, 1, 2, 3],
                'var_s': [1, 2],
            },
        )

    def test_slice_range_call_bounds(self):
        def lo():
            return Expr.int(1)

        def hi():
            return Expr.int(3)

        var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_s = var_l[lo():hi()]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('int', 1), ('int', 3), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2, 3], 'var_s': [1, 2]})

    def test_slice_range_binop_bounds(self):
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_s = var_l[Expr.int(0) + 1:Expr.int(1) + 2]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('int', 0), ('int', 1), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2, 3], 'var_s': [1, 2]})

    def test_slice_range_subscript_bounds(self):
        var_bounds = [Expr.int(1), Expr.int(3)]
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_s = var_l[var_bounds[0]:var_bounds[1]]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [('int', 1), ('int', 3), ('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_bounds': [1, 3], 'var_l': [0, 1, 2, 3], 'var_s': [1, 2]})

    def test_slice_range_dict_bounds(self):
        var_d = {'lo': Expr.int(1), 'hi': Expr.int(3)}
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_s = var_l[var_d['lo']:var_d['hi']]
        Expr.str(var_s[0])

        self.assertEqual(
            Expr.trace,
            [('int', 1), ('int', 3), ('int', 0), ('int', 1), ('int', 2), ('int', 3), ('str', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_d': {'lo': 1, 'hi': 3}, 'var_l': [0, 1, 2, 3], 'var_s': [1, 2]})
