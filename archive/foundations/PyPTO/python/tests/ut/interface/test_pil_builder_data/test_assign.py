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


class ExprTest(PILTestCase):

    def test_expr_binop(self):
        Expr.int(0) + Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_expr_named_expr(self):
        (var_a := Expr.int(0))
        Expr.str(var_a)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0})


class NamedExprTest(PILTestCase):

    def test_named_expr_binop(self):
        var_x = (var_a := Expr.int(0)) + (var_b := Expr.int(1))
        Expr.str(var_x)
        Expr.str(var_a)
        Expr.str(var_b)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_x': 1})

    def test_named_expr_unary(self):
        var_x = -(var_a := Expr.int(5))
        Expr.str(var_x)
        Expr.str(var_a)

        self.assertEqual(Expr.trace, [('int', 5), ('str', -5), ('str', 5)])
        self.assertEqual(self.get_var(locals()), {'var_a': 5, 'var_x': -5})

    def test_named_expr_if_true(self):
        if var_a := Expr.true(0):
            Expr.str(1)
        else:
            Expr.str(2)
        Expr.str(var_a)

        self.assertEqual(Expr.trace, [('true', 0), ('str', 1), ('str', True)])
        self.assertEqual(self.get_var(locals()), {'var_a': True})

    def test_named_expr_if_false(self):
        if var_a := Expr.false(0):
            Expr.str(1)
        else:
            Expr.str(2)
        Expr.str(var_a)

        self.assertEqual(Expr.trace, [('false', 0), ('str', 2), ('str', False)])
        self.assertEqual(self.get_var(locals()), {'var_a': False})

    def test_named_expr_for_iter(self):
        for var_x in (var_it := [Expr.int(0), Expr.int(1)]):
            Expr.str(var_x)
        Expr.str(len(var_it))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_it': [0, 1], 'var_x': 1})

    def test_named_expr_while(self):
        var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
        var_i = [0]
        while var_n := (var_i[0] < len(var_items)):
            Expr.str(var_items[var_i[0]])
            var_i[0] = var_i[0] + 1
        Expr.str(var_n)

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('str', 0), ('str', 1), ('str', 2), ('str', False)],
        )
        self.assertEqual(self.get_var(locals()), {'var_items': [0, 1, 2], 'var_i': [3], 'var_n': False})

    def test_named_expr_call_pos_arg(self):
        Expr.str(var_a := Expr.int(0))
        Expr.str(var_a)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0})

    def test_named_expr_call_kw_arg(self):
        def func(x):
            Expr.str(x)

        func(x=(var_a := Expr.int(0)))
        Expr.str(var_a)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0})

    def test_named_expr_in_tuple(self):
        var_t = ((var_a := Expr.int(0)), (var_b := Expr.int(1)))
        Expr.str(var_t[0])
        Expr.str(var_t[1])
        Expr.str(var_a)
        Expr.str(var_b)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_t': (0, 1)})

    def test_named_expr_in_list(self):
        var_l = [(var_a := Expr.int(0)), (var_b := Expr.int(1))]
        Expr.str(var_l[0])
        Expr.str(var_l[1])
        Expr.str(var_a)
        Expr.str(var_b)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 0), ('str', 1), ('str', 0), ('str', 1)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_l': [0, 1]})

    def test_named_expr_dict_key(self):
        var_d = {(var_k := Expr.int(0)): Expr.int(1)}
        Expr.str(var_d[0])
        Expr.str(var_k)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('str', 1), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_k': 0, 'var_d': {0: 1}})

    def test_named_expr_dict_value(self):
        var_d = {Expr.int(0): (var_v := Expr.int(99))}
        Expr.str(var_d[0])
        Expr.str(var_v)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 99), ('str', 99), ('str', 99)])
        self.assertEqual(self.get_var(locals()), {'var_v': 99, 'var_d': {0: 99}})

    def test_named_expr_in_set(self):
        var_s = {(var_a := Expr.int(1)), (var_b := Expr.int(2))}
        Expr.str(1 in var_s)
        Expr.str(var_a)
        Expr.str(var_b)

        self.assertEqual(Expr.trace, [('int', 1), ('int', 2), ('str', True), ('str', 1), ('str', 2)])
        self.assertEqual(self.get_var(locals()), {'var_a': 1, 'var_b': 2, 'var_s': {1, 2}})

    def test_named_expr_subscript_index(self):
        var_arr = Expr(0)
        var_arr[0] = Expr.int(99)
        var_x = var_arr[(var_i := Expr.int(0))]
        Expr.str(var_x)
        Expr.str(var_i)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 99), ('setitem', 0, 0, 99), ('int', 0), ('getitem', 0, 0), ('str', 99), ('str', 0)],
        )
        self.assertEqual(self.get_var(locals()), {'var_arr': Expr(0, {0: 99}, {}), 'var_i': 0, 'var_x': 99})

    def test_named_expr_slice_bound(self):
        var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
        var_s = var_l[(var_lo := Expr.int(1)):(var_hi := Expr.int(3))]
        Expr.str(var_s[0])
        Expr.str(var_lo)
        Expr.str(var_hi)

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('int', 1), ('int', 3), ('str', 1), ('str', 1), ('str', 3)],
        )
        self.assertEqual(self.get_var(locals()), {'var_l': [0, 1, 2], 'var_lo': 1, 'var_hi': 3, 'var_s': [1, 2]})

    def test_named_expr_annotation_value(self):
        var_x: int = (var_a := Expr.int(0))
        Expr.str(var_x)
        Expr.str(var_a)

        self.assertEqual(Expr.trace, [('int', 0), ('str', 0), ('str', 0)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_x': 0})

    def test_named_expr_as_annotation(self):
        _ns = {'Expr': Expr}
        exec(
            '_var_x: (var_ann := Expr.str(0))\nExpr.str(var_ann)',
            _ns,
        )

        self.assertEqual(Expr.trace, [('str', 0), ('str', 'str(0)')])
        self.assertEqual(self.get_var(_ns), {'var_ann': 'str(0)'})

    def test_named_expr_as_annotation_with_value(self):
        _ns = {'Expr': Expr}
        exec(
            'var_x: (var_ann := Expr.str(0)) = Expr.int(1)  # fmt: skip\nExpr.str(var_x)\nExpr.str(var_ann)',
            _ns,
        )

        self.assertEqual(Expr.trace, [('int', 1), ('str', 0), ('str', 1), ('str', 'str(0)')])
        self.assertEqual(self.get_var(_ns), {'var_x': 1, 'var_ann': 'str(0)'})


class AssignTest(PILTestCase):

    def test_assign_name(self):
        _var_x = Expr.int(0)

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_name_rhs_call(self):
        _var_x = Expr.int(0) + Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_multi_target(self):
        # e.g. a = b = expr - both names get the same value
        _var_x = _var_y = Expr.int(0)

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_attr(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 0), ('setattr', 0, 'val', 0)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 0})})

    def test_assign_attr_rhs_call(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(0) + Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 0), ('int', 1), ('setattr', 0, 'val', 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_assign_attr_chain(self):
        # e.g. obj.val.val += rhs - chain of attribute loads
        var_obj = Expr(0)
        var_obj.val = Expr(1)
        var_obj.val.val = Expr.int(0)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('init', 1),
                ('setattr', 0, 'val', Expr(1, {}, {'val': 0})),
                ('int', 0),
                ('getattr', 0, 'val'),
                ('setattr', 1, 'val', 0),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': Expr(1, {}, {'val': 0})})})

    def test_assign_subscript_const_index(self):
        var_obj = Expr(0)
        var_obj[0] = Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 1), ('setitem', 0, 0, 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {})})

    def test_assign_subscript_expr_index(self):
        var_obj = Expr(0)
        var_obj[Expr.int(0)] = Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 1), ('int', 0), ('setitem', 0, 0, 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {})})

    def test_assign_subscript_attr_index(self):
        # e.g. obj[other.val] = rhs - index is an attribute load
        var_obj = Expr(0)
        var_idx = Expr(1)
        var_idx.val = Expr.int(0)
        var_obj[var_idx.val] = Expr.int(1)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('init', 1),
                ('int', 0),
                ('setattr', 1, 'val', 0),
                ('int', 1),
                ('getattr', 1, 'val'),
                ('setitem', 0, 0, 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {}), 'var_idx': Expr(1, {}, {'val': 0})})

    def test_assign_subscript_subscript_index(self):
        # e.g. obj[idx[k]] = rhs - index is itself a subscript
        var_obj = Expr(0)
        var_idx = Expr(1)
        var_idx[0] = Expr.int(0)
        var_obj[var_idx[0]] = Expr.int(1)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('init', 1),
                ('int', 0),
                ('setitem', 1, 0, 0),
                ('int', 1),
                ('getitem', 1, 0),
                ('setitem', 0, 0, 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {}), 'var_idx': Expr(1, {0: 0}, {})})

    def test_assign_subscript_binop_index(self):
        # e.g. obj[a + b] = rhs - index is a binop
        var_obj = Expr(0)
        var_obj[Expr.int(0) + Expr.int(1)] = Expr.int(2)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 2), ('int', 0), ('int', 1), ('setitem', 0, 1, 2)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {1: 2}, {})})

    def test_assign_subscript_slice(self):
        var_obj = Expr(0)
        var_obj[0:2] = Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 1), ('setitem', 0, slice(0, 2, None), 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {('slice', 0, 2, None): 1}, {})})

    def test_assign_subscript_slice_with_step(self):
        var_obj = Expr(0)
        var_obj[0:4:2] = Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 1), ('setitem', 0, slice(0, 4, 2), 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {('slice', 0, 4, 2): 1}, {})})

    def test_assign_subscript_expr_slice(self):
        # slice bounds are side-effectful expressions
        var_obj = Expr(0)
        var_obj[Expr.int(0):Expr.int(1)] = Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 2), ('int', 0), ('int', 1), ('setitem', 0, slice(0, 1, None), 2)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {('slice', 0, 1, None): 2}, {})})

    def test_assign_subscript_attr_slice(self):
        # e.g. obj[a.val:b.val] = rhs - slice bounds are attribute loads
        var_obj = Expr(0)
        var_lo = Expr(1)
        var_lo.val = Expr.int(0)
        var_hi = Expr(2)
        var_hi.val = Expr.int(2)
        var_obj[var_lo.val:var_hi.val] = Expr.int(3)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('init', 1),
                ('int', 0),
                ('setattr', 1, 'val', 0),
                ('init', 2),
                ('int', 2),
                ('setattr', 2, 'val', 2),
                ('int', 3),
                ('getattr', 1, 'val'),
                ('getattr', 2, 'val'),
                ('setitem', 0, slice(0, 2, None), 3),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {
                'var_obj': Expr(0, {('slice', 0, 2, None): 3}, {}),
                'var_lo': Expr(1, {}, {'val': 0}),
                'var_hi': Expr(2, {}, {'val': 2}),
            },
        )

    def test_assign_subscript_subscript_slice(self):
        # e.g. obj[lo[0]:hi[0]] = rhs - slice bounds are subscripts
        var_obj = Expr(0)
        var_lo = Expr(1)
        var_lo[0] = Expr.int(0)
        var_hi = Expr(2)
        var_hi[0] = Expr.int(2)
        var_obj[var_lo[0]:var_hi[0]] = Expr.int(3)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('init', 1),
                ('int', 0),
                ('setitem', 1, 0, 0),
                ('init', 2),
                ('int', 2),
                ('setitem', 2, 0, 2),
                ('int', 3),
                ('getitem', 1, 0),
                ('getitem', 2, 0),
                ('setitem', 0, slice(0, 2, None), 3),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {
                'var_obj': Expr(0, {('slice', 0, 2, None): 3}, {}),
                'var_lo': Expr(1, {0: 0}, {}),
                'var_hi': Expr(2, {0: 2}, {}),
            },
        )

    def test_assign_subscript_binop_slice(self):
        # e.g. obj[a+1 : b*2] = rhs - slice bounds are binops
        var_obj = Expr(0)
        var_obj[Expr.int(0) + 1:Expr.int(1) * 2] = Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 2), ('int', 0), ('int', 1), ('setitem', 0, slice(1, 2, None), 2)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {('slice', 1, 2, None): 2}, {})})

    def test_assign_attr_subscript(self):
        # e.g. obj.val[k] = rhs - subscript index is an attribute load
        var_obj = Expr(0)
        var_obj.val = Expr(1)
        var_obj.val[0] = Expr.int(1)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('init', 1),
                ('setattr', 0, 'val', Expr(1, {0: 1}, {})),
                ('int', 1),
                ('getattr', 0, 'val'),
                ('setitem', 1, 0, 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': Expr(1, {0: 1}, {})})})

    def test_assign_subscript_attr(self):
        # e.g. obj[k].val = rhs - subscript index is a subscript
        var_obj = Expr(0)
        var_obj[0] = Expr(1)
        var_obj[0].val = Expr.int(1)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('init', 1),
                ('setitem', 0, 0, Expr(1, {}, {'val': 1})),
                ('int', 1),
                ('getitem', 0, 0),
                ('setattr', 1, 'val', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: Expr(1, {}, {'val': 1})}, {})})

    def test_assign_subscript_attr_subscript_attr(self):
        # e.g. obj[k].val[k].val = rhs - assignment through a four-level access chain
        var_obj = Expr(0)
        var_obj[0] = Expr(1)
        var_obj[0].val = Expr(2)
        var_obj[0].val[0] = Expr(3)
        var_obj[0].val[0].val = Expr.int(1)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('init', 1),
                ('setitem', 0, 0, Expr(1, {}, {'val': Expr(2, {0: Expr(3, {}, {'val': 1})}, {})})),
                ('init', 2),
                ('getitem', 0, 0),
                ('setattr', 1, 'val', Expr(2, {0: Expr(3, {}, {'val': 1})}, {})),
                ('init', 3),
                ('getitem', 0, 0),
                ('getattr', 1, 'val'),
                ('setitem', 2, 0, Expr(3, {}, {'val': 1})),
                ('int', 1),
                ('getitem', 0, 0),
                ('getattr', 1, 'val'),
                ('getitem', 2, 0),
                ('setattr', 3, 'val', 1),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(0, {0: Expr(1, {}, {'val': Expr(2, {0: Expr(3, {}, {'val': 1})}, {})})}, {})},
        )

    def test_assign_tuple_unpack(self):
        _var_x, _var_y = Expr.int(0), Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_list_unpack(self):
        [_var_x, _var_y] = [Expr.int(0), Expr.int(1)]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_starred_unpack(self):
        _var_x, *_var_y, _var_z = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_nested_tuple_unpack(self):
        (_var_x, (_var_y, _var_z)) = (Expr.int(0), (Expr.int(1), Expr.int(2)))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_unpack_to_attr_subscript(self):
        # lhs elements can be attribute / subscript targets
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        var_arr = Expr(1)
        var_arr[0] = Expr.int(0)
        var_obj.val, var_arr[0] = Expr.int(1), Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 0),
                ('setattr', 0, 'val', 0),
                ('init', 1),
                ('int', 0),
                ('setitem', 1, 0, 0),
                ('int', 1),
                ('int', 2),
                ('setattr', 0, 'val', 1),
                ('setitem', 1, 0, 2),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1}), 'var_arr': Expr(1, {0: 2}, {})})

    def test_assign_chain_name_name(self):
        # e.g. x = y = expr - both names bound to same value
        _var_x = _var_y = Expr.int(0)

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_chain_name_attr(self):
        # e.g. x = obj.val = expr - name bound to attribute load of object
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        _var_x = var_obj.val = Expr.int(1)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 0), ('setattr', 0, 'val', 0), ('int', 1), ('setattr', 0, 'val', 1)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_assign_chain_name_subscript(self):
        # e.g. x = obj[k] = expr - name bound to subscript of object
        var_obj = Expr(0)
        var_obj[0] = Expr.int(0)
        _var_x = var_obj[0] = Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 0), ('setitem', 0, 0, 0), ('int', 1), ('setitem', 0, 0, 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {})})

    def test_assign_chain_attr_subscript(self):
        # e.g. obj.val = arr[k] = expr - attribute load of object bound to subscript of array target
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        var_arr = Expr(1)
        var_arr[0] = Expr.int(0)
        var_obj.val = var_arr[0] = Expr.int(1)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 0),
                ('setattr', 0, 'val', 0),
                ('init', 1),
                ('int', 0),
                ('setitem', 1, 0, 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('setitem', 1, 0, 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1}), 'var_arr': Expr(1, {0: 1}, {})})

    def test_assign_chain_three(self):
        # e.g. x = obj.val = arr[k] = expr - three targets bound to same value
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        var_arr = Expr(1)
        var_arr[0] = Expr.int(0)
        _var_x = var_obj.val = var_arr[0] = Expr.int(1)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 0),
                ('setattr', 0, 'val', 0),
                ('init', 1),
                ('int', 0),
                ('setitem', 1, 0, 0),
                ('int', 1),
                ('setattr', 0, 'val', 1),
                ('setitem', 1, 0, 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1}), 'var_arr': Expr(1, {0: 1}, {})})

    def test_assign_chain_tuple_name(self):
        # e.g. (a, b) = x = expr - tuple elements bound to names
        _var_x = var_a, var_b = Expr.int(0), Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1})

    def test_assign_chain_tuple_tuple(self):
        # e.g. (a, b) = (c, d) = expr - two tuple lhs targets
        _var_a, _var_b = _var_c, _var_d = Expr.int(0), Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_chain_list_list(self):
        # e.g. [a, b] = [c, d] = expr: list elements bound to names
        [_var_a, _var_b] = [_var_c, _var_d] = [Expr.int(0), Expr.int(1)]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_assign_chain_tuple_nested_2(self):
        # e.g. (a, (b, c)) = x = expr - 2-level nested tuple on first target
        _var_x = var_a, (var_b, var_c) = Expr.int(0), (Expr.int(1), Expr.int(2))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_c': 2})

    def test_assign_chain_tuple_nested_3(self):
        # e.g. x = (a, (b, (c, d))) = expr - 3-level nested tuple
        _var_x = var_a, (var_b, (var_c, var_d)) = Expr.int(0), (Expr.int(1), (Expr.int(2), Expr.int(3)))

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_c': 2, 'var_d': 3})

    def test_assign_chain_list_nested_3(self):
        # e.g. x = [a, [b, [c, d]]] = expr - 3-level nested list
        _var_x = [var_a, [var_b, [var_c, var_d]]] = [Expr.int(0), [Expr.int(1), [Expr.int(2), Expr.int(3)]]]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_c': 2, 'var_d': 3})

    def test_assign_chain_mixed_nested_3(self):
        # e.g. x = (a, [b, (c, d)]) = expr - mixed tuple/list 3-level
        _var_x = var_a, [var_b, (var_c, var_d)] = Expr.int(0), [Expr.int(1), (Expr.int(2), Expr.int(3))]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2), ('int', 3)])
        self.assertEqual(self.get_var(locals()), {'var_a': 0, 'var_b': 1, 'var_c': 2, 'var_d': 3})

    def test_assign_chain_starred_nested(self):
        # e.g. (a, *b, c) = x = expr - chained assignment with a starred unpack target
        _var_x = [var_a, var_b, var_c, var_d] = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
        var_a, *var_rest, var_z = _var_x = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]

        self.assertEqual(
            Expr.trace,
            [('int', 0), ('int', 1), ('int', 2), ('int', 3), ('int', 0), ('int', 1), ('int', 2), ('int', 3)],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_a': 0, 'var_b': 1, 'var_c': 2, 'var_d': 3, 'var_rest': [1, 2], 'var_z': 3},
        )

    def test_assign_chain_three_nested(self):
        # e.g. (a, b) = [c, d] = x = expr - three targets, two of them are nested
        _var_x = [var_c, var_d] = var_a, var_b = [Expr.int(0), Expr.int(1)]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_c': 0, 'var_d': 1, 'var_a': 0, 'var_b': 1})


class AugAssignTest(PILTestCase):

    def test_aug_assign_name(self):
        var_x = Expr.int(0)
        var_x += Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {'var_x': 1})

    def test_aug_assign_name_rhs_call(self):
        var_x = Expr.int(0)
        var_x += Expr.int(1) * Expr.int(2)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1), ('int', 2)])
        self.assertEqual(self.get_var(locals()), {'var_x': 2})

    def test_aug_assign_attr(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        var_obj.val += Expr.int(1)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 0),
                ('setattr', 0, 'val', 0),
                ('getattr', 0, 'val'),
                ('int', 1),
                ('setattr', 0, 'val', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_aug_assign_attr_rhs_call(self):
        var_obj = Expr(0)
        var_obj.val = Expr.int(0)
        var_obj.val += Expr.int(1) + Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 0),
                ('setattr', 0, 'val', 0),
                ('getattr', 0, 'val'),
                ('int', 1),
                ('int', 2),
                ('setattr', 0, 'val', 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 3})})

    def test_aug_assign_subscript(self):
        var_obj = Expr(0)
        var_obj[0] = Expr.int(1)
        var_obj[0] += Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 1), ('setitem', 0, 0, 1), ('getitem', 0, 0), ('int', 2), ('setitem', 0, 0, 3)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 3}, {})})

    def test_aug_assign_subscript_rhs_call(self):
        var_obj = Expr(0)
        var_obj[0] = Expr.int(1)
        var_obj[0] += Expr.int(2) * Expr.int(3)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setitem', 0, 0, 1),
                ('getitem', 0, 0),
                ('int', 2),
                ('int', 3),
                ('setitem', 0, 0, 7),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 7}, {})})

    def test_aug_assign_subscript_expr_index(self):
        var_obj = Expr(0)
        var_obj[Expr.int(0)] = Expr.int(1)
        var_obj[Expr.int(0)] += Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('int', 0),
                ('setitem', 0, 0, 1),
                ('int', 0),
                ('getitem', 0, 0),
                ('int', 2),
                ('setitem', 0, 0, 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 3}, {})})

    def test_aug_assign_nested_attr_subscript(self):
        var_obj = Expr(0)
        var_obj.val = Expr(1)
        var_obj.val[0] = Expr.int(1)
        var_obj.val[0] += Expr.int(2) + Expr.int(3)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('init', 1),
                ('setattr', 0, 'val', Expr(1, {0: 6}, {})),
                ('int', 1),
                ('getattr', 0, 'val'),
                ('setitem', 1, 0, 1),
                ('getattr', 0, 'val'),
                ('getitem', 1, 0),
                ('int', 2),
                ('int', 3),
                ('setitem', 1, 0, 6),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': Expr(1, {0: 6}, {})})})

    def test_aug_assign_subscript_attr_chain(self):
        # e.g. obj[k].val += rhs - subscript then attribute cases bound
        var_obj = Expr(0)
        var_obj[0] = Expr(1)
        var_obj[0].val = Expr.int(1)
        var_obj[0].val += Expr.int(2)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('init', 1),
                ('setitem', 0, 0, Expr(1, {}, {'val': 3})),
                ('int', 1),
                ('getitem', 0, 0),
                ('setattr', 1, 'val', 1),
                ('getitem', 0, 0),
                ('getattr', 1, 'val'),
                ('int', 2),
                ('setattr', 1, 'val', 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: Expr(1, {}, {'val': 3})}, {})})

    def test_aug_assign_attr_subscript_attr_subscript(self):
        # e.g. obj.val[k].val[k] += rhs - assignment through a four-level access chain
        var_obj = Expr(0)
        var_obj.val = Expr(1)
        var_obj.val[0] = Expr(2)
        var_obj.val[0].val = Expr(3)
        var_obj.val[0].val[0] = Expr.int(1)
        var_obj.val[0].val[0] += Expr.int(2)

        self.assertEqual(
            list(Expr.trace),
            [
                ('init', 0),
                ('init', 1),
                ('setattr', 0, 'val', Expr(1, {0: Expr(2, {}, {'val': Expr(3, {0: 3}, {})})}, {})),
                ('init', 2),
                ('getattr', 0, 'val'),
                ('setitem', 1, 0, Expr(2, {}, {'val': Expr(3, {0: 3}, {})})),
                ('init', 3),
                ('getattr', 0, 'val'),
                ('getitem', 1, 0),
                ('setattr', 2, 'val', Expr(3, {0: 3}, {})),
                ('int', 1),
                ('getattr', 0, 'val'),
                ('getitem', 1, 0),
                ('getattr', 2, 'val'),
                ('setitem', 3, 0, 1),
                ('getattr', 0, 'val'),
                ('getitem', 1, 0),
                ('getattr', 2, 'val'),
                ('getitem', 3, 0),
                ('int', 2),
                ('setitem', 3, 0, 3),
            ],
        )
        self.assertEqual(
            self.get_var(locals()),
            {'var_obj': Expr(0, {}, {'val': Expr(1, {0: Expr(2, {}, {'val': Expr(3, {0: 3}, {})})}, {})})},
        )

    def test_aug_assign_subscript_slice(self):
        # e.g. obj[a:b] += rhs - augmented assignment on a basic slice
        var_obj = Expr(0)
        var_obj[0:1] = Expr.int(1)
        var_obj[0:1] += Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setitem', 0, slice(0, 1, None), 1),
                ('getitem', 0, slice(0, 1, None)),
                ('int', 2),
                ('setitem', 0, slice(0, 1, None), 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {('slice', 0, 1, None): 3}, {})})

    def test_aug_assign_subscript_slice_with_step(self):
        # e.g. obj[a:b:c] += rhs - augmented assignment on a slice with step
        var_obj = Expr(0)
        var_obj[0:4:2] = Expr.int(1)
        var_obj[0:4:2] += Expr.int(2)

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 1),
                ('setitem', 0, slice(0, 4, 2), 1),
                ('getitem', 0, slice(0, 4, 2)),
                ('int', 2),
                ('setitem', 0, slice(0, 4, 2), 3),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {('slice', 0, 4, 2): 3}, {})})


class AnnAssignTest(PILTestCase):

    def test_ann_assign_only_call_annotation(self):
        _var_x: Expr.str(0)

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_ann_assign_only_const_annotation(self):
        # constant annotation: no side effect, trace stays empty
        _var_x: int

        self.assertEqual(Expr.trace, [])
        self.assertEqual(self.get_var(locals()), {})

    def test_ann_assign_name_call_annotation(self):
        _var_x: Expr.str(0) = Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_ann_assign_name_const_annotation(self):
        _var_x: int = Expr.int(0)

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_ann_assign_attr_target(self):
        var_obj = Expr(0)
        var_obj.val: Expr.str(0) = Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 1), ('setattr', 0, 'val', 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {'val': 1})})

    def test_ann_assign_subscript_target(self):
        var_obj = Expr(0)
        var_obj[0]: Expr.str(0) = Expr.int(1)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 1), ('setitem', 0, 0, 1)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 1}, {})})

    def test_ann_assign_subscript_target_obj_from_call(self):
        def make():
            return Expr(0)

        make()[0]: Expr.str(0) = Expr.int(1)

        self.assertEqual(Expr.trace, [('int', 1), ('init', 0), ('setitem', 0, 0, 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_ann_assign_subscript_target_slice_from_call(self):
        var_obj = Expr(0)
        var_obj[Expr.int(0)]: Expr.str(1) = Expr.int(2)

        self.assertEqual(Expr.trace, [('init', 0), ('int', 2), ('int', 0), ('setitem', 0, 0, 2)])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {0: 2}, {})})

    def test_ann_assign_subscript_target_obj_and_slice_from_calls(self):
        def make():
            return Expr(0)

        make()[Expr.int(0)]: Expr.str(1) = Expr.int(2)

        self.assertEqual(Expr.trace, [('int', 2), ('init', 0), ('int', 0), ('setitem', 0, 0, 2)])
        self.assertEqual(self.get_var(locals()), {})

    def test_ann_assign_subscript_target_slice_range_from_calls(self):
        var_obj = Expr(0)
        var_obj[Expr.int(0):Expr.int(1)]: Expr.str(2) = Expr.int(3)

        self.assertEqual(
            Expr.trace,
            [('init', 0), ('int', 3), ('int', 0), ('int', 1), ('setitem', 0, slice(0, 1, None), 3)],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {('slice', 0, 1, None): 3}, {})})

    def test_ann_assign_subscript_target_obj_call_slice_range(self):
        def make():
            return Expr(0)

        make()[Expr.int(0):Expr.int(1)]: Expr.str(2) = Expr.int(3)

        self.assertEqual(
            Expr.trace,
            [('int', 3), ('init', 0), ('int', 0), ('int', 1), ('setitem', 0, slice(0, 1, None), 3)],
        )
        self.assertEqual(self.get_var(locals()), {})

    def test_ann_assign_binop_annotation(self):
        _var_x: Expr.int(0) + Expr.int(1) = Expr.int(2)

        self.assertEqual(Expr.trace, [('int', 2)])
        self.assertEqual(self.get_var(locals()), {})


class DeleteTest(PILTestCase):

    def test_delete_name(self):
        var_x = Expr.int(0)
        del var_x

        self.assertEqual(Expr.trace, [('int', 0)])
        self.assertEqual(self.get_var(locals()), {})

    def test_delete_attribute(self):
        var_obj = Expr(0)
        var_obj.val = Expr.str(1)
        del var_obj.val

        self.assertEqual(Expr.trace, [('init', 0), ('str', 1), ('setattr', 0, 'val', 'str(1)'), ('delattr', 0, 'val')])
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {})})

    def test_delete_subscript(self):
        var_obj = Expr(0)
        var_obj[Expr.str(1)] = Expr.str(2)
        del var_obj[Expr.str(1)]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('str', 2),
                ('str', 1),
                ('setitem', 0, 'str(1)', 'str(2)'),
                ('str', 1),
                ('delitem', 0, 'str(1)'),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {})})

    def test_delete_subscript_slice(self):
        var_obj = Expr(0)
        var_obj[Expr.int(1):Expr.int(2)] = Expr.str(2)
        del var_obj[Expr.int(1):Expr.int(2)]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('str', 2),
                ('int', 1),
                ('int', 2),
                ('setitem', 0, slice(1, 2, None), 'str(2)'),
                ('int', 1),
                ('int', 2),
                ('delitem', 0, slice(1, 2, None)),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {})})

    def test_delete_subscript_tuple(self):
        var_obj = Expr(0)
        var_obj[Expr.int(1), Expr.int(2)] = Expr.str(2)
        del var_obj[Expr.int(1), Expr.int(2)]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('str', 2),
                ('int', 1),
                ('int', 2),
                ('setitem', 0, (1, 2), 'str(2)'),
                ('int', 1),
                ('int', 2),
                ('delitem', 0, (1, 2)),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {})})

    def test_delete_subscript_tuple_slice(self):
        var_obj = Expr(0)
        var_obj[Expr.int(1), Expr.int(2):Expr.int(3)] = Expr.str(2)
        del var_obj[Expr.int(1), Expr.int(2):Expr.int(3)]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('str', 2),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('setitem', 0, (1, slice(2, 3, None)), 'str(2)'),
                ('int', 1),
                ('int', 2),
                ('int', 3),
                ('delitem', 0, (1, slice(2, 3, None))),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {})})

    def test_delete_tuple(self):
        var_a = Expr.int(0)
        var_b = Expr.int(1)
        del var_a, var_b

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_delete_nested_tuple(self):
        var_a = Expr.int(0)
        var_b = Expr.int(1)
        del (var_a, var_b)

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_delete_nested_list(self):
        a = Expr.int(0)
        b = Expr.int(1)
        del [a, b]

        self.assertEqual(Expr.trace, [('int', 0), ('int', 1)])
        self.assertEqual(self.get_var(locals()), {})

    def test_delete_mixed(self):
        var_obj = Expr(0)
        var_obj.val = Expr.str(1)
        var_obj[Expr.str(2)] = Expr.str(3)
        var_x = Expr.int(4)
        del var_x, var_obj.val, var_obj[Expr.str(2)]

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('str', 1),
                ('setattr', 0, 'val', 'str(1)'),
                ('str', 3),
                ('str', 2),
                ('setitem', 0, 'str(2)', 'str(3)'),
                ('int', 4),
                ('delattr', 0, 'val'),
                ('str', 2),
                ('delitem', 0, 'str(2)'),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_obj': Expr(0, {}, {})})
