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


class ClassTest(PILTestCase):

    def test_cls_basic(self):
        class Cls:
            a = Expr.int(20) + Expr.int(30)
            b = a + Expr.int(20)
            c = b + a

        _var_a = Cls.a
        _var_b = Cls.b
        _var_c = Cls.c

        self.assertEqual(Expr.trace, [('int', 20), ('int', 30), ('int', 20)])
        self.assertEqual(self.get_var(locals()), {})

    def test_cls_decorate(self):
        var_e = Expr(0)

        @var_e.decorate(1)
        @var_e.decorate(2)
        class Cls:
            a = Expr.int(20) + Expr.int(30)
            b = a + Expr.int(20)
            c = b + a

        _var_a = Cls.a
        _var_b = Cls.b
        _var_c = Cls.c

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('decorate', 1),
                ('decorate', 2),
                ('int', 20),
                ('int', 30),
                ('int', 20),
                ('decorate.wrapper', 2),
                ('decorate.wrapper', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_e': Expr(0, {}, {})})

    def test_cls_inherit(self):
        var_e = Expr(0)

        class Base:
            d = Expr.int(20) + Expr.int(30)

        e = [Base]

        @var_e.decorate(1)
        @var_e.decorate(2)
        class Cls(e[0]):
            a = Expr.int(20) + Expr.int(30)
            b = a + Expr.int(20)
            c = b + a

        _var_a = Cls.a
        _var_b = Cls.b
        _var_c = Cls.c
        _var_d = Cls.d

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 20),
                ('int', 30),
                ('decorate', 1),
                ('decorate', 2),
                ('int', 20),
                ('int', 30),
                ('int', 20),
                ('decorate.wrapper', 2),
                ('decorate.wrapper', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_e': Expr(0, {}, {})})

    def test_cls_inherit_list(self):
        var_e = Expr(0)

        class Base:
            d = Expr.int(20) + Expr.int(30)

        class Base2:
            d = Expr.int(100) + Expr.int(110)

        e = [Base, Base2]

        @var_e.decorate(1)
        @var_e.decorate(2)
        class Cls(*e):
            a = Expr.int(320) + Expr.int(330)
            b = a + Expr.int(420)
            c = b + a

        _var_a = Cls.a
        _var_b = Cls.b
        _var_c = Cls.c
        _var_d = Cls.d

        self.assertEqual(
            Expr.trace,
            [
                ('init', 0),
                ('int', 20),
                ('int', 30),
                ('int', 100),
                ('int', 110),
                ('decorate', 1),
                ('decorate', 2),
                ('int', 320),
                ('int', 330),
                ('int', 420),
                ('decorate.wrapper', 2),
                ('decorate.wrapper', 1),
            ],
        )
        self.assertEqual(self.get_var(locals()), {'var_e': Expr(0, {}, {})})


class WithTest(PILTestCase):

    def test_with_basic(self):
        class Cls:
            @classmethod
            def __enter__(cls):
                return Expr.str(20) + Expr.str(30)

            @classmethod
            def __exit__(cls, exc_type, exc_value, traceback):
                Expr.int(40)

        with Cls():
            _var_g = Expr.int(20)

        self.assertEqual(Expr.trace, [('str', 20), ('str', 30), ('int', 20), ('int', 40)])
        self.assertEqual(self.get_var(locals()), {})

    def test_with_target(self):
        class Cls:
            @classmethod
            def __enter__(cls):
                return Expr.str(20) + Expr.str(30)

            @classmethod
            def __exit__(cls, exc_type, exc_value, traceback):
                Expr.int(40)

        with Cls() as var_f:
            _var_g = var_f[0]

        self.assertEqual(Expr.trace, [('str', 20), ('str', 30), ('int', 40)])
        self.assertEqual(self.get_var(locals()), {'var_f': 'str(20)str(30)'})

    def test_with_multiple_target(self):
        class Cls:
            @classmethod
            def __enter__(cls):
                return Expr.str(20) + Expr.str(30)

            @classmethod
            def __exit__(cls, exc_type, exc_value, traceback):
                Expr.int(40)

        with Cls() as var_f0, Cls() as var_f1:
            _var_g0 = var_f0[0]
            _var_g1 = var_f1[1]

        self.assertEqual(Expr.trace, [('str', 20), ('str', 30), ('str', 20), ('str', 30), ('int', 40), ('int', 40)])
        self.assertEqual(self.get_var(locals()), {'var_f0': 'str(20)str(30)', 'var_f1': 'str(20)str(30)'})
