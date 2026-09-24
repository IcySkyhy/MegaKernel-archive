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
import itertools

import pypto
from pypto import SatStatus, SymbolicScalar


def _sym(name: str):
    return [pypto.symbolic_scalar(x) for x in name.split()]


def _val(v):
    return pypto.symbolic_scalar(v)


def check(conds):
    return pypto.SymbolicScalar.check(conds)


def find_feasible_paths(conds, pre_conds):
    ans = []
    choices: list = [None] * len(conds)

    def dfs(i, path):
        if i == len(conds):
            ans.append(list(choices))  # choices should be copied
            return
        for choice, cond in ((True, conds[i]), (False, ~conds[i])):
            choices[i] = choice
            new_path = path + [cond.simplify()]
            if SymbolicScalar.check(new_path) != SatStatus.UNSAT:
                dfs(i + 1, new_path)

    dfs(0, pre_conds)
    return ans

# origin fa pattern
#
# for (i = 0; i < n; i += step, unroll_list=[4, 1]) {
#     if (i == 0) { do something0 } else { do something1 }
#     if (i + step >= n) { do something2 } else { do something3 }
#

# unrolled from origin fa pattern
# for (i = 0; i + step * 4 <= n; i += step * 4) {
#   if (i == 0) { do something0 } else { do something1 }
#   if (i + step >= n) { do something2 } else { do something3 }
#   if (i == step) { do something0 } else { do something1 }
#   if (i + step *2 >= n) { do something2 } else { do something3 }
#   if (i == step *2) { do something0 } else { do something1 }
#   if (i + step *3 >= n) { do something2 } else { do something3 }
#   if (i == step *3) { do something0 } else { do something1 }
#   if (i + step *4 >= n) { do something2 } else { do something3 }
def test_find_feasible_paths():
    unroll, step = 32, 10
    x = pypto.symbolic_scalar('x')  # loop var
    n = pypto.symbolic_scalar('n')  # loop end
    s = pypto.symbolic_scalar('s')  # loop start
    n = (n + 8 - 1) // 8

    # simulate loop unroll, [0, unroll)
    pre_cond = [x >= s, x + unroll * step <= n]
    # x + k == s simulate is loop begin
    conds = [x + k * step == s for k in range(unroll)]
    # x + k * step >= n simulate is loop end (inequality)
    conds += [x + (k + 1) * step >= n for k in range(unroll)]
    choices = find_feasible_paths(conds, pre_cond)
    assert len(choices) == 4


def test():
    a, b, c, d, x, n = _sym("a b c d x n")

    # Equalities that contradict: a shared operand pinned to a nonzero constant diff.
    assert check([a == b, a == b + 1]) == SatStatus.UNSAT
    assert check([a == 5, a == 6]) == SatStatus.UNSAT
    assert check([a == b + 1, a == b + 2]) == SatStatus.UNSAT
    assert check([b == a, b + 1 == a]) == SatStatus.UNSAT
    assert check([b == b + 1]) == SatStatus.UNSAT
    # Two equalities relating distinct targets, no contradiction -> Unknown.
    assert check([a == b, a == c]) == SatStatus.UNKNOWN

    # Nested And chains whose equalities contradict -> UNSAT.
    assert check([((a == 0) & (a == b)) & (a + 1 == 0)]) == SatStatus.UNSAT
    assert check([((a == 0) & (a == b)) & (a + 1 == b)]) == SatStatus.UNSAT
    assert check([((x == 0) & (x == n)) & (x + 1 == n)]) == SatStatus.UNSAT
    # Nested but satisfiable: distinct targets, no shared-operand constant diff.
    assert check([((a == b) & (a == c)) & (b == d)]) != SatStatus.UNSAT

    # Tautology: Eq(x, x) is provably true -> SAT.
    assert check([x == x]) == SatStatus.SAT
    # A lone symbolic comparison is decidable neither way -> Unknown.
    assert check([x >= 0]) == SatStatus.UNKNOWN

    # A literal false (0) makes the conjunction UNSAT.
    assert check([pypto.SymbolicScalar(0)]) == SatStatus.UNSAT
    # A constant-false comparison (3 < 2) folds to false -> UNSAT, alone or conjoined.
    assert check([_val(3) < _val(2)]) == SatStatus.UNSAT
    assert check([x >= 0, _val(3) < _val(2)]) == SatStatus.UNSAT
    # Ground-true atoms are recognised as SAT.
    assert check([_val(2) < _val(3)]) == SatStatus.SAT
    assert check([_val(5) == _val(5)]) == SatStatus.SAT

    # Two Eqs pinning the same symbol to distinct constants -> UNSAT, even when no
    # operand is structurally shared (x+1==0 pins x=-1; 2-x==0 pins x=2).
    assert check([x + 1 == 0, 2 - x == 0]) == SatStatus.UNSAT


def test_linear_feasibility():
    # Cases that exercise the z3-inspired linear-feasibility passes (integer
    # Gaussian elimination over arbitrary coefficients, and per-symbol [lo,hi]
    # bound propagation). These used to be kUnknown; they are now detected.

    # --- Non-unit-coefficient equality contradictions -> UNSAT (Gauss) ---
    x, y = _sym("x y")
    assert check([2 * x + 2 * y == 4, 2 * x + 2 * y == 6]) == SatStatus.UNSAT
    assert check([3 * x == 6, 3 * x == 9]) == SatStatus.UNSAT

    # Equality contradiction surfaced only after substitution, leaving residual
    # non-unit-coefficient equalities: y==x+1 is substituted into nothing here,
    # the 4*x rows have no ±1-coeff symbol, but Gauss still finds the conflict.
    assert check([y == x + 1, 4 * x == 2, 4 * x == 6]) == SatStatus.UNSAT

    # --- Pure inequality bound contradictions -> UNSAT (bounds) ---
    assert check([x >= 0, x <= -1]) == SatStatus.UNSAT
    assert check([x >= 10, x < 10]) == SatStatus.UNSAT
    assert check([x > 10, x < 10]) == SatStatus.UNSAT

    # --- Satisfiable conjunctions that must NOT regress to UNSAT ---
    # Multi-symbol inequalities sharing one symbolic part are now compared,
    # so this lo(0) > hi(-1) contradiction on (x + y) is detected.
    assert check([x + y >= 0, x + y <= -1]) == SatStatus.UNSAT
    # Equality pin collapses an inequality to single-symbol, with a feasible range.
    assert check([y == -1, x + y >= 0, x <= 5]) != SatStatus.UNSAT
    # Negative-coefficient bound that is satisfiable: -2*x <= -4  =>  x >= 2.
    assert check([-2 * x <= -4, x <= 10]) != SatStatus.UNSAT
    # After subst the comparison folds to ground true (0 >= 0); not UNSAT.
    assert check([y == x, x - y >= 0]) != SatStatus.UNSAT


def test_multisymbol_inequality_bounds():
    x, y, n = _sym("x y n")

    # --- Strict atoms with |coeff| > 1: must stay satisfiable (no false UNSAT).
    # The strict->non-strict fold must happen BEFORE dividing by the gcd, or
    # 2*x > 5 would tighten to x >= 4 instead of x >= 3.
    assert check([2 * x > 5, x <= 3]) != SatStatus.UNSAT            # x = 3
    assert check([2 * x < 5, x >= 2]) != SatStatus.UNSAT            # x = 2
    assert check([-2 * x > 5, x >= -3]) != SatStatus.UNSAT          # x = -3
    assert check([2 * x + 2 * y > 5, x + y <= 3]) != SatStatus.UNSAT  # x + y = 3

    # --- Multi-symbol contradictions sharing a symbolic part -> UNSAT ---
    assert check([x + 2 >= n, x + 8 <= n]) == SatStatus.UNSAT       # lo 8 > hi 2 on n - x
    assert check([2 * x + 2 * y >= 6, x + y <= 2]) == SatStatus.UNSAT  # gcd(2, 2) = 2

    # --- Non-strict |coeff| > 1, incl. Euclidean rounding on negatives ---
    assert check([2 * x >= 5, x <= 2]) == SatStatus.UNSAT           # x >= 3
    assert check([2 * x >= -5, x <= -3]) == SatStatus.UNSAT         # x >= -2
    assert check([2 * x >= -5, x <= -2]) != SatStatus.UNSAT         # x = -2

    # --- A single-point bound (lo == hi) must NOT be UNSAT ---
    assert check([x + 8 >= n, x + 8 <= n]) != SatStatus.UNSAT       # n - x == 8

    # --- Canonicalization: n - 2 <= x is the same hi bound as x + 2 >= n ---
    assert check([n - 2 <= x, x + 8 <= n]) == SatStatus.UNSAT


def test_non_unit_equality():
    # Non-±1-coefficient equalities now feed integer reasoning: a residual row
    # c*x + bias == 0 either pins x (c | bias) or is integer-unsat (c ∤ bias).
    x, y = _sym("x y")

    # --- Single-symbol pin surfaces a contradiction with a dependent inequality ---
    assert check([2 * x == 4, x <= 1]) == SatStatus.UNSAT  # x=2 violates x<=1
    assert check([2 * x == 4, x >= 3]) == SatStatus.UNSAT  # x=2 violates x>=3
    assert check([3 * x == 6, x <= 1]) == SatStatus.UNSAT  # x=2 violates x<=1

    # --- Lone integer-unsatisfiable equality (no ±1 coeff, c ∤ bias) ---
    assert check([2 * x == 3]) == SatStatus.UNSAT  # no integer x with 2x=3

    # --- Multi-symbol gcd-divisibility: gcd(2,2)=2 does not divide 5 ---
    assert check([2 * x + 2 * y == 5]) == SatStatus.UNSAT

    # --- Integer-unsat surfaced only after a ±1 substitution ---
    # y==0 reduces 2*x+y==1 to 2*x==1, which has no integer solution.
    assert check([2 * x + y == 1, y == 0]) == SatStatus.UNSAT

    # --- Satisfiable conjunctions that must NOT regress to UNSAT ---
    assert check([2 * x == 4, x <= 3]) != SatStatus.UNSAT  # x=2, 2<=3 holds
    assert check([2 * x + 3 * y == 5]) != SatStatus.UNSAT  # x=1, y=1 works
    assert check([2 * x + 2 * y == 4, x + y <= 3]) != SatStatus.UNSAT  # x+y=2<=3


def test_gcd_coefficient_substitution():
    # A non-±1 equality whose coefficient equals ±gcd(c_i) is solved exactly and
    # propagated into inequalities. These used to be kUnknown (no ±1 to start the
    # substitution, and Gauss sees no equality-vs-equality contradiction).
    x, y = _sym("x y")

    # 2x+2y==4 => x+y==2, contradicts x+y>=3. g=2 and the coeff of x is 2 == g.
    assert check([2 * x + 2 * y == 4, x + y >= 3]) == SatStatus.UNSAT
    # Same divisibility, satisfiable side: must not regress to UNSAT.
    assert check([2 * x + 2 * y == 4, x + y <= 3]) != SatStatus.UNSAT

    # gcd(4,2)=2; the ±g coefficient is on y. Solving y=-2x+3 and feeding x+y>=4
    # yields -x+3>=4 (i.e. x<=-1), which is satisfiable -> not UNSAT.
    assert check([4 * x + 2 * y == 6, x + y >= 4]) != SatStatus.UNSAT

    # Three-symbol cascade: 2x+4y+2z==8 => x+2y+z==4, contradicts x+2y+z>=5.
    z = pypto.symbolic_scalar("z")
    assert check([2 * x + 4 * y + 2 * z == 8, x + 2 * y + z >= 5]) == SatStatus.UNSAT


def test_ne_atom():
    # NE (T_BOP_NE): FoldComparisons folds a substituted NE to false -> UNSAT;
    # a lone NE is undecidable -> UNKNOWN.
    x = pypto.symbolic_scalar("x")
    assert check([x != x]) == SatStatus.UNSAT  # self-NE folds to 0 != 0
    assert check([x != 5, x == 5]) == SatStatus.UNSAT  # NE contradicts the equality pin
    assert check([x != 5]) != SatStatus.UNSAT  # lone NE: undecidable


def test_ground_and_empty():
    # Empty conjunction is vacuously true; ground constants decide directly.
    assert check([]) == SatStatus.SAT
    assert check([pypto.SymbolicScalar(1)]) == SatStatus.SAT
    assert check([pypto.SymbolicScalar(5)]) == SatStatus.SAT
    # A ground-false atom anywhere makes the whole conjunction UNSAT.
    assert check([pypto.SymbolicScalar(1), pypto.SymbolicScalar(0)]) == SatStatus.UNSAT


def test_affine_tautology():
    # IsProvablyTrue accepts an Eq whose two sides are affine-equivalent (ConstDiff 0).
    x = pypto.symbolic_scalar("x")
    assert check([2 * x == 2 * x]) == SatStatus.SAT
    assert check([x + 1 == x + 1]) == SatStatus.SAT
    assert check([2 * x + 1 == 1 + 2 * x]) == SatStatus.SAT  # affine-equivalent, not syntactic


def test_inequality_contradictions():
    # Bound-contradiction family: subst-into-inequality, negative coefficients,
    # mixed strict/non-strict bounds, and negative-value bounds.
    x, y = _sym("x y")

    # Equality pin collapses a 2-symbol row to single-symbol, exposing lo > hi.
    assert check([y == 5, x + y >= 10, x <= 3]) == SatStatus.UNSAT  # x>=5 vs x<=3
    assert check([y == 2, x - y >= 5, x <= 3]) == SatStatus.UNSAT  # x>=7 vs x<=3

    # Negative-coefficient rows: normalization (FlipRel) must preserve the verdict.
    assert check([-2 * x <= -4, x <= 1]) == SatStatus.UNSAT  # x>=2 vs x<=1
    assert check([-2 * x >= 4, x >= -1]) == SatStatus.UNSAT  # x<=-2 vs x>=-1

    # Mixed strict / non-strict bounds.
    assert check([x > 5, x <= 5]) == SatStatus.UNSAT  # x>=6 vs x<=5
    assert check([x >= 5, x < 5]) == SatStatus.UNSAT  # x>=5 vs x<=4
    assert check([x > 5, x < 6]) == SatStatus.UNSAT  # no integer strictly in (5,6)

    # Negative-value bounds.
    assert check([x >= -3, x <= -10]) == SatStatus.UNSAT
    assert check([x >= -10, x <= -3]) != SatStatus.UNSAT  # feasible, e.g. x=-5


def test_substitution_chains():
    # Multi-equality systems: ±1-substitution chains and Gauss over >2 symbols.
    a, b, c, x, y = _sym("a b c x y")

    # Chain that only contradicts after propagating a ±1 substitution to a fixed point.
    assert check([a == b, b == c, c == a + 1]) == SatStatus.UNSAT  # a=b=c, but c=a+1

    # Consistent 2-symbol system (unique rational+integer solution x=y=1): not UNSAT.
    assert check([x + y == 2, x - y == 0]) != SatStatus.UNSAT
    # Gauss catches a dependent non-±1 row contradicting a ±1 row.
    assert check([x + y == 2, 2 * x + 2 * y == 6]) == SatStatus.UNSAT


def test_gauss_multi_row():
    # Fraction-free integer Gauss across rows that are non-trivial multiples of each
    # other, including a 3-symbol sum. The contradiction surfaces only after clearing
    # a pivot column, leaving a 0 == nonzero ground row.
    x, y, z = _sym("x y z")

    # 2*(x+2y) is the LHS of row1; a contradictory constant survives elimination.
    assert check([x + 2 * y == 5, 2 * x + 4 * y == 11]) == SatStatus.UNSAT
    # Same shape, consistent constant -> not UNSAT.
    assert check([x + 2 * y == 5, 2 * x + 4 * y == 10]) != SatStatus.UNSAT

    # 3-symbol sum: row1 is exactly 2*row0 in coefficients, conflicting constant.
    assert check([x + y + z == 1, 2 * x + 2 * y + 2 * z == 5]) == SatStatus.UNSAT
    assert check([x + y + z == 6, 2 * x + 2 * y + 2 * z == 12]) != SatStatus.UNSAT

    # Scaled mixed-coefficient rows; the conflict appears only after eliminating x.
    assert check([2 * x + y == 3, 4 * x + 2 * y == 5]) == SatStatus.UNSAT
    assert check([2 * x + y == 3, 4 * x + 2 * y == 6]) != SatStatus.UNSAT


def test_dependent_row_systems_unsat():
    # row3 == 2*row2 in coefficients, but 30 != 2*14
    x, y, z = _sym("x y z")
    assert (
        check(
            [
                2 * x - y + 3 * z == 5,
                x + 4 * y - z == 2,
                3 * x + 3 * y + 2 * z == 7,
                6 * x + 6 * y + 4 * z == 20,
            ]
        )
        == SatStatus.UNSAT
    )

    # row4 == 2*row3 in coefficients, but 30 != 2*12
    w = pypto.symbolic_scalar("w")
    assert (
        check(
            [
                x + 2 * y - z + w == 4,
                3 * x - y + 2 * z - 2 * w == 1,
                2 * x + y + z + w == 7,
                5 * x + 3 * y + 4 * z - w == 12,
                10 * x + 6 * y + 8 * z - 2 * w == 30,
            ]
        )
        == SatStatus.UNSAT
    )


def test_substitution_cascades():
    # Equalities solved to a fixed point, then folded into a comparison. The
    # contradiction is invisible until the substitution has fully cascaded.
    a, b, c, x, y, z = _sym("a b c x y z")
    a0, a1, a2, a3, a4, a5 = _sym("a0 a1 a2 a3 a4 a5")

    # Three-step chain a=b=c+1=6 pins a; a<=5 contradicts, a<=6 does not.
    assert check([a == b, b == c + 1, c == 5, a <= 5]) == SatStatus.UNSAT
    assert check([a == b, b == c + 1, c == 5, a <= 6]) != SatStatus.UNSAT

    # y==x collapses x+y>=10 to 2x>=10 (x>=5); x<=4 contradicts, x<=5 does not.
    assert check([y == x, x + y >= 10, x <= 4]) == SatStatus.UNSAT
    assert check([y == x, x + y >= 10, x <= 5]) != SatStatus.UNSAT

    # x+y==10 and y+z==10 force x==z; x-z>=5 folds to 0>=5 after substitution.
    assert check([x + y == 10, y + z == 10, x - z >= 5]) == SatStatus.UNSAT
    assert check([x + y == 10, y + z == 10, x - z <= 0]) != SatStatus.UNSAT

    # Six-deep chain pins a0=5; a0>=10 contradicts, a0<=9 does not.
    deep = [a0 == a1, a1 == a2, a2 == a3, a3 == a4, a4 == a5, a5 == 5]
    assert check(deep + [a0 >= 10]) == SatStatus.UNSAT
    assert check(deep + [a0 <= 9]) != SatStatus.UNSAT


def test_subst_moves_bounds():
    # An equality can rewrite every bound onto a different symbol; PropagateBounds
    # must still catch lo>hi on the substituted variable.
    x, y = _sym("x y")

    # 2*y==x: 3*x>=10 becomes 6*y>=10 (y>=2); x<=3 becomes 2*y<=3 (y<=1) -> UNSAT.
    assert check([2 * y == x, 3 * x >= 10, x <= 3]) == SatStatus.UNSAT
    # x<=4 becomes 2*y<=4 (y<=2); with y>=2 it is feasible (y=2, x=4).
    assert check([2 * y == x, 3 * x >= 10, x <= 4]) != SatStatus.UNSAT


def test_bound_propagation_coefficients():
    # Non-unit-coefficient and negative-coefficient bounds exercise CeilDiv/FloorDiv
    # and the strict->non-strict / FlipRel normalization in PropagateBounds.
    x = pypto.symbolic_scalar("x")

    # 3*x>=10 => x>=ceil(10/3)=4; contradicts x<=3, not x<=4.
    assert check([3 * x >= 10, x <= 3]) == SatStatus.UNSAT
    assert check([3 * x >= 10, x <= 4]) != SatStatus.UNSAT

    # Strict with coefficient: 2*x>5 => x>=3; 2*x<6 => x<=2 -> UNSAT.
    assert check([2 * x > 5, 2 * x < 6]) == SatStatus.UNSAT
    # 2*x<=6 => x<=3; with x>=3 feasible (x=3).
    assert check([2 * x > 5, 2 * x <= 6]) != SatStatus.UNSAT

    # Negative coefficients: -2*x>-5 => x<=2; -2*x<-6 => x>=4 -> UNSAT.
    assert check([-2 * x > -5, -2 * x < -6]) == SatStatus.UNSAT
    # -2*x>-7 => x<=3; combined with x<=2 stays feasible.
    assert check([-2 * x > -5, -2 * x > -7]) != SatStatus.UNSAT


def test_propagate_bounds_negative_division():
    # FloorDiv rounds toward -inf, CeilDiv toward +inf. The negative-numerator
    # arms (a%b<0 / a%b>0) decide the bound; these never fold to ground or hit an
    # equality conflict, so they reach PropagateBounds.
    x = pypto.symbolic_scalar("x")

    # 3*x<=-10  =>  x<=floor(-10/3)=-4.
    assert check([3 * x <= -10, x >= -3]) == SatStatus.UNSAT
    assert check([3 * x <= -10, x >= -4]) != SatStatus.UNSAT
    # 3*x>=-10  =>  x>=ceil(-10/3)=-3.
    assert check([3 * x >= -10, x <= -4]) == SatStatus.UNSAT
    assert check([3 * x >= -10, x <= -3]) != SatStatus.UNSAT

    # Negative coefficient on a non-strict row: FlipRel, then divide negated t.
    # -3*x<=10  =>  3*x>=-10  =>  x>=-3.
    assert check([-3 * x <= 10, x <= -4]) == SatStatus.UNSAT
    assert check([-3 * x <= 10, x <= -3]) != SatStatus.UNSAT
    # -3*x>=10  =>  3*x<=-10  =>  x<=-4.
    assert check([-3 * x >= 10, x >= -3]) == SatStatus.UNSAT
    assert check([-3 * x >= 10, x >= -4]) != SatStatus.UNSAT


def test_propagate_bounds_strict_nonunit():
    # Strict inequalities with a non-unit coefficient: the ++t (>) / --t (<)
    # adjustment composes with CeilDiv/FloorDiv.
    x = pypto.symbolic_scalar("x")

    # 3*x>10  =>  3*x>=11  =>  x>=ceil(11/3)=4.
    assert check([3 * x > 10, x <= 3]) == SatStatus.UNSAT
    assert check([3 * x > 10, x <= 4]) != SatStatus.UNSAT
    # 3*x<10  =>  3*x<=9  =>  x<=floor(9/3)=3.
    assert check([3 * x < 10, x >= 4]) == SatStatus.UNSAT
    assert check([3 * x < 10, x >= 3]) != SatStatus.UNSAT


def test_propagate_bounds_aggregation():
    # Multiple constraints on one symbol: TightenLo takes the max, TightenHi the min.
    x = pypto.symbolic_scalar("x")

    assert check([x >= 2, x >= 5, x <= 4]) == SatStatus.UNSAT  # max(2,5)=5 > 4
    assert check([x >= 2, x >= 5, x <= 5]) != SatStatus.UNSAT
    assert check([x <= 10, x <= 7, x >= 8]) == SatStatus.UNSAT  # min(10,7)=7 < 8
    assert check([x <= 10, x <= 7, x >= 7]) != SatStatus.UNSAT


def test_propagate_bounds_after_subst():
    # Substitution collapses a 2-symbol comparison to single-symbol; the residual
    # coefficient then drives FloorDiv/CeilDiv inside PropagateBounds.
    x, y = _sym("x y")

    # y==3; 2*x+y<=10  =>  2*x<=7  =>  x<=floor(7/2)=3.
    assert check([y == 3, 2 * x + y <= 10, x >= 4]) == SatStatus.UNSAT
    assert check([y == 3, 2 * x + y <= 10, x >= 3]) != SatStatus.UNSAT
    # y==-3; 2*x+y>=-10  =>  2*x>=-7  =>  x>=ceil(-7/2)=-3.
    assert check([y == -3, 2 * x + y >= -10, x <= -4]) == SatStatus.UNSAT
    assert check([y == -3, 2 * x + y >= -10, x <= -3]) != SatStatus.UNSAT


def test_combined_gauss_and_subst():
    # A 2x2 system Gauss-reduces, then a ±1 coefficient solves x exactly; the pinned
    # value finally contradicts an inequality via FoldComparisons.
    x, y = _sym("x y")

    # 3x+y==7, 2x-y==3 => x=2, y=1; x>=3 contradicts.
    assert check([3 * x + y == 7, 2 * x - y == 3, x >= 3]) == SatStatus.UNSAT
    # x<=2 is consistent with the pinned x=2.
    assert check([3 * x + y == 7, 2 * x - y == 3, x <= 2]) != SatStatus.UNSAT


def test_ne_with_substitution():
    # NE atoms fold after substitution: a substituted NE whose operands become equal
    # folds to 0!=0 (false) -> UNSAT; one that stays distinct does not.
    x, y = _sym("x y")

    assert check([y == x, y != x]) == SatStatus.UNSAT  # y-x folds to 0; 0!=0
    assert check([y == x + 1, y != x]) != SatStatus.UNSAT  # folds to 1!=0
    assert check([x == 5, x != 5]) == SatStatus.UNSAT  # folds to 5!=5
    assert check([x == 5, x != 6]) != SatStatus.UNSAT  # folds to 5!=6


def test_post_subst_ground_fold():
    # Div/Mod/Min/Max atoms are non-affine: undecidable until an equality pins their
    # operands. Once subst resolves a symbol to a constant the atom is concretely
    # folded (C++ truncated /,% semantics); a ground-false fold is UNSAT.
    x, y = _sym("x y")

    # Mod.
    assert check([x == 6, x % 4 == 2]) == SatStatus.SAT  # 6 % 4 == 2
    assert check([x == 6, x % 4 == 3]) == SatStatus.UNSAT  # 6 % 4 == 2 != 3
    # Div (truncated toward zero).
    assert check([x == 7, x / 2 == 3]) == SatStatus.SAT  # 7 / 2 == 3
    assert check([x == 8, x / 2 == 3]) == SatStatus.UNSAT  # 8 / 2 == 4
    # Negative mod is truncated, not Euclidean: -6 % 4 == -2.
    assert check([x == -6, x % 4 == -2]) == SatStatus.SAT

    # Min / Max.
    assert check([x == 5, y == 8, pypto.min(x, y) == 5]) == SatStatus.SAT
    assert check([x == 6, y == 8, pypto.min(x, y) == 5]) == SatStatus.UNSAT  # min(6,8)=6
    assert check([x == 5, y == 8, pypto.max(x, y) == 8]) == SatStatus.SAT
    assert check([x == 5, y == 8, pypto.max(x, y) == 7]) == SatStatus.UNSAT  # max(5,8)=8

    # Transitive substitution chain pins a, then folds a % 4.
    a, b = _sym("a b")
    assert check([a == b, b == 6, a % 4 == 2]) == SatStatus.SAT

    # Divisor pinned to zero only through subst: refuse to fold -> UNKNOWN, no crash.
    assert check([x == 5, y == 0, x / y == 1]) == SatStatus.UNKNOWN
    assert check([x == 5, y == 0, x % y == 1]) == SatStatus.UNKNOWN

    # Unpinned operands stay UNKNOWN (no false SAT/UNSAT).
    assert check([x % 4 == 2]) == SatStatus.UNKNOWN
    assert check([x / 2 == 3]) == SatStatus.UNKNOWN


def test_nonlinear_and_non_affine():
    # Nonlinear (symbol*symbol) atoms are not affine-linear, so the checker must
    # leave them undecidable: never crash, and never a false UNSAT.
    x, y = _sym("x y")

    assert check([x * y == 0]) != SatStatus.UNSAT
    # Mathematically contradictory, but nonlinear -> conservatively UNKNOWN.
    assert check([x * y >= 1, x * y <= 0]) != SatStatus.UNSAT


def test_affine_mop_call():
    c = pypto.Tensor([-1, 32], pypto.DT_FP32, 'c')
    b = c.shape[0]
    assert str(b) == "RUNTIME_GetInputShapeDim(ARG_c,0)"
    a = pypto.symbolic_scalar("a")

    # Three equalities pin a-b to three distinct constants -> UNSAT.
    assert check([a + 1 == b - 1, a + 2 == b - 1, a + 3 == b - 1]) == SatStatus.UNSAT
    # The plain-symbol baseline (same shape) is UNSAT too.
    bs = pypto.symbolic_scalar("b")
    assert check([a + 1 == bs - 1, a + 2 == bs - 1, a + 3 == bs - 1]) == SatStatus.UNSAT
    # A satisfiable equality over the runtime dim must never be flagged UNSAT.
    assert check([a + 1 == b - 1]) != SatStatus.UNSAT


def test_affine_tautology_complex():
    # Multi-symbol equalities whose two sides are affine-equivalent (ConstDiff 0) are
    # provably true -> SAT, even when the surface forms differ.
    x, y = _sym("x y")

    assert check([2 * x - 2 * y == 2 * (x - y)]) == SatStatus.SAT
    assert check([(x + y) - (y + x) == 0]) == SatStatus.SAT
    assert check([3 * (x + 1) == 3 * x + 3]) == SatStatus.SAT


def _queens_constraints(cols, n):
    # For queens fixed in distinct rows, columns are `cols`. Rows i<j already differ,
    # so a pair is non-attacking iff columns differ and |ci - cj| != (j - i). The
    # diagonal test splits into two NE atoms: ci - d != cj  AND  ci + d != cj.
    atoms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            atoms.append(cols[i] != cols[j])
            atoms.append(cols[i] - d != cols[j])
            atoms.append(cols[i] + d != cols[j])
    return atoms


def test_three_queens_is_unsat():
    # Classic result: 3 non-attacking queens cannot be placed on a 3x3 board. The
    # checker is conjunctive+affine, so it cannot express the disjunction directly;
    # instead we enumerate every column assignment, pin the columns, and let each NE
    # atom fold to ground. Zero feasible assignments => the puzzle is UNSAT.
    n = 3
    cols = [pypto.symbolic_scalar(f"c{i}") for i in range(n)]
    cons = _queens_constraints(cols, n)

    feasible = 0
    for assignment in itertools.product(range(n), repeat=n):
        pins = [cols[i] == assignment[i] for i in range(n)]
        if check(pins + cons) != SatStatus.UNSAT:
            feasible += 1
    assert feasible == 0

    # Positive control: the encoding admits the known 4-queens solution (1,3,0,2),
    # so the UNSAT above is a property of 3-queens, not of a broken encoding.
    n4 = 4
    q = [pypto.symbolic_scalar(f"q{i}") for i in range(n4)]
    cons4 = _queens_constraints(q, n4)
    sol = (1, 3, 0, 2)
    assert check([q[i] == sol[i] for i in range(n4)] + cons4) != SatStatus.UNSAT
