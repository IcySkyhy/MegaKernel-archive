/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file symbolic_scalar_solver.h
 * \brief Affine-linear SAT checker for SymbolicScalar conditions (SAT/UNSAT/Unknown).
 *
 * Limitations:
 *   - Symbol identity is by Dump() string (one-name-per-variable assumed).
 *   - Only LT/LE/GT/GE/EQ/NE atoms; callers should pre-simplify Not/Or
 *     (e.g. ~(x>=0) -> x<0).
 *   - int64 arithmetic, unguarded against overflow.
 *   - Returns kUnknown for non-affine atoms (Div/Mod/Min/Max/Call) and nonlinear
 *     terms (x*y). Multi-symbol affine inequalities ARE compared when they share
 *     one symbolic part (x+2>=n && x+8<=n).
 */
#pragma once

#include "symbolic_scalar_simplify.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace npu::tile_fwk {

// Affine form: {coeffs, bias}; ok=false on anything not affine-linear.
class AffineForm {
public:
    std::map<std::string, int64_t> coeffs;
    int64_t bias = 0;
    bool ok = false;

    AffineForm() = default;

    explicit AffineForm(const RawPtr& n)
    {
        if (auto v = GetConstVal(n)) { // immediate
            bias = *v;
            ok = true;
            return;
        }
        if (n->IsSymbol()) {
            coeffs[n->Dump()] = 1;
            ok = true;
            return;
        }
        auto e = std::static_pointer_cast<Expr>(n);
        const auto& ops = e->OperandList();
        switch (e->Opcode()) {
            case SymbolicOpcode::T_MOP_CALL:
            case SymbolicOpcode::T_BOP_DIV:
            case SymbolicOpcode::T_BOP_MOD:
                coeffs[n->Dump()] = 1;
                ok = true;
                return;
            case SymbolicOpcode::T_UOP_POS:
                *this = MulS(ops[0], 1);
                return;
            case SymbolicOpcode::T_UOP_NEG:
                *this = MulS(ops[0], -1);
                return;
            case SymbolicOpcode::T_BOP_ADD:
                *this = Add(ops[0], ops[1]);
                return;
            case SymbolicOpcode::T_BOP_SUB:
                *this = Sub(ops[0], ops[1]);
                return;
            case SymbolicOpcode::T_BOP_MUL: {
                // Linear only: one operand must be a constant.
                if (auto k = GetConstVal(ops[0])) {
                    *this = MulS(ops[1], *k);
                    return;
                }
                if (auto k = GetConstVal(ops[1])) {
                    *this = MulS(ops[0], *k);
                    return;
                }
                return; // nonlinear (symbol * symbol)
            }
            default:
                return;
        }
    }

    // Drop zero coefficients.
    void Prune()
    {
        auto it = coeffs.begin();
        while (it != coeffs.end()) {
            if (it->second == 0) {
                it = coeffs.erase(it);
            } else {
                ++it;
            }
        }
    }

    static AffineForm Add(const RawPtr& a, const RawPtr& b)
    {
        auto la = AffineForm(a);
        auto lb = AffineForm(b);
        if (!la.ok || !lb.ok) {
            return AffineForm();
        }
        for (const auto& kv : lb.coeffs) {
            la.coeffs[kv.first] += kv.second;
        }
        la.bias += lb.bias;
        la.Prune();
        return la;
    }

    static AffineForm Sub(const RawPtr& a, const RawPtr& b)
    {
        auto la = AffineForm(a);
        auto lb = AffineForm(b);
        if (!la.ok || !lb.ok) {
            return AffineForm();
        }
        for (const auto& kv : lb.coeffs) {
            la.coeffs[kv.first] -= kv.second;
        }
        la.bias -= lb.bias;
        la.Prune();
        return la;
    }

    static AffineForm MulS(const RawPtr& a, int64_t k)
    {
        auto la = AffineForm(a);
        if (!la.ok) {
            return AffineForm();
        }
        for (auto& kv : la.coeffs) {
            kv.second *= k;
        }
        la.bias *= k;
        la.Prune();
        return la;
    }

    // Apply substitution (symbol -> affine form) to *this to a fixed point. Returns
    // false on a cycle (caller treats the atom as not decidable). Mutates *this.
    bool ApplySubst(const std::map<std::string, AffineForm>& subst)
    {
        if (subst.empty()) {
            return true;
        }
        // Each progressing pass eliminates at least one subst key from coeffs.
        for (size_t bound = 0; bound <= subst.size() + 1; ++bound) {
            bool changed = false;
            for (auto it = coeffs.begin(); it != coeffs.end();) {
                auto f = subst.find(it->first);
                if (f == subst.end()) {
                    ++it;
                    continue;
                }
                int64_t coef = it->second;
                it = coeffs.erase(it);
                bias += coef * f->second.bias;
                for (const auto& kv : f->second.coeffs) {
                    coeffs[kv.first] += coef * kv.second;
                }
                changed = true;
            }
            Prune();
            if (!changed) {
                return true;
            }
        }
        return false;
    }

    // row := scale * row - factor * piv (fraction-free row combination); prunes zero
    // coefficients. int64 arithmetic, unguarded against overflow.
    void RowMulSub(int64_t scale, const AffineForm& piv, int64_t factor)
    {
        std::map<std::string, int64_t> nc = coeffs;
        for (auto& kv : nc) { // scale * coeffs
            kv.second *= scale;
        }
        for (const auto& kv : piv.coeffs) {     // - factor * piv.coeffs
            nc[kv.first] -= factor * kv.second; // operator[] zeroes new keys
        }
        AffineForm out;
        out.ok = true;
        out.bias = scale * bias - factor * piv.bias;
        for (const auto& kv : nc) {
            if (kv.second != 0) {
                out.coeffs[kv.first] = kv.second;
            }
        }
        *this = std::move(out);
    }

    // Constant (e1 - e2) iff their symbolic parts are identical, else nullopt.
    static std::optional<int64_t> ConstDiff(const RawPtr& e1, const RawPtr& e2)
    {
        auto la = Sub(e1, e2);
        if (!la.ok || !la.coeffs.empty()) {
            return std::nullopt;
        }
        return la.bias;
    }

    static bool GaussCheck(std::vector<AffineForm>& rows)
    {
        const size_t n = rows.size();
        size_t pivot = 0;
        while (pivot < n) {
            size_t pr = pivot;
            while (pr < n && rows[pr].coeffs.empty()) { // skip now-ground rows
                ++pr;
            }
            if (pr == n) {
                break;
            }
            std::swap(rows[pivot], rows[pr]);
            const std::string psym = rows[pivot].coeffs.begin()->first;
            const int64_t pc = rows[pivot].coeffs.begin()->second;
            for (size_t j = 0; j < n; ++j) { // clear psym from every other row
                if (j == pivot) {
                    continue;
                }
                auto it = rows[j].coeffs.find(psym);
                if (it == rows[j].coeffs.end()) {
                    continue;
                }
                rows[j].RowMulSub(pc, rows[pivot], it->second);
            }
            ++pivot;
        }

        for (const auto& r : rows) {
            if (r.coeffs.empty() && r.bias != 0) { // 0 == nonzero contradiction
                return true;
            }
        }
        return false;
    }
};

class SatChecker {
public:
    explicit SatChecker(const std::vector<RawPtr>& conds)
    {
        for (const auto& c : conds) {
            FlattenAnd(c, conjuncts_);
        }
    }

    // Decide the conjunction (each cond may be an And chain):
    //   kSat     — every conjunct is provably true
    //   kUnsat   — a contradiction is found
    //   kUnknown — neither
    SatStatus Check()
    {
        if (conjuncts_.empty()) {
            return SatStatus::kSat;
        }

        for (const auto& c : conjuncts_) { // false atom => UNSAT
            if (EvalGroundBool(c) == false) {
                return SatStatus::kUnsat;
            }
        }

        std::vector<AffineForm> eqRows;
        for (const auto& pr : CollectEqs()) {
            auto row = AffineForm::Sub(pr.first, pr.second);
            if (row.ok) {
                eqRows.push_back(std::move(row));
            }
        }
        if (AffineForm::GaussCheck(eqRows)) {
            return SatStatus::kUnsat;
        }

        if (auto st = BuildSubst()) { // equalities => substitution
            return *st;
        }

        ComputeGround();                   // pin substituted symbols for concrete folding
        for (const auto& c : conjuncts_) { // ground-false atom (incl. Div/Mod/Min/Max) => UNSAT
            if (auto v = EvalGroundExpr(c)) {
                if (*v == 0) {
                    return SatStatus::kUnsat;
                }
            }
        }

        if (auto st = FoldComparisons()) { // false folded atom => UNSAT
            return *st;
        }

        if (PropagateBounds()) { // symbolic lo > hi => UNSAT
            return SatStatus::kUnsat;
        }

        for (const auto& c : conjuncts_) { // else SAT iff all provably true
            if (!IsProvablyTrue(c)) {
                return SatStatus::kUnknown;
            }
        }
        return SatStatus::kSat;
    }

private:
    std::vector<RawPtr> conjuncts_;
    std::map<std::string, AffineForm> subst_;
    std::map<std::string, int64_t> ground_; // symbols subst_ pins to a constant

    static void FlattenAnd(const RawPtr& node, std::vector<RawPtr>& out)
    {
        if (!node->IsExpression()) {
            out.push_back(node);
            return;
        }
        auto e = std::static_pointer_cast<Expr>(node);
        if (e->Opcode() != SymbolicOpcode::T_MOP_AND) {
            out.push_back(node);
            return;
        }
        for (const auto& op : e->OperandList()) {
            FlattenAnd(op, out);
        }
    }

    // Collect operands of every Eq atom in the conjunction.
    std::vector<std::pair<RawPtr, RawPtr>> CollectEqs() const
    {
        std::vector<std::pair<RawPtr, RawPtr>> eqs;
        for (const auto& c : conjuncts_) {
            if (c->IsExpression()) {
                auto ce = std::static_pointer_cast<Expr>(c);
                if (ce->Opcode() == SymbolicOpcode::T_BOP_EQ) {
                    eqs.emplace_back(ce->OperandList()[0], ce->OperandList()[1]);
                }
            }
        }
        return eqs;
    }

    // Constant value of an affine form with no symbols, else nullopt.
    static std::optional<int64_t> ConstFoldGround(const RawPtr& n)
    {
        AffineForm a(n);
        if (!a.ok || !a.coeffs.empty()) {
            return std::nullopt;
        }
        return a.bias;
    }

    // Evaluate a ground (symbol-free) bool atom; nullopt if not decidable.
    static std::optional<bool> EvalGroundBool(const RawPtr& c)
    {
        if (auto v = GetConstVal(c)) {
            return *v != 0;
        }
        if (!IsCompareExpr(c)) {
            return std::nullopt;
        }
        auto ce = std::static_pointer_cast<Expr>(c);
        auto lv = ConstFoldGround(ce->OperandList()[0]);
        auto rv = ConstFoldGround(ce->OperandList()[1]);
        if (!lv || !rv) {
            return std::nullopt;
        }
        return EvalBinary(ce->Opcode(), *lv, *rv);
    }

    // True iff c is ground-true (incl. Div/Mod/Min/Max once subst_ pins operands),
    // or a symbolic Eq whose affine diff is 0.
    bool IsProvablyTrue(const RawPtr& c) const
    {
        if (auto v = EvalGroundExpr(c)) {
            return *v != 0;
        }
        if (c->IsExpression()) {
            auto ce = std::static_pointer_cast<Expr>(c);
            if (ce->Opcode() == SymbolicOpcode::T_BOP_EQ) {
                if (auto d = AffineForm::ConstDiff(ce->OperandList()[0], ce->OperandList()[1])) {
                    return *d == 0;
                }
            }
        }
        return false;
    }

    // Resolve every subst_ entry whose bound collapses to a constant into ground_.
    void ComputeGround()
    {
        for (const auto& kv : subst_) {
            AffineForm r = kv.second;
            if (r.ApplySubst(subst_) && r.coeffs.empty()) {
                ground_[kv.first] = r.bias;
            }
        }
    }

    // Concretely evaluate n under ground_. Returns nullopt when any symbol is not
    // pinned, on a Call, or on a potential div/mod-by-zero -- sound (no guess).
    // Arithmetic uses C++ truncated /,% to match the language semantics.
    std::optional<int64_t> EvalGroundExpr(const RawPtr& n) const
    {
        if (auto v = GetConstVal(n)) {
            return *v;
        }
        if (n->IsSymbol()) {
            auto it = ground_.find(n->Dump());
            return it != ground_.end() ? std::optional<int64_t>(it->second) : std::nullopt;
        }
        if (!n->IsExpression()) {
            return std::nullopt;
        }
        auto e = std::static_pointer_cast<Expr>(n);
        const auto& ops = e->OperandList();
        switch (e->Opcode()) {
            case SymbolicOpcode::T_UOP_POS:
                return EvalGroundExpr(ops[0]);
            case SymbolicOpcode::T_UOP_NEG:
                if (auto v = EvalGroundExpr(ops[0])) {
                    return -*v;
                }
                return std::nullopt;
            case SymbolicOpcode::T_UOP_NOT:
                if (auto v = EvalGroundExpr(ops[0])) {
                    return (*v == 0) ? 1 : 0;
                }
                return std::nullopt;
            case SymbolicOpcode::T_BOP_ADD:
            case SymbolicOpcode::T_BOP_SUB:
            case SymbolicOpcode::T_BOP_MUL:
            case SymbolicOpcode::T_BOP_DIV:
            case SymbolicOpcode::T_BOP_MOD:
            case SymbolicOpcode::T_BOP_LT:
            case SymbolicOpcode::T_BOP_LE:
            case SymbolicOpcode::T_BOP_GT:
            case SymbolicOpcode::T_BOP_GE:
            case SymbolicOpcode::T_BOP_EQ:
            case SymbolicOpcode::T_BOP_NE: {
                auto lv = EvalGroundExpr(ops[0]);
                auto rv = EvalGroundExpr(ops[1]);
                if (!lv || !rv) {
                    return std::nullopt;
                }
                if ((e->Opcode() == SymbolicOpcode::T_BOP_DIV || e->Opcode() == SymbolicOpcode::T_BOP_MOD) &&
                    *rv == 0) {
                    return std::nullopt; // refuse to divide by zero
                }
                return EvalBinary(e->Opcode(), *lv, *rv);
            }
            case SymbolicOpcode::T_MOP_MIN: {
                std::optional<int64_t> acc;
                for (const auto& op : ops) {
                    auto v = EvalGroundExpr(op);
                    if (!v) {
                        return std::nullopt;
                    }
                    acc = acc ? std::min(*acc, *v) : *v;
                }
                return acc;
            }
            case SymbolicOpcode::T_MOP_MAX: {
                std::optional<int64_t> acc;
                for (const auto& op : ops) {
                    auto v = EvalGroundExpr(op);
                    if (!v) {
                        return std::nullopt;
                    }
                    acc = acc ? std::max(*acc, *v) : *v;
                }
                return acc;
            }
            case SymbolicOpcode::T_MOP_AND: {
                for (const auto& op : ops) {
                    auto v = EvalGroundExpr(op);
                    if (!v) {
                        return std::nullopt;
                    }
                    if (*v == 0) {
                        return 0;
                    }
                }
                return 1;
            }
            case SymbolicOpcode::T_MOP_OR: {
                for (const auto& op : ops) {
                    auto v = EvalGroundExpr(op);
                    if (!v) {
                        return std::nullopt;
                    }
                    if (*v != 0) {
                        return 1;
                    }
                }
                return 0;
            }
            default:
                return std::nullopt; // T_MOP_CALL and unhandled opcodes: undecidable
        }
    }

    static int64_t EvalBinary(SymbolicOpcode op, int64_t lv, int64_t rv)
    {
        switch (op) {
            case SymbolicOpcode::T_BOP_ADD:
                return lv + rv;
            case SymbolicOpcode::T_BOP_SUB:
                return lv - rv;
            case SymbolicOpcode::T_BOP_MUL:
                return lv * rv;
            case SymbolicOpcode::T_BOP_DIV:
                return lv / rv; // truncated toward zero
            case SymbolicOpcode::T_BOP_MOD:
                return lv % rv;
            case SymbolicOpcode::T_BOP_LT:
                return lv < rv;
            case SymbolicOpcode::T_BOP_LE:
                return lv <= rv;
            case SymbolicOpcode::T_BOP_GT:
                return lv > rv;
            case SymbolicOpcode::T_BOP_GE:
                return lv >= rv;
            case SymbolicOpcode::T_BOP_EQ:
                return lv == rv;
            case SymbolicOpcode::T_BOP_NE:
                return lv != rv;
            default:
                return 0;
        }
    }

    static bool IsCompareExpr(const RawPtr& e)
    {
        if (!e->IsExpression()) {
            return false;
        }
        auto ce = std::static_pointer_cast<Expr>(e);
        switch (ce->Opcode()) {
            case SymbolicOpcode::T_BOP_LT:
            case SymbolicOpcode::T_BOP_LE:
            case SymbolicOpcode::T_BOP_GT:
            case SymbolicOpcode::T_BOP_GE:
            case SymbolicOpcode::T_BOP_EQ:
            case SymbolicOpcode::T_BOP_NE:
                return true;
            default:
                return false;
        }
    }

    std::optional<SatStatus> SolveRow(const AffineForm& la, bool& progressed)
    {
        int64_t g = 0;
        for (const auto& kv : la.coeffs) {
            g = std::gcd(g, kv.second); // std::gcd is non-negative; gcd(0, x) = |x|
        }
        if (la.bias % g != 0) {
            return SatStatus::kUnsat; // integer equation has no solution
        }
        for (const auto& kv : la.coeffs) {
            if (kv.second == g || kv.second == -g) {
                if (UpdateSymbol(la, kv.first, kv.second)) {
                    progressed = true;
                }
                return std::nullopt;
            }
        }
        return std::nullopt;
    }

    bool UpdateSymbol(const AffineForm& la, const std::string& sym, int64_t c)
    {
        AffineForm bound;
        bound.ok = true;
        for (const auto& kv : la.coeffs) {
            if (kv.first != sym) {
                bound.coeffs[kv.first] = -kv.second / c;
            }
        }
        bound.bias = -la.bias / c;
        bound.Prune();
        return subst_.emplace(sym, bound).second;
    }

    // Solve each Eq into subst_, to a fixed point. Returns kUnsat if an equality
    // reduces to a nonzero-constant contradiction.
    std::optional<SatStatus> BuildSubst()
    {
        const auto eqs = CollectEqs();
        for (size_t round = 0; round <= eqs.size() + 1; ++round) {
            bool progressed = false;
            for (const auto& pr : eqs) {
                auto la = AffineForm::Sub(pr.first, pr.second);
                if (!la.ok || !la.ApplySubst(subst_)) {
                    continue;
                }
                if (la.coeffs.empty()) {
                    if (la.bias != 0) {
                        return SatStatus::kUnsat; // contradictory equality
                    }
                    continue; // 0 == 0
                }
                if (auto st = SolveRow(la, progressed)) {
                    return st;
                }
            }
            if (!progressed) {
                break;
            }
        }
        return std::nullopt;
    }

    // Substitute into every comparison atom; return kUnsat if one folds to false.
    std::optional<SatStatus> FoldComparisons() const
    {
        for (const auto& c : conjuncts_) {
            if (!IsCompareExpr(c)) {
                continue;
            }
            auto ce = std::static_pointer_cast<Expr>(c);
            const auto& ops = ce->OperandList();
            auto la = AffineForm::Sub(ops[0], ops[1]);
            if (!la.ok || !la.ApplySubst(subst_)) {
                continue;
            }
            if (!la.coeffs.empty()) {
                continue; // symbolic parts differ
            }
            if (!EvalBinary(ce->Opcode(), la.bias, 0)) {
                return SatStatus::kUnsat; // la OP rb folds false
            }
        }
        return std::nullopt;
    }

    // Detect lo > hi contradictions. Bounds are keyed on a canonical affine
    // signature (gcd-reduced, sign-fixed) so that inequalities sharing one
    // symbolic part -- including multi-symbol ones such as x+2>=n and x+8<=n,
    // both "n - x" -- are compared against each other.
    bool PropagateBounds() const
    {
        std::map<std::string, IntBound> bounds;
        for (const auto& c : conjuncts_) {
            if (!IsCompareExpr(c)) {
                continue;
            }
            auto ce = std::static_pointer_cast<Expr>(c);
            SymbolicOpcode op = ce->Opcode();
            if (op == SymbolicOpcode::T_BOP_EQ || op == SymbolicOpcode::T_BOP_NE) {
                continue; // handled by the equality path
            }
            auto la = AffineForm::Sub(ce->OperandList()[0], ce->OperandList()[1]); // coeffs*s + bias OP 0
            if (!la.ok || !la.ApplySubst(subst_) || la.coeffs.empty()) {
                continue; // non-affine, subst cycle, or ground (handled elsewhere)
            }
            int64_t thr = 0;
            std::string key;
            if (!NormalizeAtom(la, op, thr, key)) {
                continue;
            }
            IntBound& bd = bounds[key];
            if (op == SymbolicOpcode::T_BOP_GE) {
                bd.TightenLo(thr); // D >= thr
            } else {               // LE
                bd.TightenHi(thr); // D <= thr
            }
        }
        for (const auto& kv : bounds) {
            if (kv.second.Empty()) {
                return true;
            }
        }
        return false;
    }

    // Reduce `la OP 0` to an inclusive integer bound on D = sum (coeff/g)*s,
    // where g = gcd(coeffs) and the sign is fixed so the lexicographically
    // smallest coefficient is positive. Emits thr (D >= thr for GE, D <= thr
    // for LE) and a delimiter-safe canonical signature key. Strict comparisons
    // are folded to non-strict BEFORE dividing by g (over the integers),
    // matching the single-symbol path; dividing first would over-tighten atoms
    // such as 2*x > 5 and yield a false UNSAT. Returns false if la is ground.
    static bool NormalizeAtom(AffineForm la, SymbolicOpcode& op, int64_t& thr, std::string& key)
    {
        int64_t g = 0;
        for (const auto& kv : la.coeffs) {
            g = std::gcd(g, kv.second); // std::gcd is non-negative; gcd(0, x) = |x|
        }
        if (g <= 0) {
            return false;
        }
        if (la.coeffs.begin()->second < 0) { // canonical sign: first key positive
            for (auto& kv : la.coeffs) {
                kv.second = -kv.second;
            }
            la.bias = -la.bias;
            op = FlipRel(op);
        }
        int64_t t = -la.bias;                 // g * D OP t
        if (op == SymbolicOpcode::T_BOP_GT) { // strict -> non-strict in the g*D space
            op = SymbolicOpcode::T_BOP_GE;
            ++t;
        } else if (op == SymbolicOpcode::T_BOP_LT) {
            op = SymbolicOpcode::T_BOP_LE;
            --t;
        }
        thr = (op == SymbolicOpcode::T_BOP_GE) ? CeilDiv(t, g) : FloorDiv(t, g);
        for (const auto& kv : la.coeffs) {
            key += kv.first;
            key += '=';
            key += std::to_string(kv.second / g);
            key += ';';
        }
        return true;
    }

    struct IntBound {
        std::optional<int64_t> lo; // inclusive lower bound
        std::optional<int64_t> hi; // inclusive upper bound

        void TightenLo(int64_t v) { lo = lo ? std::max(*lo, v) : v; }
        void TightenHi(int64_t v) { hi = hi ? std::min(*hi, v) : v; }
        bool Empty() const { return lo && hi && *lo > *hi; }
    };

    // b > 0; Euclidean floor (rounds toward -inf).
    static int64_t FloorDiv(int64_t a, int64_t b)
    {
        int64_t q = a / b;
        return (a % b < 0) ? (q - 1) : q;
    }

    // b > 0; Euclidean ceil (rounds toward +inf).
    static int64_t CeilDiv(int64_t a, int64_t b)
    {
        int64_t q = a / b;
        return (a % b > 0) ? (q + 1) : q;
    }

    static SymbolicOpcode FlipRel(SymbolicOpcode op)
    {
        switch (op) {
            case SymbolicOpcode::T_BOP_LT:
                return SymbolicOpcode::T_BOP_GT;
            case SymbolicOpcode::T_BOP_LE:
                return SymbolicOpcode::T_BOP_GE;
            case SymbolicOpcode::T_BOP_GT:
                return SymbolicOpcode::T_BOP_LT;
            case SymbolicOpcode::T_BOP_GE:
                return SymbolicOpcode::T_BOP_LE;
            default:
                return op;
        }
    }
};
} // namespace npu::tile_fwk
