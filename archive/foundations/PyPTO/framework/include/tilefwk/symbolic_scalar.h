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
 * \file symbolic_scalar.h
 * \brief
 */

#pragma once

#include <memory>
#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "error.h"

#include "ir/expr.h"

namespace npu::tile_fwk {
class RawSymbolicScalar;
using RawSymbolicScalarPtr = std::shared_ptr<RawSymbolicScalar>;

using ScalarImmediateType = long long;

// Tri-state result of SymbolicScalar::Check (satisfiability of a conjunction).
enum class SatStatus { kSat, kUnsat, kUnknown };

class SymbolicScalar {
public:
    /* Immediate type */
    SymbolicScalar(int64_t value);

    /* Symbol type*/
    SymbolicScalar(const std::string& name);
    /* Symbol type with explicit value */
    SymbolicScalar(const std::string& name, int64_t value);

    SymbolicScalar() = default;
    SymbolicScalar(const SymbolicScalar& val) = default;
    SymbolicScalar& operator=(const SymbolicScalar&) = default;

    bool ConcreteValid() const { return concreteValid_; }
    int64_t Concrete() const
    {
        FE_ASSERT(concreteValid_) << "concrete value is not valid !";
        return concrete_;
    }

    bool IsImmediate() const;
    bool IsSymbol() const;
    bool IsExpression() const;
    bool IsValid() const { return raw_ != nullptr; }

    // symbolic experession maybe too compicated
    // use this as a hint to generate intermediat variable
    void AsIntermediateVariable();
    bool IsIntermediateVariable() const;

    pypto::ir::VarPtr AsVar() const;
    pypto::ir::ExprPtr AsExpr() const;
    void GetVarRefs(std::unordered_set<const pypto::ir::Var*>& out) const;
    SymbolicScalar SubstituteVars(const std::unordered_map<pypto::ir::VarPtr, pypto::ir::ExprPtr>& vars);

    operator int() const
    {
        FE_ASSERT(concreteValid_) << "concrete value is not valid for int() !" << Dump();
        return concrete_;
    }

#define SYMBOLIC_SCALAR_DEFINE_UOP(name, uop) \
    SymbolicScalar name() const;              \
    SymbolicScalar operator uop() const { return name(); }

    SYMBOLIC_SCALAR_DEFINE_UOP(Pos, +)
    SYMBOLIC_SCALAR_DEFINE_UOP(Neg, -)
    SYMBOLIC_SCALAR_DEFINE_UOP(Not, !)
#undef SYMBOLIC_SCALAR_DEFINE_UOP

#define SYMBOLIC_SCALAR_DEFINE_BOP(name, bop)                                                    \
    SymbolicScalar name(const SymbolicScalar& sval) const;                                       \
    SymbolicScalar operator bop(const SymbolicScalar sval) const { return name(sval); }          \
    template <typename TyScalar, typename = std::enable_if_t<std::is_integral<TyScalar>::value>> \
    SymbolicScalar operator bop(TyScalar immediate) const                                        \
    {                                                                                            \
        return name(SymbolicScalar(immediate));                                                  \
    }                                                                                            \
    template <typename TyScalar, typename = std::enable_if_t<std::is_integral<TyScalar>::value>> \
    friend SymbolicScalar operator bop(TyScalar immediate, const SymbolicScalar sval)            \
    {                                                                                            \
        return SymbolicScalar(immediate).name(sval);                                             \
    }
    SYMBOLIC_SCALAR_DEFINE_BOP(Add, +)
    SYMBOLIC_SCALAR_DEFINE_BOP(Sub, -)
    SYMBOLIC_SCALAR_DEFINE_BOP(Mul, *)
    SYMBOLIC_SCALAR_DEFINE_BOP(Div, /)
    SYMBOLIC_SCALAR_DEFINE_BOP(Mod, %)

    SYMBOLIC_SCALAR_DEFINE_BOP(Eq, ==)
    SYMBOLIC_SCALAR_DEFINE_BOP(Ne, !=)
    SYMBOLIC_SCALAR_DEFINE_BOP(Lt, <)
    SYMBOLIC_SCALAR_DEFINE_BOP(Le, <=)
    SYMBOLIC_SCALAR_DEFINE_BOP(Gt, >)
    SYMBOLIC_SCALAR_DEFINE_BOP(Ge, >=)
    SYMBOLIC_SCALAR_DEFINE_BOP(And, &&)
    SYMBOLIC_SCALAR_DEFINE_BOP(Or, ||)
#undef SYMBOLIC_SCALAR_DEFINE_BOP

    SymbolicScalar Min(const SymbolicScalar& sval) const;
    SymbolicScalar Max(const SymbolicScalar& sval) const;
    SymbolicScalar Ternary(const SymbolicScalar& sval1, const SymbolicScalar& sval2) const;

    std::string Dump() const;

    SymbolicScalar Simplify() const;

    SymbolicScalar operator()() const;
    SymbolicScalar operator()(const SymbolicScalar& arg0) const;
    SymbolicScalar operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1) const;
    SymbolicScalar operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1, const SymbolicScalar& arg2) const;
    SymbolicScalar operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1, const SymbolicScalar& arg2,
                              const SymbolicScalar& arg3) const;
    SymbolicScalar operator()(const SymbolicScalar& arg0, const SymbolicScalar& arg1, const SymbolicScalar& arg2,
                              const SymbolicScalar& arg3, const SymbolicScalar& arg4) const;
    SymbolicScalar operator()(const std::vector<SymbolicScalar>& argList) const;

    friend std::ostream& operator<<(std::ostream& os, const SymbolicScalar& val) { return os << val.Dump(); }

public:
    static std::vector<int64_t> Concrete(const std::vector<SymbolicScalar>& scalarList, int64_t defValue);
    static std::vector<SymbolicScalar> FromConcrete(const std::vector<int64_t>& values);
    static SymbolicScalar FromExpr(const pypto::ir::ExprPtr& expr);

    // Satisfiability of the conjunction conds[0] && conds[1] && ... : kSat / kUnsat /
    static SatStatus Check(const std::vector<SymbolicScalar>& conds);

public:
    /* internal use */
    SymbolicScalar(RawSymbolicScalarPtr raw);

    RawSymbolicScalarPtr Raw() const { return raw_; }

    void AsLoopBegin(bool value) { isLoopBegin_ = value; }
    void AsLoopEnd(bool value) { isLoopEnd_ = value; }
    bool IsLoopBegin() const { return isLoopBegin_; }
    bool IsLoopEnd() const { return isLoopEnd_; }

private:
    RawSymbolicScalarPtr raw_{nullptr};
    bool concreteValid_{false};
    int64_t concrete_{-1};
    bool isLoopBegin_{false};
    bool isLoopEnd_{false};
};

class SymbolicScalarTracker {
public:
    typedef SymbolicScalar ValueType;

    static ValueType Value(uint64_t value, const std::string& name)
    {
        std::string symbolName = name + "(" + std::to_string(value) + ")";
        SymbolicScalar result(symbolName);
        auto& symbolDict = SymbolicScalarTracker::GetSymbolDict();
        symbolDict[symbolName] = value;
        return result;
    }

    static std::unordered_map<std::string, ScalarImmediateType>& GetSymbolDict();
};

} // namespace npu::tile_fwk

namespace std {
#define SYMBOLIC_SCALAR_DEFINE(name, bfn)                                                           \
    static inline npu::tile_fwk::SymbolicScalar bfn(const npu::tile_fwk::SymbolicScalar lhs,        \
                                                    const npu::tile_fwk::SymbolicScalar rhs)        \
    {                                                                                               \
        return lhs.name(rhs);                                                                       \
    }                                                                                               \
    template <typename TyScalar, typename = std::enable_if_t<std::is_integral<TyScalar>::value>>    \
    npu::tile_fwk::SymbolicScalar bfn(const npu::tile_fwk::SymbolicScalar sval, TyScalar immediate) \
    {                                                                                               \
        return sval.name(npu::tile_fwk::SymbolicScalar(immediate));                                 \
    }                                                                                               \
    template <typename TyScalar, typename = std::enable_if_t<std::is_integral<TyScalar>::value>>    \
    npu::tile_fwk::SymbolicScalar bfn(TyScalar immediate, const npu::tile_fwk::SymbolicScalar sval) \
    {                                                                                               \
        return npu::tile_fwk::SymbolicScalar(immediate).name(sval);                                 \
    }
SYMBOLIC_SCALAR_DEFINE(Min, min)
SYMBOLIC_SCALAR_DEFINE(Max, max)
#undef SYMBOLIC_SCALAR_DEFINE

#define SYMBOLIC_SCALAR_DEFINE_TRI(name, bfn)                                                 \
    static inline npu::tile_fwk::SymbolicScalar bfn(const npu::tile_fwk::SymbolicScalar cond, \
                                                    const npu::tile_fwk::SymbolicScalar lhs,  \
                                                    const npu::tile_fwk::SymbolicScalar rhs)  \
    {                                                                                         \
        return cond.name(lhs, rhs);                                                           \
    }
SYMBOLIC_SCALAR_DEFINE_TRI(Ternary, ternary)
#undef SYMBOLIC_SCALAR_DEFINE_TRI
} // namespace std
