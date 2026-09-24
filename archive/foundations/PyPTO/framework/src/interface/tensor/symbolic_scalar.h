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

#include <cstddef>
#include <algorithm>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include <tilefwk/symbolic_scalar.h>

#include "tilefwk/error.h"
#include "interface/inner/hash_buffer.h"
#include "symbol_handler.h"
#include "interface/utils/error.h"
#include "interface/utils/string_utils.h"
#include "interface/utils/common.h"
#include "ir/scalar_expr.h"
#include "ir/expr.h"

using Json = nlohmann::json;

using namespace pypto;

namespace npu::tile_fwk {

constexpr int INVALID_SCALAR_IMMEDIATE = -1;
constexpr const char* const SPECIAL_SYMBOL_NAME_RUNTIME_PREFIX = "RUNTIME_";
constexpr const char* const SPECIAL_SYMBOL_NAME_RUNTIME_COA_PREFIX = "RUNTIME_COA_";
constexpr const char* const SPECIAL_SYMBOL_NAME_ARG_PREFIX = "ARG_";

enum class SymbolicScalarKind {
    T_SCALAR_SYMBOLIC_IMMEDIATE,
    T_SCALAR_SYMBOLIC_SYMBOL,
    T_SCALAR_SYMBOLIC_EXPRESSION,
};

inline std::string SymbolicScalarKind2Name(SymbolicScalarKind kind)
{
    std::string name;
    switch (kind) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
            name = "immediate";
            break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
            name = "symbol";
            break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION:
            name = "expression";
            break;
        default:
            FE_ASSERT(false) << " undefined kind.";
            break;
    }
    return name;
}

enum class SymbolicOpcode {
    T_UOP_POS,
    T_UOP_NEG,
    T_UOP_NOT,

    T_BOP_ADD,
    T_BOP_SUB,
    T_BOP_MUL,
    T_BOP_DIV,
    T_BOP_MOD,

    T_BOP_EQ,
    T_BOP_NE,
    T_BOP_LT,
    T_BOP_LE,
    T_BOP_GT,
    T_BOP_GE,

    T_BOP_MIN_DEPRECATED,
    T_BOP_MAX_DEPRECATED,

    T_MOP_CALL,
    T_MOP_MIN,
    T_MOP_MAX,

    T_MOP_AND,
    T_MOP_OR,

    T_UOP_BEGIN = T_UOP_POS,
    T_UOP_END = T_UOP_NOT + 1,
    T_BOP_BEGIN = T_BOP_ADD,
    T_BOP_END = T_BOP_MAX_DEPRECATED + 1
};

class RawSymbolicScalar {
public:
    SymbolicScalarKind kind;

    RawSymbolicScalar(SymbolicScalarKind tkind) : kind(tkind) {}

    SymbolicScalarKind Kind() const { return kind; }

    bool IsImmediate() const { return Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE; }
    bool IsSymbol() const { return Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL; }
    bool IsExpression() const { return Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION; }

    ScalarImmediateType GetImmediateValue() const;
    const std::string& GetSymbolName() const;
    SymbolicOpcode GetExpressionOpcode() const;
    const std::vector<RawSymbolicScalarPtr>& GetExpressionOperandList() const;
    bool IsExpressionCall(const std::string& calleeName) const;

    [[nodiscard]] bool IsIntermediateVariable() const { return intermediateVariable_; }
    void AsIntermediateVariable() { intermediateVariable_ = true; }

    virtual Json DumpJson() const = 0;
    virtual ~RawSymbolicScalar() = default;

    std::string Dump() const;

    static void FlattenOperands(const std::vector<RawSymbolicScalarPtr>& inOperandList, SymbolicOpcode objOpcode,
                                std::vector<RawSymbolicScalarPtr>& outOperandList);

private:
    friend class SymbolicScalar;
    friend class RawSymbolicExpression;
    virtual void DumpBuffer(std::ostream& buffer) const = 0;

    bool intermediateVariable_{false};
};

class RawSymbolicImmediate : public ir::ConstInt, public RawSymbolicScalar {
public:
    explicit RawSymbolicImmediate(ScalarImmediateType immediate)
        : ir::ConstInt(immediate, ir::DataType::INT64, ir::Span::Unknown()),
          RawSymbolicScalar(SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE)
    {}

    ScalarImmediateType Immediate() const { return value_; }

    Json DumpJson() const override
    {
        Json immediateDump = Json::array();
        immediateDump.push_back(kind);
        immediateDump.push_back(value_);
        return immediateDump;
    }
    static RawSymbolicScalarPtr Create(ScalarImmediateType immediate)
    {
        return std::make_shared<RawSymbolicImmediate>(immediate);
    }

private:
    void DumpBuffer(std::ostream& buffer) const override { buffer << value_; }
};

class RawSymbolicSymbol : public ir::Var, public RawSymbolicScalar {
public:
    explicit RawSymbolicSymbol(const std::string& name)
        : ir::Var(name, std::make_shared<ir::ScalarType>(ir::DataType::INT64), ir::Span::Unknown()),
          RawSymbolicScalar(SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL)
    {}

    const std::string& Name() const { return name_; }

    Json DumpJson() const override
    {
        Json symbolDump = Json::array();
        symbolDump.push_back(kind);
        symbolDump.push_back(name_);
        return symbolDump;
    }

    static RawSymbolicScalarPtr Create(const std::string& name) { return std::make_shared<RawSymbolicSymbol>(name); }

    std::shared_ptr<const ir::Var> Clone() const override;

private:
    void DumpBuffer(std::ostream& buffer) const override { buffer << name_; }
};

class RawSymbolicExpression : public ir::ScalarExpr, public RawSymbolicScalar {
public:
    RawSymbolicExpression(SymbolicOpcode opcode, const std::vector<RawSymbolicScalarPtr>& operandList)
        : ir::ScalarExpr(ir::DataType::INT64, ir::Span::Unknown()),
          RawSymbolicScalar(SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION),
          opcode_(opcode),
          operandList_(operandList)
    {
        // LoopBegin和LoopEnd目前只能作为最外层的表达式
        for (const auto& operand : operandList) {
            if (operand->IsExpression()) {
                auto expression = std::static_pointer_cast<RawSymbolicExpression>(operand);
                FE_ASSERT(!expression->IsLoopBeginCall());
                FE_ASSERT(!expression->IsLoopEndCall());
            }
        }
    }

    SymbolicOpcode Opcode() const { return opcode_; }
    const std::vector<RawSymbolicScalarPtr>& OperandList() const { return operandList_; }

    bool IsLoopBeginCall() const
    {
        if (opcode_ == SymbolicOpcode::T_MOP_CALL) {
            auto raw = operandList_[0];
            auto rawSymbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            auto callee = rawSymbol->Name();
            return callee.find(SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopBegin)) != std::string::npos;
        }
        return false;
    }

    bool IsLoopEndCall() const
    {
        if (opcode_ == SymbolicOpcode::T_MOP_CALL) {
            auto raw = operandList_[0];
            auto rawSymbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            auto callee = rawSymbol->Name();
            return callee.find(SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopEnd)) != std::string::npos;
        }
        return false;
    }

    Json DumpJson() const override
    {
        Json exprDump = Json::array();
        exprDump.push_back(kind);
        exprDump.push_back(opcode_);
        for (auto& op : operandList_) {
            exprDump.push_back(op->DumpJson());
        }
        return exprDump;
    }

#define RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(name, uop) \
    static ScalarImmediateType name(const ScalarImmediateType& val) { return uop val; }
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(CalcUopPos, +)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(CalcUopNeg, -)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP(CalcUopNot, !)
#undef RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_UOP

#define RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(name, bop)                                          \
    static ScalarImmediateType name(const ScalarImmediateType& lhs, const ScalarImmediateType& rhs) \
    {                                                                                               \
        return lhs bop rhs;                                                                         \
    }
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopAdd, +)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopSub, -)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopMul, *)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopDiv, /)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopMod, %)

    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopEq, ==)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopNe, !=)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopLt, <)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopLe, <=)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopGt, >)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopGe, >=)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopAnd, &&)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP(CalcBopOr, ||)
#undef RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOP

#define RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF(name, bfn)                                         \
    static ScalarImmediateType name(const ScalarImmediateType& lhs, const ScalarImmediateType& rhs) \
    {                                                                                               \
        return bfn(lhs, rhs);                                                                       \
    }
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF(CalcBopMin, std::min)
    RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF(CalcBopMax, std::max)
#undef RAW_SYMBOLIC_EXPRESSION_CALC_DEFINE_BOPF

    using SymbolicCalcUnary = ScalarImmediateType (*)(const ScalarImmediateType&);
    static SymbolicCalcUnary GetSymbolicCalcUnary(SymbolicOpcode opcode)
    {
        switch (opcode) {
            case SymbolicOpcode::T_UOP_POS:
                return &RawSymbolicExpression::CalcUopPos;
            case SymbolicOpcode::T_UOP_NEG:
                return &RawSymbolicExpression::CalcUopNeg;
            case SymbolicOpcode::T_UOP_NOT:
                return &RawSymbolicExpression::CalcUopNot;
            default:
                FE_ASSERT(false);
                return nullptr;
        }
    }

    using SymbolicCalcBinary = ScalarImmediateType (*)(const ScalarImmediateType&, const ScalarImmediateType&);

    static bool IsBinaryCalcOpcode(SymbolicOpcode opcode)
    {
        return (SymbolicOpcode::T_BOP_BEGIN <= opcode && opcode < SymbolicOpcode::T_BOP_END) ||
               opcode == SymbolicOpcode::T_MOP_AND || opcode == SymbolicOpcode::T_MOP_OR;
    }

    static SymbolicCalcBinary GetSymbolicCalcBinary(SymbolicOpcode opcode)
    {
        switch (opcode) {
            case SymbolicOpcode::T_BOP_ADD:
                return &RawSymbolicExpression::CalcBopAdd;
            case SymbolicOpcode::T_BOP_SUB:
                return &RawSymbolicExpression::CalcBopSub;
            case SymbolicOpcode::T_BOP_MUL:
                return &RawSymbolicExpression::CalcBopMul;
            case SymbolicOpcode::T_BOP_DIV:
                return &RawSymbolicExpression::CalcBopDiv;
            case SymbolicOpcode::T_BOP_MOD:
                return &RawSymbolicExpression::CalcBopMod;
            case SymbolicOpcode::T_BOP_EQ:
                return &RawSymbolicExpression::CalcBopEq;
            case SymbolicOpcode::T_BOP_NE:
                return &RawSymbolicExpression::CalcBopNe;
            case SymbolicOpcode::T_BOP_LT:
                return &RawSymbolicExpression::CalcBopLt;
            case SymbolicOpcode::T_BOP_LE:
                return &RawSymbolicExpression::CalcBopLe;
            case SymbolicOpcode::T_BOP_GT:
                return &RawSymbolicExpression::CalcBopGt;
            case SymbolicOpcode::T_BOP_GE:
                return &RawSymbolicExpression::CalcBopGe;
            case SymbolicOpcode::T_MOP_AND:
                return &RawSymbolicExpression::CalcBopAnd;
            case SymbolicOpcode::T_MOP_OR:
                return &RawSymbolicExpression::CalcBopOr;
            default:
                FE_ASSERT(false);
                return nullptr;
        }
    }

    using SymbolicCalcMop = ScalarImmediateType (*)(const std::vector<ScalarImmediateType>&);

    static ScalarImmediateType CalcMopMin(const std::vector<ScalarImmediateType>& immediateList)
    {
        FE_ASSERT(!immediateList.empty());
        return std::accumulate(immediateList.begin() + 1, immediateList.end(), immediateList[0],
                               [](const ScalarImmediateType& lhs, const ScalarImmediateType& rhs) {
                                   return RawSymbolicExpression::CalcBopMin(lhs, rhs);
                               });
    }

    static ScalarImmediateType CalcMopMax(const std::vector<ScalarImmediateType>& immediateList)
    {
        FE_ASSERT(!immediateList.empty());
        return std::accumulate(immediateList.begin() + 1, immediateList.end(), immediateList[0],
                               [](const ScalarImmediateType& lhs, const ScalarImmediateType& rhs) {
                                   return RawSymbolicExpression::CalcBopMax(lhs, rhs);
                               });
    }

    static SymbolicCalcMop GetSymbolicCalcMultiple(SymbolicOpcode opcode)
    {
        switch (opcode) {
            case SymbolicOpcode::T_MOP_MIN:
                return &RawSymbolicExpression::CalcMopMin;
            case SymbolicOpcode::T_MOP_MAX:
                return &RawSymbolicExpression::CalcMopMax;
            default:
                FE_ASSERT(false) << "Not a MOP extrema opcode: " << static_cast<size_t>(opcode);
                return nullptr;
        }
    }

    static std::string GetSymbolicCalcOpcode(SymbolicOpcode opcode)
    {
        static const std::string OPCODE_NAME_LIST[] = {
            "+",       "-",       "!",                        // T_UOP_POS / T_UOP_NEG / T_UOP_NOT
            "+",       "-",       "*",       "/",  "%",       // T_BOP_ADD .. T_BOP_MOD
            "==",      "!=",      "<",       "<=", ">", ">=", // T_BOP_EQ .. T_BOP_GE
            "<:min:>", "<:max:>",                             // deprecated T_BOP_MIN_DEPRECATED / T_BOP_MAX_DEPRECATED
            "",        "<:min:>", "<:max:>",                  // T_MOP_CALL / T_MOP_MIN / T_MOP_MAX (not rendered here)
            "&&",      "||",                                  // T_MOP_AND / T_MOP_OR
        };
        return OPCODE_NAME_LIST[static_cast<size_t>(opcode)];
    }

    static void Handle2NonzeroOperand(RawSymbolicScalarPtr& raw, SymbolicOpcode opcode,
                                      std::vector<RawSymbolicScalarPtr>& nonzeroOperandList)
    {
        constexpr int size2 = 2;
        FE_ASSERT(nonzeroOperandList.size() == size2)
            << "Lvalue: " << nonzeroOperandList.size() << ", Rvalue: " << size2;
        if (nonzeroOperandList[0]->IsImmediate()) {
            if (nonzeroOperandList[1]->IsImmediate()) {
                raw = std::make_shared<RawSymbolicImmediate>(
                    std::static_pointer_cast<RawSymbolicImmediate>(nonzeroOperandList[0])->Immediate() +
                    std::static_pointer_cast<RawSymbolicImmediate>(nonzeroOperandList[1])->Immediate());
            } else {
                std::swap(nonzeroOperandList[0], nonzeroOperandList[1]);
                raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
            }
        } else {
            if (!nonzeroOperandList[1]->IsImmediate() || !nonzeroOperandList[0]->IsExpression()) {
                raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
            } else {
                auto expression = std::static_pointer_cast<RawSymbolicExpression>(nonzeroOperandList[0]);
                if (expression->Opcode() == SymbolicOpcode::T_BOP_ADD) {
                    FE_ASSERT(expression->OperandList().size() == size2)
                        << "Lvalue: " << expression->OperandList().size() << ", Rvalue: " << size2;
                    if (!expression->OperandList()[1]->IsImmediate()) {
                        raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
                    } else {
                        auto operationList = expression->OperandList();
                        operationList[1] = std::make_shared<RawSymbolicImmediate>(
                            std::static_pointer_cast<RawSymbolicImmediate>(operationList[1])->Immediate() +
                            std::static_pointer_cast<RawSymbolicImmediate>(nonzeroOperandList[1])->Immediate());
                        raw = std::make_shared<RawSymbolicExpression>(opcode, operationList);
                    }
                } else {
                    raw = std::make_shared<RawSymbolicExpression>(opcode, nonzeroOperandList);
                }
            }
        }
    }

    static RawSymbolicScalarPtr CreateRuntimeExtrema(SymbolicOpcode opcode,
                                                     const std::vector<RawSymbolicScalarPtr>& operandList)
    {
        std::vector<RawSymbolicScalarPtr> flatOperands;
        flatOperands.reserve(operandList.size());
        FlattenOperands(operandList, opcode, flatOperands);

        bool hasImm = false;
        ScalarImmediateType immExt = 0;
        std::vector<RawSymbolicScalarPtr> nonImm;
        nonImm.reserve(flatOperands.size());
        std::unordered_set<std::string> seenStr;
        seenStr.reserve(flatOperands.size());

        auto combine = [&](ScalarImmediateType a, ScalarImmediateType b) {
            return (opcode == SymbolicOpcode::T_MOP_MAX) ? std::max(a, b) : std::min(a, b);
        };

        for (auto& operand : flatOperands) {
            if (operand->IsImmediate()) {
                auto value = std::static_pointer_cast<RawSymbolicImmediate>(operand)->Immediate();
                if (!hasImm) {
                    immExt = value;
                    hasImm = true;
                } else {
                    immExt = combine(immExt, value);
                }
                continue;
            }

            std::ostringstream oss;
            operand->DumpBuffer(oss);
            std::string tmpExpr = oss.str();
            if (seenStr.count(tmpExpr)) {
                continue;
            } else {
                seenStr.insert(std::move(tmpExpr));
                nonImm.emplace_back(operand);
            }
        }

        if (hasImm) {
            nonImm.emplace_back(std::make_shared<RawSymbolicImmediate>(immExt));
        }

        if (nonImm.empty()) {
            return std::make_shared<RawSymbolicImmediate>(immExt);
        } else if (nonImm.size() == 1) {
            return nonImm[0];
        } else {
            return std::make_shared<RawSymbolicExpression>(opcode, nonImm);
        }
    }

    static bool AllImmediate(const std::vector<RawSymbolicScalarPtr>& ops)
    {
        FE_ASSERT(!ops.empty());
        return std::all_of(ops.begin(), ops.end(), [](const RawSymbolicScalarPtr& op) { return op->IsImmediate(); });
    }

    static std::vector<ScalarImmediateType> ToImmediateList(const std::vector<RawSymbolicScalarPtr>& ops)
    {
        std::vector<ScalarImmediateType> imm(ops.size());
        std::transform(ops.begin(), ops.end(), imm.begin(), [](const RawSymbolicScalarPtr& v) {
            return std::static_pointer_cast<RawSymbolicImmediate>(v)->Immediate();
        });
        return imm;
    }

    static ScalarImmediateType FoldAllImmediate(SymbolicOpcode opcode,
                                                const std::vector<ScalarImmediateType>& immediateList)
    {
        if (SymbolicOpcode::T_UOP_BEGIN <= opcode && opcode < SymbolicOpcode::T_UOP_END) {
            FE_ASSERT(immediateList.size() == 1) << "immediateList.size():  " << immediateList.size();
            return RawSymbolicExpression::GetSymbolicCalcUnary(opcode)(immediateList[0]);
        } else if (RawSymbolicExpression::IsBinaryCalcOpcode(opcode)) {
            return std::accumulate(immediateList.begin() + 1, immediateList.end(), immediateList[0],
                                   [opcode](const ScalarImmediateType& lhs, const ScalarImmediateType& rhs) {
                                       return RawSymbolicExpression::GetSymbolicCalcBinary(opcode)(lhs, rhs);
                                   });
        } else if (opcode == SymbolicOpcode::T_MOP_MAX || opcode == SymbolicOpcode::T_MOP_MIN) {
            return RawSymbolicExpression::GetSymbolicCalcMultiple(opcode)(immediateList);
        }
        FE_ASSERT(false) << "undefined behavior.";
        return 0;
    }

    static RawSymbolicScalarPtr MakeAddWithZeroOpt(const std::vector<RawSymbolicScalarPtr>& ops)
    {
        std::vector<RawSymbolicScalarPtr> nonzero;
        nonzero.reserve(ops.size());
        for (auto& op : ops) {
            if (op->IsImmediate() && std::static_pointer_cast<RawSymbolicImmediate>(op)->Immediate() == 0) {
                continue;
            }
            nonzero.push_back(op);
        }
        if (nonzero.size() == 1)
            return nonzero[0];

        RawSymbolicScalarPtr raw;
        Handle2NonzeroOperand(raw, SymbolicOpcode::T_BOP_ADD, nonzero);
        return raw;
    }

    static RawSymbolicScalarPtr CreateRuntime(SymbolicOpcode opcode, const std::vector<RawSymbolicScalarPtr>& ops)
    {
        if (opcode == SymbolicOpcode::T_BOP_ADD) {
            return MakeAddWithZeroOpt(ops);
        } else if (opcode == SymbolicOpcode::T_MOP_MAX || opcode == SymbolicOpcode::T_MOP_MIN) {
            return CreateRuntimeExtrema(opcode, ops);
        } else {
            return std::make_shared<RawSymbolicExpression>(opcode, ops);
        }
    }

    static RawSymbolicScalarPtr Create(SymbolicOpcode opcode, const std::vector<RawSymbolicScalarPtr>& operandList)
    {
        if (AllImmediate(operandList)) {
            auto imm = ToImmediateList(operandList);
            auto result = FoldAllImmediate(opcode, imm);
            return std::make_shared<RawSymbolicImmediate>(result);
        }
        return CreateRuntime(opcode, operandList);
    }

#define RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(name, uop)                 \
    static RawSymbolicScalarPtr name(const RawSymbolicScalarPtr& val) \
    {                                                                 \
        RawSymbolicScalarPtr result = Create(uop, {val});             \
        return result;                                                \
    }
    RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(CreateUopPos, SymbolicOpcode::T_UOP_POS)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(CreateUopNeg, SymbolicOpcode::T_UOP_NEG)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP(CreateUopNot, SymbolicOpcode::T_UOP_NOT)
#undef RAW_SYMBOLIC_EXPRESSION_DEFINE_UOP
#define RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(name, bop)                                                  \
    static RawSymbolicScalarPtr name(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs) \
    {                                                                                                  \
        RawSymbolicScalarPtr result = Create(bop, {lhs, rhs});                                         \
        return result;                                                                                 \
    }
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopAdd, SymbolicOpcode::T_BOP_ADD)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopSub, SymbolicOpcode::T_BOP_SUB)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopMul, SymbolicOpcode::T_BOP_MUL)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopDiv, SymbolicOpcode::T_BOP_DIV)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopMod, SymbolicOpcode::T_BOP_MOD)

    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopEq, SymbolicOpcode::T_BOP_EQ)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopNe, SymbolicOpcode::T_BOP_NE)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopLt, SymbolicOpcode::T_BOP_LT)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopLe, SymbolicOpcode::T_BOP_LE)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopGt, SymbolicOpcode::T_BOP_GT)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP(CreateBopGe, SymbolicOpcode::T_BOP_GE)
#undef RAW_SYMBOLIC_EXPRESSION_DEFINE_BOP
#define RAW_SYMBOLIC_EXPRESSION_DEFINE_MOP(name, mop)                                   \
    static RawSymbolicScalarPtr name(const std::vector<RawSymbolicScalarPtr>& operands) \
    {                                                                                   \
        RawSymbolicScalarPtr result = Create(mop, operands);                            \
        return result;                                                                  \
    }
    RAW_SYMBOLIC_EXPRESSION_DEFINE_MOP(CreateMopMax, SymbolicOpcode::T_MOP_MAX)
    RAW_SYMBOLIC_EXPRESSION_DEFINE_MOP(CreateMopMin, SymbolicOpcode::T_MOP_MIN)
#undef RAW_SYMBOLIC_EXPRESSION_DEFINE_MOP

    static RawSymbolicScalarPtr CreateBopOr(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs)
    {
        std::vector<RawSymbolicScalarPtr> operands = {lhs, rhs};
        for (auto& op : operands) {
            if (op->IsImmediate() && std::static_pointer_cast<RawSymbolicImmediate>(op)->Immediate() != 0) {
                return std::make_shared<RawSymbolicImmediate>(1);
            }
        }
        return CreateRuntime(SymbolicOpcode::T_MOP_OR, operands);
    }

    static RawSymbolicScalarPtr CreateBopAnd(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs)
    {
        std::vector<RawSymbolicScalarPtr> operands = {lhs, rhs};
        for (auto& op : operands) {
            if (op->IsImmediate() && std::static_pointer_cast<RawSymbolicImmediate>(op)->Immediate() == 0) {
                return std::make_shared<RawSymbolicImmediate>(0);
            }
        }
        return CreateRuntime(SymbolicOpcode::T_MOP_AND, operands);
    }

    static RawSymbolicScalarPtr CreateMopCall(const RawSymbolicScalarPtr& callee)
    {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, {callee});
        return result;
    }
    static RawSymbolicScalarPtr CreateMopCall(const RawSymbolicScalarPtr& callee, const RawSymbolicScalarPtr& arg0)
    {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, {callee, arg0});
        return result;
    }
    static RawSymbolicScalarPtr CreateMopCall(const RawSymbolicScalarPtr& callee, const RawSymbolicScalarPtr& arg0,
                                              const RawSymbolicScalarPtr& arg1)
    {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, {callee, arg0, arg1});
        return result;
    }
    static RawSymbolicScalarPtr CreateMopCall(const std::vector<RawSymbolicScalarPtr>& calleeArgs)
    {
        RawSymbolicScalarPtr result = Create(SymbolicOpcode::T_MOP_CALL, calleeArgs);
        return result;
    }

private:
    void DumpRuntimeExtrema(std::ostream& buffer) const;
    void DumpBuffer(std::ostream& buffer) const override;

    SymbolicOpcode opcode_;
    std::vector<RawSymbolicScalarPtr> operandList_;
};

using RawSymbolicExpPtr = std::shared_ptr<RawSymbolicExpression>;

inline std::string AddRuntimePrefix(const std::string& name) { return SPECIAL_SYMBOL_NAME_RUNTIME_PREFIX + name; }
inline bool CheckRuntimePrefix(const std::string& name)
{
    return StringUtils::StartsWith(name, SPECIAL_SYMBOL_NAME_RUNTIME_PREFIX);
}
inline std::string AddRuntimeCoaPrefix(const std::string& name)
{
    return SPECIAL_SYMBOL_NAME_RUNTIME_COA_PREFIX + name;
}
inline std::string AddArgPrefix(const std::string& name) { return SPECIAL_SYMBOL_NAME_ARG_PREFIX + name; }
inline bool CheckArgPrefix(const std::string& name)
{
    return StringUtils::StartsWith(name, SPECIAL_SYMBOL_NAME_ARG_PREFIX);
}

static inline bool CallIsGetTensorData(const std::string& name)
{
    return StringUtils::StartsWith(name, AddRuntimePrefix("GetTensorData"));
}

static inline bool CallIsGetInputData(const std::string& name)
{
    return StringUtils::StartsWith(name, AddRuntimePrefix("GetInputData"));
}

static inline bool CallIsGetInputShapeDim(const std::string& name)
{
    return name == AddRuntimePrefix("GetInputShapeDim") || name == AddRuntimePrefix("GetInputShapeDimSize");
}

Json ToJson(const SymbolicScalar& sval);

SymbolicScalar LoadSymbolicScalar(const Json& jval);

} // namespace npu::tile_fwk

namespace npu::tile_fwk {
class SymbolicClosure {
public:
    typedef std::unordered_map<std::string, ScalarImmediateType> SymbolValueDictType;
    std::shared_ptr<SymbolValueDictType> symbolValueDict;

    SymbolicClosure() { symbolValueDict = std::make_shared<SymbolValueDictType>(); }
    SymbolicClosure(const SymbolValueDictType& symbolDict)
    {
        symbolValueDict = std::make_shared<SymbolValueDictType>();
        for (auto& [name, value] : symbolDict) {
            Insert(name, value);
        }
    }
    virtual ~SymbolicClosure() = default;
    void Insert(const std::string& name, ScalarImmediateType val) { symbolValueDict->insert({name, val}); }
    void Remove(const std::string& name) { symbolValueDict->erase(name); }
    bool Has(const std::string& name) const { return symbolValueDict->count(name) != 0; }
    ScalarImmediateType Get(const std::string& name) const { return symbolValueDict->find(name)->second; }

    ScalarImmediateType Evaluate(const SymbolicScalar& scalar) const { return Evaluate(scalar.Raw()); }

protected:
    virtual ScalarImmediateType EvaluateExpressionCall(const RawSymbolicScalarPtr ptr) const
    {
        (void)(ptr);
        FE_ASSERT(false) << " expression call is not supported";
        return 0;
    }
    ScalarImmediateType EvaluateExpression(const RawSymbolicExpPtr& expr) const
    {
        if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
            return EvaluateExpressionCall(std::static_pointer_cast<RawSymbolicScalar>(expr));
        }

        std::vector<ScalarImmediateType> dataList;
        for (auto& operand : expr->OperandList()) {
            dataList.emplace_back(Evaluate(operand));
        }
        ScalarImmediateType result = 0;
        if (SymbolicOpcode::T_UOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_UOP_END) {
            result = RawSymbolicExpression::GetSymbolicCalcUnary(expr->Opcode())(dataList[0]);
        } else if (RawSymbolicExpression::IsBinaryCalcOpcode(expr->Opcode())) {
            result = dataList[0];
            for (size_t i = 1; i < dataList.size(); i++) {
                result = RawSymbolicExpression::GetSymbolicCalcBinary(expr->Opcode())(result, dataList[i]);
            }
        } else if (expr->Opcode() == SymbolicOpcode::T_MOP_MAX || expr->Opcode() == SymbolicOpcode::T_MOP_MIN) {
            return RawSymbolicExpression::GetSymbolicCalcMultiple(expr->Opcode())(dataList);
        } else {
            FE_ASSERT(false) << " unknown opcode: " << expr->Dump();
        }
        return result;
    }

    ScalarImmediateType Evaluate(const RawSymbolicScalarPtr& raw) const
    {
        ScalarImmediateType result{INVALID_SCALAR_IMMEDIATE};
        switch (raw->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
                auto immediate = std::dynamic_pointer_cast<RawSymbolicImmediate>(raw);
                result = immediate->Immediate();
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
                auto symbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
                FE_ASSERT(symbolValueDict->count(symbol->Name()))
                    << symbol->Name() << " has not been found in symbolValueDict";
                result = symbolValueDict->find(symbol->Name())->second;
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
                RawSymbolicExpPtr expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
                result = EvaluateExpression(expr);
            } break;
            default:
                FE_ASSERT(false) << " undefined behavior.";
                break;
        }
        return result;
    }
};

struct SymbolicSymbolTable {
    OrderedSet<std::string> symbolTable_;
    std::unordered_map<std::string, RawSymbolicScalarPtr> symbolTableDict_;

    void AddSymbolFromExpression(const SymbolicScalar& ss)
    {
        if (ss.Raw()->IsImmediate()) {
            return;
        }
        AddAllSymbol(ss.Raw());
    }

    const OrderedSet<std::string>& GetSymbolTable() const { return symbolTable_; }
    const std::unordered_map<std::string, RawSymbolicScalarPtr>& GetSymbolTableDict() const { return symbolTableDict_; }

    void AddSymbol(const SymbolicScalar& ss)
    {
        auto raw = ss.Raw();
        AddSymbol(raw);
    }

    void AddSymbol(const RawSymbolicScalarPtr& raw)
    {
        FE_ASSERT(raw->Kind() == SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL)
            << "raw->Kind(): " << SymbolicScalarKind2Name(raw->Kind());
        std::string name = raw->GetSymbolName();
        if (symbolTable_.count(name)) {
            return;
        }
        symbolTable_.Insert(name);
        symbolTableDict_[name] = raw;
    }

    void NormalizeForSymbol()
    {
        std::set<std::string> nameSet;
        for (auto& [name, ss] : symbolTableDict_) {
            (void)ss;
            nameSet.insert(name);
        }
        symbolTable_.Clear();
        for (auto& name : nameSet) {
            symbolTable_.Insert(name);
        }
    }

    std::string Dump() const
    {
        std::ostringstream oss;
        for (auto& name : symbolTable_) {
            oss << "name:" << name << "\n";
        }
        return oss.str();
    }

    std::string BuildSymbolList() const
    {
        std::ostringstream oss;
        for (size_t index = 0; index < GetSymbolTable().size(); index++) {
            std::string name = GetSymbolTable()[index];
            oss << "\n"
                << "#define SYMBOLIC_INDEX_" << name << " " << index << "\n"
                << "#define VALUE_" << name << " (RUNTIME_GetSymbol(SYMBOLIC_INDEX_" << name << "))\n";
        }
        return oss.str();
    }

private:
    void AddAllSymbol(const RawSymbolicScalarPtr& raw)
    {
        switch (raw->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
                break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
                AddSymbol(raw);
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
                RawSymbolicExpPtr expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
                for (auto& operand : expr->OperandList()) {
                    AddAllSymbol(operand);
                }
            } break;
            default:
                FE_ASSERT(false) << SymbolicScalarKind2Name(raw->Kind()) << " undefined behavior";
                break;
        }
    }
};

struct SymbolicExpressionTable {
    OrderedSet<RawSymbolicScalarPtr> expressionSet;
    std::map<std::string, RawSymbolicScalarPtr> primaryExpressionDict_;
    OrderedSet<RawSymbolicScalarPtr> primaryExpressionSet;
    std::string elementKey_;
    std::string title_;
    SymbolicScalar mainBlockScalar_;

    void SetElementKeyOnce(const std::string& key);
    void SetTitleOnce(const std::string& title);

    void AddPrimaryExpression(const SymbolicScalar& ss)
    {
        if (ss.IsImmediate()) {
            return;
        }
        std::string str = BuildExpressionByRaw(ss.Raw(), {});
        if (primaryExpressionDict_.count(str)) {
            return;
        }
        primaryExpressionDict_.emplace(str, ss.Raw());
        primaryExpressionSet.Insert(ss.Raw());
        AddExpression(ss.Raw());
    }

    void NormalizeForSymbolTable(const SymbolicSymbolTable& symbolTable)
    {
        primaryExpressionSet.Clear();
        primaryExpressionSet.Insert(mainBlockScalar_.Raw());
        auto symTable = symbolTable.GetSymbolTable();
        auto symExprTable = symbolTable.GetSymbolTableDict();
        FE_ASSERT(symTable.size() == symExprTable.size())
            << "Lvalue: " << symTable.size() << ", Rvalue: " << symExprTable.size();
        auto symOrder = symTable.GetOrder();
        for (auto& sym : symOrder) {
            RawSymbolicScalarPtr symbol;
            if (CheckRuntimePrefix(sym)) {
                symbol = std::make_shared<RawSymbolicImmediate>(0);
            } else {
                symbol = symExprTable.find(sym)->second;
            }
            primaryExpressionSet.Insert(symbol);
        }
        // 按结构而非字符串排序，使等差家族在表中相邻，便于后续循环折叠。
        std::vector<RawSymbolicScalarPtr> sortedPrimaryExprs;
        sortedPrimaryExprs.reserve(primaryExpressionDict_.size());
        for (auto& [str, expr] : primaryExpressionDict_) {
            (void)str;
            sortedPrimaryExprs.emplace_back(expr);
        }
        std::sort(
            sortedPrimaryExprs.begin(), sortedPrimaryExprs.end(),
            [](const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs) { return CompareRaw(lhs, rhs) < 0; });
        for (auto& expr : sortedPrimaryExprs) {
            primaryExpressionSet.Insert(expr);
        }
    }

    int LookupPrimaryExpressionIndex(const SymbolicScalar& ss) const
    {
        std::string str = BuildExpressionByRaw(ss.Raw(), {});
        FE_ASSERT(primaryExpressionDict_.count(str)) << str << " has not been found in primaryExpressionDict_.";
        auto raw = primaryExpressionDict_.find(str)->second;
        FE_ASSERT(primaryExpressionSet.count(raw)) << raw << " has not been found in primaryExpressionSet.";
        return primaryExpressionSet.GetIndex(raw);
    }

    int GetPrimaryExpressionSize() const { return primaryExpressionSet.size(); }
    const OrderedSet<RawSymbolicScalarPtr>& GetPrimaryExpressionSet() const { return primaryExpressionSet; }

    std::string Dump() const
    {
        std::ostringstream oss;
        for (auto& expr : primaryExpressionDict_) {
            oss << "expr:" << BuildExpressionByRaw(expr.second, {}) << "\n";
        }
        return oss.str();
    }

    std::string BuildExpressionList() const;
    std::string BuildExpressionTempVarInit(int indent);

    static std::string GetExprKeyLoopBes(int funcKey) { return "EXPR_LOOP_BES_" + std::to_string(funcKey); }
    static std::string GetExprKeyLoopPathCond(int funcKey, int condKey)
    {
        return "EXPR_LOOP_PATHCOND_" + std::to_string(funcKey) + "_" + std::to_string(condKey);
    }
    static std::string GetExprKeyDevRootCoa(int funcKey) { return "EXPR_DEV_ROOT_COA_" + std::to_string(funcKey); }
    static std::string GetExprKeyDevLeafOp(int funcKey, int opKey)
    {
        return "EXPR_DEV_LEAF_OP_" + std::to_string(funcKey) + "_" + std::to_string(opKey);
    }

    static std::string GetExprNameTempVarFlag(const std::string& exprKey, int index)
    {
        return exprKey + "_" + std::to_string(index) + "_TEMP_VAR_FLAG";
    }
    static std::string GetExprNameTempVar(const std::string& exprKey, int index)
    {
        return exprKey + "_" + std::to_string(index) + "_TEMP_VAR";
    }
    static std::string GetExprNameTempVarInit(const std::string& exprKey, int index)
    {
        return exprKey + "_" + std::to_string(index) + "_TEMP_INIT";
    }
    static std::string GetExprNameCalc(const std::string& exprKey, int index)
    {
        return exprKey + "_" + std::to_string(index) + "_CALC";
    }
    static std::string GetExprNameUse(const std::string& exprKey, int index)
    {
        return exprKey + "_" + std::to_string(index) + "_USE";
    }

    // structuralCse: canonical BuildExpression text -> C identifier (e.g. CSE_sd[i]).
    // Lookup only runs on GetInputShapeDim / GetInputData CALL nodes; other nodes expand as usual.
    static std::string BuildExpressionByRaw(
        const RawSymbolicScalarPtr& raw, const std::unordered_map<RawSymbolicScalarPtr, std::string>& exprDict,
        const std::unordered_map<std::string, std::string>* structuralCse = nullptr);
    static std::string BuildExpression(const SymbolicScalar& ss);
    static std::string BuildExpression(const RawSymbolicScalarPtr& ss);
    static std::string BuildExpression(const RawSymbolicScalarPtr& ss,
                                       const std::unordered_map<std::string, std::string>* structuralCse);

    // 结构化三路比较，strcmp 语义。
    static int CompareRaw(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs);

    // 差异叶子：路径 + 两侧 Immediate 值。
    struct ImmediateDiff {
        std::vector<int> path;
        int64_t immLhs;
        int64_t immRhs;
    };

    // 在 raw 上按一组 (path, placeholderNode) 替换 Immediate 叶子并渲染整个表达式。
    static std::string BuildExpressionWithPlaceholders(
        const RawSymbolicScalarPtr& raw,
        const std::vector<std::pair<std::vector<int>, RawSymbolicScalarPtr>>& replacements,
        const std::unordered_map<std::string, std::string>* structuralCse = nullptr);

    // 找出 lhs / rhs 之间所有结构对齐、仅在 Immediate 叶子上数值不同的位置。
    // 结构不同（Kind / Opcode / Operand 数 / Symbol 名）返回 false。
    static bool FindAllImmediateDifferences(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs,
                                            std::vector<ImmediateDiff>& diffs);

    static bool CheckExprDependCore(const RawSymbolicScalarPtr& ss,
                                    const std::unordered_map<std::string, bool>& tensorNameToDependCore,
                                    std::unordered_map<RawSymbolicScalarPtr, bool>& valDependMap);

private:
    static std::string BuildSymbolName(const std::string& name);
    static void BuildExtremaExpressionCode(const RawSymbolicExpPtr& expr,
                                           const std::unordered_map<RawSymbolicScalarPtr, std::string>& exprDict,
                                           std::ostringstream& oss,
                                           const std::unordered_map<std::string, std::string>* structuralCse);
    static std::string BuildExpressionCode(const RawSymbolicExpPtr& expr,
                                           const std::unordered_map<RawSymbolicScalarPtr, std::string>& exprDict,
                                           const std::unordered_map<std::string, std::string>* structuralCse);

    // FindAllImmediateDifferences 的递归实现。
    static bool CollectImmediateDifferences(const RawSymbolicScalarPtr& lhs, const RawSymbolicScalarPtr& rhs,
                                            std::vector<int>& currentPath, std::vector<ImmediateDiff>& diffs);

    void AddExpression(const RawSymbolicScalarPtr& raw)
    {
        switch (raw->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            } break;
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
                for (auto& operand : raw->GetExpressionOperandList()) {
                    AddExpression(operand);
                }
                expressionSet.Insert(raw);
            } break;
            default:
                FE_ASSERT(false) << SymbolicScalarKind2Name(raw->Kind()) << " undefined behavior";
                break;
        }
    }
};

std::vector<uint8_t> CompileAndLoadSection(const std::string& code, const std::string& sourceFilePath,
                                           const std::string& aicpuPath, std::vector<std::string>& exprSrcFiles,
                                           const std::string& gcc, const std::string& ld, const std::string& objcopy,
                                           const std::string& sectionName, bool needDump,
                                           const std::string& extraCflag = "");

void CompileAndLink(const std::string& code, const std::string& sourceFilePath, const std::string& gcc,
                    bool isStaticLink, bool isBenchmark, bool useMakefile);

std::string CompileCopyLink(const std::string& code, const std::string& sourceFilePath, const std::string& gcc,
                            const std::string& objcopy, bool isStaticLink, bool isBenchmark,
                            const std::map<std::string, std::string>& sectionDataDict);

void RunMake(const std::string& makefilePath);

std::vector<RawSymbolicScalarPtr> LookupExpressionByOpcode(const RawSymbolicScalarPtr& value, SymbolicOpcode opcode);

} // namespace npu::tile_fwk
