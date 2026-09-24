/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class IRVisitor;

#define PYPTO_IR_PRINTER_VISIT_EXPR(Type) void VisitExpr_(const Type& op) override;
#define PYPTO_IR_PRINTER_VISIT_STMT(Type) void VisitStmt_(const Type& op) override;
#define PYPTO_IR_PRINTER_VISIT_FUNCTION(Type) void VisitFunction(const Type& func) override;
#define PYPTO_IR_PRINTER_VISIT_PROGRAM(Type) void VisitProgram(const Type& program) override;

#define PYPTO_IR_PRINTER_COMMON_VISITOR_OVERRIDES() \
    PYPTO_IR_PRINTER_VISIT_EXPR(VarPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(MemRefPtr)          \
    PYPTO_IR_PRINTER_VISIT_EXPR(ConstIntPtr)        \
    PYPTO_IR_PRINTER_VISIT_EXPR(ConstFloatPtr)      \
    PYPTO_IR_PRINTER_VISIT_EXPR(ConstBoolPtr)       \
    PYPTO_IR_PRINTER_VISIT_EXPR(CallPtr)            \
    PYPTO_IR_PRINTER_VISIT_EXPR(MakeTuplePtr)       \
    PYPTO_IR_PRINTER_VISIT_EXPR(GetItemExprPtr)     \
    PYPTO_IR_PRINTER_VISIT_EXPR(AddPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(SubPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(MulPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(FloorDivPtr)        \
    PYPTO_IR_PRINTER_VISIT_EXPR(FloorModPtr)        \
    PYPTO_IR_PRINTER_VISIT_EXPR(FloatDivPtr)        \
    PYPTO_IR_PRINTER_VISIT_EXPR(MinPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(MaxPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(PowPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(EqPtr)              \
    PYPTO_IR_PRINTER_VISIT_EXPR(NePtr)              \
    PYPTO_IR_PRINTER_VISIT_EXPR(LtPtr)              \
    PYPTO_IR_PRINTER_VISIT_EXPR(LePtr)              \
    PYPTO_IR_PRINTER_VISIT_EXPR(GtPtr)              \
    PYPTO_IR_PRINTER_VISIT_EXPR(GePtr)              \
    PYPTO_IR_PRINTER_VISIT_EXPR(AndPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(OrPtr)              \
    PYPTO_IR_PRINTER_VISIT_EXPR(XorPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(BitAndPtr)          \
    PYPTO_IR_PRINTER_VISIT_EXPR(BitOrPtr)           \
    PYPTO_IR_PRINTER_VISIT_EXPR(BitXorPtr)          \
    PYPTO_IR_PRINTER_VISIT_EXPR(BitShiftLeftPtr)    \
    PYPTO_IR_PRINTER_VISIT_EXPR(BitShiftRightPtr)   \
    PYPTO_IR_PRINTER_VISIT_EXPR(AbsPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(NegPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(NotPtr)             \
    PYPTO_IR_PRINTER_VISIT_EXPR(BitNotPtr)          \
    PYPTO_IR_PRINTER_VISIT_EXPR(CastPtr)            \
    PYPTO_IR_PRINTER_VISIT_STMT(AssignStmtPtr)      \
    PYPTO_IR_PRINTER_VISIT_STMT(IfStmtPtr)          \
    PYPTO_IR_PRINTER_VISIT_STMT(YieldStmtPtr)       \
    PYPTO_IR_PRINTER_VISIT_STMT(ReturnStmtPtr)      \
    PYPTO_IR_PRINTER_VISIT_STMT(ForStmtPtr)         \
    PYPTO_IR_PRINTER_VISIT_STMT(WhileStmtPtr)       \
    PYPTO_IR_PRINTER_VISIT_STMT(SeqStmtsPtr)        \
    PYPTO_IR_PRINTER_VISIT_STMT(EvalStmtPtr)        \
    PYPTO_IR_PRINTER_VISIT_STMT(BreakStmtPtr)       \
    PYPTO_IR_PRINTER_VISIT_STMT(ContinueStmtPtr)    \
    PYPTO_IR_PRINTER_VISIT_STMT(StmtPtr)            \
    PYPTO_IR_PRINTER_VISIT_FUNCTION(FunctionPtr)    \
    PYPTO_IR_PRINTER_VISIT_PROGRAM(ProgramPtr)

enum class Precedence : int {
    kOr = 1,
    kXor = 2,
    kAnd = 3,
    kNot = 4,
    kComparison = 5,
    kBitOr = 6,
    kBitXor = 7,
    kBitAnd = 8,
    kBitShift = 9,
    kAddSub = 10,
    kMulDivMod = 11,
    kUnary = 12,
    kPow = 13,
    kCall = 14,
    kAtom = 15
};

Precedence GetPrecedence(const ExprPtr& expr);
bool IsRightAssociative(const ExprPtr& expr);
bool NeedsParensForPrint(const ExprPtr& parent, const ExprPtr& child, bool is_left);
void PrintIRNodeWithVisitor(IRVisitor& visitor, std::ostream& stream, const IRNodePtr& node);
void PrintChildExprWithParens(IRVisitor& visitor, std::ostream& stream, const ExprPtr& parent, const ExprPtr& child,
                              bool is_left);
void PrintReturnStmtValues(IRVisitor& visitor, std::ostream& stream, const std::vector<ExprPtr>& values);
void PrintFunctionReturnAnnotation(std::ostream& stream, const std::vector<TypePtr>& return_types,
                                   const std::function<std::string(const TypePtr&)>& print_type);

std::string PythonPrint(const IRNodePtr& node, const std::string& prefix = "ir", bool concise = false);
std::string PythonPrint(const TypePtr& type, const std::string& prefix = "ir");

} // namespace ir
} // namespace pypto
