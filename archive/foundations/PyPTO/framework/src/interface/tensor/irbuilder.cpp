/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "irbuilder.h"

#include "logical_tensor.h"
#include "raw_tensor.h"
#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/program/program.h"
#include "interface/utils/id_gen.h"

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/type.h"
#include "ir/transforms/printer.h" // for test
#include "ir/transforms/infer_token_pass.h"

using namespace pypto;

namespace npu::tile_fwk {

IRContext& IRContext::Get()
{
    static IRContext ctx;
    return ctx;
}

std::vector<ir::VarPtr>& IRContext::GetDependToken(const ir::ExprPtr& val)
{
    static std::vector<ir::VarPtr> empty;
    if (token_map_.find(val) != token_map_.end()) {
        return token_map_[val];
    }
    if (ir::As<ir::ScalarExpr>(val) == nullptr) {
        return empty;
    }

    auto getToken = [this](const RawSymbolicScalarPtr& s) -> std::vector<ir::VarPtr> {
        if (s->IsExpression()) {
            auto ss = std::static_pointer_cast<RawSymbolicExpression>(s);
            auto tokens = GetDependToken(ss);
            return tokens;
        }
        return empty;
    };

    std::set<ir::VarPtr> tokenList;
    auto s = std::dynamic_pointer_cast<const RawSymbolicExpression>(val);
    auto start = s->Opcode() == SymbolicOpcode::T_MOP_CALL;
    for (size_t i = start; i < s->OperandList().size(); i++) {
        auto tokens = getToken(s->OperandList()[i]);
        tokenList.insert(tokens.begin(), tokens.end());
    }
    token_map_[val] = std::vector<ir::VarPtr>(tokenList.begin(), tokenList.end());
    return token_map_[val];
}

std::vector<ir::VarPtr> IRContext::GetDependToken(Operation& op)
{
    std::set<ir::VarPtr> tokenSet;
    for (auto& scalar : op.GetDynamicAttributeList()) {
        auto token = GetDependToken(scalar.get().AsExpr());
        tokenSet.insert(token.begin(), token.end());
    }
    std::vector<ir::VarPtr> tokenList(tokenSet.begin(), tokenSet.end());
    std::sort(tokenList.begin(), tokenList.end(), [](ir::VarPtr a, ir::VarPtr b) { return a->name_ < b->name_; });
    return tokenList;
}

ir::VarPtr IRContext::MakeSemanticToken(std::string name, ir::TokenKind kind, ir::Span span)
{
    FE_ASSERT(FeError::INVALID_VAL, kind == ir::TokenKind::READ || kind == ir::TokenKind::WRITE)
        << "MakeSemanticToken requires READ or WRITE kind";
    auto semantic = MakeVar(std::move(name), ir::GetTokenType(kind), span);
    auto normal = MakeVar(semantic->name_ + "_n", ir::GetTokenType(ir::TokenKind::NORMAL), span);
    semantic_to_normal_.emplace(semantic, normal);
    return semantic;
}

ir::VarPtr IRContext::GetNormalToken(const ir::VarPtr& semantic) const
{
    if (!semantic) {
        return nullptr;
    }
    auto it = semantic_to_normal_.find(semantic);
    return it != semantic_to_normal_.end() ? it->second : nullptr;
}

Function& DummyFunc()
{
    static auto func = []() {
        auto funcId = IdGen<IdType::FUNCTION>::Inst().CurId();
        auto dummyFunc = std::make_unique<Function>(Program::GetInstance(), "Dummy", "Dummy", nullptr);
        IdGen<IdType::FUNCTION>::Inst().SetId(funcId);
        return dummyFunc;
    }();
    return *func;
}

IRBuilder::IRBuilder() : irContext_(IRContext::Get()) {}

LogicalTensorPtr IRBuilder::CreateTensorVar(DataType t, Shape shape, TileOpFormat format, std::string name)
{
    return std::make_shared<LogicalTensor>(DummyFunc(), t, std::move(shape), format, name);
}

LogicalTensorPtr IRBuilder::CreateTensorVar(DataType t, Shape shape, std::vector<SymbolicScalar> validShape,
                                            TileOpFormat format, std::string name)
{
    return std::make_shared<LogicalTensor>(DummyFunc(), t, std::move(shape), std::move(validShape), format, name);
}

LogicalTensorPtr IRBuilder::CreateTensorVar(std::shared_ptr<RawTensor> rawTensor, Offset offset, Shape shape,
                                            std::vector<SymbolicScalar> validShape)
{
    LogicalTensorPtr tensor;
    if (validShape.empty()) {
        tensor = std::make_shared<LogicalTensor>(DummyFunc(), std::move(rawTensor), std::move(offset),
                                                 std::move(shape));
    } else {
        tensor = std::make_shared<LogicalTensor>(DummyFunc(), std::move(rawTensor), std::move(offset), std::move(shape),
                                                 std::move(validShape));
    }
    return tensor;
}

LogicalTensorPtr IRBuilder::CreateTensorVar(Function& f, DataType t, Shape shape, TileOpFormat format, std::string name)
{
    return std::make_shared<LogicalTensor>(f, t, std::move(shape), format, name);
}

LogicalTensorPtr IRBuilder::CreateTensorVar(Function& f, DataType t, Shape shape,
                                            std::vector<SymbolicScalar> validShape, TileOpFormat format,
                                            std::string name)
{
    return std::make_shared<LogicalTensor>(f, t, std::move(shape), std::move(validShape), format, name);
}

LogicalTensorPtr IRBuilder::CreateTensorVar(Function& f, std::shared_ptr<RawTensor> rawTensor, Offset offset,
                                            Shape shape, std::vector<SymbolicScalar> validShape)
{
    LogicalTensorPtr tensor;
    if (validShape.empty()) {
        tensor = std::make_shared<LogicalTensor>(f, std::move(rawTensor), std::move(offset), std::move(shape));
    } else {
        tensor = std::make_shared<LogicalTensor>(f, std::move(rawTensor), std::move(offset), std::move(shape),
                                                 std::move(validShape));
    }
    return tensor;
}

Operation& IRBuilder::CreateTensorOpStmt(Function& f, const Opcode opCode, const LogicalTensors& iOperands,
                                         const LogicalTensors& oOperands, ir::Span span)
{
    return f.AddRawOperation(opCode, iOperands, oOperands, span);
}

std::shared_ptr<RawTensor> IRBuilder::CreateRawTensor(DataType t, Shape shape, TileOpFormat format, std::string name)
{
    auto magic = IdGen<IdType::RAW_TENSOR>::Inst().NewId();
    return std::make_shared<RawTensor>(t, std::move(shape), format, std::move(name), magic);
}

std::shared_ptr<RawTensor> IRBuilder::CreateRawTensor(DataType t, std::vector<SymbolicScalar> dynShape,
                                                      TileOpFormat format, std::string name)
{
    auto shape = SymbolicScalar::Concrete(dynShape, -1);
    auto magic = IdGen<IdType::RAW_TENSOR>::Inst().NewId();
    auto raw = std::make_shared<RawTensor>(t, shape, format, std::move(name), magic);
    raw->UpdateDynRawShape(dynShape);
    return raw;
}

ir::TensorOpStmtPtr IRBuilder::CreateTensorOpStmt(std::vector<ir::VarPtr> result, ir::VarPtr result_token,
                                                  std::string opcode, std::vector<ir::ExprPtr> args,
                                                  std::vector<ir::VarPtr> tokens,
                                                  std::vector<std::pair<std::string, std::any>> attrs, ir::Span span)
{
    return std::make_shared<ir::TensorOpStmt>(result, result_token, opcode, args, tokens, attrs, span);
}

/* create symbolic scalar */
SymbolicScalar IRBuilder::CreateConstInt(int64_t value) { return SymbolicScalar(value); }

SymbolicScalar IRBuilder::CreateScalarVar(std::string sym)
{
    auto name = irContext_.GetVarName(sym);
    return SymbolicScalar(name);
}

ir::VarPtr IRBuilder::CreateVarLike(std::string name, ir::ExprPtr value)
{
    if (auto type = ir::As<ir::LogicalTensorType>(value->GetType())) {
        auto t = std::dynamic_pointer_cast<const LogicalTensor>(value);
        auto var = CreateTensorVar(DummyFunc(), t->tensor, t->offset, t->shape, t->GetDynValidShape());
        var->name_ = irContext_.GetVarName(name);
        return var;
    }
    if (auto type = ir::As<ir::ScalarType>(value->GetType())) {
        return CreateScalarVar(name).AsVar();
    }
    if (auto type = ir::As<ir::NoneType>(value->GetType())) {
        return irContext_.MakeVar(name, ir::GetNoneType(), value->span_);
    }
    if (auto tuple = ir::As<ir::MakeTuple>(value)) {
        std::vector<ir::ExprPtr> elements;
        for (size_t i = 0; i < tuple->elements_.size(); i++) {
            elements.push_back(CreateVarLike(name + std::to_string(i), tuple->elements_[i]));
        }
        return irContext_.MakeVar(name, ir::GetUnknownType(), value->span_);
    }
    ASSERT(false) << "CreateVarLike: unknown type" << value->GetType()->TypeName();
    return nullptr;
}

/* scf statement */
ir::AssignStmtPtr IRBuilder::CreateAssignStmt(ir::VarPtr var, ir::ExprPtr value, ir::Span span)
{
    return std::make_shared<ir::AssignStmt>(var, value, span);
}

ir::SeqStmtsPtr IRBuilder::CreateSeqStmts(std::vector<ir::StmtPtr> stmts, ir::Span span)
{
    return std::make_shared<ir::SeqStmts>(stmts, span);
}

ir::IfStmtPtr IRBuilder::CreateIfStmt(ir::ExprPtr cond, ir::StmtPtr thenBody, std::optional<ir::StmtPtr> elseBody,
                                      std::vector<ir::VarPtr> returnVars, ir::Span span)
{
    return std::make_shared<ir::IfStmt>(cond, thenBody, elseBody, returnVars, span);
}

ir::YieldStmtPtr IRBuilder::CreateYieldStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::YieldStmt>(values, span);
}

ir::ReturnStmtPtr IRBuilder::CreateReturnStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::ReturnStmt>(values, span);
}

ir::ForStmtPtr IRBuilder::CreateForStmt(ir::VarPtr loopVar, ir::ExprPtr start, ir::ExprPtr stop, ir::ExprPtr step,
                                        std::vector<ir::IterArgPtr> iterArgs, ir::StmtPtr body,
                                        std::vector<ir::VarPtr> returnVars, ir::Span span,
                                        std::vector<std::pair<std::string, std::any>> attrs)
{
    return std::make_shared<ir::ForStmt>(loopVar, start, stop, step, std::move(iterArgs), body, returnVars, span,
                                         std::move(attrs));
}

ir::WhileStmtPtr IRBuilder::CreateWhileStmt(ir::ExprPtr cond, std::vector<ir::IterArgPtr> iterArgs, ir::StmtPtr body,
                                            std::vector<ir::VarPtr> returnVars, ir::Span span)
{
    return std::make_shared<ir::WhileStmt>(cond, iterArgs, body, returnVars, span);
}

ir::BreakStmtPtr IRBuilder::CreateBreakStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::BreakStmt>(values, span);
}

ir::ContinueStmtPtr IRBuilder::CreateContinueStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::ContinueStmt>(values, span);
}

ir::FunctionPtr IRBuilder::CreateFunction(std::string name, std::vector<ir::VarPtr> params,
                                          std::vector<ir::TypePtr> returnTypes, ir::StmtPtr body, ir::Span span)
{
    return std::make_shared<ir::Function>(name, params, returnTypes, body, span);
}

ir::ProgramPtr IRBuilder::CreateProgram(std::vector<ir::FunctionPtr> functions, std::string name, ir::Span span)
{
    return std::make_shared<ir::Program>(functions, name, span);
}

ir::VarPtr IRBuilder::CreateTokenVar(ir::Span span) { return irContext_.MakeTempVar(ir::GetTokenType(), span); }

ir::ExprPtr IRBuilder::None()
{
    static auto none = irContext_.MakeVar("None", ir::GetNoneType(), ir::Span::Unknown());
    return none;
}

ir::IterArgPtr IRBuilder::CreateIterArg(std::string name, ir::TypePtr type, ir::ExprPtr initValue, ir::Span span)
{
    return std::make_shared<ir::IterArg>(name, type, initValue, span);
}

ir::IterArgPtr IRBuilder::CreateIterArg(ir::VarPtr var, ir::ExprPtr initValue)
{
    return std::make_shared<ir::IterArg>(var, initValue);
}

void IRBuilder::EmitTensorStmts()
{
    auto func = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "__entry__");
    for (auto& op : func->Operations(false)) {
        auto stmt = std::dynamic_pointer_cast<ir::TensorOpStmt>(op.shared_from_this());
        stmt->span_ = ir::Span::Current();
        stmt->tokens_ = irContext_.GetDependToken(op);
        Emit(stmt);
    }
    func->ResetOperations();
}

void IRBuilder::Checkpoint() { Program::GetInstance().GetTensorSlotManager()->Checkpoint(); }

void IRBuilder::Restore() { Program::GetInstance().GetTensorSlotManager()->Restore(); }
} // namespace npu::tile_fwk
