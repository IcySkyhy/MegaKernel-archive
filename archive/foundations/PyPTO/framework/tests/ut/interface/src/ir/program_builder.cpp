/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */
#include "program_builder.h"

#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/stmt.h"
#include "ir/transforms/structural_comparison.h"

namespace npu::tile_fwk {
constexpr const char* ENTRY = "__entry__";
struct ProgramBuilder::Impl {
    ir::SeqStmtsPtr PushInsertPoint()
    {
        auto stmt = std::make_shared<ir::SeqStmts>(span_);
        builder_.SetInsertPoint(std::make_shared<ir::InsertPoint>(stmt));
        return stmt;
    }

    void PopInsertPoint() { builder_.ClearInsertPoint(); }

    void Emit(ir::StmtPtr stmt)
    {
        builder_.EmitTensorStmts();
        builder_.Emit(stmt);
    }

    void TensorOp(const std::string& op, std::vector<ir::ExprPtr> inputs, std::vector<ir::VarPtr> outputs)
    {
        Emit(builder_.CreateTensorOpStmt(std::move(outputs), nullptr, op, std::move(inputs), {}, {}, span_));
    }

    ir::SeqStmtsPtr RunBranch(const std::function<void()>& fn)
    {
        auto block = PushInsertPoint();
        fn();
        builder_.EmitTensorStmts();
        PopInsertPoint(); // restore enclosing block so the parent receives the next Emit
        return block;
    }

    bool SameType(ir::ExprPtr a, ir::ExprPtr b)
    {
        bool ret = ir::structural_equal(a->GetType(), b->GetType());
        if (ret) {
            if (ir::As<ir::LogicalTensorType>(b->GetType())) {
                auto ta = std::dynamic_pointer_cast<LogicalTensor>(std::const_pointer_cast<ir::Expr>(a));
                auto tb = std::dynamic_pointer_cast<LogicalTensor>(std::const_pointer_cast<ir::Expr>(b));
                if (ta && tb) {
                    return TypeEqual(ta, tb);
                }
            }
        }
        return ret;
    }

    void Checkpoint() { Program::GetInstance().GetTensorSlotManager()->Checkpoint(); }

    void Restore() { Program::GetInstance().GetTensorSlotManager()->Restore(); }

    IRBuilder builder_;
    std::vector<std::reference_wrapper<const Tensor>> inputs_;
    std::vector<LogicalTensorPtr> params_;
    ir::Span span_ = ir::Span::Unknown();
    ir::SeqStmtsPtr body_;
    std::string name_;

}; // namespace npu::tile_fwk

ProgramBuilder::ProgramBuilder() : impl_(std::make_unique<Impl>()) {}

ProgramBuilder::~ProgramBuilder() = default;

void ProgramBuilder::BeginFunction(const std::string& name, std::vector<std::reference_wrapper<const Tensor>> inputs)
{
    impl_->name_ = name;
    impl_->inputs_ = std::move(inputs);
    for (auto& p : impl_->inputs_) {
        impl_->params_.push_back(p.get().GetStorage(false));
    }
    impl_->body_ = impl_->PushInsertPoint();
    Program::GetInstance().Reset();
    Program::GetInstance().BeginFunction(FUNCTION_PREFIX + ENTRY, FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH,
                                         impl_->inputs_);
}

ir::ProgramPtr ProgramBuilder::EndFunction()
{
    std::vector<ir::ExprPtr> exprs;
    for (auto& p : impl_->inputs_) {
        exprs.push_back(p.get().GetStorage(false));
    }
    impl_->Emit(impl_->builder_.CreateReturnStmt(std::move(exprs), impl_->span_));

    impl_->PopInsertPoint();
    Program::GetInstance().EndFunction(FUNCTION_PREFIX + ENTRY, false);

    std::vector<ir::VarPtr> irParams;
    for (auto& p : impl_->params_) {
        irParams.push_back(std::static_pointer_cast<const ir::Var>(p));
    }
    auto func = std::make_shared<ir::Function>(impl_->name_, std::move(irParams), std::vector<ir::TypePtr>{},
                                               impl_->body_, impl_->span_);
    return std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{func}, "entry", impl_->span_);
}

Tensor ProgramBuilder::Alloc(DataType dtype, std::vector<int64_t> shape, std::string name)
{
    auto t = Tensor(dtype, std::move(shape), std::move(name));
    impl_->TensorOp("TENSOR_ALLOC", {}, {t.GetStorage(false)});
    return t;
}

std::vector<ir::ExprPtr> ProgramBuilder::If(SymbolicScalar cond, std::function<void()> thenFn,
                                            std::function<void()> elseFn, const char* file, int line)
{
    impl_->Checkpoint();
    auto thenBody = impl_->RunBranch(thenFn);
    impl_->Restore();

    impl_->Checkpoint();
    auto elseBody = impl_->RunBranch(elseFn);
    impl_->Restore();

    auto thenYield = ir::As<ir::YieldStmt>(thenBody->stmts_.back());
    auto elseYield = ir::As<ir::YieldStmt>(elseBody->stmts_.back());

    ASSERT(thenYield) << "missing yield stmt for if branch file: " << file << " line: " << line;
    ASSERT(elseYield) << "missing yield stmt for else branch file: " << file << " line: " << line;
    ASSERT(thenYield->value_.size() == elseYield->value_.size());

    std::vector<ir::VarPtr> rets;
    for (size_t i = 0; i < thenYield->value_.size(); ++i) {
        std::string name = "if_ret";
        if (auto v = ir::As<ir::Var>(thenYield->value_[i])) {
            name = v->name_;
        }
        if (impl_->SameType(thenYield->value_[i], elseYield->value_[i])) {
            rets.push_back(impl_->builder_.CreateVarLike(name, thenYield->value_[i]));
        } else if (ir::IsA<ir::NoneType>(thenYield->value_[i]->GetType())) {
            rets.push_back(impl_->builder_.CreateVarLike(name, elseYield->value_[i]));
        } else if (ir::IsA<ir::NoneType>(elseYield->value_[i]->GetType())) {
            rets.push_back(impl_->builder_.CreateVarLike(name, thenYield->value_[i]));
        } else {
            rets.push_back(impl_->builder_.CreateVarLike(name, thenYield->value_[i]));
        }
    }
    auto stmt = impl_->builder_.CreateIfStmt(cond.AsExpr(), std::move(thenBody),
                                             std::optional<ir::StmtPtr>{std::move(elseBody)}, rets,
                                             ir::Span(file, line, 0));
    impl_->Emit(stmt);
    return std::vector<ir::ExprPtr>(rets.begin(), rets.end());
}

std::vector<ir::VarPtr> ProgramBuilder::For(SymbolicScalar start, SymbolicScalar stop, SymbolicScalar step,
                                            std::vector<std::pair<std::string, std::reference_wrapper<Tensor>>> carries,
                                            std::function<void(SymbolicScalar, const std::vector<ir::VarPtr>&)> body,
                                            const char* file, int line)
{
    std::vector<ir::VarPtr> carryVars;
    std::vector<ir::IterArgPtr> iterArgs;

    auto idx = impl_->builder_.CreateScalarVar("idx");
    for (auto& [name, init] : carries) {
        auto t = init.get().GetStorage(false);
        auto cv = impl_->builder_.CreateVarLike(name, t);
        carryVars.push_back(cv);
        iterArgs.push_back(impl_->builder_.CreateIterArg(cv, t));
    }

    auto bodyStmt = impl_->RunBranch([&]() { body(idx, carryVars); });

    std::vector<ir::VarPtr> rets;
    std::vector<ir::ExprPtr> values;
    if (!bodyStmt->stmts_.empty()) {
        auto last = bodyStmt->stmts_.back();
        if (auto y = ir::As<ir::YieldStmt>(last)) {
            values = y->value_;
        } else if (auto c = ir::As<ir::ContinueStmt>(last)) {
            values = c->value_;
        } else {
            ASSERT(false) << "missing yield/continue stmt for for branch file: " << file << " line: " << line;
        }
        for (size_t i = 0; i < values.size() && i < carries.size(); ++i) {
            rets.push_back(impl_->builder_.CreateVarLike(carries[i].first, values[i]));
        }
    }
    auto forStmt = impl_->builder_.CreateForStmt(idx.AsVar(), start.AsExpr(), stop.AsExpr(), step.AsExpr(),
                                                 std::move(iterArgs), bodyStmt, rets, ir::Span(file, line, 0));
    impl_->Emit(forStmt);
    return rets;
}

void ProgramBuilder::Yield(std::vector<ir::ExprPtr> value)
{
    impl_->Emit(impl_->builder_.CreateYieldStmt(std::move(value), impl_->span_));
}

void ProgramBuilder::Continue(std::vector<ir::ExprPtr> value)
{
    impl_->Emit(impl_->builder_.CreateContinueStmt(std::move(value), impl_->span_));
}

Tensor ProgramBuilder::AsTensor(const ir::ExprPtr& e)
{
    auto lt = std::dynamic_pointer_cast<LogicalTensor>(std::const_pointer_cast<ir::Expr>(e));
    ASSERT(lt) << "AsTensor: expression is not backed by a LogicalTensor";
    return Tensor(std::move(lt));
}

SymbolicScalar ProgramBuilder::AsSymbol(const ir::ExprPtr& e) { return SymbolicScalar::FromExpr(e); }

ir::ExprPtr ProgramBuilder::Unwrap(const Tensor& t) { return t.GetStorage(false); }
ir::ExprPtr ProgramBuilder::Unwrap(SymbolicScalar s) { return s.AsExpr(); }
ir::ExprPtr ProgramBuilder::Unwrap(const ir::ExprPtr& e) { return e; }
ir::ExprPtr ProgramBuilder::Unwrap(int x) { return SymbolicScalar(x).AsExpr(); }

} // namespace npu::tile_fwk
