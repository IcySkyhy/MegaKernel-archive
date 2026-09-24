/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir.h"
#include "ir/kind_traits.h"
#include "ir/type.h"
#include "ir/transforms/passes.h"
#include "ir/transforms/utils/dead_code_elimination.h"

#include "interface/program/program.h"
#include "interface/function/function.h"

#include "irbuilder.h"
#include "symbolic_scalar.h"
#include "logical_tensor.h"
#include "token_pass.h"
#include "ir_func_builder.h"
#include "ir_finalize.h"

using npu::tile_fwk::LogicalTensor;
using npu::tile_fwk::Operation;
using npu::tile_fwk::RawSymbolicExpression;

namespace pypto::ir {
std::string DumpScalarExpr(const ScalarExprPtr& op)
{
    auto p = std::dynamic_pointer_cast<const RawSymbolicExpression>(op);
    ASSERT(p) << "not a RawSymbolicExpression";
    return p->Dump();
}

std::string DumpTensorVar(const VarPtr& var)
{
    auto t = std::dynamic_pointer_cast<const LogicalTensor>(var);
    ASSERT(t) << "not a logical tensor";
    return var->name_ + "@" + std::to_string(t->tensor->memoryId);
}

std::map<std::string, std::any> CollectTensorOpAttrs(const TensorOpStmtPtr& top)
{
    std::map<std::string, std::any> attrs;

    auto op = std::dynamic_pointer_cast<Operation>(std::const_pointer_cast<TensorOpStmt>(top));
    if (op && op->GetOpAttribute()) {
        auto opAttr = op->GetOpAttribute();
        opAttr->CollectAttrs(attrs);
    }
    return attrs;
}

std::optional<int> GetVarMemoryId(const Var* var)
{
    auto tensor = dynamic_cast<const LogicalTensor*>(var);
    if (tensor && tensor->GetRawTensor()) {
        return tensor->GetRawTensor()->memoryId;
    }
    return std::nullopt;
}

std::unordered_set<int> CollectMemoryIds(const std::unordered_set<const Var*>& vars)
{
    std::unordered_set<int> memory_ids;
    for (const auto* var : vars) {
        if (auto mid = GetVarMemoryId(var)) {
            memory_ids.insert(*mid);
        }
    }
    return memory_ids;
}

void CollectScalarVarRefs(const TensorOpStmtPtr& op, std::unordered_set<const Var*>& var_uses)
{
    auto operation = std::dynamic_pointer_cast<Operation>(std::const_pointer_cast<TensorOpStmt>(op));
    if (operation) {
        for (auto& attr : operation->GetDynamicAttributeList()) {
            attr.get().GetVarRefs(var_uses);
        }
    }
}

bool IsSameRawTensor(const VarPtr& v0, const VarPtr& v1)
{
    if (ir::IsA<LogicalTensorType>(v0->GetType()) && ir::IsA<LogicalTensorType>(v1->GetType())) {
        auto lt1 = std::dynamic_pointer_cast<const LogicalTensor>(v0);
        auto lt2 = std::dynamic_pointer_cast<const LogicalTensor>(v1);
        return lt1->GetRawTensor() == lt2->GetRawTensor();
    }
    return false;
}

Pass pass::AggressiveDCE()
{
    return pass::CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            // Collect RawTensors from function input parameters.
            std::unordered_set<int> memIds;
            for (const auto& param : func->params_) {
                if (auto type = ir::As<LogicalTensorType>(param->GetType())) {
                    auto lt = std::dynamic_pointer_cast<const LogicalTensor>(param);
                    memIds.insert(lt->tensor->memoryId);
                }
            }

            auto isRemovable = [&memIds](const StmtPtr& stmt) -> bool {
                // TensorOpStmt: written to input slot cannot be removed
                if (auto tensorOp = std::dynamic_pointer_cast<const TensorOpStmt>(stmt)) {
                    // pass_verify_print / pass_verify_save (OP_PRINT, IR opcode "OP_DUMP") have no
                    // result tensors but are side-effect ops and must survive DCE.
                    if (tensorOp->opcode_ == "OP_DUMP") {
                        return false;
                    }
                    for (auto arg : tensorOp->result_) {
                        auto lt = std::dynamic_pointer_cast<const LogicalTensor>(arg);
                        if (memIds.count(lt->tensor->memoryId)) {
                            return false;
                        }
                    }
                    return true;
                }
                if (auto assignOp = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
                    return true;
                }
                if (auto scalarOp = std::dynamic_pointer_cast<const ScalarOpStmt>(stmt)) {
                    return true;
                }
                // Everything else (ReturnStmt, YieldStmt, control flow, EvalStmt): not removable
                return false;
            };

            auto newStmts = dce::EliminateDeadCode(func->body_->stmts_, isRemovable);
            if (newStmts.size() == func->body_->stmts_.size()) {
                bool changed = false;
                for (size_t i = 0; i < newStmts.size(); ++i) {
                    if (newStmts[i].get() != func->body_->stmts_[i].get()) {
                        changed = true;
                        break;
                    }
                }
                if (!changed)
                    return func;
            }
            auto newBody = std::make_shared<ir::SeqStmts>(std::move(newStmts), func->body_->span_);
            return std::make_shared<ir::Function>(func->name_, func->params_, func->returnTypes_, newBody, func->span_,
                                                  func->funcType_, func->entry_, func->attrs_);
        },
        "AggressiveDCE");
}

Pass pass::TokenPass()
{
    return pass::CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            npu::tile_fwk::TokenPass transform;
            return transform(func);
        },
        "TokenPass");
}

Pass pass::CreateRootFunctions()
{
    return pass::CreateProgramPass(
        [](const ProgramPtr& irProgram) -> ProgramPtr {
            auto& programInst = npu::tile_fwk::Program::GetInstance();
            programInst.ResetTensorSlotManager();
            auto parentFunc = programInst.GetLastFunction();

            std::map<std::string, FunctionPtr> newFunctions;
            const auto funcMapOld = programInst.GetFunctionMap();
            std::vector<std::shared_ptr<npu::tile_fwk::Function>> dynFuncs;
            for (const auto& [funcName, irFunc] : irProgram->functions_) {
                (void)funcName;
                npu::tile_fwk::RootFunctionBuilder builder(parentFunc);
                auto dynFunc = builder.Build(irFunc);
                dynFunc->entry_ = true;
                newFunctions[dynFunc->name_] = std::static_pointer_cast<const ir::Function>(dynFunc);
                dynFuncs.push_back(dynFunc);
            }

            const auto& funcMap = programInst.GetFunctionMap();
            for (const auto& [k, v] : funcMap) {
                if (funcMapOld.find(k) == funcMapOld.end()) {
                    newFunctions[k] = std::static_pointer_cast<const ir::Function>(v);
                }
            }

            for (auto& dynFunc : dynFuncs) {
                programInst.InsertFuncToFunctionMap(dynFunc->GetMagicName(), dynFunc);
            }

            return std::make_shared<const ir::Program>(std::move(newFunctions), irProgram->name_, irProgram->span_);
        },
        "CreateRootFunctions");
}

namespace {

npu::tile_fwk::Function* AsFrameworkFunction(const FunctionPtr& func)
{
    auto fwkFunc = std::static_pointer_cast<const npu::tile_fwk::Function>(func);
    return const_cast<npu::tile_fwk::Function*>(fwkFunc.get());
}

} // namespace

Pass pass::FinalizeDynamicFunction()
{
    return pass::CreateProgramPass(
        [](const ProgramPtr& irProgram) -> ProgramPtr {
            auto& programInst = npu::tile_fwk::Program::GetInstance();
            for (const auto& [funcName, funcPtr] : irProgram->functions_) {
                (void)funcName;
                if (!funcPtr->entry_) {
                    continue;
                }
                auto* fwkFunc = AsFrameworkFunction(funcPtr);
                if (fwkFunc == nullptr) {
                    continue;
                }
                programInst.SetLastFunction(fwkFunc);
                npu::tile_fwk::FinalizeDynamicFunction(fwkFunc);
            }
            return irProgram;
        },
        "FinalizeDynamicFunction");
}

} // namespace pypto::ir
