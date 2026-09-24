/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ir_backend.h
 * \brief New SCF IR based control flow building functions.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>

#include "machine/host/backend.h"
#include "machine/host/expr_generator.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "interface/tensor/irbuilder.h"

namespace npu::tile_fwk {

struct IrBackendContext {
    std::unordered_map<const pypto::ir::ForStmt*, std::shared_ptr<Function>> forStmtLoopFuncMap;
    // Align with legacy EmitControlFlowSources: launch-level GetInputShapeDim CSE.
    const GetInputCse* getInputCse{nullptr};
};

SymbolicScalar ExprPtrToSymbolicScalar(const ir::ExprPtr& expr);
Function* ResolveCalleeFromOpCall(const ir::StmtPtr& stmt);
bool IsOpCallStmt(const ir::StmtPtr& stmt);

Function* IrBuildVirtualLoopFunc(IrBackendContext& ctx, const ir::ForStmt* forStmt, Function* dynFunc);
void IrParseValueDependDesc(Function* func, std::initializer_list<ir::ExprPtr> exprs);
void InsertCacheStopForContrlFlow(IrBackendContext& ctx, const ir::ForStmt* forStmt, Function* dynFunc,
                                  ValDependTensorMeta& valDependTensorMeta);
void InsertWaitAicoreStartForControlFlow(const ir::ForStmt* forStmt, int indent, std::ostringstream& controlFlowOss,
                                         ValDependTensorMeta& valDependTensorMeta);
void VisitForStmtForControlFlow(IrBackendContext& ctx, FunctionCache& cache, Linker& linker,
                                const std::string& sectionName, const ir::ForStmtPtr& forStmt, Function* dynFunc,
                                std::unordered_map<int, int>& slotIdxMapping,
                                DyndevFunctionAttribute::FunctionGroup& group,
                                std::unordered_map<Function*, Function*>& rootTileDict,
                                std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                                std::ostringstream& exprHeaderOss, int indent, const std::string& expName,
                                std::vector<std::string>& exprSrcFiles, ValDependTensorMeta& valDependTensorMeta);

void FindAllExpressionFromIR(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, Function* func);
void FindExprFromIRStmt(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, const ir::StmtPtr& stmt,
                        Function* dynFunc, std::vector<ir::ExprPtr>& condStack);

void VisitIRStmtForControlFlow(IrBackendContext& ctx, FunctionCache& cache, Linker& linker,
                               const std::string& sectionName, const ir::StmtPtr& stmt, Function* dynFunc,
                               std::unordered_map<int, int>& slotIdxMapping,
                               DyndevFunctionAttribute::FunctionGroup& group,
                               std::unordered_map<Function*, Function*>& rootTileDict,
                               std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                               std::ostringstream& exprHeaderOss, int indent, const std::string& expName,
                               std::vector<std::string>& exprSrcFiles, ValDependTensorMeta& valDependTensorMeta,
                               const std::vector<ir::VarPtr>& currentReturnVars = {});

void BuildControlFlowFromIR(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, const std::string& sectionName,
                            Function* func, std::unordered_map<int, int>& slotIdxMapping,
                            DyndevFunctionAttribute::FunctionGroup& group,
                            std::unordered_map<Function*, Function*>& rootTileDict, std::ostringstream& controlFlowOss,
                            std::ostringstream& expressionOss, std::ostringstream& exprHeaderOss, int indent,
                            const std::string& expName, std::vector<std::string>& exprSrcFiles,
                            ValDependTensorMeta& valDependTensorMeta);

} // namespace npu::tile_fwk
