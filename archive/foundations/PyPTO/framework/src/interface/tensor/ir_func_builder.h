/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/function.h"
#include "ir/stmt.h"
#include "ir/transforms/base/mutator.h"
#include "interface/tensor/tensor_slot.h"

using namespace pypto;

namespace npu::tile_fwk {

class ConfigScope;
class Function;
class LogicalTensor;
class Program;
class TensorSlotManager;
class TensorSlotScope;
class Tensor;
class Operation;
class StmtTransformer;
using LogicalTensorPtr = std::shared_ptr<LogicalTensor>;
using LogicalTensors = std::vector<LogicalTensorPtr>;

class RootFunctionBuilder {
public:
    friend class StmtTransformer;
    explicit RootFunctionBuilder(Function* parentFunc);
    std::shared_ptr<Function> Build(const ir::FunctionPtr& irFunc);

private:
    Program& program_;
    Function* parentFunc_;
    std::shared_ptr<Function> dynFunc_;
    LogicalTensors logicalParams_;
    std::unordered_set<LogicalTensorPtr> consumedTensors_;
    std::unordered_set<LogicalTensorPtr> paramTensors_;
    std::unordered_map<std::string, int> loopNameCounters_;
    std::unordered_map<LogicalTensorPtr, Function*> tensorDefineFunc_;
    std::unordered_map<LogicalTensorPtr, Function*> tensorConstructFunc_;

    void InitDynFunc(const ir::FunctionPtr& irFunc);
    void FinalizeDynFunc(const ir::FunctionPtr& irFunc);
    void CollectInOutCastFromReturn();

    ir::StmtPtr TransformBody(ir::StmtPtr stmt);
    void ReplacePlaceholders(ir::StmtPtr stmt);
    void LinkReturnSlots(const ir::StmtPtr& stmt);
    void LinkControlFlowSlots(const ir::StmtPtr& stmt);
    void LinkForStmtSlots(const ir::ForStmt& forStmt);
    void LinkIfStmtSlots(const ir::IfStmt& ifStmt);

    ir::StmtPtr CreatePathFuncAndPlaceholder(const ir::SeqStmtsPtr& seq, const std::string& loopVarName);
    ir::StmtPtr FinalizePathFunc(const ir::StmtPtr& placeholder);

    std::shared_ptr<Function> CreateHiddenFunc(const ir::SeqStmtsPtr& seq, const std::string& loopVarName);
    std::pair<LogicalTensors, LogicalTensors> FinalizeHiddenFunc(Function* hiddenFunc, const ir::StmtPtr& placeholder);
    void AddHiddenFuncValueDepend(Function* hiddenFunc);
    void CreateAndFinalizePathFunc(Function* pathFunc, Function* hiddenFunc, const LogicalTensors& hiddenInArgs,
                                   const LogicalTensors& hiddenOutArgs, const ir::StmtPtr& placeholder);
    ir::StmtPtr ProcessTensorOp(std::shared_ptr<Function> pathFunc, const ir::StmtPtr& stmt,
                                std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
                                std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs,
                                std::unordered_map<int, std::shared_ptr<RawTensor>>& memIdMap);
    void RecordDefinedTensors(const ir::TensorOpStmtPtr& tensorOpStmt, Function* pathFunc);
    void MigrateInplaceLinks(const std::shared_ptr<const Operation>& sourceOp, const LogicalTensors& oOperands,
                             Function* pathFunc);
    void CollectAssembleConsumed(const std::shared_ptr<const Operation>& sourceOp, const LogicalTensors& oOperands,
                                 Function* pathFunc);
    void ComputeIncast(Function& pathFunc, const std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
                       const std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs);
    void ComputeOutcast(Function& pathFunc);

    void BuildDynSlotScope();
    void BuildPathFuncSlotScope(Function* pathFunc, const std::shared_ptr<TensorSlotScope>& scope,
                                const LogicalTensors& originalIncasts, const LogicalTensors& originalOutcasts);
    void FlushConstructAssembleSlots();

    bool IsPureTensorOpSeq(const ir::SeqStmtsPtr& seq);
    std::vector<std::vector<ir::StmtPtr>> SplitIntoTensorOpSegments(const ir::SeqStmtsPtr& seq);
    bool IsPlaceholderCallStmt(const ir::StmtPtr& stmt);
    std::string GetPlaceholderFuncname(const ir::StmtPtr& stmt);
    void DumpFunctionGraph(Function* func);
};

} // namespace npu::tile_fwk
