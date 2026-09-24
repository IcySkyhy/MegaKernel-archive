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
 * \file loop_unroll.h
 * \brief
 */

#ifndef PASS_LOOP_UNROLL_H_
#define PASS_LOOP_UNROLL_H_

#include <vector>
#include <map>
#include <climits>
#include <unordered_map>
#include <set>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "interface/cache/hash.h"
#include "interface/utils/id_gen.h"
#include "interface/machine/host/host_machine.h"

namespace npu {
namespace tile_fwk {
class LoopUnroll : public Pass {
public:
    LoopUnroll() : Pass("LoopUnroll"), topFunction_(nullptr) {}
    ~LoopUnroll() override = default;

private:
    std::vector<std::string> staticFuncNames_;
    std::shared_ptr<EvaluateSymbol> evaluateSymbol_;
    std::unordered_map<int, std::pair<LogicalTensorPtr, bool>> lastWriteMap_;
    Function* topFunction_;
    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar& ss)
    {
        return evaluateSymbol_->EvaluateSymbolicScalar(ss);
    }

    Status RunOnFunction(Function& function) override;

    Status GetCallee(const Operation* callop, Function*& callFunc);
    std::vector<SymbolicScalar> ConvertToSymbolicScalar(std::vector<int64_t> staticShape);
    Status MapLocalTensorToGlobal(const LogicalTensors& localTensor, LogicalTensors& globalTensor,
                                  std::unordered_map<int, LogicalTensorPtr> tensorLocal2Global);
    Status AddNewOperation(Operation* localOp, const std::unordered_map<int, LogicalTensorPtr> tensorLocal2Global,
                           std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap,
                           std::unordered_map<Operation*, std::vector<int64_t>> opDynShapeMap);
    Status UpdateCloneOpAttributes(Operation* localOp, Operation* cloneOp,
                                   std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap,
                                   std::unordered_map<Operation*, std::vector<int64_t>> opDynShapeMap);
    void UpdateOutTensorDynAttributes(Operation* originalOp, Operation* clonedOp,
                                      std::unordered_map<Operation*, std::vector<int64_t>>& opDynOffsetMap,
                                      std::unordered_map<Operation*, std::vector<int64_t>>& opDynShapeMap);
    void DeriveTensorStaticAttributes(LogicalTensorPtr tensor, EvaluateSymbol& evaluator,
                                      std::vector<int64_t>& staticShape);
    void EvaluateDynamicOpParams(Operation* op, EvaluateSymbol& evaluator,
                                 std::unordered_map<Operation*, std::vector<int64_t>>& opDynOffsetMap,
                                 std::unordered_map<Operation*, std::vector<int64_t>>& opDynShapeMap);
    Operation* ExecuteFunctionLoopLookupSat(const std::shared_ptr<DynloopFunctionAttribute>& controlFlowExecution);
    Status ExpandDynamicLoop(Operation* callop);
    Status ExpandDynamicFunction(Operation* callop);
    Status TopFunctionUnroll(Function* function, std::vector<Operation*> callopList);
    Status UpdateTopFuncInoutCast(Function* function);
    Status TraverseCallOp(Function* function);
    bool IsConvertingToStatic(Function* function);

    void UpdateGlobalTensorWAW();
    Status CreateGlobalTensor(std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap,
                              std::unordered_map<int, LogicalTensorPtr>& tensor2Global, const Operation* op,
                              Function* curFunc);
    bool IsWARDepend(const int slotIdx, std::set<LogicalTensorPtr> input2Global);
    void FindSlotDepend(const Operation* op, std::set<LogicalTensorPtr> input2Global, bool& isDepend);
    bool IsNoOverlapWAW(int slotIdx, LogicalTensorPtr tensor,
                        std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap);
    bool IsTensorOverlap(std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& tensors);
    bool IsOverlapping(std::pair<std::vector<int64_t>, std::vector<int64_t>> tensor1,
                       std::pair<std::vector<int64_t>, std::vector<int64_t>> tensor2);
    Status FindOutputGlobalTensor(int slotIdx, std::unordered_map<int, LogicalTensorPtr>& tensor2Global,
                                  std::set<LogicalTensorPtr> input2Global, LogicalTensorPtr tensor,
                                  std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap);
    Status FindInputGlobalTensor(int slotIdx, std::unordered_map<int, LogicalTensorPtr>& tensor2Global,
                                 LogicalTensorPtr tensor);
    Status CreateLocal2Global(std::unordered_map<int, LogicalTensorPtr>& tensor2Global, LogicalTensorPtr tensor);
    Function* CreateLoopFunc(Function* func, Function* callerParentFunc);
    Status CreateLoopUnrollFunc(Function* function);
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_LOOP_UNROLL_H_
