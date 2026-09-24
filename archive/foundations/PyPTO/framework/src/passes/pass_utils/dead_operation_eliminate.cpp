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
 * \file dead_operation_eliminate.cpp
 * \brief
 */

#include "dead_operation_eliminate.h"
#include <queue>
#include <chrono>
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

Status DeadOperationEliminator::EliminateDeadOperation(Function& function)
{
    DeadOperationEliminator eliminator;
    eliminator.EliminateDeadOperationBackward(function);
    return SUCCESS;
}

// Delete Operation without oOperand
void DeadOperationEliminator::EliminateDeadOperationBackward(Function& function)
{
    for (auto& op : function.Operations(false)) {
        op.SetAsNotDeleted();
    }
    EliminateOperation(function);
}

std::set<Operation*, LogicalTensor::CompareOp> FindProducers(
    Operation& op, std::unordered_set<std::shared_ptr<LogicalTensor>>& visitedOperands, Function& function)
{
    std::set<Operation*, LogicalTensor::CompareOp> producerOps;
    for (const auto& input : op.iOperand) {
        if (visitedOperands.count(input) != 0) {
            continue;
        }
        for (const auto& producer : input->GetProducers()) {
            if (producer->BelongTo() == &function) {
                producerOps.emplace(producer);
            }
        }
        visitedOperands.emplace(input);
    }
    return producerOps;
}

void DeadOperationEliminator::EliminateOperation(Function& function, bool useSortedOperations, bool sortAfterErase)
{
    std::queue<Operation*> q;
    std::unordered_set<Operation*> visited;
    std::unordered_set<std::shared_ptr<LogicalTensor>> visitedOperands;
    for (auto& op : function.Operations(useSortedOperations, SortOperationsMode::LIGHTWEIGHT)) {
        bool dontTouch = op.GetBoolAttribute(OpAttributeKey::dontTouch);
        if (dontTouch) {
            visited.emplace(&op);
            q.emplace(&op);
        }
    }

    for (auto& outcast : function.GetOutcast()) {
        for (auto op : outcast->GetProducers()) {
            if (visited.count(op) != 0) {
                continue;
            }
            visited.emplace(op);
            q.emplace(op);
        }
    }
    while (!q.empty()) {
        auto op = q.front();
        q.pop();
        std::set<Operation*, LogicalTensor::CompareOp> producerOps = FindProducers(*op, visitedOperands, function);
        for (const auto& producerOp : producerOps) {
            if (visited.count(producerOp) != 0) {
                continue;
            }
            visited.emplace(producerOp);
            q.emplace(producerOp);
        }
    }
    for (auto& op : function.Operations(false)) {
        if (visited.count(&op) == 0) {
            op.SetAsDeleted();
        }
    }
    function.EraseOperations(false, sortAfterErase, SortOperationsMode::LIGHTWEIGHT);
}
} // namespace npu::tile_fwk
