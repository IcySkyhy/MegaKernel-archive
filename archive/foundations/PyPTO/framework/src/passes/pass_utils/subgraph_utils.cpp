/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file subgraph_utils.cpp
 * \brief
 */

#include <unordered_set>
#include "subgraph_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
namespace {
bool HasCrossSubgraphEdges(const LogicalTensor& tensor)
{
    int expectedSubgraphId = -1;

    auto check_op = [&](Operation* op) {
        if (op == nullptr)
            return false;
        int opSubgraphId = op->GetSubgraphID();
        if (expectedSubgraphId == -1) {
            expectedSubgraphId = opSubgraphId;
        } else if (expectedSubgraphId != opSubgraphId) {
            return true;
        }
        return false;
    };

    for (const auto& op : tensor.GetProducers()) {
        if (check_op(op))
            return true;
    }
    for (const auto& op : tensor.GetConsumers()) {
        if (check_op(op))
            return true;
    }
    return false;
}

bool IsBoundaryAsInput(const LogicalTensor& tensor, const Operation& op)
{
    constexpr int kDistCopyInBoundaryIndex = 1;

    if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        return !op.GetIOperands().empty() && op.GetIOperands().front().get() == &tensor;
    }
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        return !op.GetOOperands().empty() &&
               op.GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    }

    bool isDistCopyOut = false;
    if (op.GetAttr<bool>(OpAttributeKey::isDistCopyOut, isDistCopyOut) && !isDistCopyOut) {
        return op.GetIOperands().size() > kDistCopyInBoundaryIndex &&
               op.GetIOperands()[kDistCopyInBoundaryIndex].get() == &tensor;
    }

    return false;
}

bool IsBoundaryAsOutput(const LogicalTensor& tensor, const Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        return !op.HasAttribute(OpAttributeKey::inplaceIdx) && !op.GetOOperands().empty() &&
               op.GetOOperands().front().get() == &tensor;
    }
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        return tensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    }
    if (op.GetOpcode() == Opcode::OP_RESHAPE) {
        return !op.GetIOperands().empty() &&
               op.GetIOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
               tensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    }

    bool isDistCopyOut = false;
    if (op.GetAttr<bool>(OpAttributeKey::isDistCopyOut, isDistCopyOut) && isDistCopyOut) {
        return !op.GetOOperands().empty() && op.GetOOperands().front().get() == &tensor;
    }

    // 针对通信算子 OP_SHMEM_SIGNAL 处理，在图切分时，其输出 tensor 作为子图边界
    if (op.GetOpcode() == Opcode::OP_SHMEM_SIGNAL) {
        return !op.GetOOperands().empty() && op.GetOOperands().front().get() == &tensor;
    }

    return false;
}

bool IsBaseBoundary(const LogicalTensor& tensor)
{
    if (tensor.GetProducers().empty() && !tensor.GetConsumers().empty()) {
        return true;
    }

    // 跨子图边缘
    if (HasCrossSubgraphEdges(tensor)) {
        return true;
    }

    // 当 Tensor 作为输入时
    for (const auto& op : tensor.GetConsumers()) {
        if (op == nullptr) {
            continue;
        }
        if (IsBoundaryAsInput(tensor, *op)) {
            return true;
        }
    }

    // 当 Tensor 作为输出时
    for (const auto& op : tensor.GetProducers()) {
        if (op == nullptr) {
            continue;
        }
        if (IsBoundaryAsOutput(tensor, *op)) {
            return true;
        }
    }

    return false;
}

bool IsBoundaryImpl(const LogicalTensor& tensor, std::unordered_set<const LogicalTensor*>& visited)
{
    if (!visited.insert(&tensor).second) {
        return false;
    }

    if (IsBaseBoundary(tensor)) {
        return true;
    }

    // Reshape边界属性传播
    for (const auto& op : tensor.GetProducers()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE && !op->GetIOperands().empty()) {
            if (IsBoundaryImpl(*(op->GetIOperands().front()), visited))
                return true;
        }
    }
    for (const auto& op : tensor.GetConsumers()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE && !op->GetOOperands().empty()) {
            if (IsBoundaryImpl(*(op->GetOOperands().front()), visited))
                return true;
        }
    }

    return false;
}

} // anonymous namespace

bool SubgraphUtils::IsBoundary(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    std::unordered_set<const LogicalTensor*> visited;
    return IsBoundaryImpl(*tensor, visited);
}

bool SubgraphUtils::IsBoundary(const LogicalTensor& tensor)
{
    std::unordered_set<const LogicalTensor*> visited;
    return IsBoundaryImpl(tensor, visited);
}

} // namespace npu::tile_fwk
