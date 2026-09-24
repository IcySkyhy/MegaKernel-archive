/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file axis_combine_marker.cpp
 * \brief
 */

#include "axis_combine_marker.h"
namespace npu {
namespace tile_fwk {
const std::unordered_set<Opcode> whiteList{Opcode::OP_RESHAPE, Opcode::OP_VEC_DUP};
const std::unordered_set<OpCalcType> propagationCalcType{
    OpCalcType::ELMWISE,
    OpCalcType::BROADCAST,
    OpCalcType::CAST,
    OpCalcType::REDUCE,
};

void AxisCombineMarker::Run(Function& function)
{
    Init(function);
    ForwardVisit();
    BuildUnionFind();
    ResolveGroupStatus();
}

bool AxisCombineMarker::IsTensorEnableAxisCombine(LogicalTensorPtr tensor) const
{
    auto it = tensorStatus_.find(tensor);
    return it != tensorStatus_.end() && it->second == AxisReorderStatus::ENABLE;
}

void AxisCombineMarker::Init(Function& function)
{
    tensorStatus_.clear();
    parent_.clear();
    rank_.clear();
    opList_ = function.Operations(true, SortOperationsMode::LIGHTWEIGHT).DuplicatedOpList();
}

void UpdateCopyinStatus(Operation* op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus>& tensorStatus)
{
    // 约束: COPY_IN 为单输入单输出。
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (outputTensor->GetShape().back() != 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
        return;
    }
    if (outputTensor->GetShape().back() == inputTensor->GetShape().back()) {
        tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
        return;
    }
    tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
    return;
}

void UpdateViewStatus(Operation* op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus>& tensorStatus)
{
    // 约束: VIEW 为单输入单输出。
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (inputTensor->GetShape().back() != outputTensor->GetShape().back()) {
        tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    if (outputTensor->GetShape().back() != 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
        return;
    }
    if (tensorStatus.find(inputTensor) != tensorStatus.end()) {
        tensorStatus[outputTensor] = tensorStatus[inputTensor];
        return;
    }
    if (inputTensor->GetShape().back() == 1 && outputTensor->GetShape().back() == 1) {
        tensorStatus[inputTensor] = AxisReorderStatus::ENABLE;
        tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
        return;
    }
    tensorStatus[inputTensor] = AxisReorderStatus::DISABLE;
    tensorStatus[outputTensor] = AxisReorderStatus::DISABLE; // DDR场景，不涉及。
}

void UpdateAssembleStatus(Operation* op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus>& tensorStatus)
{
    // 约束: ASSEMBLE/COPY_OUT 为单输入单输出。
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (tensorStatus.find(inputTensor) != tensorStatus.end()) {
        if (tensorStatus[inputTensor] == AxisReorderStatus::ENABLE) {
            if (inputTensor->GetShape().back() != outputTensor->GetShape().back()) {
                tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
                tensorStatus[inputTensor] = AxisReorderStatus::DISABLE; // 如果尾轴有assemble，那么不能支持合轴
            } else {
                tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
            }
            return;
        }
        tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    // 正向推导不应该存在assemble输入没被访问过的场景
}

void UpdateExpandStatus(Operation* op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus>& tensorStatus)
{
    // 约束: EXPAND 为单输入单输出。
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (tensorStatus[inputTensor] == AxisReorderStatus::ENABLE) {
        auto dimSize = static_cast<int>(inputTensor->GetShape().size());
        auto axes = op->GetVectorIntAttribute(OpAttributeKey::expandDims);
        // 在尾轴为1的条件下，要求尾轴没有发生broadcast。[n, 1, 1]->expand->[n, 8, 1]??
        bool hasTailExpand = false;
        for (auto axis : axes) {
            if (axis >= dimSize - 1) {
                hasTailExpand = true;
                break;
            }
        }

        if (!hasTailExpand) {
            tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
            return;
        } else {
            // 如果是尾轴broadcast，不支持交换轴
            tensorStatus[inputTensor] = AxisReorderStatus::DISABLE;
            tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
        }
        return;
    }
    // 如果expand输出尾轴为1，并且输入就不支持合轴，那么输出也不支持合轴
    if (outputTensor->GetShape().back() == 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    // 如果expand的输出尾轴不为1，那么不涉及到合轴优化。
    tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
}

void UpdateReduceStatus(Operation* op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus>& tensorStatus)
{
    constexpr int kDimOffset = 2;
    // 最后两根轴不发生reduce，并且尾轴为1。那么支持交换轴，如果倒数第二根轴发生reduce，不支持。尾轴reduce，需不需要交换轴要看后继节点
    auto inputTensor = op->GetIOperands()[0];
    auto dimSize = static_cast<int>(inputTensor->GetShape().size());
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    for (auto outputTensor : op->GetOOperands()) {
        if (outputTensor->GetConsumers().empty()) {
            tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
            continue;
        }
        if (dimSize > 1 && axis < dimSize - kDimOffset) {
            tensorStatus[outputTensor] = tensorStatus[inputTensor];
            continue;
        }
        if (axis == dimSize - kDimOffset) {
            // reduce倒数第二轴，当前不支持合轴优化
            tensorStatus[inputTensor] = AxisReorderStatus::DISABLE;
            tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
            continue;
        }
        if (inputTensor->GetShape().back() == outputTensor->GetShape().back()) {
            // reduce tensor shape尾轴为1, 且语义为尾轴Reduce
            tensorStatus[inputTensor] = AxisReorderStatus::DISABLE;
            tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
            continue;
        }
        // Reduce尾轴，默认可以
        tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
    }
}

void UpdateElewiseStatus(Operation* op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus>& tensorStatus)
{
    auto outputTensor = op->GetOOperands()[0];
    bool multiOutput = op->GetOOperands().size() > 1;
    if (!multiOutput) {
        for (auto inputTensor : op->GetIOperands()) {
            if (tensorStatus[inputTensor] == AxisReorderStatus::UNKNOWN && inputTensor->GetShape().back() == 1) {
                tensorStatus[inputTensor] = AxisReorderStatus::ENABLE;
            }
        }
    }
    bool allEnable = true;
    bool hasDisable = false;
    AxisReorderStatus lastStatus;
    for (auto inputTensor : op->GetIOperands()) {
        auto status = tensorStatus[inputTensor];
        if (status != AxisReorderStatus::ENABLE) {
            allEnable = false;
            lastStatus = status;
        }
        if (status == AxisReorderStatus::DISABLE) {
            hasDisable = true;
        }
    }
    if (!allEnable && SUPPORT_BRC_INLINE.count(op->GetOpcode()) == 0) {
        for (auto inputTensor : op->GetIOperands()) {
            if (tensorStatus.find(inputTensor) != tensorStatus.end() &&
                tensorStatus[inputTensor] == AxisReorderStatus::ENABLE) {
                tensorStatus[inputTensor] = lastStatus;
            }
        }
    }
    if (hasDisable || multiOutput) {
        for (auto out : op->GetOOperands()) {
            tensorStatus[out] = AxisReorderStatus::DISABLE;
        }
    } else if (outputTensor->GetShape().back() == 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
    } else {
        tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
    }
}

void AxisCombineMarker::DisableNoneWhiteListTensor(Operation* op)
{
    for (auto inputTensor : op->GetIOperands()) {
        if (inputTensor->GetShape().back() == 1) {
            tensorStatus_[inputTensor] = AxisReorderStatus::DISABLE;
        }
    }
    for (auto outputTensor : op->GetOOperands()) {
        if (outputTensor->GetShape().back() == 1) {
            tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
        } else {
            tensorStatus_[outputTensor] = AxisReorderStatus::UNKNOWN;
        }
    }
}

void AxisCombineMarker::UpdateOpACEnableForward(size_t opIdx)
{
    auto op = opList_[opIdx];
    auto outputTensor = op->GetOOperands()[0];
    if (outputTensor->GetShape().back() != outputTensor->GetRawTensor()->GetRawShape().back()) {
        tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    if (op->GetOpcode() == Opcode::OP_COPY_IN) {
        UpdateCopyinStatus(op, tensorStatus_);
        return;
    }
    if (op->GetOpcode() == Opcode::OP_VIEW) {
        UpdateViewStatus(op, tensorStatus_);
        return;
    }
    if (op->GetOpcode() == Opcode::OP_ASSEMBLE || op->GetOpcode() == Opcode::OP_COPY_OUT) {
        UpdateAssembleStatus(op, tensorStatus_);
        return;
    }
    if (op->GetOpcode() == Opcode::OP_EXPAND) {
        UpdateExpandStatus(op, tensorStatus_);
        return;
    }
    if (OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::REDUCE) {
        UpdateReduceStatus(op, tensorStatus_);
        return;
    }
    if (propagationCalcType.find(OpcodeManager::Inst().GetOpCalcType(op->GetOpcode())) != propagationCalcType.end()) {
        UpdateElewiseStatus(op, tensorStatus_);
        return;
    }
    if (whiteList.find(op->GetOpcode()) == whiteList.end()) { // 非白名单Op，尾轴为1均不支持合轴
        DisableNoneWhiteListTensor(op);
        return;
    }
    tensorStatus_[outputTensor] = AxisReorderStatus::UNKNOWN;
}

LogicalTensorPtr AxisCombineMarker::Find(LogicalTensorPtr x)
{
    if (parent_.find(x) == parent_.end()) {
        parent_[x] = x;
    }
    if (parent_[x] != x) {
        parent_[x] = Find(parent_[x]); // 路径压缩
    }
    return parent_[x];
}

void AxisCombineMarker::Union(LogicalTensorPtr x, LogicalTensorPtr y)
{
    auto px = Find(x);
    auto py = Find(y);
    if (px == py) {
        return;
    }
    // 按秩合并
    if (rank_[px] < rank_[py]) {
        std::swap(px, py);
    }
    parent_[py] = px;
    if (rank_[px] == rank_[py]) {
        rank_[px]++;
    }
}

bool AxisCombineMarker::IsEligibleUnionOutput(Operation* op, OpCalcType calcType, LogicalTensorPtr outputTensor) const
{
    bool isPropagationOp = propagationCalcType.find(calcType) != propagationCalcType.end();
    bool isTailPreserved = (op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_ASSEMBLE) &&
                           outputTensor->GetShape().back() == op->GetIOperands()[0]->GetShape().back();
    return isPropagationOp || isTailPreserved;
}

void AxisCombineMarker::UnionOutputWithInputs(Operation* op, OpCalcType calcType, LogicalTensorPtr outputTensor)
{
    if (!IsEligibleUnionOutput(op, calcType, outputTensor) || outputTensor->GetShape().back() != 1) {
        return;
    }
    for (auto inputTensor : op->GetIOperands()) {
        if (inputTensor->GetShape().back() != 1) {
            continue;
        }
        Union(inputTensor, outputTensor);
    }
}

void AxisCombineMarker::UnionMultiOutputTensors(Operation* op)
{
    if (op->GetOOperands().size() <= 1) {
        return;
    }
    LogicalTensorPtr firstOut = nullptr;
    for (auto outputTensor : op->GetOOperands()) {
        if (outputTensor->GetShape().back() != 1) {
            continue;
        }
        if (firstOut == nullptr) {
            firstOut = outputTensor;
            continue;
        }
        Union(firstOut, outputTensor);
    }
}

void AxisCombineMarker::BuildUnionFind()
{
    for (size_t opIdx = 0; opIdx < opList_.size(); opIdx++) {
        auto op = opList_[opIdx];
        auto calcType = OpcodeManager::Inst().GetOpCalcType(op->GetOpcode());
        for (auto outputTensor : op->GetOOperands()) {
            UnionOutputWithInputs(op, calcType, outputTensor);
        }
        UnionMultiOutputTensors(op);
    }
}

void AxisCombineMarker::ResolveGroupStatus()
{
    // 按并查集根节点对tensor分组
    std::unordered_map<LogicalTensorPtr, std::vector<LogicalTensorPtr>> groups;
    for (auto& tensorStatus : tensorStatus_) {
        if (tensorStatus.first->GetShape().back() != 1) {
            continue;
        }
        groups[Find(tensorStatus.first)].push_back(tensorStatus.first);
    }
    // 对每个分组统一状态：DISABLE优先，其次ENABLE传播给UNKNOWN
    for (auto& group : groups) {
        auto members = group.second;
        bool hasDisable = false;
        bool hasEnable = false;
        for (auto tensor : members) {
            if (tensorStatus_[tensor] == AxisReorderStatus::DISABLE) {
                hasDisable = true;
            }
            if (tensorStatus_[tensor] == AxisReorderStatus::ENABLE) {
                hasEnable = true;
            }
        }
        if (hasDisable) {
            for (auto tensor : members) {
                tensorStatus_[tensor] = AxisReorderStatus::DISABLE;
            }
        } else if (hasEnable) {
            for (auto tensor : members) {
                if (tensorStatus_[tensor] == AxisReorderStatus::UNKNOWN) {
                    tensorStatus_[tensor] = AxisReorderStatus::ENABLE;
                }
            }
        }
    }
}

void AxisCombineMarker::ForwardVisit()
{
    for (size_t opIdx = 0; opIdx < opList_.size(); ++opIdx) {
        UpdateOpACEnableForward(opIdx);
    }
}
} // namespace tile_fwk
} // namespace npu
