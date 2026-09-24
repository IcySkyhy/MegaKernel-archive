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
 * \file last_use_mark.cpp
 * \brief 标记算子的LastUse属性
 */

#include "last_use_mark.h"
#include "passes/pass_log/pass_log.h"

#ifndef MODULE_NAME
#define MODULE_NAME "LastUseMark"
#endif

namespace npu::tile_fwk {

Status LastUseMark::CollectLastUseInfo(Function& function)
{
    lastUseMap_.clear();
    APASS_LOG_INFO_F(Elements::Function, "===> Start CollectLastUseInfo.");
    for (auto& program : function.rootFunc_->programs_) {
        std::map<std::pair<int, int>, std::pair<LogicalTensorPtr, Operation*>> recordMemMap;
        auto opList = program.second->Operations(false);
        for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
            Operation* op = &opList[opIdx];
            if (LASTUSE_OPS.find(op->GetOpcode()) != LASTUSE_OPS.end()) {
                int tensorSize = op->GetIOperands().size() + op->GetOOperands().size();
                std::vector<int> initVec(tensorSize, false);
                op->SetAttribute(OpAttributeKey::lastUse, initVec);
            }
            for (size_t inputIdx = 0; inputIdx < op->GetIOperands().size(); inputIdx++) {
                auto inTensor = op->GetInputOperand(inputIdx);
                recordMemMap[{inTensor->memoryrange.start, inTensor->memoryrange.end}] = {inTensor, op};
                APASS_LOG_INFO_F(Elements::Operation, "Record OP_%s[%d] input Tensor %d Memory Range{%zu, %zu}",
                                 op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(),
                                 inTensor->memoryrange.start, inTensor->memoryrange.end);
            }
        }
        for (auto& entry : recordMemMap) {
            auto& value = entry.second;
            auto inTensor = value.first;
            auto op = value.second;
            lastUseMap_[inTensor] = op;
            APASS_LOG_INFO_F(Elements::Operation, "Record lastUseMap Key: inTensor[%d], Value: OP_%s[%d]",
                             inTensor->GetMagic(), op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End CollectLastUseInfo.");
    return SUCCESS;
}

void LastUseMark::SetLastUseAttributes()
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start SetLastUseAttributes.");
    std::unordered_map<Operation*, std::vector<int>> opInputIdxMap;
    std::unordered_set<Opcode> reduceOp = {Opcode::OP_ROWSUM_SINGLE, Opcode::OP_ROWMAX_SINGLE,
                                           Opcode::OP_ROWMIN_SINGLE};
    for (auto& entry : lastUseMap_) {
        auto lastUseOp = entry.second;
        auto lastUseTensor = entry.first;
        if (LASTUSE_OPS.find(lastUseOp->GetOpcode()) == LASTUSE_OPS.end()) {
            continue;
        }
        if (opInputIdxMap.find(lastUseOp) == opInputIdxMap.end()) {
            int tensorSize = lastUseOp->GetIOperands().size() + lastUseOp->GetOOperands().size();
            std::vector<int> tensorIdxVec(tensorSize, false);
            int inputIdx = lastUseOp->GetIOperandIndex(lastUseTensor) + lastUseOp->GetOOperands().size();
            if (reduceOp.find(lastUseOp->GetOpcode()) != reduceOp.end() && inputIdx == tensorSize - 1) {
                tensorIdxVec[inputIdx] = false;
            } else {
                tensorIdxVec[inputIdx] = true;
            }
            opInputIdxMap[lastUseOp] = tensorIdxVec;
        } else {
            int inputIdx = lastUseOp->GetIOperandIndex(lastUseTensor) + lastUseOp->GetOOperands().size();
            opInputIdxMap[lastUseOp][inputIdx] = true;
        }
    }
    for (auto& entry : opInputIdxMap) {
        auto op = entry.first;
        if (op->HasAttribute(OpAttributeKey::brcOperand) || op->GetOpcode() == Opcode::OP_EXPAND) {
            APASS_LOG_INFO_F(Elements::Operation, "Skip Process OP_%s[%d] LastUse Attribute",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic());
            continue;
        }
        op->SetAttribute(OpAttributeKey::lastUse, opInputIdxMap[op]);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End SetLastUseAttributes.");
}

Status LastUseMark::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "=============== START LastUseMark ===============");
    if (CollectLastUseInfo(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "Run CollectLastUseInfo Failed.");
        return FAILED;
    }
    SetLastUseAttributes();
    APASS_LOG_INFO_F(Elements::Operation, "=============== END LastUseMark ===============");
    return SUCCESS;
}

} // namespace npu::tile_fwk
