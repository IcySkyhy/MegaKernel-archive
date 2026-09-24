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
 * \file add_alloc.cpp
 * \brief
 */

#include "add_alloc.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "AddAlloc"

namespace npu::tile_fwk {
Status AddAlloc::GenTensorAllocMsgMap(Function& function,
                                      std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const
{
    for (auto& op : function.Operations(false).DuplicatedOpList()) {
        if (FindTensorAllocMsg(*op, tensorAllocMsgMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "FindTensorAllocMsg failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AddAlloc::GenAllocNode(Function& function)
{
    std::unordered_map<int, TensorAllocMsg> tensorAllocMsgMap;
    if (GenTensorAllocMsgMap(function, tensorAllocMsgMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "GenTensorAllocMsgMap failed.");
        return FAILED;
    }
    std::vector<std::pair<int, TensorAllocMsg>> sortedAllocs(tensorAllocMsgMap.begin(), tensorAllocMsgMap.end());
    std::stable_sort(sortedAllocs.begin(), sortedAllocs.end(), [](const auto& a, const auto& b) {
        int magicA = a.second.producer[0].get().GetOpMagic();
        int magicB = b.second.producer[0].get().GetOpMagic();
        if (magicA == magicB) {
            return a.second.firstOperandIdx < b.second.firstOperandIdx;
        }
        return magicA < magicB;
    });

    for (auto& tensorAllocMsg : sortedAllocs) {
        if (tensorAllocMsg.second.isAllocated == false) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "Create alloc node for tensor [%d]", tensorAllocMsg.first);
            CreateAllocNode(tensorAllocMsg.second, function);
        }
    }
    return SUCCESS;
}

Status AddAlloc::AddAndCheckAlloc(Function& function)
{
    if (GenAllocNode(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "GenAllocNode failed.");
        return FAILED;
    }
    std::vector<Operation*> newOperations;
    for (auto& op : function.Operations(false).DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
            continue;
        }
        newOperations.push_back(op);
    }
    function.ScheduleBy(newOperations);
    return SUCCESS;
}

TensorAllocMsg AddAlloc::ConstructTensorAllocMsg(Operation& op, size_t i, int memId) const
{
    TensorAllocMsg tensorAllocMsg;
    tensorAllocMsg.producer.push_back(std::ref(op));
    tensorAllocMsg.memType = op.GetOutputOperand(i)->GetMemoryTypeOriginal();
    tensorAllocMsg.memId = memId;
    tensorAllocMsg.firstOperandIdx = i;
    return tensorAllocMsg;
}

Status AddAlloc::UpdateTensorAllocMsg(Operation& op, size_t i,
                                      std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const
{
    auto memId = op.GetOutputOperand(i)->memoryrange.memId;
    if (memId == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get memId in memoryrange failed, op:%d, operand: %zu.%s", op.GetOpMagic(),
                          i, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (tensorAllocMsgMap.find(memId) == tensorAllocMsgMap.end()) {
        tensorAllocMsgMap.emplace(memId, ConstructTensorAllocMsg(op, i, memId));
        return SUCCESS;
    }
    tensorAllocMsgMap[memId].producer.push_back(op);
    return SUCCESS;
}

Status AddAlloc::SetTensorAllocMsg(Operation& op, std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const
{
    for (size_t i = 0; i < op.GetOOperands().size(); i++) {
        if (op.GetOutputOperand(i)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (UpdateTensorAllocMsg(op, i, tensorAllocMsgMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "UpdateTensorAllocMsg failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AddAlloc::FindTensorAllocMsg(Operation& op, std::unordered_map<int, TensorAllocMsg>& tensorAllocMsgMap) const
{
    // 遍历所有节点，找到需要分配Alloc的tensor以及其第一次出现时候的位置
    if (SetTensorAllocMsg(op, tensorAllocMsgMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "SetTensorAllocMsg failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status AddAlloc::GenAllocOpcode(const Opcode& allocOpcode, const TensorAllocMsg& tensorAllocMsg, Function& function)
{
    for (auto& oOperand : tensorAllocMsg.producer[0].get().GetOOperands()) {
        if (oOperand->memoryrange.memId != tensorAllocMsg.memId) {
            continue;
        }
        PassOperationUtils::AddOperation(function, allocOpcode, {},
                                         std::vector<std::shared_ptr<LogicalTensor>>({oOperand}));
    }
    return SUCCESS;
}

Status AddAlloc::CreateAllocNode(const TensorAllocMsg& tensorAllocMsg, Function& function)
{
    auto iter = allocOpcodeMap.find(tensorAllocMsg.memType);
    if (iter != allocOpcodeMap.end()) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Create alloc node for memtype [%d]",
                          static_cast<int>(tensorAllocMsg.memType));
        if (tensorAllocMsg.producer.size() == 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "TensorAllocMsg's producer size cannot be 0.");
            return FAILED;
        }
        Opcode allocOpcode = iter->second;
        if (GenAllocOpcode(allocOpcode, tensorAllocMsg, function)) {
            APASS_LOG_ERROR_F(Elements::Tensor, "GenAllocOpcode failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
