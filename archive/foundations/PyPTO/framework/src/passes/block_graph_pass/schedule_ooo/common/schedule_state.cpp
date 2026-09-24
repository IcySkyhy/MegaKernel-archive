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
 * \file schedule_state.cpp
 * \brief Implementation of ScheduleState methods.
 */

#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"

namespace npu::tile_fwk {

std::vector<int>& ScheduleState::GetOpMemIds(Operation* op)
{
    auto it = opReqMemIdsMap.find(op);
    if (it != opReqMemIdsMap.end()) {
        return it->second;
    }
    std::vector<int> memIds;
    for (auto tensor : GetInOutOperandCached(op)) {
        memIds.push_back(tensor->memoryrange.memId);
    }
    auto inserted = opReqMemIdsMap.emplace(op, std::move(memIds));
    return inserted.first->second;
}

bool ScheduleState::ReplaceOpMemId(Operation* op, int oldMemId, int newMemId)
{
    auto& memIds = opReqMemIdsMap[op];
    bool replaced = false;
    for (auto& memId : memIds) {
        if (memId == oldMemId) {
            memId = newMemId;
            replaced = true;
        }
    }
    return replaced;
}

Status ScheduleState::InitLocalBuffer(LogicalTensorPtr oOperand, int memId)
{
    if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        return SUCCESS;
    }
    if (static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()) !=
        ShapeCeilAlign(oOperand->tensor->rawshape, oOperand->tensor->datatype)) {
        APASS_LOG_WARN_F(Elements::Tensor,
                         "InitLocalBuffer Failed at ShapeCeilAlign! "
                         "Please ensure that the rawTensor[%d] shapes are aligned.",
                         oOperand->GetRawMagic());
    }
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        localBufferMap[memId] = std::make_shared<LocalBuffer>(memId, oOperand->tensor->GetRawDataSize(),
                                                              oOperand->GetMemoryTypeOriginal());
    } else {
        localBufferMap[memId]->size = std::max(localBufferMap[memId]->size,
                                               static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()));
    }
    return SUCCESS;
}

std::string ScheduleState::GetOpInfo(Operation* op) const
{
    if (op == nullptr)
        return "nullptr";
    return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
}

Status ScheduleState::DelBufRefCount(const int memId)
{
    if (bufRefCount.find(memId) == bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", memId);
        return FAILED;
    }
    bufRefCount[memId]--;
    APASS_LOG_DEBUG_F(Elements::Tensor, "DelBufRefCount: memId [%d], refcount [%d].", memId, bufRefCount[memId]);
    if (bufRefCount[memId] < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount cannot less than 0.", memId);
        return FAILED;
    }
    return SUCCESS;
}

uint64_t ScheduleState::ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype)
{
    uint64_t bytes = 0;
    if (shape.size() == DIM_FIVE) {
        bytes = BytesPerElement(dtype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        bytes = CeilAlign(bytes, UB_BLOCK_SIZE);
    } else {
        uint64_t preDimSize = 1;
        uint64_t lastTwoDimSize = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            if ((shape.size() != 1) && (i < (shape.size() - LAST_TWO_DIM))) {
                preDimSize *= shape[i];
            } else {
                lastTwoDimSize *= shape[i];
            }
        }
        bytes = preDimSize * CeilAlign(lastTwoDimSize * BytesPerElement(dtype), UB_BLOCK_SIZE);
    }
    return bytes;
}

const LogicalTensors& ScheduleState::GetInOutOperandCached(Operation* op)
{
    auto it = inOutOperandsCache.find(op);
    if (it != inOutOperandsCache.end())
        return it->second;
    LogicalTensors inOutOperand;
    inOutOperand.reserve(op->GetOOperands().size() + op->GetIOperands().size());
    for (auto o : op->GetOOperands()) {
        if (o->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            inOutOperand.push_back(o);
        }
    }
    for (auto i : op->GetIOperands()) {
        if (i->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            inOutOperand.push_back(i);
        }
    }
    auto cacheIt = inOutOperandsCache.emplace(op, std::move(inOutOperand)).first;
    return cacheIt->second;
}

void ScheduleState::UpdateBufRefCount(Operation* op, LogicalTensorPtr tensor)
{
    int memId = tensor->memoryrange.memId;
    if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
        bufRefCount[memId]++;
        opReqMemIdsMap[op].push_back(memId);
    }
}

Status ScheduleState::InitBufRefCount(std::vector<Operation*>& list)
{
    bufRefCount.clear();
    depManager.ClearDependencies();
    localBufferMap.clear();
    inOutOperandsCache.clear();
    opReqMemIdsMap.clear();
    for (const auto& op : list) {
        for (auto& tensor : op->GetIOperands()) {
            UpdateBufRefCount(op, tensor);
            int memId = tensor->memoryrange.memId;
            if (InitLocalBuffer(tensor, memId) == FAILED) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                return FAILED;
            }
        }
        for (auto& tensor : op->GetOOperands()) {
            UpdateBufRefCount(op, tensor);
            int memId = tensor->memoryrange.memId;
            if (InitLocalBuffer(tensor, memId) == FAILED) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool ScheduleState::IsOpAlloc(Operation* op)
{
    if (op == nullptr) {
        return false;
    }
    return op->GetOpcodeStr().find("ALLOC") != std::string::npos;
}

Status ScheduleState::CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t>& bufferSize,
                                     std::set<int>& memIdMap)
{
    for (auto tensor : tensors) {
        if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        const auto& shape = tensor->tensor->GetRawShape();
        if (std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d < 0; })) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "Dynamic axis detected in %s, "
                              "OoOSchedule requires static rawShape!",
                              tensor->Dump().c_str());
            return FAILED;
        }
        if (memIdMap.find(tensor->memoryrange.memId) == memIdMap.end()) {
            bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->tensor->GetRawDataSize();
            memIdMap.insert(tensor->memoryrange.memId);
        }
    }
    return SUCCESS;
}

std::string ScheduleState::DumpOpInfo(Operation& op)
{
    std::ostringstream os;
    os << "op: " << op.GetOpcodeStr().c_str() << "[" << op.GetOpMagic() << "] | ";
    os << "inputs: { ";
    for (size_t i = 0; i < op.iOperand.size(); i++) {
        os << "RawTensor [" << op.GetInputOperand(i)->tensor->GetRawMagic() << "] ";
        os << op.iOperand[i]->tensor->DumpSSA(true, true);
        if (i != op.iOperand.size() - 1) {
            os << ", ";
        }
    }
    os << " }"
       << " | ";
    os << "outputs: { ";
    for (size_t i = 0; i < op.oOperand.size(); i++) {
        os << "RawTensor [" << op.GetOutputOperand(i)->tensor->GetRawMagic() << "] ";
        os << op.oOperand[i]->tensor->DumpSSA(true, true);
        if (i != op.oOperand.size() - 1) {
            os << ", ";
        }
    }
    os << "} ";
    return os.str();
}

Status ScheduleState::CheckOpBufferSize(Operation* op)
{
    std::map<MemoryType, int64_t> bufferSizeMap;
    std::set<int> memIdMap;
    if (CalcBufferSize(op->GetIOperands(), bufferSizeMap, memIdMap) != SUCCESS ||
        CalcBufferSize(op->GetOOperands(), bufferSizeMap, memIdMap) != SUCCESS) {
        return FAILED;
    }
    for (auto& bufferPair : bufferSizeMap) {
        if (localMemSize.find(bufferPair.first) == localMemSize.end()) {
            continue;
        }
        if (bufferPair.second <= localMemSize[bufferPair.first]) {
            continue;
        }
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_MEMORY_ALLOCATION, Elements::Operation,
                              "Alloc tensor [%d] size [%ld] exceeds %s size [%ld]! %s",
                              op->GetOutputOperand(0)->GetMagic(), bufferPair.second,
                              MemoryTypeToString(bufferPair.first).c_str(), localMemSize[bufferPair.first],
                              GetFormatBacktrace(*op).c_str());
            APASS_LOG_ERROR_F(Elements::Operation, "Tensor [%d] producer info:", op->GetOutputOperand(0)->GetMagic());
            for (auto producer : op->GetOutputOperand(0)->GetProducers()) {
                if (producer == op) {
                    continue;
                }
                APASS_LOG_ERROR_F(Elements::Operation, "      %s.", DumpOpInfo(*producer).c_str());
            }
        } else {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_MEMORY_ALLOCATION, Elements::Operation,
                              "OP %s[%d] in/output total size [%ld] exceeds %s size [%ld]!", op->GetOpcodeStr().c_str(),
                              op->GetOpMagic(), bufferPair.second, MemoryTypeToString(bufferPair.first).c_str(),
                              localMemSize[bufferPair.first]);
            APASS_LOG_ERROR_F(Elements::Operation, " %s.", DumpOpInfo(*op).c_str());
        }
        return FAILED;
    }
    return SUCCESS;
}

Status ScheduleState::Init(std::vector<Operation*>& opList)
{
    localMemSize = CommonUtils::GetLocalMemorySize();
    localMemoryCurrentSize = localMemSize;
    operations = opList;
    for (auto& op : operations) {
        if (CheckOpBufferSize(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] checkOpBufferSize failed! %s", op->GetOpcodeStr().c_str(),
                              op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
    }
    InitBufRefCount(operations);
    if (depManager.InitDependencies(operations, true) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    depManager.PrintDependencies(operations);

    opList = operations;
    return SUCCESS;
}

CoreLocationType ScheduleState::GetCoreLocation(Operation* op) const
{
    auto it = schedInfoMap.find(op);
    if (it != schedInfoMap.end())
        return it->second.coreLocation;
    return CoreLocationType::UNKNOWN;
}

int ScheduleState::GetExecOrder(Operation* op) const
{
    auto it = schedInfoMap.find(op);
    if (it != schedInfoMap.end())
        return it->second.execOrder;
    return -1;
}

bool ScheduleState::IsOpAllocInSchedInfo(Operation* op) const
{
    auto it = schedInfoMap.find(op);
    if (it != schedInfoMap.end())
        return it->second.isAlloc;
    return false;
}

bool ScheduleState::IsOpRetired(Operation* op) const
{
    auto it = schedInfoMap.find(op);
    if (it != schedInfoMap.end())
        return it->second.isRetired;
    return false;
}

void ScheduleState::InsertOrdered(Operation* insertOp)
{
    int execOrder = schedInfoMap[insertOp].execOrder;
    auto it = orderedOps.begin();
    for (; it != orderedOps.end(); it++) {
        if (schedInfoMap[*it].execOrder >= execOrder) {
            break;
        }
    }
    auto insertPos = orderedOps.insert(it, insertOp);
    for (auto adjustIt = insertPos + 1; adjustIt != orderedOps.end(); adjustIt++) {
        if (schedInfoMap[*adjustIt].execOrder >= execOrder) {
            schedInfoMap[*adjustIt].execOrder++;
        }
    }
}

CoreLocationType ScheduleState::ResolveCoreForFree(int memId)
{
    auto overrideIt = dualDstMemIdCoreOverride.find(memId);
    if (overrideIt != dualDstMemIdCoreOverride.end()) {
        return overrideIt->second;
    }

    auto allocIt = tensorAllocMap.find(memId);
    if (allocIt == tensorAllocMap.end() || allocIt->second == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "ResolveCoreForFree cannot find alloc op for tensor[%d].", memId);
        return CoreLocationType::UNKNOWN;
    }
    auto schedIt = schedInfoMap.find(allocIt->second);
    if (schedIt == schedInfoMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "ResolveCoreForFree cannot find schedule info for %s.",
                          GetOpInfo(allocIt->second).c_str());
        return CoreLocationType::UNKNOWN;
    }
    return schedIt->second.coreLocation;
}

bool ScheduleState::IsDualDstAlloc(Operation* allocOp)
{
    if (!enableDualDst)
        return false;
    if (allocOp == nullptr)
        return false;
    auto it = schedInfoMap.find(allocOp);
    if (it == schedInfoMap.end() || !it->second.isAlloc)
        return false;
    for (auto* succ : depManager.GetSuccessors(allocOp)) {
        if (succ != nullptr && succ->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            return true;
        }
    }
    return false;
}

} // namespace npu::tile_fwk
