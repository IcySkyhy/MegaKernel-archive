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
 * \file spill_engine.cpp
 * \brief Spill execution engine — pure execution layer only.
 *        Orchestration methods (GenBufferSpill, SelectSpillBuffers, HasEnoughBuffer,
 *        RearrangeBuffer, ApplySpillContext, PrintSpillFailedInfo, GetSpillGroup,
 *        GetDualSpillGroup, GetGroupNextUseTime) are in OoOScheduler (spill_buffer.cpp).
 *        SpillEngine only retains execution-layer methods that work purely through
 *        state_ / notifier_ / function_ / irBuilder_ / ddrKindMap_.
 */

#include "spill_engine.h"
#include "tilefwk/symbolic_scalar.h"
#include "passes/pass_utils/reschedule_utils.h"

namespace npu::tile_fwk {

constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;

void SpillEngine::EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind)
{
    if (t == nullptr)
        return;
    int memId = t->memoryrange.memId;
    if (ddrKindMap_.count(memId) != 0)
        return;
    ddrKindMap_[memId] = kind;
    if (!state_.HasObservers())
        return;
    InitDDRBufferEvent event;
    event.clock = -1;
    event.memId = memId;
    event.kind = kind;
    event.magic = t->GetMagic();
    event.dtype = t->Datatype();
    auto dynShape = t->GetDynValidShape();
    if (!dynShape.empty()) {
        for (const auto& s : dynShape) {
            event.shape.push_back(s.Dump());
        }
    } else {
        for (auto d : t->GetShape()) {
            event.shape.push_back(std::to_string(d));
        }
    }
    for (auto* obs : state_.observers_) {
        obs->OnInitDDRBuffer(event);
    }
}

int64_t SpillEngine::CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset, DataType dataType)
{
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }

    int64_t linearOffset = 0;
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        linearOffset += offset[i - 1] * stride;
        if (i > 0) {
            stride *= shape[i - 1];
        }
    }
    return linearOffset * BytesOf(dataType);
}

bool SpillEngine::IsBelongSpillBlackList(Operation* spillOp, Operation* op)
{
    std::set<Operation*> filterLtags;
    FindFilterLtags(op, filterLtags);
    if (state_.schedInfoMap[spillOp].isAlloc || filterLtags.count(spillOp) != 0 || !CheckMachineAndL1(spillOp, op)) {
        return true;
    }
    return false;
}

void SpillEngine::FindFilterLtags(Operation* allocOp, std::set<Operation*>& filterLtags)
{
    auto dstOpList = state_.depManager.GetSuccessors(allocOp);
    for (auto dstOp : dstOpList) {
        if (COPY_IN_OPS.find(dstOp->GetOpcode()) == COPY_IN_OPS.end()) {
            for (auto& inOp : state_.depManager.GetPredecessors(dstOp)) {
                filterLtags.insert(inOp);
            }
            continue;
        }
        for (auto& dstOpId : state_.depManager.GetSuccessors(dstOp)) {
            auto dstOp_level0 = dstOpId;
            for (auto& inOp : state_.depManager.GetPredecessors(dstOp_level0)) {
                filterLtags.insert(inOp);
            }
        }
    }
}

bool SpillEngine::CheckMachineAndL1(Operation* spillOp, Operation* allocOp)
{
    if (!spillOp->GetInputOperand(0) && allocOp->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        APASS_LOG_WARN_F(Elements::Tensor, "CheckMachineAndL1: spillOp %s has no inputOperand.",
                         state_.GetOpInfo(spillOp).c_str());
        return false;
    }
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
        allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos &&
        spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos &&
        spillOp->GetOpcodeStr().find("RESHAPE") == std::string::npos &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        return false;
    }
    return true;
}

LogicalTensorPtr SpillEngine::CreateLocalTensor(LogicalTensorPtr spillTensor)
{
    LogicalTensorPtr localTensor = irBuilder_.CreateTensorVar(spillTensor->Datatype(), spillTensor->GetShape(),
                                                              std::vector<SymbolicScalar>{}, spillTensor->Format());
    localTensor->SetMemoryTypeToBe(spillTensor->GetMemoryTypeOriginal());
    localTensor->SetMemoryTypeOriginal(spillTensor->GetMemoryTypeOriginal());
    localTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    localTensor->tensor->rawshape = spillTensor->tensor->rawshape;
    int rawMagic = localTensor->GetRawTensor()->GetRawMagic();
    localTensor->memoryrange.memId = rawMagic;
    state_.localBufferMap[rawMagic] = std::make_shared<LocalBuffer>(rawMagic, localTensor->tensor->GetRawDataSize(),
                                                                    localTensor->GetMemoryTypeOriginal());
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);
    APASS_LOG_DEBUG_F(Elements::Operation, "Create local tensor[%d].", localTensor->memoryrange.memId);
    return localTensor;
}

const std::vector<int64_t>& SpillEngine::GetLargerShape(const std::vector<int64_t>& shape1,
                                                        const std::vector<int64_t>& shape2)
{
    for (size_t i = 0; i < shape1.size(); i++) {
        if (shape1[i] > shape2[i]) {
            return shape1;
        }
    }
    return shape2;
}

LogicalTensorPtr SpillEngine::CreateGMTensor(LogicalTensorPtr spillTensor, LogicalTensorPtr actualSpillTensor,
                                             int spillMemId, DataType gmDtype)
{
    DataType dtype = (gmDtype == DT_BOTTOM) ? spillTensor->Datatype() : gmDtype;
    std::shared_ptr<RawTensor> gmRawTensor = std::make_shared<RawTensor>(
        dtype, GetLargerShape(spillTensor->tensor->rawshape, actualSpillTensor->tensor->rawshape),
        TileOpFormat::TILEOP_ND, "WorkspaceGm");
    LogicalTensorPtr gmTensor = irBuilder_.CreateTensorVar(
        gmRawTensor, spillTensor->GetOffset(), actualSpillTensor->GetShape(), std::vector<SymbolicScalar>{});
    gmTensor->SetAttr(OpAttributeKey::workspaceBaseOffset, state_.workspaceOffset);
    gmTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
    gmTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    gmTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    gmTensor->tensor->rawshape = GetLargerShape(spillTensor->tensor->rawshape, actualSpillTensor->tensor->rawshape);
    if (state_.localBufferMap.find(spillMemId) == state_.localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] in localBufferMap.", spillMemId);
        return nullptr;
    }
    gmTensor->memoryrange = TileRange(state_.workspaceOffset, state_.workspaceOffset + gmRawTensor->GetRawDataSize(),
                                      state_.workspaceMemId++);
    state_.workspaceOffset += gmRawTensor->GetRawDataSize();
    EmitInitDDRBuffer(gmTensor, DDRBufferKind::SPILL_TEMP);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create gm tensor[%d].", gmTensor->memoryrange.memId);
    return gmTensor;
}

LogicalTensorPtr SpillEngine::CreateParticalTensor(LogicalTensorPtr iOperand, LogicalTensorPtr oriOperand,
                                                   LogicalTensorPtr spillTensor, std::vector<int64_t> toOffset)
{
    LogicalTensorPtr particalTensor = irBuilder_.CreateTensorVar(iOperand->Datatype(), iOperand->GetShape(),
                                                                 std::vector<SymbolicScalar>{}, iOperand->Format());
    particalTensor->SetMemoryTypeToBe(oriOperand->GetMemoryTypeToBe());
    particalTensor->SetMemoryTypeOriginal(oriOperand->GetMemoryTypeOriginal());
    particalTensor->tensor = oriOperand->tensor;
    particalTensor->memoryrange.memId = oriOperand->memoryrange.memId;
    particalTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    particalTensor->offset = toOffset;
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create partical tensor[%d].", particalTensor->memoryrange.memId);
    return particalTensor;
}

Operation* SpillEngine::CreateAllocOp(LogicalTensorPtr oOperand)
{
    Opcode opcode = oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    Operation& allocOp = irBuilder_.CreateTensorOpStmt(function_, opcode, {}, {oOperand});
    allocOp.UpdateLatency(1);
    state_.tensorAllocMap[oOperand->memoryrange.memId] = &allocOp;
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&allocOp).c_str());
    return &allocOp;
}

Operation* SpillEngine::CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand)
{
    Operation& copyinOp = spillOp->CloneOperation(function_, {iOperand}, {oOperand});
    copyinOp.SetIOpAtt(0, spillOp->GetIOpAttrOffset(0));
    copyinOp.SetOpAttribute(spillOp->GetOpAttribute()->Clone());
    copyinOp.inParamLocation_ = spillOp->inParamLocation_;
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Clone %s", state_.GetOpInfo(&copyinOp).c_str());
    return &copyinOp;
}

Operation* SpillEngine::CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                                       std::vector<OpImmediate> offset, bool isND2NZ)
{
    Operation& copyinOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_COPY_IN, {iOperand}, {oOperand});
    copyinOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offset, oOperand->GetMemoryTypeOriginal(), OpImmediate::Specified(oOperand->GetShape()),
        OpImmediate::Specified(oOperand->tensor->GetDynRawShape()),
        OpImmediate::Specified(oOperand->GetDynValidShape())));
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    bool isCube = true;
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    copyinOp.SetAttribute(OpAttributeKey::isCube, isCube);
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 || isND2NZ) {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2NZ));
        } else {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2ND));
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&copyinOp).c_str());
    return &copyinOp;
}

Operation* SpillEngine::CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                                        std::vector<OpImmediate> offset)
{
    Operation& copyoutOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_COPY_OUT, {iOperand}, {oOperand});
    copyoutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        iOperand->GetMemoryTypeOriginal(), offset, OpImmediate::Specified(iOperand->GetShape()),
        OpImmediate::Specified(iOperand->GetRawTensor()->GetDynRawShape()),
        OpImmediate::Specified(iOperand->GetDynValidShape())));
    if (spillOp->HasAttribute(OpAttributeKey::scaleValue)) {
        Element scaleValue = Element(DataType::DT_UINT64, 0);
        spillOp->GetAttr(OpAttributeKey::scaleValue, scaleValue);
        copyoutOp.SetAttribute(OpAttributeKey::scaleValue, scaleValue);
    }
    bool isCube = true;
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    copyoutOp.SetAttribute(OpAttributeKey::isCube, isCube);
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
        copyoutOp.SetAttribute(OpAttributeKey::copyIsNZ, 0);
    } else if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 &&
               iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        copyoutOp.SetAttribute(OpAttributeKey::copyOutMode, static_cast<int64_t>(Matrix::CopyOutMode::ND2ND));
    }
    copyoutOp.UpdateLatency(DEFAULT_LATENCY);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&copyoutOp).c_str());
    return &copyoutOp;
}

Operation* SpillEngine::CreateReshapeOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand)
{
    Operation& reshapeOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_RESHAPE, {iOperand}, {oOperand});
    bool isCube = true;
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    reshapeOp.SetAttribute(OpAttributeKey::isCube, isCube);
    reshapeOp.UpdateLatency(0);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&reshapeOp).c_str());
    return &reshapeOp;
}

Operation* SpillEngine::CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
                                         std::vector<int64_t> toOffset, std::vector<SymbolicScalar> toDynOffset,
                                         std::vector<SymbolicScalar> fromDynValidShape)
{
    Operation& assembleOp = irBuilder_.CreateTensorOpStmt(function_, Opcode::OP_ASSEMBLE, {iOperand}, {oOperand});
    assembleOp.UpdateLatency(1);
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(iOperand->GetMemoryTypeOriginal(), toOffset,
                                                                    toDynOffset, fromDynValidShape));
    bool isCube = true;
    if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        isCube = false;
    }
    assembleOp.SetAttribute(OpAttributeKey::isCube, isCube);
    APASS_LOG_DEBUG_F(Elements::Operation, "Spill: Create %s", state_.GetOpInfo(&assembleOp).c_str());
    return &assembleOp;
}

Operation* SpillEngine::GetSpillOp(int memId)
{
    if (state_.tensorOccupyMap.count(memId)) {
        return state_.tensorOccupyMap[memId];
    }
    return nullptr;
}

LogicalTensorPtr SpillEngine::GetSpillTensor(Operation* spillOp, int spillMemId)
{
    for (size_t i = 0; i < spillOp->GetOOperands().size(); i++) {
        if (spillOp->GetOOperands()[i]->memoryrange.memId == spillMemId) {
            return spillOp->GetOutputOperand(i);
        }
    }
    return nullptr;
}

Status SpillEngine::GetActualSpillForNd2nz(Operation*& spillOp, LogicalTensorPtr& spillTensor)
{
    if (spillOp->GetOpcode() == Opcode::OP_UB_COPY_ND2NZ) {
        for (auto producer : spillOp->ProducerOps()) {
            if (state_.schedInfoMap[producer].isAlloc)
                continue;
            spillTensor = spillOp->GetInputOperand(0);
            spillOp = producer;
            if (spillTensor == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Get %s spill tensor failed.",
                                  state_.GetOpInfo(spillOp).c_str());
                return FAILED;
            }
        }
        if (spillOp->GetOpcode() == Opcode::OP_UB_COPY_ND2NZ) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot spill %s.", state_.GetOpInfo(spillOp).c_str());
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Spill UB_COPY_ND2NZ producer %s", state_.GetOpInfo(spillOp).c_str());
    }
    return SUCCESS;
}

Status SpillEngine::GetActualSpill(Operation* op, Operation*& actualOp, LogicalTensorPtr& actualTensor)
{
    auto iOperand = op->GetInputOperand(0);
    if (iOperand == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "SmallShape spill: producer %s has null input.",
                          state_.GetOpInfo(op).c_str());
        return FAILED;
    }
    auto iOperandMemType = iOperand->GetMemoryTypeOriginal();
    if (iOperandMemType != MemoryType::MEM_UB && iOperandMemType != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F(Elements::Operation, "SmallShape spill: producer %s input memType is %s, expect UB/L0C.",
                          state_.GetOpInfo(op).c_str(), MemoryTypeToString(iOperandMemType).c_str());
        return FAILED;
    }
    actualOp = op;
    actualTensor = iOperand;
    if (iOperandMemType == MemoryType::MEM_UB) {
        Operation* prevOp = nullptr;
        for (auto& preOp : state_.depManager.GetPredecessors(op)) {
            if (!state_.schedInfoMap[preOp].isAlloc) {
                prevOp = preOp;
            }
        }
        if (prevOp == nullptr || prevOp->GetOpcode() != Opcode::OP_UB_COPY_ND2NZ) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "SmallShape spill: UB-producer %s does not have UB_COPY_ND2NZ predecessor.",
                              state_.GetOpInfo(op).c_str());
            return FAILED;
        }
        actualOp = prevOp;
        actualTensor = prevOp->GetInputOperand(0);
    }
    if (actualTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "SmallShape spill: actualTensor is null for producer %s.",
                          state_.GetOpInfo(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

void SpillEngine::CollectL0CConsumers(LogicalTensorPtr spillTensor, std::vector<Operation*>& consumers)
{
    for (auto* consumer : spillTensor->GetConsumers()) {
        if (consumer == nullptr || state_.schedInfoMap[consumer].isRetired) {
            continue;
        }
        auto output = consumer->GetOutputOperand(0);
        if (output == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation, "L0C spill: skip consumer %s without output operand.",
                             state_.GetOpInfo(consumer).c_str());
            continue;
        }
        auto outMem = output->GetMemoryTypeOriginal();
        if (outMem == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (outMem != MemoryType::MEM_UB && outMem != MemoryType::MEM_L1) {
            APASS_LOG_WARN_F(Elements::Operation, "L0C spill: skip consumer %s with output memType %s.",
                             state_.GetOpInfo(consumer).c_str(), MemoryTypeToString(outMem).c_str());
            continue;
        }
        consumers.push_back(consumer);
    }
    std::sort(consumers.begin(), consumers.end(), [this](Operation* a, Operation* b) {
        return state_.schedInfoMap[a].execOrder < state_.schedInfoMap[b].execOrder;
    });
}

bool SpillEngine::IsMultiProducerTensor(LogicalTensorPtr tensor)
{
    int producerCount = 0;
    for (auto& producer : tensor->GetProducers()) {
        if (producer->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            producerCount++;
        }
    }
    return producerCount > 1 ? true : false;
}

Status SpillEngine::GetPartialWriteReplayAttr(Operation* producerOp, std::vector<int64_t>& toOffset,
                                              std::vector<SymbolicScalar>& toDynOffset,
                                              std::vector<SymbolicScalar>& fromDynValidShape) const
{
    if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto attr = std::static_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid AssembleOpAttribute.");
            return FAILED;
        }
        toOffset = attr->GetToOffset();
        toDynOffset = attr->GetToDynOffset();
        fromDynValidShape = attr->GetFromDynValidShape();
        return SUCCESS;
    } else if (producerOp->GetOpcode() == Opcode::OP_L0C_TO_L1 || producerOp->GetOpcode() == Opcode::OP_L0C_COPY_UB) {
        auto attr = std::static_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid CopyOpAttribute.");
            return FAILED;
        }
        auto iOperand = producerOp->GetInputOperand(0);
        for (const auto& offsetImm : attr->GetToOffset()) {
            if (!offsetImm.IsSpecified() || !offsetImm.GetSpecifiedValue().ConcreteValid()) {
                APASS_LOG_ERROR_F(Elements::Operation, "L0C_TO_L1 replay only supports static concrete toOffset.");
                return FAILED;
            }
            toOffset.push_back(static_cast<int64_t>(offsetImm.GetSpecifiedValue()));
        }
        fromDynValidShape = iOperand->GetDynValidShape();
        if (fromDynValidShape.empty() && !attr->GetToDynValidShape().empty()) {
            fromDynValidShape = OpImmediate::ToSpecified(attr->GetToDynValidShape());
        }
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "Unsupported producer opcode in SpillParticalBuffer.");
    return FAILED;
}

bool SpillEngine::IsUnusedTensor(Operation* spillOp)
{
    if (spillOp == nullptr) {
        return false;
    }
    for (auto& succOp : state_.depManager.GetSuccessors(spillOp)) {
        if (state_.schedInfoMap[succOp].isRetired) {
            return false;
        }
    }
    return true;
}

Status SpillEngine::UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr reloadTensor,
                                         int spillMemId)
{
    for (size_t index = 0; index < targetOp->GetIOperands().size(); index++) {
        if (targetOp->GetIOperands()[index]->memoryrange.memId != spillMemId) {
            continue;
        }
        for (auto& inOp : targetOp->GetIOperands()[index]->GetProducers()) {
            if (IsSkipOp(*inOp)) {
                if (UpdateSkipOpInput(inOp, spillOp, targetOp, reloadTensor, index) != SUCCESS) {
                    return FAILED;
                }
            } else if (inOp == spillOp) {
                targetOp->UpdateInputOperand(index, reloadTensor);
            }
        }
    }
    return SUCCESS;
}

// 一律给 targetOp 重建一条链, 不判是否被共享: 原链原封不动留给别的读者 (含已发射的), 语义上永远安全。
// 代价只是多几个 skip op —— 它们零 codegen、不申请内存、不进流水线, 判"能不能省下这份克隆"反而更贵更险。
Status SpillEngine::UpdateSkipOpInput(Operation* chainTail, Operation* spillOp, Operation* targetOp,
                                      LogicalTensorPtr reloadTensor, size_t index)
{
    // 从链尾往上取祖先路径: 链中间的张量允许有旁支消费者, 沿 consumers 走会在分叉处停下, 到不了链尾。
    std::vector<Operation*> chain = SkipChainPath(chainTail);
    if (chain.empty()) {
        return SUCCESS;
    }
    std::reverse(chain.begin(), chain.end());
    const auto& producers = chain.front()->GetInputOperand(0)->GetProducers();
    if (producers.find(spillOp) == producers.end()) {
        return SUCCESS;
    }

    LogicalTensorPtr clonedTail = CloneSkipChain(targetOp, chain, reloadTensor);
    if (clonedTail == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Clone skip chain failed for %s.", state_.GetOpInfo(targetOp).c_str());
        return FAILED;
    }
    targetOp->UpdateInputOperand(index, clonedTail);
    DetachOrphanedSkipChain(chain, targetOp);
    return SUCCESS;
}

// 从链尾往链首扫: 摘掉尾巴才会让它的上游变成无读者, 一趟连锁清干净。
void SpillEngine::DetachOrphanedSkipChain(const std::vector<Operation*>& chain, Operation* targetOp)
{
    auto& skipOps = state_.schedInfoMap[targetOp].skipOps;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        Operation* oldOp = *it;
        auto pos = std::find(skipOps.begin(), skipOps.end(), oldOp);
        if (pos != skipOps.end()) {
            skipOps.erase(pos);
        }
        const auto& consumers = oldOp->GetOutputOperand(0)->GetConsumers();
        if (std::any_of(consumers.begin(), consumers.end(), [](Operation* c) { return !c->IsDeleted(); })) {
            continue;
        }
        EraseSchedulerSideMaps(oldOp);
        oldOp->SetAsDeleted();
        APASS_LOG_DEBUG_F(Elements::Operation, "Detached orphaned skip op %s.", state_.GetOpInfo(oldOp).c_str());
    }
}

// 逐个克隆 chain 上的 skip op, 把上一个克隆的输出接给下一个的输入, 链首的输入直接用 reload copyin 的产物。
// 返回克隆链尾的输出张量, 供 targetOp 改指。
// Clone(function_, true) 新建 RawTensor 并深拷 shape/rawshape/offset/dtype/format/dynValidShape, 这里只覆写 memId。
LogicalTensorPtr SpillEngine::CloneSkipChain(Operation* targetOp, const std::vector<Operation*>& chain,
                                             LogicalTensorPtr reloadTensor)
{
    std::vector<Operation*> clones;
    LogicalTensorPtr inTensor = reloadTensor;
    for (Operation* op : chain) {
        LogicalTensorPtr outTensor = op->GetOutputOperand(0)->Clone(function_, true);
        if (outTensor == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Clone skip op %s operand failed.", state_.GetOpInfo(op).c_str());
            return nullptr;
        }
        outTensor->memoryrange.memId = reloadTensor->memoryrange.memId;
        Operation& cloneOp = op->CloneOperation(function_, {inTensor}, {outTensor});
        UpdateOpInternalSubgraphID(cloneOp, op);
        clones.push_back(&cloneOp);
        inTensor = outTensor;
    }
    auto& skipOps = state_.schedInfoMap[targetOp].skipOps;
    skipOps.insert(skipOps.end(), clones.begin(), clones.end());
    return inTensor;
}

void SpillEngine::ReplaceSkipOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId)
{
    std::vector<Operation*> skipConsumers;
    for (auto* consumer : startTensor->GetConsumers()) {
        if (IsSkipOp(*consumer)) {
            skipConsumers.push_back(consumer);
        }
    }

    while (!skipConsumers.empty()) {
        Operation* skipOp = skipConsumers.back();
        skipConsumers.pop_back();
        auto skipOutTensor = skipOp->GetOutputOperand(0);
        if (skipOutTensor == nullptr) {
            continue;
        }
        if (skipOutTensor->memoryrange.memId == oldMemId) {
            skipOutTensor->memoryrange.memId = newMemId;
        }
        for (auto* consumer : skipOutTensor->GetConsumers()) {
            if (IsSkipOp(*consumer)) {
                skipConsumers.push_back(consumer);
            }
        }
    }
}

void SpillEngine::ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId)
{
    auto& reqMemIds = state_.opReqMemIdsMap[op];
    for (auto memId : reqMemIds) {
        if (memId == oldMemId || memId == newMemId) {
            state_.bufRefCount[newMemId]++;
        }
        if (memId == oldMemId) {
            std::replace(reqMemIds.begin(), reqMemIds.end(), oldMemId, newMemId);
        }
    }
    for (auto& outTensor : op->GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
            ReplaceSkipOpChainMemId(outTensor, oldMemId, newMemId);
        }
    }
}

void SpillEngine::UpdateOpInternalSubgraphID(Operation& op, Operation* srcOp)
{
    if (srcOp->GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        op.UpdateInternalSubgraphID(srcOp->GetInternalSubgraphID());
        op.SetAIVCore(srcOp->GetAIVCore());
    }
}

Status SpillEngine::UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId)
{
    auto& successors = state_.depManager.GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (!state_.schedInfoMap[succOp].isRetired) {
            auto& reqMemIds = state_.opReqMemIdsMap[succOp];
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0 &&
                UpdateOperationInput(succOp, spillOp, newTensor, spillMemId) != SUCCESS) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status SpillEngine::UpdateSuccessorDependencies(Operation* succOp, Operation* spillOp, Operation* reloadCopyin,
                                                int spillMemId, int reloadMemId)
{
    auto& reqMemIds = state_.GetOpMemIds(succOp);
    if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0) {
        state_.depManager.InsertSuccessor(reloadCopyin, succOp);
        std::replace(reqMemIds.begin(), reqMemIds.end(), spillMemId, reloadMemId);
        for (auto& outTensor : succOp->GetOOperands()) {
            if (outTensor->memoryrange.memId == spillMemId) {
                outTensor->memoryrange.memId = reloadMemId;
                ReplaceSkipOpChainMemId(outTensor, spillMemId, reloadMemId);
            }
        }
        state_.depManager.RemovePredecessor(succOp, spillOp);
        state_.depManager.InsertPredecessor(succOp, reloadCopyin);
        return UpdateOperationInput(succOp, spillOp, reloadCopyin->GetOutputOperand(0), spillMemId);
    }
    return SUCCESS;
}

void SpillEngine::UpdatePredecessorAllocDependencies(Operation* succOp, Operation* reloadAlloc, int spillMemId)
{
    auto predecessors = state_.depManager.GetPredecessors(succOp);
    for (auto predOp : predecessors) {
        if (state_.schedInfoMap[predOp].isAlloc) {
            auto& predReqMemIds = state_.GetOpMemIds(predOp);
            if (std::find(predReqMemIds.begin(), predReqMemIds.end(), spillMemId) != predReqMemIds.end()) {
                state_.depManager.RemovePredecessor(succOp, predOp);
                state_.depManager.InsertPredecessor(succOp, reloadAlloc);
            }
        }
    }
}

Status SpillEngine::UpdateSmallShapeDependAndBuf(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap,
                                                 int spillMemId, Operation* spillOp)
{
    if (opMemidMap.size() != TWO_ISSUE) {
        APASS_LOG_ERROR_F(Elements::Tensor, "The number of elements in opMemidMap is invalid: %zu.", opMemidMap.size());
        return FAILED;
    }
    Operation* reloadAlloc = opMemidMap[0].first;
    Operation* reloadCopyin = opMemidMap[1].first;
    if (state_.bufRefCount.find(spillMemId) == state_.bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]. ", spillMemId);
        return FAILED;
    }
    int reloadMemId = state_.GetOpMemIds(reloadAlloc)[0];
    state_.bufRefCount[reloadMemId] = TWO_ISSUE;

    auto& successors = state_.depManager.GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (state_.schedInfoMap[succOp].isRetired) {
            continue;
        }
        state_.bufRefCount[reloadMemId]++;
        if (UpdateSuccessorDependencies(succOp, spillOp, reloadCopyin, spillMemId, reloadMemId) != SUCCESS) {
            return FAILED;
        }
        UpdatePredecessorAllocDependencies(succOp, reloadAlloc, spillMemId);
    }
    return SUCCESS;
}

void SpillEngine::CollectUBSceneOpsAndTensors(Operation* producerOp, std::vector<Operation*>& opsToDelete,
                                              std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    opsToDelete.push_back(producerOp);
    auto ubTensor2 = producerOp->GetInputOperand(0);
    if (ubTensor2 == nullptr)
        return;
    for (auto* op : ubTensor2->GetProducers()) {
        if (op != nullptr && op->GetOpcodeStr().find("UB_COPY_ND2NZ") != std::string::npos) {
            if (state_.depManager.GetSuccessors(op).size() > 1) {
                return;
            }
        }
    }
    tensorsToDelete.push_back(ubTensor2);
    for (auto* op : ubTensor2->GetProducers()) {
        if (op != nullptr &&
            (state_.schedInfoMap[op].isAlloc || op->GetOpcodeStr().find("UB_COPY_ND2NZ") != std::string::npos)) {
            opsToDelete.push_back(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "UB scene: collect %s[%d]", op->GetOpcodeStr().c_str(),
                              op->GetOpMagic());
        }
    }
}

void SpillEngine::CollectProducerChainForDeletion(LogicalTensorPtr spillTensor, std::vector<Operation*>& opsToDelete,
                                                  std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    tensorsToDelete.push_back(spillTensor);

    for (auto* producerOp : spillTensor->GetProducers()) {
        if (producerOp == nullptr) {
            continue;
        }
        bool isUBCopyL1 = producerOp->GetOpcode() == Opcode::OP_UB_COPY_L1;
        if (isUBCopyL1) {
            CollectUBSceneOpsAndTensors(producerOp, opsToDelete, tensorsToDelete);
            APASS_LOG_DEBUG_F(Elements::Operation, "UB scene: collect UB tensor and op");
        } else {
            opsToDelete.push_back(producerOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "collect L0C_COPY_L1/L1_ALLOC only");
        }
    }
}

void SpillEngine::ReleaseDeletedOpBufRefs(Operation* op, const std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    if (state_.schedInfoMap[op].isRetired) {
        return;
    }
    for (int memId : state_.GetOpMemIds(op)) {
        bool willErase = false;
        for (const auto& tensor : tensorsToDelete) {
            if (tensor->memoryrange.memId == memId) {
                willErase = true;
                break;
            }
        }
        if (willErase) {
            continue;
        }
        auto refIt = state_.bufRefCount.find(memId);
        if (refIt != state_.bufRefCount.end() && refIt->second > 0) {
            refIt->second--;
        }
    }
}

void SpillEngine::CleanupCollectedTensors(const std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    for (size_t i = 0; i < tensorsToDelete.size(); i++) {
        auto& tensor = tensorsToDelete[i];
        int memId = tensor->memoryrange.memId;

        state_.tensorAllocMap.erase(memId);
        state_.bufRefCount.erase(memId);

        APASS_LOG_DEBUG_F(Elements::Tensor, "Cleaned tensor[%d] scheduler.", memId);
    }
}

void SpillEngine::EraseOrphanedTensors(const std::vector<LogicalTensorPtr>& tensorsToDelete,
                                       const std::vector<Operation*>& opsToDelete)
{
    for (auto& tensor : tensorsToDelete) {
        for (auto* op : opsToDelete) {
            tensor->RemoveProducer(op);
        }
        for (auto* op : opsToDelete) {
            tensor->RemoveConsumer(op);
        }
    }
}

Status SpillEngine::SpillBuffer(int memId, Operation* spillAllocOp, SpillContext& ctx)
{
    Operation* spillOp = GetSpillOp(memId);
    if (spillOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
        return FAILED;
    }
    if (state_.schedInfoMap[spillOp].isAlloc || !CheckMachineAndL1(spillOp, spillAllocOp)) {
        return SUCCESS;
    }
    LogicalTensorPtr spillTensor = GetSpillTensor(spillOp, memId);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Find %s spill tensor[%d] failed.", state_.GetOpInfo(spillOp).c_str(),
                          memId);
        return FAILED;
    }
    SingleSpillCreatedOps created;
    if (HandleSpillMode(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Spill %s tensor[%d] failed.", state_.GetOpInfo(spillOp).c_str(), memId);
        return FAILED;
    }
    NotifySpill(state_, spillTensor, memId, spillAllocOp, created);
    CoreLocationType freeCore = state_.schedInfoMap[spillAllocOp].coreLocation;
    if (state_.enableDualDst) {
        freeCore = state_.ResolveCoreForFree(memId);
    }
    if (state_.bufferManagerMap[freeCore][state_.localBufferMap[memId]->memType].Free(memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", memId);
        return FAILED;
    }
    state_.tensorOccupyMap.erase(memId);
    return SUCCESS;
}

Status SpillEngine::HandleSpillMode(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                    Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Begin spill %s, Tensor[%d]", state_.GetOpInfo(spillOp).c_str(), memId);
    if (spillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        if (SpillBufferFromDDR(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillBufferFromDDR failed!");
            return FAILED;
        }
    } else if (state_.localBufferMap[memId]->memType == MemoryType::MEM_L1 &&
               Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        if (SpillL1BufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillL1BufferFor3510 failed!");
            return FAILED;
        }
    } else if (IsMultiProducerTensor(spillTensor)) {
        if (SpillMultiProducerBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiProducerBuffer failed!");
            return FAILED;
        }
    } else if (state_.localBufferMap[memId]->memType == MemoryType::MEM_L0C) {
        if (SpillL0CBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillL0CBuffer failed!");
            return FAILED;
        }
    } else {
        if (SpillGeneralBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillGeneralBuffer failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SpillEngine::SpillBufferFromDDR(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                       Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillBufferFromDDR begin.");
    LogicalTensorPtr gmTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);
    Operation* allocOp = CreateAllocOp(localTensor);
    Operation* copyinOp = CloneCopyinOp(spillOp, gmTensor, localTensor);

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {{allocOp, {localTensor->memoryrange.memId}},
                                                                       {copyinOp, {localTensor->memoryrange.memId}}};

    if (UpdateScheduleStatus(opMemidMap, memId, spillAllocOp, localTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newAllocOps.push_back(allocOp);
    created.Record(nullptr, allocOp, copyinOp, nullptr);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillBufferFromDDR end.");
    return SUCCESS;
}

Status SpillEngine::SpillGeneralBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                       Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralBuffer begin.");
    if (GetActualSpillForNd2nz(spillOp, spillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpillForNd2nz failed.");
        return FAILED;
    }

    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillTensor, spillMemId);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);

    Operation* copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor,
                                           OpImmediate::Specified(gmTensor->GetOffset()));
    Operation* allocOp = CreateAllocOp(localTensor);
    Operation* copyinOp = CreateCopyinOp(gmTensor, localTensor, OpImmediate::Specified(gmTensor->GetOffset()));

    if (UpdateCopyoutScheduleInfo(copyoutOp, spillTensor, spillMemId, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
        return FAILED;
    }

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {{allocOp, {localTensor->memoryrange.memId}},
                                                                       {copyinOp, {localTensor->memoryrange.memId}}};

    if (UpdateScheduleStatus(opMemidMap, spillMemId, spillAllocOp, localTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralBuffer end.");
    return SUCCESS;
}

Status SpillEngine::SpillMultiProducerBufferFor3510(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor,
                                                    Operation* spillAllocOp, SpillContext& ctx,
                                                    SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillMultiProducerBufferFor3510 begin.");
    Operation* actualTriggerOp = nullptr;
    LogicalTensorPtr actualTriggerTensor = nullptr;
    if (GetActualSpill(spillOp, actualTriggerOp, actualTriggerTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
        return FAILED;
    }
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, actualTriggerTensor, spillMemid);
    LogicalTensorPtr l1Tensor = CreateLocalTensor(spillTensor);
    if (CopyoutParticalBuffer(spillTensor, gmTensor, ctx) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CopyoutParticalBuffer failed.");
        return FAILED;
    }
    Operation* allocOp = CreateAllocOp(l1Tensor);
    Operation* copyinOp = CreateCopyinOp(gmTensor, l1Tensor, OpImmediate::Specified(gmTensor->GetOffset()));

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {{allocOp, {l1Tensor->memoryrange.memId}},
                                                                       {copyinOp, {l1Tensor->memoryrange.memId}}};

    if (!IsUnusedTensor(spillOp)) {
        if (UpdateScheduleStatus(opMemidMap, spillMemid, spillAllocOp, l1Tensor, spillOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
            return FAILED;
        }
    } else {
        if (UpdateNeedDeleteScheduleStatus(opMemidMap, spillMemid, spillAllocOp, spillTensor, spillOp, ctx) !=
            SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateNeedDeleteScheduleStatus failed.");
            return FAILED;
        }
    }
    ctx.newAllocOps.push_back(allocOp);
    created.Record(nullptr, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillMultiProducerBufferFor3510 end.");
    return SUCCESS;
}

Status SpillEngine::SpillL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                         Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created)
{
    if (IsMultiProducerTensor(spillTensor)) {
        if (SpillMultiProducerBufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx, created) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiProducerBufferFor3510 failed.");
            return FAILED;
        }
    } else if (spillOp->GetOpcode() != Opcode::OP_RESHAPE) {
        if (spillOp->GetIOperands().size() == 1) {
            SpillGeneralL1BufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx, created);
        } else {
            return FAILED;
        }
    } else {
        Operation* actualSpillOp = nullptr;
        for (auto& preOp : state_.depManager.GetPredecessors(spillOp)) {
            if (!state_.schedInfoMap[preOp].isAlloc) {
                actualSpillOp = preOp;
            }
        }
        if (actualSpillOp == nullptr || actualSpillOp->GetIOperands().size() != 1) {
            return FAILED;
        }
        if (actualSpillOp->GetOpcode() == Opcode::OP_COPY_IN) {
            SpillReshapeFromDDRFor3510(memId, actualSpillOp, spillOp, spillTensor, spillAllocOp, ctx, created);
        } else {
            SpillReshapeL1BufferFor3510(memId, actualSpillOp, spillOp, spillTensor, spillAllocOp, ctx, created);
        }
    }
    return SUCCESS;
}

Status SpillEngine::SpillGeneralL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                                Operation* spillAllocOp, SpillContext& ctx,
                                                SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralL1BufferFor3510 begin.");
    Operation* actualOp = nullptr;
    LogicalTensorPtr actualSpillTensor = nullptr;
    if (GetActualSpill(spillOp, actualOp, actualSpillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
        return FAILED;
    }
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillTensor, actualSpillTensor, memId, spillTensor->Datatype());
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);

    Operation* copyoutOp = CreateCopyoutOp(spillOp, actualSpillTensor, gmTensor,
                                           OpImmediate::Specified(gmTensor->GetOffset()));
    Operation* allocOp = CreateAllocOp(localTensor);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(spillOp->GetOpAttribute());
    if (attr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Op %s attribute is nullptr", state_.GetOpInfo(spillOp).c_str());
        return FAILED;
    }
    Operation* copyinOp = CreateCopyinOp(gmTensor, localTensor, attr->GetFromOffset());

    if (UpdateCopyoutScheduleInfo(copyoutOp, actualSpillTensor, actualSpillTensor->memoryrange.memId, actualOp) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
        return FAILED;
    }

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {{allocOp, {localTensor->memoryrange.memId}},
                                                                       {copyinOp, {localTensor->memoryrange.memId}}};

    if (UpdateScheduleStatus(opMemidMap, memId, spillAllocOp, localTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillGeneralL1BufferFor3510 end.");
    return SUCCESS;
}

Status SpillEngine::SpillReshapeFromDDRFor3510(int memId, Operation* actualSpillOp, Operation* spillOp,
                                               LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext& ctx,
                                               SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeFromDDRFor3510 begin.");
    LogicalTensorPtr preSpillTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr ddrTensor = actualSpillOp->GetInputOperand(0);
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);
    LogicalTensorPtr copyinTensor = CreateParticalTensor(preSpillTensor, reshapeTensor, preSpillTensor,
                                                         preSpillTensor->GetOffset());

    Operation* allocOp = CreateAllocOp(copyinTensor);
    Operation* copyinOp = CloneCopyinOp(actualSpillOp, ddrTensor, copyinTensor);
    Operation* reshapeOp = CreateReshapeOp(copyinTensor, reshapeTensor);

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {reshapeTensor->memoryrange.memId}},
        {copyinOp, {reshapeTensor->memoryrange.memId}},
        {reshapeOp, {reshapeTensor->memoryrange.memId, reshapeTensor->memoryrange.memId}}};

    if (UpdateScheduleStatus(opMemidMap, memId, spillAllocOp, reshapeTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newAllocOps.push_back(allocOp);
    created.Record(nullptr, allocOp, copyinOp, nullptr);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeFromDDRFor3510 end.");
    return SUCCESS;
}

Status SpillEngine::SpillReshapeL1BufferFor3510(int spillMemId, Operation* actualSpillOp, Operation* spillOp,
                                                LogicalTensorPtr spillTensor, Operation* spillAllocOp,
                                                SpillContext& ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeL1BufferFor3510 begin.");
    LogicalTensorPtr preSpillTensor = spillOp->GetInputOperand(0);
    Operation* actualOp = nullptr;
    LogicalTensorPtr actualSpillTensor = nullptr;
    if (GetActualSpill(actualSpillOp, actualOp, actualSpillTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
        return FAILED;
    }
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillTensor, actualSpillTensor, spillMemId,
                                               preSpillTensor->Datatype());
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);
    LogicalTensorPtr l1Tensor = CreateParticalTensor(preSpillTensor, reshapeTensor, preSpillTensor,
                                                     preSpillTensor->GetOffset());

    Operation* copyoutOp = CreateCopyoutOp(actualSpillOp, actualOp->GetInputOperand(0), gmTensor,
                                           OpImmediate::Specified(gmTensor->GetOffset()));

    Operation* allocOp = CreateAllocOp(l1Tensor);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(actualSpillOp->GetOpAttribute());
    if (attr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Op %s attribute is nullptr", state_.GetOpInfo(actualSpillOp).c_str());
        return FAILED;
    }
    Operation* copyinOp = CreateCopyinOp(gmTensor, l1Tensor, attr->GetFromOffset());
    Operation* reshapeOp = CreateReshapeOp(l1Tensor, reshapeTensor);

    if (UpdateCopyoutScheduleInfo(copyoutOp, actualSpillTensor, actualSpillTensor->memoryrange.memId, actualSpillOp) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
        return FAILED;
    }

    std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap = {
        {allocOp, {l1Tensor->memoryrange.memId}},
        {copyinOp, {l1Tensor->memoryrange.memId}},
        {reshapeOp, {l1Tensor->memoryrange.memId, l1Tensor->memoryrange.memId}}};

    if (UpdateScheduleStatus(opMemidMap, spillMemId, spillAllocOp, reshapeTensor, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateScheduleStatus failed.");
        return FAILED;
    }
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, copyinOp, gmTensor);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillReshapeL1BufferFor3510 end.");
    return SUCCESS;
}

Status SpillEngine::SpillL0CBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor,
                                   Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillL0CBuffer begin.");
    std::vector<Operation*> consumers;
    CollectL0CConsumers(spillTensor, consumers);

    std::map<DataType, std::vector<Operation*>> dtypeGroups;
    for (auto* consumer : consumers) {
        auto oOperand = consumer->GetOutputOperand(0);
        if (oOperand == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "L0C spill: consumer %s has no output operand.",
                              state_.GetOpInfo(consumer).c_str());
            return FAILED;
        }
        dtypeGroups[oOperand->Datatype()].push_back(consumer);
    }

    auto emitGroup = [&](DataType dtype, const std::vector<Operation*>& groupConsumers) -> Status {
        LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillTensor, spillMemId, dtype);
        Operation* copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor,
                                               OpImmediate::Specified(gmTensor->GetOffset()));
        if (UpdateCopyoutScheduleInfo(copyoutOp, spillTensor, spillMemId, spillOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed for dtype group.");
            return FAILED;
        }
        for (auto* consumer : groupConsumers) {
            auto oOperand = consumer->GetOutputOperand(0);
            Operation* copyinOp = CreateCopyinOp(gmTensor, oOperand, OpImmediate::Specified(gmTensor->GetOffset()),
                                                 true);
            UpdateOpScheduleInfo(copyinOp, {oOperand->memoryrange.memId}, spillAllocOp);
            std::replace(state_.orderedOps.begin(), state_.orderedOps.end(), consumer, copyinOp);
            APASS_LOG_DEBUG_F(Elements::Operation, "L0C spill: replace %s with %s.", state_.GetOpInfo(consumer).c_str(),
                              state_.GetOpInfo(copyinOp).c_str());
            consumer->SetAsDeleted();
            EraseSchedulerSideMaps(consumer);
        }
        ctx.newCopyoutOps.push_back(copyoutOp);
        created.Record(copyoutOp, nullptr, nullptr, gmTensor);
        return SUCCESS;
    };

    for (auto& [dtype, groupConsumers] : dtypeGroups) {
        if (emitGroup(dtype, groupConsumers) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillL0CBuffer: emit dtype group failed.");
            return FAILED;
        }
    }
    state_.depManager.InitDependencies(state_.orderedOps, false);
    state_.bufRefCount[spillMemId] = 0;
    function_.EraseOperations(false, false);
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillL0CBuffer end.");
    return SUCCESS;
}

Status SpillEngine::SpillMultiProducerBuffer(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor,
                                             Operation* spillAllocOp, SpillContext& ctx, SingleSpillCreatedOps& created)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "---- SpillMultiProducerBuffer begin.");
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillTensor, spillMemid);
    LogicalTensorPtr assembleTensor = CreateLocalTensor(spillTensor);

    Operation* copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor,
                                           OpImmediate::Specified(gmTensor->GetOffset()));

    if (UpdateCopyoutScheduleInfo(copyoutOp, spillTensor, spillMemid, spillOp) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, assembleTensor, spillMemid) != SUCCESS) {
        return FAILED;
    }

    for (auto& op : spillTensor->GetProducers()) {
        if (op->GetOpcode() != Opcode::OP_ASSEMBLE)
            continue;
        for (auto& producer : op->ProducerOps()) {
            if (state_.schedInfoMap[producer].isAlloc)
                producer->UpdateOutputOperand(0, spillTensor);
        }
    }
    Operation* allocOp = CreateAllocOp(assembleTensor);
    UpdateOpScheduleInfo(allocOp, {assembleTensor->memoryrange.memId}, spillAllocOp);
    if (InsertOps({{allocOp, {assembleTensor->memoryrange.memId}}}, spillAllocOp, spillMemid) != SUCCESS) {
        return FAILED;
    }
    Operation* wholeCopyinOp = nullptr;
    if (FillSpillAssembleBuffer(spillMemid, spillTensor, assembleTensor, copyoutOp, gmTensor, spillAllocOp,
                                wholeCopyinOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FillSpillAssembleBuffer failed.");
        return FAILED;
    }

    if (UpdateRemainMemid(spillMemid, assembleTensor->memoryrange.memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    state_.depManager.InitDependencies(state_.orderedOps, false);
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    created.Record(copyoutOp, allocOp, wholeCopyinOp, gmTensor);
    return SUCCESS;
}

Status SpillEngine::FillSpillAssembleBuffer(int spillMemid, LogicalTensorPtr spillTensor,
                                            LogicalTensorPtr assembleTensor, Operation* copyoutOp,
                                            LogicalTensorPtr gmTensor, Operation* spillAllocOp,
                                            Operation*& wholeCopyinOut)
{
    wholeCopyinOut = nullptr;
    bool allRetired = true;
    for (auto& op : spillTensor->GetProducers()) {
        if (state_.schedInfoMap[op].isAlloc) {
            continue;
        }
        if (!state_.schedInfoMap[op].isRetired) {
            allRetired = false;
            break;
        }
    }
    if (allRetired) {
        wholeCopyinOut = CreateCopyinOp(gmTensor, assembleTensor, OpImmediate::Specified(gmTensor->GetOffset()));
        UpdateOpScheduleInfo(wholeCopyinOut, {assembleTensor->memoryrange.memId}, spillAllocOp);
        return InsertOps({{wholeCopyinOut, {assembleTensor->memoryrange.memId}}}, spillAllocOp, spillMemid);
    }
    std::vector<Operation*> replaceOps;
    for (auto& op : spillTensor->GetProducers()) {
        if (state_.schedInfoMap[op].isAlloc) {
            continue;
        }
        if (state_.schedInfoMap[op].isRetired) {
            CreateParticalBuffer(spillMemid, op, assembleTensor, copyoutOp, spillAllocOp);
        } else {
            replaceOps.push_back(op);
        }
    }
    for (auto& op : replaceOps) {
        op->ReplaceOutput(assembleTensor, spillTensor);
    }
    return SUCCESS;
}

Status SpillEngine::CopyoutParticalBuffer(LogicalTensorPtr spillTensor, LogicalTensorPtr gmTensor, SpillContext& ctx)
{
    for (auto& op : spillTensor->GetProducers()) {
        if (state_.schedInfoMap[op].isAlloc) {
            continue;
        }
        Operation* actualOp = nullptr;
        LogicalTensorPtr actualTensor = nullptr;
        if (GetActualSpill(op, actualOp, actualTensor) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetActualSpill failed.");
            return FAILED;
        }
        auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Op %s attribute is nullptr", state_.GetOpInfo(op).c_str());
            return FAILED;
        }
        Operation* copyoutOp = CreateCopyoutOp(op, actualTensor, gmTensor, attr->GetToOffset());
        if (UpdateCopyoutScheduleInfo(copyoutOp, actualTensor, actualTensor->memoryrange.memId, actualOp,
                                      state_.schedInfoMap[op].isRetired) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyoutScheduleInfo failed.");
            return FAILED;
        }
        if (!state_.schedInfoMap[op].isRetired) {
            state_.bufRefCount[actualTensor->memoryrange.memId]++;
            ctx.newNotRetiredCopyOutSize++;
        } else {
            ctx.newCopyoutOps.push_back(copyoutOp);
        }
    }
    return SUCCESS;
}

Status SpillEngine::CreateParticalBuffer(int spillMemid, Operation* producerOp, LogicalTensorPtr assembleTensor,
                                         Operation* copyoutOp, Operation* spillAllocOp)
{
    LogicalTensorPtr gmTensor = copyoutOp->GetOutputOperand(0);
    LogicalTensorPtr spillTensor = copyoutOp->GetInputOperand(0);

    std::vector<int64_t> toOffset;
    std::vector<SymbolicScalar> toDynOffset;
    std::vector<SymbolicScalar> fromDynValidShape;
    if (GetPartialWriteReplayAttr(producerOp, toOffset, toDynOffset, fromDynValidShape) != SUCCESS) {
        return FAILED;
    }

    LogicalTensorPtr copyinTensor = CreateParticalTensor(producerOp->GetInputOperand(0), assembleTensor,
                                                         producerOp->GetInputOperand(0), toOffset);
    Operation* copyinOp = CreateCopyinOp(gmTensor, copyinTensor, OpImmediate::Specified(toOffset));
    Operation* assembleOp = CreateAssembleOp(copyinTensor, assembleTensor, toOffset, toDynOffset, fromDynValidShape);

    int64_t isNZ = 0;
    producerOp->GetAttr(OpAttributeKey::copyIsNZ, isNZ);
    copyinOp->SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    assembleOp->SetAttr(OpAttributeKey::copyIsNZ, isNZ);
    UpdateOpScheduleInfo(copyinOp, {assembleTensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(assembleOp, {assembleTensor->memoryrange.memId, assembleTensor->memoryrange.memId},
                         spillAllocOp);
    if (InsertOps({{copyinOp, {assembleTensor->memoryrange.memId}},
                   {assembleOp, {assembleTensor->memoryrange.memId, assembleTensor->memoryrange.memId}}},
                  spillAllocOp, spillMemid) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertOps failed.");
        return FAILED;
    }
    return SUCCESS;
}

void SpillEngine::EraseSchedulerSideMaps(Operation* op)
{
    auto it = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), op);
    if (it != state_.orderedOps.end()) {
        state_.orderedOps.erase(it);
    }
    state_.schedInfoMap.erase(op);
    state_.opReqMemIdsMap.erase(op);
    state_.inOutOperandsCache.erase(op);
    state_.depManager.RemoveSuccessorOp(op);
    state_.depManager.RemovePredecessorOp(op);
}

int SpillEngine::GetBufNextUseTime(int curMemId)
{
    for (size_t i = 0; i < state_.orderedOps.size(); i++) {
        auto& op = state_.orderedOps[i];
        if (state_.schedInfoMap[op].isRetired)
            continue;
        auto& reqMemids = state_.GetOpMemIds(op);
        if (std::find(reqMemids.begin(), reqMemids.end(), curMemId) == reqMemids.end())
            continue;

        int opExecOrder = state_.schedInfoMap[op].execOrder;
        int minAlloc = INT_MAX;
        int allocCnt = 0;
        for (auto pre : state_.depManager.GetPredecessors(op)) {
            if (state_.schedInfoMap[pre].isRetired)
                continue;
            if (state_.schedInfoMap[pre].isAlloc) {
                int preOrder = state_.schedInfoMap[pre].execOrder;
                if (preOrder < minAlloc)
                    minAlloc = preOrder;
                allocCnt++;
            }
        }
        if (allocCnt == 0) {
            return opExecOrder;
        }
        return (opExecOrder - minAlloc <= allocCnt) ? minAlloc : opExecOrder;
    }
    return -1;
}

Status SpillEngine::UpdateCopyoutScheduleInfo(Operation* op, LogicalTensorPtr spillTensor, int spillMemId,
                                              Operation* spillOp, bool isRetired)
{
    state_.opReqMemIdsMap[op] = {spillMemId};
    state_.schedInfoMap[op].isRetired = isRetired;
    state_.schedInfoMap[op].isAlloc = false;
    state_.schedInfoMap[op].pipeType = RescheduleUtils::GetOpPipeType(op);
    state_.depManager.RegisterOp(op);
    Operation* allocOp = state_.tensorAllocMap[spillTensor->memoryrange.memId];
    state_.schedInfoMap[op].coreLocation = state_.schedInfoMap[allocOp].coreLocation;
    UpdateOpInternalSubgraphID(*op, allocOp);
    int bufNextUseTime = state_.schedInfoMap[spillOp].execOrder;
    for (auto succOp : state_.depManager.GetSuccessors(spillOp)) {
        if (!state_.schedInfoMap[succOp].isRetired)
            continue;
        if (succOp == op)
            continue;
        if (succOp->GetOpcodeStr().find("COPY_OUT") != std::string::npos) {
            bufNextUseTime = std::max(bufNextUseTime, state_.schedInfoMap[succOp].execOrder);
        }
    }
    state_.schedInfoMap[op].execOrder = bufNextUseTime + 1;
    state_.InsertOrdered(op);
    return SUCCESS;
}

void SpillEngine::UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp)
{
    state_.schedInfoMap[op].pipeType = RescheduleUtils::GetOpPipeType(op);
    state_.schedInfoMap[op].isAlloc = op->GetOpcodeStr().find("ALLOC") != std::string::npos;
    state_.schedInfoMap[op].isRetired = false;
    state_.opReqMemIdsMap[op] = memIds;
    state_.depManager.RegisterOp(op);
    state_.schedInfoMap[op].coreLocation = state_.schedInfoMap[AllocOp].coreLocation;
    UpdateOpInternalSubgraphID(*op, AllocOp);
    state_.numTotalIssues++;
}

Status SpillEngine::InsertOps(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, Operation* spillAllocOp,
                              int memId)
{
    if (memId == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "MemId: %d illegal.", memId);
        return FAILED;
    }
    int bufNextUseTime = GetBufNextUseTime(memId);
    if (bufNextUseTime == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use time failed.", memId);
        return FAILED;
    }
    bufNextUseTime = bufNextUseTime <= state_.schedInfoMap[spillAllocOp].execOrder ?
                         state_.schedInfoMap[spillAllocOp].execOrder + 1 :
                         bufNextUseTime;
    for (auto& op : opMemidMap) {
        state_.schedInfoMap[op.first].execOrder = bufNextUseTime++;
        state_.InsertOrdered(op.first);
    }
    return SUCCESS;
}

Status SpillEngine::UpdateScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
                                         Operation* spillAllocOp, LogicalTensorPtr localTensor, Operation* spillOp)
{
    CoreLocationType coreLocation = state_.schedInfoMap[spillAllocOp].coreLocation;
    if (!state_.enableDualDst || (coreLocation != CoreLocationType::AIV0 && coreLocation != CoreLocationType::AIV1)) {
        for (auto& [op, memid] : opMemidMap) {
            UpdateOpScheduleInfo(op, memid, spillAllocOp);
        }
    } else {
        UpdateScheduleStatusForDualDst(opMemidMap, memId, spillAllocOp, spillOp);
    }

    if (InsertOps(opMemidMap, spillAllocOp, memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertOps failed.");
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, localTensor, memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateSpillOpDepend failed.");
        return FAILED;
    }
    if (UpdateRemainMemid(memId, localTensor->memoryrange.memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    state_.depManager.InitDependencies(state_.orderedOps, false);
    return SUCCESS;
}

Status SpillEngine::UpdateNeedDeleteScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap,
                                                   int memId, Operation* spillAllocOp, LogicalTensorPtr spillTensor,
                                                   Operation* spillOp, SpillContext& ctx)
{
    for (auto& [op, memid] : opMemidMap) {
        UpdateOpScheduleInfo(op, memid, spillAllocOp);
    }

    if (UpdateSmallShapeDependAndBuf(opMemidMap, memId, spillOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateSmallShapeDependAndBuf failed");
        return FAILED;
    }
    if (RemoveSmallShapeSpillResources(memId, spillTensor, ctx) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RemoveSmallShapeSpillResources failed");
        return FAILED;
    }

    int newMemid = -1;
    for (auto& op : opMemidMap) {
        if (state_.schedInfoMap[op.first].isAlloc) {
            newMemid = op.first->GetOutputOperand(0)->memoryrange.memId;
        }
    }
    if (InsertOps(opMemidMap, spillAllocOp, newMemid) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertOps failed.");
        return FAILED;
    }
    state_.depManager.InitDependencies(state_.orderedOps, false);
    return SUCCESS;
}

Status SpillEngine::UpdateRemainMemid(int oldMemId, int newMemId)
{
    if (state_.bufRefCount.find(oldMemId) == state_.bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]. ", oldMemId);
        return FAILED;
    }
    state_.bufRefCount[newMemId] = 0;
    state_.bufRefCount[oldMemId] = 0;
    for (auto& op : state_.orderedOps) {
        if (state_.schedInfoMap[op].isRetired) {
            continue;
        }
        ReplaceTensorMemId(op, oldMemId, newMemId);
    }
    return SUCCESS;
}

size_t SpillEngine::CleanupCollectedOperations(const std::vector<Operation*>& opsToDelete,
                                               const std::vector<LogicalTensorPtr>& tensorsToDelete)
{
    size_t deleteNum = 0;
    for (auto* op : opsToDelete) {
        if (op == nullptr) {
            continue;
        }
        auto it = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), op);
        if (it != state_.orderedOps.end()) {
            size_t opIndex = std::distance(state_.orderedOps.begin(), it);
            if (state_.schedInfoMap[op].isRetired) {
                deleteNum++;
            }
            int deletedOrder = state_.schedInfoMap[op].execOrder;

            auto nextIt = state_.orderedOps.erase(it);

            for (auto adjustIt = nextIt; adjustIt != state_.orderedOps.end(); adjustIt++) {
                if (state_.schedInfoMap.count(*adjustIt) > 0 &&
                    state_.schedInfoMap[*adjustIt].execOrder > deletedOrder) {
                    state_.schedInfoMap[*adjustIt].execOrder--;
                }
            }

            APASS_LOG_DEBUG_F(Elements::Operation, "Deleted op %s at index %zu (order %d).",
                              state_.GetOpInfo(op).c_str(), opIndex, deletedOrder);
        }
        ReleaseDeletedOpBufRefs(op, tensorsToDelete);
        EraseSchedulerSideMaps(op);

        auto predecessors = state_.depManager.GetPredecessors(op);
        auto successors = state_.depManager.GetSuccessors(op);
        for (auto* pred : predecessors) {
            state_.depManager.RemoveSuccessor(pred, op);
        }
        for (auto* succ : successors) {
            state_.depManager.RemovePredecessor(succ, op);
        }
        auto newOpsIt = std::find(state_.newOperations.begin(), state_.newOperations.end(), op);
        if (newOpsIt != state_.newOperations.end()) {
            state_.newOperations.erase(newOpsIt);
            APASS_LOG_DEBUG_F(Elements::Operation, "Removed op %s from newOperations.", state_.GetOpInfo(op).c_str());
        }
        op->SetAsDeleted();
        APASS_LOG_DEBUG_F(Elements::Operation, "Marked op %s as deleted.", state_.GetOpInfo(op).c_str());
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Delete pcidx num: %zu", deleteNum);
    return deleteNum;
}

Status SpillEngine::RemoveSmallShapeSpillResources(int spillMemId, LogicalTensorPtr spillTensor, SpillContext& ctx)
{
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "spillTensor is null.");
        return FAILED;
    }

    std::vector<Operation*> opsToDelete;
    std::vector<LogicalTensorPtr> tensorsToDelete;
    CollectProducerChainForDeletion(spillTensor, opsToDelete, tensorsToDelete);
    APASS_LOG_DEBUG_F(Elements::Operation, "Collected %zu ops and %zu tensors.", opsToDelete.size(),
                      tensorsToDelete.size());
    for (auto deleteOp : opsToDelete) {
        if (state_.schedInfoMap[deleteOp].isAlloc) {
            ctx.deleteAllocOps.push_back({deleteOp, deleteOp->GetOutputOperand(0)->GetMemoryTypeOriginal(),
                                          state_.schedInfoMap[deleteOp].coreLocation});
        }
    }
    auto deleteNum = CleanupCollectedOperations(opsToDelete, tensorsToDelete);
    if (deleteNum > opsToDelete.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Delete number greater than totalDeleteNumber");
        return FAILED;
    }
    CleanupCollectedTensors(tensorsToDelete);

    function_.EraseOperations(false, true);

    EraseOrphanedTensors(tensorsToDelete, opsToDelete);

    ctx.deleteRetiredOpSize += deleteNum;
    ctx.deleteNotRetiredOpSize = static_cast<int>(opsToDelete.size() - deleteNum);
    APASS_LOG_DEBUG_F(Elements::Operation, "Deleted spill tensor[%d] and %zu ops (%zu tensors).", spillMemId,
                      opsToDelete.size(), tensorsToDelete.size());

    return SUCCESS;
}

// ============================================================================
// DualDst 跨池 spill 专用，主线 enableDualDst_=false 时不会进入。
// ============================================================================

// UpdateScheduleStatus 的 dualdst 分支。入口约定 (调用方保证): spillAllocOp 是 dualdst
// dualdst 保留的 A（跨池场景）。
//
// 关键性质: spilled tensor 必然是 UB,
// 它的消费者必然在 AIV0/AIV1，直接拿消费者的属性即对。通过 GetSpillTensor(spillOp, memId)
// 反查 spillTensor 拿到消费者列表。
void SpillEngine::UpdateScheduleStatusForDualDst(const std::vector<std::pair<Operation*, std::vector<int>>>& opMemidMap,
                                                 int memId, Operation* spillAllocOp, Operation* spillOp)
{
    LogicalTensorPtr spillTensor = GetSpillTensor(spillOp, memId);
    Operation* consumer = nullptr;
    if (spillTensor != nullptr) {
        for (auto* cons : spillTensor->GetConsumers()) {
            if (cons == nullptr || cons->IsDeleted())
                continue;
            if (cons->HasAttribute(OpAttributeKey::isCube) && cons->GetBoolAttribute(OpAttributeKey::isCube))
                continue;
            consumer = cons;
            break;
        }
    }

    for (auto& [op, memid] : opMemidMap) {
        UpdateOpScheduleInfo(op, memid, spillAllocOp); // 先走原始继承 (拿到默认值)
        if (consumer != nullptr) {
            // 覆写核、subgraph 和 aivCore，全部从消费者继承。
            state_.schedInfoMap[op].coreLocation = state_.schedInfoMap[consumer].coreLocation;
            UpdateOpInternalSubgraphID(*op, consumer);
        }
        // 没找到消费者就保留 spillAllocOp 继承的默认值（极罕见，dualdst 触发但 UB 无消费方）。
    }
}

} // namespace npu::tile_fwk
