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
 * \file ooo_scheduler_notify.cpp
 * \brief Observer notification helpers split from ooo_scheduler.cpp.
 */

#include "passes/block_graph_pass/schedule_ooo/post_schedule/ooo_scheduler.h"
#include "tilefwk/symbolic_scalar.h"

namespace npu::tile_fwk {

namespace {

std::vector<std::string> FormatShape(const LogicalTensorPtr& tensor)
{
    std::vector<std::string> result;
    const auto& dynShape = tensor->GetDynValidShape();
    if (!dynShape.empty()) {
        for (const auto& s : dynShape) {
            result.push_back(s.Dump());
        }
        return result;
    }
    for (auto d : tensor->GetShape()) {
        result.push_back(std::to_string(d));
    }
    return result;
}

} // namespace

CoreLocation OoOScheduler::ToCoreLocation(CoreLocationType c)
{
    switch (c) {
        case CoreLocationType::AIC:
            return {CoreClass::AIC, 0};
        case CoreLocationType::AIV0:
            return {CoreClass::AIV, 0};
        case CoreLocationType::AIV1:
            return {CoreClass::AIV, 1};
        default:
            return {CoreClass::UNKNOWN, 0};
    }
}

void OoOScheduler::NotifyOpLaunch(Operation* op, int cycleEnd)
{
    if (observers_.empty())
        return;
    OpLaunchEvent event;
    event.clock = state_.clock;
    event.cycleEnd = cycleEnd;
    event.opMagic = op->GetOpMagic();
    event.pipeType = state_.schedInfoMap[op].pipeType;
    event.coreLocation = ToCoreLocation(state_.schedInfoMap[op].coreLocation);
    for (size_t i = 0; i < op->GetIOperands().size(); ++i) {
        event.inputMemIds.push_back(op->GetInputOperand(i)->memoryrange.memId);
    }
    for (size_t i = 0; i < op->GetOOperands().size(); ++i) {
        event.outputMemIds.push_back(op->GetOutputOperand(i)->memoryrange.memId);
    }
    event.ddrRefs = BuildDDRRefs(op);
    for (auto* obs : observers_) {
        obs->OnOpLaunch(event);
    }
}

void OoOScheduler::NotifyOpRetire(Operation* op, const std::vector<int>& freedMemIds)
{
    if (observers_.empty())
        return;
    if (state_.schedInfoMap[op].isAlloc)
        return; // alloc retire only awakens successors, no OP_RETIRE event
    OpRetireEvent event{state_.clock, op->GetOpMagic(), freedMemIds};
    for (auto* obs : observers_) {
        obs->OnOpRetire(event);
    }
}

void OoOScheduler::NotifyAllocExec(Operation* op, int memId)
{
    if (observers_.empty())
        return;
    auto& buf = state_.localBufferMap.at(memId);
    AllocExecEvent event;
    event.clock = state_.clock;
    event.memId = memId;
    event.memType = buf->memType;
    event.coreLocation = ToCoreLocation(state_.schedInfoMap[op].coreLocation);
    event.addrStart = buf->start;
    event.addrEnd = buf->end;
    uint64_t validSize = buf->size;
    for (auto& oOp : op->GetOOperands()) {
        if (oOp == nullptr)
            continue;
        event.logicalTensors.push_back({oOp->GetMagic(), validSize});
    }

    for (auto* obs : observers_) {
        obs->OnAllocExec(event);
    }
}

void OoOScheduler::NotifyBufferRearrange(Operation* triggerOp, MemoryType memType,
                                         std::vector<BufferRearrangeEvent::Change> changes)
{
    if (observers_.empty() || changes.empty())
        return;
    BufferRearrangeEvent event;
    event.clock = state_.clock;
    event.memType = memType;
    event.coreLocation = ToCoreLocation(state_.schedInfoMap[triggerOp].coreLocation);
    event.triggerOpMagic = triggerOp->GetOpMagic();
    event.changes = std::move(changes);
    for (auto* obs : observers_) {
        obs->OnBufferRearrange(event);
    }
}

void OoOScheduler::NotifyAllocFail(Operation* triggerOp, MemoryType memType, uint64_t requestSize)
{
    if (observers_.empty())
        return;
    auto coreLocation = state_.schedInfoMap[triggerOp].coreLocation;
    auto& pool = state_.bufferManagerMap[coreLocation][memType];

    AllocFailEvent event;
    event.clock = state_.clock;
    event.triggerOpMagic = triggerOp->GetOpMagic();
    event.memType = memType;
    event.coreLocation = ToCoreLocation(coreLocation);
    event.requestSize = requestSize;
    event.capacity = pool.GetMemSize();

    for (int memId : pool.GetAddrSortedBufs()) {
        AllocFailEvent::OccupiedSlice slice;
        slice.memId = memId;
        slice.addrStart = pool.GetBufferOffset(memId);
        slice.size = pool.GetBufferSize(memId);
        slice.addrEnd = slice.addrStart + slice.size;
        auto it = state_.tensorOccupyMap.find(memId);
        slice.ownerOpMagic = (it != state_.tensorOccupyMap.end()) ? it->second->GetOpMagic() : -1;
        event.occupiedSlices.push_back(slice);
    }

    // BufferPool::FindFreeIntervals returns map<intervalSize, map<start, end>>.
    // Flatten into a plain list of {start, size} for the event.
    auto freeIntervalMap = pool.FindFreeIntervals();
    for (auto& [intervalSize, intervals] : freeIntervalMap) {
        (void)intervalSize;
        for (auto& [start, end] : intervals) {
            event.freeIntervals.push_back({start, end - start});
        }
    }

    for (auto* obs : observers_) {
        obs->OnAllocFail(event);
    }
}

void OoOScheduler::NotifyScheduleEnd(bool success)
{
    if (observers_.empty())
        return;
    ScheduleEndEvent event{state_.clock, state_.workspaceOffset, success};
    for (auto* obs : observers_) {
        obs->OnScheduleEnd(event);
    }
}

void OoOScheduler::NotifyMainLoopBegin()
{
    for (auto* obs : observers_) {
        obs->OnMainLoopBegin();
    }
}

void OoOScheduler::NotifyMainLoopEnd()
{
    for (auto* obs : observers_) {
        obs->OnMainLoopEnd();
    }
}

void OoOScheduler::EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind)
{
    if (t == nullptr)
        return;
    int memId = t->memoryrange.memId;
    if (ddrKindMap_.count(memId) != 0)
        return;
    ddrKindMap_[memId] = kind;
    if (observers_.empty())
        return;
    InitDDRBufferEvent event;
    event.clock = -1;
    event.memId = memId;
    event.kind = kind;
    event.magic = t->GetMagic();
    event.dtype = t->Datatype();
    event.shape = FormatShape(t);
    for (auto* obs : observers_) {
        obs->OnInitDDRBuffer(event);
    }
}

void OoOScheduler::NotifyInitDDRBuffers()
{
    for (const auto& t : function_.GetIncast()) {
        EmitInitDDRBuffer(t, DDRBufferKind::INCAST);
    }
    for (const auto& t : function_.GetOutcast()) {
        EmitInitDDRBuffer(t, DDRBufferKind::OUTCAST);
    }
}

std::vector<DDRRef> OoOScheduler::BuildDDRRefs(Operation* op) const
{
    std::vector<DDRRef> refs;
    auto add = [this, &refs](const LogicalTensorPtr& operand) {
        if (operand == nullptr)
            return;
        if (operand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR)
            return;
        DDRRef ref;
        ref.memId = operand->memoryrange.memId;
        ref.memType = operand->GetMemoryTypeOriginal();
        ref.addrStart = operand->memoryrange.start;
        ref.addrEnd = operand->memoryrange.end;
        ref.size = ref.addrEnd - ref.addrStart;
        auto it = ddrKindMap_.find(ref.memId);
        ref.kind = (it != ddrKindMap_.end()) ? it->second : DDRBufferKind::FUNCTION_TEMP;
        refs.push_back(ref);
    };
    for (const auto& iOp : op->GetIOperands())
        add(iOp);
    for (const auto& oOp : op->GetOOperands())
        add(oOp);
    return refs;
}

CoreLocation ToCoreLocation(CoreLocationType c)
{
    switch (c) {
        case CoreLocationType::AIC:
            return {CoreClass::AIC, 0};
        case CoreLocationType::AIV0:
            return {CoreClass::AIV, 0};
        case CoreLocationType::AIV1:
            return {CoreClass::AIV, 1};
        default:
            return {CoreClass::UNKNOWN, 0};
    }
}

void NotifySpill(ScheduleState& state, LogicalTensorPtr spillTensor, int spillMemId, Operation* spillAllocOp,
                 const SingleSpillCreatedOps& created)
{
    if (!state.HasObservers())
        return;
    auto& buf = state.localBufferMap.at(spillMemId);
    SpillEvent event;
    int curClock = state.clock;
    event.clock = curClock;
    event.beginClock = curClock;
    event.endClock = created.copyoutOp != nullptr ? curClock + created.copyoutOp->GetLatency() : curClock;
    event.spillMemId = spillMemId;
    event.memType = spillTensor->GetMemoryTypeOriginal();
    event.coreLocation = ToCoreLocation(state.schedInfoMap[spillAllocOp].coreLocation);
    event.addrStart = buf->start;
    event.addrEnd = buf->end;
    event.triggerOpMagic = spillAllocOp->GetOpMagic();
    event.triggerTensorSize = spillAllocOp->GetOutputOperand(0)->tensor->GetRawDataSize();
    event.spillCopyoutOpMagic = created.copyoutOp ? created.copyoutOp->GetOpMagic() : -1;
    event.reloadAllocOpMagic = created.allocOp ? created.allocOp->GetOpMagic() : -1;
    event.reloadCopyInOpMagic = created.copyinOp ? created.copyinOp->GetOpMagic() : -1;
    event.spillTensorMagic = spillTensor->GetMagic();
    event.spillCopyoutSize = (created.copyoutOp != nullptr) ? spillTensor->tensor->GetRawDataSize() : 0;
    for (const auto& [memId, ownerOp] : state.tensorOccupyMap) {
        if (state.schedInfoMap[ownerOp].isAlloc) {
            event.allocOccupiedSize += state.localBufferMap.at(memId)->size;
        }
    }
    auto& pool = state.bufferManagerMap[state.schedInfoMap[spillAllocOp].coreLocation][buf->memType];
    event.bufferCurrentUsage = pool.GetAllocatedSize();
    event.bufferCapacity = pool.GetMemSize();
    if (created.gmTensor != nullptr) {
        event.spillDdrMemId = created.gmTensor->memoryrange.memId;
        event.ddrKind = DDRBufferKind::SPILL_TEMP;
        event.ddrMemType = MemoryType::MEM_DEVICE_DDR;
        event.ddrAddrStart = created.gmTensor->memoryrange.start;
        event.ddrAddrEnd = created.gmTensor->memoryrange.end;
        event.ddrSize = event.ddrAddrEnd - event.ddrAddrStart;
    }
    for (auto* obs : state.observers_) {
        obs->OnSpill(event);
    }
}

} // namespace npu::tile_fwk
