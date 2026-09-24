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
 * \file device_stitch_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/dump_device_topo.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "machine/utils/dynamic/dev_stitch_dependency_enhanced.h"

namespace npu::tile_fwk::dynamic {
namespace {
struct HandleCellMatchFull {
    static inline uint32_t Process(int index, const DevCellMatchTableDesc& desc, uint32_t* cellMatchTableData,
                                   uint64_t* matchCount, DevAscendFunctionDupped* prevDup,
                                   DevAscendFunctionDupped* nextDup, size_t devNextIdx, int consumerOperationIdx,
                                   DeviceWorkspaceAllocator* workspace, int debugSlotIdx, size_t devTaskId,
                                   uint32_t preFuncIndex)
    {
        UNUSED(desc);
        auto producerOperationIdx = cellMatchTableData[index];
        if (producerOperationIdx != static_cast<uint32_t>(-1)) {
            (*matchCount)++;
            DEV_TRACE_DEBUG(
                DEvent(DUid(none()), DActStitchEdge(Producer(LUid(none(), 0, none(), producerOperationIdx, none()),
                                                             none(), none(), debugSlotIdx, none(), none()),
                                                    Consumer(LUid(none(), 0, none(), consumerOperationIdx, none()),
                                                             none(), none(), debugSlotIdx, none(), none()),
                                                    StitchReasonUniqueMatch())));
            DEV_VERBOSE_DEBUG("FullCoverUpdateStitch HandleCellMatchFull handle one stitch [%u] -> [%u!%u]",
                              producerOperationIdx, static_cast<uint32_t>(devNextIdx),
                              static_cast<uint32_t>(consumerOperationIdx));
            DeviceStitchContext::HandleOneStitch(
                *prevDup, *nextDup, preFuncIndex, producerOperationIdx, devNextIdx, consumerOperationIdx, workspace,
                DeviceStitchContext::StitchKind::StitchDefault, debugSlotIdx, static_cast<uint64_t>(devTaskId));
        }
        return 0;
    }
};

// Direct stitch: partial producers (producerConsumerList) → fullCover consumers
// (stitchPolicyFullCoverConsumerList).
void FullCoverPartialProdToFullConsStitch(DevAscendFunctionDupped& prevDup, DevAscendFunctionDupped& nextDup,
                                          DevAscendFunction* prevSrc, DevAscendFunction* nextSrc,
                                          DevAscendFunctionOutcast& outcast, DevAscendFunctionIncast& incast,
                                          size_t devNextIdx, int slotIdx, size_t devTaskId, uint32_t stitchDupIdx,
                                          DeviceWorkspaceAllocator* workspace, DevAscendFunctionDupped* stitchedList,
                                          size_t stitchedListSize)
{
    const size_t prodSize = outcast.producerConsumerList.size();
    const size_t consSize = incast.stitchPolicyFullCoverConsumerList.size();
    if (prodSize == 0 || consSize == 0) {
        return;
    }
    auto producerList = &prevSrc->At(outcast.producerConsumerList, 0);
    auto consumerList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerList, 0);
    for (size_t prodIndex = 0; prodIndex < prodSize; prodIndex++) {
        auto& producer = producerList[prodIndex];
        if (producer.opType == CellMatchOpType::READ) {
            continue;
        }
        auto& producerStitchList = prevDup.GetOperationStitch(producer.operationIdx, false);
        for (size_t conIndex = 0; conIndex < consSize; conIndex++) {
            auto& consumer = consumerList[conIndex];
            DeviceStitchContext::HandleOneStitch(prevDup, nextDup, producerStitchList, stitchDupIdx,
                                                 producer.operationIdx, devNextIdx, consumer.stitchOpIdx, workspace,
                                                 DeviceStitchContext::StitchKind::StitchFullCover, slotIdx,
                                                 static_cast<uint64_t>(devTaskId));
        }
    }
    DEV_IF_NONDEVICE { DeviceStitchContext::CheckStitch(stitchedList, static_cast<int>(stitchedListSize), &nextDup); }
}

void PartialUpdateStitchOneConsumerList(DevAscendFunctionDupped& nextDup, DevAscendFunction* nextSrc,
                                        uint64_t* expressionList, DevCellMatchTableDesc& cellMatchTableDesc,
                                        uint64_t* partialUpdateTableData, DevAscendFunctionDupped* stitchedList,
                                        size_t stitchedListSize, DeviceWorkspaceAllocator* workspace,
                                        size_t cellMatchTagId, size_t devTaskId, size_t devNextIdx, int slotIdx,
                                        int incastDim, DevLocalVector<DevAscendFunctionCallOperandUse>& consumerListRef,
                                        uint64_t* matchCount)
{
    const size_t consCnt = consumerListRef.size();
    if (consCnt == 0) {
        return;
    }
    auto* consumers = &nextSrc->At(consumerListRef, 0);
    for (size_t n = 0; n < consCnt; n++) {
        auto& consumer = consumers[n];
        uint64_t consumerOffset[DEV_SHAPE_DIM_MAX];
        uint64_t consumerValidShape[DEV_SHAPE_DIM_MAX];
        GetTensorOffsetAndValidShape(nextSrc, consumerOffset, consumerValidShape, expressionList, cellMatchTableDesc,
                                     incastDim, consumer.offsetAttrIdx, consumer.cachedAttrBase, consumer.operationIdx);

        DEV_IF_VERBOSE_DEBUG
        {
            for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                DEV_VERBOSE_DEBUG(
                    "PartialUpdateStitchConsumer consumer cell match, operation[%d] -> dimension[%d] = (offset:%lu "
                    ",shape:%lu, "
                    "cellshape:%d)",
                    consumer.operationIdx, j, consumerOffset[j], consumerValidShape[j],
                    cellMatchTableDesc.cellShape.dim[j]);
            }
        }
        DEV_IF_NONDEVICE
        {
            topo_dump::DumpConsumerCellAccess(static_cast<uint32_t>(devTaskId), slotIdx,
                                              static_cast<uint32_t>(devNextIdx), *nextSrc, consumer, cellMatchTableDesc,
                                              expressionList);
        }

        CellMatchStitchEnhance(consumerOffset, consumerValidShape, cellMatchTableDesc,
                               static_cast<uint32_t>(consumer.opType), partialUpdateTableData, stitchedList,
                               stitchedListSize, &nextDup, cellMatchTagId, static_cast<uint64_t>(devTaskId), devNextIdx,
                               workspace, consumer.stitchOpIdx, slotIdx, matchCount);
    }
}
} // namespace
void DeviceStitchContext::Init(DevAscendProgram* devProg, DeviceWorkspaceAllocator& workspace)
{
    workspace_ = &workspace;
    workspace_->SetupVector(stitchedList_);
    devProg_ = devProg;

    Reset();
}

void DeviceStitchContext::Reset()
{
    stitchedList_.clear();
    stitchReuseContext_.firstDupIdx = 0;
    stitchReuseContext_.lastNonEmptyDupIdx = -1;
}

void DeviceStitchContext::DumpStitchInfo() { DumpStitchInfo(stitchedList_.data(), stitchedList_.size()); }

void DeviceStitchContext::CheckStitch(DevAscendFunctionDupped* stitchedList, int size, DevAscendFunctionDupped* nextDup)
{
    DEV_IF_NONDEVICE
    {
        uint32_t dynPredCount = 0;
        uint32_t dynSuccCount = 0;
        for (int k = 0; k <= size; k++) {
            DevAscendFunctionDupped* dup = nullptr;
            if (k < size) {
                dup = &stitchedList[k];
            } else if (nextDup != nullptr) {
                dup = nextDup;
            } else {
                break;
            }
            auto src = dup->GetSource();
            for (size_t i = 0; i < dup->GetOperationSize(); i++) {
                auto opPredCount = src->GetOperationDepGraphPredCount(i);
                auto opDynPredCount = dup->GetOperationCurrPredCount(i);
                dynPredCount += opDynPredCount - opPredCount;
                auto succStitchList = dup->GetOperationStitch(i);
                for (auto p = succStitchList.Head(); p != nullptr; p = p->Next()) {
                    dynSuccCount += p->Size();
                }
            }
        }
        if (dynPredCount != dynSuccCount) {
            DEV_ERROR(ProgEncodeErr::STITCH_PRED_SUCC_MISMATCH,
                      "#ctrl.task.pre.stitch.check: dynPredCount %u does not match dynSuccCount %u", dynPredCount,
                      dynSuccCount);
        }
    }
}

void DeviceStitchContext::CheckStitch(DynDeviceTask* dyntask)
{
    DevAscendFunctionDupped* stitchedList = &dyntask->stitchedList[0];
    int stitchedSize = dyntask->stitchedList.size();
    CheckStitch(stitchedList, stitchedSize, nullptr);
}

uint64_t DeviceStitchContext::Stitch(DeviceSlotContext& slotContext, DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                     size_t devNextIdx, uint32_t cellMatchTagSeq)
{
    uint64_t count = FastStitch(slotContext.GetSlotList(), nextDup, devTaskId, devNextIdx, cellMatchTagSeq);
    if (stitchedList_.capacity() == 0) {
        /* This stitchedList_ vector can only allocate sufficient space once,
            during a single device task construction process.*/
        stitchedList_.reserve(MAX_STITCH_FUNC_NUM);
    }
    Append(nextDup);
    stitchedCallOpSize_ += (nextDup.GetSource()->GetOperationSize() - nextDup.GetSource()->hubOpCount_);
    return count;
}

void DeviceStitchContext::RecycleTensorWorkspace()
{
    // recycle submitted tasks' workspace memory
    workspace_->RecycleDevFuncWorkspace();
    workspace_->TriggerDelayedRecycle();
}

void DeviceStitchContext::DumpSlotInfo(const char* label, DeviceExecuteSlot* slotList, size_t slotSize)
{
    UNUSED(label);
    UNUSED(slotList);
    UNUSED(slotSize);
    DEV_IF_VERBOSE_DEBUG
    {
        DEV_DEBUG("[DecideSlotAddress] %s.", label);
        for (size_t slotIdx = 0; slotIdx < slotSize; slotIdx++) {
            [[maybe_unused]] const char* extraAttr = "";
            if (slotList[slotIdx].isOutputSlot) {
                extraAttr = " <output>";
            } else if (slotList[slotIdx].isAssembleSlot) {
                extraAttr = " <assemble>";
            }

            if (slotList[slotIdx].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                DEV_DEBUG("[DecideSlotAddress]   Slot [%3lu]: <no tensor>%s", slotIdx, extraAttr);
                continue;
            }
            [[maybe_unused]] auto& outcastDesc = workspace_->GetRuntimeOutcastTensor(slotList[slotIdx].rtOutcastIter);
            DEV_DEBUG("[DecideSlotAddress]   Slot [%3lu]: %s%s | mask=0x%x partial=%d stitchDupIdx=%u", slotIdx,
                      outcastDesc.Dump().c_str(), extraAttr, static_cast<unsigned>(slotList[slotIdx].stitchCtrlBitMask),
                      slotList[slotIdx].isPartialUpdateStitch, slotList[slotIdx].stitchDupIdx);
        }
    }
}

void DeviceStitchContext::DecideSlotAddress(DeviceExecuteSlot* slotList, size_t slotSize)
{
    [[maybe_unused]] static constexpr uint64_t NON_ADDR_MASK = UINT64_C(1) << 62;

    DumpSlotInfo("Update before", slotList, slotSize);
#if !DEBUG_INFINITE_LIFETIME
    for (size_t slotIdx = 0; slotIdx < slotSize; ++slotIdx) {
        auto& slot = slotList[slotIdx];
        if (slot.rtOutcastIter != ITEM_POOL_INVALID_INDEX &&
            workspace_->GetRuntimeOutcastTensor(slot.rtOutcastIter).property ==
                RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST) {
            workspace_->RuntimeOutcastTensorReplaceAddrWithoutRecycle(slot.rtOutcastIter,
                                                                      workspace_->AllocateBoundaryOutcastSlot(),
                                                                      RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
        }
    }
#endif // !DEBUG_INFINITE_LIFETIME
    DumpSlotInfo("Update after", slotList, slotSize);
}

int DeviceStitchContext::DecideIncastOutcast(uint64_t taskId)
{
    (void)taskId;
    const size_t funcNum = stitchedList_.size();
    for (size_t funcIndex = 0; funcIndex < funcNum; ++funcIndex) {
        auto& dup = stitchedList_[funcIndex];
        auto* src = dup.GetSource();
        // decide incast address
        size_t incastSize = src->GetIncastSize();
        for (size_t i = 0; i < incastSize; ++i) {
            auto& desc = dup.GetIncastAddress(i);
            DEV_ASSERT(CtrlErr::DEVICE_TASK_BUILD_FAILED, desc.IsRtOutcast());
            ItemPoolIter iter = desc.GetRtOutcastIter();
            auto& rtOutcast = workspace_->GetRuntimeOutcastTensor(iter);
            uintdevptr_t addr = rtOutcast.allocation.ptr;
            workspace_->RuntimeOutcastTensorDeref(rtOutcast);
            desc = AddressDescriptor::MakeFromAddress(addr);
        }

        // decide outcast address
        size_t outcastSize = src->GetOutcastSize();
        for (size_t i = 0; i < outcastSize; ++i) {
            auto& desc = dup.GetOutcastAddress(i);
            DEV_ASSERT(CtrlErr::DEVICE_TASK_BUILD_FAILED, desc.IsRtOutcast());
            ItemPoolIter iter = desc.GetRtOutcastIter();
            auto& rtOutcast = workspace_->GetRuntimeOutcastTensor(iter);
            uintdevptr_t addr = rtOutcast.Addr();
            workspace_->RuntimeOutcastTensorDeref(rtOutcast);
            desc = AddressDescriptor::MakeFromAddress(addr);
        }
    }
    return DEVICE_MACHINE_OK;
}

int DeviceStitchContext::MoveTo(DynDeviceTask* dynTask)
{
    dynTask->stitchedList = std::move(stitchedList_);
    stitchedList_.clear();
    dynTask->devTask.coreFunctionCnt = stitchedCallOpSize_;
    stitchedCallOpSize_ = 0;

    if (dynTask->stitchedList.size() > MAX_STITCH_FUNC_NUM) {
        DEV_ERROR(ProgEncodeErr::STITCH_LIST_TOO_LARGE,
                  "#ctrl.stitch.toomany_root: Stitch list size:%u exceeds maximum allowed cached function number:%zu.",
                  dynTask->stitchedList.size(), MAX_STITCH_FUNC_NUM);
        return DEVICE_MACHINE_ERROR;
    }
    int size = static_cast<int>(dynTask->stitchedList.size());
    for (int i = 0; i < size; ++i) {
        auto& funcDup = dynTask->stitchedList[i];
        auto* src = funcDup.GetSource();
        dynTask->dynFuncDataCacheList[i] = {src, &funcDup.GetOperationCurrPredCount(0), src->GetCalleeIndexAddr(),
                                            funcDup.DupDataForDynFuncData()};
        dynTask->devTask.mixTaskData.opWrapList[i] = PtrToValue(src->GetOpWrapListAddr());
    }
    dynTask->dynFuncDataCacheListSize = size;
    return DEVICE_MACHINE_OK;
}

static void ValidateAndDumpStitchEdge(const DevAscendFunctionDupped& producerDup,
                                      const DevAscendFunctionDupped& consumerDup, size_t producerOperationIdx,
                                      size_t consumerIdx, size_t consumerOperationIdx,
                                      DeviceStitchContext::StitchKind debugStitchKind, int debugSlotIdx)
{
    if (producerOperationIdx >= producerDup.GetSource()->GetOperationSize()) {
        DEV_ERROR(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                  "#ctrl.task.pre.stitch.handle: producerOperationIdx %zu exceeds the size of GetOperation %zu",
                  producerOperationIdx, producerDup.GetSource()->GetOperationSize());
        return;
    }
    if (consumerOperationIdx >= consumerDup.GetSource()->GetOperationSize()) {
        DEV_ERROR(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                  "#ctrl.task.pre.stitch.handle: consumerOperationIdx %zu exceeds the size of GetOperation %zu",
                  consumerOperationIdx, consumerDup.GetSource()->GetOperationSize());
        return;
    }
    DEV_VERBOSE_DEBUG("[Stitch] slot:%d kind:%s dupIdx:%d funcKey:%d,op:%d -> funcKey:%d,op:%d\n", debugSlotIdx,
                      DeviceStitchContext::GetStitchKindName(debugStitchKind).c_str(), (int)consumerIdx,
                      producerDup.GetSource()->GetFuncKey(), (int)producerOperationIdx,
                      consumerDup.GetSource()->GetFuncKey(), (int)consumerOperationIdx);

    topo_dump::DumpStitchEdge(producerDup, consumerDup, producerOperationIdx, consumerIdx, consumerOperationIdx,
                              debugStitchKind, debugSlotIdx);
}

void DeviceStitchContext::HandleOneStitch(DevAscendFunctionDupped& producerDup, DevAscendFunctionDupped& consumerDup,
                                          DevAscendFunctionDuppedStitchList& producerStitchList,
                                          uint32_t producerFuncIndex, size_t producerOperationIdx, size_t consumerIdx,
                                          size_t consumerOperationIdx, DeviceWorkspaceAllocator* workspace,
                                          StitchKind debugStitchKind, int debugSlotIdx, uint64_t devTaskId)
{
    (void)debugStitchKind;
    (void)debugSlotIdx;
    DEV_VERBOSE_DEBUG("DeviceStitchContext::HandleOneStitch %p stitchlist %p [%u!%d] -> [%u!%u]", &producerDup,
                      &producerStitchList, producerFuncIndex, static_cast<int>(producerOperationIdx),
                      static_cast<uint32_t>(consumerIdx), static_cast<uint32_t>(consumerOperationIdx));

    auto* stitchCacheAddr = workspace->StitchCacheAddr();
    const uint32_t rootFuncMaxCallOpsize = workspace->RootFuncMaxCallOpsize();
    const uint16_t stitchCacheEpoch = workspace->StitchCacheEpoch();
    if (CheckStitchCacheDuplicate(stitchCacheAddr, rootFuncMaxCallOpsize, stitchCacheEpoch, producerFuncIndex,
                                  static_cast<uint32_t>(producerOperationIdx), static_cast<uint32_t>(consumerIdx),
                                  consumerOperationIdx, devTaskId)) {
        DEV_VERBOSE_DEBUG("Duplicate stitch ignore.");
        return;
    }

    PushBackTask(producerStitchList, MakeTaskID(consumerIdx, consumerOperationIdx), workspace);
    consumerDup.GetOperationCurrPredCount(consumerOperationIdx)++;

    auto* producerFunc = producerDup.GetSource();
    auto producerIdx = static_cast<uint32_t>(producerOperationIdx);
    producerFunc->ClearTailTask(producerIdx);
    if (producerFunc->ClearDeadEndHub(producerIdx)) {
        producerFunc->PropagateDeadHubClear(producerIdx);
    }

    DEV_IF_NONDEVICE
    {
        ValidateAndDumpStitchEdge(producerDup, consumerDup, producerOperationIdx, consumerIdx, consumerOperationIdx,
                                  debugStitchKind, debugSlotIdx);
    }
}

void DeviceStitchContext::HandleOneStitch(DevAscendFunctionDupped& producerDup, DevAscendFunctionDupped& consumerDup,
                                          uint32_t producerFuncIndex, size_t producerOperationIdx, size_t consumerIdx,
                                          size_t consumerOperationIdx, DeviceWorkspaceAllocator* workspace,
                                          StitchKind debugStitchKind, int debugSlotIdx, uint64_t devTaskId)
{
    auto& producerStitchList = producerDup.GetOperationStitch(producerOperationIdx, false);
    HandleOneStitch(producerDup, consumerDup, producerStitchList, producerFuncIndex, producerOperationIdx, consumerIdx,
                    consumerOperationIdx, workspace, debugStitchKind, debugSlotIdx, devTaskId);
}

uint64_t DeviceStitchContext::PartialUpdateStitchConsumer(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                          size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                          DevAscendFunctionIncast& incast, uint32_t cellMatchTagSeq)
{
    uint64_t matchCount = 0;
    auto* nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto& cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
    auto partialUpdateTableData = slot.partialUpdate->cellMatchRuntimePartialUpdateTable.Data();
    size_t tableSize = slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size();

    DEV_VERBOSE_DEBUG("[PartialUpdateStitch] enter slotIdx=%d devTaskId=%lu devNextIdx=%lu consumerFuncKey=%d "
                      "partialConsumerCount=%zu fullCoverConsumerCount=%zu tableSize=%zu descDimSize=%d",
                      slotIdx, (uint64_t)devTaskId, (uint64_t)devNextIdx, nextSrc->GetFuncKey(),
                      incast.consumerList.size(), incast.stitchPolicyFullCoverConsumerList.size(), tableSize,
                      cellMatchTableDesc.GetDimensionSize());
    size_t cellMatchTagId = CellMatchBuildTagId(slot.slotAllocIterId, cellMatchTagSeq);

    PartialUpdateStitchOneConsumerList(nextDup, nextSrc, expressionList, cellMatchTableDesc, partialUpdateTableData,
                                       stitchedList_.data(), stitchedList_.size(), workspace_, cellMatchTagId,
                                       devTaskId, devNextIdx, slotIdx, incast.dim, incast.consumerList, &matchCount);
    PartialUpdateStitchOneConsumerList(nextDup, nextSrc, expressionList, cellMatchTableDesc, partialUpdateTableData,
                                       stitchedList_.data(), stitchedList_.size(), workspace_, cellMatchTagId,
                                       devTaskId, devNextIdx, slotIdx, incast.dim,
                                       incast.stitchPolicyFullCoverConsumerList, &matchCount);
    return matchCount;
}

uint64_t DeviceStitchContext::FullCoverDefaultUpdateStitch(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                           size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                           DevAscendFunctionDupped& prevDup,
                                                           DevAscendFunctionOutcast& outcast,
                                                           DevAscendFunctionIncast& incast)
{
    const size_t consumerCount = incast.consumerList.size();
    const size_t tableSize = outcast.cellMatchRuntimeFullUpdateTable.size();
    if (consumerCount == 0 || tableSize == 0) {
        return 0;
    }
    uint64_t matchCount = 0;
    auto* prevSrc = prevDup.GetSource();
    auto* nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto& cellMatchTableDesc = outcast.cellMatchTableDesc;
    auto fullUpdateTableData = &prevSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
    auto* consumers = &nextSrc->At(incast.consumerList, 0);

    DEV_VERBOSE_DEBUG(
        "[FullCoverDefaultStitch] enter slotIdx=%d devTaskId=%lu devNextIdx=%lu producerFuncKey=%d "
        "consumerFuncKey=%d stitchDupIdx=%u stitchOutcastIdx=%u partialConsumerCount=%zu tableSize=%zu descDimSize=%d",
        slotIdx, (uint64_t)devTaskId, (uint64_t)devNextIdx, prevSrc->GetFuncKey(), nextSrc->GetFuncKey(),
        slot.stitchDupIdx, slot.stitchOutcastIdx, consumerCount, tableSize, cellMatchTableDesc.GetDimensionSize());

    for (size_t n = 0; n < consumerCount; n++) {
        auto& consumer = consumers[n];
        uint64_t fullCoverOffset[DEV_SHAPE_DIM_MAX];
        uint64_t fullCoverValidShape[DEV_SHAPE_DIM_MAX];
        GetTensorOffsetAndValidShape(nextSrc, fullCoverOffset, fullCoverValidShape, expressionList, cellMatchTableDesc,
                                     incast.dim, consumer.offsetAttrIdx, consumer.cachedAttrBase,
                                     consumer.operationIdx);
        DEV_IF_NONDEVICE
        {
            topo_dump::DumpConsumerCellAccess(static_cast<uint32_t>(devTaskId), slotIdx,
                                              static_cast<uint32_t>(devNextIdx), *nextSrc, consumer, cellMatchTableDesc,
                                              expressionList);
        }
        CellMatchHandle<HandleCellMatchFull>(fullCoverOffset, fullCoverValidShape, cellMatchTableDesc,
                                             fullUpdateTableData, &matchCount, &prevDup, &nextDup, devNextIdx,
                                             consumer.stitchOpIdx, workspace_, slotIdx, devTaskId, slot.stitchDupIdx);
    }
    DEV_IF_NONDEVICE { DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup); }
    return matchCount;
}

uint64_t DeviceStitchContext::FullCoverUpdateStitch(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                    size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                    DevAscendFunctionIncast& incast)
{
    DevAscendFunctionDupped& prevDup = stitchedList_[slot.stitchDupIdx];
    auto* prevSrc = prevDup.GetSource();
    auto& outcast = prevSrc->GetOutcast(slot.stitchOutcastIdx);
    auto* nextSrc = nextDup.GetSource();
    DEV_VERBOSE_DEBUG("outcast %lu fullcover update stitch\n", (unsigned long)slot.stitchOutcastIdx);
    DEV_VERBOSE_DEBUG("=================FullCoverUpdateStitch partialProducerCount=%zu partialConsumerCount=%zu "
                      "fullCoverConsumerCount=%zu=================\n",
                      outcast.producerConsumerList.size(), incast.consumerList.size(),
                      incast.stitchPolicyFullCoverConsumerList.size());

    // stitchPolicyFullCover hub
    auto producerHubOpIdx = outcast.stitchPolicyFullCoverProducerHubOpIdx;
    if (producerHubOpIdx != -1) {
        auto& producerStitchList = prevDup.GetOperationStitch(producerHubOpIdx, false);
        auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
        for (size_t conIndex = 0, conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); conIndex < conSize;
             conIndex++) {
            auto& consumerOpIdx = consumerAllOpIdxList[conIndex];
            DEV_VERBOSE_DEBUG("FullCoverUpdateStitch hub handle one stitch [%u!%u] -> [%u!%u]", slot.stitchDupIdx,
                              static_cast<uint32_t>(producerHubOpIdx), static_cast<uint32_t>(devNextIdx),
                              consumerOpIdx);
            DeviceStitchContext::HandleOneStitch(
                prevDup, nextDup, producerStitchList, slot.stitchDupIdx, producerHubOpIdx, devNextIdx, consumerOpIdx,
                workspace_, StitchKind::StitchFullCover, slotIdx, static_cast<uint64_t>(devTaskId));
        }
        DEV_IF_NONDEVICE { DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup); }
    } else {
        // stitchPolicyFullCover producer
        auto producerList = &prevSrc->At(outcast.stitchPolicyFullCoverProducerList, 0);
        auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
        const size_t conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size();
        for (size_t prodIndex = 0, prodSize = outcast.stitchPolicyFullCoverProducerList.size(); prodIndex < prodSize;
             prodIndex++) {
            auto& producer = producerList[prodIndex];
            auto producerOperationIdx = producer.operationIdx;
            if (producer.opType == CellMatchOpType::READ) {
                continue;
            }
            auto& producerStitchList = prevDup.GetOperationStitch(producerOperationIdx, false);
            for (size_t conIndex = 0; conIndex < conSize; conIndex++) {
                auto& consumerOpIdx = consumerAllOpIdxList[conIndex];
                DEV_VERBOSE_DEBUG("FullCoverUpdateStitch handle one stitch [%u!%u] -> [%u!%u]", slot.stitchDupIdx,
                                  static_cast<uint32_t>(producerOperationIdx), static_cast<uint32_t>(devNextIdx),
                                  consumerOpIdx);
                DeviceStitchContext::HandleOneStitch(
                    prevDup, nextDup, producerStitchList, slot.stitchDupIdx, producerOperationIdx, devNextIdx,
                    consumerOpIdx, workspace_, StitchKind::StitchFullCover, slotIdx, static_cast<uint64_t>(devTaskId));
            }
        }
        DEV_IF_NONDEVICE { DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup); }
    }

    // partial write → fullCover read (fullCover consumers are no longer in DEFAULT consumerList)
    FullCoverPartialProdToFullConsStitch(prevDup, nextDup, prevSrc, nextSrc, outcast, incast, devNextIdx, slotIdx,
                                         devTaskId, slot.stitchDupIdx, workspace_, stitchedList_.data(),
                                         stitchedList_.size());

    return FullCoverDefaultUpdateStitch(nextDup, devTaskId, devNextIdx, slot, slotIdx, prevDup, outcast, incast);
}

uint64_t DeviceStitchContext::PartialUpdateStitchProducer(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                          size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                          DevAscendFunctionOutcast& outcast, uint32_t cellMatchTagSeq)
{
    uint64_t matchCount = 0;
    auto* nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto& cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
    if (slot.partialUpdate->Empty()) {
        return matchCount;
    }
    auto partialUpdateTableData = slot.partialUpdate->cellMatchRuntimePartialUpdateTable.Data();
    size_t cellMatchTagId = CellMatchBuildTagId(slot.slotAllocIterId, cellMatchTagSeq);

    auto* stitchedData = stitchedList_.data();
    const size_t stitchedSize = stitchedList_.size();
    auto processProducerList = [&](auto& producerListRef) {
        const size_t prodCnt = producerListRef.size();
        if (prodCnt == 0) {
            return;
        }
        auto* producerList = &nextSrc->At(producerListRef, 0);
        for (size_t i = 0; i < prodCnt; i++) {
            auto& producer = producerList[i];
            if (producer.opType == CellMatchOpType::READ) {
                continue;
            }
            uint64_t producerOffset[DEV_SHAPE_DIM_MAX];
            uint64_t producerValidShape[DEV_SHAPE_DIM_MAX];
            GetTensorOffsetAndValidShape(nextSrc, producerOffset, producerValidShape, expressionList,
                                         cellMatchTableDesc, outcast.dim, producer.offsetAttrIdx,
                                         producer.cachedAttrBase, producer.operationIdx);

            DEV_IF_VERBOSE_DEBUG
            {
                for (int k = 0; k < cellMatchTableDesc.GetDimensionSize(); k++) {
                    DEV_VERBOSE_DEBUG(
                        "PartialUpdateStitchProducer cell match, operation[%d] -> dimension[%d] = (offset:%lu "
                        ",validshape:%lu, "
                        "cellshape:%d)",
                        producer.operationIdx, k, producerOffset[k], producerValidShape[k],
                        cellMatchTableDesc.cellShape.dim[k]);
                }
            }

            CellMatchStitchEnhance(producerOffset, producerValidShape, cellMatchTableDesc,
                                   static_cast<uint32_t>(producer.opType), partialUpdateTableData, stitchedData,
                                   stitchedSize, &nextDup, cellMatchTagId, static_cast<uint64_t>(devTaskId), devNextIdx,
                                   workspace_, producer.stitchOpIdx, slotIdx, &matchCount);
        }
    };

    DEV_VERBOSE_DEBUG("Begin PartialUpdateStitchProducer partialProducerList.");
    processProducerList(outcast.producerConsumerList);

    DEV_VERBOSE_DEBUG("Begin PartialUpdateStitchProducer fullCoverProducerList.");
    processProducerList(outcast.stitchPolicyFullCoverProducerList);

    return matchCount;
}

void DeviceStitchContext::ReuseStitch(DevAscendFunctionDupped& nextDup, size_t devNextIdx, size_t devTaskId)
{
    auto* nextSrc = nextDup.GetSource();
    if (nextSrc->rootInnerTensorWsMemoryRequirement == 0) {
        // 0 length workspace, no dependency in need
        return;
    }

    uintdevptr_t nextAddrL = nextDup.RuntimeWorkspace();
    uintdevptr_t nextAddrR = nextAddrL + nextSrc->rootInnerTensorWsMemoryRequirement;
    auto nextReuseInfo = nextDup.GetRuntimeReuseInfo();
    if (auto& firstDup = stitchedList_[stitchReuseContext_.firstDupIdx];
        firstDup.GetRuntimeReuseInfo().poolResetTimes >= nextReuseInfo.poolResetTimes) {
        return;
    }

    auto needsDependency = [&](uint32_t prevIdx) -> int {
        if (prevIdx >= devNextIdx) {
            // invalid idx
            return INVALID_TOO_AHEAD;
        }

        auto& prevDup = stitchedList_[prevIdx];
        if (prevDup.GetSource()->rootInnerTensorWsMemoryRequirement == 0) {
            // empty workspace
            return SKIP_EMPTY;
        }

        auto prevReuseInfo = prevDup.GetRuntimeReuseInfo();
        if (prevReuseInfo.poolResetTimes + 1 != nextReuseInfo.poolResetTimes) {
            return prevReuseInfo.poolResetTimes >= nextReuseInfo.poolResetTimes ? INVALID_TOO_AHEAD : NO_DEP;
        }

        // proper poolResetTimes
        stitchReuseContext_.lastNonEmptyDupIdx = prevIdx;

        uintdevptr_t prevAddrL = prevDup.RuntimeWorkspace();
        uintdevptr_t prevAddrR = prevAddrL + prevDup.GetSource()->rootInnerTensorWsMemoryRequirement;
        return !(prevAddrR <= nextAddrL || prevAddrL >= nextAddrR) ? NEEDS_DEP : NO_DEP;
    };

    auto skipBefore = [](int result) { return result == NO_DEP || result == SKIP_EMPTY; };
    int firstResult;
    for (; skipBefore(firstResult = needsDependency(stitchReuseContext_.firstDupIdx));
         stitchReuseContext_.firstDupIdx++) {
    }

    if (firstResult == NEEDS_DEP) {
        for (uint32_t prevIdx = stitchReuseContext_.firstDupIdx;; prevIdx++) {
            int res = needsDependency(prevIdx);
            if (res == NO_DEP || res == INVALID_TOO_AHEAD) {
                break;
            }
            if (res != SKIP_EMPTY) {
                auto& prevDup = stitchedList_[prevIdx];
                StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx,
                                        workspace_, static_cast<uint64_t>(devTaskId), prevIdx);
                stitchReuseContext_
                    .firstDupIdx = prevIdx; // Risk on time complexity: Duplicated access to empty-workspace funcs
            }
        }
    } else {
        if (stitchReuseContext_.lastNonEmptyDupIdx != -1) {
            auto& prevDup = stitchedList_[stitchReuseContext_.lastNonEmptyDupIdx];
            StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx,
                                    workspace_, static_cast<uint64_t>(devTaskId),
                                    stitchReuseContext_.lastNonEmptyDupIdx);
        }
    }
}

uint64_t DeviceStitchContext::FastStitchConsumer(DeviceExecuteSlot* slotList, DevAscendFunctionDupped& nextDup,
                                                 size_t devTaskId, size_t devNextIdx, uint32_t cellMatchTagSeq)
{
    auto* nextSrc = nextDup.GetSource();
    uint64_t matchCount = 0;
    size_t listSize = nextSrc->GetStitchIncastSlotListSize();
    for (size_t i = 0; i < listSize; ++i) {
        auto& entry = nextSrc->GetStitchIncastSlot(i);
        auto slotIdx = entry.slotIdx;
        auto& slot = slotList[slotIdx];
        if (slot.stitchDupIdx == INVALID_STITCH_IDX) {
            continue;
        }
        auto& incast = nextSrc->GetIncast(entry.incastIdx);
        DEV_VERBOSE_DEBUG("FastStitch slot %d, incastindex %d, ispartial %d, stitchDupIdx %u", slotIdx, entry.incastIdx,
                          slot.isPartialUpdateStitch, slot.stitchDupIdx);
        if (slot.isPartialUpdateStitch) {
            matchCount = PartialUpdateStitchConsumer(nextDup, devTaskId, devNextIdx, slot, slotIdx, incast,
                                                     cellMatchTagSeq);
            continue;
        }
        if (slot.rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
            continue;
        }
        matchCount = FullCoverUpdateStitch(nextDup, devTaskId, devNextIdx, slot, slotIdx, incast);
    }
    return matchCount;
}

uint64_t DeviceStitchContext::FastStitchProducer(DeviceExecuteSlot* slotList, DevAscendFunctionDupped& nextDup,
                                                 size_t devTaskId, size_t devNextIdx, uint32_t cellMatchTagSeq)
{
    auto* nextSrc = nextDup.GetSource();
    uint64_t matchCount = 0;
    size_t listSize = nextSrc->GetStitchOutCastSlotListSize();
    for (size_t i = 0; i < listSize; ++i) {
        auto& entry = nextSrc->GetStitchOutCastSlot(i);
        auto slotIdx = entry.slotIdx;
        auto& slot = slotList[slotIdx];
        if (slot.stitchDupIdx == INVALID_STITCH_IDX) {
            continue;
        }
        auto& outcast = nextSrc->GetOutcast(entry.outcastIdx);
        DEV_VERBOSE_DEBUG("FastStitch slot %d, outcastindex %d, ispartial %d, stitchDupIdx %u", slotIdx,
                          entry.outcastIdx, slot.isPartialUpdateStitch, slot.stitchDupIdx);
        matchCount += PartialUpdateStitchProducer(nextDup, devTaskId, devNextIdx, slot, slotIdx, outcast,
                                                  cellMatchTagSeq);
    }
    return matchCount;
}

uint64_t DeviceStitchContext::FastStitch(DeviceExecuteSlot* slotList, DevAscendFunctionDupped& nextDup,
                                         size_t devTaskId, size_t devNextIdx, uint32_t cellMatchTagSeq)
{
    AutoScopedPerf asp(PERF_EVT_FAST_STITCH);
#if !ENABLE_STITCH
    return 0;
#endif
    auto* nextSrc = nextDup.GetSource();
    nextSrc->GetFuncidx() = static_cast<int>(devNextIdx);
    if (devNextIdx == 0) {
        return 0;
    }
    uint64_t matchCount = FastStitchConsumer(slotList, nextDup, devTaskId, devNextIdx, cellMatchTagSeq);
    matchCount += FastStitchProducer(slotList, nextDup, devTaskId, devNextIdx, cellMatchTagSeq);
#if !DEBUG_INFINITE_LIFETIME
    ReuseStitch(nextDup, devNextIdx, devTaskId);
#endif
    return matchCount;
}

void DeviceStitchContext::DumpStitchInfo(DevAscendFunctionDupped* stitchedList, int stitchedSize)
{
    int funcId = 0;
    for (int i = 0; i < stitchedSize; i++) {
        auto& funcDup = stitchedList[i];
        for (size_t opIndex = 0; opIndex < funcDup.GetSource()->GetOperationSize(); opIndex++) {
            auto& stitch = funcDup.GetOperationStitch(opIndex);

            std::stringstream oss;
            oss << stitch.Dump();
            DEV_VERBOSE_DEBUG("func %d opIndex %zu stitch list: %p stitchindex:%u %s.", funcId, opIndex, &stitch,
                              funcDup.GetSource()->GetOperationStitchIndex(opIndex), oss.str().c_str());
        }
        funcId++;
    }
}

void DeviceStitchContext::StitchForWorkspaceReuse(DevAscendFunctionDupped* stitchingList, int stitchingSize,
                                                  DevAscendFunctionDupped& prevDup, DevAscendFunctionDupped& currDup,
                                                  size_t devCurrIdx, DeviceWorkspaceAllocator* workspace,
                                                  uint64_t devTaskId, uint32_t preFuncIndex)
{
    // Add dependency between root functions
    auto* prevSrc = prevDup.GetSource();
    auto* currSrc = currDup.GetSource();

    size_t prevNoSuccOpSize = prevSrc->GetNoSuccOpSize();
    size_t currNoPredOpSize = currSrc->GetNoPredOpSize();
    if (unlikely(prevNoSuccOpSize == 0 || currNoPredOpSize == 0)) {
        // Empty root function
        return;
    }

    // Graph has been optimized when encoding, we just put trivial full connection logics here
    for (size_t i = 0; i < prevNoSuccOpSize; ++i) {
        int prevNoSucc = prevSrc->GetNoSuccOpIdx(i);
        auto& stitch = prevDup.GetOperationStitch(prevNoSucc);
        for (size_t j = 0; j < currNoPredOpSize; ++j) {
            int currNoPred = currSrc->GetNoPredOpIdx(j);
            DEV_TRACE_DEBUG(DEvent(DUid(none()), DActStitchEdge(Producer(LUid(none(), 0, none(), prevNoSucc, none()),
                                                                         none(), none(), none(), none(), none()),
                                                                Consumer(LUid(none(), 0, none(), currNoPred, none()),
                                                                         none(), none(), none(), none(), none()),
                                                                StitchReasonWorkspaceReuse())));
            DEV_VERBOSE_DEBUG("StitchForWorkspaceReuse handle one stitch [%u] -> [%u!%u]",
                              static_cast<uint32_t>(prevNoSucc), static_cast<uint32_t>(devCurrIdx),
                              static_cast<uint32_t>(currNoPred));
            DeviceStitchContext::HandleOneStitch(prevDup, currDup, stitch, preFuncIndex, prevNoSucc, devCurrIdx,
                                                 currNoPred, workspace, DeviceStitchContext::StitchKind::StitchReuse,
                                                 -1, devTaskId);
            DEV_IF_NONDEVICE { DeviceStitchContext::CheckStitch(stitchingList, stitchingSize, &currDup); }
        }
    }
}
} // namespace npu::tile_fwk::dynamic
