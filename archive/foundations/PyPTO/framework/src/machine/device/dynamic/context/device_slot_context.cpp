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
 * \file device_slot_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_slot_context.h"
#include "machine/device/dynamic/context/dump_device_topo.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "machine/utils/dynamic/dev_cell_match_dump.h"

namespace npu::tile_fwk::dynamic {

static void PrepareRuntimeDynamicPartialUpdateTable(DeviceWorkspaceAllocator* workspace,
                                                    DevAscendProgramUpdate* partialUpdate)
{
    auto& desc = partialUpdate->cellMatchTableDesc;
    const int dim = desc.GetDimensionSize();

    uint64_t cellCount = 1;
    for (int d = 0; d < dim; ++d) {
        int stride = desc.GetStrideShape(d);
        DEV_IF_NONDEVICE
        {
            DEV_ASSERT_MSG(ProgEncodeErr::CELL_MATCH_PARAM_INVALID, stride > 0,
                           "Dynamic cell match launch prepare missing for slot=%d dim=%d stride[%d]=%d",
                           partialUpdate->slotIndex, dim, d, stride);
        }
        cellCount *= static_cast<uint64_t>(stride);
    }

    uint32_t cellUint64Size = desc.cellUint64Size;
    DEV_IF_NONDEVICE
    {
        uint64_t slotTableCapacity = workspace->DynamicCellMatchSlotByteSize() / (cellUint64Size * sizeof(uint64_t));
        DEV_ASSERT_MSG(
            ProgEncodeErr::CELL_MATCH_PARAM_INVALID, cellCount > 0 && cellCount <= slotTableCapacity,
            "Dynamic cell match table size invalid for slot=%d, cellCount=%lu capacity=%lu cellUint64Size=%u",
            partialUpdate->slotIndex, cellCount, slotTableCapacity, cellUint64Size);
    }

    partialUpdate->cellMatchRuntimePartialUpdateTable.HostAssignDataSize(
        reinterpret_cast<uintdevptr_t>(partialUpdate->cellMatchRuntimePartialUpdateTable.Data()),
        cellCount * cellUint64Size);
}

void DeviceSlotContext::InitAllocator(DeviceWorkspaceAllocator& workspace, uint64_t slotSize)
{
    workspace.SetupVector(slotList_);
    workspace_ = &workspace;
    slotList_.resize(slotSize);
}

void DeviceSlotContext::FillInputOutputSlot(DevAscendProgram* devProg, DevStartArgs* args)
{
    const uint64_t progBegin = reinterpret_cast<uint64_t>(devProg);
    const uint64_t progEnd = progBegin + devProg->GetSize();
    const size_t ie = devProg->updateList.size();
    for (size_t i = 0; i < ie; ++i) {
        auto& partialUpdate = devProg->At(devProg->updateList, i);
        uint64_t tablePtr = reinterpret_cast<uint64_t>(partialUpdate.cellMatchRuntimePartialUpdateTable.Data());
        if (tablePtr != 0 && (tablePtr < progBegin || tablePtr >= progEnd)) {
            partialUpdate.cellMatchRuntimePartialUpdateTable.HostAssignDataSize(0, 0);
        }
    }
    FillInputOutputSlot(slotList_.data(), slotList_.size(), devProg, args);
}

static uint32_t UpdateSlotsForOutCastPartialStitch(int slotIdx, DeviceExecuteSlot& slot, DevAscendFunction* devRootSrc,
                                                   DevAscendFunctionOutcast& outcast,
                                                   DevAscendFunctionCallOperandUse* producerList,
                                                   uint32_t cellMatchTagId, uint32_t devNextIdx,
                                                   uint64_t* expressionList)
{
    if (slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size() == 0) {
        return 0;
    }
    auto& cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
    auto* tableData = slot.partialUpdate->cellMatchRuntimePartialUpdateTable.Data();
    auto producerSize = outcast.producerConsumerList.size();
    uint32_t errCode = 0;

    if (unlikely(cellMatchTableDesc.MaybeHaveAtomic())) {
        if (producerSize != 0) {
            errCode = CellMatchFillIncastOutcast(devRootSrc, producerList, producerSize, expressionList,
                                                 cellMatchTableDesc, tableData, cellMatchTagId, devNextIdx);
        }
        if (outcast.stitchPolicyFullCoverProducerList.size() != 0) {
            errCode = CellMatchFillIncastOutcast(devRootSrc,
                                                 &devRootSrc->At(outcast.stitchPolicyFullCoverProducerList, 0),
                                                 outcast.stitchPolicyFullCoverProducerList.size(), expressionList,
                                                 cellMatchTableDesc, tableData, cellMatchTagId, devNextIdx);
        }
    } else if (producerSize != 0) {
        errCode = CellMatchFillIncastOutcast(devRootSrc, producerList, producerSize, expressionList, cellMatchTableDesc,
                                             tableData, cellMatchTagId, devNextIdx);
        DEV_VERBOSE_DEBUG("Fill cell match table slot %d outcastIndex %u, "
                          "outcast producer list, size = %zu.",
                          slotIdx, slot.stitchOutcastIdx, producerSize);
    } else {
        errCode = CellMatchFillIncastOutcast(devRootSrc, &devRootSrc->At(outcast.stitchPolicyFullCoverProducerList, 0),
                                             outcast.stitchPolicyFullCoverProducerList.size(), expressionList,
                                             cellMatchTableDesc, tableData, cellMatchTagId, devNextIdx);
        DEV_VERBOSE_DEBUG("Fill cell match table slot %d outcastIndex %u, "
                          "outcast full cover producer list.",
                          slotIdx, slot.stitchOutcastIdx);
    }

    DEV_VERBOSE_DEBUG_SPLIT(
        "[UpdateSlots]  slot %d, cellMatchTagId=%x, ret=0x%x, CellMatchPartial=%s.\n", slotIdx, cellMatchTagId, errCode,
        DumpCellMatchPartialUpdateTable(tableData, slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size(),
                                        cellMatchTableDesc)
            .c_str());
    return errCode;
}

static uint32_t UpdateSlotsForOutCastFullCoverStitch(int slotIdx, DevAscendFunction* devRootSrc,
                                                     DevAscendFunctionOutcast& outcast, uint32_t cellMatchTagId,
                                                     uint64_t* expressionList)
{
    auto producerList = &devRootSrc->At(outcast.producerConsumerList, 0);
    // Full Fill only covers producerConsumerList (partial). FullCover producers stitch
    // via POLICY direct edges; they are filled only on the partial path.
    auto& cellMatchTableDesc = outcast.cellMatchTableDesc;
    auto tableData = &devRootSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
    uint32_t errCode = CellMatchFillIncastOutcast(devRootSrc, producerList, outcast.producerConsumerList.size(),
                                                  expressionList, cellMatchTableDesc, tableData);
    DEV_VERBOSE_DEBUG(
        "[UpdateSlots] slot %d  CellMatchFull=%s cellMatchTagId=%x, ret=0x%x\n", slotIdx,
        DevAscendFunctionDuppedStitchList::DumpTask(tableData, outcast.cellMatchRuntimeFullUpdateTable.size()).c_str(),
        cellMatchTagId, errCode);
    return errCode;
}

static uint32_t UpdateSlotsForIncastStitch(int slotIdx, DeviceExecuteSlot& slot, DevAscendFunction* devRootSrc,
                                           DevAscendFunctionIncast& incast, uint32_t devTaskId, uint32_t devNextIdx,
                                           uint64_t* expressionList, uint32_t cellMatchTagSeq)
{
    UNUSED(slotIdx);
    UNUSED(devTaskId);
    // Caller requires stitchCtrlBitMask & RAW (⇒ partialUpdate set by Mark).
    if (slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size() == 0 ||
        slot.partialUpdate->cellMatchTableDesc.GetCacheOpMaxCount(CELL_MATCH_OP_TYPE_READ) == 0) {
        return 0;
    }

    auto& cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
    auto* tableData = slot.partialUpdate->cellMatchRuntimePartialUpdateTable.Data();
    uint32_t cellMatchTagId = CellMatchBuildTagId(slot.slotAllocIterId, cellMatchTagSeq);

    uint32_t errCode = 0;
    if (incast.consumerList.size() != 0) {
        errCode = CellMatchFillIncastOutcast(devRootSrc, &devRootSrc->At(incast.consumerList, 0),
                                             incast.consumerList.size(), expressionList, cellMatchTableDesc, tableData,
                                             cellMatchTagId, devNextIdx);
    }
    if (errCode == 0 && incast.stitchPolicyFullCoverConsumerList.size() != 0) {
        errCode = CellMatchFillIncastOutcast(devRootSrc, &devRootSrc->At(incast.stitchPolicyFullCoverConsumerList, 0),
                                             incast.stitchPolicyFullCoverConsumerList.size(), expressionList,
                                             cellMatchTableDesc, tableData, cellMatchTagId, devNextIdx);
    }
    DEV_VERBOSE_DEBUG_SPLIT(
        "[UpdateSlots]  incast slot %d  cellMatchTagId=%x, ret=0x%x partialConsumerCount=%zu "
        "fullCoverConsumerCount=%zu CellMatchPartial=%s\n",
        slotIdx, cellMatchTagId, errCode, incast.consumerList.size(), incast.stitchPolicyFullCoverConsumerList.size(),
        DumpCellMatchPartialUpdateTable(tableData, slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size(),
                                        cellMatchTableDesc)
            .c_str());
    return errCode;
}

static uint32_t UpdateSlotsImpl(DeviceWorkspaceAllocator* workspace, DeviceExecuteSlot* slotList,
                                DevAscendFunctionDupped& devRootDup, uint32_t devTaskId, uint32_t devNextIdx,
                                uint32_t cellMatchTagSeq)
{
    AutoScopedPerf asp(PERF_EVT_UPDATE_SLOT);
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    uint32_t retCode = 0;

    uint64_t* expressionList = &devRootDup.GetExpression(0);
    size_t updateOutCastListSize = devRootSrc->GetUpdateOutCastSlotListSize();
    for (size_t i = 0; i < updateOutCastListSize; ++i) {
        auto& entry = devRootSrc->GetUpdateOutCastSlot(i);
        auto& outcast = devRootSrc->GetOutcast(entry.outcastIdx);
        const auto& outcastDesc = devRootDup.GetOutcastAddress(entry.outcastIdx);
        auto& slot = slotList[entry.slotIdx];
        slot.stitchDupIdx = devNextIdx;
        slot.stitchOutcastIdx = static_cast<uint32_t>(entry.outcastIdx);
        DEV_IF_NONDEVICE
        {
            topo_dump::DumpProducerCellAccess(devTaskId, entry.slotIdx, devNextIdx, *devRootSrc, outcast, slot,
                                              expressionList);
        }
        uint32_t cellMatchTagId = CellMatchBuildTagId(slot.slotAllocIterId, cellMatchTagSeq);
        uint32_t errCode = 0;
        if (slot.stitchCtrlBitMask & (STITCH_CTRL_WAR | STITCH_CTRL_WAW)) {
            if (slot.isPartialUpdateStitch) {
                if (slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size() == 0 &&
                    slot.partialUpdate->cellMatchRuntimePartialUpdateTable.Data() != nullptr) {
                    PrepareRuntimeDynamicPartialUpdateTable(workspace, slot.partialUpdate);
                }
                auto producerList = &devRootSrc->At(outcast.producerConsumerList, 0);
                errCode = UpdateSlotsForOutCastPartialStitch(entry.slotIdx, slot, devRootSrc, outcast, producerList,
                                                             cellMatchTagId, devNextIdx, expressionList);
            } else {
                errCode = UpdateSlotsForOutCastFullCoverStitch(entry.slotIdx, devRootSrc, outcast, cellMatchTagId,
                                                               expressionList);
            }
        }
        workspace->RuntimeOutcastTensorAssign(slot.rtOutcastIter, outcastDesc.GetRtOutcastIter());
        DEV_VERBOSE_DEBUG("[UpdateSlots]   Outcast [%zu] to slot [%d], address %s, ret = 0x%x.", i, entry.slotIdx,
                          outcastDesc.Dump().c_str(), errCode);
        if (errCode != 0) {
            retCode = errCode;
        }
    }
    if (retCode != 0) {
        return retCode;
    }

    size_t updateIncastListSize = devRootSrc->GetUpdateIncastSlotListSize();
    for (size_t i = 0; i < updateIncastListSize; ++i) {
        auto& entry = devRootSrc->GetUpdateIncastSlot(i);
        auto& incast = devRootSrc->GetIncast(entry.incastIdx);
        auto& slot = slotList[entry.slotIdx];
        DEV_VERBOSE_DEBUG("[UpdateSlots]   Begin update Incast [%d] from slot [%3d].", entry.incastIdx, entry.slotIdx);
        uint32_t errCode = UpdateSlotsForIncastStitch(entry.slotIdx, slot, devRootSrc, incast, devTaskId, devNextIdx,
                                                      expressionList, cellMatchTagSeq);
        DEV_VERBOSE_DEBUG("[UpdateSlots]   Incast [%d] from slot [%3d], ret = 0x%x.", entry.incastIdx, entry.slotIdx,
                          errCode);
        if (errCode != 0) {
            return errCode;
        }
    }
    return 0;
}

uint32_t DeviceSlotContext::UpdateSlots(DevAscendFunctionDupped& devRootDup, uint32_t devTaskId, uint32_t devNextIdx,
                                        uint32_t cellMatchTagSeq)
{
    return UpdateSlotsImpl(workspace_, slotList_.data(), devRootDup, devTaskId, devNextIdx, cellMatchTagSeq);
}

static void MarkPartialUpdateSlots(DeviceExecuteSlot* slotList, size_t slotSize, DevAscendProgram* devProg)
{
    for (size_t index = 0, ie = devProg->updateList.size(); index < ie; index++) {
        auto& update = devProg->At(devProg->updateList, index);
        int slotIndex = update.slotIndex;
        if (slotIndex < 0) {
            continue;
        }
        DEV_ASSERT_MSG(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, slotIndex < static_cast<int>(slotSize),
                       "Invalid slot index %d", slotIndex);
        slotList[slotIndex].stitchCtrlBitMask = update.stitchCtrlBitMask;
        if (update.isPartial) {
            slotList[slotIndex].isPartialUpdateStitch = true;
            slotList[slotIndex].partialUpdate = &update;
            DEV_VERBOSE_DEBUG("Partial Update Slot %d mask=0x%x.\n", slotIndex,
                              static_cast<unsigned>(update.stitchCtrlBitMask));
        } else {
            DEV_VERBOSE_DEBUG("FullCover Update Slot %d mask=0x%x.\n", slotIndex,
                              static_cast<unsigned>(update.stitchCtrlBitMask));
        }
    }
}

static void FillExternalTensorSlot(DeviceExecuteSlot* slotList, size_t slotSize, DeviceWorkspaceAllocator* workspace,
                                   int slotIndex, uint64_t tensorAddr, int tensorIndex, bool isInput)
{
    DEV_ASSERT_MSG(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                   slotIndex >= 0 && slotIndex < static_cast<int>(slotSize), "Invalid slot index %d", slotIndex);
    slotList[slotIndex].rtOutcastIter = workspace->MakeRuntimeOutcastTensorBump(WsAllocation(tensorAddr, 0),
                                                                                RuntimeTensorMemProperty::EXTERNAL);
    slotList[slotIndex].isOutputSlot = true;
    DEV_INFO("Param %d %s Slot %d = %lx.", tensorIndex, isInput ? "Input" : "Output", slotIndex, tensorAddr);
}

void DeviceSlotContext::FillInputOutputSlot(DeviceExecuteSlot* slotList, [[maybe_unused]] size_t slotSize,
                                            DevAscendProgram* devProg, DevStartArgs* args)
{
    const int inputSize = args->GetInputTensorSize();
    const int outputSize = args->GetOutputTensorSize();
    const uint64_t* inputSlotIdx = devProg->startArgsInputTensorSlotIndexList.Data();
    const uint64_t* outputSlotIdx = devProg->startArgsOutputTensorSlotIndexList.Data();
    DevTensorData* tensors = args->devTensorList;

    DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorCount(inputSize)));
    for (int index = 0; index < inputSize; ++index) {
        FillExternalTensorSlot(slotList, slotSize, workspace_, static_cast<int>(inputSlotIdx[index]),
                               tensors[index].address, index, true);
    }
    DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorCount(outputSize)));
    for (int index = 0; index < outputSize; ++index) {
        FillExternalTensorSlot(slotList, slotSize, workspace_, static_cast<int>(outputSlotIdx[index]),
                               tensors[inputSize + index].address, index, false);
    }
    const size_t outputSlotNum = devProg->startArgsOutputTensorSlotIndexList.size();
    const uint64_t* inplaceSlots = devProg->outputInplaceSlotList.Data();
    for (size_t index = static_cast<size_t>(outputSize); index < outputSlotNum; ++index) {
        int outSlot = static_cast<int>(outputSlotIdx[index]);
        int inSlot = static_cast<int>(inplaceSlots[index]);
        if (inSlot != -1) {
            DEV_ASSERT_MSG(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                           outSlot >= 0 && outSlot < static_cast<int>(slotSize), "Invalid slot index %d", outSlot);
            DEV_ASSERT_MSG(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                           inSlot >= 0 && inSlot < static_cast<int>(slotSize), "Invalid slot index %d", inSlot);
            workspace_->RuntimeOutcastTensorAssign(slotList[outSlot].rtOutcastIter, slotList[inSlot].rtOutcastIter);
            slotList[outSlot].isOutputSlot = true;
            DEV_VERBOSE_DEBUG("Param %zu Output Slot %d = inSlot %d.", index, outSlot, inSlot);
        }
    }
    const uint64_t* assembleSlots = devProg->assembleSlotIndexList.Data();
    const size_t assembleNum = devProg->assembleSlotIndexList.size();
    for (size_t index = 0; index < assembleNum; ++index) {
        int slotIndex = static_cast<int>(assembleSlots[index]);
        DEV_ASSERT_MSG(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                       slotIndex >= 0 && slotIndex < static_cast<int>(slotSize), "Invalid slot index %d", slotIndex);
        slotList[slotIndex].isAssembleSlot = true;
        DEV_VERBOSE_DEBUG("Assemble Slot %d .", slotIndex);
    }
    MarkPartialUpdateSlots(slotList, slotSize, devProg);
}

} // namespace npu::tile_fwk::dynamic
