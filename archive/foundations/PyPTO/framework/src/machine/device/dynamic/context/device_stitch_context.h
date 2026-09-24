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
 * \file device_stitch_context.h
 * \brief
 */

#pragma once

#include "machine/device/dynamic/context/device_slot_context.h"
#include "machine/utils/dynamic/dev_workspace.h"

#define ENABLE_STITCH 1

namespace npu::tile_fwk::dynamic {
const int SKIP_EMPTY = -2;
const int INVALID_TOO_AHEAD = -1;
const int NO_DEP = 0;
const int NEEDS_DEP = 1;
using StitchedList = Vector<DevAscendFunctionDupped, WsMemCategory::VECTOR_STITCHED_LIST, DeviceWorkspaceAllocator>;
struct DeviceStitchContext {
    struct StitchReuseContext {
        // changing with stitching progress
        uint32_t firstDupIdx{0};
        int32_t lastNonEmptyDupIdx{-1};
    } stitchReuseContext_;

    void Init(DevAscendProgram* devProg, DeviceWorkspaceAllocator& workspace);
    void Reset();

    void DumpStitchInfo();

    size_t Size() const { return stitchedList_.size(); }
    bool Empty() const { return stitchedList_.empty(); }

    void Append(DevAscendFunctionDupped& devRootDup) { stitchedList_.push_back(devRootDup); }

    const auto& GetStitchedList() const { return stitchedList_; }

    static void CheckStitch(DevAscendFunctionDupped* stitchedList, int size, DevAscendFunctionDupped* nextDup);

    static void CheckStitch(DynDeviceTask* dyntask);

    uint64_t Stitch(DeviceSlotContext& slotContext, DevAscendFunctionDupped& nextDup, size_t devTaskId,
                    size_t devNextIdx, uint32_t cellMatchTagSeq);

    void RecycleTensorWorkspace();

    void DumpSlotInfo(const char* label, DeviceExecuteSlot* slotList, size_t slotSize);

    void DecideSlotAddress(DeviceExecuteSlot* slotList, size_t slotSize);

    int DecideIncastOutcast(uint64_t taskId);

    int MoveTo(DynDeviceTask* dynTask);

    void VerifyStitchedListMemory(DevStartArgs& args) const
    {
        workspace_->VerifyStitchedListMemory(args, stitchedList_.data(), stitchedList_.size());
    }

    static void PushBackTask(DevAscendFunctionDuppedStitchList& stitch, uint32_t coreTask,
                             DeviceWorkspaceAllocator* workspace)
    {
        // Template AllocFn avoids std::function type-erasure on the stitch hot path.
        stitch.PushBack(coreTask, [workspace]() { return workspace->AllocateStitch(); });
    }

    uint32_t stitchedCallOpSize() { return stitchedCallOpSize_; }

private:
    uint32_t stitchedCallOpSize_{0};
    StitchedList stitchedList_;
    DeviceWorkspaceAllocator* workspace_{nullptr};
    DevAscendProgram* devProg_{nullptr};

public:
    enum class StitchKind {
        StitchDefault,
        StitchPartial,
        StitchFullCover,
        StitchReuse,
    };

    static std::string GetStitchKindName(StitchKind kind)
    {
        static std::unordered_map<StitchKind, std::string> stitchNameDict = {
            {StitchKind::StitchDefault, "default"},
            {StitchKind::StitchPartial, "partial"},
            {StitchKind::StitchFullCover, "fullCover"},
            {StitchKind::StitchReuse, "reuse"},
        };
        return stitchNameDict.count(kind) == 0 ? "invalid stitch kind" : stitchNameDict.find(kind)->second;
    }

    static void HandleOneStitch(DevAscendFunctionDupped& producerDup, DevAscendFunctionDupped& consumerDup,
                                DevAscendFunctionDuppedStitchList& producerStitchList, uint32_t producerFuncIndex,
                                size_t producerOperationIdx, size_t consumerIdx, size_t consumerOperationIdx,
                                DeviceWorkspaceAllocator* workspace, StitchKind debugStitchKind, int debugSlotIdx,
                                uint64_t devTaskId);

    static void HandleOneStitch(DevAscendFunctionDupped& producerDup, DevAscendFunctionDupped& consumerDup,
                                uint32_t producerFuncIndex, size_t producerOperationIdx, size_t consumerIdx,
                                size_t consumerOperationIdx, DeviceWorkspaceAllocator* workspace,
                                StitchKind debugStitchKind, int debugSlotIdx, uint64_t devTaskId);

    template <typename T>
    static inline std::string IntVecToStr(DevAscendFunctionDupped& dup, DevLocalVector<T>& vec)
    {
        std::stringstream ss;
        ss << "[";
        ss << dup.GetSource()->At(vec, 0);
        for (size_t i = 1; i < vec.size(); ++i) {
            ss << ", " << dup.GetSource()->At(vec, i);
        }
        ss << "]";
        return ss.str();
    }
    static inline std::string IntVecToStr(const uint64_t shape[DEV_SHAPE_DIM_MAX], int dim)
    {
        std::stringstream ss;
        ss << "[";
        ss << shape[0];
        for (size_t i = 1; i < static_cast<size_t>(dim); ++i) {
            ss << ", " << shape[i];
        }
        ss << "]";
        return ss.str();
    }

    uint64_t PartialUpdateStitchConsumer(DevAscendFunctionDupped& nextDup, size_t devTaskId, size_t devNextIdx,
                                         DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast,
                                         uint32_t cellMatchTagSeq);

    uint64_t FullCoverDefaultUpdateStitch(DevAscendFunctionDupped& nextDup, size_t devTaskId, size_t devNextIdx,
                                          DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionDupped& prevDup,
                                          DevAscendFunctionOutcast& outcast, DevAscendFunctionIncast& incast);

    uint64_t FullCoverUpdateStitch(DevAscendFunctionDupped& nextDup, size_t devTaskId, size_t devNextIdx,
                                   DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast);

    uint64_t PartialUpdateStitchProducer(DevAscendFunctionDupped& nextDup, size_t devTaskId, size_t devNextIdx,
                                         DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionOutcast& outcast,
                                         uint32_t cellMatchTagSeq);

    void ReuseStitch(DevAscendFunctionDupped& nextDup, size_t devNextIdx, size_t devTaskId);

    uint64_t FastStitchConsumer(DeviceExecuteSlot* slotList, DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                size_t devNextIdx, uint32_t cellMatchTagSeq);

    uint64_t FastStitchProducer(DeviceExecuteSlot* slotList, DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                size_t devNextIdx, uint32_t cellMatchTagSeq);

    uint64_t FastStitch(DeviceExecuteSlot* slotList, DevAscendFunctionDupped& nextDup, size_t devTaskId,
                        size_t devNextIdx, uint32_t cellMatchTagSeq);

    static void DumpStitchInfo(DevAscendFunctionDupped* stitchedList, int stitchedSize);

private:
    static bool MemOverlap(uint64_t ahead, uint64_t alength, uint64_t bhead, uint64_t blength)
    {
        return !(ahead + alength <= bhead || bhead + blength <= ahead);
    }

    static void StitchForWorkspaceReuse(DevAscendFunctionDupped* stitchingList, int stitchingSize,
                                        DevAscendFunctionDupped& prevDup, DevAscendFunctionDupped& currDup,
                                        size_t devCurrIdx, DeviceWorkspaceAllocator* workspace, uint64_t devTaskId,
                                        uint32_t preFuncIndex);
};
} // namespace npu::tile_fwk::dynamic
