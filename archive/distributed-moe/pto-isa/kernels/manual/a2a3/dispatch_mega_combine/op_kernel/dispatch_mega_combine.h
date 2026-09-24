/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_H
#define DISPATCH_MEGA_COMBINE_H

#include "kernel_operator.h"

#include "combine.h"
#include "dispatch_mega_combine_tiling.h"
#include "dispatch.h"
#include "front_reorder.h"
#include "front_fullload_sort.h"
#include "front_vms_sort.h"
#include "gmm1.h"
#include "gmm2.h"
#include "kernel_launch.hpp"
#include "swiglu.h"
#include "unpermute.h"
#include "utils/mega_expert_sync.hpp"

template <typename InputElement, uint32_t ExpertPerRank>
AICORE inline void FrontRunVmsSort(FrontReorderVmsSort<InputElement>& path)
{
    if ASCEND_IS_AIV {
        path.RunSort();
        FrontRunPostSortPipeline<InputElement, ExpertPerRank>(path.common());
    }
}

template <typename InputElement, uint32_t ExpertPerRank>
AICORE inline void FrontRunFullLoad(FrontReorderFullLoad<InputElement>& path)
{
    if ASCEND_IS_AIV {
        path.RunFullLoadSort();
        path.StoreExpandedRowIdxToGm();
        path.BuildLocalTokenPerExpertFromSort();
        path.QuantAndScatterPackedRows();
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        FrontFinalizeRankMetadata<ExpertPerRank>(path.common());
    }
}

template <typename InputElement, uint32_t ExpertPerRank>
AICORE inline void FrontReorderProcess(
    GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    FrontReorderCommonState state;
    const uint32_t frontCase = tilingData->frontReorderTiling.frontCase;
    if (frontCase == kFrontCaseFullLoadDynamic) {
        FrontReorderFullLoad<InputElement> path(state);
        path.Init(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
        FrontRunFullLoad<InputElement, ExpertPerRank>(path);
        return;
    }
    if (frontCase == kFrontCaseOneCoreDynamic || frontCase == kFrontCaseMultiCoreDynamic) {
        FrontReorderVmsSort<InputElement> path(state);
        path.Init(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
        FrontRunVmsSort<InputElement, ExpertPerRank>(path);
    }
}

template <typename CType_, uint32_t ExpertPerRank>
class MegaMoe {
public:
    __aicore__ inline void Init(
        GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
        GM_ADDR probs, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessFixedGroups(uint16_t stageNum);
    __aicore__ inline void ProcessFixedGmm1(uint32_t physicalBlockId);

    GM_ADDR xGM_ = nullptr;
    GM_ADDR weight1GM_ = nullptr;
    GM_ADDR weight2GM_ = nullptr;
    GM_ADDR scale1GM_ = nullptr;
    GM_ADDR scale2GM_ = nullptr;
    GM_ADDR expertIdGM_ = nullptr;
    GM_ADDR expertTokenNumsGM_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR probsGM_ = nullptr;
    GM_ADDR outGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
};

template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::Init(
    GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
    GM_ADDR probs, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    xGM_ = xGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    scale1GM_ = scale1GM;
    scale2GM_ = scale2GM;
    expertIdGM_ = expertIdGM;
    expertTokenNumsGM_ = expertTokenNums;
    workspaceGM_ = workspaceGM;
    probsGM_ = probs;
    outGM_ = outGM;
    tilingData_ = tilingData;
}

template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::ProcessFixedGmm1(uint32_t physicalBlockId)
{
    Gmm1<CType_> gmm1;
    gmm1.Init(weight1GM_, scale1GM_, expertTokenNumsGM_, workspaceGM_, tilingData_);
    gmm1.ProcessFixed(physicalBlockId, tilingData_->fixedGroupTiling.physicalAicNum);
}

template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::ProcessFixedGroups(uint16_t stageNum)
{
    const MegaMoeFixedCoreRoleInfo role = FixedCoreRole(tilingData_);
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
    const bool rankStreaming = tilingData_->unpermuteTiling.unpermuteImplMode == kMegaMoeUnpermuteImplRankStreaming;
    const uint32_t rankStreamingWorkerCount = fixed.physicalAivNum;

    if (role.role == kMegaMoeFixedRoleDispatch && role.groupLocalId < tilingData_->runtimeInfo.rankSize &&
        stageNum >= 9U) {
        DispatchGather<CType_> dispatchGather;
        dispatchGather.Init(expertTokenNumsGM_, workspaceGM_, tilingData_);
        dispatchGather.ProcessFixed(role.groupLocalId, role.groupSize);
    } else if ((role.role == kMegaMoeFixedRoleGmm1 || role.role == kMegaMoeFixedRoleGmm2) && stageNum >= 10U) {
        ProcessFixedGmm1(role.physicalBlockId);
        const bool dynamicGmm2Join = stageNum >= 12U;
        if (dynamicGmm2Join && role.role == kMegaMoeFixedRoleGmm1 && role.groupLocalId == 0U) {
            pipe_barrier(PIPE_ALL);
            dsb(DSB_DDR);
            PublishScalarEpoch(
                FixedSyncSlot(workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).gmm1DoneSlot),
                kMegaMoeFixedGmm1DoneMarker);
        }
        if (stageNum >= 12U && (role.role == kMegaMoeFixedRoleGmm2 || dynamicGmm2Join)) {
            Gmm2<CType_> gmm2;
            gmm2.Init(weight2GM_, scale2GM_, expertTokenNumsGM_, workspaceGM_, tilingData_);
            if (role.role == kMegaMoeFixedRoleGmm2) {
                gmm2.ProcessFixed(role.groupLocalId, role.groupSize);
            } else {
                gmm2.ProcessFixedHelper(role.groupLocalId);
            }
        }
    } else if (
        role.role == kMegaMoeFixedRoleSwiglu && role.groupLocalId < fixed.swigluActiveGroupSize && stageNum >= 11U) {
        Swiglu<CType_> swiglu;
        swiglu.Init(expertTokenNumsGM_, workspaceGM_, tilingData_);
        swiglu.ProcessFixed(role.groupLocalId, fixed.swigluActiveGroupSize);
    } else if (role.role == kMegaMoeFixedRoleCombine && role.groupLocalId < role.groupSize && stageNum >= 13U) {
        Combine<half> combine;
        combine.Init(workspaceGM_, tilingData_);
        combine.ProcessFixed(role.groupLocalId, role.groupSize);
    }

    if ASCEND_IS_AIV {
        if (!rankStreaming && stageNum >= 13U) {
            Combine<half> finalBoundary;
            finalBoundary.Init(workspaceGM_, tilingData_);
            finalBoundary.ProcessFixedFinalBoundary(
                role.role, role.flatAivId, role.role != kMegaMoeFixedRoleCombine || role.groupLocalId < role.groupSize);
        }
        if (rankStreaming && stageNum == 13U && role.physicalBlockId == 0U && role.subblockId == 0U) {
            PtoRemoteWindow remoteWindow;
            remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
            const int32_t epoch = remoteWindow.DataReadyEpoch();
            for (uint32_t producerRank = 0U; producerRank < tilingData_->runtimeInfo.rankSize; ++producerRank) {
                WaitEpochAcquire(remoteWindow.LocalDataReadySlot(producerRank), epoch);
            }
        }

        const uint32_t initialUnpermuteGroupSize = fixed.gmm1GroupSize * 2U;
        const bool initialUnpermuteGroup = role.physicalBlockId < fixed.gmm1GroupSize;
        const uint32_t rankStreamingWorkerIdx = initialUnpermuteGroup ?
                                                    role.physicalBlockId + role.subblockId * fixed.gmm1GroupSize :
                                                    initialUnpermuteGroupSize + role.groupLocalId;
        const bool unpermuteWorker = !rankStreaming || rankStreamingWorkerIdx < rankStreamingWorkerCount;
        if (rankStreaming && stageNum >= 14U && initialUnpermuteGroup) {
            PtoRemoteWindow remoteWindow;
            remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
            const int32_t epoch = remoteWindow.DataReadyEpoch();
            if (role.role == kMegaMoeFixedRoleDispatch) {
                remoteWindow.PublishDispatchRelease(role.groupLocalId, epoch);
            } else if (role.role == kMegaMoeFixedRoleSwiglu) {
                remoteWindow.PublishSwigluRelease(role.groupLocalId, epoch);
            }

            if (rankStreamingWorkerIdx == 0U) {
                remoteWindow.WaitDispatchReleaseMte(fixed.dispatchGroupSize, epoch);
                remoteWindow.WaitSwigluReleaseMte(fixed.swigluGroupSize, epoch);

                uint32_t readyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
                const uint32_t rankCount = tilingData_->runtimeInfo.rankSize;
                const uint32_t expertPerRank = tilingData_->megaMoeInfo.expertPerRank;
                const uint32_t configuredCut = fixed.unpermutePhase1ReadyExpertCount;
                const uint32_t readyCut = configuredCut < expertPerRank ? configuredCut : expertPerRank;
                uint32_t minimumReady = 0U;
                while (minimumReady < readyCut) {
                    minimumReady = remoteWindow.ReadExpertProgressMte(epoch, expertPerRank, readyExpertCounts);
                    if (minimumReady < readyCut) {
                        RemoteWindowSyncPollBackoff();
                    }
                }
                remoteWindow.AcquireDataReady();
                uint32_t readyRankMask = 0U;
                for (uint32_t producerRank = 0U; producerRank < rankCount; ++producerRank) {
                    if (readyExpertCounts[producerRank] >= readyCut) {
                        readyRankMask |= 1U << producerRank;
                    }
                }
                remoteWindow.PublishUnpermutePhase1Progress(readyExpertCounts, rankCount, readyRankMask, epoch);
                const uint32_t initialWorkerCount = rankStreamingWorkerCount < initialUnpermuteGroupSize ?
                                                        rankStreamingWorkerCount :
                                                        initialUnpermuteGroupSize;
                remoteWindow.PublishUnpermuteStartRangeMte(0U, initialWorkerCount, epoch);
            }
        }
        if (stageNum >= 14U && unpermuteWorker) {
            uint32_t workerIdx = role.flatAivId;
            uint32_t workerCount = tilingData_->fixedGroupTiling.physicalAivNum;
            if (rankStreaming) {
                workerIdx = rankStreamingWorkerIdx;
                workerCount = rankStreamingWorkerCount;
                PtoRemoteWindow remoteWindow;
                remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
                WaitEpochAcquire(remoteWindow.LocalUnpermuteStartSlot(workerIdx), remoteWindow.DataReadyEpoch());
            }
            Unpermute<half> unpermute;
            unpermute.Init(workspaceGM_, expertIdGM_, probsGM_, outGM_, tilingData_, workerIdx, workerCount);
            unpermute.Process();
        }
    }
}

template <typename CType_, uint32_t ExpertPerRank>
__aicore__ inline void MegaMoe<CType_, ExpertPerRank>::Process()
{
    const uint16_t stageNum = tilingData_->frontReorderTiling.stageNum;

    FrontReorderProcess<CType_, ExpertPerRank>(xGM_, expertIdGM_, expertTokenNumsGM_, workspaceGM_, tilingData_);
    ProcessFixedGroups(stageNum);
}

#endif // DISPATCH_MEGA_COMBINE_H
