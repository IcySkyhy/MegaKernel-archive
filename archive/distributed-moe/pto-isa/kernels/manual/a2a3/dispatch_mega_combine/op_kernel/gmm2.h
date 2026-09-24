/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_H
#define DISPATCH_MEGA_COMBINE_GMM2_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"

using Gmm2Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm2 {
public:
    AICORE inline void Init(
        GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize);
    AICORE inline void ProcessFixedHelper(uint32_t groupLocalId);

private:
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, outputN_, tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline void RunGmmTile(
        Gmm2Pipeline& gmmPipeline, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM, uint32_t loopIdx) const
    {
        GmmCommonRunTile(
            gmmPipeline, gmPermutedTokenPtr_, weight2Ptr_, gmm2OutputPtr_, scale2Ptr_, groupIdx, groupBase, currentM,
            loopIdx, outputN_, inputK_, inputK_, inputK_, outputN_, outputN_, tilingData_->gmm2Tiling.l1TileM,
            tilingData_->gmm2Tiling.l1TileN);
    }
    AICORE inline uint32_t GroupBaseBefore(uint32_t groupIdx) const;
    AICORE inline uint32_t StartCoreBefore(uint32_t groupIdx, uint32_t coreNum) const;
    AICORE inline int32_t PrimaryJoinDecision(uint32_t groupIdx) const;
    AICORE inline uint32_t HelperJoinExpert() const;
    AICORE inline void ProcessImpl(bool helperGroup);

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;

    __gm__ int8_t* gmPermutedTokenPtr_ = nullptr;
    __gm__ half* gmm2OutputPtr_ = nullptr;
    __gm__ int8_t* weight2Ptr_ = nullptr;
    __gm__ uint64_t* scale2Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;

    uint32_t inputK_ = 0;
    uint32_t outputN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t primaryLocalId_ = 0;
};

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::Init(
    GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    (void)sizeof(InputElement);
    (void)expertTokenNumsGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    const uint32_t problemN = tilingData_->megaMoeInfo.N;
    const uint32_t problemK = tilingData_->megaMoeInfo.K;
    inputK_ = problemN / 2U;
    outputN_ = problemK;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmPermutedTokenPtr_ =
        reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData_->swigluTiling.gmPermutedTokenOffset);
    gmm2OutputPtr_ = reinterpret_cast<__gm__ half*>(workspaceGM + tilingData_->gmm2Tiling.gmm2OutputOffset);
    weight2Ptr_ = reinterpret_cast<__gm__ int8_t*>(weight2GM);
    scale2Ptr_ = reinterpret_cast<__gm__ uint64_t*>(scale2GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
{
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    primaryLocalId_ = groupLocalId;
    ProcessImpl(false);
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::ProcessFixedHelper(uint32_t groupLocalId)
{
    coreIdx_ = groupLocalId;
    coreNum_ = tilingData_->fixedGroupTiling.physicalAicNum;
    primaryLocalId_ = groupLocalId;
    ProcessImpl(true);
}

template <typename InputElement>
AICORE inline uint32_t Gmm2<InputElement>::GroupBaseBefore(uint32_t groupIdx) const
{
    uint32_t groupBase = 0U;
    for (uint32_t expert = 0U; expert < groupIdx; ++expert) {
        const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, expert);
        groupBase += MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
    }
    return groupBase;
}

template <typename InputElement>
AICORE inline uint32_t Gmm2<InputElement>::StartCoreBefore(uint32_t groupIdx, uint32_t coreNum) const
{
    uint32_t groupBase = 0U;
    uint32_t startCoreIdx = 0U;
    for (uint32_t expert = 0U; expert < groupIdx; ++expert) {
        const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, expert);
        const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
        groupBase += currentM;
        startCoreIdx = GmmCommonNextStartCoreIdx(startCoreIdx, coreNum, CoreLoops(currentM));
    }
    return startCoreIdx;
}

template <typename InputElement>
AICORE inline int32_t Gmm2<InputElement>::PrimaryJoinDecision(uint32_t groupIdx) const
{
    volatile __gm__ int32_t* joinSlot =
        FixedSyncSlot(workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).gmm2JoinSlot);
    const int32_t expectedDecision = static_cast<int32_t>(groupIdx + 1U);
    if (primaryLocalId_ != 0U) {
        return WaitEpochAcquire(joinSlot, expectedDecision);
    }

    const int32_t done =
        ReadScalarEpoch(FixedSyncSlot(workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).gmm1DoneSlot));
    const int32_t decision =
        done >= kMegaMoeFixedGmm1DoneMarker ? (kMegaMoeFixedGmm2JoinDecisionBit | expectedDecision) : expectedDecision;
    PublishScalarEpoch(joinSlot, decision);
    return decision;
}

template <typename InputElement>
AICORE inline uint32_t Gmm2<InputElement>::HelperJoinExpert() const
{
    const int32_t decision = WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).gmm2JoinSlot),
        kMegaMoeFixedGmm2JoinDecisionBit);
    const int32_t encodedExpert = decision & kMegaMoeFixedGmm2JoinDecisionMask;
    return encodedExpert > 0 ? static_cast<uint32_t>(encodedExpert - 1) : expertPerRank_;
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::ProcessImpl(bool helperGroup)
{
    if ASCEND_IS_AIV {
        return;
    }
    Gmm2Pipeline gmmPipeline;
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    uint32_t firstGroupIdx = 0U;
    bool joinedGroup = helperGroup;
    if (helperGroup) {
        firstGroupIdx = HelperJoinExpert();
        if (firstGroupIdx >= expertPerRank_) {
            return;
        }
        groupBase = GroupBaseBefore(firstGroupIdx);
        startCoreIdx = StartCoreBefore(firstGroupIdx, tilingData_->fixedGroupTiling.gmm2GroupSize);
    }
    for (uint32_t groupIdx = firstGroupIdx; groupIdx < expertPerRank_; ++groupIdx) {
        const MegaMoeSyncLayout sync = FixedSyncLayout(tilingData_);
        const uint32_t readyLocalId = coreIdx_ % tilingData_->fixedGroupTiling.gmm2GroupSize;
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, sync.gmm2ReadyBase + readyLocalId),
            static_cast<int32_t>(groupIdx * 2U + 2U));
        if (!helperGroup && !joinedGroup && groupIdx >= tilingData_->fixedGroupTiling.gmm2JoinCheckStartExpert) {
            const int32_t decision = PrimaryJoinDecision(groupIdx);
            const uint32_t encodedJoinExpert = static_cast<uint32_t>(decision & kMegaMoeFixedGmm2JoinDecisionMask);
            const bool joinThisExpert = (decision & kMegaMoeFixedGmm2JoinDecisionBit) != 0 && encodedJoinExpert != 0U &&
                                        groupIdx + 1U >= encodedJoinExpert;
            if (joinThisExpert) {
                joinedGroup = true;
                coreIdx_ = tilingData_->fixedGroupTiling.gmm1GroupSize + primaryLocalId_;
                coreNum_ = tilingData_->fixedGroupTiling.physicalAicNum;
            }
        }
        const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
        const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
        const uint32_t coreLoops = CoreLoops(currentM);
        const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
            RunGmmTile(gmmPipeline, groupIdx, groupBase, currentM, loopIdx);
        }
        gmmPipeline.SynchronizeBlock();
        if (tilingData_->frontReorderTiling.stageNum >= 13U) {
            const uint32_t arrivalLocalId =
                helperGroup ? primaryLocalId_ : tilingData_->fixedGroupTiling.gmm1GroupSize + primaryLocalId_;
            PublishGroupArrival(workspaceGM_, tilingData_, sync.gmm2ArrivalBase, arrivalLocalId, groupIdx);
        }
        groupBase += currentM;
        startCoreIdx = GmmCommonNextStartCoreIdx(startCoreIdx, coreNum_, coreLoops);
    }
    if (!helperGroup && !joinedGroup && primaryLocalId_ == 0U) {
        const int32_t sentinel = kMegaMoeFixedGmm2JoinDecisionBit + static_cast<int32_t>(expertPerRank_ + 1U);
        PublishScalarEpoch(
            FixedSyncSlot(workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).gmm2JoinSlot), sentinel);
    }
}

#endif // DISPATCH_MEGA_COMBINE_GMM2_H
