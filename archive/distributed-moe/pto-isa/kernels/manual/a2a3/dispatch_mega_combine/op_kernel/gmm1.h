/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM1_H
#define DISPATCH_MEGA_COMBINE_GMM1_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"

using Gmm1Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm1 {
public:
    AICORE inline void Init(
        GM_ADDR weight1GM, GM_ADDR scale1GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize);

private:
    AICORE inline bool ParticipantBalancedNPartition(uint32_t currentM) const
    {
        const uint32_t tileM = tilingData_->gmm1Tiling.l1TileM;
        const uint32_t tileN = tilingData_->gmm1Tiling.l1TileN;
        const uint32_t defaultLoops = GmmCommonCoreLoops(currentM, problemN_, tileM, tileN);
        constexpr uint32_t kNAlign = 32U;
        return GmmCommonTileM(currentM, tileM) == 1U && defaultLoops < coreNum_ && problemN_ % kNAlign == 0U &&
               problemN_ / kNAlign >= coreNum_;
    }
    AICORE inline uint32_t CoreLoops(uint32_t currentM, bool participantBalanced) const
    {
        if (participantBalanced) {
            return coreNum_;
        }
        return GmmCommonCoreLoops(
            currentM, problemN_, tilingData_->gmm1Tiling.l1TileM, tilingData_->gmm1Tiling.l1TileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline void WaitDispatchGroupReady(uint32_t groupIdx) const
    {
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).dispatchReadyBase + coreIdx_),
            static_cast<int32_t>(groupIdx * 2U + 2U));
    }
    AICORE inline GmmCommonTileInfo ParticipantBalancedTileInfo(uint32_t currentM, uint32_t loopIdx) const
    {
        constexpr uint32_t kNAlign = 32U;
        const uint32_t nUnits = problemN_ / kNAlign;
        const uint32_t baseUnits = nUnits / coreNum_;
        const uint32_t wideTileCount = nUnits % coreNum_;
        const uint32_t widePrefix = loopIdx < wideTileCount ? loopIdx : wideTileCount;

        GmmCommonTileInfo tileInfo;
        tileInfo.tileM = 1U;
        tileInfo.tileN = coreNum_;
        tileInfo.blockM = 0U;
        tileInfo.blockN = loopIdx;
        tileInfo.actualM = currentM;
        tileInfo.actualN = (baseUnits + (loopIdx < wideTileCount ? 1U : 0U)) * kNAlign;
        tileInfo.blockRowStart = 0U;
        tileInfo.blockColStart = (loopIdx * baseUnits + widePrefix) * kNAlign;
        return tileInfo;
    }
    AICORE inline void RunGmmTile(
        Gmm1Pipeline& gmmPipeline, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM, bool participantBalanced,
        uint32_t loopIdx) const
    {
        if (participantBalanced) {
            const GmmCommonTileInfo tileInfo = ParticipantBalancedTileInfo(currentM, loopIdx);
            GmmCommonRunTileInfo(
                gmmPipeline, gmAPtr_, weight1Ptr_, gmCPtr_, scale1Ptr_, groupIdx, groupBase, tileInfo, problemN_,
                problemK_, problemK_, problemK_, problemN_, problemN_);
            return;
        }
        GmmCommonRunTile(
            gmmPipeline, gmAPtr_, weight1Ptr_, gmCPtr_, scale1Ptr_, groupIdx, groupBase, currentM, loopIdx, problemN_,
            problemK_, problemK_, problemK_, problemN_, problemN_, tilingData_->gmm1Tiling.l1TileM,
            tilingData_->gmm1Tiling.l1TileN);
    }
    AICORE inline void ProcessImpl();

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;

    __gm__ int8_t* gmAPtr_ = nullptr;
    __gm__ half* gmCPtr_ = nullptr;
    __gm__ int8_t* weight1Ptr_ = nullptr;
    __gm__ uint64_t* scale1Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t problemN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::Init(
    GM_ADDR weight1GM, GM_ADDR scale1GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    (void)expertTokenNumsGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    problemK_ = tilingData_->megaMoeInfo.K;
    problemN_ = tilingData_->megaMoeInfo.N;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmAPtr_ = reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
    gmCPtr_ = reinterpret_cast<__gm__ half*>(workspaceGM + tilingData_->gmm1Tiling.gmCOffset);
    weight1Ptr_ = reinterpret_cast<__gm__ int8_t*>(weight1GM);
    scale1Ptr_ = reinterpret_cast<__gm__ uint64_t*>(scale1GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
}

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
{
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    ProcessImpl();
}

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::ProcessImpl()
{
    if ASCEND_IS_AIV {
        return;
    }
    Gmm1Pipeline gmmPipeline;
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) {
        const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
        coreNum_ = groupIdx < fixed.fullAicGmm1ExpertCount ? fixed.physicalAicNum : fixed.gmm1GroupSize;
        if (coreIdx_ >= coreNum_) {
            break;
        }
        startCoreIdx %= coreNum_;
        WaitDispatchGroupReady(groupIdx);
        const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
        const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
        const bool participantBalanced = ParticipantBalancedNPartition(currentM);
        const uint32_t coreLoops = CoreLoops(currentM, participantBalanced); // 当前 expert 一共有多少个 GMM tile
        const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx); // 当前 AIC 在这个 expert 里的第一个 tile id
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
            RunGmmTile(gmmPipeline, groupIdx, groupBase, currentM, participantBalanced, loopIdx);
        }
        gmmPipeline.SynchronizeBlock();
        if (groupIdx < fixed.fullAicGmm1ExpertCount) {
            pto::SYNCALL<pto::SyncCoreType::AICOnly>();
            if (coreIdx_ == 0U) {
                pipe_barrier(PIPE_ALL);
                dsb(DSB_DDR);
                PublishScalarEpochRange(
                    workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).swigluReadyBase,
                    fixed.swigluActiveGroupSize, static_cast<int32_t>(groupIdx * 2U + 2U));
            }
        } else {
            PublishGroupArrival(
                workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).gmm1ArrivalBase, coreIdx_, groupIdx);
        }
        groupBase += currentM;
        startCoreIdx = GmmCommonNextStartCoreIdx(startCoreIdx, coreNum_, coreLoops);
    }
}

#endif // DISPATCH_MEGA_COMBINE_GMM1_H
