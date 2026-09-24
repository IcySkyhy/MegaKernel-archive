/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_COMBINE_H
#define DISPATCH_MEGA_COMBINE_COMBINE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "kernel_operator.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kCombineVecTileElems = 8192U;
constexpr uint32_t kCombineBufferNum = 2U;

template <typename OutputElement>
class Combine {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize);
    AICORE inline void ProcessFixedFinalBoundary(uint32_t role, uint32_t flatAivId, bool combineActive);

private:
    static_assert(
        std::is_same_v<OutputElement, half> || std::is_same_v<OutputElement, bfloat16_t>,
        "combine output must be half or bfloat16");

    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CBlockGlobal = pto::GlobalTensor<half, BlockShape, BlockStride, pto::Layout::ND>;
    using DBlockGlobal = pto::GlobalTensor<OutputElement, BlockShape, BlockStride, pto::Layout::ND>;
    using TileC = pto::Tile<pto::TileType::Vec, half, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileFp32 = pto::Tile<pto::TileType::Vec, float, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileD = pto::Tile<pto::TileType::Vec, OutputElement, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    AICORE inline bool RankStreamingEnabled() const
    {
        return tilingData_->unpermuteTiling.unpermuteImplMode == kMegaMoeUnpermuteImplRankStreaming;
    }
    AICORE inline uint32_t CurrentM(uint32_t groupIdx) const
    {
        return MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
    }
    AICORE inline uint32_t CumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        if (srcRank == 0U) {
            return 0U;
        }
        return static_cast<uint32_t>(cumsumMMPtr_[static_cast<uint64_t>(srcRank - 1U) * expertPerRank_ + groupIdx]);
    }
    AICORE inline uint32_t GlobalExpert(uint32_t groupIdx) const { return rank_ * expertPerRank_ + groupIdx; }
    AICORE inline uint32_t RowsRaw(uint32_t srcRank, uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(
            tokenPerExpertPtr_[static_cast<uint64_t>(srcRank) * expertNumAligned_ + GlobalExpert(groupIdx)]);
    }
    AICORE inline uint32_t DstRowOffset(uint32_t srcRank, uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(preSumBeforeRankPtr_[static_cast<uint64_t>(srcRank) * expertPerRank_ + groupIdx]);
    }
    AICORE inline uint32_t RowsClipped(uint32_t srcRowOffset, uint32_t rowsRaw) const
    {
        if (srcRowOffset >= maxOutputSize_) {
            return 0U;
        }
        const uint32_t remaining = maxOutputSize_ - srcRowOffset;
        return rowsRaw > remaining ? remaining : rowsRaw;
    }
    AICORE inline uint32_t LargeLanesPerRank() const
    {
        if (rankSize_ == 0U) {
            return 1U;
        }
        uint32_t lanes = coreNum_ / rankSize_;
        if (lanes == 0U) {
            lanes = 1U;
        }
        const uint32_t configuredLanes = tilingData_->fixedGroupTiling.combineLargeLanesPerRank;
        return lanes > configuredLanes ? configuredLanes : lanes;
    }
    AICORE inline uint32_t DirectLargeTaskCount() const { return rankSize_ * LargeLanesPerRank(); }
    AICORE inline uint32_t DirectLargeWorkerCount() const
    {
        const uint32_t taskCount = DirectLargeTaskCount();
        return coreNum_ < taskCount ? coreNum_ : taskCount;
    }
    AICORE inline uint32_t ReadyCoordinatorCore() const
    {
        const uint32_t workerCount = DirectLargeWorkerCount();
        if (workerCount == 0U) {
            return 0U;
        }
        const uint64_t firstLocalTask = static_cast<uint64_t>(rank_) * LargeLanesPerRank();
        return static_cast<uint32_t>(firstLocalTask % workerCount);
    }
    AICORE inline uint32_t LargeLaneRowBegin(uint32_t rows, uint32_t laneIdx, uint32_t lanesPerRank) const
    {
        const uint32_t safeLanes = lanesPerRank == 0U ? 1U : lanesPerRank;
        return static_cast<uint32_t>((static_cast<uint64_t>(rows) * laneIdx) / safeLanes);
    }
    AICORE inline uint32_t LargeLaneRowNum(uint32_t rows, uint32_t laneIdx, uint32_t lanesPerRank) const
    {
        const uint32_t begin = LargeLaneRowBegin(rows, laneIdx, lanesPerRank);
        const uint32_t end = LargeLaneRowBegin(rows, laneIdx + 1U, lanesPerRank);
        return end > begin ? end - begin : 0U;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId + 2U); }
    AICORE inline event_t StoreFreeEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t StoreReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId + 2U); }
    AICORE inline void InitUbLayout();
    AICORE inline void SetInitialFlags() const;
    AICORE inline void FinalizeLocalPipe() const;
    AICORE inline void FinalizeExpertStores() const;
    AICORE inline uint32_t TokenPerExpertResetElems() const;
    AICORE inline bool ResetTokenPerExpertByOwner(uint32_t elems, bool resetOwner) const;
    AICORE inline void PublishAssignedExpertProgress(uint32_t readyExpertCount) const;
    AICORE inline void FinalizeRankStreamingLane();
    AICORE inline uint32_t Gmm2ProducerCount(uint32_t groupIdx) const;
    AICORE inline void WaitGmm2Ready(uint32_t groupIdx) const;
    AICORE inline void ProcessDirectLargeSegmentRows(
        uint32_t srcRank, uint32_t srcRowOffset, uint32_t rows, uint32_t dstRowOffset);
    AICORE inline void ProcessDirectLargeTokenPath();

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    __gm__ half* gmm2OutputPtr_ = nullptr;
    __gm__ float* perTokenScale2Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    __gm__ int32_t* preSumBeforeRankPtr_ = nullptr;
    __gm__ int32_t* tokenPerExpertPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t expertNumAligned_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t pingpongId_ = 0;
    int32_t dataReadyEpoch_ = 0;
    uint64_t ubCOffset_[kCombineBufferNum] = {0, 0};
    uint64_t ubFp32Offset_[kCombineBufferNum] = {0, 0};
    uint64_t ubDOffset_[kCombineBufferNum] = {0, 0};
};

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    pingpongId_ = 0;

    problemK_ = tilingData_->megaMoeInfo.K;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    expertNumAligned_ = tilingData_->frontReorderTiling.expertNumAligned;
    rank_ = tilingData_->runtimeInfo.rank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();
    if ASCEND_IS_AIV {
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();
    }

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    peerMemoryLayout_.Init(remoteWindow_);

    gmm2OutputPtr_ = reinterpret_cast<__gm__ half*>(workspaceGM + tilingData_->combineTiling.gmm2OutputOffset);
    perTokenScale2Ptr_ = reinterpret_cast<__gm__ float*>(workspaceGM + tilingData_->combineTiling.perTokenScale2Offset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
    preSumBeforeRankPtr_ =
        reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.preSumBeforeRankOffset);
    tokenPerExpertPtr_ =
        reinterpret_cast<__gm__ int32_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetPeerTokenPerExpert);
    dataReadyEpoch_ = RankStreamingEnabled() ? remoteWindow_.DataReadyEpoch() : 0;

    InitUbLayout();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::InitUbLayout()
{
    uint64_t ubOffset = 0;
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        ubCOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(problemK_) * sizeof(half), UB_ALIGN);
        ubDOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(problemK_) * sizeof(OutputElement), UB_ALIGN);
        ubFp32Offset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(problemK_) * sizeof(float), UB_ALIGN);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::SetInitialFlags() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::FinalizeLocalPipe() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::FinalizeExpertStores() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::TokenPerExpertResetElems() const
{
    return rankSize_ * expertNumAligned_;
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::ResetTokenPerExpertByOwner(uint32_t elems, bool resetOwner) const
{
    if (!resetOwner) {
        return false;
    }
    PtoFillUb<int32_t>(0U, 0, elems);
    pipe_barrier(PIPE_ALL);
    PtoStoreVector<int32_t>(tokenPerExpertPtr_, 0U, elems);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return true;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFixedFinalBoundary(
    uint32_t role, uint32_t flatAivId, bool combineActive)
{
    if (role == kMegaMoeFixedRoleCombine && combineActive) {
        FinalizeLocalPipe();
    }
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    ResetTokenPerExpertByOwner(TokenPerExpertResetElems(), flatAivId + 1U == kMegaMoeFixedPhysicalAivNum);
    remoteWindow_.CrossRankSync();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PublishAssignedExpertProgress(uint32_t readyExpertCount) const
{
    const uint32_t lanesPerRank = LargeLanesPerRank();
    const uint32_t taskCount = DirectLargeTaskCount();
    for (uint32_t taskIdx = coreIdx_; taskIdx < taskCount; taskIdx += coreNum_) {
        const uint32_t ownerRank = taskIdx / lanesPerRank;
        remoteWindow_.PublishExpertProgress(static_cast<int32_t>(ownerRank), readyExpertCount, dataReadyEpoch_);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::FinalizeRankStreamingLane()
{
    const uint32_t taskCount = DirectLargeTaskCount();
    const uint32_t combineWorkerCount = DirectLargeWorkerCount();
    if (coreIdx_ >= combineWorkerCount) {
        return;
    }

    remoteWindow_.AcquireDataReady();
    const uint32_t lanesPerRank = LargeLanesPerRank();
    for (uint32_t taskIdx = coreIdx_; taskIdx < taskCount; taskIdx += coreNum_) {
        const uint32_t ownerRank = taskIdx / lanesPerRank;
        remoteWindow_.PublishExpertProgress(static_cast<int32_t>(ownerRank), expertPerRank_, dataReadyEpoch_);
        remoteWindow_.PublishDataReady(static_cast<int32_t>(ownerRank), dataReadyEpoch_);
    }
    remoteWindow_.PublishLocalCombineDone(coreIdx_, dataReadyEpoch_);

    if (coreIdx_ != 0U) {
        return;
    }
    remoteWindow_.WaitLocalCombineDoneMte(combineWorkerCount, dataReadyEpoch_);
    ResetTokenPerExpertByOwner(TokenPerExpertResetElems(), true);
    const uint32_t workerCount = tilingData_->fixedGroupTiling.physicalAivNum;
    const uint32_t initialWorkerCount = tilingData_->fixedGroupTiling.gmm1GroupSize * 2U;
    const uint32_t helperCount = workerCount > initialWorkerCount ? workerCount - initialWorkerCount : 0U;
    remoteWindow_.PublishUnpermuteStartRangeMte(initialWorkerCount, helperCount, dataReadyEpoch_);
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::Gmm2ProducerCount(uint32_t groupIdx) const
{
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
    if (groupIdx < fixed.gmm2JoinCheckStartExpert) {
        return fixed.gmm2GroupSize;
    }

    const int32_t decision = WaitEpochAcquire(
        FixedSyncSlot(workspaceGM_, tilingData_, FixedSyncLayout(tilingData_).gmm2JoinSlot),
        static_cast<int32_t>(groupIdx + 1U));
    const uint32_t encodedExpert = static_cast<uint32_t>(decision & kMegaMoeFixedGmm2JoinDecisionMask);
    const bool joined =
        (decision & kMegaMoeFixedGmm2JoinDecisionBit) != 0 && encodedExpert != 0U && groupIdx + 1U >= encodedExpert;
    return joined ? fixed.physicalAicNum : fixed.gmm2GroupSize;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::WaitGmm2Ready(uint32_t groupIdx) const
{
    const MegaMoeSyncLayout sync = FixedSyncLayout(tilingData_);
    const uint32_t readyLocalId = coreIdx_ % tilingData_->fixedGroupTiling.gmm2GroupSize;
    const uint32_t readySlot = sync.combineReadyBase + readyLocalId;
    const int32_t expectedEpoch = static_cast<int32_t>(groupIdx * 2U + 2U);
    if (coreIdx_ == ReadyCoordinatorCore()) {
        const uint32_t producerCount = Gmm2ProducerCount(groupIdx);
        const uint32_t producerBaseOffset = producerCount == tilingData_->fixedGroupTiling.physicalAicNum ?
                                                0U :
                                                tilingData_->fixedGroupTiling.gmm1GroupSize;
        CoordinateGroupConsumersMte(
            workspaceGM_, tilingData_, sync.gmm2ArrivalBase + producerBaseOffset, sync.combineReadyBase, producerCount,
            tilingData_->fixedGroupTiling.gmm2GroupSize, groupIdx);
    } else {
        WaitEpochAcquire(FixedSyncSlot(workspaceGM_, tilingData_, readySlot), expectedEpoch);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectLargeSegmentRows(
    uint32_t srcRank, uint32_t srcRowOffset, uint32_t rows, uint32_t dstRowOffset)
{
    if (rows == 0U) {
        return;
    }
    __gm__ OutputElement* dstBase = reinterpret_cast<__gm__ OutputElement*>(
        remoteWindow_.RemoteBase(peerMemoryLayout_.offsetD, static_cast<int32_t>(srcRank)));
    if (dstBase == nullptr) {
        return;
    }

    for (uint32_t row = 0; row < rows; ++row) { // 遍历每个token
        const uint32_t bufferId = pingpongId_;
        pingpongId_ = (pingpongId_ + 1U) % kCombineBufferNum;
        const uint32_t srcRow = srcRowOffset + row;
        const uint32_t dstRow = dstRowOffset + row;

        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        TileC cTile(1, problemK_);
        pto::TASSIGN(cTile, ubCOffset_[bufferId]);
        BlockShape cShape(1, problemK_);
        BlockStride cStride(problemK_, problemK_, problemK_, problemK_);
        CBlockGlobal cGlobal(gmm2OutputPtr_ + static_cast<uint64_t>(srcRow) * problemK_, cShape, cStride);
        pto::TLOAD(cTile, cGlobal); // load一行token到UB
        set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));

        wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
        TileFp32 fp32Tile(1, problemK_);
        pto::TASSIGN(fp32Tile, ubFp32Offset_[bufferId]);
        pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE); // 将GMM2结果转成FP32
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

        const float scaleValue = perTokenScale2Ptr_[srcRow];
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        pto::TMULS(fp32Tile, fp32Tile, scaleValue); // 将GMM2结果做反量化
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
        TileD dTile(1, problemK_);
        pto::TASSIGN(dTile, ubDOffset_[bufferId]);
        pto::TCVT(dTile, fp32Tile, pto::RoundMode::CAST_RINT); // FP32的结果转成bf16输出
        set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));

        wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
        BlockShape dShape(1, problemK_);
        BlockStride dStride(problemK_, problemK_, problemK_, problemK_);
        DBlockGlobal dGlobal(dstBase + static_cast<uint64_t>(dstRow) * problemK_, dShape, dStride);
        pto::TSTORE(dGlobal, dTile); // store 一行结果到远端GM
        set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectLargeTokenPath()
{
    uint32_t groupBase = 0;
    const bool delayedStart = expertPerRank_ != 0U;
    const uint32_t initialReadyExpert = expertPerRank_ > tilingData_->fixedGroupTiling.combineStartAfterGmm2Expert ?
                                            tilingData_->fixedGroupTiling.combineStartAfterGmm2Expert :
                                            expertPerRank_ - 1U;
    if (delayedStart && coreIdx_ < DirectLargeTaskCount()) {
        WaitGmm2Ready(initialReadyExpert);
    }
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) { // 逐个group遍历
        const uint32_t currentM = CurrentM(groupIdx);                    // expert 总共有多少输出 row
        const uint32_t lanesPerRank = LargeLanesPerRank();
        const uint32_t taskCount = rankSize_ * lanesPerRank;
        if (!delayedStart || groupIdx > initialReadyExpert) {
            WaitGmm2Ready(groupIdx);
        }
        if (lanesPerRank == 0U) {
            continue;
        }
        for (uint32_t taskIdx = coreIdx_; taskIdx < taskCount; taskIdx += coreNum_) {
            const uint32_t safeLanes = lanesPerRank == 0U ? 1U : lanesPerRank;
            const uint32_t srcRank = taskIdx / safeLanes;
            const uint32_t laneIdx = taskIdx - srcRank * lanesPerRank;
            const uint32_t cumsumBeforeSrc = CumsumBeforeSource(srcRank, groupIdx);
            const uint32_t srcRowOffset = groupBase + cumsumBeforeSrc;
            const uint32_t rows = RowsClipped(srcRowOffset, RowsRaw(srcRank, groupIdx));
            const uint32_t rowBegin = LargeLaneRowBegin(rows, laneIdx, lanesPerRank);
            const uint32_t rowNum = LargeLaneRowNum(rows, laneIdx, lanesPerRank); // 分配当前core处理的rownnum
            const uint32_t dstRowOffset = DstRowOffset(srcRank, groupIdx);
            ProcessDirectLargeSegmentRows(srcRank, srcRowOffset + rowBegin, rowNum, dstRowOffset + rowBegin);
        }
        const uint32_t readyExpertCount = groupIdx + 1U;
        const uint32_t phase1ReadyExpertCount = tilingData_->fixedGroupTiling.unpermutePhase1ReadyExpertCount;
        if (RankStreamingEnabled() && lanesPerRank == 1U && coreIdx_ < DirectLargeWorkerCount() &&
            readyExpertCount == phase1ReadyExpertCount && readyExpertCount < expertPerRank_) {
            FinalizeExpertStores();
            PublishAssignedExpertProgress(readyExpertCount);
        }
        groupBase += currentM;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
{
    if ASCEND_IS_AIC {
        return;
    }
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    SetInitialFlags();
    if (coreIdx_ < DirectLargeTaskCount()) {
        ProcessDirectLargeTokenPath();
    }
    if (RankStreamingEnabled()) {
        FinalizeLocalPipe();
        FinalizeRankStreamingLane();
    }
}

#endif // DISPATCH_MEGA_COMBINE_COMBINE_H
