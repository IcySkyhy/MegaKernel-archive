/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HCCL_WINDOW_HPP
#define HCCL_WINDOW_HPP

#include "const_args.hpp"
#include "hccl_window_context.hpp"
#include "kernel_operator.h"

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

constexpr int32_t PTO_REMOTE_WINDOW_MEM = 700 * MB_SIZE;

constexpr uint32_t COMBINE_BARRIER_COUNTER_BASE_INDEX = 8192;
constexpr uint32_t COMBINE_BARRIER_EPOCH_INDEX = 12288;
constexpr uint32_t COMBINE_BARRIER_COUNTER_STRIDE = 16;
constexpr uint32_t START_AIV_BARRIER_COUNTER_BASE_INDEX = 14336;
constexpr uint32_t START_AIV_BARRIER_EPOCH_INDEX = 18432;
constexpr uint32_t START_AIC_BARRIER_COUNTER_BASE_INDEX = 20480;
constexpr uint32_t START_AIC_BARRIER_EPOCH_INDEX = 24576;
// Keep one cache line per producer slot.  These slots are independent from the
// older all-rank barrier counters above and are reused with a launch epoch.
constexpr uint32_t COMBINE_DATA_READY_BASE_INDEX = 26624;
constexpr uint32_t COMBINE_DATA_READY_STRIDE = 16;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_OFFSET = 1;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_COUNT_BITS = 6;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_COUNT_MASK = (1U << COMBINE_EXPERT_PROGRESS_COUNT_BITS) - 1U;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_MAX_RANKS = kMegaMoeExpertProgressMaxRanks;
constexpr uint32_t COMBINE_DATA_READY_EPOCH_INDEX = 28672;
constexpr uint32_t UNPERMUTE_ALL_READY_BASE_INDEX = COMBINE_DATA_READY_EPOCH_INDEX + COMBINE_DATA_READY_STRIDE;
constexpr uint32_t UNPERMUTE_ALL_READY_STRIDE = 16;
constexpr uint32_t UNPERMUTE_ALL_READY_SLOT_COUNT = kMegaMoeFixedPhysicalAivNum;
constexpr uint32_t UNPERMUTE_START_BASE_INDEX =
    UNPERMUTE_ALL_READY_BASE_INDEX + UNPERMUTE_ALL_READY_SLOT_COUNT * UNPERMUTE_ALL_READY_STRIDE;
constexpr uint32_t UNPERMUTE_START_STRIDE = 16;
constexpr uint32_t UNPERMUTE_START_SLOT_COUNT = kMegaMoeFixedPhysicalAivNum;
constexpr uint32_t COMBINE_LOCAL_DONE_BASE_INDEX =
    UNPERMUTE_START_BASE_INDEX + UNPERMUTE_START_SLOT_COUNT * UNPERMUTE_START_STRIDE;
constexpr uint32_t COMBINE_LOCAL_DONE_STRIDE = 16;
constexpr uint32_t COMBINE_LOCAL_DONE_SLOT_COUNT = kMegaMoeFixedPhysicalAivNum;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_BASE_INDEX =
    COMBINE_LOCAL_DONE_BASE_INDEX + COMBINE_LOCAL_DONE_SLOT_COUNT * COMBINE_LOCAL_DONE_STRIDE;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_STRIDE = 32;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_EPOCH_OFFSET = 0;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_MASK_OFFSET = 1;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_COUNTS_OFFSET = 2;
constexpr uint32_t REMOTE_WINDOW_CACHE_LINE_VALUES = 64U / sizeof(int32_t);
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_VALUE_COUNT =
    UNPERMUTE_PHASE1_PROGRESS_COUNTS_OFFSET + COMBINE_EXPERT_PROGRESS_MAX_RANKS;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_MAX_CACHE_LINE_COUNT =
    (UNPERMUTE_PHASE1_PROGRESS_VALUE_COUNT + REMOTE_WINDOW_CACHE_LINE_VALUES - 1U) / REMOTE_WINDOW_CACHE_LINE_VALUES;
constexpr uint32_t UNPERMUTE_DISPATCH_RELEASE_BASE_INDEX =
    UNPERMUTE_PHASE1_PROGRESS_BASE_INDEX + UNPERMUTE_PHASE1_PROGRESS_STRIDE;
constexpr uint32_t UNPERMUTE_DISPATCH_RELEASE_STRIDE = 16;
constexpr uint32_t UNPERMUTE_DISPATCH_RELEASE_SLOT_COUNT = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t UNPERMUTE_SWIGLU_RELEASE_BASE_INDEX =
    UNPERMUTE_DISPATCH_RELEASE_BASE_INDEX + UNPERMUTE_DISPATCH_RELEASE_SLOT_COUNT * UNPERMUTE_DISPATCH_RELEASE_STRIDE;
constexpr uint32_t UNPERMUTE_SWIGLU_RELEASE_STRIDE = 16;
constexpr uint32_t UNPERMUTE_SWIGLU_RELEASE_SLOT_COUNT = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t UNPERMUTE_PHASE1_DONE_BASE_INDEX =
    UNPERMUTE_SWIGLU_RELEASE_BASE_INDEX + UNPERMUTE_SWIGLU_RELEASE_SLOT_COUNT * UNPERMUTE_SWIGLU_RELEASE_STRIDE;
constexpr uint32_t UNPERMUTE_PHASE1_DONE_STRIDE = 16;
constexpr uint32_t UNPERMUTE_PHASE1_DONE_SLOT_COUNT = kMegaMoeFixedPhysicalAivNum;
constexpr uint32_t REMOTE_WINDOW_SYNC_MAX_SLOTS = kMegaMoeFixedPhysicalAivNum;
constexpr uint32_t REMOTE_WINDOW_SYNC_VALUES_PER_SLOT = 16;
constexpr uint32_t REMOTE_WINDOW_SYNC_MAX_VALUES = REMOTE_WINDOW_SYNC_MAX_SLOTS * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
constexpr uint64_t REMOTE_WINDOW_SYNC_SNAPSHOT_BYTES =
    static_cast<uint64_t>(REMOTE_WINDOW_SYNC_MAX_VALUES) * sizeof(int32_t);
constexpr uint64_t REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET = AtlasA2::UB_SIZE - REMOTE_WINDOW_SYNC_SNAPSHOT_BYTES;
constexpr event_t REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT = EVENT_ID0;
static_assert(
    COMBINE_DATA_READY_BASE_INDEX + COMBINE_EXPERT_PROGRESS_MAX_RANKS * COMBINE_DATA_READY_STRIDE <=
    COMBINE_DATA_READY_EPOCH_INDEX);
static_assert(COMBINE_LOCAL_DONE_SLOT_COUNT >= COMBINE_EXPERT_PROGRESS_MAX_RANKS);
static_assert(UNPERMUTE_PHASE1_PROGRESS_VALUE_COUNT <= UNPERMUTE_PHASE1_PROGRESS_STRIDE);
static_assert(
    UNPERMUTE_PHASE1_PROGRESS_MAX_CACHE_LINE_COUNT * REMOTE_WINDOW_CACHE_LINE_VALUES <=
    UNPERMUTE_PHASE1_PROGRESS_STRIDE);
static_assert(UNPERMUTE_PHASE1_PROGRESS_BASE_INDEX % REMOTE_WINDOW_CACHE_LINE_VALUES == 0U);
static_assert(
    UNPERMUTE_PHASE1_DONE_BASE_INDEX + UNPERMUTE_PHASE1_DONE_SLOT_COUNT * UNPERMUTE_PHASE1_DONE_STRIDE <
    MB_SIZE / sizeof(int32_t));
static_assert(REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET % UB_ALIGN == 0U);

AICORE inline void RemoteWindowSyncPollBackoff()
{
    constexpr uint32_t kDelayTicks = 3U;
    const uint64_t deadline = get_sys_cnt() + kDelayTicks;
    while (get_sys_cnt() < deadline) {
        __asm__ __volatile__("");
    }
}

AICORE inline void DcciUnpermutePhase1Progress(volatile __gm__ int32_t* base, uint32_t rankCount)
{
    const uint32_t valueCount = UNPERMUTE_PHASE1_PROGRESS_COUNTS_OFFSET + rankCount;
    const uint32_t cacheLineCount =
        (valueCount + REMOTE_WINDOW_CACHE_LINE_VALUES - 1U) / REMOTE_WINDOW_CACHE_LINE_VALUES;
    for (uint32_t line = 0U; line < cacheLineCount; ++line) {
        dcci((__gm__ void*)(base + line * REMOTE_WINDOW_CACHE_LINE_VALUES), SINGLE_CACHE_LINE);
    }
}

class PtoRemoteWindow {
public:
    AICORE inline PtoRemoteWindow() { segmentBytes_ = PTO_REMOTE_WINDOW_MEM; }

    AICORE inline void Init(GM_ADDR remoteWindowContext)
    {
        context_ = reinterpret_cast<__gm__ PtoRemoteWindowContext*>(remoteWindowContext);
        rank_ = static_cast<int32_t>(context_->rank);
        rankSize_ = static_cast<int32_t>(context_->rankSize);
        segmentBytes_ = static_cast<size_t>(context_->windowBytes);
    }

    AICORE inline GM_ADDR LocalBase() const { return reinterpret_cast<GM_ADDR>(context_->windowIn[rank_]); }

    AICORE inline GM_ADDR RemoteBase(int64_t offset, int32_t rankId) const
    {
        if (offset < 0 || offset >= static_cast<int64_t>(segmentBytes_) || rankId < 0 || rankId >= rankSize_) {
            return nullptr;
        }
        return reinterpret_cast<GM_ADDR>(context_->windowIn[rankId]) + offset;
    }

    template <typename T>
    AICORE inline __gm__ T* RemotePtr(__gm__ T* localPtr, int32_t rankId) const
    {
        const uint64_t localBase = context_->windowIn[rank_];
        const uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
        return reinterpret_cast<__gm__ T*>(context_->windowIn[rankId] + offset);
    }

    AICORE inline size_t SegmentSize() const { return segmentBytes_; }

    AICORE inline int32_t Rank() const { return rank_; }

    AICORE inline int32_t RankSize() const { return rankSize_; }

    AICORE inline __gm__ int32_t* LocalSignalBase() const
    {
        return reinterpret_cast<__gm__ int32_t*>(LocalBase() + segmentBytes_ - MB_SIZE);
    }

    AICORE inline __gm__ int32_t* RemoteSignalBase(int32_t rankId) const
    {
        return RemotePtr(LocalSignalBase(), rankId);
    }

    // The epoch is advanced by one AIV before the first stage starts.  This
    // permits repeated launches even when the host does not clear the window.
    AICORE inline int32_t PrepareDataReadyEpoch() const
    {
        __gm__ int32_t* epochSlot = LocalSignalBase() + COMBINE_DATA_READY_EPOCH_INDEX;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        int32_t epoch = *epochSlot + 1;
        if (epoch <= 0) {
            epoch = 1;
        }
        *epochSlot = epoch;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        PublishUnpermuteAllReady(UNPERMUTE_ALL_READY_SLOT_COUNT, 0);
        dsb(DSB_DDR);
        return epoch;
    }

    AICORE inline int32_t DataReadyEpoch() const
    {
        __gm__ int32_t* epochSlot = LocalSignalBase() + COMBINE_DATA_READY_EPOCH_INDEX;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        return *epochSlot;
    }

    AICORE inline volatile __gm__ int32_t* LocalDataReadySlot(int32_t producerRank) const
    {
        if (producerRank < 0 || producerRank >= rankSize_) {
            return nullptr;
        }
        return LocalSignalBase() + COMBINE_DATA_READY_BASE_INDEX +
               static_cast<uint32_t>(producerRank) * COMBINE_DATA_READY_STRIDE;
    }

    AICORE inline void PublishDataReady(int32_t ownerRank, int32_t epoch) const
    {
        if (ownerRank < 0 || ownerRank >= rankSize_) {
            return;
        }
        if (ownerRank == rank_) {
            volatile __gm__ int32_t* localSlot = LocalDataReadySlot(rank_);
            *localSlot = epoch;
            dcci((__gm__ void*)localSlot, SINGLE_CACHE_LINE);
            __asm__ __volatile__("");
            return;
        }
        __gm__ int32_t* remoteSlot = RemoteSignalBase(ownerRank) + COMBINE_DATA_READY_BASE_INDEX +
                                     static_cast<uint32_t>(rank_) * COMBINE_DATA_READY_STRIDE;
        auto signal = pto::comm::Signal(remoteSlot);
        pto::comm::TNOTIFY(signal, epoch, pto::comm::NotifyOp::Set);
    }

    AICORE inline int32_t EncodeExpertProgress(int32_t epoch, uint32_t readyExpertCount) const
    {
        const uint32_t count = readyExpertCount & COMBINE_EXPERT_PROGRESS_COUNT_MASK;
        return static_cast<int32_t>((static_cast<uint32_t>(epoch) << COMBINE_EXPERT_PROGRESS_COUNT_BITS) | count);
    }

    AICORE inline void PublishExpertProgress(int32_t consumerRank, uint32_t readyExpertCount, int32_t epoch) const
    {
        if (consumerRank < 0 || consumerRank >= rankSize_ || readyExpertCount > COMBINE_EXPERT_PROGRESS_COUNT_MASK ||
            readyExpertCount > kMegaMoeFixedMaxExperts) {
            return;
        }
        const int32_t encoded = EncodeExpertProgress(epoch, readyExpertCount);
        if (consumerRank == rank_) {
            volatile __gm__ int32_t* localSlot = LocalDataReadySlot(rank_) + COMBINE_EXPERT_PROGRESS_OFFSET;
            *localSlot = encoded;
            dcci((__gm__ void*)localSlot, SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            return;
        }
        __gm__ int32_t* remoteSlot = RemoteSignalBase(consumerRank) + COMBINE_DATA_READY_BASE_INDEX +
                                     static_cast<uint32_t>(rank_) * COMBINE_DATA_READY_STRIDE +
                                     COMBINE_EXPERT_PROGRESS_OFFSET;
        auto signal = pto::comm::Signal(remoteSlot);
        pto::comm::TNOTIFY(signal, encoded, pto::comm::NotifyOp::Set);
    }

    AICORE inline void AcquireDataReady() const
    {
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
    }

    // AIV-only snapshot helpers. Each signal owns one 64-byte cache line and
    // only the first int32_t carries the epoch or mask.
    AICORE inline uint32_t ReadEpochMaskMte(__gm__ int32_t* base, uint32_t count, int32_t epoch) const
    {
        if (base == nullptr || count == 0U || count > REMOTE_WINDOW_SYNC_MAX_SLOTS) {
            return 0U;
        }

        using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
        using SnapshotTile =
            pto::Tile<pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_MAX_VALUES, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t snapshotValues = count * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
        SnapshotShape snapshotShape(snapshotValues);
        SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
        SnapshotGlobal snapshotGlobal(base, snapshotShape, snapshotStride);
        SnapshotTile snapshotTile(1U, snapshotValues);
        pto::TASSIGN(snapshotTile, REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET);
        pto::TLOAD(snapshotTile, snapshotGlobal);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);

        uint32_t readyMask = 0U;
        for (uint32_t slot = 0U; slot < count; ++slot) {
            if (snapshotTile.GetValue(slot * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT) >= epoch) {
                readyMask |= 1U << slot;
            }
        }
        return readyMask;
    }

    AICORE inline void PublishEpochRangeMte(__gm__ int32_t* base, uint32_t count, int32_t value) const
    {
        if (base == nullptr || count == 0U || count > REMOTE_WINDOW_SYNC_MAX_SLOTS) {
            return;
        }

        using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
        using SnapshotTile =
            pto::Tile<pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_MAX_VALUES, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t snapshotValues = count * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
        SnapshotShape snapshotShape(snapshotValues);
        SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
        SnapshotGlobal snapshotGlobal(base, snapshotShape, snapshotStride);
        SnapshotTile snapshotTile(1U, snapshotValues);
        pto::TASSIGN(snapshotTile, REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET);
        for (uint32_t slot = 0U; slot < count; ++slot) {
            snapshotTile.SetValue(slot * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, value);
        }
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);
        pto::TSTORE(snapshotGlobal, snapshotTile);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);
    }

    AICORE inline uint32_t ReadDataReadyMaskMte(int32_t epoch) const
    {
        return ReadEpochMaskMte(
            const_cast<__gm__ int32_t*>(LocalDataReadySlot(0)), static_cast<uint32_t>(rankSize_), epoch);
    }

    AICORE inline uint32_t ReadExpertProgressMte(
        int32_t epoch, uint32_t expertPerRank, uint32_t* readyExpertCounts) const
    {
        if (readyExpertCounts == nullptr || rankSize_ <= 0 ||
            static_cast<uint32_t>(rankSize_) > COMBINE_EXPERT_PROGRESS_MAX_RANKS) {
            return 0U;
        }

        using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
        using SnapshotTile =
            pto::Tile<pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_MAX_VALUES, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t count = static_cast<uint32_t>(rankSize_);
        const uint32_t snapshotValues = count * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
        SnapshotShape snapshotShape(snapshotValues);
        SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
        SnapshotGlobal snapshotGlobal(
            const_cast<__gm__ int32_t*>(LocalDataReadySlot(0)), snapshotShape, snapshotStride);
        SnapshotTile snapshotTile(1U, snapshotValues);
        pto::TASSIGN(snapshotTile, REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET);
        pto::TLOAD(snapshotTile, snapshotGlobal);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);

        uint32_t minimumReady = expertPerRank;
        const uint32_t expectedEpoch = static_cast<uint32_t>(epoch);
        for (uint32_t producerRank = 0U; producerRank < count; ++producerRank) {
            const uint32_t encoded = static_cast<uint32_t>(snapshotTile.GetValue(
                producerRank * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT + COMBINE_EXPERT_PROGRESS_OFFSET));
            const uint32_t observedEpoch = encoded >> COMBINE_EXPERT_PROGRESS_COUNT_BITS;
            uint32_t readyCount = observedEpoch == expectedEpoch ? encoded & COMBINE_EXPERT_PROGRESS_COUNT_MASK : 0U;
            if (readyCount > expertPerRank) {
                readyCount = expertPerRank;
            }
            readyExpertCounts[producerRank] = readyCount;
            minimumReady = readyCount < minimumReady ? readyCount : minimumReady;
        }
        return minimumReady;
    }

    AICORE inline volatile __gm__ int32_t* LocalUnpermuteAllReadySlot(uint32_t workerIdx) const
    {
        if (workerIdx >= UNPERMUTE_ALL_READY_SLOT_COUNT) {
            return nullptr;
        }
        return LocalSignalBase() + UNPERMUTE_ALL_READY_BASE_INDEX + workerIdx * UNPERMUTE_ALL_READY_STRIDE;
    }

    AICORE inline void PublishUnpermuteAllReady(uint32_t workerCount, int32_t epoch) const
    {
        const uint32_t count =
            workerCount < UNPERMUTE_ALL_READY_SLOT_COUNT ? workerCount : UNPERMUTE_ALL_READY_SLOT_COUNT;
        PublishEpochRangeMte(const_cast<__gm__ int32_t*>(LocalUnpermuteAllReadySlot(0)), count, epoch);
    }

    AICORE inline volatile __gm__ int32_t* LocalUnpermuteStartSlot(uint32_t workerIdx) const
    {
        if (workerIdx >= UNPERMUTE_START_SLOT_COUNT) {
            return nullptr;
        }
        return LocalSignalBase() + UNPERMUTE_START_BASE_INDEX + workerIdx * UNPERMUTE_START_STRIDE;
    }

    AICORE inline void PublishUnpermuteStart(uint32_t workerCount, int32_t epoch) const
    {
        const uint32_t count = workerCount < UNPERMUTE_START_SLOT_COUNT ? workerCount : UNPERMUTE_START_SLOT_COUNT;
        for (uint32_t workerIdx = 0U; workerIdx < count; ++workerIdx) {
            *LocalUnpermuteStartSlot(workerIdx) = epoch;
        }
        for (uint32_t workerIdx = 0U; workerIdx < count; ++workerIdx) {
            dcci((__gm__ void*)LocalUnpermuteStartSlot(workerIdx), SINGLE_CACHE_LINE);
        }
        dsb(DSB_DDR);
        __asm__ __volatile__("");
    }

    AICORE inline void PublishUnpermuteStartRangeMte(uint32_t firstWorker, uint32_t workerCount, int32_t epoch) const
    {
        if (firstWorker >= UNPERMUTE_START_SLOT_COUNT) {
            return;
        }
        const uint32_t remaining = UNPERMUTE_START_SLOT_COUNT - firstWorker;
        const uint32_t count = workerCount < remaining ? workerCount : remaining;
        PublishEpochRangeMte(const_cast<__gm__ int32_t*>(LocalUnpermuteStartSlot(firstWorker)), count, epoch);
    }

    AICORE inline volatile __gm__ int32_t* LocalUnpermutePhase1ProgressEpochSlot() const
    {
        return LocalSignalBase() + UNPERMUTE_PHASE1_PROGRESS_BASE_INDEX + UNPERMUTE_PHASE1_PROGRESS_EPOCH_OFFSET;
    }

    AICORE inline void PublishUnpermutePhase1Progress(
        const uint32_t* readyExpertCounts, uint32_t rankCount, uint32_t readyRankMask, int32_t epoch) const
    {
        if (readyExpertCounts == nullptr || rankCount == 0U || rankCount > COMBINE_EXPERT_PROGRESS_MAX_RANKS) {
            return;
        }
        volatile __gm__ int32_t* base = LocalSignalBase() + UNPERMUTE_PHASE1_PROGRESS_BASE_INDEX;
        base[UNPERMUTE_PHASE1_PROGRESS_MASK_OFFSET] = static_cast<int32_t>(readyRankMask);
        for (uint32_t producerRank = 0U; producerRank < rankCount; ++producerRank) {
            base[UNPERMUTE_PHASE1_PROGRESS_COUNTS_OFFSET + producerRank] =
                static_cast<int32_t>(readyExpertCounts[producerRank]);
        }
        DcciUnpermutePhase1Progress(base, rankCount);
        dsb(DSB_DDR);
        base[UNPERMUTE_PHASE1_PROGRESS_EPOCH_OFFSET] = epoch;
        dcci((__gm__ void*)base, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline uint32_t ReadUnpermutePhase1Progress(uint32_t* readyExpertCounts, uint32_t rankCount) const
    {
        if (readyExpertCounts == nullptr || rankCount == 0U || rankCount > COMBINE_EXPERT_PROGRESS_MAX_RANKS) {
            return 0U;
        }
        volatile __gm__ int32_t* base = LocalSignalBase() + UNPERMUTE_PHASE1_PROGRESS_BASE_INDEX;
        DcciUnpermutePhase1Progress(base, rankCount);
        dsb(DSB_DDR);
        for (uint32_t producerRank = 0U; producerRank < rankCount; ++producerRank) {
            readyExpertCounts[producerRank] =
                static_cast<uint32_t>(base[UNPERMUTE_PHASE1_PROGRESS_COUNTS_OFFSET + producerRank]);
        }
        return static_cast<uint32_t>(base[UNPERMUTE_PHASE1_PROGRESS_MASK_OFFSET]);
    }

    AICORE inline volatile __gm__ int32_t* LocalDispatchReleaseSlot(uint32_t workerIdx) const
    {
        if (workerIdx >= UNPERMUTE_DISPATCH_RELEASE_SLOT_COUNT) {
            return nullptr;
        }
        return LocalSignalBase() + UNPERMUTE_DISPATCH_RELEASE_BASE_INDEX +
               workerIdx * UNPERMUTE_DISPATCH_RELEASE_STRIDE;
    }

    AICORE inline volatile __gm__ int32_t* LocalSwigluReleaseSlot(uint32_t workerIdx) const
    {
        if (workerIdx >= UNPERMUTE_SWIGLU_RELEASE_SLOT_COUNT) {
            return nullptr;
        }
        return LocalSignalBase() + UNPERMUTE_SWIGLU_RELEASE_BASE_INDEX + workerIdx * UNPERMUTE_SWIGLU_RELEASE_STRIDE;
    }

    AICORE inline void PublishDispatchRelease(uint32_t workerIdx, int32_t epoch) const
    {
        volatile __gm__ int32_t* slot = LocalDispatchReleaseSlot(workerIdx);
        if (slot == nullptr) {
            return;
        }
        *slot = epoch;
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline void PublishSwigluRelease(uint32_t workerIdx, int32_t epoch) const
    {
        volatile __gm__ int32_t* slot = LocalSwigluReleaseSlot(workerIdx);
        if (slot == nullptr) {
            return;
        }
        *slot = epoch;
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline void WaitDispatchReleaseMte(uint32_t workerCount, int32_t epoch) const
    {
        const uint32_t count =
            workerCount < UNPERMUTE_DISPATCH_RELEASE_SLOT_COUNT ? workerCount : UNPERMUTE_DISPATCH_RELEASE_SLOT_COUNT;
        const uint32_t allReadyMask = count >= 32U ? 0xFFFFFFFFU : ((1U << count) - 1U);
        __gm__ int32_t* base = const_cast<__gm__ int32_t*>(LocalDispatchReleaseSlot(0U));
        while (ReadEpochMaskMte(base, count, epoch) != allReadyMask) {
            RemoteWindowSyncPollBackoff();
        }
        AcquireDataReady();
    }

    AICORE inline void WaitSwigluReleaseMte(uint32_t workerCount, int32_t epoch) const
    {
        const uint32_t count =
            workerCount < UNPERMUTE_SWIGLU_RELEASE_SLOT_COUNT ? workerCount : UNPERMUTE_SWIGLU_RELEASE_SLOT_COUNT;
        const uint32_t allReadyMask = count >= 32U ? 0xFFFFFFFFU : ((1U << count) - 1U);
        __gm__ int32_t* base = const_cast<__gm__ int32_t*>(LocalSwigluReleaseSlot(0U));
        while (ReadEpochMaskMte(base, count, epoch) != allReadyMask) {
            RemoteWindowSyncPollBackoff();
        }
        AcquireDataReady();
    }

    AICORE inline volatile __gm__ int32_t* LocalPhase1DoneSlot(uint32_t workerIdx) const
    {
        if (workerIdx >= UNPERMUTE_PHASE1_DONE_SLOT_COUNT) {
            return nullptr;
        }
        return LocalSignalBase() + UNPERMUTE_PHASE1_DONE_BASE_INDEX + workerIdx * UNPERMUTE_PHASE1_DONE_STRIDE;
    }

    AICORE inline void PublishPhase1Done(uint32_t workerIdx, int32_t epoch) const
    {
        volatile __gm__ int32_t* slot = LocalPhase1DoneSlot(workerIdx);
        if (slot == nullptr) {
            return;
        }
        *slot = epoch;
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline void WaitPhase1DoneMte(uint32_t workerCount, int32_t epoch) const
    {
        const uint32_t count =
            workerCount < UNPERMUTE_PHASE1_DONE_SLOT_COUNT ? workerCount : UNPERMUTE_PHASE1_DONE_SLOT_COUNT;
        const uint32_t allReadyMask = count == 32U ? 0xFFFFFFFFU : ((1U << count) - 1U);
        __gm__ int32_t* base = const_cast<__gm__ int32_t*>(LocalPhase1DoneSlot(0U));
        while (ReadEpochMaskMte(base, count, epoch) != allReadyMask) {
            RemoteWindowSyncPollBackoff();
        }
        AcquireDataReady();
    }

    AICORE inline volatile __gm__ int32_t* LocalCombineDoneSlot(uint32_t laneIdx) const
    {
        if (laneIdx >= COMBINE_LOCAL_DONE_SLOT_COUNT) {
            return nullptr;
        }
        return LocalSignalBase() + COMBINE_LOCAL_DONE_BASE_INDEX + laneIdx * COMBINE_LOCAL_DONE_STRIDE;
    }

    AICORE inline void PublishLocalCombineDone(uint32_t laneIdx, int32_t epoch) const
    {
        volatile __gm__ int32_t* slot = LocalCombineDoneSlot(laneIdx);
        if (slot == nullptr) {
            return;
        }
        *slot = epoch;
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }

    AICORE inline void WaitLocalCombineDoneMte(uint32_t laneCount, int32_t epoch) const
    {
        const uint32_t count = laneCount < COMBINE_LOCAL_DONE_SLOT_COUNT ? laneCount : COMBINE_LOCAL_DONE_SLOT_COUNT;
        const uint32_t allReadyMask = count >= 32U ? 0xFFFFFFFFU : ((1U << count) - 1U);
        __gm__ int32_t* base = const_cast<__gm__ int32_t*>(LocalCombineDoneSlot(0));
        while (ReadEpochMaskMte(base, count, epoch) != allReadyMask) {
            RemoteWindowSyncPollBackoff();
        }
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
    }

    AICORE inline void CrossRankSync() const
    {
        __gm__ int32_t* localSignalBase = LocalSignalBase();
        __gm__ int32_t* syncBase = localSignalBase + COMBINE_BARRIER_EPOCH_INDEX;
        const int32_t count = *syncBase + 1;
        int32_t vecId = static_cast<int32_t>(get_block_idx());
        int32_t vecSize = static_cast<int32_t>(get_block_num());
        if ASCEND_IS_AIV {
            vecId += static_cast<int32_t>(get_subblockid()) * static_cast<int32_t>(get_block_num());
            vecSize *= static_cast<int32_t>(get_subblockdim());
        }
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        for (int32_t i = vecId; i < rankSize_; i += vecSize) {
            __gm__ int32_t* remoteSignalBase = RemoteSignalBase(i);
            auto remoteBarrier = pto::comm::Signal(
                remoteSignalBase + COMBINE_BARRIER_COUNTER_BASE_INDEX + rank_ * COMBINE_BARRIER_COUNTER_STRIDE);
            auto localBarrier = pto::comm::Signal(
                localSignalBase + COMBINE_BARRIER_COUNTER_BASE_INDEX + i * COMBINE_BARRIER_COUNTER_STRIDE);
            pto::comm::TNOTIFY(remoteBarrier, 1, pto::comm::NotifyOp::AtomicAdd);
            pto::comm::TWAIT(localBarrier, count, pto::comm::WaitCmp::GE);
        }

        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        *syncBase = count;
    }

    AICORE inline void CrossRankStartSyncAiv() const
    {
        if ASCEND_IS_AIV {
            const int32_t coreId = static_cast<int32_t>(get_block_idx()) +
                                   static_cast<int32_t>(get_subblockid()) * static_cast<int32_t>(get_block_num());
            const int32_t coreNum = static_cast<int32_t>(get_block_num()) * static_cast<int32_t>(get_subblockdim());
            const int32_t count = CrossRankSyncSignals(
                START_AIV_BARRIER_COUNTER_BASE_INDEX, START_AIV_BARRIER_EPOCH_INDEX, coreId, coreNum);
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
            PublishCrossRankSyncEpoch(START_AIV_BARRIER_EPOCH_INDEX, count);
        }
    }

    AICORE inline void CrossRankStartSyncAic() const
    {
        if ASCEND_IS_AIC {
            const int32_t coreId = static_cast<int32_t>(get_block_idx());
            const int32_t coreNum = static_cast<int32_t>(get_block_num());
            const int32_t count = CrossRankSyncSignals(
                START_AIC_BARRIER_COUNTER_BASE_INDEX, START_AIC_BARRIER_EPOCH_INDEX, coreId, coreNum);
            pto::SYNCALL<pto::SyncCoreType::AICOnly>();
            PublishCrossRankSyncEpoch(START_AIC_BARRIER_EPOCH_INDEX, count);
        }
    }

private:
    AICORE inline int32_t CrossRankSyncSignals(
        uint32_t counterBaseIndex, uint32_t epochIndex, int32_t coreId, int32_t coreNum) const
    {
        __gm__ int32_t* localSignalBase = LocalSignalBase();
        __gm__ int32_t* syncBase = localSignalBase + epochIndex;
        const int32_t count = *syncBase + 1;
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        for (int32_t i = coreId; i < rankSize_; i += coreNum) {
            __gm__ int32_t* remoteSignalBase = RemoteSignalBase(i);
            auto remoteBarrier =
                pto::comm::Signal(remoteSignalBase + counterBaseIndex + rank_ * COMBINE_BARRIER_COUNTER_STRIDE);
            auto localBarrier =
                pto::comm::Signal(localSignalBase + counterBaseIndex + i * COMBINE_BARRIER_COUNTER_STRIDE);
            pto::comm::TNOTIFY(remoteBarrier, 1, pto::comm::NotifyOp::AtomicAdd);
            pto::comm::TWAIT(localBarrier, count, pto::comm::WaitCmp::GE);
        }
        return count;
    }

    AICORE inline void PublishCrossRankSyncEpoch(uint32_t epochIndex, int32_t count) const
    {
        *(LocalSignalBase() + epochIndex) = count;
    }

    __gm__ PtoRemoteWindowContext* context_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    size_t segmentBytes_ = 0;
};

struct MegaMoePeerMemoryLayout {
    int64_t offsetA = 0;
    int64_t offsetPeerPerTokenScale = 0;
    int64_t offsetPeerTokenPerExpert = 0;
    int64_t offsetD = 0;

    AICORE inline void Init(const PtoRemoteWindow& remoteWindow)
    {
        constexpr int64_t alignBytes = 512;
        const int64_t segmentSize = static_cast<int64_t>(remoteWindow.SegmentSize());
        offsetA = 0;
        offsetPeerPerTokenScale = offsetA + ((segmentSize / 3 + alignBytes - 1) / alignBytes * alignBytes);
        offsetD = offsetPeerPerTokenScale + MB_SIZE;
        offsetPeerTokenPerExpert = segmentSize - 2 * MB_SIZE;
    }
};

#endif // HCCL_WINDOW_HPP
