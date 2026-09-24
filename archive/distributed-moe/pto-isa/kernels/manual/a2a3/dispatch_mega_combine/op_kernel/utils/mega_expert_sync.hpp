/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP
#define DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "const_args.hpp"

struct MegaMoeFixedCoreRoleInfo {
    uint32_t role = 0U;
    uint32_t physicalBlockId = 0U;
    uint32_t subblockId = 0U;
    uint32_t flatAivId = 0U;
    uint32_t groupLocalId = 0U;
    uint32_t groupSize = 0U;
};

AICORE inline MegaMoeFixedCoreRoleInfo FixedCoreRole(const __gm__ MegaMoeTilingData* tilingData)
{
    MegaMoeFixedCoreRoleInfo info;
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    info.physicalBlockId = get_block_idx();
    const bool firstGroup = info.physicalBlockId < fixed.gmm1GroupSize;
    if ASCEND_IS_AIC {
        if (firstGroup) {
            info.role = kMegaMoeFixedRoleGmm1;
            info.groupLocalId = info.physicalBlockId;
            info.groupSize = fixed.gmm1GroupSize;
        } else {
            info.role = kMegaMoeFixedRoleGmm2;
            info.groupLocalId = info.physicalBlockId - fixed.gmm1GroupSize;
            info.groupSize = fixed.gmm2GroupSize;
        }
        return info;
    }

    info.subblockId = get_subblockid();
    info.flatAivId = info.physicalBlockId + info.subblockId * fixed.physicalAicNum;
    if (firstGroup) {
        info.groupLocalId = info.physicalBlockId;
        if (info.subblockId == 0U) {
            info.role = kMegaMoeFixedRoleDispatch;
            info.groupSize = fixed.dispatchGroupSize;
        } else {
            info.role = kMegaMoeFixedRoleSwiglu;
            info.groupSize = fixed.swigluGroupSize;
        }
    } else {
        info.role = kMegaMoeFixedRoleCombine;
        const uint32_t pairedGmm2LocalId = info.physicalBlockId - fixed.gmm1GroupSize;
        info.groupLocalId = pairedGmm2LocalId + info.subblockId * fixed.gmm2GroupSize;
        info.groupSize = fixed.combineGroupSize;
    }
    return info;
}

AICORE inline bool FixedRoleActiveForStage(const MegaMoeFixedCoreRoleInfo& info, uint32_t stageNum)
{
    return (info.role == kMegaMoeFixedRoleDispatch && stageNum >= 9U) ||
           (info.role == kMegaMoeFixedRoleGmm1 && stageNum >= 10U) ||
           (info.role == kMegaMoeFixedRoleSwiglu && stageNum >= 11U) ||
           (info.role == kMegaMoeFixedRoleGmm2 && stageNum >= 12U) ||
           (info.role == kMegaMoeFixedRoleCombine && info.groupLocalId < info.groupSize && stageNum >= 13U);
}

AICORE inline MegaMoeSyncLayout FixedSyncLayout(const __gm__ MegaMoeTilingData* tilingData)
{
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    MegaMoeSyncLayout layout;
    layout.dispatchArrivalBase = kMegaMoeFixedSyncHeadCanarySlot + 1U;
    layout.dispatchReadyBase = layout.dispatchArrivalBase + fixed.dispatchGroupSize;
    layout.gmm1ArrivalBase = layout.dispatchReadyBase + fixed.physicalAicNum;
    layout.swigluReadyBase = layout.gmm1ArrivalBase + fixed.gmm1GroupSize;
    layout.swigluArrivalBase = layout.swigluReadyBase + fixed.swigluGroupSize;
    layout.gmm2ReadyBase = layout.swigluArrivalBase + fixed.swigluGroupSize;
    layout.gmm2ArrivalBase = layout.gmm2ReadyBase + fixed.gmm2GroupSize;
    layout.combineReadyBase = layout.gmm2ArrivalBase + fixed.physicalAicNum;
    layout.gmm1DoneSlot = layout.combineReadyBase + fixed.gmm2GroupSize;
    layout.gmm2JoinSlot = layout.gmm1DoneSlot + 1U;
    layout.tailCanarySlot = layout.gmm2JoinSlot + 1U;
    layout.slotCount = layout.tailCanarySlot + 1U;
    return layout;
}

AICORE inline volatile __gm__ int32_t* FixedSyncSlot(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t slot)
{
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    return reinterpret_cast<volatile __gm__ int32_t*>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(slot) * fixed.syncSlotBytes);
}

// The doorbell is a scalar GM store. The caller drains payload pipelines before publishing.
AICORE inline void PublishScalarEpoch(volatile __gm__ int32_t* slot, int32_t epoch)
{
    *slot = epoch;
    dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}

AICORE inline void PublishScalarEpochRange(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t baseSlot, uint32_t count, int32_t epoch)
{
    for (uint32_t localId = 0U; localId < count; ++localId) {
        *FixedSyncSlot(workspaceGM, tilingData, baseSlot + localId) = epoch;
    }
    for (uint32_t localId = 0U; localId < count; ++localId) {
        dcci((__gm__ void*)FixedSyncSlot(workspaceGM, tilingData, baseSlot + localId), SINGLE_CACHE_LINE);
    }
    __asm__ __volatile__("");
}

AICORE inline void ResetFixedSyncWorkspace(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    const MegaMoeSyncLayout sync = FixedSyncLayout(tilingData);
    for (uint32_t slotId = 0U; slotId < sync.slotCount; ++slotId) {
        volatile __gm__ int32_t* slot = FixedSyncSlot(workspaceGM, tilingData, slotId);
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        uint32_t value = 0U;
        if (slotId == kMegaMoeFixedSyncHeadCanarySlot) {
            value = kMegaMoeFixedHeadCanary;
        } else if (slotId == sync.tailCanarySlot) {
            value = kMegaMoeFixedTailCanary;
        }
        *slot = static_cast<int32_t>(value);
    }
    for (uint32_t slotId = 0U; slotId < sync.slotCount; ++slotId) {
        dcci((__gm__ void*)FixedSyncSlot(workspaceGM, tilingData, slotId), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}

AICORE inline int32_t ReadScalarEpoch(volatile __gm__ int32_t* slot)
{
    dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    return *slot;
}

AICORE inline bool TestEpoch(volatile __gm__ int32_t* slot, int32_t epoch, int32_t& observed)
{
    observed = ReadScalarEpoch(slot);
    return observed >= epoch;
}

AICORE inline void EpochPollBackoff()
{
    constexpr uint32_t kDelayTicks = 3U;
    const uint64_t deadline = get_sys_cnt() + kDelayTicks;
    while (get_sys_cnt() < deadline) {
        __asm__ __volatile__("");
    }
}

AICORE inline int32_t WaitEpochRaw(volatile __gm__ int32_t* slot, int32_t epoch)
{
    while (true) {
        int32_t observed = 0;
        if (TestEpoch(slot, epoch, observed)) {
            return observed;
        }
        EpochPollBackoff();
    }
}

AICORE inline int32_t WaitEpochAcquire(volatile __gm__ int32_t* slot, int32_t epoch)
{
    const int32_t observed = WaitEpochRaw(slot, epoch);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return observed;
}

// These snapshot helpers require AIV UB and MTE2/MTE3; AIC callers publish scalar arrivals only.
constexpr uint32_t kMegaMoeSyncSnapshotValuesPerSlot = kMegaMoeFixedSyncSlotBytes / sizeof(int32_t);
constexpr uint32_t kMegaMoeSyncSnapshotMaxValues = kMegaMoeFixedPhysicalAicNum * kMegaMoeSyncSnapshotValuesPerSlot;
constexpr uint64_t kMegaMoeSyncSnapshotUbBytes = static_cast<uint64_t>(kMegaMoeSyncSnapshotMaxValues) * sizeof(int32_t);
constexpr uint64_t kMegaMoeSyncSnapshotUbOffset = AtlasA2::UB_SIZE - kMegaMoeSyncSnapshotUbBytes;
constexpr event_t kMegaMoeSyncSnapshotEvent = EVENT_ID0;

static_assert(kMegaMoeFixedSyncSlotBytes % sizeof(int32_t) == 0U);
static_assert(kMegaMoeSyncSnapshotUbOffset % UB_ALIGN == 0U);

AICORE inline int32_t ReadArrivalMinMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t producerCount)
{
    if (producerCount == 0U || producerCount > kMegaMoeFixedPhysicalAicNum) {
        return 0;
    }

    using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
    using SnapshotTile =
        pto::Tile<pto::TileType::Vec, int32_t, 1, kMegaMoeSyncSnapshotMaxValues, pto::BLayout::RowMajor, -1, -1>;

    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    __gm__ int32_t* arrivalBase = reinterpret_cast<__gm__ int32_t*>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(arrivalBaseSlot) * fixed.syncSlotBytes);
    const uint32_t snapshotValues = producerCount * kMegaMoeSyncSnapshotValuesPerSlot;
    SnapshotShape snapshotShape(snapshotValues);
    SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
    SnapshotGlobal snapshotGlobal(arrivalBase, snapshotShape, snapshotStride);
    SnapshotTile snapshotTile(1U, snapshotValues);
    pto::TASSIGN(snapshotTile, kMegaMoeSyncSnapshotUbOffset);
    pto::TLOAD(snapshotTile, snapshotGlobal);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);

    int32_t observedMin = snapshotTile.GetValue(0U);
    for (uint32_t peer = 1U; peer < producerCount; ++peer) {
        const int32_t observed = snapshotTile.GetValue(peer * kMegaMoeSyncSnapshotValuesPerSlot);
        observedMin = observed < observedMin ? observed : observedMin;
    }
    return observedMin;
}

AICORE inline int32_t WaitArrivalMinMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t producerCount,
    int32_t epoch)
{
    while (true) {
        const int32_t observedMin = ReadArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount);
        if (observedMin >= epoch) {
            return observedMin;
        }
        EpochPollBackoff();
    }
}

AICORE inline void PublishEpochRangeMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t baseSlot, uint32_t count, int32_t epoch)
{
    if (count == 0U || count > kMegaMoeFixedPhysicalAicNum) {
        return;
    }

    using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
    using SnapshotTile =
        pto::Tile<pto::TileType::Vec, int32_t, 1, kMegaMoeSyncSnapshotMaxValues, pto::BLayout::RowMajor, -1, -1>;

    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    __gm__ int32_t* readyBase = reinterpret_cast<__gm__ int32_t*>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(baseSlot) * fixed.syncSlotBytes);
    const uint32_t snapshotValues = count * kMegaMoeSyncSnapshotValuesPerSlot;
    SnapshotShape snapshotShape(snapshotValues);
    SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
    SnapshotGlobal snapshotGlobal(readyBase, snapshotShape, snapshotStride);
    SnapshotTile snapshotTile(1U, snapshotValues);
    pto::TASSIGN(snapshotTile, kMegaMoeSyncSnapshotUbOffset);
    for (uint32_t localId = 0U; localId < count; ++localId) {
        snapshotTile.SetValue(localId * kMegaMoeSyncSnapshotValuesPerSlot, epoch);
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);
    pto::TSTORE(snapshotGlobal, snapshotTile);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);
}

AICORE inline void PublishGroupArrival(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t localId,
    uint32_t notifyCall)
{
    const int32_t epoch = static_cast<int32_t>(notifyCall * 2U + 1U);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    PublishScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, arrivalBaseSlot + localId), epoch);
}

AICORE inline void CoordinateGroupConsumersMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t readyBaseSlot,
    uint32_t producerCount, uint32_t consumerCount, uint32_t notifyCall)
{
    const int32_t arriveEpoch = static_cast<int32_t>(notifyCall * 2U + 1U);
    const int32_t releaseEpoch = arriveEpoch + 1;
    WaitArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount, arriveEpoch);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    PublishEpochRangeMte(workspaceGM, tilingData, readyBaseSlot, consumerCount, releaseEpoch);
}

AICORE inline void NotifyGroupConsumersMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t readyBaseSlot,
    uint32_t producerCount, uint32_t consumerCount, uint32_t localId, uint32_t coordinatorLocalId, uint32_t notifyCall)
{
    const int32_t arriveEpoch = static_cast<int32_t>(notifyCall * 2U + 1U);
    const int32_t releaseEpoch = arriveEpoch + 1;
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    PublishScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, arrivalBaseSlot + localId), arriveEpoch);

    if (localId == coordinatorLocalId) {
        WaitArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount, arriveEpoch);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PublishEpochRangeMte(workspaceGM, tilingData, readyBaseSlot, consumerCount, releaseEpoch);
    }
}

#endif // DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP
