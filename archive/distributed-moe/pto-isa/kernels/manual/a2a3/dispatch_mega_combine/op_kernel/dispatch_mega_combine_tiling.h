/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file dispatch_mega_combine_tiling.h
 * \brief
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

constexpr uint32_t kFrontCaseFullLoadDynamic = 21000U;
constexpr uint32_t kFrontCaseOneCoreDynamic = 11000U;
constexpr uint32_t kFrontCaseMultiCoreDynamic = 11010U;

inline constexpr bool FrontCaseIsSupported(uint32_t frontCase)
{
    return frontCase == kFrontCaseFullLoadDynamic || frontCase == kFrontCaseOneCoreDynamic ||
           frontCase == kFrontCaseMultiCoreDynamic;
}

struct MegaMoeInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t maxOutputSize;
    uint32_t aivNum;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t reservedInfo[2] = {0, 0};
};

struct MegaMoeRuntimeInfo {
    uint64_t remoteWindowContext = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
};

// Keep uint16_t control fields paired or explicitly padded before uint32_t offset/size fields.
struct MegaMoeFrontReorderTiling {
    uint32_t routeElems = 0;
    uint16_t frontCase = 0;
    uint16_t stageNum = 0;
    uint32_t expertNum = 0;
    uint32_t expertNumAligned = 0;
    uint32_t sortLoopMaxElement = 0;
    uint32_t sortLastCorePerLoopElems = 0;
    uint32_t expandedRowIdxOffset = 0;
    uint32_t localTokenPerExpertOffset = 0;
    uint32_t cumsumMMOffset = 0;
    uint32_t preSumBeforeRankOffset = 0;
    uint32_t reservedFrontCountPad = 0;
    uint32_t alignedRouteElems = 0;
    uint32_t sortPerCoreElems = 0;
    uint32_t sortLastCoreElems = 0;
    uint32_t sortPerCoreLoops = 0;
    uint32_t sortPerCorePerLoopElems = 0;
    uint32_t sortPerCoreLastLoopElems = 0;
    uint32_t sortLastCoreLoops = 0;
    uint32_t sortLastCoreLastLoopElems = 0;
    uint16_t sortNeedCoreNum = 0;
    uint16_t sortOutLoopMaxElems = 0;
    uint16_t reservedFrontSortPad = 0;
    uint32_t frontExpandedExpertOffset = 0;
    uint32_t frontExpandDstToSrcOffset = 0;
    uint32_t frontSortWs0Offset = 0;
    uint32_t frontSortWs1Offset = 0;
    uint32_t reservedFrontQuantTmpOffset = 0;
    uint32_t reservedFrontQuantTmpBytes = 0;
    uint32_t reservedCoreCountOffset = 0;
    uint32_t reservedExpertBaseOffset = 0;
    uint32_t reservedCoreBaseOffset = 0;
    uint32_t frontWorkspaceBytes = 0;
    uint32_t reservedFrontCountScratchOffset = 0;
};

struct MegaMoeDispatchTiling {
    uint64_t gmAOffset = 0;
    uint64_t perTokenScaleOffset = 0;
    uint64_t reservedDispatchScratchOffset = 0;
    uint64_t reservedDispatchScratchBytes = 0;
    uint64_t reservedDispatchScratchBytesPerAiv = 0;
    uint64_t dispatchTileBytes = 0;
    uint32_t reservedDispatchGatherMode = 0;
    uint32_t reservedDispatchPad = 0;
    uint64_t dispatchGatherScratchOffset = 0;
    uint64_t dispatchGatherScratchBytes = 0;
    uint64_t dispatchGatherScratchBytesPerAiv = 0;
    uint64_t dispatchGatherTileBytes = 0;
};

struct MegaMoeGmm1Tiling {
    uint64_t gmCOffset = 0;
    uint64_t reserved0 = 0;
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 256;
    uint32_t l1TileK = 512;
    uint32_t l0TileM = 128;
    uint32_t l0TileN = 256;
    uint32_t l0TileK = 128;
    uint32_t reserved1 = 0;
};

struct MegaMoeSwigluTiling {
    uint64_t gmPermutedTokenOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint32_t reserved[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct MegaMoeGmm2Tiling {
    uint64_t gmm2OutputOffset = 0;
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 256;
    uint32_t l1TileK = 512;
    uint32_t l0TileM = 128;
    uint32_t l0TileN = 256;
    uint32_t l0TileK = 128;
    uint32_t reserved0[5] = {0, 0, 0, 0, 0};
};

struct MegaMoeCombineTiling {
    uint64_t gmm2OutputOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint64_t reservedCombineScratchOffset = 0;
    uint64_t reservedCombineScratchBytes = 0;
    uint64_t reservedCombineScratchBytesPerAiv = 0;
    uint32_t reservedCombineTileCols = 0;
    uint32_t reserved0 = 0;
};

struct MegaMoeUnpermuteTiling {
    uint32_t unpermuteTileCols = 1024;
    uint32_t unpermuteTokenBatch = 256;
    uint32_t unpermuteImplMode = 0;
    uint32_t reserved0[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

constexpr uint32_t kMegaMoeUnpermuteImplBarrier = 0U;
constexpr uint32_t kMegaMoeUnpermuteImplRankStreaming = 1U;
constexpr uint32_t kMegaMoeFixedLayoutVersion = 11U;
constexpr uint32_t kMegaMoeFixedPhysicalAicNum = 24U;
constexpr uint32_t kMegaMoeFixedPhysicalAivNum = 48U;
constexpr uint32_t kMegaMoeExpertProgressMaxRanks = 16U;
constexpr uint32_t kMegaMoeRankStreamingMaxTokensPerWorker = 256U;
constexpr uint32_t kMegaMoeFixedMaxExperts = 32U;

constexpr uint32_t kMegaMoeFixedSyncSlotBytes = 64U;
constexpr uint32_t kMegaMoeFixedSyncHeadCanarySlot = 0U;
constexpr uint32_t kMegaMoeFixedSyncBytes = 8704U;
constexpr uint32_t kMegaMoeFixedHeadCanary = 0x13579BDFU;
constexpr uint32_t kMegaMoeFixedTailCanary = 0x2468ACE0U;
constexpr int32_t kMegaMoeFixedGmm1DoneMarker = 1;
constexpr int32_t kMegaMoeFixedGmm2JoinDecisionBit = 0x40000000;
constexpr int32_t kMegaMoeFixedGmm2JoinDecisionMask = kMegaMoeFixedGmm2JoinDecisionBit - 1;

enum MegaMoeFixedRole : uint32_t {
    kMegaMoeFixedRoleGmm1 = 1U,
    kMegaMoeFixedRoleGmm2 = 2U,
    kMegaMoeFixedRoleSwiglu = 3U,
    kMegaMoeFixedRoleCombine = 4U,
    kMegaMoeFixedRoleDispatch = 5U,
};

struct MegaMoeFixedGroupTiling {
    uint64_t syncOffset = 0;
    uint64_t syncBytes = 0;
    uint32_t layoutVersion = kMegaMoeFixedLayoutVersion;
    uint32_t physicalAicNum = 0;
    uint32_t physicalAivNum = 0;
    uint32_t dispatchGroupSize = 16U;
    uint32_t gmm1GroupSize = 16U;
    uint32_t gmm2GroupSize = 8U;
    uint32_t swigluGroupSize = 16U;
    uint32_t swigluActiveGroupSize = 16U;
    uint32_t combineGroupSize = 8U;
    uint32_t shapeConfigM = 0;
    uint32_t fullAicGmm1ExpertCount = 2U;
    uint32_t unpermutePhase1ReadyExpertCount = 13U;
    uint32_t gmm2JoinCheckStartExpert = 13U;
    uint32_t combineStartAfterGmm2Expert = 2U;
    uint32_t combineLargeLanesPerRank = 1U;
    uint32_t reserved0 = 0;
    uint32_t syncSlotBytes = kMegaMoeFixedSyncSlotBytes;
    uint32_t maxExperts = kMegaMoeFixedMaxExperts;
};

struct MegaMoeSyncLayout {
    uint32_t dispatchArrivalBase = 0;
    uint32_t dispatchReadyBase = 0;
    uint32_t gmm1ArrivalBase = 0;
    uint32_t swigluReadyBase = 0;
    uint32_t swigluArrivalBase = 0;
    uint32_t gmm2ReadyBase = 0;
    uint32_t gmm2ArrivalBase = 0;
    uint32_t combineReadyBase = 0;
    uint32_t gmm1DoneSlot = 0;
    uint32_t gmm2JoinSlot = 0;
    uint32_t tailCanarySlot = 0;
    uint32_t slotCount = 0;
};

template <typename FixedT>
inline MegaMoeSyncLayout MakeMegaMoeSyncLayout(const FixedT& fixed)
{
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

static_assert(sizeof(MegaMoeSwigluTiling) == 64);
static_assert(sizeof(MegaMoeGmm2Tiling) == 56);
static_assert(sizeof(MegaMoeCombineTiling) == 48);
static_assert(sizeof(MegaMoeUnpermuteTiling) == 56);
static_assert(sizeof(MegaMoeFixedGroupTiling) == 88);

struct MegaMoeTilingData {
    MegaMoeInfo megaMoeInfo;
    MegaMoeRuntimeInfo runtimeInfo;
    MegaMoeFrontReorderTiling frontReorderTiling;
    MegaMoeDispatchTiling dispatchTiling;
    MegaMoeGmm1Tiling gmm1Tiling;
    MegaMoeSwigluTiling swigluTiling;
    MegaMoeGmm2Tiling gmm2Tiling;
    MegaMoeCombineTiling combineTiling;
    MegaMoeUnpermuteTiling unpermuteTiling;
    MegaMoeFixedGroupTiling fixedGroupTiling;
};
