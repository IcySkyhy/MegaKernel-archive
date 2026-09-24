/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "machine/utils/dynamic/workspace_budget_calculator.h"

#include "interface/utils/common.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/error_code.h"

namespace {
constexpr uint64_t kAlignment32K = 32 * 1024;

static uint64_t MinU64(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }

static uint64_t MaxU64(uint64_t lhs, uint64_t rhs) { return lhs > rhs ? lhs : rhs; }
} // namespace

namespace npu::tile_fwk {
namespace dynamic {

static bool Uint64MulWouldOverflow(uint64_t lhs, uint64_t rhs) { return lhs != 0 && rhs > UINT64_MAX / lhs; }

// stitchDepthK counts root functions (pre-unroll_list semantics); unrollTimes does not scale tensor budget.
static uint64_t RootFunctionBudgetAtStitchDepth(uint64_t rawMem, uint32_t stitchDepthK)
{
    if (rawMem == 0) {
        return 0;
    }
    const uint64_t alignedRaw = AlignUp(rawMem, TENSOR_ADDR_ALIGNMENT);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, !Uint64MulWouldOverflow(alignedRaw, stitchDepthK))
        << "aligned root budget * stitchDepthK would overflow";
    return alignedRaw * stitchDepthK;
}

static bool RootBudgetAtStitchDepthWouldOverflow(const WorkspaceDesc& desc, uint32_t stitchDepthK)
{
    for (const auto& root : desc.rootFuncDescList) {
        const uint64_t alignedInner = AlignUp(root.rootInnerSpilledRawMem, TENSOR_ADDR_ALIGNMENT);
        const uint64_t alignedExclusive = AlignUp(root.rootTotalExclusiveOutcastRawMem, TENSOR_ADDR_ALIGNMENT);
        if (Uint64MulWouldOverflow(alignedInner, stitchDepthK) ||
            Uint64MulWouldOverflow(alignedExclusive, stitchDepthK)) {
            return true;
        }
    }
    return false;
}

static bool CanDoubleStitchDepthK(uint64_t k) { return k <= static_cast<uint64_t>(UINT32_MAX) / 2; }

static uint32_t NormalizeTensorStitchDepthK(uint32_t tensorStitchDepthK)
{
    return tensorStitchDepthK == 0 ? 1 : tensorStitchDepthK;
}

static uint32_t DeriveOutcastCacheDepth(uint32_t tensorStitchDepthK)
{
    return NormalizeTensorStitchDepthK(tensorStitchDepthK);
}

static uint32_t DeriveRuntimeOutcastPoolDepth(uint32_t tensorStitchDepthK)
{
    const uint32_t kForPool = std::min(tensorStitchDepthK, static_cast<uint32_t>(MAX_STITCH_FUNC_NUM));
    return NormalizeTensorStitchDepthK(kForPool);
}

uint64_t EstimateCtrlFlowCacheSlottedBlockCount(uint64_t totalSlot, uint32_t outcastCacheDepth)
{
    return totalSlot * (static_cast<uint64_t>(outcastCacheDepth) + SLOTS_NEED_ALLOC_SIZE);
}

static void BuildTensorWorkspaceFromDescriptor(WorkspaceDesc& desc, uint32_t stitchDepthK)
{
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, stitchDepthK > 0)
        << "Invalid stitchDepthK: " << stitchDepthK << ", must be greater than 0";

    uint64_t maxInner = 0;
    uint64_t maxExclusive = 0;
    for (const auto& root : desc.rootFuncDescList) {
        maxInner = std::max(maxInner, RootFunctionBudgetAtStitchDepth(root.rootInnerSpilledRawMem, stitchDepthK));
        maxExclusive = std::max(maxExclusive,
                                RootFunctionBudgetAtStitchDepth(root.rootTotalExclusiveOutcastRawMem, stitchDepthK));
    }
    desc.maxRootInnerSpilledMem = maxInner;
    desc.maxRootTotalExclusiveOutcastMem = maxExclusive;
    desc.devTaskBoundaryOutcastNum = desc.totalExclusiveOutcastSlot * SLOTS_NEED_ALLOC_SIZE +
                                     desc.totalAssembleOutcastSlot * SLOTS_NEED_ALLOC_SIZE;
    desc.devTaskInnerTemporalOutcastNum = desc.totalAssembleOutcastSlot * stitchDepthK;
}

static uint64_t WorkspaceTotalFromDesc(const WorkspaceDesc& desc, uint32_t parallelism, uint64_t aicoreSpilled,
                                       uint64_t debugTotal)
{
    const uint64_t boundaryAndInnerTemporal = desc.devTaskBoundaryOutcastNum + desc.devTaskInnerTemporalOutcastNum;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
           !Uint64MulWouldOverflow(desc.maxStaticOutcastMem, boundaryAndInnerTemporal))
        << "maxStaticOutcastMem * boundaryAndInnerTemporalOutcastNum would overflow";
    const uint64_t staticOutcast = desc.maxStaticOutcastMem * boundaryAndInnerTemporal;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED,
           desc.maxRootInnerSpilledMem <= UINT64_MAX - desc.maxRootTotalExclusiveOutcastMem)
        << "tensor workspace inner sum would overflow";
    const uint64_t innerSum = desc.maxRootInnerSpilledMem + desc.maxRootTotalExclusiveOutcastMem;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, innerSum <= UINT64_MAX - staticOutcast)
        << "tensor workspace sum would overflow";
    const uint64_t raw = innerSum + staticOutcast;
    const uint64_t alignedRaw = AlignUp(raw, kAlignment32K);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, !Uint64MulWouldOverflow(alignedRaw, parallelism))
        << "aligned workspace * parallelism would overflow";
    const uint64_t alignedParallel = alignedRaw * parallelism;
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, alignedParallel <= UINT64_MAX - aicoreSpilled - debugTotal)
        << "encoded workspace total would overflow";
    return alignedParallel + aicoreSpilled + debugTotal;
}

static uint64_t TensorWorkspaceBytesAtStitchDepth(const WorkspaceDesc& desc, uint32_t stitchDepthK,
                                                  uint32_t parallelism, uint64_t aicoreSpilled, uint64_t debugTotal)
{
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, stitchDepthK > 0) << "Invalid stitchDepthK";
    WorkspaceDesc scratch = desc;
    BuildTensorWorkspaceFromDescriptor(scratch, stitchDepthK);
    return WorkspaceTotalFromDesc(scratch, parallelism, aicoreSpilled, debugTotal);
}

// k_min=1: smallest stitch-depth in the budget model.
uint64_t TensorWorkspaceBytesAtMinimumStitchDepth(const WorkspaceDesc& desc, uint32_t parallelism,
                                                  uint64_t aicoreSpilled, uint64_t debugTotal)
{
    return TensorWorkspaceBytesAtStitchDepth(desc, 1, parallelism, aicoreSpilled, debugTotal);
}

namespace {
struct TensorBudgetModel {
    const WorkspaceDesc& desc;
    uint32_t parallelism;
    uint64_t aicoreSpilled;
    uint64_t debugTotal;
    uint64_t cap;

    uint64_t BytesAt(uint32_t stitchDepthK) const
    {
        if (RootBudgetAtStitchDepthWouldOverflow(desc, stitchDepthK)) {
            return UINT64_MAX;
        }
        WorkspaceDesc scratch = desc;
        BuildTensorWorkspaceFromDescriptor(scratch, stitchDepthK);
        return WorkspaceTotalFromDesc(scratch, parallelism, aicoreSpilled, debugTotal);
    }

    bool Fits(uint32_t stitchDepthK) const { return BytesAt(stitchDepthK) <= cap; }
};

struct StitchDepthSearchBounds {
    uint64_t lo{1};
    uint64_t searchHi{1};
};

// raw(k) = c0 + c1*k before 32KB alignment; c1=0 means stitch depth does not change tensor bytes.
static void ComputeTensorBudgetLinearCoeffs(const WorkspaceDesc& desc, uint64_t& c0, uint64_t& c1)
{
    uint64_t maxAlignedInner = 0;
    uint64_t maxAlignedExclusive = 0;
    for (const auto& root : desc.rootFuncDescList) {
        maxAlignedInner = MaxU64(maxAlignedInner, AlignUp(root.rootInnerSpilledRawMem, TENSOR_ADDR_ALIGNMENT));
        maxAlignedExclusive = MaxU64(maxAlignedExclusive,
                                     AlignUp(root.rootTotalExclusiveOutcastRawMem, TENSOR_ADDR_ALIGNMENT));
    }
    const uint64_t boundaryFixed = desc.totalExclusiveOutcastSlot * SLOTS_NEED_ALLOC_SIZE +
                                   desc.totalAssembleOutcastSlot * SLOTS_NEED_ALLOC_SIZE;
    c0 = desc.maxStaticOutcastMem * boundaryFixed;
    c1 = maxAlignedInner + maxAlignedExclusive + desc.maxStaticOutcastMem * desc.totalAssembleOutcastSlot;
}

// Loose upper bound for binary-search range only (not the final k_eff).
// Intentionally over-estimates: kUpper >= true max fitting k; must never tighten below it.
// raw(k) = c0 + c1*k matches pre-align tensor bytes; + (32K-1) and +1 are align/ceil slack.
static uint32_t EstimateMaxStitchDepthKUpperBound(const WorkspaceDesc& desc, uint32_t parallelism,
                                                  uint64_t aicoreSpilled, uint64_t debugTotal, uint64_t cap)
{
    if (parallelism == 0) {
        return 1;
    }
    const uint64_t fixed = aicoreSpilled + debugTotal;
    if (cap <= fixed) {
        return 1;
    }
    const uint64_t tensorCap = cap - fixed;

    uint64_t c0 = 0;
    uint64_t c1 = 0;
    ComputeTensorBudgetLinearCoeffs(desc, c0, c1);
    if (c1 == 0) {
        return UINT32_MAX;
    }

    const uint64_t perCoreCap = tensorCap / parallelism;
    if (perCoreCap <= c0) {
        return UINT32_MAX;
    }

    // Over-estimate: headroom adds (32K-1); kUpper adds +1 after divide.
    uint64_t headroom = perCoreCap - c0 + (kAlignment32K - 1);
    const uint64_t kUpper = headroom / c1 + 1;
    if (kUpper > UINT32_MAX) {
        return UINT32_MAX;
    }
    return static_cast<uint32_t>(kUpper);
}

static uint64_t InitialStitchDepthSearchHi(uint32_t kCapUpper)
{
    const uint64_t cap = kCapUpper == 0 ? 1 : static_cast<uint64_t>(kCapUpper);
    return cap < 32 ? cap : 32;
}

StitchDepthSearchBounds ExpandStitchDepthSearchBounds(const TensorBudgetModel& model, uint32_t kCapUpper)
{
    StitchDepthSearchBounds bounds;
    const uint64_t capLimit = kCapUpper == 0 ? 1 : static_cast<uint64_t>(kCapUpper);
    uint64_t hi = InitialStitchDepthSearchHi(kCapUpper);
    while (model.Fits(static_cast<uint32_t>(hi))) {
        bounds.lo = hi;
        if (hi >= capLimit || !CanDoubleStitchDepthK(hi)) {
            break;
        }
        const uint64_t nextHi = MinU64(hi * 2, capLimit);
        if (nextHi <= hi) {
            break;
        }
        if (model.BytesAt(static_cast<uint32_t>(nextHi)) == model.BytesAt(static_cast<uint32_t>(hi))) {
            break;
        }
        hi = nextHi;
    }

    bounds.searchHi = MinU64(hi, capLimit);
    if (!model.Fits(static_cast<uint32_t>(hi)) && hi > bounds.lo) {
        bounds.searchHi = MinU64(hi - 1, capLimit);
    }
    return bounds;
}

uint32_t BinarySearchMaxFittingStitchDepth(const TensorBudgetModel& model, const StitchDepthSearchBounds& bounds)
{
    uint64_t kCandidate = bounds.lo;
    if (bounds.lo >= bounds.searchHi) {
        return static_cast<uint32_t>(kCandidate);
    }

    uint64_t searchLo = bounds.lo;
    uint64_t searchHiVal = bounds.searchHi;
    while (searchLo < searchHiVal) {
        const uint64_t mid = searchLo + (searchHiVal - searchLo + 1) / 2;
        if (model.Fits(static_cast<uint32_t>(mid))) {
            searchLo = mid;
        } else {
            searchHiVal = mid - 1;
        }
    }
    return static_cast<uint32_t>(searchLo);
}

// When 32KB alignment plateaus, BytesAt(k) may stay flat as k grows. Pick the smallest fitting k.
uint32_t PreferMinimalKWithSameBudget(const TensorBudgetModel& model, uint32_t kCandidate)
{
    if (kCandidate <= 1) {
        return kCandidate;
    }
    const uint64_t targetBytes = model.BytesAt(kCandidate);
    uint64_t searchLo = 1;
    uint64_t searchHi = kCandidate;
    while (searchLo < searchHi) {
        const uint64_t mid = searchLo + (searchHi - searchLo) / 2;
        if (model.BytesAt(static_cast<uint32_t>(mid)) == targetBytes) {
            searchHi = mid;
        } else {
            searchLo = mid + 1;
        }
    }
    return static_cast<uint32_t>(searchLo);
}

static uint32_t DeriveEffectiveStitchNum(uint64_t cap, const WorkspaceDesc& desc, uint32_t parallelism,
                                         uint64_t aicoreSpilled, uint64_t debugTotal)
{
    const TensorBudgetModel model{desc, parallelism, aicoreSpilled, debugTotal, cap};
    const uint32_t kCapUpper = EstimateMaxStitchDepthKUpperBound(desc, parallelism, aicoreSpilled, debugTotal, cap);
    const StitchDepthSearchBounds bounds = ExpandStitchDepthSearchBounds(model, kCapUpper);
    const uint32_t maxK = BinarySearchMaxFittingStitchDepth(model, bounds);
    return PreferMinimalKWithSameBudget(model, maxK);
}
} // namespace

StitchDepthConfig ResolveStitchDepthConfig(WorkspaceDesc& desc, const RuntimeWorkspaceConfig& runtimeCfg)
{
    StitchDepthConfig config;
    config.kEff = runtimeCfg.stitchNumMax;
    const bool memoryDriven = runtimeCfg.maxWorkspaceBytes > 0 &&
                              runtimeCfg.maxWorkspaceBytes > runtimeCfg.workspaceStitchMin;
    if (memoryDriven) {
        const uint32_t derivedK = DeriveEffectiveStitchNum(runtimeCfg.maxWorkspaceBytes, desc, runtimeCfg.parallelism,
                                                           runtimeCfg.aicoreSpilled, runtimeCfg.debugTotal);
        config.kEff = derivedK;
        config.encodedWorkspaceSize = runtimeCfg.maxWorkspaceBytes;
        config.memoryDrivenWorkspace = 1;
        config.stitchMaxFunctionNum = config.kEff;
    } else {
        WorkspaceDesc scratch = desc;
        BuildTensorWorkspaceFromDescriptor(scratch, runtimeCfg.stitchNumMax);
        config.encodedWorkspaceSize = WorkspaceTotalFromDesc(scratch, runtimeCfg.parallelism, runtimeCfg.aicoreSpilled,
                                                             runtimeCfg.debugTotal);
        config.memoryDrivenWorkspace = 0;
        config.stitchMaxFunctionNum = runtimeCfg.stitchNumMax;
    }

    config.outcastCacheDepth = DeriveOutcastCacheDepth(config.kEff);
    config.runtimeOutcastPoolDepth = DeriveRuntimeOutcastPoolDepth(config.kEff);
    BuildTensorWorkspaceFromDescriptor(desc, config.kEff);
    return config;
}

uint64_t WorkspaceBytesToKbCeil(uint64_t bytes) { return bytes == 0 ? UINT64_C(0) : (bytes + 1023) / 1024; }

} // namespace dynamic
} // namespace npu::tile_fwk
