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
 * \file test_encode.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>

#include "interface/function/rebuildable_attribute.h"
#include "interface/tensor/irbuilder.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_workspace.h"
#include "machine/utils/dynamic/workspace_budget_calculator.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/program/program.h"

namespace npu::tile_fwk::dynamic {
void AddTokenDependEdgesToColorGraph(std::vector<Operation*>& callopList,
                                     std::unordered_map<Operation*, int>& callopIndexDict,
                                     std::unordered_map<int, std::vector<int>>& colorOutGraph);
}

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestDevEncode : public testing::Test {};

namespace {

constexpr int kMemoryDrivenLoopCount = 4;
constexpr int kMemoryDrivenTileSize = 32;

void SetupMemoryDrivenEncodeTest(uint32_t stitchNumMax, uint64_t maxWorkspaceKb = 0)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, static_cast<int>(stitchNumMax));
    if (maxWorkspaceKb > 0) {
        config::SetRuntimeOption(MAX_WORKSPACE_KB, static_cast<long>(maxWorkspaceKb));
    }
    TileShape::Current().SetVecTile(kMemoryDrivenTileSize, kMemoryDrivenTileSize);
    TileShape::Current().SetCubeTile({kMemoryDrivenTileSize, kMemoryDrivenTileSize},
                                     {kMemoryDrivenTileSize, kMemoryDrivenTileSize},
                                     {kMemoryDrivenTileSize, kMemoryDrivenTileSize});
}

void BuildSimpleLoopFunction(const char* funcName, const char* loopName, const Tensor& t0, const Tensor& t1,
                             Tensor& out)
{
    FUNCTION(funcName, {t0, t1}, {out})
    {
        LOOP(loopName, FunctionType::DYNAMIC_LOOP, i, LoopRange(kMemoryDrivenLoopCount))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * kMemoryDrivenTileSize, 0}, out);
        }
    }
}

DevAscendProgram* GetLastDevProg()
{
    std::shared_ptr<DyndevFunctionAttribute>
        funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    EXPECT_NE(funcDynDev, nullptr);
    if (funcDynDev == nullptr) {
        return nullptr;
    }
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    EXPECT_NE(devProg, nullptr);
    return devProg;
}

uint64_t ComputeWorkspaceStitchMin(DevAscendProgram* devProg)
{
    Function* func = Program::GetInstance().GetLastFunction();
    auto dyndev = func->GetDyndevAttribute();
    const uint64_t progAddr = reinterpret_cast<uint64_t>(devProg);
    devProg->RelocProgram(0, progAddr);
    const auto flex = CollectWorkspaceDesc(func, *devProg, dyndev->constructAssembleNeedAllocRuntimeSlots);
    devProg->RelocProgram(progAddr, 0);
    const uint64_t aicoreSpilled = flex.maxLeafPerCoreSpilledMem * static_cast<uint64_t>(GetPlatformMaxAicoreNum());
    const uint64_t debugTotal = devProg->memBudget.debug.dumpTensor + devProg->memBudget.debug.leafDump;
    return TensorWorkspaceBytesAtMinimumStitchDepth(flex, devProg->GetParallelism(), aicoreSpilled, debugTotal);
}

uint32_t OutcastCacheDepthFromPool(const DevAscendProgram* devProg)
{
    const uint32_t parallelism = devProg->memBudget.tensor.parallelism;
    if (devProg->slotSize == 0 || parallelism == 0) {
        return 0;
    }
    return devProg->memBudget.tensor.runtimeOutcastPoolSize / (devProg->slotSize * parallelism) - 1;
}

uint32_t TensorStitchDepthK(const DevAscendProgram* devProg, uint32_t configuredStitchNumMax)
{
    if (devProg->memBudget.tensor.memoryDrivenWorkspace != 0) {
        return devProg->stitchMaxFunctionNum;
    }
    return configuredStitchNumMax;
}

RuntimeWorkspaceConfig MakeNonMemoryDrivenCfg(uint32_t stitchNumMax, uint32_t parallelism = 1,
                                              uint64_t aicoreSpilled = 0, uint64_t debugTotal = 0)
{
    RuntimeWorkspaceConfig cfg;
    cfg.stitchNumMax = stitchNumMax;
    cfg.parallelism = parallelism;
    cfg.aicoreSpilled = aicoreSpilled;
    cfg.debugTotal = debugTotal;
    cfg.maxWorkspaceBytes = 0;
    cfg.workspaceStitchMin = UINT64_MAX;
    return cfg;
}

WorkspaceDesc ResolvedWorkspaceAtStitchDepth(const WorkspaceDesc& baseDesc, uint32_t stitchDepthK,
                                             uint32_t parallelism = 1, uint64_t aicoreSpilled = 0,
                                             uint64_t debugTotal = 0)
{
    WorkspaceDesc desc = baseDesc;
    (void)ResolveStitchDepthConfig(desc, MakeNonMemoryDrivenCfg(stitchDepthK, parallelism, aicoreSpilled, debugTotal));
    return desc;
}

uint64_t EncodedWorkspaceBytesAtStitchDepth(const WorkspaceDesc& baseDesc, uint32_t stitchDepthK,
                                            uint32_t parallelism = 1, uint64_t aicoreSpilled = 0,
                                            uint64_t debugTotal = 0)
{
    WorkspaceDesc desc = baseDesc;
    const StitchDepthConfig depth = ResolveStitchDepthConfig(
        desc, MakeNonMemoryDrivenCfg(stitchDepthK, parallelism, aicoreSpilled, debugTotal));
    return depth.encodedWorkspaceSize;
}

WorkspaceDesc MakeLinearBudgetDesc(uint64_t maxStaticOutcastMem, uint64_t totalExclusiveOutcastSlot,
                                   uint64_t totalAssembleOutcastSlot, uint64_t rootInnerSpilledRawMem,
                                   uint64_t rootExclusiveRawMem = 0)
{
    WorkspaceDesc desc;
    desc.maxStaticOutcastMem = maxStaticOutcastMem;
    desc.totalExclusiveOutcastSlot = totalExclusiveOutcastSlot;
    desc.totalAssembleOutcastSlot = totalAssembleOutcastSlot;
    WorkspaceDesc::WorkspacePerRootFunctionDesc profile;
    profile.unroll = 1;
    profile.rootInnerSpilledRawMem = rootInnerSpilledRawMem;
    profile.rootTotalExclusiveOutcastRawMem = rootExclusiveRawMem;
    desc.rootFuncDescList = {profile};
    return desc;
}

uint32_t BruteForceDeriveEffectiveStitchNum(uint64_t cap, const WorkspaceDesc& desc, uint32_t parallelism,
                                            uint64_t aicoreSpilled, uint64_t debugTotal,
                                            uint32_t searchLimit = UINT32_MAX)
{
    uint32_t maxFit = 1;
    for (uint32_t k = 1; k <= searchLimit; ++k) {
        if (EncodedWorkspaceBytesAtStitchDepth(desc, k, parallelism, aicoreSpilled, debugTotal) <= cap) {
            maxFit = k;
        } else {
            break;
        }
    }
    uint32_t minK = maxFit;
    while (minK > 1 && EncodedWorkspaceBytesAtStitchDepth(desc, minK, parallelism, aicoreSpilled, debugTotal) ==
                           EncodedWorkspaceBytesAtStitchDepth(desc, minK - 1, parallelism, aicoreSpilled, debugTotal)) {
        --minK;
    }
    return minK;
}

StitchDepthConfig ResolveMemoryDrivenDepth(WorkspaceDesc& desc, const RuntimeWorkspaceConfig& baseCfg,
                                           uint64_t maxWorkspaceBytes)
{
    RuntimeWorkspaceConfig cfg = baseCfg;
    cfg.maxWorkspaceBytes = maxWorkspaceBytes;
    cfg.workspaceStitchMin = TensorWorkspaceBytesAtMinimumStitchDepth(desc, cfg.parallelism, cfg.aicoreSpilled,
                                                                      cfg.debugTotal);
    return ResolveStitchDepthConfig(desc, cfg);
}

} // namespace

TEST_F(TestDevEncode, DevSymShape)
{
    DevSymShape shape;
    shape.SetShape({SymInt(true, 0), SymInt(true, 2), SymInt(2)}); // 4, 8, 2
    uint64_t exprTbl[] = {4, 6, 8};
    uint64_t strides[3] = {0};
    shape.ToStride(strides, exprTbl);
    EXPECT_EQ(strides[0], 16);
    EXPECT_EQ(strides[1], 2);
    EXPECT_EQ(strides[2], 1);
}

TEST_F(TestDevEncode, test_parse_unroll_times_from_name)
{
    EXPECT_EQ(ParseUnrollTimesFromName("TENSOR_main_LoopUnroll2_L0_Unroll1_PATH0_hiddenfunc0_root"), 2u);
    EXPECT_EQ(ParseUnrollTimesFromName("kernel_LoopUnroll512_suffix"), 512u);
    EXPECT_EQ(ParseUnrollTimesFromName("kernel_Unroll64"), 64u);
    EXPECT_EQ(ParseUnrollTimesFromName("foo_LoopUnroll2_bar_Unroll4"), 8u);
    EXPECT_EQ(ParseUnrollTimesFromName("plain_name"), 1u);
}

TEST_F(TestDevEncode, test_dev_encode_program)
{
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int LOOP_COUNT_INNER = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {LOOP_COUNT_INNER * s, s}, "out");

    // clc
    FUNCTION("main_LoopUnroll2", {t0, t1, t2}, {out})
    {
        LOOP("main_LoopUnroll2_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT_INNER))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);

            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }

    std::shared_ptr<DyndevFunctionAttribute>
        funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    constexpr uint32_t kConfiguredStitchNumMax = 64;
    EXPECT_EQ(devProg->stitchMaxFunctionNum, kConfiguredStitchNumMax);
    const uint32_t expectedOutcastDepth = kConfiguredStitchNumMax;
    EXPECT_EQ(OutcastCacheDepthFromPool(devProg), expectedOutcastDepth);

    Function* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    auto* wsAttr = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableWorkspaceDesc>(func);
    ASSERT_NE(wsAttr, nullptr);
    EXPECT_EQ(devProg->memBudget.tensor.rootInnerSpilledMem, wsAttr->data.maxRootInnerSpilledMem);
    EXPECT_EQ(devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts, wsAttr->data.maxRootTotalExclusiveOutcastMem);

    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t contextWorkspaceAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr, nullptr,
                                                        devProg->GetParallelism());
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
        reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    devProg->Dump(0, true);
    devProg->DumpFile("./dum_dev_program.txt");
    devProg->ResetRerun();
    devProg->RuntimeVerify(0, 0);
    EXPECT_NE(devProg->GetInputTensorSlotIndexList().empty(), true);
    EXPECT_NE(devProg->GetOutputTensorSlotIndexList().empty(), true);
    EXPECT_NE(devProg->GetAssembleTensorSlotIndexList().empty(), true);
    (void)devProg->GetDevControlFlowBinary();
    (void)devProg->GetHostControlFlowBinary();
    (void)devProg->GetExpressionTableBinary();

    DevAscendFunction* devFunc = devProg->GetFunction(0);
    ASSERT_NE(devFunc, nullptr);
    EXPECT_NE(devProg->GetFunctionByRawName(devFunc->GetRawName()), nullptr);
    devFunc->Dump();
    EXPECT_EQ(devFunc->HasValueDepend(), false);
    EXPECT_EQ(devFunc->LookupIncastBySlotIndex(0), 0);
    EXPECT_EQ(devFunc->LookupOutcastBySlotIndex(0), -1);
    std::vector<int> slotIndexList = {0};
    (void)devFunc->LookupIncastBySlotIndexList(slotIndexList);
    (void)devFunc->LookupOutcastBySlotIndexList(slotIndexList);

    DevAscendFunction* devFunc1 = devProg->GetFunction(1);
    if (devFunc1 != nullptr) {
        devFunc->LookupConnectionSlotIndexFrom(devFunc1);
    }

    DevAscendFunctionDuppedData* devFuncDuppedData = devFunc->GetDuppedData();
    ASSERT_NE(devFuncDuppedData, nullptr);
    devFuncDuppedData->source_ = devFunc;
    (void)devFuncDuppedData->Dump();

    devProg->ResetFromLaunch();
}
static DevAscendProgram* BuildAndGetDevProgForExpectedMaxCachedNum()
{
    constexpr int LOOP_COUNT_INNER = 4;
    int s = 32;
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {LOOP_COUNT_INNER * s, s}, "out");
    FUNCTION("stitch_max_cached_num", {t0, t1, t2}, {out})
    {
        LOOP("stitch_max_cached_num_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT_INNER))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);
            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }
    std::shared_ptr<DyndevFunctionAttribute> dynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    EXPECT_NE(dynDev, nullptr);
    if (dynDev == nullptr) {
        return nullptr;
    }
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(dynDev->devProgBinary.data());
    EXPECT_NE(devProg, nullptr);
    return devProg;
}
TEST_F(TestDevEncode, test_max_stitch_function_num)
{
    // case1:
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 256);
    DevAscendProgram* devProg1 = BuildAndGetDevProgForExpectedMaxCachedNum();
    ASSERT_NE(devProg1, nullptr);
    constexpr uint32_t kConfiguredStitchNumMax = 256;
    EXPECT_EQ(devProg1->stitchMaxFunctionNum, kConfiguredStitchNumMax);
    const uint32_t outcastDepth = OutcastCacheDepthFromPool(devProg1);
    EXPECT_EQ(outcastDepth, kConfiguredStitchNumMax);
    const uint32_t expectedPool = devProg1->slotSize * (outcastDepth + 1) * devProg1->GetParallelism();
    EXPECT_EQ(devProg1->memBudget.tensor.runtimeOutcastPoolSize, expectedPool);
}

TEST_F(TestDevEncode, test_memory_driven_derives_k_eff_at_minimum)
{
    SetupMemoryDrivenEncodeTest(64);
    int s = kMemoryDrivenTileSize;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {kMemoryDrivenLoopCount * s, s}, "out");
    BuildSimpleLoopFunction("memory_driven_k_eff_probe", "memory_driven_k_eff_probe_L0", t0, t1, out);
    DevAscendProgram* devProg = GetLastDevProg();
    ASSERT_NE(devProg, nullptr);
    const uint64_t workspaceStitchMin = ComputeWorkspaceStitchMin(devProg);
    if (workspaceStitchMin == 0) {
        GTEST_SKIP() << "Spill-free kernel has zero stitch-min workspace; skip minimum-cap k_eff probe.";
    }
    const uint64_t minKb = std::max<uint64_t>(1, (workspaceStitchMin + 1023) / 1024);

    SetupMemoryDrivenEncodeTest(64, minKb);
    RecordFunc recEqual("memory_driven_k_eff_cap", {t0, t1}, {out});
    for ([[maybe_unused]] auto& step : recEqual) {
        LOOP("memory_driven_k_eff_cap_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kMemoryDrivenLoopCount))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * kMemoryDrivenTileSize, 0}, out);
        }
    }
    EXPECT_THROW(recEqual.EndFunction(), Error);

    const uint64_t capKb = minKb + 1;
    SetupMemoryDrivenEncodeTest(64, capKb);
    BuildSimpleLoopFunction("memory_driven_k_eff_cap_plus1", "memory_driven_k_eff_cap_plus1_L0", t0, t1, out);
    devProg = GetLastDevProg();
    ASSERT_NE(devProg, nullptr);
    EXPECT_EQ(devProg->memBudget.tensor.memoryDrivenWorkspace, 1u);
    EXPECT_GE(minKb * 1024, workspaceStitchMin);
}

TEST_F(TestDevEncode, test_validate_max_workspace_or_throw)
{
    constexpr uint64_t kStitchMin = 32ULL * 1024;
    EXPECT_NO_THROW(ValidateMaxWorkspaceOrThrow(0, kStitchMin));
    EXPECT_THROW(ValidateMaxWorkspaceOrThrow(kStitchMin, kStitchMin), Error);
    EXPECT_NO_THROW(ValidateMaxWorkspaceOrThrow(kStitchMin + 1, kStitchMin));
    EXPECT_THROW(ValidateMaxWorkspaceOrThrow(1024, kStitchMin), Error);
    EXPECT_THROW(ValidateMaxWorkspaceOrThrow(10ULL * 1024, kStitchMin), Error);
}

TEST_F(TestDevEncode, test_max_workspace_kb_below_operator_min_throws)
{
    SetupMemoryDrivenEncodeTest(64);
    int s = kMemoryDrivenTileSize;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {kMemoryDrivenLoopCount * s, s}, "out");
    BuildSimpleLoopFunction("max_ws_below_min_probe", "max_ws_below_min_probe_L0", t0, t1, out);
    DevAscendProgram* devProg = GetLastDevProg();
    ASSERT_NE(devProg, nullptr);
    const uint64_t workspaceStitchMin = ComputeWorkspaceStitchMin(devProg);
    if (workspaceStitchMin == 0) {
        GTEST_SKIP() << "Spill-free kernel has zero stitch-min workspace; skip max_workspace_kb below-min check.";
    }

    constexpr uint64_t kTinyMaxWorkspaceKb = 1;
    ASSERT_LT(kTinyMaxWorkspaceKb * 1024, workspaceStitchMin);

    SetupMemoryDrivenEncodeTest(64, kTinyMaxWorkspaceKb);
    RecordFunc rec("max_ws_below_min", {t0, t1}, {out});
    for ([[maybe_unused]] auto& step : rec) {
        LOOP("max_ws_below_min_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kMemoryDrivenLoopCount))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * kMemoryDrivenTileSize, 0}, out);
        }
    }
    EXPECT_THROW(rec.EndFunction(), Error);
}

TEST_F(TestDevEncode, test_memory_driven_runtime_outcast_cache_depth)
{
    SetupMemoryDrivenEncodeTest(64, 40000);
    int s = kMemoryDrivenTileSize;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {kMemoryDrivenLoopCount * s, s}, "out");
    BuildSimpleLoopFunction("memory_driven_outcast_depth", "memory_driven_outcast_depth_L0", t0, t1, out);
    std::shared_ptr<DyndevFunctionAttribute>
        funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = GetLastDevProg();
    ASSERT_NE(devProg, nullptr);
    constexpr uint32_t kConfiguredStitchNumMax = 64;
    const uint32_t kEff = TensorStitchDepthK(devProg, kConfiguredStitchNumMax);
    const uint32_t kForPool = std::min(kEff, static_cast<uint32_t>(npu::tile_fwk::dynamic::MAX_STITCH_FUNC_NUM));
    const uint32_t outcastDepth = OutcastCacheDepthFromPool(devProg);
    EXPECT_LE(outcastDepth, kForPool);
    EXPECT_EQ(outcastDepth, kForPool);
    EXPECT_EQ(devProg->memBudget.tensor.runtimeOutcastPoolSize,
              devProg->slotSize * (outcastDepth + 1) * devProg->GetParallelism());

    const uint64_t totalOutcastSlots = devProg->memBudget.tensor.BoundaryAndInnerTemporalOutcastSlotNum();
    EXPECT_GE(totalOutcastSlots, static_cast<uint64_t>(std::max(outcastDepth, 2u)));
    EXPECT_GT(devProg->memBudget.tensor.devTaskBoundaryOutcastNum, 0u);
    EXPECT_LE(devProg->memBudget.tensor.devTaskInnerTemporalOutcastNum,
              devProg->memBudget.tensor.slottableOutcastSlotSize * kEff);
    EXPECT_EQ(devProg->controlFlowCache.runtimeBackup.workspace.tensorAllocators[0].slottedOutcastsBlockList.size(),
              devProg->GetCtrlFlowCacheSlottedOutcastBlockCount(devProg->slotSize));
}

TEST_F(TestDevEncode, test_memory_driven_unroll_from_name)
{
    SetupMemoryDrivenEncodeTest(64, 40000);
    int s = kMemoryDrivenTileSize;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {kMemoryDrivenLoopCount * s, s}, "out");
    FUNCTION("main_LoopUnroll2", {t0, t1, t2}, {out})
    {
        LOOP("main_LoopUnroll2_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kMemoryDrivenLoopCount))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);
            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }
    std::shared_ptr<DyndevFunctionAttribute>
        funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = GetLastDevProg();
    ASSERT_NE(devProg, nullptr);
    uint32_t maxDevRootUnroll = 1;
    for (const auto& encoded : funcDynDev->devEncodeList) {
        if (encoded.empty()) {
            continue;
        }
        auto* devRoot = reinterpret_cast<DevAscendFunction*>(const_cast<uint8_t*>(encoded.data()));
        const uint32_t unrollFromName = ParseUnrollTimesFromName(devRoot->GetRawName());
        EXPECT_EQ(devRoot->unrollTimes, unrollFromName);
        maxDevRootUnroll = std::max(maxDevRootUnroll, devRoot->unrollTimes);
    }
    EXPECT_GE(maxDevRootUnroll, 2u);
}

TEST_F(TestDevEncode, test_dev_func_dupped)
{
    DevAscendRawTensor rawTensor;
    std::vector<std::string> lines;
    std::stringstream oss;
    DevAscendFunctionDupped funcDuppped;
    funcDuppped.DumpRawShape(&rawTensor, 0, lines, oss);
}

TEST_F(TestDevEncode, test_init_wrap_info)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");

    FUNCTION("test_wrap_info", {t0, t1}, {out})
    {
        auto temp = Add(t0, t1);
        Assemble(temp, {0, 0}, out);
    }

    std::shared_ptr<DyndevFunctionAttribute>
        funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);

    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t contextWorkspaceAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr, nullptr,
                                                        devProg->GetParallelism());
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
        reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    DevAscendFunction* devFunc = devProg->GetFunction(0);
    ASSERT_NE(devFunc, nullptr);

    EXPECT_GE(devFunc->wrapIdNum_, 0);
}

TEST_F(TestDevEncode, test_workspace_budget_calculator_unroll_ge_depth)
{
    WorkspaceDesc desc;
    desc.maxStaticOutcastMem = 200;
    desc.totalExclusiveOutcastSlot = 2;
    desc.totalAssembleOutcastSlot = 1;

    WorkspaceDesc::WorkspacePerRootFunctionDesc lowUnroll;
    lowUnroll.unroll = 1;
    lowUnroll.rootInnerSpilledRawMem = 100;
    lowUnroll.rootTotalExclusiveOutcastRawMem = 50;

    WorkspaceDesc::WorkspacePerRootFunctionDesc highUnroll;
    highUnroll.unroll = 64;
    highUnroll.rootInnerSpilledRawMem = 8000;
    highUnroll.rootTotalExclusiveOutcastRawMem = 4000;

    desc.rootFuncDescList = {lowUnroll, highUnroll};

    const WorkspaceDesc wsK1 = ResolvedWorkspaceAtStitchDepth(desc, 1);
    EXPECT_EQ(wsK1.maxRootInnerSpilledMem, 8192u);
    EXPECT_EQ(wsK1.maxRootTotalExclusiveOutcastMem, 4096u);

    const WorkspaceDesc wsK64 = ResolvedWorkspaceAtStitchDepth(desc, 64);
    EXPECT_EQ(wsK64.maxRootInnerSpilledMem, 524288u);
    EXPECT_EQ(wsK64.maxRootTotalExclusiveOutcastMem, 262144u);

    const WorkspaceDesc wsK128 = ResolvedWorkspaceAtStitchDepth(desc, 128);
    EXPECT_EQ(wsK128.maxRootInnerSpilledMem, 1048576u);
    EXPECT_EQ(wsK128.maxRootTotalExclusiveOutcastMem, 524288u);
}

TEST_F(TestDevEncode, test_workspace_budget_calculator_resolve_depth)
{
    WorkspaceDesc desc;
    desc.maxStaticOutcastMem = 4096;
    desc.totalExclusiveOutcastSlot = 1;

    WorkspaceDesc::WorkspacePerRootFunctionDesc profile;
    profile.unroll = 1;
    profile.rootInnerSpilledRawMem = 4096;
    desc.rootFuncDescList = {profile};

    RuntimeWorkspaceConfig cfg;
    cfg.stitchNumMax = 64;
    cfg.parallelism = 1;
    cfg.aicoreSpilled = 0;
    cfg.debugTotal = 0;

    const StitchDepthConfig depth = ResolveStitchDepthConfig(desc, cfg);
    EXPECT_EQ(depth.kEff, 64u);
    EXPECT_EQ(depth.outcastCacheDepth, 64u);
    EXPECT_EQ(depth.stitchMaxFunctionNum, 64u);
    EXPECT_GT(depth.encodedWorkspaceSize, 0u);
}

TEST_F(TestDevEncode, test_workspace_budget_calculator_outcast_cache_depth)
{
    WorkspaceDesc desc8 = MakeLinearBudgetDesc(4096, 1, 0, 4096, 0);
    const StitchDepthConfig depth8 = ResolveStitchDepthConfig(desc8, MakeNonMemoryDrivenCfg(8));
    EXPECT_EQ(depth8.kEff, 8u);
    EXPECT_EQ(depth8.outcastCacheDepth, 8u);

    WorkspaceDesc desc64 = MakeLinearBudgetDesc(4096, 1, 0, 4096, 0);
    StitchDepthConfig depth64 = ResolveStitchDepthConfig(desc64, MakeNonMemoryDrivenCfg(64));
    EXPECT_EQ(depth64.outcastCacheDepth, 64u);

    WorkspaceDesc descMd = MakeLinearBudgetDesc(4096, 1, 0, 4096, 0);
    RuntimeWorkspaceConfig mdCfg;
    mdCfg.stitchNumMax = 64;
    mdCfg.parallelism = 1;
    const StitchDepthConfig depthMd = ResolveMemoryDrivenDepth(descMd, mdCfg, 40ULL * 1024 * 1024);
    EXPECT_GT(depthMd.kEff, 200u);
    EXPECT_EQ(depthMd.outcastCacheDepth, depthMd.kEff);
}

TEST_F(TestDevEncode, test_runtime_outcast_pool_depth_capped_at_max_stitch)
{
    const uint32_t kPoolCap = static_cast<uint32_t>(npu::tile_fwk::dynamic::MAX_STITCH_FUNC_NUM);

    WorkspaceDesc descMd = MakeLinearBudgetDesc(4096, 1, 0, 4096, 0);
    RuntimeWorkspaceConfig mdCfg;
    mdCfg.stitchNumMax = 64;
    mdCfg.parallelism = 1;
    const StitchDepthConfig depthMd = ResolveMemoryDrivenDepth(descMd, mdCfg, 40ULL * 1024 * 1024);
    EXPECT_GT(depthMd.kEff, kPoolCap);
    EXPECT_EQ(depthMd.runtimeOutcastPoolDepth, kPoolCap);
    EXPECT_EQ(depthMd.outcastCacheDepth, depthMd.kEff);
    EXPECT_GT(depthMd.outcastCacheDepth, depthMd.runtimeOutcastPoolDepth);

    WorkspaceDesc desc64 = MakeLinearBudgetDesc(4096, 1, 0, 4096, 0);
    const StitchDepthConfig depth64 = ResolveStitchDepthConfig(desc64, MakeNonMemoryDrivenCfg(64));
    EXPECT_EQ(depth64.runtimeOutcastPoolDepth, depth64.outcastCacheDepth);
    EXPECT_EQ(depth64.runtimeOutcastPoolDepth, 64u);
}

TEST_F(TestDevEncode, test_memory_driven_k_eff_from_memory_cap)
{
    WorkspaceDesc desc;
    desc.maxStaticOutcastMem = 4096;
    desc.totalExclusiveOutcastSlot = 1;
    desc.totalAssembleOutcastSlot = 0;

    WorkspaceDesc::WorkspacePerRootFunctionDesc profile;
    profile.unroll = 1;
    profile.rootInnerSpilledRawMem = 4096;
    desc.rootFuncDescList = {profile};

    RuntimeWorkspaceConfig cfg;
    cfg.stitchNumMax = 64;
    cfg.parallelism = 1;
    cfg.aicoreSpilled = 0;
    cfg.debugTotal = 0;
    cfg.maxWorkspaceBytes = 40ULL * 1024 * 1024;
    cfg.workspaceStitchMin = TensorWorkspaceBytesAtMinimumStitchDepth(desc, cfg.parallelism, 0, 0);

    const uint32_t expectedK = BruteForceDeriveEffectiveStitchNum(cfg.maxWorkspaceBytes, desc, cfg.parallelism,
                                                                  cfg.aicoreSpilled, cfg.debugTotal);
    const StitchDepthConfig depth = ResolveStitchDepthConfig(desc, cfg);
    EXPECT_EQ(depth.memoryDrivenWorkspace, 1u);
    EXPECT_EQ(depth.kEff, expectedK);
    EXPECT_GT(depth.kEff, cfg.stitchNumMax);
    EXPECT_EQ(depth.outcastCacheDepth, depth.kEff);
    EXPECT_EQ(depth.runtimeOutcastPoolDepth,
              depth.kEff > static_cast<uint32_t>(npu::tile_fwk::dynamic::MAX_STITCH_FUNC_NUM) ?
                  static_cast<uint32_t>(npu::tile_fwk::dynamic::MAX_STITCH_FUNC_NUM) :
                  depth.kEff);
    EXPECT_LE(depth.runtimeOutcastPoolDepth, static_cast<uint32_t>(npu::tile_fwk::dynamic::MAX_STITCH_FUNC_NUM));
    EXPECT_EQ(depth.stitchMaxFunctionNum, depth.kEff);
}

TEST_F(TestDevEncode, test_memory_driven_ctrlflow_backup_decoupled_from_stitch_num_max)
{
    WorkspaceDesc desc;
    desc.maxStaticOutcastMem = 4096;
    desc.totalExclusiveOutcastSlot = 0;
    desc.totalAssembleOutcastSlot = 2;

    WorkspaceDesc::WorkspacePerRootFunctionDesc profile;
    profile.unroll = 1;
    profile.rootInnerSpilledRawMem = 4096;
    desc.rootFuncDescList = {profile};

    RuntimeWorkspaceConfig cfg;
    cfg.stitchNumMax = 64;
    cfg.parallelism = 1;
    cfg.aicoreSpilled = 0;
    cfg.debugTotal = 0;
    cfg.maxWorkspaceBytes = 40ULL * 1024 * 1024;
    cfg.workspaceStitchMin = TensorWorkspaceBytesAtMinimumStitchDepth(desc, cfg.parallelism, 0, 0);

    const StitchDepthConfig depth = ResolveStitchDepthConfig(desc, cfg);
    EXPECT_GT(depth.kEff, cfg.stitchNumMax);

    const uint64_t requiredSlotBlocks = desc.devTaskBoundaryOutcastNum + desc.devTaskInnerTemporalOutcastNum;
    const uint64_t totalSlot = desc.totalExclusiveOutcastSlot + desc.totalAssembleOutcastSlot;
    const uint32_t estimatedStitching = cfg.stitchNumMax * profile.unroll;
    const uint32_t legacyBackupDepth = estimatedStitching < depth.kEff ? estimatedStitching : depth.kEff;
    const uint64_t legacyBackupCount = EstimateCtrlFlowCacheSlottedBlockCount(totalSlot, legacyBackupDepth);
    EXPECT_LT(legacyBackupCount, requiredSlotBlocks);

    const uint64_t alignedBackupCount = requiredSlotBlocks;
    EXPECT_GE(alignedBackupCount, requiredSlotBlocks);
    EXPECT_EQ(alignedBackupCount, desc.totalAssembleOutcastSlot * depth.kEff + desc.devTaskBoundaryOutcastNum);
}

TEST_F(TestDevEncode, test_derive_k_eff_matches_brute_force)
{
    const WorkspaceDesc desc = MakeLinearBudgetDesc(4096, 1, 2, 4096, 2048);
    const uint32_t parallelism = 1;
    const uint64_t aicoreSpilled = 0;
    const uint64_t debugTotal = 0;
    const uint64_t caps[] = {
        TensorWorkspaceBytesAtMinimumStitchDepth(desc, parallelism, aicoreSpilled, debugTotal) + 32768,
        512ULL * 1024,
        4ULL * 1024 * 1024,
        40ULL * 1024 * 1024,
    };

    for (const uint64_t cap : caps) {
        WorkspaceDesc descCopy = desc;
        RuntimeWorkspaceConfig cfg;
        cfg.stitchNumMax = 64;
        cfg.parallelism = parallelism;
        cfg.aicoreSpilled = aicoreSpilled;
        cfg.debugTotal = debugTotal;
        const uint32_t expected = BruteForceDeriveEffectiveStitchNum(cap, desc, parallelism, aicoreSpilled, debugTotal);
        const StitchDepthConfig depth = ResolveMemoryDrivenDepth(descCopy, cfg, cap);
        EXPECT_EQ(depth.kEff, expected) << "cap=" << cap;
    }
}

TEST_F(TestDevEncode, test_derive_k_eff_flat_budget_returns_minimal_k)
{
    const WorkspaceDesc desc = MakeLinearBudgetDesc(4096, 1, 0, 0, 0);
    const uint32_t parallelism = 1;
    RuntimeWorkspaceConfig cfg;
    cfg.stitchNumMax = 64;
    cfg.parallelism = parallelism;
    const uint64_t flatBytes = EncodedWorkspaceBytesAtStitchDepth(desc, 1, parallelism, 0, 0);
    cfg.maxWorkspaceBytes = flatBytes;
    cfg.workspaceStitchMin = flatBytes > 0 ? flatBytes - 1 : 0;
    WorkspaceDesc descCopy = desc;
    EXPECT_EQ(ResolveStitchDepthConfig(descCopy, cfg).kEff, 1u);
    descCopy = desc;
    cfg.maxWorkspaceBytes = flatBytes * 100;
    EXPECT_EQ(ResolveStitchDepthConfig(descCopy, cfg).kEff, 1u);
}

TEST_F(TestDevEncode, test_workspace_flex_io_outcast_skips_assemble_mark)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 128);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    const int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {kLoopCount * s, s}, "out");

    FUNCTION("workspace_flex_io_skip", {t0, t1, t2}, {out})
    {
        LOOP("workspace_flex_io_skip_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);
            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }

    Function* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    auto dyndev = func->GetDyndevAttribute();
    ASSERT_NE(dyndev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(dyndev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);

    const uint64_t progAddr = reinterpret_cast<uint64_t>(devProg);
    devProg->RelocProgram(0, progAddr);
    const WorkspaceDesc flex = CollectWorkspaceDesc(func, *devProg, dyndev->constructAssembleNeedAllocRuntimeSlots);
    devProg->RelocProgram(progAddr, 0);

    EXPECT_EQ(flex.totalAssembleOutcastSlot, dyndev->inoutLink.assembleSlotIndexList.size());
    EXPECT_LE(flex.totalExclusiveOutcastSlot, devProg->slotSize);
    EXPECT_LT(flex.totalAssembleOutcastSlot, dyndev->inoutLink.inputSlotIndexList.size());

    const uint32_t stitchNumMax = ConfiguredStitchFunctionMaxNum();
    WorkspaceDesc flexWs = flex;
    (void)ResolveStitchDepthConfig(flexWs, MakeNonMemoryDrivenCfg(stitchNumMax));
    EXPECT_EQ(flexWs.totalAssembleOutcastSlot, dyndev->inoutLink.assembleSlotIndexList.size());
    EXPECT_LT(flexWs.devTaskBoundaryOutcastNum, devProg->slotSize * stitchNumMax);
}

TEST_F(TestDevEncode, test_encode_multi_op_chain)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {kLoopCount * s, s}, "out");

    FUNCTION("multi_op_chain", {t0, t1}, {out})
    {
        LOOP("multi_op_chain_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto a = Add(t0, t1);
            auto b = Add(a, t0);
            auto c = Add(b, t1);
            Assemble(c, {i * s, 0}, out);
        }
    }

    auto funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    EXPECT_GT(devProg->GetFunctionSize(), 0u);

    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t ctxAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(ctxAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(ctxAddr, 0, nullptr, nullptr, nullptr,
                                                        devProg->GetParallelism());
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(ctxAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
        reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    DevAscendFunction* devFunc = devProg->GetFunction(0);
    ASSERT_NE(devFunc, nullptr);
    devFunc->Dump();
    devProg->ResetRerun();
    devProg->RuntimeVerify(0, 0);
    devProg->ResetFromLaunch();
}

TEST_F(TestDevEncode, test_encode_multi_output_assemble)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out1(DT_FP32, {kLoopCount * s, s}, "out1");
    Tensor out2(DT_FP32, {kLoopCount * s, s}, "out2");

    FUNCTION("multi_output", {t0, t1}, {out1, out2})
    {
        LOOP("multi_output_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto a = Add(t0, t1);
            auto b = Mul(t0, t1);
            Assemble(a, {i * s, 0}, out1);
            Assemble(b, {i * s, 0}, out2);
        }
    }

    auto funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    EXPECT_GT(devProg->GetFunctionSize(), 0u);
}

namespace {
ir::VarPtr MakeEncodeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetTokenType(), ir::Span::Unknown());
}

void BindToken(Function& func, Operation* producer, Operation* consumer, const std::string& name)
{
    auto token = MakeEncodeToken(name);
    auto& dep = func.GetVarDependency();
    dep.AddProducer(token, std::static_pointer_cast<const ir::Stmt>(producer->shared_from_this()));
    dep.AddConsumer(token, std::static_pointer_cast<const ir::Stmt>(consumer->shared_from_this()));
    producer->result_token_ = {token};
    consumer->tokens_.push_back(token);
}
} // namespace

TEST_F(TestDevEncode, test_add_token_depend_edges_to_color_graph)
{
    Program::GetInstance().Reset();
    config::Reset();
    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DT_FP32, {32, 32}, "input");
    Tensor output(DT_FP32, {32, 32}, "output");
    FUNCTION("TokenEdgeCallop") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenEdgeCallop");
    ASSERT_NE(func, nullptr);

    IRBuilder builder;
    (void)builder.CreateTensorOpStmt(*func, Opcode::OP_VIEW, {}, {}, ir::Span::Unknown());
    (void)builder.CreateTensorOpStmt(*func, Opcode::OP_CALL, {}, {}, ir::Span::Unknown());
    (void)builder.CreateTensorOpStmt(*func, Opcode::OP_CALL, {}, {}, ir::Span::Unknown());
    (void)builder.CreateTensorOpStmt(*func, Opcode::OP_CALL, {}, {}, ir::Span::Unknown());

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 4);
    Operation* viewOp = opList[opList.size() - 4];
    Operation* callOp0 = opList[opList.size() - 3];
    Operation* callOp1 = opList[opList.size() - 2];
    Operation* callOp2 = opList[opList.size() - 1];

    BindToken(*func, viewOp, callOp0, "tokenVC");
    BindToken(*func, callOp0, callOp1, "token01");
    BindToken(*func, callOp1, callOp2, "token12");

    std::vector<Operation*> callopList = {viewOp, callOp0, callOp1, callOp2};
    std::unordered_map<Operation*, int> callopIndexDict = {{viewOp, 0}, {callOp0, 1}, {callOp1, 2}, {callOp2, 3}};
    std::unordered_map<int, std::vector<int>> colorOutGraph;
    AddTokenDependEdgesToColorGraph(callopList, callopIndexDict, colorOutGraph);

    EXPECT_EQ(colorOutGraph.count(0), 0u);
    ASSERT_EQ(colorOutGraph.size(), 2u);
    EXPECT_EQ(colorOutGraph[1].size(), 1u);
    EXPECT_EQ(colorOutGraph[1][0], 2);
    EXPECT_EQ(colorOutGraph[2].size(), 1u);
    EXPECT_EQ(colorOutGraph[2][0], 3);
    EXPECT_EQ(colorOutGraph.count(3), 0u);
}

TEST_F(TestDevEncode, test_encode_conditional_branches)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {kLoopCount * s, s}, "out");

    FUNCTION("cond_branches", {t0, t1, t2}, {out})
    {
        LOOP("cond_branches_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto temp = Add(t0, t1);
            SymbolicScalar sel = std::ternary(i < 2, i, i + 1);
            IF(sel == i) { temp = Add(temp, t1); }
            ELSE IF(sel == i + 1) { temp = Add(temp, t2); }
            ELSE { temp = Add(temp, t0); }
            Assemble(temp, {i * s, 0}, out);
        }
    }

    auto funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);

    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t ctxAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(ctxAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(ctxAddr, 0, nullptr, nullptr, nullptr,
                                                        devProg->GetParallelism());
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(ctxAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
        reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    devProg->Dump(0, true);
    devProg->ResetRerun();
    devProg->RuntimeVerify(0, 0);
    devProg->ResetFromLaunch();
}

TEST_F(TestDevEncode, test_encode_single_iteration_loop)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");

    FUNCTION("single_iter", {t0, t1}, {out})
    {
        LOOP("single_iter_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * s, 0}, out);
        }
    }

    auto funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    EXPECT_GT(devProg->GetFunctionSize(), 0u);
}

TEST_F(TestDevEncode, test_workspace_budget_parallelism)
{
    WorkspaceDesc desc;
    desc.maxStaticOutcastMem = 8192;
    desc.totalExclusiveOutcastSlot = 2;
    desc.totalAssembleOutcastSlot = 1;

    WorkspaceDesc::WorkspacePerRootFunctionDesc profile;
    profile.unroll = 2;
    profile.rootInnerSpilledRawMem = 2048;
    profile.rootTotalExclusiveOutcastRawMem = 1024;
    desc.rootFuncDescList = {profile};

    const WorkspaceDesc wsP1 = ResolvedWorkspaceAtStitchDepth(desc, 8, 1);
    const WorkspaceDesc wsP2 = ResolvedWorkspaceAtStitchDepth(desc, 8, 2);
    EXPECT_GE(wsP2.maxRootInnerSpilledMem, wsP1.maxRootInnerSpilledMem);
}

TEST_F(TestDevEncode, test_encode_fan_out)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out1(DT_FP32, {kLoopCount * s, s}, "out1");
    Tensor out2(DT_FP32, {kLoopCount * s, s}, "out2");
    Tensor out3(DT_FP32, {kLoopCount * s, s}, "out3");

    FUNCTION("fan_out", {t0, t1}, {out1, out2, out3})
    {
        LOOP("fan_out_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto a = Add(t0, t1);
            Assemble(a, {i * s, 0}, out1);
            Assemble(a, {i * s, 0}, out2);
            Assemble(a, {i * s, 0}, out3);
        }
    }

    auto funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    EXPECT_GT(devProg->GetFunctionSize(), 0u);
    EXPECT_GE(devProg->slotSize, 3u);
}

TEST_F(TestDevEncode, test_encode_deep_chain)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {kLoopCount * s, s}, "out");

    FUNCTION("deep_chain", {t0, t1}, {out})
    {
        LOOP("deep_chain_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto a = Add(t0, t1);
            auto b = Add(a, t0);
            auto c = Add(b, t1);
            auto d = Add(c, t0);
            auto e = Add(d, t1);
            auto f = Add(e, t0);
            Assemble(f, {i * s, 0}, out);
        }
    }

    auto funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);

    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t ctxAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(ctxAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(ctxAddr, 0, nullptr, nullptr, nullptr,
                                                        devProg->GetParallelism());
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(ctxAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
        reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    devProg->Dump(0, true);
    devProg->ResetRerun();
    devProg->RuntimeVerify(0, 0);
    devProg->ResetFromLaunch();
}

TEST_F(TestDevEncode, test_encode_wide_parallel)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor t3(DT_FP32, {s, s}, "t3");
    Tensor out(DT_FP32, {kLoopCount * s, s}, "out");

    FUNCTION("wide_parallel", {t0, t1, t2, t3}, {out})
    {
        LOOP("wide_parallel_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto a = Add(t0, t1);
            auto b = Add(t2, t3);
            auto c = Mul(a, b);
            Assemble(c, {i * s, 0}, out);
        }
    }

    auto funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    EXPECT_GT(devProg->GetFunctionSize(), 0u);
}
