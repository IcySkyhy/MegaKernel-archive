/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "adapter/api/runtime_api.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class CellMatchDynamicTest : public testing::Test {};

TEST_F(CellMatchDynamicTest, ResetRuntimeDynamicCellMatchPoolHost_AllCases)
{
    ResetRuntimeDynamicCellMatchPoolHost(0, 1024, false);

    uint64_t dummy = 0;
    ResetRuntimeDynamicCellMatchPoolHost(reinterpret_cast<uint64_t>(&dummy), 0, false);

    constexpr size_t kNumWords = 16;
    uint64_t table[kNumWords];
    std::memset(table, 0, sizeof(table));
    ResetRuntimeDynamicCellMatchPoolHost(reinterpret_cast<uint64_t>(table), kNumWords * sizeof(uint64_t), false);
    for (size_t i = 0; i < kNumWords; ++i) {
        EXPECT_EQ(table[i], AICORE_TASK_INIT);
    }
}

TEST_F(CellMatchDynamicTest, ResetRuntimeDynamicCellMatchPoolHost_DevicePath)
{
    auto& pool = DevMemoryPool::Instance();
    uint8_t* devAddr = nullptr;
    pool.AllocDevAddr(&devAddr, 4096);
    if (devAddr == nullptr) {
        GTEST_SKIP() << "Failed to allocate device memory";
    }

    ResetRuntimeDynamicCellMatchPoolHost(reinterpret_cast<uint64_t>(devAddr), 4096, true);

    uint64_t hostCheck[512];
    auto copyRet = NormalizedRtMemcpy(hostCheck, 4096, devAddr, 4096, RtMemcpyKind::DEVICE_TO_HOST);
    EXPECT_EQ(copyRet, RT_SUCCESS);

    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(hostCheck[i], AICORE_TASK_INIT);
    }

    pool.FreeDevAddr(devAddr);
}

TEST_F(CellMatchDynamicTest, PatchHostDynamicCellMatchTableDesc_AllCases)
{
    std::vector<DevDynamicCellMatchStridePatch> patches;
    patches.push_back({});
    PatchHostDynamicCellMatchTableDesc(nullptr, patches);

    DevAscendProgram prog{};
    std::vector<DevDynamicCellMatchStridePatch> emptyPatches;
    PatchHostDynamicCellMatchTableDesc(&prog, emptyPatches);

    alignas(64) uint8_t buffer[4096];
    std::memset(buffer, 0, sizeof(buffer));
    auto* progPtr = reinterpret_cast<DevAscendProgram*>(buffer);
    size_t descOffset = offsetof(DevAscendProgram, devArgs);
    DevDynamicCellMatchStridePatch patch;
    patch.descOffset = descOffset;
    patch.stride.dimStride[0] = 42;
    patch.stride.dimStride[1] = 99;
    std::vector<DevDynamicCellMatchStridePatch> validPatches = {patch};
    PatchHostDynamicCellMatchTableDesc(progPtr, validPatches);
}

TEST_F(CellMatchDynamicTest, WriteDynamicCellMatchStridePatchesToLaunchArgs_AllCases)
{
    std::vector<DevDynamicCellMatchStridePatch> patches;
    patches.push_back({});
    WriteDynamicCellMatchStridePatchesToLaunchArgs(nullptr, patches);

    int64_t launchInputs[64];
    std::memset(launchInputs, 0, sizeof(launchInputs));
    launchInputs[0] = 2;
    launchInputs[1] = 2;
    std::vector<DevDynamicCellMatchStridePatch> emptyPatches;
    WriteDynamicCellMatchStridePatchesToLaunchArgs(launchInputs, emptyPatches);
    auto* patchCountPtr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<DevTensorData*>(launchInputs + DEV_TENSOR_DATA_OFFSET) + 2 + 2);
    EXPECT_EQ(*patchCountPtr, 0u);

    int64_t launchInputs2[128];
    std::memset(launchInputs2, 0, sizeof(launchInputs2));
    launchInputs2[0] = 1;
    launchInputs2[1] = 1;
    DevDynamicCellMatchStridePatch patch;
    patch.descOffset = 0x1234;
    patch.stride.dimStride[0] = 10;
    patch.stride.dimStride[1] = 20;
    std::vector<DevDynamicCellMatchStridePatch> validPatches = {patch};
    WriteDynamicCellMatchStridePatchesToLaunchArgs(launchInputs2, validPatches);
    auto* patchCountPtr2 = reinterpret_cast<uint64_t*>(
        reinterpret_cast<DevTensorData*>(launchInputs2 + DEV_TENSOR_DATA_OFFSET) + 1 + 1);
    EXPECT_EQ(*patchCountPtr2, 1u);
    auto* patchArr = reinterpret_cast<DevDynamicCellMatchStridePatch*>(patchCountPtr2 + 1);
    EXPECT_EQ(patchArr[0].descOffset, 0x1234u);
    EXPECT_EQ(patchArr[0].stride.dimStride[0], 10u);
    EXPECT_EQ(patchArr[0].stride.dimStride[1], 20u);
}

TEST_F(CellMatchDynamicTest, ValidateDynamicCellMatchTableMemBudget_NullProg)
{
    DyndevFunctionAttribute dynAttr;
    ValidateDynamicCellMatchTableMemBudget(dynAttr, nullptr);
}

TEST_F(CellMatchDynamicTest, ValidateDynamicCellMatchTableMemBudget_EmptySlots)
{
    DyndevFunctionAttribute dynAttr;
    DevAscendProgram prog{};
    ValidateDynamicCellMatchTableMemBudget(dynAttr, &prog);
}

TEST_F(CellMatchDynamicTest, ValidateDynamicCellMatchTableMemBudget_WithLaunchMeta)
{
    DyndevFunctionAttribute dynAttr;

    DyndevFunctionAttribute::DynamicCellMatchLaunchMeta meta;
    meta.slotIndex = 0;
    meta.descOffset = offsetof(DevAscendProgram, devArgs);
    meta.cellShape = {32, 32};
    dynAttr.dynamicCellMatchLaunchMetaList.push_back(meta);

    alignas(64) uint8_t buffer[8192];
    std::memset(buffer, 0, sizeof(buffer));
    auto* prog = reinterpret_cast<DevAscendProgram*>(buffer);

    ValidateDynamicCellMatchTableMemBudget(dynAttr, prog);
}

TEST_F(CellMatchDynamicTest, RefillDynamicMemBudgets_NullProg)
{
    DyndevFunctionAttribute dynAttr;
    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    RefillDynamicMemBudgets(nullptr, dynAttr, eval);
}

TEST_F(CellMatchDynamicTest, RefillDynamicMemBudgets_NoValidExpr)
{
    DyndevFunctionAttribute dynAttr;
    DevAscendProgram prog{};

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    RefillDynamicMemBudgets(&prog, dynAttr, eval);
}

TEST_F(CellMatchDynamicTest, PrepareDynamicCellMatchDescPatches_EmptyList)
{
    DyndevFunctionAttribute dynAttr;

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    auto patches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    EXPECT_TRUE(patches.empty());
}

TEST_F(CellMatchDynamicTest, PrepareDynamicCellMatchDescPatches_WithLaunchMeta)
{
    DyndevFunctionAttribute dynAttr;

    DyndevFunctionAttribute::DynamicCellMatchLaunchMeta meta;
    meta.slotIndex = 0;
    meta.descOffset = 100;
    meta.cellShape = {16, 16};

    std::vector<SymbolicScalar> candidateDim;
    candidateDim.push_back(SymbolicScalar(32));
    candidateDim.push_back(SymbolicScalar(32));
    meta.candidateRawDims.push_back(candidateDim);

    dynAttr.dynamicCellMatchLaunchMetaList.push_back(meta);

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    auto patches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    EXPECT_EQ(patches.size(), 1u);
    EXPECT_EQ(patches[0].descOffset, 100u);
}

TEST_F(CellMatchDynamicTest, WriteDynamicCellMatchStridePatchesToLaunchArgs_MultiplePatches)
{
    int64_t launchInputs[256];
    std::memset(launchInputs, 0, sizeof(launchInputs));
    launchInputs[0] = 2;
    launchInputs[1] = 2;

    std::vector<DevDynamicCellMatchStridePatch> patches;
    for (int i = 0; i < 3; ++i) {
        DevDynamicCellMatchStridePatch patch;
        patch.descOffset = 100 * (i + 1);
        patch.stride.dimStride[0] = 10 * (i + 1);
        patch.stride.dimStride[1] = 20 * (i + 1);
        patches.push_back(patch);
    }

    WriteDynamicCellMatchStridePatchesToLaunchArgs(launchInputs, patches);

    auto* patchCountPtr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<DevTensorData*>(launchInputs + DEV_TENSOR_DATA_OFFSET) + 2 + 2);
    EXPECT_EQ(*patchCountPtr, 3u);

    auto* patchArr = reinterpret_cast<DevDynamicCellMatchStridePatch*>(patchCountPtr + 1);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(patchArr[i].descOffset, static_cast<uint64_t>(100 * (i + 1)));
        EXPECT_EQ(patchArr[i].stride.dimStride[0], static_cast<uint64_t>(10 * (i + 1)));
        EXPECT_EQ(patchArr[i].stride.dimStride[1], static_cast<uint64_t>(20 * (i + 1)));
    }
}

TEST_F(CellMatchDynamicTest, PatchHostDynamicCellMatchTableDesc_MultiplePatches)
{
    alignas(64) uint8_t buffer[8192];
    std::memset(buffer, 0, sizeof(buffer));
    auto* prog = reinterpret_cast<DevAscendProgram*>(buffer);

    std::vector<DevDynamicCellMatchStridePatch> patches;
    for (int i = 0; i < 3; ++i) {
        DevDynamicCellMatchStridePatch patch;
        patch.descOffset = offsetof(DevAscendProgram, devArgs) + i * 64;
        patch.stride.dimStride[0] = 10 * (i + 1);
        patch.stride.dimStride[1] = 20 * (i + 1);
        patches.push_back(patch);
    }

    PatchHostDynamicCellMatchTableDesc(prog, patches);
}
