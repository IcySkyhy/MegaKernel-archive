/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <new>
#define private public
#include "machine/runtime/runner/kernel_binary.h"
#undef private

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
struct KernelBinaryTestHelper {
    void* rawMem{nullptr};
    KernelBinary* kb{nullptr};

    KernelBinaryTestHelper()
    {
        rawMem = std::malloc(sizeof(KernelBinary));
        std::memset(rawMem, 0, sizeof(KernelBinary));
        kb = reinterpret_cast<KernelBinary*>(rawMem);
        new (&kb->dynFunc) std::shared_ptr<Function>();
        new (&kb->pinnedGraph_) std::vector<std::shared_ptr<Function>>();
        new (&kb->inferShapeCaches) std::vector<ControlFlowCache>();
        new (&kb->originShapeCaches) std::vector<ControlFlowCache>();
        new (&kb->hostCtrlFlowCaches_) std::vector<HostControlFlowCache>();
        new (&kb->aicpuArgBuf) std::vector<int64_t>();
        new (&kb->argTypes) std::vector<DeviceTensorData>();
        new (&kb->dynamicCellMatchDescPatches_) std::vector<DevDynamicCellMatchStridePatch>();
        new (&kb->kernelName_) std::string();
        new (&kb->kernelArgs_) std::vector<void*>();
    }

    ~KernelBinaryTestHelper()
    {
        if (kb) {
            kb->kernelArgs_.~vector();
            kb->kernelName_.~basic_string();
            kb->dynamicCellMatchDescPatches_.~vector();
            kb->argTypes.~vector();
            kb->aicpuArgBuf.~vector();
            kb->hostCtrlFlowCaches_.~vector();
            kb->originShapeCaches.~vector();
            kb->inferShapeCaches.~vector();
            kb->pinnedGraph_.~vector();
            kb->dynFunc.~shared_ptr();
        }
        if (rawMem) {
            std::free(rawMem);
        }
    }
};
} // namespace

TEST(KernelBinaryTest, DefaultStateGetters)
{
    KernelBinaryTestHelper helper;
    EXPECT_EQ(helper.kb->GetKernelBin(), nullptr);
    EXPECT_EQ(helper.kb->GetFunction(), nullptr);
    EXPECT_TRUE(helper.kb->GetKernelname().empty());
    EXPECT_EQ(helper.kb->GetRuntimeDynamicCellMatchAddr(), 0u);
    EXPECT_EQ(helper.kb->GetRuntimeDynamicCellMatchCapacity(), 0u);
    EXPECT_EQ(helper.kb->GetSyncMode(), 0u);
    EXPECT_FALSE(helper.kb->DisableHostCtrlFlowCacheBuild());
}

TEST(KernelBinaryTest, RefReturningGetters)
{
    KernelBinaryTestHelper helper;
    auto& config = helper.kb->GetMachineConfig();
    (void)config;
    EXPECT_TRUE(helper.kb->GetHostCtrlFlowCaches().empty());
    EXPECT_TRUE(helper.kb->GetArgTypes().empty());
    auto& aicpuLaunchDesc = helper.kb->GetAicpuLaunchDesc();
    (void)aicpuLaunchDesc;
    auto& aicoreArgs = helper.kb->GetRtAicoreArgs();
    (void)aicoreArgs;
    auto& taskCfg = helper.kb->GetRtTaskCfg();
    (void)taskCfg;
    EXPECT_TRUE(helper.kb->GetKernelArgs().empty());
}

TEST(KernelBinaryTest, SetSyncMode_And_GetSyncMode)
{
    KernelBinaryTestHelper helper;
    helper.kb->SetSyncMode(1);
    EXPECT_EQ(helper.kb->GetSyncMode(), 1u);
    helper.kb->SetSyncMode(0);
    EXPECT_EQ(helper.kb->GetSyncMode(), 0u);
}

TEST(KernelBinaryTest, DisableHostCtrlFlowCacheBuild_WithDevProg)
{
    KernelBinaryTestHelper helper;
    DevAscendProgram prog{};
    prog.disableCtrlFlowCache = 1;
    helper.kb->devProg = &prog;
    EXPECT_TRUE(helper.kb->DisableHostCtrlFlowCacheBuild());
    helper.kb->devProg = nullptr;
    EXPECT_FALSE(helper.kb->DisableHostCtrlFlowCacheBuild());
}

TEST(KernelBinaryTest, ResetRuntimeDynamicCellMatchPool_AllCases)
{
    KernelBinaryTestHelper helper;
    helper.kb->ResetRuntimeDynamicCellMatchPool(false);
    helper.kb->ResetRuntimeDynamicCellMatchPool(true);

    helper.kb->runtimeDynamicCellMatchCapacity_ = 100;
    helper.kb->runtimeDynamicCellMatchHostAddr_ = 0;
    helper.kb->ResetRuntimeDynamicCellMatchPool(true);

    helper.kb->runtimeDynamicCellMatchHostAddr_ = 1;
    helper.kb->runtimeDynamicCellMatchAddr_ = 0;
    helper.kb->ResetRuntimeDynamicCellMatchPool(false);

    helper.kb->runtimeDynamicCellMatchCapacity_ = 0;
}

TEST(KernelBinaryTest, PatchHostDynamicCellMatchAddr)
{
    KernelBinaryTestHelper helper;
    helper.kb->PatchHostDynamicCellMatchAddr(nullptr);

    DevAscendProgram prog{};
    helper.kb->runtimeDynamicCellMatchHostAddr_ = 0x1234;
    helper.kb->runtimeDynamicCellMatchCapacity_ = 5678;
    helper.kb->PatchHostDynamicCellMatchAddr(&prog);
    EXPECT_EQ(prog.devArgs.dynamicCellMatchAddr, 0x1234u);
    EXPECT_EQ(prog.devArgs.dynamicCellMatchCapacity, 5678u);
}

TEST(KernelBinaryTest, FindCtrlFlowCache_EmptyInputs)
{
    KernelBinaryTestHelper helper;
    std::vector<std::vector<int64_t>> intInputs;
    EXPECT_EQ(helper.kb->FindCtrlFlowCache(intInputs, true), nullptr);
    EXPECT_EQ(helper.kb->FindCtrlFlowCache(intInputs, false), nullptr);

    std::vector<DeviceTensorData> tensorInputs;
    EXPECT_EQ(helper.kb->FindCtrlFlowCache(tensorInputs, true), nullptr);
    EXPECT_EQ(helper.kb->FindCtrlFlowCache(tensorInputs, false), nullptr);
}

TEST(KernelBinaryTest, CheckArgs)
{
    KernelBinaryTestHelper helper;
    std::vector<DeviceTensorData> empty;
    EXPECT_TRUE(helper.kb->CheckArgs(empty));

    std::vector<DeviceTensorData> nonEmpty;
    nonEmpty.push_back(DeviceTensorData());
    EXPECT_FALSE(helper.kb->CheckArgs(nonEmpty));
}
