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
 * \file test_aot_code_pool.cpp
 * \brief Cover AOT CF pool when distinct kernels exceed pool entry capacity.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "machine/device/dynamic/aot_binary.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/context/device_launcher_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
// Device/host AOT pool has AOT_CODE_POOL_NUM entries. Run beyond capacity to exercise LRU eviction.
constexpr int kAotPoolEntries = AOT_CODE_POOL_NUM;
constexpr int kKernelCountBeyondPool = kAotPoolEntries + 4;
constexpr int kTile = 32;
} // namespace

class AotCodePoolTest : public testing::Test {
public:
    void SetUp() override
    {
        DeviceLauncherContext::Get().Initialize();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }

    void TearDown() override { DeviceLauncherContext::Get().Finalize(); }
};

namespace {

void ResetProgramState()
{
    ProgramData::GetInstance().Reset();
    Program::GetInstance().Reset();
    config::Reset();
}

int ExpectedResult(int kernelIndex)
{
    const int fillA = 1;
    const int fillB = 2 + kernelIndex;
    int value = fillA + fillB;
    for (int step = 0; step < (kernelIndex % 3); ++step) {
        (void)step;
        value += fillB;
    }
    return value;
}

// Build a unique CF binary per kernelIndex so each launch maps to a distinct AOT pool key.
void BuildUniqueAddKernel(int kernelIndex, int n)
{
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    const int fillA = 1;
    const int fillB = 2 + kernelIndex;
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, fillA),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, fillB),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    const std::string funcName = "aot_pool_main_" + std::to_string(kernelIndex);
    const std::string loopName = "aot_pool_loop_" + std::to_string(kernelIndex);
    FUNCTION(funcName, {inputA, inputB}, {output})
    {
        LOOP(loopName, FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            // Vary op chain length with kernelIndex so CF hash differs even if names collide.
            Tensor acc = Add(inputA, inputB);
            for (int step = 0; step < (kernelIndex % 3); ++step) {
                (void)step;
                acc = Add(acc, inputB);
            }
            output = acc;
        }
    }
}

void ExpectOutputEquals(int kernelIndex, int n, const char* tag)
{
    auto outputResult = ProgramData::GetInstance().GetOutputData(0);
    ASSERT_NE(outputResult, nullptr) << tag << " null output at kernelIndex=" << kernelIndex;
    std::vector<int32_t> golden(static_cast<size_t>(n) * static_cast<size_t>(n), ExpectedResult(kernelIndex));
    EXPECT_TRUE(resultCmp(golden, reinterpret_cast<int32_t*>(outputResult->data()), 0.001f))
        << tag << " precision mismatch at kernelIndex=" << kernelIndex;
}

} // namespace

TEST_F(AotCodePoolTest, ExceedPoolCapacitySequentialKernels)
{
    ASSERT_GT(kKernelCountBeyondPool, kAotPoolEntries);
    ASSERT_GT(kKernelCountBeyondPool, static_cast<int>(AOT_CODE_POOL_NUM));

    const int n = kTile * 2;
    DeviceLauncherConfig config;
    config.blockdim = 24;

    for (int kernelIndex = 0; kernelIndex < kKernelCountBeyondPool; ++kernelIndex) {
        ResetProgramState();
        TileShape::Current().SetVecTile(kTile, kTile);
        BuildUniqueAddKernel(kernelIndex, n);

        // Emulation does not execute aicore leaves (CostModel); only check rc for host AOT pool.
        EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr, config))
            << "Emulation failed at kernelIndex=" << kernelIndex;

#ifdef BUILD_WITH_CANN
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction(), nullptr, config))
            << "Device failed at kernelIndex=" << kernelIndex;
        ExpectOutputEquals(kernelIndex, n, "Device");
#endif
    }
}
