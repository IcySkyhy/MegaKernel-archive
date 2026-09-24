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
 * \file test_dynamic_bin.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime//context/stream_context.h"
#include "machine/runtime//context/device_launcher_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

static constexpr int tiling32 = 32;

class DynamicBindingTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        DeviceLauncherContext::Get().Initialize();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }

    void TearDown() override { DeviceLauncherContext::Get().Finalize(); }
};

namespace {

TEST_F(DynamicBindingTest, TestDefaultCompute)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(tiling32, tiling32);
    TileShape::Current().SetCubeTile({tiling32, tiling32}, {tiling32, tiling32}, {tiling32, tiling32});

    int n = 1 * tiling32;
    int m = 2 * tiling32;

    std::vector<int32_t> inputAData(n * m, 0);
    std::vector<int32_t> inputBData(n * m, 0);
    std::vector<int32_t> outputData(n * m, 0);
    std::vector<int32_t> outputGolden(n * m, 0);
    for (int i = 0; i < n * m; i++) {
        inputAData[i] = i;
        inputBData[i] = i * 2;
        outputGolden[i] = i * 3;
    }

    Tensor inputA(DT_INT32, {n, m}, "inputA");
    Tensor inputB(DT_INT32, {n, m}, "inputB");
    Tensor output(DT_INT32, {n, m}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<int32_t>(inputB, inputBData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(output, outputData),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(m / tiling32))
        {
            auto tmpA = View(inputA, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpB = View(inputB, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpO = Add(tmpA, tmpB);
            Assemble(tmpO, {0, i * tiling32}, output);
        }
    }

#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBindingTest, TestDeviceRunDataFromHost)
{
    SetInterpreterConfig();
    int n = 2 * tiling32;

    std::vector<int32_t> inputData(n * n, 0);
    std::vector<int32_t> outputData(n * n, 0);
    std::vector<int32_t> outputGolden(n * n, 0);
    for (int i = 0; i < n * n; i++) {
        inputData[i] = i;
        outputGolden[i] = i * 11;
    }

    Tensor input(DT_INT32, {n, n}, "input");
    Tensor output(DT_INT32, {n, n}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(input, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(output, outputData),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    TileShape::Current().SetVecTile(tiling32, tiling32);
    FUNCTION("main", {input}, {output})
    {
        LOOP("s0", FunctionType::DYNAMIC_LOOP, k, LoopRange(10))
        {
            IF(k == 0) { output = Add(input, input); }
            ELSE { output = Add(input, output); }
        }
    }

#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBindingTest, TestDeviceCompute)
{
    SetInterpreterConfig();
    AclInit(nullptr);
    RuntimeSetDevice(GetDeviceIdByEnvVar());

    TileShape::Current().SetVecTile(tiling32, tiling32);
    TileShape::Current().SetCubeTile({tiling32, tiling32}, {tiling32, tiling32}, {tiling32, tiling32});

    int n = 1 * tiling32;
    int m = 2 * tiling32;
    uint8_t* inputADevAddr = nullptr;
    uint8_t* inputBDevAddr = nullptr;
    uint8_t* outputDevAddr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&inputADevAddr, n * m * sizeof(int32_t));
    DevMemoryPool::Instance().AllocDevAddr(&inputBDevAddr, n * m * sizeof(int32_t));
    DevMemoryPool::Instance().AllocDevAddr(&outputDevAddr, n * m * sizeof(int32_t));

    std::vector<int32_t> inputAData(n * m, 0);
    std::vector<int32_t> inputBData(n * m, 0);
    std::vector<int32_t> outputData(n * m, 0);
    std::vector<int32_t> outputGolden(n * m, 0);
    for (int i = 0; i < n * m; i++) {
        inputAData[i] = i;
        inputBData[i] = i * 2;
        outputGolden[i] = i * 3;
    }

    RuntimeMemcpy(inputADevAddr, inputAData.size() * sizeof(int32_t), inputAData.data(),
                  inputAData.size() * sizeof(int32_t), RtMemcpyKind::HOST_TO_DEVICE);
    RuntimeMemcpy(inputBDevAddr, inputBData.size() * sizeof(int32_t), inputBData.data(),
                  inputAData.size() * sizeof(int32_t), RtMemcpyKind::HOST_TO_DEVICE);

    Tensor inputA(DT_INT32, {n, m}, "inputA");
    Tensor inputB(DT_INT32, {n, m}, "inputB");
    Tensor output(DT_INT32, {n, m}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<int32_t>(inputB, inputBData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(output, outputData),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    ExportedOperator* op = ExportedOperatorBegin();

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(m / tiling32))
        {
            auto tmpA = View(inputA, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpB = View(inputB, {tiling32, tiling32}, {0, i * tiling32});
            auto tmpO = Add(tmpA, tmpB);
            Assemble(tmpO, {0, i * tiling32}, output);
        }
    }

    ExportedOperatorEnd(op);

    std::vector<DeviceTensorData> inputList = {
        DeviceTensorData(inputA.GetDataType(), inputADevAddr, inputA.GetShape()),
        DeviceTensorData(inputB.GetDataType(), inputBDevAddr, inputB.GetShape()),
    };
    std::vector<DeviceTensorData> outputList = {
        DeviceTensorData(output.GetDataType(), outputDevAddr, output.GetShape()),
    };

    auto aicoreStream = reinterpret_cast<DeviceStream>(GetStreamContext().GetAiCoreStream());
    EXPECT_EQ(0, ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(op, inputList, outputList, aicoreStream, true));

    RuntimeMemcpy(outputData.data(), outputData.size() * sizeof(int32_t), outputDevAddr,
                  outputData.size() * sizeof(int32_t), RtMemcpyKind::DEVICE_TO_HOST);

    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputData.data(), 0.001f));
}
} // namespace
