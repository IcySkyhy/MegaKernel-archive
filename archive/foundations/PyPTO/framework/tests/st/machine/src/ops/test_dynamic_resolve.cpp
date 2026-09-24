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
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/runtime//context/device_launcher_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicResolveTest : public testing::Test {
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

TEST_F(DynamicResolveTest, TestResolve)
{
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 100 * 1024 * 1024);
    config::SetPassOption(SG_PG_LOWER_BOUND, 1024);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 32}});
    config::SetPassOption(SG_PARALLEL_NUM, 2);
    config::SetPassOption<std::map<int64_t, int64_t>>(VEC_NBUFFER_SETTING, {{-1, 16}});
    config::SetPassOption<int>(COPYOUT_RESOLVE_COALESCING, 10);

    static constexpr int v64 = 64;
    static constexpr int v128 = 128;

    TileShape::Current().SetVecTile(v64, v128);
    TileShape::Current().SetCubeTile({v64, v64}, {v128, v128}, {v128, v128});

    Tensor inputA(DT_BF16, {v64, v128}, "inputA");
    Tensor inputB(DT_BF16, {v128, v128 * v64}, "inputB");
    Tensor inputC(DT_FP32, {v64, v128 * v64}, "inputC");
    Tensor output(DT_FP32, {v64, v128 * v64}, "output");

    std::vector<bfloat16> inputBData(v128 * v128 * v64, bfloat16(0));
    for (int i = 0; i < v128 * v128 * v64; i++) {
        inputBData[i] = bfloat16(1.0 * (i % (v128 * v64) / v128));
    }
    std::vector<float> outputGolden(v64 * v128 * v64, 0);
    for (int i = 0; i < v64 * v128 * v64; i++) {
        outputGolden[i] = float(2.0 * (v128 * (i % (v128 * v64) / v128)) + 3.0);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<bfloat16>(inputA, 2.0),
        RawTensorData::CreateTensor<bfloat16>(inputB, inputBData),
        RawTensorData::CreateConstantTensor<float>(inputC, 3.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0),
    });

    FUNCTION("main", {inputA, inputB, inputC}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            std::vector<Tensor> tensorList;
            for (int j = 0; j < v64; j++) {
                auto t = View(inputB, {v128, v128}, {0, v128 * j});                  // <128 x 128 x FP32>
                auto mm = Matrix::Matmul(DataType::DT_FP32, inputA, t, false, true); // <64 x 128 x FP32>
                tensorList.emplace_back(mm);
            }
            auto mmConcat = Cat(tensorList, -1); // <64 x (128 * 64) x FP32>
            output = Add(inputC, mmConcat);
        }
    }

    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), nullptr));

#ifdef BUILD_WITH_CANN
    EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
    auto outputResult = (float*)npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0)->data();
    EXPECT_TRUE(resultCmp(outputGolden, outputResult, 0.001f));
#endif
}

} // namespace
