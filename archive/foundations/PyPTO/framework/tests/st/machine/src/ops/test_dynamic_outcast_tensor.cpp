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
 * \file test_dynamic_outcast_tensor.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/runtime/context/device_launcher_context.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/emulation_launcher.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicOutcastTensorTest : public testing::Test {
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

TEST_F(DynamicOutcastTensorTest, TensorAllocateIntermediate)
{
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 100);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int round = 4;
    int width = 4;
    int n = tiling * width;
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {round * n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    std::vector<int32_t> inputBData(n * round * n, 0);
    for (int i = 0; i < round; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                inputBData[i * (n * n) + j * n + k] = i;
            }
        }
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateTensor<int32_t>(inputB, inputBData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    std::vector<int32_t> outputGolden(n * n, 1 * round + (round - 1) * round / 2);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, k, LoopRange(round))
        {
            Tensor mid(DT_INT32, {n, n}, "O");
            LOOP("s0", FunctionType::DYNAMIC_LOOP, i, LoopRange(width))
            {
                LOOP("s1", FunctionType::DYNAMIC_LOOP, j, LoopRange(width))
                {
                    Tensor t0 = View(inputA, {tiling, tiling}, {i * tiling, j * tiling});
                    Tensor t1 = View(inputB, {tiling, tiling}, {k * n + i * tiling, j * tiling});
                    Tensor ts = Add(t0, t1);
                    Assemble(ts, {i * tiling, j * tiling}, mid);
                }
            }
            LOOP("sum", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1))
            {
                (void)_;
                IF(k == 0) { output = Add(mid, Element(DT_INT32, 0)); }
                ELSE { output = Add(output, mid); }
            }
        }
    }

#ifdef BUILD_WITH_CANN
    for (int k = 0; k < 0x1; k++) {
        EXPECT_EQ(0, DeviceLauncher::DeviceRunOnce(Program::GetInstance().GetLastFunction()));
        auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
        EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
    }
#endif
}

} // namespace
