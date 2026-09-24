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
 * \file test_dynamic_binry_brc.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicBrcTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(DynamicBrcTest, TestDynamicMulBrcUnalign)
{
    SetInterpreterConfig();
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    TileShape::Current().SetVecTile(32, 128);

    int b = 2;
    int sq = 32;
    int d = 72;
    std::vector<int64_t> inputShape_a = {b * sq, d};
    std::vector<int64_t> inputShape_b = {b * sq, 8};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor input_a(DT_FP32, inputShape_a, "intput_a");
    Tensor input_b(DT_FP32, inputShape_b, "intput_b");
    Tensor curSeq(DT_INT32, {b, 1}, "curSeq");
    Tensor out(DT_FP32, outShape, "out");

    std::vector<int> actSeqsData(b, 24); // 有效值应该是block对齐的
    std::vector<float> golden(b * sq * d, 0.001f);
    for (int i = 0; i < b; i++) {
        int offset = i * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[i] * d, 48.0);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input_a, 2.0),
        RawTensorData::CreateConstantTensor<float>(input_b, 3.0),
        RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });
    FUNCTION("main", {input_a, input_b, curSeq}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b))
        {
            auto seq = GetTensorData(curSeq, {batchId, 0});
            Tensor input_a0 = View(input_a, {sq, d}, {seq, d}, {batchId * sq, 0});
            Tensor input_b0 = View(input_b, {sq, 8}, {seq, 8}, {batchId * sq, 0});
            auto input_c = Sum(input_b0, -1, true);
            auto tmp = Mul(input_a0, input_c);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}
