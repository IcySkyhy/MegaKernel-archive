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
 * \file test_dynamic_reshape_unalign.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "tilefwk/function.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicReshapeUnalignTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override
    {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }
};

// add dim
TEST_F(DynamicReshapeUnalignTest, test_add_dim)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(64, 64);

    int b = 2;
    int sq = 64;
    int d = 64;
    std::vector<int64_t> qShape2Dim = {b * sq, d};
    std::vector<int64_t> qShape3Dim = {b, sq, d};

    Tensor q(DT_FP32, qShape2Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 1, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape3Dim, "out");

    float inputValue = 2.0f;
    float initValue = 0.5f;

    std::vector<int> actSeqsData(b, 63);
    std::vector<float> golden(b * sq * d, initValue);
    for (int bidx = 0; bidx < b; ++bidx) {
        int offset = bidx * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[bidx] * d, exp(inputValue));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, inputValue),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {q, actSeqs}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (sq)))
        {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0, 0});
            Tensor q0 = View(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp0 = Reshape(q0, {1, sq, d}, {1, curSeq, d});
            TileShape::Current().SetVecTile(1, 64, 64);
            auto tmp = Exp(tmp0);
            Assemble(tmp, {batchId, 0, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

// merge dim, not last dim
TEST_F(DynamicReshapeUnalignTest, test_merge_dim)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(1, 16, 16);

    int b = 2;
    int sq = 10;
    int d = 10;
    std::vector<int64_t> qShape3Dim = {b, sq, d};
    std::vector<int64_t> qShape2Dim = {b, sq * d};

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 1, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape2Dim, "out");

    float inputValue = 2.0f;
    float initValue = 0.5f;

    std::vector<int> actSeqsData(b, 8);
    std::vector<float> golden(b * sq * d, initValue);
    for (int bidx = 0; bidx < b; ++bidx) {
        int offset = bidx * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[bidx] * d, exp(inputValue));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, inputValue),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {q, actSeqs}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b))
        {
            //
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0, 0});

            Tensor q0 = View(q, {1, sq, d}, {1, curSeq, d}, {batchId, 0, 0});
            auto tmp0 = Reshape(q0, {1, sq * d}, {1, curSeq * d});
            TileShape::Current().SetVecTile(1, 16);
            auto tmp = Exp(tmp0);
            Assemble(tmp, {batchId, 0}, out); // 1, sq * d -> b, sq * d
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

// split dim
TEST_F(DynamicReshapeUnalignTest, test_split_dim)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(1, 16, 16);

    int b = 2;
    int sq = 6;
    int d = 10;
    std::vector<int64_t> qShape3Dim = {b, sq, d};
    std::vector<int64_t> qShape4Dim = {b, sq, 5, d / 5};

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 2, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape4Dim, "out");

    float initValue = 0.5f;

    std::vector<int> actSeqsData = {5, 8, 5, 8};
    std::vector<float> inputValueData;
    for (int i = 0; i < b * sq * d; i++) {
        inputValueData.push_back(static_cast<float>(i));
    }

    std::vector<float> golden(b * sq * d, initValue);
    int count = 0;
    for (int bidx = 0; bidx < b; ++bidx) {
        int offset = bidx * sq * d;
        count = offset;
        for (int row = 0; row < actSeqsData[0]; row++) {
            for (int col = 0; col < actSeqsData[1]; col++) {
                if (count % d == 8)
                    count += 2;
                golden[offset + row * d + col] = count++;
            }
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, inputValueData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {q, actSeqs}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b))
        {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0, 0});
            SymbolicScalar curDim = GetTensorData(actSeqs, {batchId, 1, 0});
            Tensor q0 = View(q, {1, sq, d}, {1, curSeq, curDim}, {batchId, 0, 0});
            auto tmp0 = Reshape(q0, {1, sq, 5, d / 5}, {1, curSeq, 4, curDim / 4});
            TileShape::Current().SetVecTile(1, 16, 16, 16);

            auto tmp = Mul(tmp0, Element(tmp0.GetStorage()->Datatype(), 1.0));
            Assemble(tmp, {batchId, 0, 0, 0}, out); // 1, sq * d -> b, sq * d
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}
