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
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicBinTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(DynamicBinTest, TestDynamicAddUnalign)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(64, 64);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> inputShape = {b * sq, d};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor input1(DT_FP32, inputShape, "intput1");
    Tensor input2(DT_FP32, inputShape, "intput2");
    Tensor curSeq(DT_INT32, {b, 1}, "curSeq");
    Tensor out(DT_FP32, outShape, "out");

    std::vector<int> actSeqsData(b, 100);
    std::vector<float> golden(b * sq * d, 0.001f);
    for (int i = 0; i < b; i++) {
        int offset = i * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[i] * d, 2.0);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input1, 1.0),
        RawTensorData::CreateConstantTensor<float>(input2, 1.0),
        RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {input1, input2, curSeq}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b))
        {
            auto seq = GetTensorData(curSeq, {batchId, 0});
            Tensor intput11 = View(input1, {sq, d}, {seq, d}, {batchId, 0});
            Tensor intput22 = View(input2, {sq, d}, {seq, d}, {batchId, 0});
            auto tmp = Add(intput11, intput22);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

TEST_F(DynamicBinTest, testDynMulsUnalign)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(64, 64);

    std::vector<uint8_t> devProgBinary;

    int b = 4;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> qShape = {b * sq, d};   // input_src0 shape: {512, 64}
    std::vector<int64_t> outShape = {b * sq, d}; // output shape: {4*128, 64}

    Tensor q(DT_FP32, qShape, "q");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(DT_FP32, outShape, "out");

    std::vector<int> actSeqsData(b, 100);
    std::vector<float> golden(b * sq * d, 0.001f);
    for (int bidx = 0; bidx < b; ++bidx) {
        for (int seq = 0; seq < actSeqsData[bidx]; ++seq) {
            for (int dim = 0; dim < d; ++dim) {
                int idx = bidx * sq * d + seq * d + dim;
                golden[idx] = 1.0f;
            }
        }
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, 1.0),
        RawTensorData::CreateTensor<int>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {q, actSeqs}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (sq)))
        {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0});
            Element value(DataType::DT_FP32, 1.0);
            Tensor q0 = View(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp = Mul(q0, value);

            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}
