/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_stitch_cellmatch_bound.cpp
 * \brief ST cases for cellMatchTable fill/read range
 */

#include <gtest/gtest.h>

#include "tilefwk/function.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicStitchCellMatchBoundTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override
    {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }
};

namespace {
// Build a linearly-numbered input and the expected golden after `opCount`
// successive `+0.01f` ops. The host-side data shape is taken from `in.GetShape()`
// and the same flat buffer is the golden of `out` because Reshape/View do not
// change element values.
std::vector<float> PrepareLinearInputAndGolden(const Tensor& in, const Tensor& out, int opCount)
{
    std::vector<int64_t> shape = in.GetShape();
    size_t elemCnt = 1;
    for (int64_t s : shape) {
        elemCnt *= static_cast<size_t>(s);
    }

    std::vector<float> inputData(elemCnt, 0.0f);
    std::vector<float> golden(elemCnt, 0.0f);
    for (size_t i = 0; i < elemCnt; ++i) {
        inputData[i] = static_cast<float>(i);
        golden[i] = static_cast<float>(i) + 0.01f * opCount;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(in, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });
    return golden;
}
} // namespace

/*
 * Case: tail-iteration validShape < declared shape.
 *   q [b=2, sq=-1(18), d=8] --L1--> addTmp[b, sq, d]
 *                           --Reshape--> qReshape[b*sq, d] = [36, d]
 *                           --L2--> out[36, d], offSet=32
 */
TEST_F(DynamicStitchCellMatchBoundTest, test_cellmatch_tail_valid_shape)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(1, 4, 32);
    config::SetPassConfig("PVC2_OOO", "SplitReshape", KEY_DISABLE_PASS, true);

    int b = 2;
    int sq = -1;
    int d = 8;
    int bSq = (sq == -1) ? -1 : b * sq;
    std::vector<int64_t> qShape3Dim = {b, sq, d};
    std::vector<int64_t> qShape2Dim = {bSq, d};

    Tensor q(DT_FP32, qShape3Dim, "q");
    Tensor out(DT_FP32, qShape2Dim, "out");

    sq = 18;
    Tensor qReal(DT_FP32, {b, sq, d});
    Tensor outReal(DT_FP32, {b * sq, d});
    std::vector<float> golden = PrepareLinearInputAndGolden(qReal, outReal, 2);

    FUNCTION("main", {q}, {out})
    {
        Tensor addTmp(DT_FP32, {b, GetInputShape(q, 1), d});
        LOOP("L1", FunctionType::DYNAMIC_LOOP, sqIdx, LoopRange(GetInputShape(q, 1)))
        {
            Tensor viewTmp = View(q, {b, 1, d}, {0, sqIdx, 0});
            auto res = Add(viewTmp, Element(viewTmp.GetStorage()->Datatype(), 0.01));
            Assemble(res, {0, sqIdx, 0}, addTmp);
        }

        Tensor qReshape(DT_FP32, {GetInputShape(addTmp, 0) * GetInputShape(addTmp, 1), d});
        LOOP("ReshapeInplace", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1))
        {
            (void)idx;
            qReshape = Reshape(addTmp, {GetInputShape(addTmp, 0) * GetInputShape(addTmp, 1), d}, true);
        }

        int offSet = 32;
        LOOP("L2", FunctionType::DYNAMIC_LOOP, loopIdx,
             LoopRange((GetInputShape(addTmp, 0) * GetInputShape(addTmp, 1) + offSet - 1) / offSet))
        {
            TileShape::Current().SetVecTile(4, 32);
            Tensor tmp0 = View(qReshape, {offSet, d},
                               {min(GetInputShape(addTmp, 0) * GetInputShape(addTmp, 1) - loopIdx * offSet, offSet), d},
                               {loopIdx * offSet, 0});
            Tensor tmp = Add(tmp0, Element(tmp0.GetStorage()->Datatype(), 0.01));
            Assemble(tmp, {loopIdx * offSet, 0}, out);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(qReal.GetStorage()->GetDataSize()));
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}
