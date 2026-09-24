/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_interp_loop_unroll_if.cpp
 * \brief Test case for LOOP with UnrollList [2,1] and IF ELSE with IsLoopEnd condition
 */

#include <gtest/gtest.h>
#include <set>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {
class LoopUnrollIfTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Verify not supported skip the verify test";
        }
        Program::GetInstance().Reset();
        config::Reset();
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override {}
};

TEST_F(LoopUnrollIfTest, TestLoopUnrollWithIsLoopEnd)
{
    int s = 32;
    int n = 5; // Loop length

    Tensor accum(DT_FP32, {s, s}, "accum");
    Tensor output(DT_FP32, {s, s}, "output");
    config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(accum, 0.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    // Calculate golden value:
    // Loop runs n=5 times (i = 0, 1, 2, 3, 4)
    // Last iteration (i=4): IsLoopEnd(4, 5) returns true, add 1.0
    // Other iterations (i=0,1,2,3): IsLoopEnd returns false, add 2.0 each
    // Total: 4 * 2.0 + 1 * 1.0 = 9.0
    std::vector<float> goldenData(s * s, 9.0f);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, goldenData),
    });

    FUNCTION("main", {accum}, {output})
    {
        SymbolicScalar len(n);
        // LOOP with UnrollList [2, 1]
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(len), std::set<int>{2, 1})
        {
            // IF ELSE with IsLoopEnd condition
            IF(IsLoopEnd(i, len))
            {
                // IF branch: when loop ends, add 1.0
                accum = Add(accum, Element(DataType::DT_FP32, 1.0f));
            }
            ELSE
            {
                // ELSE branch: when not loop end, add 2.0
                accum = Add(accum, Element(DataType::DT_FP32, 2.0f));
            }
        }
        // Copy accum to output
        output = Add(accum, Element(DataType::DT_FP32, 0.0f));
    }
}

} // namespace npu::tile_fwk
