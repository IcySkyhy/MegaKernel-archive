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
 * \file test_raw_tensor_data_stride.cpp
 * \brief Unit tests for logical-vs-packed stride semantics in RawTensorData/LogicalTensorData.
 */

#include <gtest/gtest.h>
#include <vector>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {

TEST(RawTensorDataFp4RWTest, Fp4E2M1WriteThenReadBackViaCast)
{
    if (!calc::IsVerifyEnabled()) {
        GTEST_SKIP() << "Torch verifier not enabled, skip FP4 read/write cast test";
    }
    Program::GetInstance().Reset();
    config::Reset();

    // Write path: FP32 -> FP4(E2M1X2 packed)
    Tensor fp32Tensor(DT_FP32, {1, 4});
    auto srcFp32 = std::make_shared<LogicalTensorData>(
        RawTensorData::CreateTensor<float>(fp32Tensor, std::vector<float>{0.5f, 1.0f, -0.5f, -1.0f}));

    Tensor fp4Tensor(DT_FP4_E2M1, {1, 4});
    auto dstFp4 = std::make_shared<LogicalTensorData>(
        std::make_shared<RawTensorData>(fp4Tensor.GetDataType(), fp4Tensor.GetShape()));
    calc::Cast(dstFp4, srcFp32);

    // Packed bytes check (high nibble first): [0.5,1.0] -> 0x12, [-0.5,-1.0] -> 0x9A
    ASSERT_EQ(dstFp4->GetData()->size(), static_cast<size_t>(2));
    EXPECT_EQ(dstFp4->GetData()->at(0), static_cast<uint8_t>(0x12));
    EXPECT_EQ(dstFp4->GetData()->at(1), static_cast<uint8_t>(0x9A));

    // Read path: FP4(E2M1X2 packed) -> FP32
    Tensor outTensor(DT_FP32, {1, 4});
    auto outFp32 = std::make_shared<LogicalTensorData>(
        std::make_shared<RawTensorData>(outTensor.GetDataType(), outTensor.GetShape()));
    calc::Cast(outFp32, dstFp4);

    EXPECT_FLOAT_EQ(outFp32->Get<float>(0), 0.5f);
    EXPECT_FLOAT_EQ(outFp32->Get<float>(1), 1.0f);
    EXPECT_FLOAT_EQ(outFp32->Get<float>(2), -0.5f);
    EXPECT_FLOAT_EQ(outFp32->Get<float>(3), -1.0f);
}

TEST(RawTensorDataFp4RWTest, Fp4E2M1ReadWithLogicalOffset)
{
    if (!calc::IsVerifyEnabled()) {
        GTEST_SKIP() << "Torch verifier not enabled, skip FP4 offset-read test";
    }
    Program::GetInstance().Reset();
    config::Reset();

    // Full logical data [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]
    // Packed bytes (high nibble first): [0x12, 0x34, 0x9A, 0xBC]
    Tensor fp4Tensor(DT_FP4_E2M1, {1, 8});
    auto fullFp4 = std::make_shared<LogicalTensorData>(
        RawTensorData::CreateTensor<uint8_t>(fp4Tensor, std::vector<uint8_t>{0x12, 0x34, 0x9A, 0xBC}));

    // Read sub-view starting from logical offset col=2, shape [1,4]
    auto subView = std::make_shared<LogicalTensorData>(fullFp4->GetData(), std::vector<int64_t>{1, 4},
                                                       std::vector<int64_t>{1, 4}, std::vector<int64_t>{0, 2});

    Tensor outTensor(DT_FP32, {1, 4});
    auto outFp32 = std::make_shared<LogicalTensorData>(
        std::make_shared<RawTensorData>(outTensor.GetDataType(), outTensor.GetShape()));
    calc::Cast(outFp32, subView);

    EXPECT_FLOAT_EQ(outFp32->Get<float>(0), 1.5f);
    EXPECT_FLOAT_EQ(outFp32->Get<float>(1), 2.0f);
    EXPECT_FLOAT_EQ(outFp32->Get<float>(2), -0.5f);
    EXPECT_FLOAT_EQ(outFp32->Get<float>(3), -1.0f);
}

} // namespace npu::tile_fwk
