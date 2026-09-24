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
 * \file test_memory_attribute.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "machine/runtime/launcher/device_launcher_types.h"
#include "tilefwk/data_type.h"

TEST(TestMemoryAttribute, MemorySizeTest)
{
    std::vector<std::vector<int64_t>> tshapes = {{1, 1},     {1, 10},    {5, 10},     {32, 64},
                                                 {100, 100}, {255, 255}, {256, 1024}, {1024, 2048}};
    std::vector<npu::tile_fwk::DataType> dtypes = {
        npu::tile_fwk::DT_INT4, npu::tile_fwk::DT_INT8, npu::tile_fwk::DT_INT16, npu::tile_fwk::DT_INT32,
        npu::tile_fwk::DT_FP8,  npu::tile_fwk::DT_FP16, npu::tile_fwk::DT_FP32,  npu::tile_fwk::DT_BF16,
        npu::tile_fwk::DT_HF8,  npu::tile_fwk::DT_HF4};
    for (auto& tshape : tshapes) {
        for (auto dt : dtypes) {
            auto numel = tshape[0] * tshape[1];
            if (numel * BitsOf(dt) % 0x8 != 0) {
                continue;
            }
            npu::tile_fwk::Tensor A(dt, tshape, "A_" + std::string(DataType2String(dt)));
            A.GetStorage()->SetMemoryTypeToBe(npu::tile_fwk::MEM_UB);
            auto dataSize = DataSizeOf(numel, dt);
            EXPECT_EQ(A.GetStorage()->MemorySize(), (dataSize + 31) / 32 * 32);

            A.GetStorage()->SetMemoryTypeToBe(npu::tile_fwk::MEM_L2);
            EXPECT_EQ(A.GetStorage()->MemorySize(), dataSize);
        }
    }
}

TEST(TestMemoryAttribute, DeviceTensorPackedDataSize)
{
    npu::tile_fwk::dynamic::DeviceTensorData tensor(npu::tile_fwk::DT_FP4_E2M1, nullptr, {2, 4});
    EXPECT_EQ(tensor.GetDataSize(), 4);
}
