/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "machine/runtime/memory_utils/emulation_memory_utils.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(EmulationMemoryUtilsTest, StaticMethods_AndAllocDev)
{
    EXPECT_FALSE(EmulationMemoryUtils::IsDevice());
    EXPECT_EQ(EmulationMemoryUtils::GetL2Offset(), 0u);

    EmulationMemoryUtils mu;
    uint8_t* ptr = mu.AllocDev(128, nullptr);
    EXPECT_NE(ptr, nullptr);

    uint8_t* holder = nullptr;
    uint8_t* ptr2 = mu.AllocDev(64, &holder);
    EXPECT_NE(ptr2, nullptr);

    EXPECT_ANY_THROW(mu.AllocDev(0, nullptr));
    EXPECT_ANY_THROW(mu.AllocDev(0xFFFFFFFFF, nullptr));
}

TEST(EmulationMemoryUtilsTest, AllocZero_AndCopyRoundtrip)
{
    EmulationMemoryUtils mu;
    uint8_t* zptr = mu.AllocZero(64, nullptr);
    EXPECT_NE(zptr, nullptr);
    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(zptr[i], 0);
    }

    uint8_t src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t* devPtr = mu.CopyToDev(src, 8, nullptr);
    EXPECT_NE(devPtr, nullptr);
    EXPECT_EQ(memcmp(devPtr, src, 8), 0);

    uint8_t dst[8] = {0};
    mu.CopyFromDev(dst, devPtr, 8);
    EXPECT_EQ(memcmp(dst, src, 8), 0);

    std::vector<int64_t> data = {100, 200, 300};
    int64_t* vecPtr = mu.CopyToDev(data, nullptr);
    EXPECT_NE(vecPtr, nullptr);
    EXPECT_EQ(vecPtr[0], 100);
    EXPECT_EQ(vecPtr[1], 200);
    EXPECT_EQ(vecPtr[2], 300);
}

TEST(EmulationMemoryUtilsTest, RawTensorData_CopyRoundtrip)
{
    EmulationMemoryUtils mu;
    std::vector<int64_t> shape = {4};
    RawTensorData data(DT_FP32, shape);
    uint8_t* hostBuf = reinterpret_cast<uint8_t*>(data.data());
    for (size_t i = 0; i < data.size(); ++i) {
        hostBuf[i] = static_cast<uint8_t>(i + 1);
    }

    uint8_t* devPtr = mu.CopyToDev(data);
    EXPECT_NE(devPtr, nullptr);
    EXPECT_EQ(devPtr, data.GetDevPtr());

    uint8_t* secondCall = mu.CopyToDev(data);
    EXPECT_EQ(secondCall, devPtr);

    RawTensorData recv(DT_FP32, shape);
    recv.SetDevPtr(data.GetDevPtr());
    mu.CopyFromDev(recv);
    EXPECT_EQ(memcmp(data.data(), recv.data(), data.size()), 0);
}
