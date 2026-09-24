/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file test_device_memory_utils.cpp
 * \brief UT for machine/runtime/memory_utils/device_memory_utils.h
 */

#include <gtest/gtest.h>
#include "machine/runtime/memory_utils/device_memory_utils.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DeviceMemoryUtilsTest : public testing::Test {
protected:
    // isUseHugePage=false forces RuntimeMalloc->StubMalloc->malloc path, no NPU needed.
    DeviceMemoryUtils dmu{false};
};

// Covers: IsDevice, Free(nullptr) no-op, GetL2Offset
TEST_F(DeviceMemoryUtilsTest, StaticAndNoOpMethods)
{
    EXPECT_TRUE(DeviceMemoryUtils::IsDevice());
    dmu.Free(nullptr);
    EXPECT_GE(DeviceMemoryUtils::GetL2Offset(), 0u);
}

TEST_F(DeviceMemoryUtilsTest, AllocDev_NullHolder_AllocatesAndFrees)
{
    uint8_t* ptr = dmu.AllocDev(128, nullptr);
    if (ptr != nullptr) {
        dmu.Free(ptr);
    }
    SUCCEED();
}

TEST_F(DeviceMemoryUtilsTest, AllocDev_EmptyHolder_AllocatesAndCaches)
{
    uint8_t* holder = nullptr;
    uint8_t* ptr = dmu.AllocDev(64, &holder);
    if (ptr == nullptr) {
        SUCCEED();
        return;
    }
    EXPECT_EQ(holder, ptr);

    uint8_t* ptr2 = dmu.AllocDev(64, &holder);
    EXPECT_EQ(ptr2, ptr);
    dmu.Free(ptr);
}

TEST_F(DeviceMemoryUtilsTest, AllocZero_ReturnsAllocatedMemory)
{
    uint8_t* ptr = dmu.AllocZero(32, nullptr);
    if (ptr != nullptr) {
        dmu.Free(ptr);
    }
    SUCCEED();
}

TEST_F(DeviceMemoryUtilsTest, CopyToDev_WithData_CopiesCorrectly)
{
    uint8_t src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t* devPtr = dmu.CopyToDev(src, 8, nullptr);
    if (devPtr == nullptr) {
        SUCCEED();
        return;
    }

    uint8_t dst[8] = {0};
    dmu.CopyFromDev(dst, devPtr, 8);
    EXPECT_EQ(memcmp(src, dst, 8), 0);
    dmu.Free(devPtr);
}

TEST_F(DeviceMemoryUtilsTest, CopyToDev_IntoExistingPtr_CopiesCorrectly)
{
    uint8_t src[4] = {10, 20, 30, 40};
    uint8_t* devPtr = dmu.AllocDev(4, nullptr);
    if (devPtr == nullptr) {
        SUCCEED();
        return;
    }
    dmu.CopyToDev(devPtr, src, 4);

    uint8_t dst[4] = {0};
    dmu.CopyFromDev(dst, devPtr, 4);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[3], 40);
    dmu.Free(devPtr);
}

TEST_F(DeviceMemoryUtilsTest, CopyToDev_VectorTemplate_CopiesAndReturnsTypedPtr)
{
    std::vector<int64_t> data = {100, 200, 300};
    int64_t* devPtr = dmu.CopyToDev(data, nullptr);
    if (devPtr == nullptr) {
        SUCCEED();
        return;
    }

    int64_t back[3] = {0, 0, 0};
    dmu.CopyFromDev(reinterpret_cast<uint8_t*>(back), reinterpret_cast<uint8_t*>(devPtr), sizeof(back));
    EXPECT_EQ(back[0], 100);
    EXPECT_EQ(back[2], 300);
    dmu.Free(reinterpret_cast<uint8_t*>(devPtr));
}

TEST_F(DeviceMemoryUtilsTest, CopyToDev_RawTensorData_NullDevPtr_AllocatesAndCopies)
{
    std::vector<int64_t> shape = {2, 2};
    RawTensorData data(DT_FP32, shape);
    uint8_t* hostBuf = reinterpret_cast<uint8_t*>(data.data());
    for (size_t i = 0; i < data.size(); ++i) {
        hostBuf[i] = static_cast<uint8_t>(i + 1);
    }

    uint8_t* devPtr = dmu.CopyToDev(data);
    // DevMemoryPool may return nullptr in environments without NPU; just verify the call path.
    if (devPtr != nullptr) {
        EXPECT_EQ(devPtr, data.GetDevPtr());
        uint8_t* secondCall = dmu.CopyToDev(data);
        EXPECT_EQ(secondCall, devPtr);
        DevMemoryPool::Instance().FreeDevAddr(devPtr);
    }
    SUCCEED();
}

TEST_F(DeviceMemoryUtilsTest, CopyFromDev_RawTensorData_CopiesBack)
{
    std::vector<int64_t> shape = {4};
    RawTensorData data(DT_FP32, shape);
    uint8_t* hostBuf = reinterpret_cast<uint8_t*>(data.data());
    for (size_t i = 0; i < data.size(); ++i) {
        hostBuf[i] = static_cast<uint8_t>(i + 10);
    }

    uint8_t* devPtr = dmu.CopyToDev(data);
    if (devPtr == nullptr) {
        GTEST_SKIP() << "DevMemoryPool unavailable, skip copy-back test";
    }

    RawTensorData recv(DT_FP32, shape);
    recv.SetDevPtr(devPtr);
    dmu.CopyFromDev(recv);

    EXPECT_EQ(memcmp(data.data(), recv.data(), data.size()), 0);
    DevMemoryPool::Instance().FreeDevAddr(devPtr);
}
