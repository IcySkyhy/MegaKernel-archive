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
#include "machine/runtime/runner/runtime_utils.h"

using namespace npu::tile_fwk;

TEST(RuntimeUtilsExtraTest, MemcpySWithCheck_SuccessAndOverflow)
{
    uint8_t dst[16] = {0};
    uint8_t src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    MemcpySWithCheck(dst, 16, src, 16, "test", "test.cpp", 1);
    EXPECT_EQ(memcmp(dst, src, 16), 0);

    uint8_t small[4] = {0};
    EXPECT_ANY_THROW(MemcpySWithCheck(small, 4, src, 16, "test", "test.cpp", 1));
}

TEST(RuntimeUtilsExtraTest, AlignSize_CustomAndZeroAlignment)
{
    EXPECT_EQ(AlignSize(64, 64), 64u);
    EXPECT_EQ(AlignSize(128, 64), 128u);
    EXPECT_EQ(AlignSize(1, 0), sizeof(uintptr_t));
}

TEST(RuntimeUtilsExtraTest, DeviceId_AndL2Offset_AndCaptureMode)
{
    CheckDeviceId();

    int32_t devId = GetUserDeviceId();
    (void)devId;

    uint64_t offset = GetRuntimeL2Offset();
    (void)offset;

    ExchangeCaptureModeRelax();
    ExchangeCaptureModeGlobal();
    SUCCEED();
}
