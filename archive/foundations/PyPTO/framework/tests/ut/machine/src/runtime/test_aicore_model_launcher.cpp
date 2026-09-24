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
 * \file test_aicore_model_launcher.cpp
 * \brief UT for machine/runtime/launcher/aicore_model_launcher.cpp and eslmodel_launcher.cpp
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/launcher/aicore_model_launcher.h"
#include "machine/runtime/launcher/eslmodel_launcher.h"
#undef private
#include "machine/simulation/aicore_hardware.h"
#include "tilefwk/aicpu_common.h"
#include "interface/program/program.h"
#include "interface/interpreter/raw_tensor_data.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(AicoreModelLauncherTest, MemoryUtils_AllocAndCopy)
{
    AicoreModelMemoryUtils utils;

    EXPECT_ANY_THROW(utils.AllocDev(0, nullptr));
    EXPECT_ANY_THROW(utils.AllocDev(0x500000001ULL, nullptr));
    EXPECT_ANY_THROW(utils.AllocZero(0, nullptr));

    uint8_t* ptr = utils.AllocDev(128, nullptr);
    EXPECT_NE(ptr, nullptr);

    uint8_t* zeroPtr = utils.AllocZero(256, nullptr);
    ASSERT_NE(zeroPtr, nullptr);
    for (int i = 0; i < 256; i++) {
        EXPECT_EQ(zeroPtr[i], 0);
    }

    uint8_t data[] = {1, 2, 3, 4};
    uint8_t* devPtr = utils.CopyToDev(data, 4, nullptr);
    ASSERT_NE(devPtr, nullptr);
    EXPECT_EQ(devPtr[0], 1);
    EXPECT_EQ(devPtr[3], 4);

    uint8_t src[] = {10, 20, 30};
    uint8_t dst[3] = {0};
    utils.CopyFromDev(dst, src, 3);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 20);
    EXPECT_EQ(dst[2], 30);

    std::vector<int> vecData = {100, 200, 300};
    int* vecPtr = utils.CopyToDev(vecData, nullptr);
    ASSERT_NE(vecPtr, nullptr);
    EXPECT_EQ(vecPtr[0], 100);
    EXPECT_EQ(vecPtr[1], 200);
    EXPECT_EQ(vecPtr[2], 300);
}

TEST(AicoreModelLauncherTest, MemoryUtils_Properties)
{
    AicoreModelMemoryUtils utils;
    EXPECT_EQ(utils.GetL2Offset(), 0u);
    EXPECT_FALSE(AicoreModelMemoryUtils::IsDevice());
}

TEST(EslModelLauncherTest, CopyInputOutputData_NoData_NoCrash)
{
    ProgramData::GetInstance().Reset();
    EslModelLauncher::CopyInputOutputData();
    SUCCEED();
}
