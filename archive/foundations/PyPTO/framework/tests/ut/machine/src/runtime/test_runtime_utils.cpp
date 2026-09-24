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
 * \file test_runtime_utils.cpp
 * \brief UT for machine/runtime/runner/runtime_utils.cpp
 */

#include <gtest/gtest.h>
#include <unistd.h>
#include "machine/runtime/runner/runtime_utils.h"

using namespace npu::tile_fwk;

TEST(RuntimeUtilsTest, AlignSize_GetProcessId_RegisterKernelBin)
{
    EXPECT_EQ(AlignSize(0), 0u);
    EXPECT_EQ(AlignSize(1), 512u);
    EXPECT_EQ(AlignSize(512), 512u);
    EXPECT_EQ(AlignSize(513), 1024u);
    EXPECT_EQ(AlignSize(1, 64), 64u);
    EXPECT_EQ(AlignSize(65, 64), 128u);
    EXPECT_TRUE(AlignSize(1, 0) > 0u);

    std::vector<uint8_t> empty;
    EXPECT_ANY_THROW(RegisterKernelBin(empty));
}
