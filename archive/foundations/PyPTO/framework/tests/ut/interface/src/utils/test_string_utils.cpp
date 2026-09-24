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
 * \file test_string_utils.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <string>

#include "interface/utils/string_utils.h"

using namespace npu::tile_fwk;

TEST(StringUtils, basic)
{
    std::string s = "abc";
    EXPECT_TRUE(StringUtils::StartsWith(s, ""));
    EXPECT_TRUE(StringUtils::StartsWith(s, "a"));
    EXPECT_TRUE(StringUtils::StartsWith(s, "ab"));
    EXPECT_TRUE(StringUtils::StartsWith(s, "abc"));
    EXPECT_FALSE(StringUtils::StartsWith(s, "abcd"));
}
