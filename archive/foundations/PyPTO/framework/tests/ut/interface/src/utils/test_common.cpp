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
 * \file test_common.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <string>

#include "interface/utils/common.h"

using namespace npu::tile_fwk;

TEST(CommonTest, OrderedMap)
{
    std::vector<int> key;
    std::vector<int> val;
    for (int i = 0x10; i > 1; i--) {
        key.push_back(i);
        val.push_back(i + 1);
    }

    OrderedMap<int, int> m;
    for (size_t i = 0; i < key.size(); i++) {
        m[key[i]] = val[i];
    }

    std::vector<int> mkey;
    std::vector<int> mval;
    for (auto& [k, v] : m) {
        mkey.push_back(k);
        mval.push_back(v);
    }
    EXPECT_EQ(key, mkey);
    EXPECT_EQ(val, mval);
}

TEST(CommonTest, GetBacktrace)
{
    auto str = GetBacktrace(1, 0x8)->Get();
    EXPECT_TRUE(str.find("tile_fwk_utest") != std::string::npos);
}
