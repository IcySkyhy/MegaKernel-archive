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
 * \file test_op_info_manager.cpp
 * \brief Unit tests for OpInfoManager singleton (tiling keys, op type, control buffer, bin handles)
 */

#include <cstdint>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "interface/utils/op_info_manager.h"

using namespace npu::tile_fwk;

class TestOpInfoManager : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestOpInfoManager, GetOpTilingKeyDefault)
{
    auto& mgr = OpInfoManager::GetInstance();
    EXPECT_EQ(mgr.GetOpTilingKey(), DEFAULT_OP_TILING_KEY);
    EXPECT_EQ(mgr.GetOpTilingKey(), 0UL);
}

TEST_F(TestOpInfoManager, SetAndGetOpType)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.SetOpType("custom_op");
    EXPECT_EQ(mgr.GetOpType(), "custom_op");
    mgr.SetOpType("PyPTO");
    EXPECT_EQ(mgr.GetOpType(), "PyPTO");
}

TEST_F(TestOpInfoManager, GetCustomJsonDefault)
{
    auto& mgr = OpInfoManager::GetInstance();
    auto& json = mgr.GetCustomJson();
    EXPECT_EQ(json.size(), 1u);
    EXPECT_EQ(json[0], '0');
}

TEST_F(TestOpInfoManager, GetCustomOpJsonPathDefault)
{
    auto& mgr = OpInfoManager::GetInstance();
    std::string& path = mgr.GetCustomOpJsonPath();
    EXPECT_EQ(path, "");
}

TEST_F(TestOpInfoManager, SetAndGetOpFuncName)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.GetOpFuncName() = "my_func";
    EXPECT_EQ(mgr.GetOpFuncName(), "my_func");
    mgr.GetOpFuncName() = "";
}

TEST_F(TestOpInfoManager, GetControlBinHandleNotFound)
{
    auto& mgr = OpInfoManager::GetInstance();
    void* result = mgr.GetControlBinHandle("/nonexistent/path.so");
    EXPECT_EQ(result, nullptr);
}

TEST_F(TestOpInfoManager, SetAndGetControlBinHandle)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.GetCustomOpJsonPath() = "/test/path.so";
    int dummy = 0;
    void* handle = &dummy;
    mgr.SetControlBinHandle(handle);
    void* result = mgr.GetControlBinHandle("/test/path.so");
    EXPECT_NE(result, nullptr);
    mgr.GetCustomOpJsonPath() = "";
}
