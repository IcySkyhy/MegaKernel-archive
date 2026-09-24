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
 * \file test_runtime.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <regex>
#include <iostream>
#include "interface/inner/tilefwk.h"
#include "machine/runtime/runner/runtime_agent.h"

#include "interface/configs/config_manager.h"
using namespace npu::tile_fwk;

class RuntimeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        npu::tile_fwk::Program::GetInstance().Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST(RuntimeTest, Runtime01)
{
    std::cout << "start to test runtime" << std::endl;
    std::vector<int64_t> aiv;
    std::vector<int64_t> aic;
    EXPECT_EQ(RuntimeAgent::GetAgent().GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL), 0);
}
