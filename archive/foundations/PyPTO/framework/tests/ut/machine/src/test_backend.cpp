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
 * \file test_backend.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/cache/function_cache.h"
#include "machine/host/backend.h"

using namespace npu::tile_fwk;

class TestSuite_Backend : public testing::Test {};

extern "C" int32_t Execute(MachineTask* task, FunctionCache& cache);

TEST_F(TestSuite_Backend, AihacBackend_Err1)
{
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    try {
        config::Reset();
    } catch (std::runtime_error&) {
    }
}

TEST_F(TestSuite_Backend, SimulationBackend_Err1)
{
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false);
    config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
    try {
        config::Reset();
    } catch (std::runtime_error&) {
    }
}

TEST_F(TestSuite_Backend, Execute_NullTask_ReturnsZero)
{
    FunctionCache cache;
    EXPECT_THROW(Execute(nullptr, cache), npu::tile_fwk::Error);
}
