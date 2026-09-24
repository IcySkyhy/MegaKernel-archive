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
 * \file test_expected_value.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class ExpectedValueTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false);
    }

    void TearDown() override
    {
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    }
};

TEST_F(ExpectedValueTest, TestCheck)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c;

    FUNCTION("Expected")
    {
        config::SetPassConfig(config::GetPassStrategy(), "MergeViewAssemble", KEY_EXPECTED_VALUE_CHECK, true);
        config::SetPassConfig(config::GetPassStrategy(), "SplitReshapeOp", KEY_EXPECTED_VALUE_CHECK, true);
        config::SetPassConfig(config::GetPassStrategy(), "ConvertRegCopyPass", KEY_EXPECTED_VALUE_CHECK, true);
        c = Add(a, b);
    }
}
