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
 * \file test_split_reshape_pvc2.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class TestSplitReshapeOpPVC2 : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(TestSplitReshapeOpPVC2, Test_Reshape_1to1)
{
    Function* currentFunction = nullptr;

    TileShape::Current().SetVecTile(8, 8, 8, 8);
    Tensor input(DT_FP32, {8, 16, 16}, "a");
    Tensor res1;

    FUNCTION("Test_Reshape_1to1")
    {
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {8, 16, 1, 16});
        res1 = Exp(test);
        currentFunction = Program::GetInstance().GetCurrentFunction();
    }
    EXPECT_NE(currentFunction, nullptr);
    std::vector<int64_t> expiInShape = {8, 8, 8};
    std::vector<int64_t> expOutShape = {8, 8, 1, 8};
    for (auto& op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->shape, expiInShape);
            }

            for (auto& out : op.oOperand) {
                EXPECT_EQ(out->shape, expOutShape);
            }
        }
    }
}

TEST_F(TestSplitReshapeOpPVC2, Test_Reshape_1toMulti)
{
    Function* currentFunction = nullptr;

    TileShape::Current().SetVecTile(8, 8, 8, 8, 8);
    Tensor input(DT_FP32, {16, 4, 4}, "a");
    Tensor res1;

    FUNCTION("Test_Reshape_1toMulti")
    {
        auto res = Exp(input);
        auto test = Reshape(res, {16, 16});
        res1 = Exp(test);
        currentFunction = Program::GetInstance().GetCurrentFunction();
    }
    EXPECT_NE(currentFunction, nullptr);
    std::vector<int64_t> expiInShape = {8, 4, 4};
    std::vector<int64_t> expOutShape = {8, 16};
    for (auto& op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->shape, expiInShape);
            }

            for (auto& out : op.oOperand) {
                EXPECT_EQ(out->shape, expOutShape);
            }
        }
    }
}
