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
 * \file test_function_utils.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class FunctionUtilsTest : public testing::Test {
public:
    static void SetUpTestCase()
    {
        std::cout << "FunctionUtilsTest SetUpTestCase" << std::endl;
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    static void TearDownTestCase() { std::cout << "FunctionUtilsTest TearDownTestCase" << std::endl; }

    void SetUp() override
    {
        std::cout << "FunctionUtilsTest SetUp" << std::endl;
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        std::cout << "FunctionUtilsTest TearDown" << std::endl;
        Program::GetInstance().Reset();
    }
};

TEST_F(FunctionUtilsTest, TestCloneOperation)
{
    std::vector<int64_t> shape{8, 16};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    TileShape::Current().SetVecTile(shape);
    config::SetBuildStatic(true);
    FUNCTION("main") { output = Add(input, Element(DT_FP32, 1.0)); }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_main");
    ASSERT_NE(func, nullptr);
    for (const auto& op : func->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ADDS) {
            std::vector<std::shared_ptr<LogicalTensor>> ioperands;
            std::vector<std::shared_ptr<LogicalTensor>> ooperands;
            for (auto iOperand : op.GetIOperands()) {
                std::shared_ptr<LogicalTensor> tensor = iOperand->Clone(*func, true);
                ioperands.push_back(tensor);
            }
            for (auto oOperand : op.GetOOperands()) {
                std::shared_ptr<LogicalTensor> tensor = oOperand->Clone(*func, true);
                ooperands.push_back(tensor);
            }
            Operation& opClone = op.CloneOperation(*func, ioperands, ooperands);
            EXPECT_EQ(op.GetOOperands()[0]->GetShape(), opClone.GetOOperands()[0]->GetShape());
            break;
        }
    }
}
