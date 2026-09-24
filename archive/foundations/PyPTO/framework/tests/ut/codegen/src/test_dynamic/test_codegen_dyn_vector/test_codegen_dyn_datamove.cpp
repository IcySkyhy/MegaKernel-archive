/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_datamove.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynDataMove : public CodegenTestBase {
public:
    TestCodegenDynDataMove() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}

    static void TearDownTestCase() {}

    struct DataMoveTestCase {
        std::string caseName;
        std::vector<int64_t> inputShape;
        std::vector<int64_t> vecTile;
        DataType dtype;
        std::vector<int64_t> outputShape;
        std::vector<int32_t> perm;
        std::string expect;
    };

    void RunDataMoveTest(const DataMoveTestCase& tc)
    {
        TileShape::Current().SetVecTile(tc.vecTile);

        Tensor inputSrc(tc.dtype, tc.inputShape, "input");
        Tensor output(tc.dtype, tc.outputShape, "output");
        ConfigManager::Instance();
        FUNCTION(tc.caseName, {inputSrc, output})
        {
            LOOP(tc.caseName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
            {
                (void)i;
                output = Transpose(inputSrc, tc.perm);
            }
        }
        auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + tc.caseName + SUB_FUNC_SUFFIX +
                                                                    HIDDEN_FUNC_SUFFIX);
        std::string res = GenCodeByFunction(*function);
        CheckStringExist(tc.expect, res);
    }
};

TEST_F(TestCodegenDynDataMove, TestDatamoveUnalignDim3)
{
    RunDataMoveTest({.caseName = "TestDatamoveUnalignDim3",
                     .inputShape = {1, 32, 437},
                     .vecTile = {1, 32, 512},
                     .dtype = DT_FP32,
                     .outputShape = {32, 1, 437},
                     .perm = {0, 1},
                     .expect = R"!!!(TTransMoveOut<2, 3>(gmTensor_2, ubTensor_0, Coord3Dim(0, 0, 0));)!!!"});
}

TEST_F(TestCodegenDynDataMove, TestDatamoveUnalignDim4)
{
    RunDataMoveTest(
        {.caseName = "TestDatamoveUnalignDim4",
         .inputShape = {4, 1, 32, 437},
         .vecTile = {2, 1, 32, 512},
         .dtype = DT_FP32,
         .outputShape = {4, 32, 1, 437},
         .perm = {1, 2},
         .expect =
             R"!!!(TTransMoveOut<2, 3>(gmTensor_2, ubTensor_0, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(3, -2, 4, 18, 0)), 0, 0, 0));)!!!"});
}

TEST_F(TestCodegenDynDataMove, TestDatamoveAlignDim4)
{
    RunDataMoveTest(
        {.caseName = "TestDatamoveAlignDim4",
         .inputShape = {4, 1, 32, 437},
         .vecTile = {2, 1, 32, 512},
         .dtype = DT_FP32,
         .outputShape = {4, 32, 1, 437},
         .perm = {1, 2},
         .expect =
             R"!!!(TTransMoveOut<2, 3>(gmTensor_2, ubTensor_0, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(3, -2, 4, 18, 0)), 0, 0, 0));)!!!"});
}
} // namespace npu::tile_fwk
