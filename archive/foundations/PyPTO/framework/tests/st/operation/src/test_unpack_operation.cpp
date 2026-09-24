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
 * \file test_unpack_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

struct UnPackOpFuncArgs : public OpFuncArgs {
    UnPackOpFuncArgs(const std::vector<int64_t>& tileShape) : tileShape_(tileShape) {}

    std::vector<int64_t> tileShape_;
};

struct UnPackOpMetaData {
    explicit UnPackOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void UnPackOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                   const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        auto args = static_cast<const UnPackOpFuncArgs*>(opArgs);
        TileShape::Current().SetVecTile(args->tileShape_);
        DataType dstDType = outputs[0].GetDataType();
        auto res = UnPack(inputs[0], dstDType);
        Assemble(res, {0}, outputs[0]);
    }
}

class UnPackOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<UnPackOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestUnPack, UnPackOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<UnPackOpMetaData>({UnPackOperationExeFunc}, "UnPack")));

TEST_P(UnPackOperationTest, TestUnPack)
{
    auto test_data = GetParam().test_data_;
    auto args = UnPackOpFuncArgs(GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<UnPackOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace
