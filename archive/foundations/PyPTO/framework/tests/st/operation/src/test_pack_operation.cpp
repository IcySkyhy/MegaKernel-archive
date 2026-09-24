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
 * \file test_pack_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {

struct PackOpFuncArgs : public OpFuncArgs {
    PackOpFuncArgs(const std::vector<int64_t>& tileShape) : tileShape_(tileShape) {}

    std::vector<int64_t> tileShape_;
};

struct PackOpMetaData {
    explicit PackOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void PackOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                 const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        auto args = static_cast<const PackOpFuncArgs*>(opArgs);
        TileShape::Current().SetVecTile(args->tileShape_);
        auto res = Pack(inputs[0]);
        Assemble(res, {0}, outputs[0]);
    }
}

class PackOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PackOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestPack, PackOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<PackOpMetaData>(
                             {PackOperationExeFunc, PackOperationExeFunc, PackOperationExeFunc}, "Pack")));

TEST_P(PackOperationTest, TestPack)
{
    auto test_data = GetParam().test_data_;
    auto args = PackOpFuncArgs(GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<PackOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace
