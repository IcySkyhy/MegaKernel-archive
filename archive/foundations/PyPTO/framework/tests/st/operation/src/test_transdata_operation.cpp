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
 * \file test_transdata_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct TransDataOpFuncArgs : public OpFuncArgs {
    TransDataOpFuncArgs(const int& type, const int& group, const std::vector<int64_t>& validShape,
                        const std::vector<int64_t>& viewShape, const std::vector<int64_t> tileShape)
        : type_(type), group_(group), validShape_(validShape), viewShape_(viewShape), tileShape_(tileShape)
    {}

    int type_;
    int group_;
    std::vector<int64_t> validShape_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct TransDataOpMetaData {
    explicit TransDataOpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void TransDataOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    const TransDataOpFuncArgs* transDataInfo = static_cast<const TransDataOpFuncArgs*>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        if (transDataInfo->type_ == 2) {
            Tensor tileTensor0 = View(
                inputs[0],
                {inputs[0].GetShape()[0], inputs[0].GetShape()[1], inputs[0].GetShape()[2], inputs[0].GetShape()[3]},
                {SymbolicScalar(inputs[0].GetShape()[0]), SymbolicScalar(inputs[0].GetShape()[1]),
                 SymbolicScalar(inputs[0].GetShape()[2]), SymbolicScalar(inputs[0].GetShape()[3])},
                {0, 0, 0, 0});
            TileShape::Current().SetVecTile(transDataInfo->tileShape_);
            auto res = TransData(tileTensor0, static_cast<TileOpFormat>(transDataInfo->type_), outputs[0].GetShape(),
                                 SymbolicScalar::FromConcrete(transDataInfo->validShape_), transDataInfo->group_);
            Assemble(res, {0, 0, 0, 0, 0}, outputs[0]);
        } else if (transDataInfo->type_ == 4) {
            Tensor tileTensor0 = View(
                inputs[0],
                {inputs[0].GetShape()[0], inputs[0].GetShape()[1], inputs[0].GetShape()[2], inputs[0].GetShape()[3]},
                {SymbolicScalar(inputs[0].GetShape()[0]), SymbolicScalar(inputs[0].GetShape()[1]),
                 SymbolicScalar(inputs[0].GetShape()[2]), SymbolicScalar(inputs[0].GetShape()[3])},
                {0, 0, 0, 0});
            TileShape::Current().SetVecTile(transDataInfo->tileShape_);
            auto res = TransData(tileTensor0, static_cast<TileOpFormat>(transDataInfo->type_), outputs[0].GetShape(),
                                 SymbolicScalar::FromConcrete(transDataInfo->validShape_), transDataInfo->group_);
            Assemble(res, {0, 0, 0, 0}, outputs[0]);
        }
    }
}

static void TransDataOperationExeFunc5Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    const TransDataOpFuncArgs* transDataInfo = static_cast<const TransDataOpFuncArgs*>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        if (transDataInfo->type_ == 0) {
            if (inputs[0].GetShape().size() == 5) {
                Tensor tileTensor0 = View(
                    inputs[0],
                    {inputs[0].GetShape()[0], inputs[0].GetShape()[1], inputs[0].GetShape()[2], inputs[0].GetShape()[3],
                     inputs[0].GetShape()[4]},
                    {SymbolicScalar(inputs[0].GetShape()[0]), SymbolicScalar(inputs[0].GetShape()[1]),
                     SymbolicScalar(inputs[0].GetShape()[2]), SymbolicScalar(inputs[0].GetShape()[3]),
                     SymbolicScalar(inputs[0].GetShape()[4])},
                    {0, 0, 0, 0, 0});
                TileShape::Current().SetVecTile(transDataInfo->tileShape_);
                auto res = TransData(tileTensor0, static_cast<TileOpFormat>(transDataInfo->type_),
                                     outputs[0].GetShape(), SymbolicScalar::FromConcrete(transDataInfo->validShape_),
                                     transDataInfo->group_);
                Assemble(res, {0, 0, 0, 0}, outputs[0]);
            } else {
                Tensor tileTensor0 = View(
                    inputs[0],
                    {inputs[0].GetShape()[0], inputs[0].GetShape()[1], inputs[0].GetShape()[2], inputs[0].GetShape()[3],
                     inputs[0].GetShape()[4], inputs[0].GetShape()[5]},
                    {SymbolicScalar(inputs[0].GetShape()[0]), SymbolicScalar(inputs[0].GetShape()[1]),
                     SymbolicScalar(inputs[0].GetShape()[2]), SymbolicScalar(inputs[0].GetShape()[3]),
                     SymbolicScalar(inputs[0].GetShape()[4]), SymbolicScalar(inputs[0].GetShape()[5])},
                    {0, 0, 0, 0, 0, 0});
                TileShape::Current().SetVecTile(transDataInfo->tileShape_);
                auto res = TransData(tileTensor0, static_cast<TileOpFormat>(transDataInfo->type_),
                                     outputs[0].GetShape(), SymbolicScalar::FromConcrete(transDataInfo->validShape_),
                                     transDataInfo->group_);
                Assemble(res, {0, 0, 0, 0, 0}, outputs[0]);
            }
        } else if (transDataInfo->type_ == 3) {
            Tensor tileTensor0 = View(inputs[0],
                                      {inputs[0].GetShape()[0], inputs[0].GetShape()[1], inputs[0].GetShape()[2],
                                       inputs[0].GetShape()[3], inputs[0].GetShape()[4]},
                                      {SymbolicScalar(inputs[0].GetShape()[0]), SymbolicScalar(inputs[0].GetShape()[1]),
                                       SymbolicScalar(inputs[0].GetShape()[2]), SymbolicScalar(inputs[0].GetShape()[3]),
                                       SymbolicScalar(inputs[0].GetShape()[4])},
                                      {0, 0, 0, 0, 0});
            TileShape::Current().SetVecTile(transDataInfo->tileShape_);
            auto res = TransData(tileTensor0, static_cast<TileOpFormat>(transDataInfo->type_), outputs[0].GetShape(),
                                 SymbolicScalar::FromConcrete(transDataInfo->validShape_), transDataInfo->group_);
            Assemble(res, {0, 0, 0, 0, 0, 0}, outputs[0]);
        } else if (transDataInfo->type_ == 5) {
            Tensor tileTensor0 = View(inputs[0],
                                      {inputs[0].GetShape()[0], inputs[0].GetShape()[1], inputs[0].GetShape()[2],
                                       inputs[0].GetShape()[3], inputs[0].GetShape()[4]},
                                      {SymbolicScalar(inputs[0].GetShape()[0]), SymbolicScalar(inputs[0].GetShape()[1]),
                                       SymbolicScalar(inputs[0].GetShape()[2]), SymbolicScalar(inputs[0].GetShape()[3]),
                                       SymbolicScalar(inputs[0].GetShape()[4])},
                                      {0, 0, 0, 0, 0});
            TileShape::Current().SetVecTile(transDataInfo->tileShape_);
            auto res = TransData(tileTensor0, static_cast<TileOpFormat>(transDataInfo->type_), outputs[0].GetShape(),
                                 SymbolicScalar::FromConcrete(transDataInfo->validShape_), transDataInfo->group_);
            Assemble(res, {0, 0, 0, 0}, outputs[0]);
        }
    }
}

static void TransDataOperationExeFunc6Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                           const OpFuncArgs* opArgs)
{
    const TransDataOpFuncArgs* transDataInfo = static_cast<const TransDataOpFuncArgs*>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]})
    {
        if (transDataInfo->type_ == 0) {
            Tensor tileTensor0 = View(
                inputs[0],
                {inputs[0].GetShape()[0], inputs[0].GetShape()[1], inputs[0].GetShape()[2], inputs[0].GetShape()[3],
                 inputs[0].GetShape()[4], inputs[0].GetShape()[5]},
                {SymbolicScalar(inputs[0].GetShape()[0]), SymbolicScalar(inputs[0].GetShape()[1]),
                 SymbolicScalar(inputs[0].GetShape()[2]), SymbolicScalar(inputs[0].GetShape()[3]),
                 SymbolicScalar(inputs[0].GetShape()[4]), SymbolicScalar(inputs[0].GetShape()[5])},
                {0, 0, 0, 0, 0, 0});
            TileShape::Current().SetVecTile(transDataInfo->tileShape_);
            auto res = TransData(tileTensor0, static_cast<TileOpFormat>(transDataInfo->type_), outputs[0].GetShape(),
                                 SymbolicScalar::FromConcrete(transDataInfo->validShape_), transDataInfo->group_);
            Assemble(res, {0, 0, 0, 0, 0}, outputs[0]);
        }
    }
}

class TransDataOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TransDataOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestTransData, TransDataOperationTest,
                         ::testing::ValuesIn(GetOpMetaData<TransDataOpMetaData>(
                             {TransDataOperationExeFunc4Dims, TransDataOperationExeFunc4Dims,
                              TransDataOperationExeFunc4Dims, TransDataOperationExeFunc5Dims,
                              TransDataOperationExeFunc6Dims},
                             "TransData")));

TEST_P(TransDataOperationTest, TestTransData)
{
    auto test_data = GetParam().test_data_;
    int type = GetValueByName<int>(test_data, "type");
    int group = GetValueByName<int>(test_data, "group");
    std::vector<int> tmpValidShape = GetValueByName<std::vector<int>>(test_data, "valid_shape");
    std::vector<int64_t> validShape = {tmpValidShape.begin(), tmpValidShape.end()};

    auto args = TransDataOpFuncArgs(type, group, validShape, GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<TransDataOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
