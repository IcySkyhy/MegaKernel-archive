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
 * \file test_derivation_tile_shape.cpp
 * \brief Unit test for DerivationTileShape pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#include "interface/operation/operation.h"
#include "passes/tensor_graph_pass/derivation_tile_shape.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class DerivationTileShapeTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ReshapeTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        TileShape::Current().SetVecTile({64, 64});
    }
    void TearDown() override {}
};

static void BuildGraphAndCheck(ComputationalGraphBuilder& G)
{
    std::vector<std::string> tensorNames{"t1"};
    std::vector<std::string> tensorNames1{"t2"};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"RESHAPE"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {8, 6}, tensorNames), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {2, 4, 6}, tensorNames1), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t2"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
}

static void BuildShapeAndCheckSucc(Function* function, const Shape& inShape, const Shape& outShape,
                                   const std::vector<int64_t>& inTileShape, const std::vector<int64_t>& resultTileShape)
{
    std::vector<int64_t> outTileShape;
    DerivationTileShape derivationTileShapePass;
    auto opList = function->Operations().DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i]->GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        auto op = opList[i];
        auto status = derivationTileShapePass.DerivationReshapeTileShape(op, inShape, outShape, inTileShape,
                                                                         outTileShape);

        EXPECT_EQ(status, SUCCESS);
        EXPECT_EQ(outTileShape, resultTileShape);
    }
}

static void BuildShapeAndCheckFail(Function* function, const Shape& inShape, const Shape& outShape,
                                   const std::vector<int64_t>& inTileShape)
{
    std::vector<int64_t> outTileShape;
    DerivationTileShape derivationTileShapePass;
    auto opList = function->Operations().DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        if (opList[i]->GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        auto op = opList[i];
        auto status = derivationTileShapePass.DerivationReshapeTileShape(op, inShape, outShape, inTileShape,
                                                                         outTileShape);

        EXPECT_EQ(status, WARNING);
    }
}

TEST_F(DerivationTileShapeTest, DerivationSplitSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {8, 6};
    Shape outShape = {2, 4, 6};
    std::vector<int64_t> inTileShape = {2, 3};
    std::vector<int64_t> resultTileShape = {1, 2, 3};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationSplitAndMergeSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {30, 6};
    Shape outShape = {2, 45, 2};
    std::vector<int64_t> inTileShape = {5, 6};
    std::vector<int64_t> resultTileShape = {1, 15, 2};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationSplitAndMergeInputShape1Success)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {30, 6, 1, 1};
    Shape outShape = {2, 45, 2};
    std::vector<int64_t> inTileShape = {5, 6, 1, 1};
    std::vector<int64_t> resultTileShape = {1, 15, 2};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationSplitAndMergeOutputShape1Success)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {30, 6};
    Shape outShape = {2, 45, 2, 1, 1};
    std::vector<int64_t> inTileShape = {5, 6};
    std::vector<int64_t> resultTileShape = {1, 15, 2, 1, 1};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationLargeSizeSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {30000, 60000};
    Shape outShape = {300, 100, 60000};
    std::vector<int64_t> inTileShape = {100, 100};
    std::vector<int64_t> resultTileShape = {1, 100, 100};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationAlignShapeFailed)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 3};
    Shape outShape = {3, 2};
    std::vector<int64_t> inTileShape = {2, 1};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationAlignTileFail)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {30, 6};
    Shape outShape = {2, 45, 2};
    std::vector<int64_t> inTileShape = {4, 2};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationOutTileFail)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {30, 6};
    Shape outShape = {2, 45, 2};
    std::vector<int64_t> inTileShape = {5, 2};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationAlignShapeInputProductFailed)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {4, 3};
    Shape outShape = {2, 3, 2};
    std::vector<int64_t> inTileShape = {2, 1};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationAlignShapeOutputProductFailed)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 3, 2};
    Shape outShape = {4, 3};
    std::vector<int64_t> inTileShape = {2, 1, 1};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationAlignShapeZeroFailed)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 3, 2};
    Shape outShape = {4, 3};
    std::vector<int64_t> inTileShape = {2, 0, 1};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationBiggerTileSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 2};
    Shape outShape = {1, 2, 2};
    std::vector<int64_t> inTileShape = {2, 32};
    std::vector<int64_t> resultTileShape = {1, 2, 32};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationSplitBiggerTileSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 4};
    Shape outShape = {1, 2, 2, 2};
    std::vector<int64_t> inTileShape = {2, 32};
    std::vector<int64_t> resultTileShape = {1, 2, 2, 16};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationMergeBiggerTileSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 2, 2};
    Shape outShape = {2, 4};
    std::vector<int64_t> inTileShape = {2, 2, 32};
    std::vector<int64_t> resultTileShape = {2, 64};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationSplitAndMergeBiggerTileSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {30, 6};
    Shape outShape = {2, 45, 2};
    std::vector<int64_t> inTileShape = {5, 36};
    std::vector<int64_t> resultTileShape = {1, 15, 12};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationMergeBiggerTileFail)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 4};
    Shape outShape = {1, 2, 2, 2};
    std::vector<int64_t> inTileShape = {2, 33};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationNonComplianceTileFail)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 27};
    Shape outShape = {2, 3, 3, 3};
    std::vector<int64_t> inTileShape = {2, 6};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationNonComplianceTileSuccess)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 27};
    Shape outShape = {2, 3, 3, 3};
    std::vector<int64_t> inTileShape = {2, 18};
    std::vector<int64_t> resultTileShape = {2, 2, 3, 3};
    BuildShapeAndCheckSucc(G.GetFunction(), inShape, outShape, inTileShape, resultTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationDiffSizeFail)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 3};
    Shape outShape = {2, 3, 3, 3};
    std::vector<int64_t> inTileShape = {2, 2, 2};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationDiffSizeFail1)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {1, 1, 1};
    Shape outShape = {1, 1, 1, 1, 2};
    std::vector<int64_t> inTileShape = {1, 1, 2};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationCheckTileShapeSizeMismatch)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 4};
    Shape outShape = {1, 2, 2, 2};
    std::vector<int64_t> inTileShape = {2, 33};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationCheckTileShapeDistanceMismatch)
{
    ComputationalGraphBuilder G;
    BuildGraphAndCheck(G);

    Shape inShape = {2, 2, 6};
    Shape outShape = {2, 4, 3};
    std::vector<int64_t> inTileShape = {2, 2, 2};
    BuildShapeAndCheckFail(G.GetFunction(), inShape, outShape, inTileShape);
}

TEST_F(DerivationTileShapeTest, DerivationReshapeTileShapeNonReshapeOp)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {8, 8}, "a");
    G.AddTensor(DataType::DT_FP32, {8, 8}, "b");
    G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add1");
    G.SetInCast({"a"});
    G.SetOutCast({"b"});
    Function* function = G.GetFunction();
    DerivationTileShape pass;
    std::vector<int64_t> outTileShape;
    auto addOp = function->Operations().DuplicatedOpList()[0];
    EXPECT_EQ(pass.DerivationReshapeTileShape(addOp, {8, 8}, {8, 8}, {4, 4}, outTileShape), WARNING);
}

} // namespace tile_fwk
} // namespace npu
