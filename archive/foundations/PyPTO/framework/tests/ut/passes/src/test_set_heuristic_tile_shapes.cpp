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
 * \file test_set_heuristic_tile_shapes.cpp
 * \brief Unit test for SetHeuristicTileShapes.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/set_heuristic_tile_shapes.h"
#include "computational_graph_builder.h"
#include "ut_json/ut_json_tool.h"
#include "interface/tensor/irbuilder.h"
#include "interface/operation/operation_impl.h"

namespace npu::tile_fwk {

void RunPassAndVerifySuccess(Function& function)
{
    SetHeuristicTileShapes pass;
    auto status = pass.RunOnFunction(function);
    EXPECT_EQ(status, SUCCESS);
}

void PushInOutCasts(Function& function, const std::vector<std::shared_ptr<LogicalTensor>>& inCasts,
                    const std::vector<std::shared_ptr<LogicalTensor>>& outCasts)
{
    for (auto& t : inCasts)
        function.inCasts_.push_back(t);
    for (auto& t : outCasts)
        function.outCasts_.push_back(t);
}

struct GatherMatmulSetup {
    Operation& gatherOp;
    std::shared_ptr<LogicalTensor> gatherOut;
    std::shared_ptr<LogicalTensor> matmulBIn;
    std::shared_ptr<LogicalTensor> matmulOut;
};

GatherMatmulSetup SetupGatherInL1WithMatmul(Function& function, std::shared_ptr<LogicalTensor> input,
                                            std::shared_ptr<LogicalTensor> offsets,
                                            std::shared_ptr<LogicalTensor> blockTable,
                                            const std::vector<int64_t>& shapeOut,
                                            const std::vector<int64_t>& matmulBShape,
                                            const std::vector<int64_t>& matmulOutShape, bool isB, bool isTrans,
                                            bool matmulUsesGatherAsFirstInput = true)
{
    auto gatherOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeOut, CreateTestConstIntVector(shapeOut));
    auto matmulBIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                                CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    Operation& gatherOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_GATHER_IN_L1, {input, offsets, blockTable}, {gatherOut}, [isB, isTrans](Operation& op) {
            op.SetAttribute("isB", isB);
            op.SetAttribute("isTrans", isTrans);
            op.SetAttribute(OpAttributeKey::startOffset, 0);
        });

    if (matmulUsesGatherAsFirstInput) {
        PassOperationUtils::AddOperation(function, Opcode::OP_A_MUL_B, {gatherOut, matmulBIn}, {matmulOut});
        PushInOutCasts(function, {input, offsets, blockTable, matmulBIn}, {matmulOut});
    } else {
        PassOperationUtils::AddOperation(function, Opcode::OP_A_MUL_B, {matmulBIn, gatherOut}, {matmulOut});
        PushInOutCasts(function, {matmulBIn, input, offsets, blockTable}, {matmulOut});
    }

    return {gatherOp, gatherOut, matmulBIn, matmulOut};
}

std::string VectorToString(const std::vector<int64_t>& vec)
{
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); i++) {
        result += std::to_string(vec[i]);
        if (i < vec.size() - 1)
            result += ", ";
    }
    result += "]";
    return result;
}

class TestSetHeuristicTileShapes : public ::testing::Test {
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

TEST_F(TestSetHeuristicTileShapes, TestCube)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSetHeuristicTileShapes",
                                                      "TestSetHeuristicTileShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inputAShape = {64, 128};
    std::vector<int64_t> inputBShape = {128, 64};
    std::vector<int64_t> outputCShape = {64, 64};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputAShape,
                                                             CreateTestConstIntVector(inputAShape));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputBShape,
                                                             CreateTestConstIntVector(inputBShape));
    auto outputC = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputCShape,
                                                              CreateTestConstIntVector(outputCShape));

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA, inputB},
                                                      {outputC});

    PushInOutCasts(*currFunctionPtr, {inputA, inputB}, {outputC});

    RunPassAndVerifySuccess(*currFunctionPtr);

    // Verify cube tile values match full shape dimensions
    auto tileShape = matmulOp.GetTileShape();
    auto cubeTile = tileShape.GetCubeTile();

    EXPECT_EQ(cubeTile.m[0], 64) << "cubeTile.m should match shape M dimension";
    EXPECT_EQ(cubeTile.k[0], 128) << "cubeTile.k should match shape K dimension";
    EXPECT_EQ(cubeTile.n[0], 64) << "cubeTile.n should match shape N dimension";
}
TEST_F(TestSetHeuristicTileShapes, TestGatherInL1TileSetting)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherInL1", "TestGatherInL1",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {128, 64};
    std::vector<int64_t> shapeB = {1, 128};
    std::vector<int64_t> shapeOffsets = {1, 128};
    std::vector<int64_t> outputShape = {128, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {128, 64};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeA, CreateTestConstIntVector(shapeA));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeB, CreateTestConstIntVector(shapeB));
    inputB->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(128)});
    auto offsets = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shapeOffsets,
                                                              CreateTestConstIntVector(shapeOffsets));
    offsets->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(128)});
    auto gatherOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputShape,
                                                                CreateTestConstIntVector(outputShape));
    auto matmulBIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                                CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    auto& gatherOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_GATHER_IN_L1,
                                                      {inputA, offsets, inputB}, {gatherOut}, [](Operation& op) {
                                                          op.SetAttribute("isB", false);
                                                          op.SetAttribute("isTrans", false);
                                                          op.SetAttribute(OpAttributeKey::startOffset, 0);
                                                      });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {gatherOut, matmulBIn}, {matmulOut});

    PushInOutCasts(*currFunctionPtr, {inputA, offsets, inputB, matmulBIn}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = gatherOp.GetTileShape();
    auto vecTile = tileShape.GetVecTile().tile;
    EXPECT_TRUE(vecTile.size() >= 2);
    EXPECT_EQ(vecTile[0], 128) << "vecTile[0] should match output M dimension";
    EXPECT_EQ(vecTile[1], 128) << "vecTile[1] should match output K dimension";

    auto cubeTile = tileShape.GetCubeTile();
    EXPECT_EQ(cubeTile.m[1], 128) << "cubeTile.m[1] should match vecTile[0]";
    EXPECT_EQ(cubeTile.k[1], 128) << "cubeTile.k[1] should match vecTile[1]";
    EXPECT_EQ(cubeTile.n[1], -1) << "cubeTile.n[1] should be -1 for non-B gather";
}

TEST_F(TestSetHeuristicTileShapes, TestTileFilteringWithMatmul)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestTileFilter", "TestTileFilter",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> assembleInput = {16, 128};
    std::vector<int64_t> assembleOutput = {128, 128};
    std::vector<int64_t> matmulB = {128, 64};
    std::vector<int64_t> matmulOutput = {128, 64};

    auto assembleIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, assembleInput,
                                                                 CreateTestConstIntVector(assembleInput));
    auto assembleOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, assembleOutput,
                                                                  CreateTestConstIntVector(assembleOutput));
    auto matmulBIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulB, CreateTestConstIntVector(matmulB));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutput,
                                                                CreateTestConstIntVector(matmulOutput));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assembleIn}, {assembleOut});
    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {assembleOut, matmulBIn},
                                                      {matmulOut});

    PushInOutCasts(*currFunctionPtr, {assembleIn, matmulBIn}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = matmulOp.GetTileShape();
    auto cubeTile = tileShape.GetCubeTile();

    EXPECT_EQ(cubeTile.m[0], 128) << "cubeTile.m should match matmul M dimension";
    EXPECT_EQ(cubeTile.k[0], 128) << "cubeTile.k should match K dimension";
    EXPECT_EQ(cubeTile.n[0], 64) << "cubeTile.n should match N dimension";
}

TEST_F(TestSetHeuristicTileShapes, TestTransposeVariants)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestTranspose", "TestTranspose",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {32, 128};
    std::vector<int64_t> shapeBT = {64, 128};
    std::vector<int64_t> outputC = {32, 64};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeA, CreateTestConstIntVector(shapeA));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeBT, CreateTestConstIntVector(shapeBT));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputC, CreateTestConstIntVector(outputC));

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_BT, {inputA, inputB},
                                                      {output});
    matmulOp.SetAttribute(Matrix::A_MUL_B_TRANS_A, false);
    matmulOp.SetAttribute(Matrix::A_MUL_B_TRANS_B, true);

    PushInOutCasts(*currFunctionPtr, {inputA, inputB}, {output});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = matmulOp.GetTileShape();
    auto cubeTile = tileShape.GetCubeTile();

    EXPECT_EQ(cubeTile.m[0], 32) << "cubeTile.m should match output M dimension";
    EXPECT_EQ(cubeTile.k[0], 128) << "cubeTile.k should match K dimension";
    EXPECT_EQ(cubeTile.n[0], 64) << "cubeTile.n should match output N dimension";
}

TEST_F(TestSetHeuristicTileShapes, TestEmptyTileFallback)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestEmptyFallback", "TestEmptyFallback",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {4, 8};
    std::vector<int64_t> shapeB = {8, 16};
    std::vector<int64_t> outputC = {4, 16};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeA, CreateTestConstIntVector(shapeA));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, shapeB, CreateTestConstIntVector(shapeB));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP16, outputC, CreateTestConstIntVector(outputC));

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA, inputB}, {output});

    PushInOutCasts(*currFunctionPtr, {inputA, inputB}, {output});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = matmulOp.GetTileShape();
    auto cubeTile = tileShape.GetCubeTile();

    EXPECT_EQ(cubeTile.m[0], 16) << "cubeTile.m should be minimum tile size 16";
    EXPECT_EQ(cubeTile.k[0], 16) << "cubeTile.k should be minimum tile size 16";
    EXPECT_EQ(cubeTile.n[0], 16) << "cubeTile.n should be minimum tile size 16";
}

TEST_F(TestSetHeuristicTileShapes, TestReshapeChainWithMatmulEnd)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeMatmulEnd",
                                                      "TestReshapeMatmulEnd", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inputShape = {1024, 64};
    std::vector<int64_t> reshape1Shape = {-1, 64};
    std::vector<int64_t> reshape2Shape = {512, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {512, 64};

    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    auto reshape1Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape1Shape,
                                                                  CreateTestConstIntVector(reshape1Shape));
    auto reshape2Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape2Shape,
                                                                  CreateTestConstIntVector(reshape2Shape));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    auto& reshape1Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {input}, {reshape1Out});
    auto& reshape2Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshape1Out},
                                                        {reshape2Out});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {reshape2Out, matmulB}, {matmulOut});

    PushInOutCasts(*currFunctionPtr, {input, matmulB}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto reshape1Tile = reshape1Op.GetTileShape();
    auto reshape1VecTile = reshape1Tile.GetVecTile().tile;
    EXPECT_EQ(reshape1VecTile.size(), 2);
    EXPECT_EQ(reshape1VecTile[0], 256) << "reshape1 vectile[0] should be 256";
    EXPECT_EQ(reshape1VecTile[1], 64) << "reshape1 vectile[1] should be 64";

    auto reshape2Tile = reshape2Op.GetTileShape();
    auto reshape2VecTile = reshape2Tile.GetVecTile().tile;
    EXPECT_EQ(reshape2VecTile.size(), 2);
    EXPECT_EQ(reshape2VecTile[0], 256) << "reshape2 vectile[0] should be 256";
    EXPECT_EQ(reshape2VecTile[1], 64) << "reshape2 vectile[1] should be 64";
}

TEST_F(TestSetHeuristicTileShapes, TestReshapeChainWithMatmulStart)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeMatmulStart",
                                                      "TestReshapeMatmulStart", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> matmulAShape = {1024, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {1024, 64};
    std::vector<int64_t> reshape1Shape = {-1, 64};
    std::vector<int64_t> reshape2Shape = {512, 128};
    std::vector<int64_t> expOutShape = {512, 128};

    auto matmulA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulAShape,
                                                              CreateTestConstIntVector(matmulAShape));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));
    auto reshape1Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape1Shape,
                                                                  CreateTestConstIntVector(reshape1Shape));
    auto reshape2Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape2Shape,
                                                                  CreateTestConstIntVector(reshape2Shape));
    auto expOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, expOutShape,
                                                             CreateTestConstIntVector(expOutShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulOut});

    auto& reshape1Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {matmulOut},
                                                        {reshape1Out});
    auto& reshape2Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshape1Out},
                                                        {reshape2Out});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {reshape2Out}, {expOut});

    PushInOutCasts(*currFunctionPtr, {matmulA, matmulB}, {expOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto reshape1Tile = reshape1Op.GetTileShape();
    auto reshape1VecTile = reshape1Tile.GetVecTile().tile;
    EXPECT_EQ(reshape1VecTile.size(), 2);
    EXPECT_EQ(reshape1VecTile[0], 128) << "reshape1 vectile[0] should be 1024";
    EXPECT_EQ(reshape1VecTile[1], 64) << "reshape1 vectile[1] should be 64";

    auto reshape2Tile = reshape2Op.GetTileShape();
    auto reshape2VecTile = reshape2Tile.GetVecTile().tile;
    EXPECT_EQ(reshape2VecTile.size(), 2);
    EXPECT_EQ(reshape2VecTile[0], 128) << "reshape2 vectile[0] should be 512";
    EXPECT_EQ(reshape2VecTile[1], 64) << "reshape2 vectile[1] should be 128";
}

TEST_F(TestSetHeuristicTileShapes, TestMatmulWithAddAfter)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMatmulAddAfter",
                                                      "TestMatmulAddAfter", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> matmulAShape = {32, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {32, 64};
    std::vector<int64_t> viewInShape = {32, 64};
    std::vector<int64_t> viewOutShape = {32, 64};
    std::vector<int64_t> addOutShape = {32, 64};

    auto matmulA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulAShape,
                                                              CreateTestConstIntVector(matmulAShape));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));
    auto viewIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, viewInShape,
                                                             CreateTestConstIntVector(viewInShape));
    auto viewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, viewOutShape,
                                                              CreateTestConstIntVector(viewOutShape));
    auto addOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, addOutShape,
                                                             CreateTestConstIntVector(addOutShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulOut});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewIn}, {viewOut});

    auto& addOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {matmulOut, viewOut}, {addOut});

    PushInOutCasts(*currFunctionPtr, {matmulA, matmulB, viewIn}, {addOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto addTile = addOp.GetTileShape();
    auto addVecTile = addTile.GetVecTile().tile;
    EXPECT_EQ(addVecTile.size(), 2);
    EXPECT_EQ(addVecTile[0], 32) << "ADD vectile[0] should be 32";
    EXPECT_EQ(addVecTile[1], 64) << "ADD vectile[1] should be 64";
}

TEST_F(TestSetHeuristicTileShapes, TestAddBeforeMatmul)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddBeforeMatmul",
                                                      "TestAddBeforeMatmul", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> view1InShape = {32, 128};
    std::vector<int64_t> view1OutShape = {32, 128};
    std::vector<int64_t> view2InShape = {32, 128};
    std::vector<int64_t> view2OutShape = {32, 128};
    std::vector<int64_t> addOutShape = {32, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {32, 64};

    auto view1In = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1InShape,
                                                              CreateTestConstIntVector(view1InShape));
    auto view1Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1OutShape,
                                                               CreateTestConstIntVector(view1OutShape));
    auto view2In = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view2InShape,
                                                              CreateTestConstIntVector(view2InShape));
    auto view2Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view2OutShape,
                                                               CreateTestConstIntVector(view2OutShape));
    auto addOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, addOutShape,
                                                             CreateTestConstIntVector(addOutShape));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {view1In}, {view1Out});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {view2In}, {view2Out});

    auto& addOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {view1Out, view2Out}, {addOut});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {addOut, matmulB}, {matmulOut});

    PushInOutCasts(*currFunctionPtr, {view1In, view2In, matmulB}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto addTile = addOp.GetTileShape();
    auto addVecTile = addTile.GetVecTile().tile;
    EXPECT_EQ(addVecTile.size(), 2);
    EXPECT_EQ(addVecTile[0], 32) << "ADD vectile[0] should be 32";
    EXPECT_EQ(addVecTile[1], 128) << "ADD vectile[1] should be 128";
}

TEST_F(TestSetHeuristicTileShapes, TestMoveOperations)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMoveOps", "TestMoveOps", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> matmulAShape = {64, 64};
    std::vector<int64_t> matmulBShape = {64, 128};
    std::vector<int64_t> matmulOutShape = {64, 128};
    std::vector<int64_t> convertedShape = {64, 128};
    std::vector<int64_t> outputShape = {64, 128};

    auto matmulA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulAShape,
                                                              CreateTestConstIntVector(matmulAShape));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));
    auto converted = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, convertedShape,
                                                                CreateTestConstIntVector(convertedShape));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputShape,
                                                             CreateTestConstIntVector(outputShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulA, matmulB}, {matmulOut});

    auto& convertOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_CONVERT, {matmulOut}, {converted});

    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {converted}, {output});

    PushInOutCasts(*currFunctionPtr, {matmulA, matmulB}, {output});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto convertTile = convertOp.GetTileShape();
    auto convertVecTile = convertTile.GetVecTile().tile;
    EXPECT_EQ(convertVecTile.size(), 2);
    EXPECT_EQ(convertVecTile[0], 64) << "Convert vectile[0] should be 64";
    EXPECT_EQ(convertVecTile[1], 128) << "Convert vectile[1] should be 128";

    auto expTile = expOp.GetTileShape();
    auto expVecTile = expTile.GetVecTile().tile;
    EXPECT_EQ(expVecTile.size(), 2);
    EXPECT_EQ(expVecTile[0], 64) << "EXP vectile[0] should be 64";
    EXPECT_EQ(expVecTile[1], 128) << "EXP vectile[1] should be 128";
}

TEST_F(TestSetHeuristicTileShapes, TestMinimalShapeFinding)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMinimalShape", "TestMinimalShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> view1Input = {256, 128};
    std::vector<int64_t> view1Output = {16, 128};
    std::vector<int64_t> assembleInput = {16, 128};
    std::vector<int64_t> assembleOutput = {64, 512};
    std::vector<int64_t> view2Input = {64, 512};
    std::vector<int64_t> view2Output = {64, 512};
    std::vector<int64_t> matmulA = {64, 512};
    std::vector<int64_t> matmulB = {512, 64};
    std::vector<int64_t> matmulOutput = {64, 64};

    auto input1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1Input, CreateTestConstIntVector(view1Input));
    auto ub1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1Output, CreateTestConstIntVector(view1Output));
    auto ub2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, assembleOutput,
                                                          CreateTestConstIntVector(assembleOutput));
    auto ub3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view2Output, CreateTestConstIntVector(view2Output));
    auto matmulAIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulA, CreateTestConstIntVector(matmulA));
    auto matmulBIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulB, CreateTestConstIntVector(matmulB));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutput,
                                                                CreateTestConstIntVector(matmulOutput));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {input1}, {ub1});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub1}, {ub2});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ub2}, {ub3});

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {matmulAIn, matmulBIn},
                                                      {matmulOut});

    PushInOutCasts(*currFunctionPtr, {input1, matmulAIn, matmulBIn}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = matmulOp.GetTileShape();
    auto cubeTile = tileShape.GetCubeTile();

    EXPECT_EQ(cubeTile.m[0], 64) << "cubeTile.m should be heuristic minimal 16";
    EXPECT_EQ(cubeTile.k[0], 256) << "cubeTile.k should be heuristic minimal 128";
    EXPECT_EQ(cubeTile.n[0], 64) << "cubeTile.n should be heuristic minimal 64";
}

TEST_F(TestSetHeuristicTileShapes, TestMatmulTransposeAT)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMatmulTransposeAT",
                                                      "TestMatmulTransposeAT", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {128, 64};
    std::vector<int64_t> shapeB = {128, 32};
    std::vector<int64_t> outputC = {64, 32};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeA, CreateTestConstIntVector(shapeA));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeB, CreateTestConstIntVector(shapeB));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputC, CreateTestConstIntVector(outputC));

    Operation& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_AT_MUL_B, {inputA, inputB},
                                                           {output});

    PushInOutCasts(*currFunctionPtr, {inputA, inputB}, {output});

    RunPassAndVerifySuccess(*currFunctionPtr);

    // Check matmul tiles - OP_AT_MUL_B: A^T[128,64] × B[128,32] = C[64,32]
    auto matmulTile = matmulOp.GetTileShape();
    auto cubeTile = matmulTile.GetCubeTile();
    auto matmulVecTile = matmulTile.GetVecTile().tile;

    // cubeTile: {m=128, k=64, n=32}
    // vecTile: [128, 64] (INPUT A^T tile matching cubeTile {m, k})
    EXPECT_EQ(cubeTile.m[0], 128) << "cubeTile.m should be 128";
    EXPECT_EQ(cubeTile.k[0], 64) << "cubeTile.k should be 64";
    EXPECT_EQ(cubeTile.n[0], 32) << "cubeTile.n should be 32";

    EXPECT_EQ(matmulVecTile.size(), 2) << "Matmul input A^T is 2D";
    EXPECT_EQ(matmulVecTile[0], 128) << "vecTile[0] should match cubeTile.m";
    EXPECT_EQ(matmulVecTile[1], 64) << "vecTile[1] should match cubeTile.k";
}

TEST_F(TestSetHeuristicTileShapes, TestMatmulTransposeBoth)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMatmulTransposeBoth",
                                                      "TestMatmulTransposeBoth", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {128, 64};
    std::vector<int64_t> shapeB = {32, 128};
    std::vector<int64_t> outputC = {64, 32};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeA, CreateTestConstIntVector(shapeA));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeB, CreateTestConstIntVector(shapeB));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputC, CreateTestConstIntVector(outputC));

    Operation& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_AT_MUL_BT, {inputA, inputB},
                                                           {output});

    PushInOutCasts(*currFunctionPtr, {inputA, inputB}, {output});

    RunPassAndVerifySuccess(*currFunctionPtr);

    // Check matmul tiles - OP_AT_MUL_BT: A^T[128,64] × B^T[32,128] = C[64,32]
    auto matmulTile = matmulOp.GetTileShape();
    auto cubeTile = matmulTile.GetCubeTile();
    auto matmulVecTile = matmulTile.GetVecTile().tile;

    // cubeTile: {m=128, k=64, n=128}
    // vecTile: [128, 64] (INPUT A^T tile matching cubeTile {m, k})
    EXPECT_EQ(cubeTile.m[0], 128) << "cubeTile.m should be 128";
    EXPECT_EQ(cubeTile.k[0], 64) << "cubeTile.k should be 64";
    EXPECT_EQ(cubeTile.n[0], 128) << "cubeTile.n should be 128";

    EXPECT_EQ(matmulVecTile.size(), 2) << "Matmul input A^T is 2D";
    EXPECT_EQ(matmulVecTile[0], 128) << "vecTile[0] should match cubeTile.m";
    EXPECT_EQ(matmulVecTile[1], 64) << "vecTile[1] should match cubeTile.k";
}

TEST_F(TestSetHeuristicTileShapes, TestMulaccOperations)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMulacc", "TestMulacc", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {64, 128};
    std::vector<int64_t> shapeB = {128, 64};
    std::vector<int64_t> shapeC = {64, 64};
    std::vector<int64_t> outputD = {64, 64};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeA, CreateTestConstIntVector(shapeA));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeB, CreateTestConstIntVector(shapeB));
    auto inputC = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeC, CreateTestConstIntVector(shapeC));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputD, CreateTestConstIntVector(outputD));

    Operation& mulaccOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MULACC_B,
                                                           {inputA, inputB, inputC}, {output});

    PushInOutCasts(*currFunctionPtr, {inputA, inputB, inputC}, {output});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = mulaccOp.GetTileShape();
    auto cubeTile = tileShape.GetCubeTile();

    EXPECT_EQ(cubeTile.m[0], 64) << "cubeTile.m should be 64";
    EXPECT_EQ(cubeTile.k[0], 128) << "cubeTile.k should be 128";
    EXPECT_EQ(cubeTile.n[0], 64) << "cubeTile.n should be 64";
}

TEST_F(TestSetHeuristicTileShapes, TestShapeAndTypeSetting_GatherMK)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherMK", "TestGatherMK", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeOut = {128, 64};
    std::vector<int64_t> shapeIn = {128, 64};
    std::vector<int64_t> shapeOffsets = {1, 128};
    std::vector<int64_t> shapeBlockTable = {1, 128};
    std::vector<int64_t> matmulBShape = {64, 32};
    std::vector<int64_t> matmulOutShape = {128, 32};

    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeIn, CreateTestConstIntVector(shapeIn));
    auto offsets = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shapeOffsets,
                                                              CreateTestConstIntVector(shapeOffsets));
    offsets->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(128)});
    auto blockTable = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeBlockTable,
                                                                 CreateTestConstIntVector(shapeBlockTable));
    blockTable->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(128)});

    auto setup = SetupGatherInL1WithMatmul(*currFunctionPtr, input, offsets, blockTable, shapeOut, matmulBShape,
                                           matmulOutShape, false, false);

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = setup.gatherOp.GetTileShape();
    auto vecTile = tileShape.GetVecTile().tile;
    EXPECT_TRUE(vecTile.size() >= 2);
    EXPECT_EQ(vecTile[0], 128) << "vecTile[0] should match output M dimension";
    EXPECT_EQ(vecTile[1], 64) << "vecTile[1] should match output K dimension";

    auto cubeTile = tileShape.GetCubeTile();
    EXPECT_EQ(cubeTile.m[1], vecTile[0]);
    EXPECT_EQ(cubeTile.k[1], vecTile[1]);
    EXPECT_EQ(cubeTile.n[1], -1);
}

TEST_F(TestSetHeuristicTileShapes, TestShapeAndTypeSetting_GatherKM)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherKM", "TestGatherKM", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeOut = {64, 128};
    std::vector<int64_t> shapeIn = {64, 128};
    std::vector<int64_t> shapeOffsets = {1, 64};
    std::vector<int64_t> shapeBlockTable = {1, 64};
    std::vector<int64_t> matmulBShape = {128, 32};
    std::vector<int64_t> matmulOutShape = {64, 32};

    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeIn, CreateTestConstIntVector(shapeIn));
    auto offsets = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shapeOffsets,
                                                              CreateTestConstIntVector(shapeOffsets));
    offsets->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(64)});
    auto blockTable = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeBlockTable,
                                                                 CreateTestConstIntVector(shapeBlockTable));
    blockTable->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(64)});

    auto setup = SetupGatherInL1WithMatmul(*currFunctionPtr, input, offsets, blockTable, shapeOut, matmulBShape,
                                           matmulOutShape, false, false);

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = setup.gatherOp.GetTileShape();
    auto vecTile = tileShape.GetVecTile().tile;
    EXPECT_TRUE(vecTile.size() >= 2);
    EXPECT_EQ(vecTile[0], 64) << "vecTile[0] should match output M dimension";
    EXPECT_EQ(vecTile[1], 128) << "vecTile[1] should match output K dimension";

    auto cubeTile = tileShape.GetCubeTile();
    EXPECT_EQ(cubeTile.m[1], vecTile[0]) << "cubeTile.m should match vecTile[0] for GatherMK";
    EXPECT_EQ(cubeTile.k[1], vecTile[1]) << "cubeTile.k should match vecTile[1] for GatherMK";
    EXPECT_EQ(cubeTile.n[1], -1);
}

TEST_F(TestSetHeuristicTileShapes, TestShapeAndTypeSetting_GatherKN)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherKN", "TestGatherKN", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeOut = {64, 32};
    std::vector<int64_t> shapeIn = {64, 32};
    std::vector<int64_t> shapeOffsets = {1, 64};
    std::vector<int64_t> shapeBlockTable = {1, 64};
    std::vector<int64_t> matmulAShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {128, 32};

    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeIn, CreateTestConstIntVector(shapeIn));
    auto offsets = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shapeOffsets,
                                                              CreateTestConstIntVector(shapeOffsets));
    offsets->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(64)});
    auto blockTable = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeBlockTable,
                                                                 CreateTestConstIntVector(shapeBlockTable));
    blockTable->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(64)});

    auto setup = SetupGatherInL1WithMatmul(*currFunctionPtr, input, offsets, blockTable, shapeOut, matmulAShape,
                                           matmulOutShape, false, false, false);

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = setup.gatherOp.GetTileShape();
    auto vecTile = tileShape.GetVecTile().tile;
    EXPECT_TRUE(vecTile.size() >= 2);
    EXPECT_EQ(vecTile[0], 64) << "vecTile[0] should match output K dimension";
    EXPECT_EQ(vecTile[1], 32) << "vecTile[1] should match output N dimension";

    auto cubeTile = tileShape.GetCubeTile();
    EXPECT_EQ(cubeTile.m[1], vecTile[0])
        << "cubeTile.m should match vecTile[0] for GatherKN with isB=false, isTrans=false";
    EXPECT_EQ(cubeTile.k[1], vecTile[1])
        << "cubeTile.k should match vecTile[1] for GatherKN with isB=false, isTrans=false";
    EXPECT_EQ(cubeTile.n[1], -1) << "cubeTile.n should be -1 for GatherKN with isB=false, isTrans=false";
}

TEST_F(TestSetHeuristicTileShapes, TestShapeAndTypeSetting_GatherNK)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherNK", "TestGatherNK", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeOut = {32, 64};
    std::vector<int64_t> shapeIn = {32, 64};
    std::vector<int64_t> shapeOffsets = {1, 32};
    std::vector<int64_t> shapeBlockTable = {1, 32};
    std::vector<int64_t> matmulAShape = {128, 32};
    std::vector<int64_t> matmulOutShape = {128, 64};

    auto input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeIn, CreateTestConstIntVector(shapeIn));
    auto offsets = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shapeOffsets,
                                                              CreateTestConstIntVector(shapeOffsets));
    offsets->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(32)});
    auto blockTable = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeBlockTable,
                                                                 CreateTestConstIntVector(shapeBlockTable));
    blockTable->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(32)});

    auto setup = SetupGatherInL1WithMatmul(*currFunctionPtr, input, offsets, blockTable, shapeOut, matmulAShape,
                                           matmulOutShape, true, true, false);

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = setup.gatherOp.GetTileShape();
    auto vecTile = tileShape.GetVecTile().tile;
    EXPECT_TRUE(vecTile.size() >= 2);
    EXPECT_EQ(vecTile[0], 32) << "vecTile[0] should match output N dimension";
    EXPECT_EQ(vecTile[1], 64) << "vecTile[1] should match output K dimension";

    auto cubeTile = tileShape.GetCubeTile();
    EXPECT_EQ(cubeTile.m[1], -1);
    EXPECT_EQ(cubeTile.k[1], vecTile[1]);
    EXPECT_EQ(cubeTile.n[1], vecTile[0]);
}

static void CreateGatherL1InputTensors(LogicalTensorPtr* input, LogicalTensorPtr* offsets, LogicalTensorPtr* blockTable)
{
    std::vector<int64_t> shapeIn = {1024, 1024};
    std::vector<int64_t> shapeOffsets = {1, 1024};
    std::vector<int64_t> shapeBlockTable = {1, 1024};

    *input = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeIn, CreateTestConstIntVector(shapeIn));
    *offsets = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shapeOffsets,
                                                          CreateTestConstIntVector(shapeOffsets));
    (*offsets)->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(1024)});
    *blockTable = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeBlockTable,
                                                             CreateTestConstIntVector(shapeBlockTable));
    (*blockTable)->UpdateDynValidShape({IRBuilder().CreateConstInt(1), IRBuilder().CreateConstInt(1024)});
}

TEST_F(TestSetHeuristicTileShapes, TestGatherInL1MemoryLimit)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherL1Mem", "TestGatherL1Mem",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    LogicalTensorPtr input;
    LogicalTensorPtr offsets;
    LogicalTensorPtr blockTable;
    CreateGatherL1InputTensors(&input, &offsets, &blockTable);

    std::vector<int64_t> shapeOut = {1024, 1024};
    std::vector<int64_t> matmulBShape = {1024, 512};
    std::vector<int64_t> matmulOutShape = {1024, 512};

    auto setup = SetupGatherInL1WithMatmul(*currFunctionPtr, input, offsets, blockTable, shapeOut, matmulBShape,
                                           matmulOutShape, false, false);

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = setup.gatherOp.GetTileShape();
    auto vecTile = tileShape.GetVecTile().tile;
    auto cubeTile = tileShape.GetCubeTile();

    EXPECT_TRUE(vecTile.size() >= 2);
    EXPECT_TRUE(vecTile[0] > 0);
    EXPECT_TRUE(vecTile[1] > 0);

    const uint64_t L1_MAX_SIZE = 512 * 1024;
    const uint64_t FP32_SIZE = 4;
    uint64_t tileSizeBytes = static_cast<uint64_t>(vecTile[0]) * static_cast<uint64_t>(vecTile[1]) * FP32_SIZE;
    EXPECT_TRUE(tileSizeBytes <= L1_MAX_SIZE)
        << "Tile size " << tileSizeBytes << " bytes exceeds L1 limit " << L1_MAX_SIZE;

    EXPECT_EQ(cubeTile.m[1], vecTile[0]);
    EXPECT_EQ(cubeTile.k[1], vecTile[1]);
    EXPECT_EQ(cubeTile.n[1], -1);
}

TEST_F(TestSetHeuristicTileShapes, TestShapeAndTypeSetting_TypeDetermination)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestTypeDetermination",
                                                      "TestTypeDetermination", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {128, 64};
    std::vector<int64_t> shapeB = {64, 32};
    std::vector<int64_t> shapeC = {128, 32};

    auto inputA = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP16, shapeA);
    auto inputB = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP16, shapeB);
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeC, CreateTestConstIntVector(shapeC));

    Operation& op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA, inputB}, {output});
    op.SetAttribute(Matrix::A_MUL_B_TRANS_A, false);
    op.SetAttribute(Matrix::A_MUL_B_TRANS_B, false);

    PushInOutCasts(*currFunctionPtr, {inputA, inputB}, {output});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto tileShape = op.GetTileShape();
    auto cubeTile = tileShape.GetCubeTile();
    auto matmulVecTile = tileShape.GetVecTile().tile;

    EXPECT_EQ(cubeTile.m[0], 128) << "cubeTile.m should be 128";
    EXPECT_EQ(cubeTile.k[0], 64) << "cubeTile.k should be 64";
    EXPECT_EQ(cubeTile.n[0], 32) << "cubeTile.n should be 32";

    EXPECT_EQ(matmulVecTile.size(), 2);
    EXPECT_EQ(matmulVecTile[0], 128) << "vecTile[0] should match cubeTile.m";
    EXPECT_EQ(matmulVecTile[1], 64) << "vecTile[1] should match cubeTile.k";
}

TEST_F(TestSetHeuristicTileShapes, TestBackwardTraversal_ReshapeChain)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeChain", "TestReshapeChain",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {8192};
    std::vector<int64_t> shape3 = {64, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {64, 64};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    auto& reshape1Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensor1}, {tensor2});
    auto& reshape2Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensor2}, {tensor3});

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {tensor3, matmulB},
                                                      {matmulOut});

    PushInOutCasts(*currFunctionPtr, {tensor1, matmulB}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto reshape1Tile = reshape1Op.GetTileShape();
    auto reshape1VecTile = reshape1Tile.GetVecTile().tile;
    EXPECT_EQ(reshape1VecTile.size(), 2);
    EXPECT_EQ(reshape1VecTile[0], 128) << "reshape1 vectile[0] should be 128";

    auto reshape2Tile = reshape2Op.GetTileShape();
    auto reshape2VecTile = reshape2Tile.GetVecTile().tile;
    EXPECT_EQ(reshape2VecTile.size(), 1);
    EXPECT_EQ(reshape2VecTile[0], 8192) << "reshape2 vectile[0] should be 8192";

    auto matmulTile = matmulOp.GetTileShape();
    auto cubeTile = matmulTile.GetCubeTile();
    EXPECT_EQ(cubeTile.m[0], 64);
    EXPECT_EQ(cubeTile.k[0], 128);
    EXPECT_EQ(cubeTile.n[0], 64);
}

TEST_F(TestSetHeuristicTileShapes, TestBackwardTraversal_ViewChain)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewChain", "TestViewChain",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {256, 128};
    std::vector<int64_t> shapeB = {256, 32, 4};
    std::vector<int64_t> reshapeOutShape = {8192, 4};
    std::vector<int64_t> matmulBShape = {4, 64};
    std::vector<int64_t> matmulOutShape = {8192, 64};

    auto tensorA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeA, CreateTestConstIntVector(shapeA));
    auto tensorB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeB, CreateTestConstIntVector(shapeB));
    auto reshapeOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeOutShape,
                                                                 CreateTestConstIntVector(reshapeOutShape));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensorA}, {tensorB});

    auto& reshapeOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensorB}, {reshapeOut});

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {reshapeOut, matmulB},
                                                      {matmulOut});

    PushInOutCasts(*currFunctionPtr, {tensorA, matmulB}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto viewTile = viewOp.GetTileShape();
    auto viewVecTile = viewTile.GetVecTile().tile;
    EXPECT_EQ(viewVecTile.size(), 3);
    EXPECT_LE(viewVecTile[0], 16);
    EXPECT_EQ(viewVecTile[1], 32);
    EXPECT_EQ(viewVecTile[2], 4);

    auto reshapeTile = reshapeOp.GetTileShape();
    auto reshapeVecTile = reshapeTile.GetVecTile().tile;
    EXPECT_EQ(reshapeVecTile.size(), 3);
    EXPECT_LE(reshapeVecTile[0], 16);
    EXPECT_EQ(reshapeVecTile[1], 32);

    auto matmulTile = matmulOp.GetTileShape();
    auto cubeTile = matmulTile.GetCubeTile();
    EXPECT_GE(cubeTile.m[0], 256);
    EXPECT_GE(cubeTile.k[0], 8);
    EXPECT_GE(cubeTile.n[0], 8);
}

TEST_F(TestSetHeuristicTileShapes, TestBackwardTraversal_MultipleProducers)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMultipleProducers",
                                                      "TestMultipleProducers", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> view1InShape = {128, 64};
    std::vector<int64_t> view1OutShape = {128, 64};
    std::vector<int64_t> view2InShape = {128, 64};
    std::vector<int64_t> view2OutShape = {128, 64};
    std::vector<int64_t> shapeOut = {128, 64};
    std::vector<int64_t> matmulBShape = {64, 32};
    std::vector<int64_t> matmulOutShape = {128, 32};

    auto view1In = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1InShape,
                                                              CreateTestConstIntVector(view1InShape));
    auto view1Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1OutShape,
                                                               CreateTestConstIntVector(view1OutShape));
    auto view2In = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view2InShape,
                                                              CreateTestConstIntVector(view2InShape));
    auto view2Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view2OutShape,
                                                               CreateTestConstIntVector(view2OutShape));
    auto tensorOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeOut, CreateTestConstIntVector(shapeOut));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {view1In}, {view1Out});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {view2In}, {view2Out});

    auto& addOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {view1Out, view2Out}, {tensorOut});

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {tensorOut, matmulB},
                                                      {matmulOut});

    PushInOutCasts(*currFunctionPtr, {view1In, view2In, matmulB}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto addTile = addOp.GetTileShape();
    auto addVecTile = addTile.GetVecTile().tile;
    EXPECT_EQ(addVecTile.size(), 2);
    EXPECT_EQ(addVecTile[0], 128) << "ADD vectile[0] should match matmul M";
    EXPECT_EQ(addVecTile[1], 64) << "ADD vectile[1] should match matmul K";

    auto matmulTile = matmulOp.GetTileShape();
    auto cubeTile = matmulTile.GetCubeTile();
    EXPECT_EQ(cubeTile.m[0], 128);
    EXPECT_EQ(cubeTile.k[0], 64);
    EXPECT_EQ(cubeTile.n[0], 32);
}

TEST_F(TestSetHeuristicTileShapes, TestBackwardTraversal_2DShapes)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "Test2DShapes", "Test2DShapes", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape2D = {256, 128};
    std::vector<int64_t> shape1D = {256};
    std::vector<int64_t> shapeOut = {256, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {256, 64};

    auto tensor2D = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2D, CreateTestConstIntVector(shape2D));
    auto tensor1D = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1D, CreateTestConstIntVector(shape1D));
    auto tensorOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeOut, CreateTestConstIntVector(shapeOut));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    auto& reshape1Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensor2D}, {tensor1D});
    auto& reshape2Op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensor1D}, {tensorOut});

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {tensorOut, matmulB},
                                                      {matmulOut});

    PushInOutCasts(*currFunctionPtr, {tensor2D, matmulB}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto reshape1Tile = reshape1Op.GetTileShape();
    auto reshape1VecTile = reshape1Tile.GetVecTile().tile;
    EXPECT_EQ(reshape1VecTile.size(), 2);

    auto reshape2Tile = reshape2Op.GetTileShape();
    auto reshape2VecTile = reshape2Tile.GetVecTile().tile;
    EXPECT_EQ(reshape2VecTile.size(), 1);
    EXPECT_EQ(reshape2VecTile[0], 256) << "reshape2 vectile[0] should match matmul M";

    auto matmulTile = matmulOp.GetTileShape();
    auto cubeTile = matmulTile.GetCubeTile();
    EXPECT_EQ(cubeTile.m[0], 128);
    EXPECT_EQ(cubeTile.k[0], 128);
    EXPECT_EQ(cubeTile.n[0], 64);
}

TEST_F(TestSetHeuristicTileShapes, TestBackwardTraversal_AssembleOp)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssemble", "TestAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {64, 128};
    std::vector<int64_t> shape2 = {64, 128};
    std::vector<int64_t> shapeOut = {128, 128};
    std::vector<int64_t> matmulBShape = {128, 64};
    std::vector<int64_t> matmulOutShape = {128, 64};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto tensorOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeOut, CreateTestConstIntVector(shapeOut));
    auto matmulB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulBShape,
                                                              CreateTestConstIntVector(matmulBShape));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tensor1, tensor2},
                                                        {tensorOut});

    auto& matmulOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {tensorOut, matmulB},
                                                      {matmulOut});

    PushInOutCasts(*currFunctionPtr, {tensor1, tensor2, matmulB}, {matmulOut});

    RunPassAndVerifySuccess(*currFunctionPtr);

    auto assembleTile = assembleOp.GetTileShape();
    auto assembleVecTile = assembleTile.GetVecTile().tile;
    EXPECT_EQ(assembleVecTile.size(), 2);
    EXPECT_EQ(assembleVecTile[0], 128) << "Assemble vectile[0] should match matmul M";
    EXPECT_EQ(assembleVecTile[1], 128) << "Assemble vectile[1] should match matmul K";

    auto matmulTile = matmulOp.GetTileShape();
    auto cubeTile = matmulTile.GetCubeTile();
    EXPECT_EQ(cubeTile.m[0], 128);
    EXPECT_EQ(cubeTile.k[0], 128);
    EXPECT_EQ(cubeTile.n[0], 64);
}

TEST_F(TestSetHeuristicTileShapes, TestForwardBroadcast_ReshapeChain)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestForwardReshape",
                                                      "TestForwardReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> reshape1 = {8192};
    std::vector<int64_t> reshape2 = {64, 128};

    auto inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputShape,
                                                                  CreateTestConstIntVector(inputShape));
    auto ub1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape1, CreateTestConstIntVector(reshape1));
    auto ub2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape2, CreateTestConstIntVector(reshape2));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inputTensor}, {ub1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ub1}, {ub2});

    PushInOutCasts(*currFunctionPtr, {inputTensor}, {ub2});

    RunPassAndVerifySuccess(*currFunctionPtr);
}

TEST_F(TestSetHeuristicTileShapes, TestForwardBroadcast_ElementwiseAdd)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestForwardAdd", "TestForwardAdd",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> view1InShape = {128, 64};
    std::vector<int64_t> view1OutShape = {128, 64};
    std::vector<int64_t> view2InShape = {128, 64};
    std::vector<int64_t> view2OutShape = {128, 64};
    std::vector<int64_t> shapeOut = {128, 64};

    auto view1In = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1InShape,
                                                              CreateTestConstIntVector(view1InShape));
    auto view1Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view1OutShape,
                                                               CreateTestConstIntVector(view1OutShape));
    auto view2In = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view2InShape,
                                                              CreateTestConstIntVector(view2InShape));
    auto view2Out = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, view2OutShape,
                                                               CreateTestConstIntVector(view2OutShape));
    auto tensorOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeOut, CreateTestConstIntVector(shapeOut));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {view1In}, {view1Out});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {view2In}, {view2Out});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {view1Out, view2Out}, {tensorOut});

    PushInOutCasts(*currFunctionPtr, {view1In, view2In}, {tensorOut});

    RunPassAndVerifySuccess(*currFunctionPtr);
}

TEST_F(TestSetHeuristicTileShapes, TestForwardBroadcast_ViewToExp)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestForwardViewExp",
                                                      "TestForwardViewExp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inputShape = {256, 128};
    std::vector<int64_t> viewShape = {256, 32, 4};
    std::vector<int64_t> outputShape = {256, 32, 4};

    auto inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputShape,
                                                                  CreateTestConstIntVector(inputShape));
    auto viewTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, viewShape,
                                                                 CreateTestConstIntVector(viewShape));
    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outputShape,
                                                                   CreateTestConstIntVector(outputShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inputTensor}, {viewTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {viewTensor}, {outputTensor});

    PushInOutCasts(*currFunctionPtr, {inputTensor}, {outputTensor});

    RunPassAndVerifySuccess(*currFunctionPtr);
}

TEST_F(TestSetHeuristicTileShapes, TestForwardBroadcast_MultipleConsumers)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestForwardMultiConsumer",
                                                      "TestForwardMultiConsumer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> reshape1Shape = {8192};
    std::vector<int64_t> reshape2Shape = {128, 64};
    std::vector<int64_t> addShape = {8192};

    auto inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inputShape,
                                                                  CreateTestConstIntVector(inputShape));
    auto reshape1Tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape1Shape,
                                                                     CreateTestConstIntVector(reshape1Shape));
    auto reshape2Tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshape2Shape,
                                                                     CreateTestConstIntVector(reshape2Shape));
    auto addTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, addShape, CreateTestConstIntVector(addShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inputTensor}, {reshape1Tensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inputTensor}, {reshape2Tensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {reshape1Tensor, reshape2Tensor}, {addTensor});

    PushInOutCasts(*currFunctionPtr, {inputTensor}, {addTensor});

    RunPassAndVerifySuccess(*currFunctionPtr);
}

TEST_F(TestSetHeuristicTileShapes, TestForwardBroadcast_2DTo1D)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestForward2DTo1D", "TestForward2DTo1D",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape2D = {256, 128};
    std::vector<int64_t> shape1D = {32768};
    std::vector<int64_t> shapeOut = {256, 128};

    auto tensor2D = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2D, CreateTestConstIntVector(shape2D));
    auto tensor1D = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1D, CreateTestConstIntVector(shape1D));
    auto tensorOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeOut, CreateTestConstIntVector(shapeOut));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensor2D}, {tensor1D});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {tensor1D}, {tensorOut});

    PushInOutCasts(*currFunctionPtr, {tensor2D}, {tensorOut});

    RunPassAndVerifySuccess(*currFunctionPtr);
}

TEST_F(TestSetHeuristicTileShapes, TestForwardBroadcast_MatmulToReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestForwardMatmulReshape",
                                                      "TestForwardMatmulReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shapeA = {128, 64};
    std::vector<int64_t> shapeB = {64, 32};
    std::vector<int64_t> matmulOutShape = {128, 32};
    std::vector<int64_t> reshapeOutShape = {4096};

    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeA, CreateTestConstIntVector(shapeA));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shapeB, CreateTestConstIntVector(shapeB));
    auto matmulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, matmulOutShape,
                                                                CreateTestConstIntVector(matmulOutShape));
    auto reshapeOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, reshapeOutShape,
                                                                 CreateTestConstIntVector(reshapeOutShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {inputA, inputB}, {matmulOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {matmulOut}, {reshapeOut});

    PushInOutCasts(*currFunctionPtr, {inputA, inputB}, {reshapeOut});

    RunPassAndVerifySuccess(*currFunctionPtr);
}

} // namespace npu::tile_fwk
