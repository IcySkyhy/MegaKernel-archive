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
 * \file test_remove_redundant_reshape.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "computational_graph_builder.h"
#define private public
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"

namespace npu {
namespace tile_fwk {
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeThirteen = 13UL;
static const size_t kSizeFourteen = 14UL;
static const size_t kSizeFifteen = 15UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

class TestRemoveRedundantReshapePass : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}

private:
    Function* BuildComplexGraphForSTest1()
    {
        std::vector<int64_t> shape1 = {kNumExpSix, kNumExpSix};
        std::vector<int64_t> shape2 = {kNumExpFive, kNumExpSeven};
        std::vector<int64_t> shape3 = {kNumExpSeven, kNumExpFive};

        ComputationalGraphBuilder subGraph;

        std::vector<MemoryType> memTypes1(2, MemoryType::MEM_DEVICE_DDR);
        std::vector<MemoryType> memTypes2(8, MemoryType::MEM_DEVICE_DDR);
        std::vector<MemoryType> memTypes3(6, MemoryType::MEM_DEVICE_DDR);

        EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape1, memTypes1, {"input", "view_out"}, 0), true);
        EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape2, memTypes2,
                                      {"reshape1", "reshape3", "exp2", "output2_pre", "assemble_out1", "assemble_out4",
                                       "assemble_out6", "output2"},
                                      0),
                  true);
        EXPECT_EQ(
            subGraph.AddTensors(DT_FP32, shape3, memTypes3,
                                {"reshape2", "exp1", "output1_pre", "assemble_out2", "assemble_out3", "output1"}, 0),
            true);

        std::vector<Opcode> opCodes = {
            Opcode::OP_VIEW,     Opcode::OP_RESHAPE,  Opcode::OP_RESHAPE,  Opcode::OP_EXP,      Opcode::OP_RESHAPE,
            Opcode::OP_RESHAPE,  Opcode::OP_EXP,      Opcode::OP_RESHAPE,  Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE,
            Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE};

        std::vector<std::vector<std::string>> ioperands = {{"input"},       {"view_out"}, {"reshape1"},   {"reshape2"},
                                                           {"exp1"},        {"reshape3"}, {"reshape1"},   {"exp2"},
                                                           {"reshape1"},    {"reshape2"}, {"exp1"},       {"reshape3"},
                                                           {"output1_pre"}, {"exp2"},     {"output2_pre"}};

        std::vector<std::vector<std::string>> ooperands = {
            {"view_out"},      {"reshape1"},      {"reshape2"},    {"exp1"},          {"reshape3"},
            {"output1_pre"},   {"exp2"},          {"output2_pre"}, {"assemble_out1"}, {"assemble_out2"},
            {"assemble_out3"}, {"assemble_out4"}, {"output1"},     {"assemble_out6"}, {"output2"}};

        std::vector<std::string> opNames = {"view",      "reshape1",  "reshape2",  "exp1",      "reshape3",
                                            "reshape4",  "exp2",      "reshape5",  "assemble1", "assemble2",
                                            "assemble3", "assemble4", "assemble5", "assemble6", "assemble7"};

        EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

        EXPECT_EQ(subGraph.SetInCast({"input"}), true);
        EXPECT_EQ(subGraph.SetOutCast({"assemble_out1", "assemble_out2", "assemble_out3", "assemble_out4", "output1",
                                       "assemble_out6", "output2"}),
                  true);

        TileShape::Current().SetVecTile({64, 64});

        return subGraph.GetFunction();
    }
};

/*
RemoveReshapeChain
inCast{8,16}->reshape->ubTensor1{16,8}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
inCast{8,16}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest1)
{
    ComputationalGraphBuilder subGraph;
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};

    std::vector<MemoryType> memTypes1(1, MemoryType::MEM_DEVICE_DDR);
    std::vector<MemoryType> memTypes2(1, MemoryType::MEM_DEVICE_DDR);
    std::vector<MemoryType> memTypes3(2, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape1, memTypes1, {"inCast"}, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape2, memTypes2, {"ubTensor1"}, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape3, memTypes3, {"ubTensor2", "outCast"}, 0), true);

    std::vector<Opcode> opCodes = {Opcode::OP_RESHAPE, Opcode::OP_RESHAPE, Opcode::OP_SQRT};
    std::vector<std::vector<std::string>> ioperands = {{"inCast"}, {"ubTensor1"}, {"ubTensor2"}};
    std::vector<std::vector<std::string>> ooperands = {{"ubTensor1"}, {"ubTensor2"}, {"outCast"}};
    std::vector<std::string> opNames = {"reshape1", "reshape2", "sqrt"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(subGraph.SetInCast({"inCast"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"outCast"}), true);

    Function* function = subGraph.GetFunction();
    auto reshape2Magic = subGraph.GetOp("reshape2")->GetOpMagic();

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    const auto& operations = function->Operations();
    uint32_t reshape_num = kNumZero;
    auto* reshape2Op = subGraph.GetOp("reshape2");
    auto* sqrtOp = subGraph.GetOp("sqrt");
    auto inCastTensor = subGraph.GetTensor("inCast");
    auto ubTensor2Tensor = subGraph.GetTensor("ubTensor2");
    for (auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(reshape2Magic, op.GetOpMagic());
            EXPECT_EQ(reshape2Op->GetInputOperand(kSizeZero), inCastTensor);
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrtOp->GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrtOp->GetInputOperand(kSizeZero), ubTensor2Tensor);
        }
    }
    EXPECT_EQ(reshape_num, kNumOne);
}

/*
RemoveSameReshape
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast{8,16}
inCast{8,16}->sqrt->outCast{8,16}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest2)
{
    ComputationalGraphBuilder subGraph;
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<MemoryType> memTypes(3, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape, memTypes, {"inCast", "ubTensor", "outCast"}, 0), true);

    std::vector<Opcode> opCodes = {Opcode::OP_RESHAPE, Opcode::OP_SQRT};
    std::vector<std::vector<std::string>> ioperands = {{"inCast"}, {"ubTensor"}};
    std::vector<std::vector<std::string>> ooperands = {{"ubTensor"}, {"outCast"}};
    std::vector<std::string> opNames = {"reshape", "sqrt"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(subGraph.SetInCast({"inCast"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"outCast"}), true);

    Function* function = subGraph.GetFunction();

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    uint32_t reshape_num = kNumZero;
    auto* sqrtOp = subGraph.GetOp("sqrt");
    auto inCastTensor = subGraph.GetTensor("inCast");
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrtOp->GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrtOp->GetInputOperand(kSizeZero), inCastTensor);
        }
    }
    EXPECT_EQ(reshape_num, kNumZero);
}

/*
RemoveReshapeChainSeveralConsumer(WARNING CASE)
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast1{8,16}
                                    ->exp->outCast2{8,16}
                                    ->reshape->outCast3{16,8}
inCast{8,16}->sqrt->outCast1{8,16}
            ->exp->outCast2{8,16}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest3)
{
    ComputationalGraphBuilder subGraph;
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};

    std::vector<MemoryType> memTypes1(4, MemoryType::MEM_DEVICE_DDR);
    std::vector<MemoryType> memTypes2(1, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape1, memTypes1, {"inCast", "ubTensor", "outCast1", "outCast2"}, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape2, memTypes2, {"outCast3"}, 0), true);

    std::vector<Opcode> opCodes = {Opcode::OP_RESHAPE, Opcode::OP_SQRT, Opcode::OP_EXP, Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands = {{"inCast"}, {"ubTensor"}, {"ubTensor"}, {"ubTensor"}};
    std::vector<std::vector<std::string>> ooperands = {{"ubTensor"}, {"outCast1"}, {"outCast2"}, {"outCast3"}};
    std::vector<std::string> opNames = {"reshape1", "sqrt", "exp", "reshape2"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(subGraph.SetInCast({"inCast"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"outCast1", "outCast2", "outCast3"}), true);

    Function* function = subGraph.GetFunction();

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.DefaultEnabledPreCheck(*function), SUCCESS);
    EXPECT_NE(pass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    uint32_t reshape_num = kNumZero;
    auto* sqrtOp = subGraph.GetOp("sqrt");
    auto* expOp = subGraph.GetOp("exp");
    auto inCastTensor = subGraph.GetTensor("inCast");
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrtOp->GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrtOp->GetInputOperand(kSizeZero), inCastTensor);
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(expOp->GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(expOp->GetInputOperand(kSizeZero), inCastTensor);
        }
    }
    EXPECT_EQ(reshape_num, kNumOne);
}

/*
RemoveReshapeChainSeveralConsumer
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
                                      ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
            ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest4)
{
    ComputationalGraphBuilder subGraph;
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};

    std::vector<MemoryType> memTypes1(1, MemoryType::MEM_DEVICE_DDR);
    std::vector<MemoryType> memTypes2(2, MemoryType::MEM_DEVICE_DDR);
    std::vector<MemoryType> memTypes3(2, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape1, memTypes1, {"inCast"}, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape2, memTypes2, {"ubTensor1", "outCast1"}, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape3, memTypes3, {"ubTensor2", "outCast2"}, 0), true);

    std::vector<Opcode> opCodes = {Opcode::OP_RESHAPE, Opcode::OP_EXP, Opcode::OP_RESHAPE, Opcode::OP_SQRT};
    std::vector<std::vector<std::string>> ioperands = {{"inCast"}, {"ubTensor1"}, {"ubTensor1"}, {"ubTensor2"}};
    std::vector<std::vector<std::string>> ooperands = {{"ubTensor1"}, {"outCast1"}, {"ubTensor2"}, {"outCast2"}};
    std::vector<std::string> opNames = {"reshape1", "exp", "reshape2", "sqrt"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(subGraph.SetInCast({"inCast"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"outCast1", "outCast2"}), true);

    Function* function = subGraph.GetFunction();
    auto inCastTensor = subGraph.GetTensor("inCast");

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    uint32_t reshape_num = kNumZero;
    auto* reshape1Op = subGraph.GetOp("reshape1");
    auto* reshape2Op = subGraph.GetOp("reshape2");
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        }
    }
    EXPECT_EQ(reshape1Op->GetInputOperand(kSizeZero), inCastTensor);
    EXPECT_EQ(reshape2Op->GetInputOperand(kSizeZero), inCastTensor);
    EXPECT_EQ(reshape_num, kNumTwo);
}

/*
view->reshape->reshape  ->exp       ->reshape   ->reshape   ->assemble
                                                ->assemble
                                    ->assemble
                        ->assemble
             ->exp      ->reshape->assemble
             ->assemble
view->reshape  ->exp        ->reshape   ->assemble
                            ->assemble
                            ->assemble
               ->assemble
    ->reshape  ->exp        ->assemble
               ->assemble
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeSTest1)
{
    Function* function = BuildComplexGraphForSTest1();

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    auto updated_operations = function->Operations();
    int reshape_num = kNumZero;
    EXPECT_EQ(updated_operations.size(), kSizeThirteen);
    for (const auto& op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_num++;
        }
    }
    EXPECT_EQ(reshape_num, kNumThree);
}

TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest5)
{
    ComputationalGraphBuilder subGraph;
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};

    std::vector<MemoryType> memTypes1(1, MemoryType::MEM_DEVICE_DDR);
    std::vector<MemoryType> memTypes2(1, MemoryType::MEM_DEVICE_DDR);
    std::vector<MemoryType> memTypes3(2, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape1, memTypes1, {"inCast"}, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape2, memTypes2, {"ubTensor1"}, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape3, memTypes3, {"ubTensor2", "outCast"}, 0), true);

    std::vector<Opcode> opCodes = {Opcode::OP_RESHAPE, Opcode::OP_RESHAPE, Opcode::OP_SQRT};
    std::vector<std::vector<std::string>> ioperands = {{"inCast"}, {"ubTensor1"}, {"ubTensor2"}};
    std::vector<std::vector<std::string>> ooperands = {{"ubTensor1"}, {"ubTensor2"}, {"outCast"}};
    std::vector<std::string> opNames = {"reshape1", "reshape2", "sqrt"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(subGraph.SetInCast({"inCast"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"outCast"}), true);

    Function* function = subGraph.GetFunction();

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);
}

/*
inCast->reShape->ubTensor1->reShape->outCast

inCast->reShape->ubTensor1->reShape->outCast
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeContainNegativeOne)
{
    ComputationalGraphBuilder subGraph;
    int64_t kSizeNegativeOne = -1;
    std::vector<int64_t> shape = {kSizeNegativeOne, kNumEight};
    std::vector<MemoryType> memTypes(3, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape, memTypes, {"inCast", "ubTensor1", "outCast"}, 0), true);

    std::vector<Opcode> opCodes = {Opcode::OP_RESHAPE, Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands = {{"inCast"}, {"ubTensor1"}};
    std::vector<std::vector<std::string>> ooperands = {{"ubTensor1"}, {"outCast"}};
    std::vector<std::string> opNames = {"reshape1", "reshape2"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(subGraph.SetInCast({"inCast"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"outCast"}), true);

    Function* function = subGraph.GetFunction();

    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    int reshapeNum = kNumZero;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshapeNum;
        }
    }
    EXPECT_EQ(reshapeNum, kNumTwo);
}

TEST_F(TestRemoveRedundantReshapePass, ReshapeNoConsumer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    std::vector<MemoryType> memTypes(2, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(subGraph.AddTensors(DT_FP32, shape, memTypes, {"inCast", "outCast"}, 0), true);

    std::vector<Opcode> opCodes = {Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands = {{"inCast"}};
    std::vector<std::vector<std::string>> ooperands = {{"outCast"}};
    std::vector<std::string> opNames = {"reshape"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(subGraph.SetInCast({"inCast"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"outCast"}), true);

    Function* function = subGraph.GetFunction();

    RemoveRedundantReshape pass;
    Status ret = pass.PreCheck(*function);
    EXPECT_EQ(ret, FAILED);
}

} // namespace tile_fwk
} // namespace npu
