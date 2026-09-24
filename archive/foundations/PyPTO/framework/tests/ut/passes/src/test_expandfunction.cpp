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
 * \file test_expandfunction.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "passes/pass_utils/graph_utils.h"
#define private public
#include "interface/operation/operation.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "computational_graph_builder.h"
#include "passes/pass_check/inplace_conflict_checker.h"

namespace npu {
namespace tile_fwk {
static const size_t kSizeZero = 0UL;
static const size_t kSizeThree = 3UL;
static const size_t kSizeEight = 8UL;
static const size_t kSizeEleven = 11UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumForteen = 14u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumSevenTeen = 17u;
static const uint16_t kNumTwentyfive = 25u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

std::shared_ptr<AssembleOpAttribute> CreateAssembleOpAttr();

void MakeExpandGrpah(std::shared_ptr<Function>& currFunctionPtr, LogicalTensorPtr& outCast)
{
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& div_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {inCast1, inCast2}, {ubTensor});
    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});
    auto op_attr = CreateAssembleOpAttr();
    assemble_op.SetOpAttribute(op_attr);
    div_op.tileShape_.SetVecTile(tile_shape);
    assemble_op.tileShape_.SetVecTile(tile_shape);

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);
}

std::shared_ptr<AssembleOpAttribute> CreateAssembleOpAttr()
{
    std::vector<int64_t> toOffset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> symbol = {CreateTestScalarVar("sym0"), CreateTestScalarVar("sym1")};
    return std::make_shared<AssembleOpAttribute>(toOffset, symbol);
}

namespace {

void SetViewOpAttribute(Operation& viewOp, const std::vector<int64_t>& offsets, const std::vector<int64_t>& shapes)
{
    auto dynOffsets = SymbolicScalar::FromConcrete(offsets);
    auto validShape = GetViewValidShape(viewOp.GetIOperands()[0]->GetDynValidShape(), offsets, dynOffsets, shapes);
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offsets, dynOffsets, validShape));
}

int CountInputSliceOpsWithSecondDimOffset(Function& function, const LogicalTensorPtr& input, int64_t offset)
{
    int count = 0;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        if (input == nullptr || op.GetIOperands().empty() || op.GetIOperands()[0] != input) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewAttr == nullptr || viewAttr->GetFromOffset().size() < 2) {
            continue;
        }
        if (viewAttr->GetFromOffset()[1] == offset) {
            count++;
        }
    }
    return count;
}

int CountSliceOpsOnInputWithShape(Function& function, const LogicalTensorPtr& input, const std::vector<int64_t>& shape)
{
    int count = 0;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        if (input == nullptr || op.GetIOperands().empty() || op.GetIOperands()[0] != input) {
            continue;
        }
        if (op.GetOOperands().empty() || op.GetOOperands()[0]->GetShape() != shape) {
            continue;
        }
        ++count;
    }
    return count;
}

} // namespace

struct ScopeCfg {
    std::string op;
    int id;
    bool parMerge;
    bool crossMerge;
};

void RunScopeInfoTest(const std::vector<std::string>& tensors, size_t numInputs, const std::vector<Opcode>& opcodes,
                      const std::vector<std::vector<std::string>>& inputs,
                      const std::vector<std::vector<std::string>>& outputs, const std::vector<std::string>& opNames,
                      const Status status, const std::vector<ScopeCfg>& scopes = {})
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape{kNumExpSix, kNumExpSix};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape, tensors), true);
    EXPECT_EQ(G.AddOps(opcodes, inputs, outputs, opNames, true), true);
    for (const auto& s : scopes) {
        Operation::ScopeInfo info(s.id);
        info.allowParallelMerge = s.parMerge;
        info.allowCrossScopeMerge = s.crossMerge;
        auto op = G.GetOp(s.op);
        op->SetScopeInfo(info);
        if (op->GetCoreType() == CoreType::AIV)
            op->tileShape_.SetVecTile(kNumExpSix, kNumExpSix);
    }
    EXPECT_EQ(G.SetInCast({tensors.begin(), tensors.begin() + numInputs}), true);
    EXPECT_EQ(G.SetOutCast({tensors.begin() + numInputs, tensors.end()}), true);
    G.GetFunction()->SetGraphType(GraphType::TENSOR_GRAPH);
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);
    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.RunOnFunction(*G.GetFunction()), status);
}

class TestExpandFunctionPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPassOption(ENABLE_SLICE, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }
};

/*
TESTExpandFunctionNotTensorGrpah
inCast{8,16}->nop->outCast{8,16}

inCast{8,16}->nop->outCast{8,16}
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& nop_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_NOP, {inCast}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t nop_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_NOP) {
            EXPECT_EQ(nop_op.GetOpMagic(), op.GetOpMagic());
            ++nop_num;
        }
    }
    EXPECT_EQ(nop_num, kNumOne);
}

TEST_F(TestExpandFunctionPass, TestCVSeperate1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_TRUE(GraphUtils::IsCVMixPlatform());
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    EXPECT_FALSE(GraphUtils::IsCVMixPlatform());
}

TEST_F(TestExpandFunctionPass, TestCVSeperate2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);
    TileShape::Current().SetCubeTile({kNumExpFive, kNumExpFive}, {kNumExpFive, kNumExpFive}, {kNumExpFive, kNumExpFive},
                                     false);

    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto L1Tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto L1Tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& opAdd = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {out1});
    auto& opMatmul = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {L1Tensor1, L1Tensor2},
                                                      {out2});

    currFunctionPtr->inCasts_.push_back(ubTensor1);
    currFunctionPtr->inCasts_.push_back(ubTensor2);
    currFunctionPtr->inCasts_.push_back(L1Tensor1);
    currFunctionPtr->inCasts_.push_back(L1Tensor2);
    currFunctionPtr->outCasts_.push_back(out1);
    currFunctionPtr->outCasts_.push_back(out2);

    opAdd.tileShape_.SetVecTile(tile_shape);
    opAdd.SetScopeId(1);
    opMatmul.SetScopeId(1);
    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, FAILED);
}
/*
TESTExpandFunctionNOP
inCast{8,16}->nop->ubTensor2{8,16}->view->outCast{8,16}
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto op_attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{kNumZero, kNumZero});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_NOP, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor1}, {outCast});

    std::shared_ptr<Operation> nop_op, view_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex) {
        auto op = currFunctionPtr->Operations().operations_[uIndex];
        if (op->GetOpcode() == Opcode::OP_NOP)
            nop_op = op;
        else if (op->GetOpcode() == Opcode::OP_VIEW)
            view_op = op;
    }

    view_op->SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    uint32_t nop_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_NE(view_op->GetOpMagic(), op.GetOpMagic());
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contract_num;
        } else if (op.GetOpcode() == Opcode::OP_NOP) {
            EXPECT_NE(nop_op->GetOpMagic(), op.GetOpMagic());
            ++nop_num;
        }
    }
    EXPECT_EQ(slice_num, kNumOne);
    EXPECT_EQ(contract_num, kNumOne);
    EXPECT_EQ(nop_num, kNumOne);
}

/*
TESTExpandFunctionAssemble
inCast{64,64}->assemble->outCast{64,64}
assemble tile-expands to 1 slice + 1 contract (single tile fallback)
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto op_attr = CreateAssembleOpAttr();
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {outCast});

    std::shared_ptr<Operation> assemble_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex) {
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_op = currFunctionPtr->Operations().operations_[uIndex];
        }
    }

    assemble_op->SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_NE(assemble_op->GetOpMagic(), op.GetOpMagic());
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            EXPECT_NE(assemble_op->GetOpMagic(), op.GetOpMagic());
            ++contract_num;
        }
    }
    EXPECT_EQ(slice_num, kNumOne);
    EXPECT_EQ(contract_num, kNumOne);
}

/*
TESTExpandFunctionAssemble
inCast1{64,64}->div->ubTensor{64,64}->assemble->outCast{64,64}
inCast2{64,64}->
inCast1{64,64}->view*4->div->ubTensor{64,64}->assemble(*4)->outCast{64,64}
inCast2{64,64}->view*4->
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest4)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);
    LogicalTensorPtr outCast;
    MakeExpandGrpah(currFunctionPtr, outCast);

    ExpandFunction expandfunctionpass;
    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest4_before0.json");
    EXPECT_EQ(expandfunctionpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest4_after0.json");
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t div_num = kNumZero;
    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_DIV) {
            EXPECT_EQ(op.GetInputOperand(kSizeZero)->shape, tile_shape);
            EXPECT_EQ(op.GetOutputOperand(kSizeZero)->shape, tile_shape);
            EXPECT_NE(op.GetOutputOperand(kSizeZero), outCast);
            ++div_num;
        } else if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contract_num;
        }
    }
    EXPECT_EQ(div_num, kNumFour);
    // div: 8 input slices + 4 assemble input slices
    EXPECT_EQ(slice_num, kNumEight + kNumFour);
    // div: 4 output contracts + 4 assemble output contracts
    EXPECT_EQ(contract_num, kNumFour + kNumFour);
}

/*
TESTExpandFunctionAssemble
inCast{32,128}->reshape->ubTensor{64,64}->assemble->outCast{32,128}
assemble with reshape producer is view-like and should remain assemble after ExpandFunction.
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest5)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph: reshape -> assemble
    std::vector<int64_t> shape1 = {kNumExpFive, kNumExpSeven};
    std::vector<int64_t> shape2 = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumExpSeven};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));

    auto op_attr = CreateAssembleOpAttr();
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});

    std::shared_ptr<Operation> reshape_op;
    std::shared_ptr<Operation> assemble_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex) {
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_op = currFunctionPtr->Operations().operations_[uIndex];
        }
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_op = currFunctionPtr->Operations().operations_[uIndex];
        }
    }

    assemble_op->SetOpAttribute(op_attr);
    reshape_op->tileShape_.SetVecTile({kNumExpFive, kNumExpFive});
    assemble_op->tileShape_.SetVecTile({kNumExpFive, kNumExpFive});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;

    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest5_before1.json");
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest5_after1.json");
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    uint32_t reshape_num = kNumZero;
    uint32_t assemble_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_NE(op.GetOpMagic(), assemble_op->GetOpMagic());
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            EXPECT_NE(op.GetOpMagic(), assemble_op->GetOpMagic());
            auto attr = op.GetOpAttribute();
            EXPECT_NE(attr, nullptr);
            ++contract_num;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(slice_num, kNumZero);
    EXPECT_EQ(contract_num, kNumZero);
    EXPECT_EQ(reshape_num, kNumOne);
    EXPECT_EQ(assemble_num, kNumOne);
}

/*
{64, 64} -> exp -> {64, 64}
{64, 64} -> (view) - > exp -> (assemble) - > {64, 64}
{64, 64} -> (view) -> (view {32, 64}) - > exp -> (assemble {32, 64}) -> (assemble) - > {64, 64}
                   -> (view {32, 64}) - > exp -> (assemble {32, 64})
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionSTest1)
{
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpSix};
    PassManager& passManager = PassManager::Instance();
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    TileShape::Current().SetVecTile(tile_shape);
    config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
    FUNCTION("STCase1") { output = Exp(input); }
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kSizeThree);
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
                                                                   {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                                               });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int exp_num = kNumZero;
    int slice_num = kNumZero;
    int contract_num = kNumZero;
    int view_num = kNumZero;
    int assemble_num = kNumZero;
    for (const auto& op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_EXP) {
            exp_num++;
            EXPECT_EQ(op.GetInputOperand(0)->shape, tile_shape);
            EXPECT_EQ(op.GetOutputOperand(0)->shape, tile_shape);
        } else if (op.GetOpcode() == Opcode::OP_SLICE) {
            slice_num++;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            contract_num++;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
            EXPECT_TRUE(op.HasAttribute(OpAttributeKey::isGlobalInput));
            EXPECT_TRUE(op.GetBoolAttribute(OpAttributeKey::isGlobalInput));
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_num++;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(exp_num, kNumTwo);
    EXPECT_GT(slice_num, kNumZero);
    EXPECT_EQ(slice_num, contract_num);
}

/*
{64, 64} -> exp -> view -> reciprocal
                        -> sqrt -> reshape
{64, 64} -> view -> exp -> view          -> sqrt         -> reshape        ->assemble(end)
                                                         -> assemble(end)
                                         -> reciprocal   -> assemble(end)
                                         -> assemble(end)
                        -> assemble(end)
view -> view(*4) -> exp(*4) -> assemble(*4) ->view  -> view(*4+4)   -> sqrt(*4)         -> assemble(*4)     -> reshape
-> assemble(end)
                                                                                        -> assemble(*4)     ->
assemble(*4) -> assemble(end)
                                            ->assemble(end)         -> reciprocal(*4)   -> assemble(*4)     ->
assemble(end)
*/
void ConstructGraphST2()
{
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> view_shape = {kNumExpSeven, kNumExpFive};
    std::vector<int64_t> reshape_shape = {kNumExpFive, kNumExpSeven};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, view_shape, "view");
    Tensor output1(DT_FP32, view_shape, "output");
    Tensor sqrt(DT_FP32, view_shape, "sqrt");
    Tensor output2(DT_FP32, reshape_shape, "sqrt");

    config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
    FUNCTION("STCase2")
    {
        TileShape::Current().SetVecTile(tile_shape);
        exp = Exp(input);
        view = View(exp, view_shape, {kNumZero, kNumZero});
        output1 = Reciprocal(view);
        sqrt = Sqrt(view);
        output2 = Reshape(sqrt, reshape_shape);
    }
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
}

TEST_F(TestExpandFunctionPass, ExpandFunctionSTest2)
{
    PassManager& passManager = PassManager::Instance();
    ConstructGraphST2();
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
                                                                   {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                                               });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int exp_num = kNumZero;
    int sqrt_num = kNumZero;
    int reshape_num = kNumZero;
    int reciprocal_num = kNumZero;
    int slice_num = kNumZero;
    int contract_num = kNumZero;
    int view_num = kNumZero;
    int assemble_num = kNumZero;
    for (const auto& op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_EXP) {
            exp_num++;
        } else if (op.GetOpcode() == Opcode::OP_SLICE) {
            slice_num++;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            contract_num++;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
            EXPECT_TRUE(op.HasAttribute(OpAttributeKey::isGlobalInput));
            EXPECT_TRUE(op.GetBoolAttribute(OpAttributeKey::isGlobalInput));
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_num++;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_num++;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrt_num++;
        } else if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            reciprocal_num++;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(assemble_num, kNumOne);
    EXPECT_GT(slice_num, kNumZero);
    EXPECT_EQ(slice_num, contract_num);
    EXPECT_EQ(exp_num, kNumFour);
    EXPECT_EQ(reshape_num, kNumOne);
    EXPECT_EQ(sqrt_num, kNumFour);
    EXPECT_EQ(reciprocal_num, kNumFour);
}

/*
TESTExpandFunctionLoop
inCast{64,64}->assemble->view->outCast{64,64}
             <-assemble<-
loop will be detected
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest6)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto op_attr = CreateAssembleOpAttr();
    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {outCast});
    assemble_op.SetOpAttribute(op_attr);

    auto& assemble_op_loop = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {outCast},
                                                              {inCast});
    assemble_op_loop.SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.DefaultEnabledPreCheck(*currFunctionPtr), FAILED);

    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);
    EXPECT_EQ(expandfunctionpass.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(TestExpandFunctionPass, DisableCombineAxisOnA5)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    currFunctionPtr->paramConfigs_.combineAxis = true;
    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->paramConfigs_.combineAxis, true);
}

TEST_F(TestExpandFunctionPass, TestScopeIdMinusOneWithMergeFlag)
{
    RunScopeInfoTest({"in1", "in2", "out1"}, 2, {Opcode::OP_ADD}, {{"in1", "in2"}}, {{"out1"}}, {"add1"}, FAILED,
                     {{"add1", -1, true, false}});
}

TEST_F(TestExpandFunctionPass, TestConflictingScopeInfoSettings)
{
    RunScopeInfoTest({"in1", "in2", "in3", "out1", "out2"}, 3, {Opcode::OP_ADD, Opcode::OP_ADD},
                     {{"in1", "in2"}, {"in2", "in3"}}, {{"out1"}, {"out2"}}, {"add1", "add2"}, FAILED,
                     {{"add1", 1, true, false}, {"add2", 1, false, true}});
}

TEST_F(TestExpandFunctionPass, TestPassScopeInfoSettingsVerify)
{
    RunScopeInfoTest({"in1", "in2", "in3", "out1", "out2"}, 3, {Opcode::OP_ADD, Opcode::OP_ADD},
                     {{"in1", "in2"}, {"in2", "in3"}}, {{"out1"}, {"out2"}}, {"add1", "add2"}, SUCCESS,
                     {{"add1", 1, true, false}, {"add2", 1, true, false}});
}

TEST_F(TestExpandFunctionPass, PreCheckForDisorderIndexOutcast)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape,
                           {"src", "index1", "dst", "index2", "result1", "result2", "tensor1", "outcast1", "outcast2"}),
              true);
    std::vector<Opcode> opLists{Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, Opcode::OP_ASSEMBLE,
                                Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> iOperands{
        {"src", "index1", "dst"}, {"src", "index2", "dst"}, {"result1"}, {"result2"}, {"tensor1"}};
    std::vector<std::vector<std::string>> oOperands{{"result1"}, {"result2"}, {"outcast1"}, {"tensor1"}, {"outcast2"}};
    std::vector<std::string> opNames{"OP_INDEX_OUTCAST_1", "OP_INDEX_OUTCAST_2", "OP_ASSEMBLE_1", "OP_ADDS_1",
                                     "OP_ASSEMBLE_2"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"src", "index1", "dst", "index2"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast1", "outcast2"}), true);

    Function* function = G.GetFunction();

    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.PreRun(*function), SUCCESS);
}

/*
    Tensor is used by both OP_VIEW and another operation (conflict scenario)
    tensor -> view -> ...
    tensor -> add -> ...
    This should fail CheckInplaceOperationConflict because tensor serves both view and other operations
*/
TEST_F(TestExpandFunctionPass, PreCheckForViewConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape,
                           {"tensor", "view_output", "add_output1", "add_output2", "other_input"}),
              true);

    std::vector<Opcode> opLists{Opcode::OP_VIEW, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"tensor", "other_input"}};
    std::vector<std::vector<std::string>> oOperands{{"view_output"}, {"add_output1"}};
    std::vector<std::string> opNames{"OP_VIEW_1", "OP_ADD_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor", "other_input"}), true);
    EXPECT_EQ(G.SetOutCast({"view_output", "add_output1"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), FAILED);
}

/*
    Tensor is used by both OP_RESHAPE and another operation (conflict scenario)
    tensor -> reshape -> ...
    tensor -> mul -> ...
    This should fail CheckInplaceOperationConflict because tensor serves both reshape and other operations
*/
TEST_F(TestExpandFunctionPass, PreCheckForReshapeConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor", "reshape_output", "mul_output", "other_input"}),
              true);

    std::vector<Opcode> opLists{Opcode::OP_RESHAPE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"tensor", "other_input"}};
    std::vector<std::vector<std::string>> oOperands{{"reshape_output"}, {"mul_output"}};
    std::vector<std::string> opNames{"OP_RESHAPE_1", "OP_MUL_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor", "other_input"}), true);
    EXPECT_EQ(G.SetOutCast({"reshape_output", "mul_output"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), FAILED);
}

/*
    Tensor is used only by OP_VIEW (no conflict scenario)
    tensor -> view -> adds
    This should succeed because tensor only serves view operation (tensor has only one consumer)
*/
TEST_F(TestExpandFunctionPass, PreCheckForViewNoConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor", "view_output", "final_output"}), true);

    std::vector<Opcode> opLists{Opcode::OP_VIEW, Opcode::OP_ADDS};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"view_output"}};
    std::vector<std::vector<std::string>> oOperands{{"view_output"}, {"final_output"}};
    std::vector<std::string> opNames{"OP_VIEW_1", "OP_ADDS_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor"}), true);
    EXPECT_EQ(G.SetOutCast({"final_output"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), SUCCESS);
}

/*
    Tensor is used only by OP_RESHAPE (no conflict scenario)
    tensor -> reshape -> exp
    This should succeed because tensor only serves reshape operation (tensor has only one consumer)
*/
TEST_F(TestExpandFunctionPass, PreCheckForReshapeNoConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor", "reshape_output", "final_output"}), true);

    std::vector<Opcode> opLists{Opcode::OP_RESHAPE, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"reshape_output"}};
    std::vector<std::vector<std::string>> oOperands{{"reshape_output"}, {"final_output"}};
    std::vector<std::string> opNames{"OP_RESHAPE_1", "OP_EXP_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor"}), true);
    EXPECT_EQ(G.SetOutCast({"final_output"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), SUCCESS);
}

/*
 * VIEW/ASSEMBLE whose tiled shape contains dynamic dim (-1) must not tile-expand;
 * otherwise TiledView/TiledAssemble loop over shape[dim]==-1 emits zero ops and breaks the graph.
 */
TEST_F(TestExpandFunctionPass, SkipExpandViewAndAssembleWithDynamicShape)
{
    std::vector<int64_t> dynShape = {-1, kNumExpSix};
    std::vector<int64_t> tileShape = {kNumExpFive, kNumExpFive};

    // ----- view: output shape has -1 -> keep OP_VIEW -----
    {
        auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSkipExpandViewDyn",
                                                          "TestSkipExpandViewDyn", nullptr);
        auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape, CreateTestConstIntVector(dynShape));
        auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape,
                                                                  CreateTestConstIntVector(dynShape));
        PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inCast}, {outCast});

        std::shared_ptr<Operation> viewOp;
        for (auto& opPtr : currFunctionPtr->Operations().operations_) {
            if (opPtr->GetOpcode() == Opcode::OP_VIEW) {
                viewOp = opPtr;
            }
        }
        ASSERT_NE(viewOp, nullptr);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{kNumZero, kNumZero}));
        viewOp->tileShape_.SetVecTile(tileShape);

        currFunctionPtr->inCasts_.push_back(inCast);
        currFunctionPtr->outCasts_.push_back(outCast);
        currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

        ExpandFunction expandPass;
        EXPECT_EQ(expandPass.RunOnFunction(*currFunctionPtr), SUCCESS);

        uint32_t viewNum = kNumZero;
        uint32_t sliceNum = kNumZero;
        uint32_t contractNum = kNumZero;
        for (auto& op : currFunctionPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                ++viewNum;
            } else if (op.GetOpcode() == Opcode::OP_SLICE) {
                ++sliceNum;
            } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
                ++contractNum;
            }
        }
        EXPECT_EQ(viewNum, kNumOne);
        EXPECT_EQ(sliceNum, kNumZero);
        EXPECT_EQ(contractNum, kNumZero);
    }

    // ----- assemble: input shape has -1 -> keep OP_ASSEMBLE -----
    {
        auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSkipExpandAssembleDyn",
                                                          "TestSkipExpandAssembleDyn", nullptr);
        auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape, CreateTestConstIntVector(dynShape));
        auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape,
                                                                  CreateTestConstIntVector(dynShape));
        PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {outCast});

        std::shared_ptr<Operation> assembleOp;
        for (auto& opPtr : currFunctionPtr->Operations().operations_) {
            if (opPtr->GetOpcode() == Opcode::OP_ASSEMBLE) {
                assembleOp = opPtr;
            }
        }
        ASSERT_NE(assembleOp, nullptr);
        assembleOp->SetOpAttribute(CreateAssembleOpAttr());
        assembleOp->tileShape_.SetVecTile(tileShape);

        currFunctionPtr->inCasts_.push_back(inCast);
        currFunctionPtr->outCasts_.push_back(outCast);
        currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

        ExpandFunction expandPass;
        EXPECT_EQ(expandPass.RunOnFunction(*currFunctionPtr), SUCCESS);

        uint32_t assembleNum = kNumZero;
        uint32_t sliceNum = kNumZero;
        uint32_t contractNum = kNumZero;
        for (auto& op : currFunctionPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                ++assembleNum;
            } else if (op.GetOpcode() == Opcode::OP_SLICE) {
                ++sliceNum;
            } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
                ++contractNum;
            }
        }
        EXPECT_EQ(assembleNum, kNumOne);
        EXPECT_EQ(sliceNum, kNumZero);
        EXPECT_EQ(contractNum, kNumZero);
    }
}

TEST_F(TestExpandFunctionPass, ExpandFunctionTest)
{
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                 });

    std::vector<int64_t> shape{kNumExpSix, kNumExpSix};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);

    FUNCTION("A") { c = Div(a, b); }

    std::string jsonFilePath = "./config/pass/json/expand_function.json";
    auto programJson = Program::GetInstance().DumpJson();
    DumpJsonFile(programJson, jsonFilePath);
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    Function* currentFunction = Program::GetInstance().GetCurrentFunction();

    auto opListBefore = currentFunction->Operations().DuplicatedOpList();
    int divNumBefore = 0;
    int divNumAfter = 0;
    for (auto& op : opListBefore) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumBefore++;
        }
    }
    ExpandFunction expandFunction;
    expandFunction.RunOnFunction(*currentFunction);
    auto opListAfter = currentFunction->Operations().DuplicatedOpList();
    for (auto& op : opListAfter) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumAfter++;
        }
    }
    EXPECT_EQ(divNumBefore, kNumOne);
    EXPECT_EQ(divNumAfter, kNumFour);
}

TEST_F(TestExpandFunctionPass, ViewDerivesVecTileFromCubeMatmulConsumer)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ViewCubeTile", "ViewCubeTile", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ViewCubeTile", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    constexpr int64_t kM = 64;
    constexpr int64_t kK = 96;
    constexpr int64_t kN = 96;
    constexpr int64_t kCubeM = 64;
    constexpr int64_t kCubeK = 32;
    constexpr int64_t kCubeN = 64;

    const std::vector<int64_t> shapeA = {kM, kK};
    const std::vector<int64_t> shapeB = {kK, kN};
    const std::vector<int64_t> viewShapeA = {kM, kCubeK};
    const std::vector<int64_t> viewShapeB = {kCubeK, kN};
    const std::vector<int64_t> shapeC = {kM, kN};
    const std::vector<int64_t> offset0 = {0, 0};

    auto incastA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto incastB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto viewOutA = IRBuilder().CreateTensorVar(DT_FP32, viewShapeA, SymbolicScalar::FromConcrete(viewShapeA));
    auto viewOutB = IRBuilder().CreateTensorVar(DT_FP32, viewShapeB, SymbolicScalar::FromConcrete(viewShapeB));
    auto matmulOut = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));

    auto& viewA = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastA}, {viewOutA});
    SetViewOpAttribute(viewA, offset0, viewShapeA);
    auto& viewB = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastB}, {viewOutB});
    SetViewOpAttribute(viewB, offset0, viewShapeB);

    TileShape::Current().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    TileShape::Current().GetVecTile().tile.clear();
    auto& matmul = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {viewOutA, viewOutB}, {matmulOut});
    matmul.GetTileShapeForSetting().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    matmul.GetTileShapeForSetting().GetVecTile().tile.clear();

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);

    EXPECT_EQ(CountInputSliceOpsWithSecondDimOffset(*func, incastB, kCubeN), 1);
}

TEST_F(TestExpandFunctionPass, AssembleDerivesVecTileFromCubeMatmulProducer)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "AssembleCubeTile", "AssembleCubeTile", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("AssembleCubeTile", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    constexpr int64_t kM = 64;
    constexpr int64_t kK = 32;
    constexpr int64_t kN = 256;
    constexpr int64_t kCubeM = 64;
    constexpr int64_t kCubeK = 32;
    constexpr int64_t kCubeN = 128;

    const std::vector<int64_t> shapeA = {kM, kK};
    const std::vector<int64_t> shapeB = {kK, kN};
    const std::vector<int64_t> shapeC = {kM, kN};
    const std::vector<int64_t> shapeOut = {kM, kN};
    const std::vector<int64_t> offset0 = {0, 0};
    const std::vector<int64_t> expectedSliceShape = {kCubeM, kCubeN};

    auto incastA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto incastB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto matmulOut = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, shapeOut, SymbolicScalar::FromConcrete(shapeOut));

    TileShape::Current().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    TileShape::Current().SetVecTile(kCubeM, kCubeM);
    auto& matmul = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {incastA, incastB}, {matmulOut});
    matmul.GetTileShapeForSetting().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    matmul.GetTileShapeForSetting().GetVecTile().tile.clear();

    auto& assemble = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ASSEMBLE, {matmulOut}, {outCast});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset0));
    assemble.GetTileShapeForSetting().GetVecTile().tile.clear();

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);

    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, matmulOut, expectedSliceShape), 2);
    EXPECT_EQ(CountInputSliceOpsWithSecondDimOffset(*func, matmulOut, kCubeN), 1);
    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, matmulOut, {kCubeM, kCubeM}), 0);
}

} // namespace tile_fwk
} // namespace npu
