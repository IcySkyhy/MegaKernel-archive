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
 * \file test_function_converage.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <iostream>
#include <memory>
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "interface/function/function.h"
#undef private
#undef protected
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/id_gen.h"

using namespace npu::tile_fwk;

class FunctionCoverageTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "FunctionCoverageTest SetUpTestCase" << std::endl; }

    static void TearDownTestCase() { std::cout << "FunctionCoverageTest TearDownTestCase" << std::endl; }

    void SetUp() override
    {
        std::cout << "FunctionCoverageTest SetUp" << std::endl;
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        std::cout << "FunctionCoverageTest TearDown" << std::endl;
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

TEST_F(FunctionCoverageTest, ConverageCase4)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{16, 64};
    std::vector<int64_t> childShape{16, 16};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int LOOP_END = 4;
    constexpr int CHILD_SHAPE_OFFSET = 16;

    FUNCTION("main", {a, b}, {c})
    {
        LOOP("D3", FunctionType::DYNAMIC_LOOP, k, LoopRange(0, LOOP_END))
        {
            auto a0 = View(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = View(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            LOOP("D4", FunctionType::DYNAMIC_LOOP, n, LoopRange(0, LOOP_END))
            {
                auto d0 = View(a, childShape, {0, n * CHILD_SHAPE_OFFSET});
                auto e0 = View(b, childShape, {0, n * CHILD_SHAPE_OFFSET});
                auto f0 = Add(a0, b0);
                Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            }
            auto func = Program::GetInstance().GetCurrentFunction();
            EXPECT_EQ(func->InsertLoopIdxNameList("i"), true);
            EXPECT_EQ(func->InsertLoopIdxNameList("i"), true);
            EXPECT_EQ(func->InsertLoopIdxNameList("k"), false);
            EXPECT_EQ(func->InsertLoopIdxNameList("n"), true);
        }
    }
}

TEST_F(FunctionCoverageTest, TestReuseTensorCase1)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    config::SetBuildStatic(true);
    FUNCTION("R1")
    {
        Tensor in0 = Exp(input);
        Tensor in1 = Reciprocal(in0);
        Tensor in2 = Exp(in1);
        Tensor in3 = Exp(in2);
        Tensor in4 = Sqrt(in3);
        output = Exp(in4);
    }

    const Function* const_func = Program::GetInstance().GetFunctionByRawName("TENSOR_R1");
    ASSERT_NE(const_func, nullptr);
    Function* func = const_cast<Function*>(const_func);
    ASSERT_NE(func, nullptr);
    std::cout << "=========" << std::endl;
    Operation* re_op = nullptr;
    Operation* sqrt_op = nullptr;
    for (auto& op : func->Operations()) {
        std::cout << "Op:" << op.opmagic << " " << op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor>& input_tensor : op.GetIOperands()) {
            for (const auto& item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor>& output_tensor : op.GetOOperands()) {
            for (const auto& item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << std::endl;
        if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            re_op = &op;
        }
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrt_op = &op;
        }
    }
    std::cout << "=========" << std::endl;
    ASSERT_NE(re_op, nullptr);
    ASSERT_NE(sqrt_op, nullptr);
    LogicalTensorPtr src_tensor = re_op->GetOutputOperand(0);
    LogicalTensorPtr dst_tensor = sqrt_op->GetOutputOperand(0);
    // EXPECT_EQ(func->TensorReuse(dst_tensor, src_tensor), false);
    // src_tensor->offset = {0, 0};
    // dst_tensor->offset = {0, 0};
    EXPECT_EQ(func->TensorReuse(dst_tensor, src_tensor), true);
    EXPECT_EQ(src_tensor->tensor, dst_tensor->tensor);
    // EXPECT_EQ(sqrt_op->GetInCtrlOperations(), re_op->GetInCtrlOperations());
    EXPECT_EQ(sqrt_op->GetInCtrlOperations().empty(), true);
}

TEST_F(FunctionCoverageTest, TestFunctionHash)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    config::SetBuildStatic(true);
    FUNCTION("R2")
    {
        Tensor in0 = Exp(input);
        Tensor in1 = Reciprocal(in0);
        Tensor in2 = Exp(in1);
        Tensor in3 = Exp(in2);
        Tensor in4 = Sqrt(in3);
        output = Exp(in4);
    }

    const Function* const_func = Program::GetInstance().GetFunctionByRawName("TENSOR_R2");
    ASSERT_NE(const_func, nullptr);
    Function* func = const_cast<Function*>(const_func);
    ASSERT_NE(func, nullptr);
}

TEST_F(FunctionCoverageTest, TestOpValidCheckViewOp)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOpValidCheckView",
                                                      "TestOpValidCheckView", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {64, 64};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    inCast->offset = {0, 0};
    outCast->offset = {0, 0};

    auto& viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {outCast});

    std::vector<int64_t> fromOffset = {0, 0};
    auto opAttr = std::make_shared<ViewOpAttribute>(fromOffset);
    viewOp.SetOpAttribute(opAttr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(attr, nullptr);
            EXPECT_EQ(op.GetIOperands().size(), 1);
            EXPECT_LE(op.GetOOperands().size(), 1);
            EXPECT_EQ(op.GetIOperands()[0]->GetOffset().size(), attr->GetFromOffset().size());
            if (!op.GetOOperands().empty()) {
                EXPECT_EQ(op.GetOOperands()[0]->GetOffset().size(), attr->GetFromOffset().size());
            }
        }
    }

    currFunctionPtr->ValidCheck();
}

TEST_F(FunctionCoverageTest, TestOpValidCheckAssembleOp)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOpValidCheckAssemble",
                                                      "TestOpValidCheckAssemble", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {64, 64};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    inCast->offset = {0, 0};
    outCast->offset = {0, 0};

    auto& assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {inCast}, {outCast});

    std::vector<int64_t> toOffset = {0, 0};
    auto opAttr = std::make_shared<AssembleOpAttribute>(toOffset);
    assembleOp.SetOpAttribute(opAttr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
            EXPECT_NE(attr, nullptr);
            EXPECT_EQ(op.GetIOperands().size(), 1);
            EXPECT_LE(op.GetOOperands().size(), 1);
            if (!op.GetIOperands().empty()) {
                EXPECT_EQ(op.GetIOperands()[0]->GetOffset().size(), attr->GetToOffset().size());
            }
            if (!op.GetOOperands().empty()) {
                EXPECT_EQ(op.GetOOperands()[0]->GetOffset().size(), attr->GetToOffset().size());
            }
        }
    }

    currFunctionPtr->ValidCheck();
}

TEST_F(FunctionCoverageTest, TestOpValidCheckProducerInOpMap)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    config::SetBuildStatic(true);
    FUNCTION("TestProducerInOpMap")
    {
        Tensor in0 = Exp(input);
        Tensor in1 = Add(in0, in0);
        Tensor in2 = Mul(in1, in0);
        output = Exp(in2);
    }

    const Function* const_func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestProducerInOpMap");
    ASSERT_NE(const_func, nullptr);
    Function* func = const_cast<Function*>(const_func);
    ASSERT_NE(func, nullptr);

    EXPECT_THROW(func->ValidCheck(), Error);
}

// Verify RunTopologicalSort produces the same order regardless of operations_ array ordering.
// The fix sorts initial ready ops by opmagic to guarantee idempotent topo sort.
TEST_F(FunctionCoverageTest, TestLightweightSortIdempotent)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output1(DT_FP32, shape, "output1");
    Tensor output2(DT_FP32, shape, "output2");

    FUNCTION("TestIdempotentSort")
    {
        // Branch A: input -> Exp -> Sqrt -> output1 (root: last Exp)
        Tensor a = Exp(input);
        Tensor b = Sqrt(a);
        output1 = Exp(b);
        // Branch B: input -> Reciprocal -> Exp -> output2 (root: last Sqrt)
        Tensor c = Reciprocal(input);
        Tensor d = Exp(c);
        output2 = Sqrt(d);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestIdempotentSort");
    ASSERT_NE(func, nullptr);

    auto baseline = func->GetLightweightSortedOperations();
    std::vector<int> baselineMagics;
    for (const auto& op : baseline) {
        baselineMagics.push_back(op->GetOpMagic());
    }
    ASSERT_FALSE(baselineMagics.empty());

    std::reverse(func->operations_.begin(), func->operations_.end());

    auto shuffled = func->GetLightweightSortedOperations();
    std::vector<int> shuffledMagics;
    for (const auto& op : shuffled) {
        shuffledMagics.push_back(op->GetOpMagic());
    }

    EXPECT_EQ(baselineMagics.size(), shuffledMagics.size());
    for (size_t i = 0; i < baselineMagics.size(); i++) {
        EXPECT_EQ(baselineMagics[i], shuffledMagics[i]) << "Topo sort order differs at index " << i;
    }
}
