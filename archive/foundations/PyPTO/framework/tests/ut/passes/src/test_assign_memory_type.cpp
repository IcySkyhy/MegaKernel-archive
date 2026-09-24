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
 * \file test_assign_memory_type.cpp
 * \brief Unit test for assign_memory_type pass.
 */

#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/data_path/assign_memory_type.h"
#include "passes/pass_mgr/pass_manager.h"
#include "computational_graph_builder.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
const int NUM_1 = 1;
const int NUM_8 = 8;
const int NUM_16 = 16;
const int NUM_32 = 32;
const int NUM_48 = 48;
const int NUM_64 = 64;
const int NUM_128 = 128;
const int NUM_256 = 256;
const int NUM_512 = 512;
const int NUM_1024 = 1024;
constexpr float F_1 = 1.0;
constexpr float F_3 = 3.0;

class AssignMemoryTypeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
    }
    void TearDown() override {}

    void SetHalfwayStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                     });
        ConfigManager::Instance();
    }

    void SetTestStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy",
                                     {
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     });
        ConfigManager::Instance();
    }

    void SetFullTestStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"AutoCast", PassName::AUTO_CAST},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"RemoveUndrivenView", PassName::REMOVE_UNDRIVEN_VIEW},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                         {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                         {"DuplicateOp", PassName::DUPLICATE_OP},
                                         {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     });
        ConfigManager::Instance();
    }

    void CheckConvertOp(const Operation& op, bool verbose = false)
    {
        /*
        1. 单输入单输出
        2. 输入/输出的tensor mem类型唯一
        3. 输入和输出的tensor的mem类型不同
        */
        EXPECT_EQ(op.GetIOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        EXPECT_EQ(op.GetOOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        auto input = op.GetIOperands().front();
        ASSERT_NE(input, nullptr) << "OP_CONVERT input is nullptr";
        auto output = op.GetOOperands().front();
        ASSERT_NE(output, nullptr) << "OP_CONVERT output is nullptr";
        auto inputMemOri = input->GetMemoryTypeOriginal();
        auto inputMemTobe = input->GetMemoryTypeToBe();
        auto outputMemOri = output->GetMemoryTypeOriginal();
        auto outputMemTobe = output->GetMemoryTypeToBe();
        if (verbose) {
            std::cout << "\t|--- iOperand " << input->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(inputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(inputMemTobe) << std::endl;
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(outputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(outputMemTobe) << std::endl;
        }
        EXPECT_EQ(inputMemOri, inputMemTobe) << "OP_CONVERT input Memory Ori should be the same as Memory Tobe.";
        EXPECT_EQ(outputMemOri, outputMemTobe) << "OP_CONVERT output Memory Ori should be the same as Memory Tobe.";
        EXPECT_NE(inputMemOri, outputMemOri) << "OP_CONVERT input should have different memory type from output.";
    }

    int CountAndCheckNewOps(Function& function, const std::vector<int64_t>& beforeMagic,
                            const std::vector<Opcode>& targetOpcodes)
    {
        int convertNum = 0;
        for (const auto& op : function.Operations()) {
            if (std::find(beforeMagic.begin(), beforeMagic.end(), op.opmagic) != beforeMagic.end()) {
                continue;
            }
            if (std::find(targetOpcodes.begin(), targetOpcodes.end(), op.GetOpcode()) == targetOpcodes.end()) {
                continue;
            }
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            CheckConvertOp(op, true);
            convertNum++;
        }
        return convertNum;
    }

    bool HasViewMemoryPath(Function* function, MemoryType from, MemoryType to)
    {
        for (const auto& op : function->Operations()) {
            if (op.GetOpcode() != Opcode::OP_VIEW || op.GetIOperands().empty() || op.GetOOperands().empty()) {
                continue;
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (input->GetMemoryTypeOriginal() == from && output->GetMemoryTypeOriginal() == to) {
                return true;
            }
        }
        return false;
    }
};

int CountMemoryPath(Function* originFunction, MemoryType from, MemoryType to);

TEST_F(AssignMemoryTypeTest, AddReshape)
{
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    Function* currFunctionPtr = nullptr;

    PROGRAM("AssignMemoryTest")
    {
        Tensor input_tensor1(DataType::DT_FP32, shape, "input_tensor1");
        Tensor input_tensor2(DataType::DT_FP32, shape, "input_tensor2");
        Tensor assemble_output(DataType::DT_FP32, shape1, "assemble_output");

        config::SetBuildStatic(true);
        SetFullTestStrategy();
        config::SetPassStrategy("AssignMemoryTypeTestStrategy");
        FUNCTION("AddReshape", {input_tensor1, input_tensor2, assemble_output})
        {
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor add_output = Add(input_tensor1, input_tensor2);
            assemble_output = Reshape(add_output, shape1);
        }

        currFunctionPtr = Program::GetInstance().GetFunctionByRawName("TENSOR_AddReshape");
        ASSERT_NE(currFunctionPtr, nullptr);

        bool hasAdd = false;
        bool hasReshape = false;
        for (const auto& op : currFunctionPtr->Operations()) {
            MemoryType expectedMemoryType = MemoryType::MEM_UNKNOWN;
            if (op.GetOpcode() == Opcode::OP_ADD) {
                hasAdd = true;
                expectedMemoryType = MemoryType::MEM_UB;
            } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
                hasReshape = true;
                expectedMemoryType = MemoryType::MEM_DEVICE_DDR;
            } else {
                continue;
            }

            for (const auto& input : op.GetIOperands()) {
                EXPECT_EQ(input->GetMemoryTypeOriginal(), expectedMemoryType)
                    << op.GetOpcodeStr() << " input should be in " << BriefMemoryTypeToString(expectedMemoryType);
            }
            for (const auto& output : op.GetOOperands()) {
                EXPECT_EQ(output->GetMemoryTypeOriginal(), expectedMemoryType)
                    << op.GetOpcodeStr() << " output should be in " << BriefMemoryTypeToString(expectedMemoryType);
            }
        }
        EXPECT_TRUE(hasAdd) << "OP_ADD should exist.";
        EXPECT_TRUE(hasReshape) << "OP_RESHAPE should exist.";
    }
}

// slice/contract的加入导致当前的校验逻辑已经失效。从after计算图来看，当前推断内存类型正常
// 考虑是否要换成full strategy， 当前halfstrategy不做split拆分回到导致图与实际存在较大差异
TEST_F(AssignMemoryTypeTest, TestVecToCubeV2)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor input1(DataType::DT_FP32, shape0, "A");
        Tensor input2(DataType::DT_FP32, shape0, "B");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestVecToCubeV2", {input1, input2, weight, out})
        {
            config::SetPassStrategy("AssignMemoryTypeTestStrategy");
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor addRes = Add(input1, input2); // 256 * 128
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(out.GetDataType(), addRes, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor sumRes = Sum(addRes, 1, true);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(mmRes, sumRes);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestVecToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        originFunction->DumpJsonFile("./config/pass/json/AssignMemoryTypeTest_TestVecToCubeV2_before0.json");
        assignMemoryType.RunOnFunction(*originFunction);
        originFunction->DumpJsonFile("./config/pass/json/AssignMemoryTypeTest_TestVecToCubeV2_after0.json");
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        auto updatedOperations = originFunction->Operations();
        int convertNum = 0;
        for (const auto& op : updatedOperations) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (std::find(beforeMagic.begin(), beforeMagic.end(), op.opmagic) != beforeMagic.end()) {
                    continue;
                }
                convertNum++;
                std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
                CheckConvertOp(op, true);
            }
        }
        constexpr int expextedConvertNum = 0;
        EXPECT_EQ(convertNum, expextedConvertNum) << "0 operations should be Convert";
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToCube)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputQ(DataType::DT_BF16, shape0, "Q");
        Tensor inputK(DataType::DT_BF16, shape0, "K");
        Tensor weight(DataType::DT_BF16, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestCubeToCube", {inputQ, inputK, weight, out})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul(out.GetDataType(), qUpdate, kUpdate, false,
                                        true); // (256 * 64) @ (64 * 256) = (256 * 256)
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Sub(QKT, Element(DataType::DT_FP32, F_3));
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToCube"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        int convertNum = CountAndCheckNewOps(*originFunction, beforeMagic, {Opcode::OP_CONVERT});
        constexpr int expextedConvertNum = 0;
        EXPECT_EQ(convertNum, expextedConvertNum) << "0 operations should be Convert";
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToCubeV2)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputQ(DataType::DT_FP32, shape0, "Q");
        Tensor inputK(DataType::DT_FP32, shape0, "K");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        config::SetBuildStatic(true);
        config::SetPassOption(ENABLE_SLICE, false);
        config::SetBuildStatic(true);
        config::SetPassOption(ENABLE_SLICE, false);
        FUNCTION("TestCubeToCubeV2", {inputQ, inputK, weight, out})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul(out.GetDataType(), qUpdate, kUpdate, false,
                                        true); // (256 * 64) @ (64 * 256) = (256 * 256)
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(QKT, Element(DataType::DT_FP32, F_1));
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestCubeToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        int convertNum = CountAndCheckNewOps(*originFunction, beforeMagic,
                                             {Opcode::OP_CONVERT, Opcode::OP_VIEW, Opcode::OP_ASSEMBLE});
        constexpr int expextedConvertNum = 0;
        EXPECT_EQ(convertNum, expextedConvertNum) << "0 operations should be Convert";
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToVec)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {NUM_64, NUM_128};
    std::vector<int64_t> shape1 = {NUM_128, NUM_64};
    std::vector<int64_t> shape2 = {NUM_64, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP32, shape0, "A1");
        Tensor inputB1(DataType::DT_FP32, shape1, "B1");
        Tensor inputA2(DataType::DT_FP32, shape0, "A2");
        Tensor inputB2(DataType::DT_FP32, shape1, "B2");
        Tensor inputV2(DataType::DT_FP32, shape0, "B2");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestCubeToVec", {inputA1, inputB1, inputA2, inputB2, inputV2, out})
        {
            TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor C1 = Matrix::Matmul(out.GetDataType(), inputA1, inputB1); // (64 * 128) @ (128 * 64) = (64 * 64)
            TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor C2 = Matrix::Matmul(out.GetDataType(), inputA2, inputB2); // (64 * 128) @ (128 * 64) = (64 * 64)
            Tensor inputV1(DataType::DT_FP32, shape0, "inputV1");
            Assemble(C1, {0, 0}, inputV1);
            Assemble(C2, {0, NUM_64}, inputV1);
            TileShape::Current().SetVecTile(NUM_64, NUM_128);
            out = Add(inputV1, inputV2);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToVec"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        constexpr int expectedL0CToDdrNum = 2;
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), expectedL0CToDdrNum)
            << "2 operations should move data from L0C to DDR";
    }
}

void GetInvalidPatternGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;
    constexpr int opMagic5 = 1006;
    constexpr int opMagic6 = 1007;
    constexpr int opMagic7 = 1008;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;
    constexpr int tensorMagic6 = 7;
    constexpr int tensorMagic7 = 8;
    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::vector<int64_t> shape2 = {8, 32};
    std::vector<int64_t> shape3 = {32, 8};
    std::shared_ptr<LogicalTensor> input_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> input_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    input_tensor1->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, CreateTestConstIntVector(shape2));
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> view_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, CreateTestConstIntVector(shape2));
    view_output2->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> reshape_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, CreateTestConstIntVector(shape3));
    reshape_output1->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> reshape_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, CreateTestConstIntVector(shape3));
    reshape_output2->SetMagic(tensorMagic5);

    std::shared_ptr<LogicalTensor> assemble_output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    assemble_output->SetMagic(tensorMagic6);

    std::shared_ptr<LogicalTensor> output_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    output_cast->SetMagic(tensorMagic7);

    auto& reshape_op0 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {input_cast},
                                                       {input_tensor1});
    reshape_op0.opmagic = opMagic0;

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor1}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{8, 0}));
    view_op2.opmagic = opMagic2;

    auto& reshape_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {view_output1},
                                                       {reshape_output1});
    reshape_op1.opmagic = opMagic3;

    auto& reshape_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {view_output2},
                                                       {reshape_output2});
    reshape_op2.opmagic = opMagic4;

    auto& assemble_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshape_output1},
                                                        {assemble_output});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    assemble_op1.opmagic = opMagic5;

    auto& assemble_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshape_output2},
                                                        {assemble_output});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{8, 0}));
    assemble_op2.opmagic = opMagic6;

    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {assemble_output},
                                                    {output_cast});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op3.opmagic = opMagic7;

    currFunctionPtr->inCasts_.push_back(input_cast);
    currFunctionPtr->outCasts_.push_back(output_cast);
}

void CallAndVerify(std::shared_ptr<Function>& currFunctionPtr, const MemoryType type)
{
    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    std::string josnFilePath = "./config/pass/json/assign_mem_type_invalidpattern.json";
    currFunctionPtr->DumpJsonFile(josnFilePath);

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto& op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto& input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
            EXPECT_EQ(input->GetMemoryTypeOriginal(), type) << " Unexpected memory type.";
            EXPECT_EQ(input->GetMemoryTypeOriginal(), input->GetMemoryTypeToBe()) << " iOperand has two memory type.";
        }
        for (auto& output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
            EXPECT_EQ(output->GetMemoryTypeOriginal(), type) << " Unexpected memory type.";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), output->GetMemoryTypeToBe()) << " oOperand has two memory type.";
        }
    }
}

TEST_F(AssignMemoryTypeTest, InValidOpPattern)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "InValidOpPattern", "InValidOpPattern",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("InValidOpPattern", currFunctionPtr);

    GetInvalidPatternGraph(currFunctionPtr);

    CallAndVerify(currFunctionPtr, MemoryType::MEM_DEVICE_DDR);
}

void GetViewReshapeGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;

    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::vector<int64_t> shape2 = {1, 32};
    std::vector<int64_t> shape3 = {8, 32};
    std::shared_ptr<LogicalTensor> input_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> transpose_out = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    transpose_out->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape1, CreateTestConstIntVector(shape1));
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> reshape_output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape, CreateTestConstIntVector(shape));
    reshape_output->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> view_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape2, CreateTestConstIntVector(shape2));
    view_output2->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> output_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, shape3, CreateTestConstIntVector(shape3));
    output_cast->SetMagic(tensorMagic5);

    auto& transpose_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_TRANSPOSE_VNCHWCONV, {input_cast},
                                                        {transpose_out});
    transpose_op.opmagic = opMagic0;

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {transpose_out}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto& reshape_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {view_output1},
                                                      {reshape_output});
    reshape_op.opmagic = opMagic2;

    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {reshape_output},
                                                    {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op2.opmagic = opMagic3;

    auto& expand_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXPAND, {view_output2},
                                                     {output_cast});
    expand_op.opmagic = opMagic4;
}
TEST_F(AssignMemoryTypeTest, ViewReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "ViewReshape", "ViewReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("ViewReshape", currFunctionPtr);

    GetViewReshapeGraph(currFunctionPtr);
    CallAndVerify(currFunctionPtr, MemoryType::MEM_UB);
}
void L1DataMoveGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> input_cast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> op_view_L1_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> op_view_L1_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> view_out3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> view_out4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> l0a_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> l0a_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> l0b_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> l0b_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> a_mul_b_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> a_mul_b_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    // std::shared_ptr<LogicalTensor> output_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
    // CreateTestConstIntVector(shape));
    auto& view_L1_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1},
                                                       {op_view_L1_out1});
    std::vector<int> newoffset{0, 0};
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    view_L1_op1.SetOpAttribute(viewAttribute);

    auto& view_L1_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2},
                                                       {op_view_L1_out2});
    view_L1_op2.SetOpAttribute(viewAttribute);

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out1}, {view_out1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out1}, {view_out2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 32}));
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out2}, {view_out3});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out2}, {view_out4});
    view_op4.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 0}));

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {view_out1}, {l0a_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {view_out2}, {l0a_out2});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {view_out3}, {l0b_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {view_out4}, {l0b_out2});

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a_out1, l0b_out1}, {a_mul_b_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a_out2, l0b_out2}, {a_mul_b_out2});

    currFunctionPtr->inCasts_.push_back(input_cast1);
    currFunctionPtr->inCasts_.push_back(input_cast2);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out1);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out2);
}
TEST_F(AssignMemoryTypeTest, L1DataMove)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "L1DataMove", "L1DataMove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("L1DataMove", currFunctionPtr);

    L1DataMoveGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assign_mem_type_L1DataMove_before.json");
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assign_mem_type_L1DataMove_after.json");
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        } else {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            auto mem_to = viewOpAttribute->GetTo();
            if (mem_to != MemoryType::MEM_L1) {
                continue;
            } else {
                EXPECT_EQ(op.GetIOperands().size(), 1) << "View op has more than one input!";
                EXPECT_EQ(op.GetOOperands().size(), 1) << "View op has more than one output!";
                auto input = op.GetIOperands().front();
                auto output = op.GetOOperands().front();
                std::cout << "\t|--- MEM_L1 VIEW iOperand " << input->GetMagic() << std::endl;
                std::cout << "\t|--- MEM_L1 VIEW oOperand " << output->GetMagic() << std::endl;
                // EXPECT_EQ(input->GetMemoryTypeToBe(),MemoryType::MEM_L1) << "View op input has unexpected memory
                // type!";
                EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_L1)
                    << "View op input has unexpected memory type!";
            }
        }
    }
}

void AssignViewTensorWithAttr(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> view_in1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_in2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_in3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_in4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> output = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in1}, {tensor1});
    auto viewAttribute1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute1->SetToType(MemoryType::MEM_L1);
    view_op1.SetOpAttribute(viewAttribute1);
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {view_out1});
    auto viewAttribute2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute2->SetToType(MemoryType::MEM_BT);
    view_op2.SetOpAttribute(viewAttribute2);
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in2}, {tensor2});
    auto viewAttribute3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute3->SetToType(MemoryType::MEM_L1);
    view_op3.SetOpAttribute(viewAttribute3);
    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor2}, {view_out2});
    auto viewAttribute4 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute4->SetToType(MemoryType::MEM_FIX_QUANT_PRE);
    view_op4.SetOpAttribute(viewAttribute4);
    auto& view_op5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in3}, {tensor3});
    auto viewAttribute5 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute5->SetToType(MemoryType::MEM_L1);
    view_op5.SetOpAttribute(viewAttribute5);
    auto& view_op6 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor3}, {view_out3});
    auto viewAttribute6 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute6->SetToType(MemoryType::MEM_L0A);
    view_op6.SetOpAttribute(viewAttribute6);
    auto& view_op7 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in4}, {tensor4});
    auto viewAttribute7 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute7->SetToType(MemoryType::MEM_L1);
    view_op7.SetOpAttribute(viewAttribute7);
    auto& view_op8 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor4}, {view_out4});
    auto viewAttribute8 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute8->SetToType(MemoryType::MEM_L0B);
    view_op8.SetOpAttribute(viewAttribute7);

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {view_out3, view_out4, view_out1, view_out2},
                                   {output});

    currFunctionPtr->inCasts_.push_back(view_in1);
    currFunctionPtr->inCasts_.push_back(view_in2);
    currFunctionPtr->inCasts_.push_back(view_in3);
    currFunctionPtr->inCasts_.push_back(view_in4);
    currFunctionPtr->outCasts_.push_back(output);
}

TEST_F(AssignMemoryTypeTest, TestViewWithAttr)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewWithAttr", "TestViewWithAttr",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestViewWithAttr", currFunctionPtr);

    AssignViewTensorWithAttr(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assignMemoryType_TestViewWithAttr_before.json");
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assignMemoryType_TestViewWithAttr_after.json");
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            MemoryType attrToType = viewOpAttribute->GetTo();
            auto output = op.GetOOperands().front();
            auto outputMemOri = output->GetMemoryTypeOriginal();
            auto outputMemTobe = output->GetMemoryTypeToBe();
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(outputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(outputMemTobe) << std::endl;
            EXPECT_EQ(attrToType, outputMemOri);
            EXPECT_EQ(attrToType, outputMemTobe);
        }
    }
}

TEST_F(AssignMemoryTypeTest, TestPostcheckFailWhenTensorMemUnknown)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPostcheckFailWhenTensorMemUnknown",
                                                      "TestPostcheckFailWhenTensorMemUnknown", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestPostcheckFailWhenTensorMemUnknown", currFunctionPtr);
    AssignViewTensorWithAttr(currFunctionPtr);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(AssignMemoryTypeTest, TestPostcheckFailWhenPathUnreachable)
{
    std::vector<int64_t> shape1{NUM_32, NUM_32};
    std::vector<int64_t> shape2{NUM_64, NUM_64};
    std::vector<int64_t> shape3{NUM_128, NUM_128};
    ComputationalGraphBuilder G;

    G.AddTensor(DataType::DT_FP32, shape3, "input");
    auto tensorInput = G.GetTensor("input");
    tensorInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape2, "a");
    auto tensorA = G.GetTensor("a");
    tensorA->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(DataType::DT_FP32, shape1, "b");
    auto tensorB = G.GetTensor("b");
    tensorB->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape3, "output");
    auto tensorOutput = G.GetTensor("output");
    tensorOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_VIEW, {"input"}, {"a"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape3, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"b"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape2, MemoryType::MEM_DEVICE_DDR));
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"output"}, "assemble1");
    G.GetOp("assemble1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, shape2));

    G.SetInCast({"input"});
    G.SetOutCast({"output"});

    Function* function = G.GetFunction();

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*function), FAILED);
}

TEST_F(AssignMemoryTypeTest, UnalignedAssembleBeforeReshapeFallbackDdr)
{
    ComputationalGraphBuilder G;
    Shape shape{NUM_16, NUM_32};
    Shape reshapeShape{NUM_32, NUM_16};
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "vec_dup_out");
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "assemble_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "reshape_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "view_out");

    G.AddOp(Opcode::OP_VEC_DUP, {}, {"vec_dup_out"}, "vec_dup");
    G.AddOp(Opcode::OP_CONTRACT, {"vec_dup_out"}, {"assemble_out"}, "assemble");
    G.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 1}));
    G.AddOp(Opcode::OP_RESHAPE, {"assemble_out"}, {"reshape_out"}, "reshape");
    G.AddOp(Opcode::OP_SLICE, {"reshape_out"}, {"view_out"}, "view");
    G.GetOp("view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_UB));

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    func->DumpJsonFile("./config/pass/json/assignMemoryType_UnalignedAssembleBeforeReshapeFallbackDdr_before2.json");
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    func->DumpJsonFile("./config/pass/json/assignMemoryType_UnalignedAssembleBeforeReshapeFallbackDdr_after2.json");
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    EXPECT_EQ(G.GetTensor("vec_dup_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(G.GetTensor("assemble_out")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetTensor("reshape_out")->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(G.GetTensor("view_out")->GetMemoryTypeOriginal(), MemoryType::MEM_UB);

    auto assembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(G.GetOp("assemble")->GetOpAttribute());
    ASSERT_NE(assembleOpAttr, nullptr);
    EXPECT_EQ(assembleOpAttr->GetFrom(), MemoryType::MEM_UB);
}

TEST_F(AssignMemoryTypeTest, AssembleAndReshapeAfterAssemble)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {NUM_64, NUM_32};
    std::vector<int64_t> shape1 = {NUM_32, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor input1(DataType::DT_FP32, shape0, "In1");
        Tensor input2(DataType::DT_FP32, shape0, "In2");
        Tensor input3(DataType::DT_FP32, shape1, "In3");
        Tensor output1(DataType::DT_FP32, shape1, "Out1");
        Tensor output2(DataType::DT_FP32, shape0, "Out2");
        SetTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("AssembleAndReshapeAfterAssemble", {input1, input2, output1, output2})
        {
            TileShape::Current().SetVecTile(NUM_256, NUM_128);
            Tensor t1 = Add(input1, input2);
            Tensor t2(DT_FP32, shape0, "t2");
            Tensor t3(DT_FP32, shape0, "t2");
            Assemble(t1, {0, 0}, t2);
            Assemble(t2, {0, 0}, t3);
            Assemble(t3, {0, 0}, output2);
            Tensor r1 = Reshape(t2, shape1);
            output1 = Add(r1, input3);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_AssembleAndReshapeAfterAssemble"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_RESHAPE) {
                EXPECT_EQ(op.iOperand[0]->GetMemoryTypeOriginal(), op.oOperand[0]->GetMemoryTypeOriginal());
            }
        }
    }
}

bool IsMatmulProducerThroughNoOpAssemble(Operation* producerOp)
{
    constexpr int MAX_TRACE_DEPTH = 8;
    for (int depth = 0; producerOp != nullptr && depth < MAX_TRACE_DEPTH; ++depth) {
        auto opcode = producerOp->GetOpcode();
        if (opcode == Opcode::OP_A_MUL_B || opcode == Opcode::OP_A_MULACC_B) {
            return true;
        }
        if (opcode != Opcode::OP_ASSEMBLE && opcode != Opcode::OP_CONTRACT) {
            return false;
        }
        if (producerOp->GetIOperands().empty() || producerOp->GetOOperands().empty()) {
            return false;
        }
        auto input = producerOp->GetIOperands().front();
        auto output = producerOp->GetOOperands().front();
        if (input == nullptr || output == nullptr ||
            input->GetMemoryTypeOriginal() != output->GetMemoryTypeOriginal() || input->GetProducers().empty()) {
            return false;
        }
        producerOp = *input->GetProducers().begin();
    }
    return false;
}

int CountL0c2l1Num(Function* originFunction)
{
    int l0c2l1Count = 0;
    for (auto& op : originFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT || op.GetOpcode() == Opcode::OP_CONVERT ||
            op.GetOpcode() == Opcode::OP_SLICE) {
            if (op.GetIOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
                op.GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
                l0c2l1Count++;
                EXPECT_TRUE(IsMatmulProducerThroughNoOpAssemble(*op.ProducerOps().begin()));
            }
        }
    }
    return l0c2l1Count;
}

int CountMemoryPath(Function* originFunction, MemoryType from, MemoryType to)
{
    int pathCount = 0;
    for (auto& op : originFunction->Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT && op.GetOpcode() != Opcode::OP_CONVERT &&
            op.GetOpcode() != Opcode::OP_SLICE && op.GetOpcode() != Opcode::OP_VIEW &&
            op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
            continue;
        }
        if (op.GetIOperands().front()->GetMemoryTypeOriginal() == from &&
            op.GetOOperands().front()->GetMemoryTypeOriginal() == to) {
            ++pathCount;
        }
    }
    return pathCount;
}

struct L0C2UBTestShapes {
    std::vector<int64_t> shapeA = {NUM_64, NUM_128};
    std::vector<int64_t> shapeB = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC = {NUM_64, NUM_64};
};

L0C2UBTestShapes PrepareA5Platform()
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    return {};
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1EqualShape)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1EqualShape", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1EqualShape"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 2);
    }
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1ParallelDdrFallback)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC1 = {NUM_64, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor outC1(DataType::DT_FP16, shapeC1, "C1");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1ParallelDdrFallback", {inputA1, inputB1, inputA2, outC1, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC1.GetDataType(), inputA1, inputB1);
            Assemble(inputB2, {0, 0}, outC1);
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestL0C2L1ParallelDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_L1), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 1);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1), 1);
    }
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1LargeToSmall)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_32};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1LargeToSmall", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_32, NUM_32});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 32) = (64 * 32)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 32) = (128 * 32)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1LargeToSmall"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 4);
    }
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1SmallToLarge)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_32};
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1SmallToLarge", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 32) = (64 * 32)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_32, NUM_32});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 32) = (128 * 32)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1SmallToLarge"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 4);
    }
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1Dim0Mismatch)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC1 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC1, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1Dim0Mismatch", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_16, NUM_16}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1Dim0Mismatch"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

// 当tensor存在非立即数dynValidShape时，直接走L0C2L1通道进行后续矩阵乘会有精度问题，当前走DDR规避
TEST_F(AssignMemoryTypeTest, TestL0C2L1WithNonImmediateValidShape)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC1 = {NUM_64, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        config::SetPassStrategy("AssignMemoryTypeTestStrategy");
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1WithNonImmediateValidShape", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor matmul1Output = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1);
            matmul1Output.GetStorage()->UpdateDynValidShape({CreateTestScalarVar("dim0"), CreateTestScalarVar("dim1")});

            Tensor viewL1Output = View(matmul1Output, shapeC1, {0, 0});
            auto& viewL1Op = **viewL1Output.GetStorage()->GetProducers().begin();
            std::dynamic_pointer_cast<ViewOpAttribute>(viewL1Op.GetOpAttribute())->SetToType(MemoryType::MEM_L1);

            Tensor viewL0AOutput = View(viewL1Output, shapeC1, {0, 0});
            auto& viewL0AOp = **viewL0AOutput.GetStorage()->GetProducers().begin();
            std::dynamic_pointer_cast<ViewOpAttribute>(viewL0AOp.GetOpAttribute())->SetToType(MemoryType::MEM_L0A);

            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_32, NUM_32}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, viewL0AOutput);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestL0C2L1WithNonImmediateValidShape");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";

        int l0c2l1Count = 0;
        int ddr2l1Count = 0;
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_SLICE) {
                if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
                    continue;
                }
                auto inputMem = op.GetIOperands().front()->GetMemoryTypeOriginal();
                auto outputMem = op.GetOOperands().front()->GetMemoryTypeOriginal();
                if (inputMem == MemoryType::MEM_L0C && outputMem == MemoryType::MEM_L1) {
                    l0c2l1Count++;
                }
                if (inputMem == MemoryType::MEM_DEVICE_DDR && outputMem == MemoryType::MEM_L1) {
                    ddr2l1Count++;
                }
            }
        }
        EXPECT_EQ(l0c2l1Count, 0) << "Should not use L0C2L1 path when validShape is non-immediate";
        EXPECT_GT(ddr2l1Count, 0) << "Should use DDR2L1 path when validShape is non-immediate";
    }
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1UnsupportDataType)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        Tensor inputA1(DataType::DT_FP32, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP32, shapeA2, "A2");
        Tensor inputB1(DataType::DT_FP32, shapeB1, "B1");
        Tensor outC2(DataType::DT_FP32, shapeC2, "C2");

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1UnsupportDataType", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1UnsupportDataType"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1UnsupportDataShape)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1UnsupportDataShape", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_8, NUM_8}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1UnsupportDataShape"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

TEST_F(AssignMemoryTypeTest, TestL0C2L1NoSupportNotMultipleCase)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeA2 = {NUM_128, NUM_64};
    std::vector<int64_t> shapeC2 = {NUM_128, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        Tensor inputA1(DataType::DT_FP16, shapeA1, "A1");
        Tensor inputB1(DataType::DT_FP16, shapeB1, "B1");
        Tensor inputA2(DataType::DT_FP16, shapeA2, "A2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2L1NoSupportNotMultipleCase", {inputA1, inputB1, inputA2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_48, NUM_48}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            Tensor inputB2 = Matrix::Matmul(outC2.GetDataType(), inputA1, inputB1); // (64 * 32) @ (32 * 16) = (64 * 16)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_16, NUM_16});
            outC2 = Matrix::Matmul(outC2.GetDataType(), inputA2, inputB2); // (128 * 64) @ (64 * 16) = (128 * 16)
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2L1NoSupportNotMultipleCase"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        EXPECT_EQ(CountL0c2l1Num(originFunction), 0);
    }
}

TEST_F(AssignMemoryTypeTest, TestCascadingAssembleViewNoDDR2L0C)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shapeA1 = {NUM_16, NUM_32};
    std::vector<int64_t> shapeB1 = {NUM_32, NUM_64};
    std::vector<int64_t> shapeC1 = {NUM_16, NUM_64};
    std::vector<int64_t> shapeT1 = {NUM_32, NUM_64};
    std::vector<int64_t> shapeT2 = {NUM_32, NUM_32};
    std::vector<int64_t> shapeA2 = {NUM_64, NUM_32};
    std::vector<int64_t> shapeB2 = {NUM_32, NUM_16};
    std::vector<int64_t> shapeC2 = {NUM_64, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA11(DataType::DT_FP16, shapeA1, "A11");
        Tensor inputB11(DataType::DT_FP16, shapeB1, "B11");
        Tensor inputA12(DataType::DT_FP16, shapeA1, "A12");
        Tensor inputB12(DataType::DT_FP16, shapeB1, "B12");
        Tensor inputA13(DataType::DT_FP16, shapeA1, "A13");
        Tensor inputB13(DataType::DT_FP16, shapeB1, "B13");
        Tensor inputA14(DataType::DT_FP16, shapeA1, "A14");
        Tensor inputB14(DataType::DT_FP16, shapeB1, "B14");
        Tensor inputB2(DataType::DT_FP16, shapeB2, "B2");
        Tensor outC2(DataType::DT_FP16, shapeC2, "C2");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestCascadingAssembleViewNoDDR2L0C",
                 {inputA11, inputB11, inputA12, inputB12, inputA13, inputB13, inputA14, inputB14, inputB2, outC2})
        {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
            Tensor C11 = Matrix::Matmul(outC2.GetDataType(), inputA11, inputB11); // (16, 32) @ (32, 64) = (16, 64)
            Tensor C12 = Matrix::Matmul(outC2.GetDataType(), inputA12, inputB12); // (16, 32) @ (32, 64) = (16, 64)
            Tensor C13 = Matrix::Matmul(outC2.GetDataType(), inputA13, inputB13); // (16, 32) @ (32, 64) = (16, 64)
            Tensor C14 = Matrix::Matmul(outC2.GetDataType(), inputA14, inputB14); // (16, 32) @ (32, 64) = (16, 64)
            Tensor T11(DT_FP16, shapeT1, "T11");                                  // (32, 64)
            Tensor T12(DT_FP16, shapeT1, "T12");                                  // (32, 64)
            Assemble(C11, {0, 0}, T11);
            Assemble(C12, {16, 0}, T11);
            Assemble(C13, {0, 0}, T12);
            Assemble(C14, {16, 0}, T12);
            Tensor T21 = View(T11, shapeT2, {0, 0}); // (32, 32)
            Tensor T22 = View(T12, shapeT2, {0, 0}); // (32, 32)
            Tensor A2(DT_FP16, shapeA2, "A2");       // (64, 32)
            Assemble(T21, {0, 0}, A2);
            Assemble(T22, {32, 0}, A2);
            outC2 = Matrix::Matmul(outC2.GetDataType(), A2, inputB2); // (64, 32) @ (32, 16) = (64, 16)
        }
        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestCascadingAssembleViewNoDDR2L0C"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        AssignMemoryType assignMemoryType;
        EXPECT_EQ(assignMemoryType.PostCheck(*originFunction),
                  SUCCESS); // postcheck中包含对DDR到L0C的不合理通路校验，直接调用
    }
}

void ConstructMultiDataLoadGraphBranch(ComputationalGraphBuilder& G, std::string name)
{
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_1, NUM_128}, MemoryType::MEM_UNKNOWN, "in" + name);
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_1, NUM_128}, MemoryType::MEM_UNKNOWN, "t1" + name);
    G.AddOp(Opcode::OP_VIEW, {"in" + name}, {"t1" + name}, "v1" + name);
    G.GetOp("v1" + name)
        ->SetOpAttribute(
            std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_1, NUM_128}, MemoryType::MEM_UNKNOWN));
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t2" + name);
    G.AddOp(Opcode::OP_RESHAPE, {"t1" + name}, {"t2" + name}, "r" + name);
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t4" + name);
    G.AddOp(Opcode::OP_SLICE, {"t2" + name}, {"t4" + name}, "v2" + name);
    G.GetOp("v2" + name)
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_128}, MemoryType::MEM_L1));
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t5" + name);
    G.AddOp(Opcode::OP_SLICE, {"t4" + name}, {"t5" + name}, "v3" + name);
}

void ConstructMultiDataLoadGraph(ComputationalGraphBuilder& G)
{
    ConstructMultiDataLoadGraphBranch(G, "a");
    ConstructMultiDataLoadGraphBranch(G, "b");
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t6");
    G.AddOp(Opcode::OP_A_MUL_B, {"t5a", "t5b"}, {"t6"}, "amulb");
    G.AddTensor(DataType::DT_FP32, {NUM_128, NUM_128}, MemoryType::MEM_UNKNOWN, "t3b");
    G.AddOp(Opcode::OP_MUL, {"t2b", "t2b"}, {"t3b"}, "mulb");
    G.GetOp("v3a")->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_128}, MemoryType::MEM_L0A));
    G.GetOp("v3b")->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(std::vector<int64_t>{NUM_128, NUM_128}, MemoryType::MEM_L0B));
}

void MultiDataLoadCheck(Function* func)
{
    for (const auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_TRUE(op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR);
        }
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_FALSE(op.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
                         op.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_L1);
        }
    }
}
TEST_F(AssignMemoryTypeTest, TestMultiDataLoad)
{
    ComputationalGraphBuilder G;
    ConstructMultiDataLoadGraph(G);
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*func), FAILED);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    MultiDataLoadCheck(func);
}
TEST_F(AssignMemoryTypeTest, TestMultiDataLoad1)
{
    ComputationalGraphBuilder G;
    ConstructMultiDataLoadGraph(G);
    G.GetOp("rb")->SetOpCode(Opcode::OP_ADDS);
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*func), FAILED);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    MultiDataLoadCheck(func);
}
TEST_F(AssignMemoryTypeTest, TestMultiDataLoad2)
{
    ComputationalGraphBuilder G;
    ConstructMultiDataLoadGraph(G);
    Shape s{NUM_128, NUM_128};
    G.GetOp("rb")->SetOpCode(Opcode::OP_A_MUL_B);
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "inb2");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t1b2");
    G.AddOp(Opcode::OP_SLICE, {"inb2"}, {"t1b2"}, "v1b2");
    G.GetOp("v1b2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L1));
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t2b2");
    G.AddOp(Opcode::OP_SLICE, {"t1b2"}, {"t2b2"}, "v2b2");
    G.GetOp("v2b2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L0A));
    G.GetTensor("t2b2")->AddConsumer(G.GetOp("rb"));
    G.GetOp("rb")->iOperand = {G.GetTensor("t1b"), G.GetTensor("t2b2")};
    // rb b
    G.GetTensor("inb")->shape = s;
    G.GetTensor("inb")->tensor->rawshape = s;
    G.GetTensor("t1b")->shape = s;
    G.GetTensor("t1b")->tensor->rawshape = s;
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t2b22");
    G.GetOp("v1b")->ReplaceInput(G.GetTensor("t2b22"), G.GetTensor("inb"));
    G.GetTensor("inb")->RemoveConsumer(G.GetOp("v1b"));
    G.AddOp(Opcode::OP_SLICE, {"inb"}, {"t2b22"}, "v2b22");
    G.GetOp("v2b22")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L1));
    G.GetOp("v1b")->SetOpCode(Opcode::OP_SLICE);
    G.GetOp("v1b")->SetOpAttribute(std::make_shared<ViewOpAttribute>(s, MemoryType::MEM_L0B));

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*func), FAILED);
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    MultiDataLoadCheck(func);
}

TEST_F(AssignMemoryTypeTest, TestMatmulL0COutputToL1ToL0A)
{
    /*
     * Before:
     *   A_MUL_B1(l0a_in, l0b_in) -> l0c_out1(MEM_L0C) -> L1_TO_L0A -> l0a_mid(MEM_L0A) -> A_MUL_B2 -> l0c_out2
     *
     * After:
     *   A_MUL_B1 -> l0c_out1(MEM_L0C) -> OP_CONVERT(L0C->L1) -> tensorL1(MEM_L1) -> L1_TO_L0A -> l0a_mid(MEM_L0A) ->
     * A_MUL_B2 -> l0c_out2
     */
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0A, "l0a_in");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0B, "l0b_in");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0C, "l0c_out1");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0A, "l0a_mid");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0B, "l0b_in2");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_16}, MemoryType::MEM_L0C, "l0c_out2");

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"l0c_out1"}, "matmul1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l0c_out1"}, {"l0a_mid"}, "l1_to_l0a");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_mid", "l0b_in2"}, {"l0c_out2"}, "matmul2");

    constexpr int producerScopeId = 42;
    G.GetOp("matmul1")->SetScopeId(producerScopeId);

    G.SetInCast({"l0a_in", "l0b_in", "l0b_in2"});
    G.SetOutCast({"l0c_out2"});

    Function* func = G.GetFunction();
    size_t beforeOpCount = func->Operations().size();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);
    auto opList = func->Operations();
    int convertNum = 0;
    bool foundL1ToL0A = false;
    for (const auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_CONVERT) {
            convertNum++;
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            EXPECT_EQ(input->GetMemoryTypeOriginal(), MemoryType::MEM_L0C) << "convert input should be MEM_L0C";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_L1) << "convert output should be MEM_L1";
            EXPECT_EQ(op.GetScopeId(), producerScopeId) << "convert should inherit scopeId from its producer (matmul1)";
        }
        if (op.GetOpcode() == Opcode::OP_L1_TO_L0A) {
            foundL1ToL0A = true;
            auto l1ToL0AInput = op.GetIOperands().front();
            EXPECT_EQ(l1ToL0AInput->GetMemoryTypeOriginal(), MemoryType::MEM_L1)
                << "L1_TO_L0A input should be reconnected to MEM_L1 tensor (convert output)";
        }
    }
    EXPECT_EQ(convertNum, 1) << "should insert exactly 1 OP_CONVERT";
    EXPECT_TRUE(foundL1ToL0A) << "L1_TO_L0A should still exist after insertion";
    EXPECT_EQ(opList.size(), beforeOpCount + 1) << "should have exactly 1 more op after insertion";
}

TEST_F(AssignMemoryTypeTest, TestAmulBInputInvalidProducer)
{
    ComputationalGraphBuilder G;
    Shape s{NUM_128, NUM_128};
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_DEVICE_DDR, "input");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_DEVICE_DDR, "temp");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_DEVICE_DDR, "output");
    G.AddOp(Opcode::OP_ADD, {"input"}, {"temp"}, "add_op");
    G.AddOp(Opcode::OP_A_MUL_B, {"temp"}, {"output"}, "a_mul_b_op");

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;

    EXPECT_EQ(assignMemoryType.PreCheck(*func), FAILED);
}

TEST_F(AssignMemoryTypeTest, TestConvertScopeId)
{
    ComputationalGraphBuilder G;
    Shape s1{NUM_16, NUM_128};
    Shape s{NUM_16, NUM_16};
    Offset o1{0, 112};
    Offset o2{0, 0};
    G.AddTensor(DataType::DT_FP32, s1, MemoryType::MEM_UNKNOWN, "input");
    G.AddTensor(DataType::DT_FP32, s1, MemoryType::MEM_UNKNOWN, "t1");
    G.AddOp(Opcode::OP_VIEW, {"input"}, {"t1"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_UNKNOWN));

    G.AddTensor(DataType::DT_FP32, s1, MemoryType::MEM_UNKNOWN, "t11");
    G.AddOp(Opcode::OP_ADDS, {"t1"}, {"t11"}, "adds1");

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t12");
    G.AddOp(Opcode::OP_VIEW, {"t11"}, {"t12"}, "view12");
    G.GetOp("view12")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o1, MemoryType::MEM_UNKNOWN));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t2");
    G.AddOp(Opcode::OP_ADDS, {"t12"}, {"t2"}, "adds");

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t3");
    G.AddOp(Opcode::OP_SLICE, {"t2"}, {"t3"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L1));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t4");
    G.AddOp(Opcode::OP_SLICE, {"t3"}, {"t4"}, "view3");
    G.GetOp("view3")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L0A));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "input_b");
    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t5");
    G.AddOp(Opcode::OP_SLICE, {"input_b"}, {"t5"}, "view4");
    G.GetOp("view4")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L1));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "t6");
    G.AddOp(Opcode::OP_SLICE, {"t5"}, {"t6"}, "view5");
    G.GetOp("view5")->SetOpAttribute(std::make_shared<ViewOpAttribute>(o2, MemoryType::MEM_L0B));

    G.AddTensor(DataType::DT_FP32, s, MemoryType::MEM_UNKNOWN, "output");
    G.AddOp(Opcode::OP_A_MUL_B, {"t4", "t6"}, {"output"}, "a_mul_b");

    constexpr int scopeId1 = 1;
    G.GetOp("adds")->SetScopeId(scopeId1);
    G.GetOp("adds1")->SetScopeId(scopeId1);

    G.SetInCast({"input", "input_b"});
    G.SetOutCast({"output"});

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            EXPECT_EQ(op.GetScopeId(), scopeId1);
        }
    }
}

TEST_F(AssignMemoryTypeTest, TestTobeMapOrdering)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape = {NUM_256, NUM_128};
    std::vector<int64_t> shape1 = {NUM_128, NUM_64};
    std::vector<int64_t> shape2 = {NUM_64, NUM_256};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP32, shape, "A");
        Tensor inputB(DataType::DT_FP32, shape, "B");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetFullTestStrategy();
        config::SetBuildStatic(true);
        FUNCTION("TestTobeMapOrdering", {inputA, inputB, weight, out})
        {
            TileShape::Current().SetCubeTile({NUM_256, NUM_256}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(out.GetDataType(), inputA, weight);
            Tensor reshapeRes = Reshape(mmRes, shape2);
            TileShape::Current().SetVecTile(NUM_256, NUM_256);
            Tensor add1Out = Add(reshapeRes, Element(DataType::DT_FP32, 1.0));
            Tensor add2Out = Add(reshapeRes, Element(DataType::DT_FP32, 2.0));
            Tensor expOut = Exp(reshapeRes);
            Tensor out1 = Add(add2Out, expOut);
            out = Add(out1, add1Out);
        }
        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestTobeMapOrdering");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        std::vector<std::pair<uint64_t, uint64_t>> tensorOpMagicPairs;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto output = op.GetOOperands().front();
            auto consumer = *output->GetConsumers().begin();
            if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
                continue;
            }
            tensorOpMagicPairs.emplace_back(output->GetMagic(), consumer->GetOpMagic());
        }
        for (size_t i = 0; i < tensorOpMagicPairs.size(); ++i) {
            for (size_t j = i + 1; j < tensorOpMagicPairs.size(); ++j) {
                if (tensorOpMagicPairs[i].second < tensorOpMagicPairs[j].second) {
                    ASSERT_LE(tensorOpMagicPairs[i].first, tensorOpMagicPairs[j].first)
                        << "TobeMap ordering violation: OpMagic " << tensorOpMagicPairs[i].second << " (TensorMagic "
                        << tensorOpMagicPairs[i].first << ") < OpMagic " << tensorOpMagicPairs[j].second
                        << " (TensorMagic " << tensorOpMagicPairs[j].first << ")";
                }
            }
        }
    }
}

TEST_F(AssignMemoryTypeTest, TestOverSizeUb)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape = {NUM_256, NUM_256};
    PROGRAM("TestOverSizeUb")
    {
        Tensor input1(DataType::DT_FP32, shape, "input1");
        Tensor input2(DataType::DT_FP32, shape, "input2");
        Tensor input3(DataType::DT_FP32, shape, "input3");
        Tensor output(DataType::DT_FP32, shape, "output");
        SetFullTestStrategy();
        config::SetBuildStatic(true);
        FUNCTION("TestOverSizeUb", {input1, input2, input3, output})
        {
            Tensor input4(DataType::DT_FP32, shape, "tmp");
            TileShape::Current().SetCubeTile({NUM_256, NUM_256}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(input4.GetDataType(), input1, input2);
            Assemble(mmRes, {0, 0}, input4);
            TileShape::Current().SetVecTile(NUM_256, NUM_256);
            output = Add(input4, input3);
        }
        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestOverSizeUb");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        int beforeViewNum = 0;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                beforeViewNum++;
            }
        }
        AssignMemoryType assignMemoryType;
        EXPECT_EQ(assignMemoryType.RunOnFunction(*originFunction), SUCCESS);
        int afterViewNum = 0;
        for (const auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                afterViewNum++;
            }
        }
        EXPECT_EQ(afterViewNum, beforeViewNum + 2);
    }
}

TEST_F(AssignMemoryTypeTest, OversizedViewInputRequirementFallback)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> fullShape = {NUM_1024, NUM_1024};
    std::vector<int64_t> viewShape = {NUM_1024, NUM_512};
    PROGRAM("OversizedViewInputRequirementFallback")
    {
        Tensor input(DataType::DT_FP32, fullShape, "input");
        Tensor output(DataType::DT_FP32, viewShape, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("OversizedViewInputRequirementFallback", {input, output})
        {
            TileShape::Current().SetVecTile(NUM_1024, NUM_1024);
            Tensor expOut = Exp(input);
            Tensor viewOut = View(expOut, viewShape, {0, 0});
            output = Exp(viewOut);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_OversizedViewInputRequirementFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_FALSE(HasViewMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR))
            << "Oversized view should not represent UB->DDR movement.";
    }
}

// input -> Exp -> Reshape -> View -> Reshape -> Exp -> output1
//                               \-> View -> Reshape -> Exp -> output2
TEST_F(AssignMemoryTypeTest, ExpReshapeDuplicatedViewReshapeExp)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    const std::vector<int64_t> tensorShape = {NUM_16, NUM_16};
    const std::vector<int64_t> reshapeShape = {NUM_8, NUM_32};
    PROGRAM("ExpReshapeDuplicatedViewReshapeExp")
    {
        Tensor input(DataType::DT_FP16, tensorShape, "input");
        Tensor output1(DataType::DT_FP16, tensorShape, "output1");
        Tensor output2(DataType::DT_FP16, tensorShape, "output2");
        SetFullTestStrategy();
        config::SetBuildStatic(true);
        FUNCTION("ExpReshapeDuplicatedViewReshapeExp", {input, output1, output2})
        {
            TileShape::Current().SetVecTile(NUM_16, NUM_16);
            Tensor expOutput = Exp(input);
            Tensor reshapeOutput = Reshape(expOutput, reshapeShape);
            Tensor viewOutput1 = View(reshapeOutput, reshapeShape, {0, 0});
            Tensor viewOutput2 = View(reshapeOutput, reshapeShape, {0, 0});
            output1 = Exp(Reshape(viewOutput1, tensorShape));
            output2 = Exp(Reshape(viewOutput2, tensorShape));
        }
    }

    Function* function = Program::GetInstance().GetFunctionByRawName("TENSOR_ExpReshapeDuplicatedViewReshapeExp");
    ASSERT_NE(function, nullptr);
    Operation* firstReshape = nullptr;
    for (auto& operation : function->Operations()) {
        if (operation.GetOpcode() != Opcode::OP_RESHAPE || operation.iOperand.size() != 1 ||
            operation.oOperand.size() != 1) {
            continue;
        }
        auto input = operation.iOperand.front();
        auto output = operation.oOperand.front();
        if (input->GetProducers().size() == 1 && (*input->GetProducers().begin())->GetOpcode() == Opcode::OP_ASSEMBLE &&
            output->GetConsumers().size() == 2) {
            firstReshape = &operation;
            break;
        }
    }
    ASSERT_NE(firstReshape, nullptr);
    EXPECT_EQ(firstReshape->iOperand.front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(firstReshape->oOperand.front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    for (auto* view : firstReshape->oOperand.front()->GetConsumers()) {
        ASSERT_EQ(view->GetOpcode(), Opcode::OP_VIEW);
        auto viewOutput = view->oOperand.front();
        EXPECT_EQ(viewOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        ASSERT_EQ(viewOutput->GetConsumers().size(), 1);
        auto* branchReshape = *viewOutput->GetConsumers().begin();
        ASSERT_EQ(branchReshape->GetOpcode(), Opcode::OP_RESHAPE);
        EXPECT_EQ(branchReshape->oOperand.front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    }
}

TEST_F(AssignMemoryTypeTest, DdrMultiReshapeWithDirectSlice)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "DdrMultiReshapeWithDirectSlice",
                                               "DdrMultiReshapeWithDirectSlice", nullptr);
    ASSERT_NE(function, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("DdrMultiReshapeWithDirectSlice", function);

    const std::vector<int64_t> tensorShape = {NUM_16, NUM_16};
    const std::vector<int64_t> reshapeShape = {NUM_8, NUM_32};
    auto tensorShapeDyn = CreateTestConstIntVector(tensorShape);
    auto reshapeShapeDyn = CreateTestConstIntVector(reshapeShape);
    auto input = IRBuilder().CreateTensorVar(DT_FP16, tensorShape, tensorShapeDyn);
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP16, tensorShape, tensorShapeDyn);
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP16, tensorShape, tensorShapeDyn);
    auto firstReshapeOutput = IRBuilder().CreateTensorVar(DT_FP16, reshapeShape, reshapeShapeDyn);
    auto viewOutput = IRBuilder().CreateTensorVar(DT_FP16, reshapeShape, reshapeShapeDyn);
    auto branchReshapeOutput = IRBuilder().CreateTensorVar(DT_FP16, tensorShape, tensorShapeDyn);
    auto branchSliceOutput = IRBuilder().CreateTensorVar(DT_FP16, tensorShape, tensorShapeDyn);
    auto directSliceOutput = IRBuilder().CreateTensorVar(DT_FP16, reshapeShape, reshapeShapeDyn);
    auto branchOutput = IRBuilder().CreateTensorVar(DT_FP16, tensorShape, tensorShapeDyn);
    auto directOutput = IRBuilder().CreateTensorVar(DT_FP16, reshapeShape, reshapeShapeDyn);

    IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_EXP, {input}, {contractInput});
    auto& contract = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {contractInput}, {contractOutput});
    contract.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_RESHAPE, {contractOutput}, {firstReshapeOutput});
    auto& view = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {firstReshapeOutput}, {viewOutput});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_RESHAPE, {viewOutput}, {branchReshapeOutput});
    auto& branchSlice = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_SLICE, {branchReshapeOutput},
                                                       {branchSliceOutput});
    branchSlice.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& directSlice = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_SLICE, {firstReshapeOutput},
                                                       {directSliceOutput});
    directSlice.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_EXP, {branchSliceOutput}, {branchOutput});
    IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_EXP, {directSliceOutput}, {directOutput});
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(branchOutput);
    function->outCasts_.push_back(directOutput);

    AssignMemoryType assignMemoryType;
    ASSERT_EQ(assignMemoryType.RunOnFunction(*function), SUCCESS);
    ASSERT_EQ(assignMemoryType.PostCheck(*function), SUCCESS);

    EXPECT_EQ(contract.GetOpcode(), Opcode::OP_ASSEMBLE);
    EXPECT_EQ(contractInput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(contractOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(firstReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(viewOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(branchReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(branchSliceOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(directSliceOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(branchSlice.GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(directSlice.GetOpcode(), Opcode::OP_VIEW);
}

TEST_F(AssignMemoryTypeTest, TestL0C2UBSmallToLarge)
{
    auto shapes = PrepareA5Platform();
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP16, shapes.shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapes.shapeB, "B");
        Tensor inputC(DataType::DT_FP32, shapes.shapeC, "C");
        Tensor out(DataType::DT_FP32, shapes.shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestL0C2UBSmallToLarge", {inputA, inputB, inputC, out})
        {
            // 设置 Cube tile shape，使 matmul 输出 L0C
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor ab = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            // 设置 Vec tile shape，使后续 Vector 操作需要 UB 输入
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            Tensor result = Add(ab, inputC);
            out = result;
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestL0C2UBSmallToLarge");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        // 验证存在 L0C -> UB 的转换
        bool hasL0C2UB = false;
        for (auto& op : originFunction->Operations()) {
            // 检查 Convert: L0C -> UB
            if (op.GetOpcode() == Opcode::OP_CONTRACT) {
                auto input = op.GetIOperands().front();
                auto output = op.GetOOperands().front();
                if (input->GetMemoryTypeOriginal() == MEM_L0C && output->GetMemoryTypeOriginal() == MEM_UB) {
                    hasL0C2UB = true;
                }
            }
            // 或者检查 Assemble 输出为 UB（小搬大场景）
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                auto output = op.GetOOperands().front();
                if (output->GetMemoryTypeOriginal() == MEM_UB) {
                    hasL0C2UB = true;
                }
            }
        }
        EXPECT_TRUE(hasL0C2UB) << "Should have L0C->UB data path for matmul then add";
    }
    // 恢复平台设置
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(AssignMemoryTypeTest, TestL0C2UBParallelDdrFallback)
{
    auto shapes = PrepareA5Platform();
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP16, shapes.shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapes.shapeB, "B");
        Tensor inputC(DataType::DT_FP32, shapes.shapeC, "C");
        Tensor ddrOut(DataType::DT_FP32, shapes.shapeC, "DdrOut");
        Tensor out(DataType::DT_FP32, shapes.shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2UBParallelDdrFallback", {inputA, inputB, inputC, ddrOut, out})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor ab = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            Assemble(ab, {0, 0}, ddrOut);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(ab, inputC);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestL0C2UBParallelDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_UB), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 1);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB), 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(AssignMemoryTypeTest, TestUB2L1SmallToLarge)
{
    // 设置 A5 平台
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    // m=32, k=64, n=64
    std::vector<int64_t> shapeA = {32, 64};
    std::vector<int64_t> shapeB = {64, 64};
    std::vector<int64_t> shapeC = {32, 64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP32, shapeA, "A1");
        Tensor inputA2(DataType::DT_FP32, shapeA, "A2");
        Tensor inputB1(DataType::DT_FP32, shapeB, "B1");
        Tensor inputB2(DataType::DT_FP32, shapeB, "B2");
        Tensor out(DataType::DT_FP32, shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestUB2L1SmallToLarge", {inputA1, inputA2, inputB1, inputB2, out})
        {
            // 1. Vector 操作: Add 输出 UB
            TileShape::Current().SetVecTile(16, 32); // vec_tile_shapes = (16, 32)
            Tensor add1 = Add(inputA1, inputA2);     // (32, 64) UB
            Tensor add2 = Add(inputB1, inputB2);     // (64, 64) UB

            // 2. Cube 操作: MatMul 需要 L1 输入，触发 UB->L1 转换
            TileShape::Current().SetCubeTile({32, 32}, {64, 64},
                                             {64, 64}); // cube_tile_shapes = ([32,32], [64,64], [64,64])
            Tensor result = Matrix::Matmul(out.GetDataType(), add1, add2); // (32, 64) @ (64, 64) = (32, 64)
            out = result;
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestUB2L1SmallToLarge");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        // 验证存在 UB->L1 转换
        bool hasUB2L1 = false;
        for (auto& op : originFunction->Operations()) {
            // 检查 Assemble 的输出是否为 L1（小搬大场景）
            if (op.GetOpcode() == Opcode::OP_CONTRACT) {
                auto output = op.GetOOperands().front();
                if (output->GetMemoryTypeOriginal() == MEM_L1) {
                    hasUB2L1 = true;
                }
            }
            // 检查 Convert: UB -> L1
            if (op.GetOpcode() == Opcode::OP_CONVERT) {
                auto input = op.GetIOperands().front();
                auto output = op.GetOOperands().front();
                if (input->GetMemoryTypeOriginal() == MEM_UB && output->GetMemoryTypeOriginal() == MEM_L1) {
                    hasUB2L1 = true;
                }
            }
        }
        EXPECT_TRUE(hasUB2L1) << "Should have UB->L1 data path for add_then_matmul";
    }
    // 恢复平台设置
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(AssignMemoryTypeTest, TestHf8CastRightMatmulUB2L1)
{
    // FP32 -> HF8 cast is only supported on A5, and this case verifies the A5 UB->L1 path.
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> castInputShape = {NUM_64, NUM_1};
    std::vector<int64_t> leftMatrixShape = {NUM_8, NUM_1};
    std::vector<int64_t> outputShape = {NUM_8, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor castInput(DataType::DT_FP32, castInputShape, "castInput");
        Tensor leftMatrix(DataType::DT_HF8, leftMatrixShape, "leftMatrix");
        Tensor out(DataType::DT_FP32, outputShape, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestHf8CastRightMatmulUB2L1", {castInput, leftMatrix, out})
        {
            TileShape::Current().SetVecTile(NUM_16, NUM_1);
            Tensor castRightMatrix = Cast(castInput, DataType::DT_HF8);
            // HF8 kL0=1 fails frontend 32-byte alignment checks before AssignMemoryType.
            TileShape::Current().SetCubeTile({NUM_8, NUM_8}, {NUM_32, NUM_32}, {NUM_32, NUM_32});
            out = Matrix::Matmul(out.GetDataType(), leftMatrix, castRightMatrix, false, true);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestHf8CastRightMatmulUB2L1");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";

        bool hasUb2L1 = false;
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() != Opcode::OP_CONVERT && op.GetOpcode() != Opcode::OP_SLICE &&
                op.GetOpcode() != Opcode::OP_CONTRACT) {
                continue;
            }
            if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
                continue;
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
                output->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
                hasUb2L1 = true;
            }
        }
        EXPECT_TRUE(hasUb2L1) << "Cast result should use UB->L1 before feeding right matrix into matmul.";
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

void BuildConvertDynShapeGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestConvertDynShape", "TestConvertDynShape",
                                                 nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestConvertDynShape", currFunctionPtr);

    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::vector<SymbolicScalar> dynShape = {IRBuilder().CreateConstInt(6), IRBuilder().CreateConstInt(6)};
    std::shared_ptr<LogicalTensor> input_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> input_tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> view_output1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> view_output2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> add_output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    std::shared_ptr<LogicalTensor> reshape_output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1,
                                                                                               dynShape);
    std::shared_ptr<LogicalTensor> assem_output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, dynShape);

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_tensor2}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {view_output1, view_output2}, {add_output});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_RESHAPE, {add_output}, {reshape_output});
    auto& assemble_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {reshape_output},
                                                       {assem_output});
    assemble_op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(input_tensor1);
    currFunctionPtr->inCasts_.push_back(input_tensor2);
    currFunctionPtr->outCasts_.push_back(assem_output);
}

TEST_F(AssignMemoryTypeTest, TestConvertOpsHaveDynValidShape)
{
    std::shared_ptr<Function> currFunctionPtr;
    BuildConvertDynShapeGraph(currFunctionPtr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto input = op.iOperand[0];
            auto output = op.oOperand[0];
            const auto& in_shape = input->GetDynValidShape();
            const auto& out_shape = output->GetDynValidShape();
            ASSERT_EQ(in_shape.size(), out_shape.size()) << "Size mismatch";
            for (size_t i = 0; i < in_shape.size(); ++i) {
                EXPECT_EQ(in_shape[i].Dump(), out_shape[i].Dump()) << "Mismatch at dimension " << i;
            }
        }
    }
}
} // namespace tile_fwk

TEST_F(AssignMemoryTypeTest, ReshapeOutputUsesUbWithMixedViewConsumers)
{
    ComputationalGraphBuilder G;
    Shape shape{NUM_16, NUM_32};
    Shape reshapeShape{NUM_32, NUM_16};
    G.AddTensor(DataType::DT_FP16, shape, MemoryType::MEM_UNKNOWN, "vec_dup_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "reshape_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "mul_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "view_l1_out");
    G.AddTensor(DataType::DT_FP16, reshapeShape, MemoryType::MEM_UNKNOWN, "l0a_out");

    G.AddOp(Opcode::OP_VEC_DUP, {}, {"vec_dup_out"}, "vec_dup");
    G.AddOp(Opcode::OP_RESHAPE, {"vec_dup_out"}, {"reshape_out"}, "reshape");
    G.AddOp(Opcode::OP_MUL, {"reshape_out", "reshape_out"}, {"mul_out"}, "mul");
    G.AddOp(Opcode::OP_VIEW, {"reshape_out"}, {"view_l1_out"}, "view_l1");
    G.GetOp("view_l1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_L1_TO_L0A, {"view_l1_out"}, {"l0a_out"}, "l1_to_l0a");

    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(assignMemoryType.PostCheck(*func), SUCCESS);

    auto reshapeInput = G.GetTensor("vec_dup_out");
    auto reshapeOutput = G.GetTensor("reshape_out");
    auto mulOutput = G.GetTensor("mul_out");
    auto l1ViewOutput = G.GetTensor("view_l1_out");
    auto l0aOutput = G.GetTensor("l0a_out");
    EXPECT_EQ(reshapeInput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(reshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(mulOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(l1ViewOutput->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(l0aOutput->GetMemoryTypeOriginal(), MemoryType::MEM_L0A);

    auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(G.GetOp("view_l1")->GetOpAttribute());
    ASSERT_NE(viewOpAttr, nullptr);
    EXPECT_EQ(viewOpAttr->GetTo(), MemoryType::MEM_L1);
}

TEST_F(AssignMemoryTypeTest, VecDupViewMatmulNoAssemble)
{
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    const std::vector<int64_t> shapeA = {NUM_32, NUM_16};
    const std::vector<int64_t> shapeB = {NUM_16, NUM_16};
    const std::vector<int64_t> shapeC = {NUM_32, NUM_16};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputB(DataType::DT_FP16, shapeB, "B");
        Tensor matmulOutput(DataType::DT_FP16, shapeC, "C");
        SetFullTestStrategy();
        config::SetPassStrategy("AssignMemoryTypeTestStrategy");
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("VecDupViewMatmulNoAssemble", {inputB, matmulOutput})
        {
            TileShape::Current().SetVecTile(NUM_32, NUM_16);
            Tensor vecDupOutput = Full(Element(DataType::DT_FP16, 0.0f), DataType::DT_FP16, shapeA);
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_16, NUM_16}, {NUM_16, NUM_16});
            matmulOutput = Matrix::Matmul(matmulOutput.GetDataType(), vecDupOutput, inputB);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_VecDupViewMatmulNoAssemble");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        ASSERT_EQ(originFunction->GetIncast().size(), 1);
        ASSERT_EQ(originFunction->GetOutcast().size(), 1);
        EXPECT_EQ(originFunction->GetIncast().front()->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
        EXPECT_EQ(originFunction->GetIncast().front()->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
        EXPECT_EQ(originFunction->GetOutcast().front()->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
        EXPECT_EQ(originFunction->GetOutcast().front()->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);

        int vecDupCount = 0;
        int assembleCount = 0;
        int ubToDdrContractCount = 0;
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VEC_DUP) {
                vecDupCount++;
                EXPECT_TRUE(op.GetIOperands().empty());
                ASSERT_EQ(op.GetOOperands().size(), 1);
                EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
                EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeToBe(), MemoryType::MEM_UB);
            }
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                assembleCount++;
            }
            if (op.GetOpcode() != Opcode::OP_CONTRACT || op.GetIOperands().empty() || op.GetOOperands().empty()) {
                continue;
            }
            if (op.GetIOperands().front()->GetMemoryTypeOriginal() != MemoryType::MEM_UB ||
                op.GetOOperands().front()->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }

            ubToDdrContractCount++;
            auto contractOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
            ASSERT_NE(contractOpAttr, nullptr);
            EXPECT_EQ(contractOpAttr->GetFrom(), MemoryType::MEM_UB);
            EXPECT_EQ(op.GetOOperands().front()->GetMemoryTypeToBe(), MemoryType::MEM_DEVICE_DDR);
        }
        EXPECT_EQ(vecDupCount, 1) << "Full should generate one OP_VEC_DUP";
        EXPECT_EQ(assembleCount, 0) << "Pure view path should not insert legacy assemble";
        EXPECT_EQ(ubToDdrContractCount, 1) << "UB -> DDR move should be represented by one OP_CONTRACT";
    }
}

// VIEW_TYPE 输出 toBeMap 未知、后继 VIEW 无 toAttr 时，验证 InferTargetTypeThroughForwardViews
// 沿未推导视图链向前查得有效内存类型。
// 链路：inA/inB/inC(DDR) -> VIEW -> GATHER_IN_UB -> VIEW -> VIEW_TYPE -> VIEW -> ADDS -> CONTRACT -> outcast
TEST_F(AssignMemoryTypeTest, ViewTypeReusesForwardViewRequirement)
{
    ComputationalGraphBuilder G;
    // 头部输入（DDR incast）：nope_cache / topk_indices / block_table
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_DEVICE_DDR, "inA");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_DEVICE_DDR, "inB");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_DEVICE_DDR, "inC");
    // 视图输出（均无 toAttr）
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_UNKNOWN, "t1792");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_UNKNOWN, "t1793");
    G.AddTensor(DataType::DT_INT32, {1, NUM_16}, MemoryType::MEM_UNKNOWN, "t1794");
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_UNKNOWN, "t1795");
    G.AddTensor(DataType::DT_INT8, {NUM_16, NUM_32}, MemoryType::MEM_UNKNOWN, "t3280");
    // VIEW_TYPE 做字节等价重解释：INT8[16,32]=512B -> FP32[16,8]=512B
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t3281");
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t3536");
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t3537");
    G.AddTensor(DataType::DT_FP32, {NUM_16, NUM_8}, MemoryType::MEM_UNKNOWN, "t8307");

    // inA/inB/inC -> VIEW[16138/16010/15881]
    G.AddOp(Opcode::OP_VIEW, {"inA"}, {"t1792"}, "view16138");
    G.GetOp("view16138")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    G.AddOp(Opcode::OP_VIEW, {"inB"}, {"t1793"}, "view16010");
    G.GetOp("view16010")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    G.AddOp(Opcode::OP_VIEW, {"inC"}, {"t1794"}, "view15881");
    G.GetOp("view15881")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    // VIEW* -> GATHER_IN_UB[10211] -> t1795
    G.AddOp(Opcode::OP_GATHER_IN_UB, {"t1792", "t1793", "t1794"}, {"t1795"}, "gather10211");
    // t1795 -> VIEW[16394] -> t3280
    G.AddOp(Opcode::OP_VIEW, {"t1795"}, {"t3280"}, "view16394");
    G.GetOp("view16394")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    // t3280 -> VIEW_TYPE[12343] -> t3281
    G.AddOp(Opcode::OP_VIEW_TYPE, {"t3280"}, {"t3281"}, "viewtype12343");
    // t3281 -> VIEW[12726] -> t3536
    G.AddOp(Opcode::OP_VIEW, {"t3281"}, {"t3536"}, "view12726");
    G.GetOp("view12726")
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN));
    // t3536 -> ADDS[12727] -> t3537
    G.AddOp(Opcode::OP_ADDS, {"t3536"}, {"t3537"}, "adds12727");
    // t3537 -> CONTRACT[19009] -> t8307（后接 outcast）
    G.AddOp(Opcode::OP_CONTRACT, {"t3537"}, {"t8307"}, "assemble19009");
    G.GetOp("assemble19009")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    G.SetInCast({"inA", "inB", "inC"});
    G.SetOutCast({"t8307"});

    Function* function = G.GetFunction();
    AssignMemoryType assignMemoryType;
    function->DumpJsonFile(
        "./config/pass/json/SplitLargeFanoutTensorTest_ViewTypeReusesForwardViewRequirement_before0.json");
    EXPECT_EQ(assignMemoryType.RunOnFunction(*function), SUCCESS);
    function->DumpJsonFile(
        "./config/pass/json/SplitLargeFanoutTensorTest_ViewTypeReusesForwardViewRequirement_after0.json");
    EXPECT_EQ(assignMemoryType.PostCheck(*function), SUCCESS);

    // 核心：VIEW_TYPE[12343] 输出由后继视图链前推得到 UB（改动前会回退 DDR）
    auto viewTypeOut = G.GetTensor("t3281");
    ASSERT_NE(viewTypeOut, nullptr);
    EXPECT_EQ(viewTypeOut->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(viewTypeOut->GetMemoryTypeToBe(), MemoryType::MEM_UB);

    // 后继 VIEW[12726] 的 toAttr 同步为 UB
    auto view12726Attr = std::dynamic_pointer_cast<ViewOpAttribute>(G.GetOp("view12726")->GetOpAttribute());
    ASSERT_NE(view12726Attr, nullptr);
    EXPECT_EQ(view12726Attr->GetTo(), MemoryType::MEM_UB);

    // 生产者 VIEW[16394] 的 toAttr 同步为 UB
    auto view16394Attr = std::dynamic_pointer_cast<ViewOpAttribute>(G.GetOp("view16394")->GetOpAttribute());
    ASSERT_NE(view16394Attr, nullptr);
    EXPECT_EQ(view16394Attr->GetTo(), MemoryType::MEM_UB);

    // ASSEMBLE[19009] 的 fromAttr 推导为 UB
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(G.GetOp("assemble19009")->GetOpAttribute());
    ASSERT_NE(assembleAttr, nullptr);
    EXPECT_EQ(assembleAttr->GetFrom(), MemoryType::MEM_UB);
}

TEST_F(AssignMemoryTypeTest, TestL0C2UBAssembleDirectPathNotDdrFallback)
{
    auto shapes = PrepareA5Platform();
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP16, shapes.shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapes.shapeB, "B");
        Tensor inputC(DataType::DT_FP32, shapes.shapeC, "C");
        Tensor ubAssembleOut(DataType::DT_FP32, shapes.shapeC, "ubAssembleOut");
        Tensor out(DataType::DT_FP32, shapes.shapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestL0C2UBAssembleDirectPathNotDdrFallback", {inputA, inputB, inputC, ubAssembleOut, out})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor ab = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            Assemble(ab, {0, 0}, ubAssembleOut);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            ubAssembleOut = Exp(ubAssembleOut);
            out = Add(ab, inputC);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2UBAssembleDirectPathNotDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_UB), 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(AssignMemoryTypeTest, SameMemorySliceContractDowngradeToLayoutOps)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSameMemorySliceContractDowngrade",
                                                      "TestSameMemorySliceContractDowngrade", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestSameMemorySliceContractDowngrade", currFunctionPtr);

    std::vector<int64_t> shape = {NUM_16, NUM_32};
    auto dynShape = CreateTestConstIntVector(shape);
    auto sliceInput = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    auto sliceOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    auto contractInput = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    auto contractOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    sliceInput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    sliceOutput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    contractInput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    contractOutput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    auto& sliceOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SLICE, {sliceInput}, {sliceOutput});
    sliceOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UB));
    auto& contractOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CONTRACT, {contractInput},
                                                      {contractOutput});
    contractOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{0, 0}));

    AssignMemoryType assignMemoryType;
    ASSERT_EQ(assignMemoryType.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(sliceOp.GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(contractOp.GetOpcode(), Opcode::OP_ASSEMBLE);
}

TEST_F(AssignMemoryTypeTest, L1DdrL1ContractSliceDowngradeToLayoutOps)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestL1DdrL1ContractSliceDowngrade",
                                                      "TestL1DdrL1ContractSliceDowngrade", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestL1DdrL1ContractSliceDowngrade", currFunctionPtr);

    std::vector<int64_t> shape = {NUM_16, NUM_32};
    auto dynShape = CreateTestConstIntVector(shape);
    auto l1Input = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    auto middle = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    auto l1Output = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    auto l0aOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, dynShape);
    l1Input->SetMemoryTypeBoth(MemoryType::MEM_L1, true);

    auto& contractOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CONTRACT, {l1Input}, {middle});
    contractOp.SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0}));
    auto& l1SliceOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SLICE, {middle}, {l1Output});
    l1SliceOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_L1));
    auto& l0aSliceOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_SLICE, {l1Output}, {l0aOutput});
    l0aSliceOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_L0A));

    AssignMemoryType assignMemoryType;
    ASSERT_EQ(assignMemoryType.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(middle->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(contractOp.GetOpcode(), Opcode::OP_ASSEMBLE);
    EXPECT_EQ(l1SliceOp.GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(l0aSliceOp.GetOpcode(), Opcode::OP_SLICE);
    EXPECT_EQ(l0aOutput->GetMemoryTypeOriginal(), MemoryType::MEM_L0A);
}

TEST_F(AssignMemoryTypeTest, TestUB2L1AssembleDirectPathNotDdrFallback)
{
    PrepareA5Platform();
    std::vector<int64_t> ub2l1ShapeA = {32, 64};
    std::vector<int64_t> ub2l1ShapeB = {64, 64};
    std::vector<int64_t> ub2l1ShapeC = {32, 64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA1(DataType::DT_FP32, ub2l1ShapeA, "A1");
        Tensor inputA2(DataType::DT_FP32, ub2l1ShapeA, "A2");
        Tensor inputB1(DataType::DT_FP32, ub2l1ShapeB, "B1");
        Tensor inputB2(DataType::DT_FP32, ub2l1ShapeB, "B2");
        Tensor l1AssembleOut(DataType::DT_FP32, ub2l1ShapeC, "l1AssembleOut");
        Tensor out(DataType::DT_FP32, ub2l1ShapeC, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestUB2L1AssembleDirectPathNotDdrFallback", {inputA1, inputA2, inputB1, inputB2, l1AssembleOut, out})
        {
            TileShape::Current().SetVecTile(16, 32);
            Tensor add1 = Add(inputA1, inputA2);
            Assemble(add1, {0, 0}, l1AssembleOut);
            Tensor add2 = Add(inputB1, inputB2);
            TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {64, 64});
            l1AssembleOut = Matrix::Matmul(out.GetDataType(), l1AssembleOut, add2);
            out = Matrix::Matmul(out.GetDataType(), add1, add2);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestUB2L1AssembleDirectPathNotDdrFallback");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR), 0);
        EXPECT_GE(CountMemoryPath(originFunction, MemoryType::MEM_UB, MemoryType::MEM_L1), 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(AssignMemoryTypeTest, FunctionExpAssembleMatmulA5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    const std::vector<int64_t> inputShape = {NUM_16, NUM_32};
    const std::vector<int64_t> assembleShape = {NUM_32, NUM_32};
    const std::vector<int64_t> cShape = {NUM_32, NUM_64};
    const std::vector<int64_t> matmulShape = {NUM_32, NUM_64};
    PROGRAM("AssignMemoryTest")
    {
        Tensor a(DataType::DT_FP32, inputShape, "a");
        Tensor b(DataType::DT_FP32, inputShape, "b");
        Tensor c(DataType::DT_FP32, cShape, "c");
        Tensor expOut(DataType::DT_FP32, assembleShape, "expOut");
        Tensor matmulOut(DataType::DT_FP32, matmulShape, "matmulOut");
        SetFullTestStrategy();
        config::SetBuildStatic(true);
        FUNCTION("FunctionExpAssembleMatmulA5", {a, b, c, expOut, matmulOut})
        {
            TileShape::Current().SetVecTile(NUM_32, NUM_32);
            Tensor ta = Exp(a);
            Tensor tb = Exp(b);
            Tensor lt(DataType::DT_FP32, assembleShape, "Lt");
            Assemble(ta, {0, 0}, lt);
            Assemble(tb, {NUM_16, 0}, lt);

            expOut = Exp(lt);
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_32, NUM_32}, {NUM_64, NUM_64});
            matmulOut = Matrix::Matmul(matmulOut.GetDataType(), lt, c);
        }
        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_FunctionExpAssembleMatmulA5");
        ASSERT_NE(originFunction, nullptr);
        for (const auto& op : originFunction->Operations()) {
            if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
                continue;
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            EXPECT_FALSE(input->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
                         output->GetMemoryTypeOriginal() == MemoryType::MEM_UB)
                << op.GetOpcodeStr() << "[" << op.GetOpMagic()
                << "] must not move data from L1 to UB; input=" << input->GetMagic()
                << ", output=" << output->GetMagic();
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// Helper: build graph for ProcessShapeTransportFallback testing.
// matmul1(s1,L0C) → [view0(s2,L0C)] → view1(s2,L0C→L1) → view2(s2,L1→L0A) → matmul2(s2)
// When withView0=false, view1's input is matmul1Out directly (s1 must equal s2).
// When withView0=true, view0 creates a shape change from s1 to s2 (穿透 endpoint shapes differ).
static void ConstructShapeTransportGraph(std::shared_ptr<Function>& func, const std::vector<int64_t>& s1,
                                         const std::vector<int64_t>& s2, bool withView0)
{
    int64_t k1 = s1.size() >= 2 ? s1[1] : 1;
    std::vector<int64_t> shapeA1 = {s1[0], k1};
    std::vector<int64_t> shapeB1 = {k1, s1.size() >= 2 ? s1[1] : 1};
    int64_t m2 = NUM_128;
    std::vector<int64_t> shapeA2 = {m2, s2[0]};
    std::vector<int64_t> shapeC2 = {m2, s2.size() >= 2 ? s2[1] : 1};

    auto inputA1 = IRBuilder().CreateTensorVar(DT_FP16, shapeA1, CreateTestConstIntVector(shapeA1));
    auto inputB1 = IRBuilder().CreateTensorVar(DT_FP16, shapeB1, CreateTestConstIntVector(shapeB1));
    auto inputA2 = IRBuilder().CreateTensorVar(DT_FP16, shapeA2, CreateTestConstIntVector(shapeA2));
    auto matmul1Out = IRBuilder().CreateTensorVar(DT_FP16, s1, CreateTestConstIntVector(s1));
    auto view0Out = IRBuilder().CreateTensorVar(DT_FP16, s2, CreateTestConstIntVector(s2));
    auto view1Out = IRBuilder().CreateTensorVar(DT_FP16, s2, CreateTestConstIntVector(s2));
    auto view2Out = IRBuilder().CreateTensorVar(DT_FP16, s2, CreateTestConstIntVector(s2));
    auto outputC2 = IRBuilder().CreateTensorVar(DT_FP16, shapeC2, CreateTestConstIntVector(shapeC2));

    IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {inputA1, inputB1}, {matmul1Out});

    LogicalTensorPtr view1Input = matmul1Out;
    if (withView0) {
        auto& view0Op = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {matmul1Out}, {view0Out});
        view0Op.SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset(s1.size(), 0), MemoryType::MEM_L0C));
        view1Input = view0Out;
    }

    auto& view1Op = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {view1Input}, {view1Out});
    view1Op.SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset(s2.size(), 0), MemoryType::MEM_L1));

    auto& view2Op = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {view1Out}, {view2Out});
    view2Op.SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset(s2.size(), 0), MemoryType::MEM_L0A));

    IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {inputA2, view2Out}, {outputC2});

    func->inCasts_.push_back(inputA1);
    func->inCasts_.push_back(inputB1);
    func->inCasts_.push_back(inputA2);
    func->outCasts_.push_back(outputC2);
}

// 单向整除放行：prodOut=(128,16), consIn=(64,16) → 仅一个方向整除 → 放行
TEST_F(AssignMemoryTypeTest, ShapeTransportFallbackOneWayDivisibleAllow)
{
    config::SetPassOption(ENABLE_SLICE, false);
    auto func = std::make_shared<Function>(Program::GetInstance(), "ShapeTransportOneWayDivisible",
                                           "ShapeTransportOneWayDivisible", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ShapeTransportOneWayDivisible", func);
    ConstructShapeTransportGraph(func, {NUM_128, NUM_16}, {NUM_64, NUM_16}, true);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_GT(CountMemoryPath(func.get(), MemoryType::MEM_L0C, MemoryType::MEM_L1), 0)
        << "L0C→L1 path should exist when one direction is divisible";
}

// 双向都不整除拦截：prodOut=(128,48), consIn=(64,32) → 两方向均不整除 → 拦截回退DDR
TEST_F(AssignMemoryTypeTest, ShapeTransportFallbackNeitherDivisibleIntercept)
{
    config::SetPassOption(ENABLE_SLICE, false);
    auto func = std::make_shared<Function>(Program::GetInstance(), "ShapeTransportNeitherDivisible",
                                           "ShapeTransportNeitherDivisible", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ShapeTransportNeitherDivisible", func);
    ConstructShapeTransportGraph(func, {NUM_128, NUM_48}, {NUM_64, NUM_32}, true);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(CountMemoryPath(func.get(), MemoryType::MEM_L0C, MemoryType::MEM_L1), 0)
        << "L0C→L1 path should not exist when neither direction is divisible";
    EXPECT_GT(CountMemoryPath(func.get(), MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0)
        << "L0C→DDR path should exist after intercept";
}

// 等大放行：prodOut=(32,16), consIn=(32,16) → 形状相同 → 放行
TEST_F(AssignMemoryTypeTest, ShapeTransportFallbackEqualShapeAllow)
{
    config::SetPassOption(ENABLE_SLICE, false);
    auto func = std::make_shared<Function>(Program::GetInstance(), "ShapeTransportEqualShape",
                                           "ShapeTransportEqualShape", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ShapeTransportEqualShape", func);
    ConstructShapeTransportGraph(func, {NUM_32, NUM_16}, {NUM_32, NUM_16}, false);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_GT(CountMemoryPath(func.get(), MemoryType::MEM_L0C, MemoryType::MEM_L1), 0)
        << "L0C→L1 path should exist when shapes are equal";
}

// rank 不一致拦截：prodOut=(128,128) rank2, consIn=(128,128,1) rank3 → rank不匹配 → 拦截回退DDR
TEST_F(AssignMemoryTypeTest, ShapeTransportFallbackRankMismatchIntercept)
{
    config::SetPassOption(ENABLE_SLICE, false);
    auto func = std::make_shared<Function>(Program::GetInstance(), "ShapeTransportRankMismatch",
                                           "ShapeTransportRankMismatch", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ShapeTransportRankMismatch", func);
    ConstructShapeTransportGraph(func, {NUM_128, NUM_128}, {NUM_128, NUM_128, NUM_1}, true);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(CountMemoryPath(func.get(), MemoryType::MEM_L0C, MemoryType::MEM_L1), 0)
        << "L0C→L1 path should not exist when rank mismatch";
    EXPECT_GT(CountMemoryPath(func.get(), MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0)
        << "L0C→DDR path should exist after rank mismatch intercept";
}

// 3510 L0C→UB 整除放行：matmul(32,128,L0C) → VIEW(L0C 透明) → ASSEMBLE(L0C→UB) → ADD(UB)
// prodOut(32,128) 与 consIn(32,128) 等大（整除特例）→ 放行，L0C→UB 直连保留。
// 反向断言：若 allow 分支回归（IsAllowedTransport 误返 false / 端点漏收），
// 该路径会被静默回退 DDR，本用例失败可见。
TEST_F(AssignMemoryTypeTest, ShapeTransportFallbackL0C2UBDivisibleAllow)
{
    config::SetPassOption(ENABLE_SLICE, false);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    ComputationalGraphBuilder G;
    Shape shape{NUM_32, NUM_128};
    G.AddTensors(DataType::DT_FP16, shape,
                 {MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_L0C,
                  MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"l0a_in", "l0b_in", "matmul_out", "view_l0c_out", "asm_out", "rc_out", "add_in2", "add_out"});

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"matmul_out"}, "matmul");
    G.AddOp(Opcode::OP_VIEW, {"matmul_out"}, {"view_l0c_out"}, "view_l0c");
    G.GetOp("view_l0c")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_ASSEMBLE, {"view_l0c_out"}, {"asm_out"}, "assemble");
    G.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"asm_out"}, {"rc_out"}, "reg_copy");
    G.AddOp(Opcode::OP_ADD, {"rc_out", "add_in2"}, {"add_out"}, "add");

    G.SetInCast({"l0a_in", "l0b_in", "add_in2"});
    G.SetOutCast({"add_out"});
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_GT(CountMemoryPath(func, MemoryType::MEM_L0C, MemoryType::MEM_UB), 0)
        << "L0C→UB allowed: prodOut(32,128) vs consIn(32,128) equal shape";
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// 3510 UB→L1 整除放行：VEC_DUP(UB,64,64) → VIEW(UB→L1) → VIEW(L1→L0A) → matmul
// prodOut(64,64) 与 consIn(64,64) 等大 → 放行，UB→L1 直连保留。
// 反向断言：allow 分支回归导致 UB→L1 被静默回退 DDR 时失败可见。
TEST_F(AssignMemoryTypeTest, ShapeTransportFallbackUB2L1DivisibleAllow)
{
    config::SetPassOption(ENABLE_SLICE, false);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    ComputationalGraphBuilder G;
    Shape shape{NUM_64, NUM_64};
    G.AddTensors(DataType::DT_FP16, shape,
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0A,
                  MemoryType::MEM_L0B, MemoryType::MEM_L0C},
                 {"vec_out", "ub_view_out", "l1_view_out", "l0a_view_out", "l0a_in", "l0b_in", "mm_out"});

    G.AddOp(Opcode::OP_VEC_DUP, {}, {"vec_out"}, "vec_dup");
    G.AddOp(Opcode::OP_VIEW, {"vec_out"}, {"ub_view_out"}, "ub_view");
    G.GetOp("ub_view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L1));
    G.AddOp(Opcode::OP_VIEW, {"ub_view_out"}, {"l1_view_out"}, "l1_view");
    G.GetOp("l1_view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_VIEW, {"l1_view_out"}, {"l0a_view_out"}, "l0a_view");
    G.GetOp("l0a_view")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0A));
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_view_out", "l0b_in"}, {"mm_out"}, "matmul");

    G.SetInCast({"l0b_in"});
    G.SetOutCast({"mm_out"});
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_GT(CountMemoryPath(func, MemoryType::MEM_UB, MemoryType::MEM_L1), 0)
        << "UB→L1 allowed: prodOut(64,64) vs consIn(64,64) equal shape";
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// L0C→UB 穿透 REGISTER_COPY 拦截验证（参照真实计算图 10367 链路）：
// A_MUL_B(L0C,32x128) → VIEW(L0C 透明) → ASSEMBLE(L0C→UB 搬运点)
//   → REGISTER_COPY(UB 透明) → VIEW(UB reshape 32x128→16x256) → ADD(UB 真实消费者)
// prodOut=matmul_out(32,128) 与 consIn=view_ub_out(16,256) 双向不整除 → 拦截回退 DDR。
// REGISTER_COPY 加入跳过集合后 CollectRealConsumerInputs 穿透 REGISTER_COPY+VIEW 抵达真实消费者；
// 否则止步于 REGISTER_COPY(输出 32x128)，与 prodOut 等大而误放行。
// 对齐真实计算图 10367 链路 post-pass 形态（matmul→view→assemble(L0C→UB)→register_copy(UB 侧,被穿透跳过)→view→add）。
// register_copy 位于 consumer 侧 UB 区，验证 CollectRealConsumerInputs 对 REGISTER_COPY 的跳过。
TEST_F(AssignMemoryTypeTest, ShapeTransportFallbackL0C2UBRegisterCopyIntercept)
{
    config::SetPassOption(ENABLE_SLICE, false);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_32}, MemoryType::MEM_L0A, "l0a_in");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_L0B, "l0b_in");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_L0C, "matmul_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_L0C, "view_l0c_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_UB, "asm_out");
    G.AddTensor(DataType::DT_FP16, {NUM_32, NUM_128}, MemoryType::MEM_UB, "rc_out");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_256}, MemoryType::MEM_UB, "view_ub_out");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_256}, MemoryType::MEM_UB, "add_in2");
    G.AddTensor(DataType::DT_FP16, {NUM_16, NUM_256}, MemoryType::MEM_UB, "add_out");

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a_in", "l0b_in"}, {"matmul_out"}, "matmul");
    G.AddOp(Opcode::OP_VIEW, {"matmul_out"}, {"view_l0c_out"}, "view_l0c");
    G.GetOp("view_l0c")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_ASSEMBLE, {"view_l0c_out"}, {"asm_out"}, "assemble");
    G.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
    G.AddOp(Opcode::OP_REGISTER_COPY, {"asm_out"}, {"rc_out"}, "reg_copy");
    G.AddOp(Opcode::OP_VIEW, {"rc_out"}, {"view_ub_out"}, "view_ub");
    G.GetOp("view_ub")->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_ADD, {"view_ub_out", "add_in2"}, {"add_out"}, "add");

    G.SetInCast({"l0a_in", "l0b_in", "add_in2"});
    G.SetOutCast({"add_out"});
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(CountMemoryPath(func, MemoryType::MEM_L0C, MemoryType::MEM_UB), 0)
        << "L0C→UB intercepted: prodOut(32,128) vs consIn(16,256) neither divisible";
    EXPECT_GT(CountMemoryPath(func, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0)
        << "L0C→DDR fallback should exist after intercept";
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

static void AddL0cToUbBranch(ComputationalGraphBuilder& G, const std::string& l0cOut, const std::string& viewOut,
                             const std::string& rcOut)
{
    G.AddOp(Opcode::OP_VIEW, {l0cOut}, {viewOut}, "view_" + viewOut);
    G.GetOp("view_" + viewOut)->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_REGISTER_COPY, {viewOut}, {rcOut}, "rc_" + rcOut);
}

static void AddAssembleMerge(ComputationalGraphBuilder& G, const std::string& in1, const std::string& in2,
                             const std::string& out)
{
    G.AddOp(Opcode::OP_ASSEMBLE, {in1}, {out}, "asm_" + in1);
    G.GetOp("asm_" + in1)->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
    G.AddOp(Opcode::OP_ASSEMBLE, {in2}, {out}, "asm_" + in2);
    G.GetOp("asm_" + in2)->SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{0, 0}));
}

static void AddUbView(ComputationalGraphBuilder& G, const std::string& input, const std::string& output)
{
    G.AddOp(Opcode::OP_VIEW, {input}, {output}, "view_" + output);
    G.GetOp("view_" + output)->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MemoryType::MEM_UB));
}

TEST_F(AssignMemoryTypeTest, TestL0C2UBViewTransportRegisterCopyAssembleIntercept)
{
    config::SetPassOption(ENABLE_SLICE, false);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    ComputationalGraphBuilder G;
    Shape l0a1Shape{NUM_128, NUM_32};
    Shape l0a2Shape{NUM_32, NUM_32};
    Shape l0bShape{NUM_32, NUM_128};
    Shape l0c1Shape{NUM_128, NUM_128};
    Shape l0c2Shape{NUM_32, NUM_128};
    Shape ubNarrowShape{NUM_16, NUM_128};
    Shape ubWideShape{NUM_16, NUM_256};
    G.AddTensors(DataType::DT_FP16, l0a1Shape, {MemoryType::MEM_L0A}, {"l0a1"});
    G.AddTensors(DataType::DT_FP16, l0a2Shape, {MemoryType::MEM_L0A}, {"l0a2"});
    G.AddTensors(DataType::DT_FP16, l0bShape, {MemoryType::MEM_L0B, MemoryType::MEM_L0B}, {"l0b1", "l0b2"});
    G.AddTensors(DataType::DT_FP16, l0c1Shape, {MemoryType::MEM_L0C}, {"mm1_out"});
    G.AddTensors(DataType::DT_FP16, l0c2Shape, {MemoryType::MEM_L0C}, {"mm2_out"});
    G.AddTensors(DataType::DT_FP16, ubNarrowShape,
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
                  MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"v1a", "r1a", "v1b", "r1b", "v2a", "r2a", "v2b", "r2b"});
    G.AddTensors(DataType::DT_FP16, ubWideShape,
                 {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                 {"mergeX", "mergeY", "vx", "vy", "add_out"});

    G.AddOp(Opcode::OP_A_MUL_B, {"l0a1", "l0b1"}, {"mm1_out"}, "mm1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a2", "l0b2"}, {"mm2_out"}, "mm2");
    AddL0cToUbBranch(G, "mm1_out", "v1a", "r1a");
    AddL0cToUbBranch(G, "mm1_out", "v1b", "r1b");
    AddL0cToUbBranch(G, "mm2_out", "v2a", "r2a");
    AddL0cToUbBranch(G, "mm2_out", "v2b", "r2b");
    AddAssembleMerge(G, "r1a", "r2a", "mergeX");
    AddAssembleMerge(G, "r1b", "r2b", "mergeY");
    AddUbView(G, "mergeX", "vx");
    AddUbView(G, "mergeY", "vy");
    G.AddOp(Opcode::OP_ADD, {"vx", "vy"}, {"add_out"}, "add");

    G.SetInCast({"l0a1", "l0b1", "l0a2", "l0b2"});
    G.SetOutCast({"add_out"});
    Function* func = G.GetFunction();
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.RunOnFunction(*func), SUCCESS);
    EXPECT_EQ(CountMemoryPath(func, MemoryType::MEM_L0C, MemoryType::MEM_UB), 0)
        << "L0C→UB intercepted: (128,128)/(32,128) vs (16,256) neither divisible";
    EXPECT_GT(CountMemoryPath(func, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0)
        << "L0C→DDR fallback should exist after cross-chain intercept";
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

// FUNCTION macro 端到端写法：走完整 pass 流水线（AutoCast/ExpandFunction/SplitReshape 等前置 pass
// 改图后）验证 ProcessShapeTransportFallback 仍能拦截。matmul(64,64,L0C) → Reshape(16,256)
// → Add(UB)：前置 pass 为 reshape 插入的 view 构成 L0C→UB 搬运点，
// prodOut(64,64) 与 consIn(16,256) 双向不整除 → 拦截回退 DDR。
// 与 ConstructShapeTransportGraph 手动构图用例互补：手动构图直接建 tile 图，
// 覆盖不了"前端入图经前置 pass 变化后"的链路形态。
TEST_F(AssignMemoryTypeTest, TestL0C2UBShapeTransportEndToEndByFunctionMacro)
{
    config::SetPassOption(ENABLE_SLICE, false);
    auto shapes = PrepareA5Platform();
    PROGRAM("AssignMemoryTest")
    {
        Tensor inputA(DataType::DT_FP16, shapes.shapeA, "A");
        Tensor inputB(DataType::DT_FP16, shapes.shapeB, "B");
        Tensor inputC(DataType::DT_FP32, {NUM_16, NUM_256}, "C");
        Tensor out(DataType::DT_FP32, {NUM_16, NUM_256}, "output");
        SetFullTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestL0C2UBShapeTransportEndToEndByFunctionMacro", {inputA, inputB, inputC, out})
        {
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_64, NUM_64}, {NUM_64, NUM_64});
            Tensor ab = Matrix::Matmul(out.GetDataType(), inputA, inputB);
            TileShape::Current().SetVecTile(NUM_16, NUM_256);
            Tensor reshaped = Reshape(ab, {NUM_16, NUM_256});
            out = Add(reshaped, inputC);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName(
            "TENSOR_TestL0C2UBShapeTransportEndToEndByFunctionMacro");
        ASSERT_NE(originFunction, nullptr) << "Function pointer is null";
        EXPECT_EQ(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_UB), 0)
            << "End-to-end L0C→UB should be intercepted: (64,64) vs (16,256) neither divisible";
        EXPECT_GT(CountMemoryPath(originFunction, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR), 0)
            << "L0C→DDR fallback should exist after end-to-end intercept";
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

} // namespace npu
