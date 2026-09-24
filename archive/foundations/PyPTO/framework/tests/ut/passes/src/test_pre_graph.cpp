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
 * \file test_pre_graph.cpp
 * \brief Unit test for PreGraph pass.
 */

#include <fstream>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/pre_graph.h"
#include "ut_json/ut_json_tool.h"
#include "computational_graph_builder.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/irbuilder.h"
#define private public

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
constexpr int SUBGRAPHID0 = 0;
constexpr int SUBGRAPHID1 = 1;
constexpr int SUBGRAPHID2 = 2;
constexpr int NUM5 = 5;
constexpr int NUM10 = 10;
constexpr int NUM128 = 128;
constexpr const char* TEST_ENABLE_TRUE_KEY = "enable_true";
void PrintGraphInfoPreGraph(Function* func, std::set<int>& tensorMagicWithColorSet)
{
    std::cout << "func->Operations().size() = " << func->Operations().size() << std::endl;
    for (auto& op : func->Operations()) {
        std::cout << "Op:" << op.GetOpMagic() << " " << op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor>& input_tensor : op.GetIOperands()) {
            for (const auto& item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
            if (input_tensor->GetMemoryTypeOriginal() == npu::tile_fwk::MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int curColor = CommonUtils::GetTensorSubgraphID(input_tensor);
            std::cout << "input tensor, cur color is " << curColor << std::endl;
            if (curColor > 0) {
                tensorMagicWithColorSet.insert(input_tensor->magic);
                std::cout << "cur input tensor magic is " << input_tensor->magic << std::endl;
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor>& output_tensor : op.GetOOperands()) {
            for (const auto& item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
            if (output_tensor->GetMemoryTypeOriginal() == npu::tile_fwk::MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int curColor = CommonUtils::GetTensorSubgraphID(output_tensor);
            std::cout << "output tensor, cur color is " << curColor << std::endl;
            if (curColor > 0) {
                tensorMagicWithColorSet.insert(output_tensor->magic);
                std::cout << "cur output tensor magic is " << output_tensor->magic << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

void SetUpPassStrategy()
{
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreGraphTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"DuplicateOp", PassName::DUPLICATE_OP},
                                     {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                     {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     {"InferDiscontinuousInput", PassName::INFER_DISCONTINUOUS_INPUT},
                                     {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                     {"ProcessAtomic", PassName::PROCESS_ATOMIC},
                                     {"GraphPartition", PassName::GRAPH_PARTITION},
                                     {"NBufferMerge", PassName::N_BUFFER_MERGE},
                                     {"IntraSubgraphAdapter", PassName::INTRA_SUBGRAPH_ADAPTER},
                                     {"GenerateMoveOp", PassName::GENERATE_MOVE_OP},
                                     {"CommonOperationEliminate", PassName::COMMON_OPERATION_ELIMINATE},
                                     {"L1CopyInReuseMerge", PassName::L1_COPY_IN_REUSE_MERGE},
                                     {"PadLocalBuffer", PassName::PAD_LOCAL_BUFFER},
                                     {"RemoveUnalignedReshape", PassName::REMOVE_UNALIGNED_RESHAPE},
                                     {"ReplaceTensor", PassName::REPLACE_TENSOR},
                                 });
}

class PreGraphTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "PreGraphTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}

    // input[B, N, S] --> Copy_In --> input_ub[1, 1, S] --> Transpose_Datamove --> outputGm, [N, B, S](i, j, 0) -->
    // Assemble --> output[N, B, S]
    void TileExpandTransposeDatamove(ComputationalGraphBuilder& G, const int B, const int N, const int S,
                                     bool isInner = false)
    {
        std::vector<int64_t> tileShape{1, 1, S};
        // 所有transpose_datamove输出partial结果都要指向同一个raw tensor
        G.AddTensor(DataType::DT_FP32, {N, B, S}, "temp_out");
        auto tempOut = G.GetTensor("temp_out");
        tempOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        auto incast = G.GetTensor("input");
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < N; j++) {
                std::vector<int64_t> offset = {i, j, 0};
                std::vector<int64_t> offsetNew = {j, i, 0};
                int subgraphId = i * N + j;

                std::string input_ub = "input_ub_" + std::to_string(subgraphId);
                G.AddTensor(DataType::DT_FP32, tileShape, input_ub);
                auto tensorUb = G.GetTensor(input_ub);
                tensorUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_COPY_IN, {"input"}, {input_ub}, "Ub_Copy_In_" + std::to_string(subgraphId));
                auto copyInOp = G.GetOp("Ub_Copy_In_" + std::to_string(subgraphId));
                auto attrCopyIn = std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(offset), MemoryType::MEM_UB, OpImmediate::Specified(incast->GetShape()),
                    OpImmediate::Specified(incast->tensor->GetRawShape()));
                copyInOp->SetOpAttribute(attrCopyIn);
                copyInOp->UpdateSubgraphID(subgraphId);

                std::string outputPartial = "output_ddr_" + std::to_string(subgraphId);
                G.AddTensor(DataType::DT_FP32, tileShape, outputPartial);
                auto outputGm = G.GetTensor(outputPartial);
                outputGm->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
                outputGm->tensor = tempOut->tensor;
                outputGm->UpdateOffset(offsetNew);
                G.AddOp(Opcode::OP_TRANSPOSE_MOVEOUT, {input_ub}, {outputPartial},
                        "Transpose_Datamove_" + std::to_string(subgraphId));
                auto transposeOp = G.GetOp("Transpose_Datamove_" + std::to_string(subgraphId));
                transposeOp->UpdateSubgraphID(subgraphId);

                G.AddOp(Opcode::OP_ASSEMBLE, {outputPartial}, {"output"}, "Assemble_" + std::to_string(subgraphId));
                auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offsetNew);
                auto assembleOp = G.GetOp("Assemble_" + std::to_string(subgraphId));
                assembleOp->SetOpAttribute(attrAssemble);
                assembleOp->UpdateSubgraphID(subgraphId);

                if (isInner) {
                    auto outInnerTemp = G.GetTensor("outInnerTemp");
                    G.AddOp(Opcode::OP_ASSEMBLE, {outputPartial}, {"outInnerTemp"},
                            "Assemble_Inner_" + std::to_string(subgraphId));
                    auto attrAssembleInner = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR,
                                                                                   offsetNew);
                    auto assembleOpInner = G.GetOp("Assemble_Inner_" + std::to_string(subgraphId));
                    assembleOpInner->SetOpAttribute(attrAssembleInner);
                    assembleOpInner->UpdateSubgraphID(subgraphId);
                }
            }
        }
    }

    // outInnerTemp[N, B, S] --> Copy_In --> input_ub [1, B, S] --> Exp --> output_ub --> Copy_Out --> output2[N, B, S]
    void TileExpandExp(ComputationalGraphBuilder& G, const int B, const int N, const int S)
    {
        auto outInnerTemp = G.GetTensor("outInnerTemp");
        auto output2 = G.GetTensor("output2");
        std::vector<int64_t> tileShape{1, B, S};
        for (int i = 0; i < N; i++) {
            std::vector<int64_t> offset = {i, 0, 0};
            int subgraphId = B * N + i;

            std::string input_ub = "input_ub_" + std::to_string(subgraphId);
            G.AddTensor(DataType::DT_FP32, tileShape, input_ub);
            auto tensorUb = G.GetTensor(input_ub);
            tensorUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
            G.AddOp(Opcode::OP_COPY_IN, {"outInnerTemp"}, {input_ub}, "Ub_Copy_In_" + std::to_string(subgraphId));
            auto copyInOp = G.GetOp("Ub_Copy_In_" + std::to_string(subgraphId));
            auto attrCopyIn = std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offset), MemoryType::MEM_UB, OpImmediate::Specified(outInnerTemp->GetShape()),
                OpImmediate::Specified(outInnerTemp->tensor->GetRawShape()));
            copyInOp->SetOpAttribute(attrCopyIn);
            copyInOp->UpdateSubgraphID(subgraphId);

            std::string outputExpUb = "output_exp_" + std::to_string(subgraphId);
            G.AddTensor(DataType::DT_FP32, tileShape, outputExpUb);
            auto outputGm = G.GetTensor(outputExpUb);
            outputGm->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
            G.AddOp(Opcode::OP_EXP, {input_ub}, {outputExpUb}, "Exp_" + std::to_string(subgraphId));
            auto expOp = G.GetOp("Exp_" + std::to_string(subgraphId));
            expOp->UpdateSubgraphID(subgraphId);

            G.AddOp(Opcode::OP_COPY_OUT, {outputExpUb}, {"output2"}, "Copy_Out_" + std::to_string(subgraphId));
            auto copyOutOp = G.GetOp("Copy_Out_" + std::to_string(subgraphId));
            auto attrCopyOut = std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offset), MemoryType::MEM_UB, OpImmediate::Specified(output2->GetShape()),
                OpImmediate::Specified(output2->tensor->GetRawShape()));
            copyOutOp->SetOpAttribute(attrCopyOut);
            copyOutOp->UpdateSubgraphID(subgraphId);
        }
    }

    /*
    [32, 32] --> View --> [16, 16] --> Add --> [16, 16] --> Assemble --> addOutUb[32, 32] -->
                                                    \--> Copy_Out --> out1
    */
    void TileExpandAdd(ComputationalGraphBuilder& G, const int N, const int T)
    {
        std::vector<int64_t> tileShape{T, T};
        auto a = G.GetTensor("a");
        auto b = G.GetTensor("b");
        auto out1 = G.GetTensor("out1");
        auto addOutUb = G.GetTensor("addOutUb");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::vector<int64_t> offset = {i * T, j * T};
                int idx = i * N + j;

                std::string localA = "a_" + std::to_string(idx);
                G.AddTensor(DataType::DT_FP32, tileShape, localA);
                auto tensorA = G.GetTensor(localA);
                tensorA->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_COPY_IN, {"a"}, {localA}, "Copy_In_A_" + std::to_string(idx));
                auto copyInA = G.GetOp("Copy_In_A_" + std::to_string(idx));
                auto attrCopyInA = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MemoryType::MEM_UB,
                                                                     OpImmediate::Specified(a->GetShape()),
                                                                     OpImmediate::Specified(a->tensor->GetRawShape()));
                copyInA->SetOpAttribute(attrCopyInA);
                copyInA->UpdateSubgraphID(0);

                std::string localB = "b_" + std::to_string(idx);
                G.AddTensor(DataType::DT_FP32, tileShape, localB);
                auto tensorB = G.GetTensor(localB);
                tensorB->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_COPY_IN, {"b"}, {localB}, "Copy_In_B_" + std::to_string(idx));
                auto copyInB = G.GetOp("Copy_In_B_" + std::to_string(idx));
                auto attrCopyInB = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MemoryType::MEM_UB,
                                                                     OpImmediate::Specified(b->GetShape()),
                                                                     OpImmediate::Specified(b->tensor->GetRawShape()));
                copyInB->SetOpAttribute(attrCopyInB);
                copyInB->UpdateSubgraphID(0);

                std::string localAddOut = "add_out_" + std::to_string(idx);
                G.AddTensor(DataType::DT_FP32, tileShape, localAddOut);
                G.AddOp(Opcode::OP_ADD, {localA, localB}, {localAddOut}, "Add_" + std::to_string(idx));
                auto tensorAddOut = G.GetTensor(localAddOut);
                tensorAddOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                tensorAddOut->tensor = addOutUb->tensor;
                tensorAddOut->UpdateOffset(offset);
                auto addOp = G.GetOp("Add_" + std::to_string(idx));
                addOp->UpdateSubgraphID(0);

                G.AddOp(Opcode::OP_ASSEMBLE, {localAddOut}, {"addOutUb"}, "Assemble_" + std::to_string(idx));
                auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset);
                auto assembleOp = G.GetOp("Assemble_" + std::to_string(idx));
                assembleOp->SetOpAttribute(attrAssemble);
                assembleOp->UpdateSubgraphID(0);

                G.AddOp(Opcode::OP_COPY_OUT, {localAddOut}, {"out1"}, "Copy_Out_" + std::to_string(idx));
                auto copyOutOp = G.GetOp("Copy_Out_" + std::to_string(idx));
                auto attrCopyOut = std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(offset), MemoryType::MEM_UB, OpImmediate::Specified(out1->GetShape()),
                    OpImmediate::Specified(out1->tensor->GetRawShape()));
                copyOutOp->SetOpAttribute(attrCopyOut);
                copyOutOp->UpdateSubgraphID(0);
            }
        }
    }

    int CountAssemble(Function& function)
    {
        int result = 0;
        for (auto& op : function.Operations()) {
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                result++;
            }
        }
        return result;
    }
};

TEST_F(PreGraphTest, TestAssemble)
{
    SetUpPassStrategy();
    int dim1 = 8;
    int dim2 = 2;
    int dim3 = 64;
    TileShape::Current().SetVecTile(dim1, dim1, dim1, dim1);
    Tensor input(DT_FP32, {1, 384}, "a");
    Tensor res1;
    FUNCTION("TestAssign")
    {
        TileShape::Current().SetVecTile(1, dim3);
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {2, 1, 1, 192});
        TileShape::Current().SetVecTile(dim2, 1, dim2, dim3);
        res1 = Exp(test);
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_assemble.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestAssign");
    npu::tile_fwk::PreGraphProcess preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 30;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 30 operations";
    EXPECT_EQ(tensorMagicWithColorSet.size() > 0, true) << "There should be many tensor magic with color";
}

TEST_F(PreGraphTest, TestView)
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    std::vector<int64_t> shape1{128, 128};
    std::vector<int64_t> shape2{64, 64};
    std::vector<int64_t> shape3{16, 256};

    SetUpPassStrategy();
    ConfigManager::Instance();

    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor1");
    Tensor out_tensor(DT_FP32, shape3, "out_tensor");

    FUNCTION("PreGraphFunction")
    {
        TileShape::Current().SetVecTile({64, 64});
        auto a = View(in_tensor, shape2, {0, 0});
        auto b = View(in_tensor1, shape2, {32, 32});
        auto a0 = Add(a, Element(DataType::DT_FP32, 0.0f));
        auto a1 = Reshape(a0, shape3);
        auto b0 = Mul(b, Element(DataType::DT_FP32, 0.1f));
        auto b1 = Reshape(b0, shape3);
        out_tensor = Add(a1, b1);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto& op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_view.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PreGraphFunction");
    npu::tile_fwk::PreGraphProcess preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 24;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 24 operations";
    EXPECT_EQ(tensorMagicWithColorSet.size() > 0, true) << "There should be many tensor magic with color";
}

TEST_F(PreGraphTest, TestTransposeDatamove)
{
    int B = 3;
    int N = 2;
    int S = 128;
    int NUM_1 = 1;

    std::vector<int64_t> shape0{B, N, S};
    std::vector<int64_t> shape1{N, B, S};
    std::vector<int64_t> tiledShape{NUM_1, NUM_1, S};

    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "input");
    G.AddTensor(DataType::DT_FP32, shape1, "output");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandTransposeDatamove(G, B, N, S);

    auto input = G.GetTensor("input");
    input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto output = G.GetTensor("output");
    output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.SetInCast({"input"});
    G.SetOutCast({"output"});
    Function* function = G.GetFunction();
    const int SUBGRAPH_NUM = 6;
    function->SetTotalSubGraphCount(SUBGRAPH_NUM);
    // 确认构图完毕
    constexpr int opNumBefore = 18;
    constexpr int assembleNumBefore = 6;
    auto assembleNumCountBefore = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(assembleNumCountBefore, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.RunOnFunction(*function);
    preGraph.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 12;
    constexpr int assembleNumafter = 0;
    auto assembleNumCountAfter = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(assembleNumCountAfter, assembleNumafter) << assembleNumafter << " OP_ASSEMBLE before pass";
    EXPECT_EQ(opNumBefore - opNumAfter, assembleNumCountBefore - assembleNumCountAfter)
        << " only OP_ASSEMBLE should be removed";
}

TEST_F(PreGraphTest, TestTransposeDatamoveExp)
{
    int B = 3;
    int N = 2;
    int S = 128;

    std::vector<int64_t> shape0{B, N, S};
    std::vector<int64_t> shape1{N, B, S};

    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "input");
    G.AddTensor(DataType::DT_FP32, shape1, "output");
    G.AddTensor(DataType::DT_FP32, shape1, "output2");
    G.AddTensor(DataType::DT_FP32, shape1, "outInnerTemp");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandTransposeDatamove(G, B, N, S, true);
    TileExpandExp(G, B, N, S);

    auto input = G.GetTensor("input");
    input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto output = G.GetTensor("output");
    output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto output2 = G.GetTensor("output2");
    output2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto outInnerTemp = G.GetTensor("outInnerTemp");
    outInnerTemp->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.SetInCast({"input"});
    G.SetOutCast({"output", "output2"});
    Function* function = G.GetFunction();
    const int SUBGRAPH_NUM = 8;
    function->SetTotalSubGraphCount(SUBGRAPH_NUM);
    // 确认构图完毕
    constexpr int opNumBefore = 30;
    constexpr int assembleNumBefore = 12;
    auto assembleNumCountBefore = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(assembleNumCountBefore, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.RunOnFunction(*function);
    preGraph.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 30;
    constexpr int assembleNumafter = 12;
    auto assembleNumCountAfter = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(assembleNumCountAfter, assembleNumafter) << assembleNumafter << " OP_ASSEMBLE before pass";
    EXPECT_EQ(opNumBefore - opNumAfter, assembleNumCountBefore - assembleNumCountAfter)
        << " only OP_ASSEMBLE should be removed";
}

TEST_F(PreGraphTest, TestAddExp)
{
    int N = 2;
    int T = 16;
    std::vector<int64_t> shape0{N * T, N * T};
    // std::vector<int64_t> shape1{T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape0, "out1");
    G.AddTensor(DataType::DT_FP32, shape0, "out2");
    G.AddTensor(DataType::DT_FP32, shape0, "addOutUb");
    G.AddTensor(DataType::DT_FP32, shape0, "expOutUb");

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto addOutUb = G.GetTensor("addOutUb");
    addOutUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto expOutUb = G.GetTensor("expOutUb");
    expOutUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    /*
    [32, 32] --> View --> [16, 16] --> Add --> [16, 16] --> Assemble --> addOutUb[32, 32] --> Exp --> expOutUb[32, 32]
    --> Copy_Out --> out2
                                                    \--> Copy_Out --> out1
    */
    TileExpandAdd(G, N, T);

    G.AddOp(Opcode::OP_EXP, {"addOutUb"}, {"expOutUb"}, "Exp_Op");
    auto expOp = G.GetOp("Exp_Op");
    expOp->UpdateSubgraphID(0);

    G.AddOp(Opcode::OP_COPY_OUT, {"expOutUb"}, {"out2"}, "Copy_Out_Exp");
    auto copyOutOp = G.GetOp("Copy_Out_Exp");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(std::vector<int64_t>{0, 0}),
                                                         MemoryType::MEM_UB, OpImmediate::Specified(out2->GetShape()),
                                                         OpImmediate::Specified(out2->tensor->GetRawShape()));
    copyOutOp->SetOpAttribute(attrCopyOut);
    copyOutOp->UpdateSubgraphID(0);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out1", "out2"});
    Function* function = G.GetFunction();
    const int SUBGRAPH_NUM = 1;
    function->SetTotalSubGraphCount(SUBGRAPH_NUM);
    // 确认构图完毕
    constexpr int opNumBefore = 22;
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.RunOnFunction(*function);
    preGraph.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
}

TEST_F(PreGraphTest, PreGraphReShapeOnOcast)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 8, 16}, "vec_in_rel");
    auto vec_in_rel = G.GetTensor("vec_in_rel");
    vec_in_rel->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 8, 16}, "vec_in");
    auto vec_in = G.GetTensor("vec_in");
    vec_in->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    // add op
    G.AddOp(Opcode::OP_VIEW, {"vec_in_rel"}, {"vec_in"}, "VIEW");
    G.AddOp(Opcode::OP_RESHAPE, {"vec_in"}, {"vec_out"}, "RESHAPE");
    // set incast and outcast
    G.SetInCast({"vec_in_rel"});
    G.SetOutCast({"vec_out"});
    // check before pass
    auto inRawMagicBefore = vec_in->GetRawMagic();
    auto outRawMagicBefore = vec_out->GetRawMagic();
    EXPECT_NE(inRawMagicBefore, outRawMagicBefore);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto inRawMagicAfter = vec_in->GetRawMagic();
    auto outRawMagicAfter = vec_out->GetRawMagic();
    EXPECT_EQ(inRawMagicAfter, outRawMagicAfter);
    EXPECT_EQ(outRawMagicBefore, outRawMagicAfter);
}

TEST_F(PreGraphTest, TestFixPipeReconnectGraph)
{
    Program::GetInstance().Reset();
    config::Reset();
    auto funcPtr = std::make_shared<Function>(Program::GetInstance(), "TestFixPipeReconnectGraph",
                                              "TestFixPipeReconnectGraph", nullptr);

    // Build graph
    std::vector<int64_t> shape = {NUM16, NUM16};
    auto tensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_L1);
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_FIX_QUANT_PRE);
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto& copyin = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_COPY_IN, {tensor0}, {tensor1});
    (void)copyin;
    auto& l1CopyFB = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_L1_TO_FIX_QUANT_PRE, {tensor1}, {tensor2});
    (void)l1CopyFB;
    auto& aMulB = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_A_MUL_B, {tensor2}, {tensor3});
    aMulB.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, NUM10));
    aMulB.SetAttribute(A_MUL_B_RELU_ATTR, 1);
    auto& aMulAccB = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_A_MULACC_B, {tensor3}, {tensor4});
    (void)aMulAccB;
    auto& copyout = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_COPY_OUT, {tensor4}, {tensor5});

    // Test Reconnect Graph
    CubeProcess cubeProcess;
    std::vector<Operation*> l0CCopyOuts{};
    cubeProcess.GetL0CCopyOuts(aMulB, l0CCopyOuts);
    EXPECT_EQ(l0CCopyOuts[0], &copyout);
    cubeProcess.ReconnectGraph(aMulB, l0CCopyOuts);
    auto tensor2Consumer = tensor2->GetConsumers().begin();
    EXPECT_EQ(*tensor2Consumer, &copyout);
    auto scaleValue = (copyout.HasAttr(A_MUL_B_SCALE_ATTR)) ? copyout.GetElementAttribute(A_MUL_B_SCALE_ATTR) :
                                                              Element(DataType::DT_UINT64, 0);
    auto reluType = (copyout.HasAttr(A_MUL_B_RELU_ATTR)) ? copyout.GetIntAttribute(A_MUL_B_RELU_ATTR) : 0;
    EXPECT_EQ(scaleValue, Element(DataType::DT_UINT64, NUM10));
    EXPECT_EQ(reluType, 1);
}

TEST_F(PreGraphTest, TestFixPipeReconnectGraphMulAcc)
{
    Program::GetInstance().Reset();
    config::Reset();
    auto funcPtr = std::make_shared<Function>(Program::GetInstance(), "TestFixPipeReconnectGraphMulAcc",
                                              "TestFixPipeReconnectGraphMulAcc", nullptr);

    // Build graph
    std::vector<int64_t> shape = {NUM16, NUM16};
    auto tensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor0->SetMemoryTypeBoth(MemoryType::MEM_L0A);
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_L0B);
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_L1);
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_FIX_QUANT_PRE);
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto& copyin = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_COPY_IN, {tensor3}, {tensor4});
    (void)copyin;
    auto& l1CopyFB = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_L1_TO_FIX_QUANT_PRE, {tensor4}, {tensor5});
    (void)l1CopyFB;
    auto& aMulAccB = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_A_MULACC_B,
                                                    {tensor0, tensor1, tensor2, tensor5}, {tensor6});
    aMulAccB.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, NUM10));
    aMulAccB.SetAttribute(A_MUL_B_RELU_ATTR, 1);
    auto& copyout = IRBuilder().CreateTensorOpStmt(*funcPtr, Opcode::OP_COPY_OUT, {tensor6}, {tensor7});

    // Test Reconnect Graph
    CubeProcess cubeProcess;
    std::vector<Operation*> l0CCopyOuts{};
    EXPECT_EQ(cubeProcess.CheckValidCube(aMulAccB), SUCCESS);
    EXPECT_EQ(cubeProcess.GetL0CCopyOuts(aMulAccB, l0CCopyOuts), SUCCESS);
    ASSERT_EQ(l0CCopyOuts.size(), 1);
    EXPECT_EQ(l0CCopyOuts.front(), &copyout);
    EXPECT_EQ(aMulAccB.GetInputOperandSize(), 4);
    EXPECT_EQ(cubeProcess.ReconnectGraph(aMulAccB, l0CCopyOuts), SUCCESS);
    EXPECT_EQ(aMulAccB.GetInputOperandSize(), 3);
    ASSERT_EQ(tensor5->GetConsumers().size(), 1);
    EXPECT_EQ(*tensor5->GetConsumers().begin(), &copyout);
    EXPECT_EQ(*tensor0->GetConsumers().begin(), &aMulAccB);
    EXPECT_EQ(*tensor1->GetConsumers().begin(), &aMulAccB);
    EXPECT_EQ(*tensor2->GetConsumers().begin(), &aMulAccB);
    auto scaleValue = (copyout.HasAttr(A_MUL_B_SCALE_ATTR)) ? copyout.GetElementAttribute(A_MUL_B_SCALE_ATTR) :
                                                              Element(DataType::DT_UINT64, 0);
    auto reluType = (copyout.HasAttr(A_MUL_B_RELU_ATTR)) ? copyout.GetIntAttribute(A_MUL_B_RELU_ATTR) : 0;
    EXPECT_EQ(scaleValue, Element(DataType::DT_UINT64, NUM10));
    EXPECT_EQ(reluType, 1);
}

void ConstructRemoveRedundantView(ComputationalGraphBuilder& G, bool multi)
{
    // add tensor
    G.AddTensor(DataType::DT_FP16, {16, 64, 64}, "t1");
    G.AddTensor(DataType::DT_FP16, {1, 64, 64}, "t2");
    G.AddTensor(DataType::DT_FP16, {64, 64}, "t3");
    G.AddTensor(DataType::DT_FP16, {64, 64}, "t4");
    G.AddTensor(DataType::DT_FP16, {64, 64}, "t41");
    G.AddTensor(DataType::DT_FP16, {64, 64}, "t42");
    G.AddTensor(DataType::DT_FP16, {64, 64}, "t43");
    // add op
    G.AddOp(Opcode::OP_VIEW, {"t1"}, {"t2"}, "VIEW");
    G.AddOp(Opcode::OP_RESHAPE, {"t2"}, {"t3"}, "RESHAPE");
    G.AddOp(Opcode::OP_COPY_IN, {"t3"}, {"t4"}, "COPYIN");
    G.AddOp(Opcode::OP_ABS, {"t4"}, {"t42"}, "ABS1");
    std::vector<int64_t> offset = {2, 0, 0};
    auto view = G.GetOp("VIEW");
    auto attrA = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{2, 0, 0}, MemoryType::MEM_UNKNOWN,
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{2, 0, 0})));
    view->SetOpAttribute(attrA);
    // set incast and outcast
    G.SetInCast({"t1"});
    G.SetOutCast({"t42", "t43"});

    // another copyin
    if (multi) {
        G.AddTensor(DataType::DT_FP32, {64, 64}, "t41");
        G.SetOutCast({"t41"});
        G.AddOp(Opcode::OP_COPY_IN, {"t3"}, {"t41"}, "COPYIN2");
        auto copyIn2 = G.GetOp("COPYIN2");
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(copyIn2->GetOpAttribute());
        copyAttr->SetFromOffset(OpImmediate::Specified(std::vector<int64_t>{32, 0}));
        G.AddOp(Opcode::OP_ABS, {"t41"}, {"t43"}, "ABS2");
    }
}

TEST_F(PreGraphTest, TestRemoveRedundantView)
{
    ComputationalGraphBuilder G;
    ConstructRemoveRedundantView(G, false);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    EXPECT_EQ(passLocal.Run(*function, "", "", 0), SUCCESS);
    // check after pass
    auto opList = function->Operations();
    int64_t viewCnt = 0;
    for (const auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewCnt;
        }
    }
    EXPECT_EQ(viewCnt, 0);

    auto copyIn = G.GetOp("COPYIN");
    std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
    auto newDynOffset = copyAttr->GetFromOffset();
    EXPECT_EQ(newDynOffset[0].Dump(), "128");
    EXPECT_EQ(newDynOffset[1].Dump(), "0");

    auto newRawShape = copyAttr->GetRawShape();
    EXPECT_EQ(newRawShape[0].Dump(), "1024");
    EXPECT_EQ(newRawShape[1].Dump(), "64");
    auto t = G.GetTensor("t3");
    EXPECT_EQ(t->GetShape(), (std::vector<int64_t>{1024, 64}));
}

TEST_F(PreGraphTest, TestRemoveRedundantViewMultiCopyIn)
{
    ComputationalGraphBuilder G;
    ConstructRemoveRedundantView(G, true);
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    EXPECT_EQ(passLocal.Run(*function, "", "", 0), SUCCESS);
    // check after pass
    auto opList = function->Operations();
    int64_t viewCnt = 0;
    for (const auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewCnt;
        }
    }
    EXPECT_EQ(viewCnt, 0);
    auto copyIn = G.GetOp("COPYIN2");
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
    auto newDynOffset = copyAttr->GetFromOffset();
    EXPECT_EQ(newDynOffset[0].Dump(), "160");
}

void ConstructDynamicValidShapeReshapeCopyInGraph(ComputationalGraphBuilder& G, int64_t l, int64_t lMax)
{
    G.AddTensor(DataType::DT_FP32, {NUM2, 1, l}, MemoryType::MEM_DEVICE_DDR, "input");
    G.AddTensor(DataType::DT_FP32, {1, lMax}, "view_out");
    G.AddTensor(DataType::DT_FP32, {1, 1, lMax}, "reshape_out");
    G.AddTensor(DataType::DT_FP32, {1, 1, lMax}, MemoryType::MEM_UB, "copy_in_out");

    auto input = G.GetTensor("input");
    input->tensor->UpdateRawShape({NUM2, 1, lMax});
    input->tensor->UpdateDynRawShape(CreateTestConstIntVector({NUM2, 1, lMax}));
    input->UpdateDynValidShape(CreateTestConstIntVector({NUM2, 1, l}));
    G.GetTensor("view_out")->UpdateDynValidShape(CreateTestConstIntVector({1, l}));
    G.GetTensor("reshape_out")->UpdateDynValidShape(CreateTestConstIntVector({1, 1, l}));
    G.GetTensor("copy_in_out")->UpdateDynValidShape({});

    G.AddOp(Opcode::OP_VIEW, {"input"}, {"view_out"}, "VIEW");
    auto view = G.GetOp("VIEW");
    std::vector<int64_t> viewOffset{1, 0, 0};
    auto viewAttr = std::make_shared<ViewOpAttribute>(viewOffset, MemoryType::MEM_UNKNOWN,
                                                      OpImmediate::ToSpecified(OpImmediate::Specified(viewOffset)));
    view->SetOpAttribute(viewAttr);
    G.AddOp(Opcode::OP_RESHAPE, {"view_out"}, {"reshape_out"}, "RESHAPE");
    G.AddOp(Opcode::OP_COPY_IN, {"reshape_out"}, {"copy_in_out"}, "COPYIN");
    auto copyIn = G.GetOp("COPYIN");
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified(std::vector<int64_t>{0, 0, 0}));
}

void VerifyDynamicValidShapeReshapeCopyInGraph(ComputationalGraphBuilder& G, int64_t l, int64_t lMax)
{
    auto input = G.GetTensor("input");
    auto view = G.GetOp("VIEW");
    auto reshape = G.GetOp("RESHAPE");
    auto copyIn = G.GetOp("COPYIN");
    EXPECT_TRUE(view->IsDeleted());
    EXPECT_EQ(reshape->GetIOperands().front(), input);

    auto reshapeOut = G.GetTensor("reshape_out");
    EXPECT_EQ(reshapeOut->GetShape(), (std::vector<int64_t>{NUM2, 1, lMax}));
    auto reshapeDynValidShape = reshapeOut->GetDynValidShape();
    ASSERT_EQ(reshapeDynValidShape.size(), 3);
    EXPECT_EQ(reshapeDynValidShape[0].Dump(), "1");
    EXPECT_EQ(reshapeDynValidShape[1].Dump(), "1");
    EXPECT_EQ(reshapeDynValidShape[2].Dump(), std::to_string(l));

    auto reshapeDynRawShape = reshapeOut->tensor->GetDynRawShape();
    ASSERT_EQ(reshapeDynRawShape.size(), 3);
    EXPECT_EQ(reshapeDynRawShape[0].Dump(), std::to_string(NUM2));
    EXPECT_EQ(reshapeDynRawShape[1].Dump(), "1");
    EXPECT_EQ(reshapeDynRawShape[2].Dump(), std::to_string(lMax));

    auto newCopyAttr = std::static_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
    auto fromOffset = newCopyAttr->GetFromOffset();
    ASSERT_EQ(fromOffset.size(), 3);
    EXPECT_EQ(fromOffset[0].Dump(), std::to_string(1));
    EXPECT_EQ(fromOffset[1].Dump(), "0");
    EXPECT_EQ(fromOffset[2].Dump(), "0");

    auto rawShape = newCopyAttr->GetRawShape();
    ASSERT_EQ(rawShape.size(), 3);
    EXPECT_EQ(rawShape[0].Dump(), std::to_string(NUM2));
    EXPECT_EQ(rawShape[1].Dump(), "1");
    EXPECT_EQ(rawShape[2].Dump(), std::to_string(lMax));

    auto toDynValidShape = newCopyAttr->GetToDynValidShape();
    ASSERT_EQ(toDynValidShape.size(), 3);
    EXPECT_EQ(toDynValidShape[0].Dump(), "1");
    EXPECT_EQ(toDynValidShape[1].Dump(), "1");
    EXPECT_EQ(toDynValidShape[2].Dump(), std::to_string(l));
}

TEST_F(PreGraphTest, TestRemoveRedundantViewDynamicValidShapeReshapeCopyIn)
{
    constexpr int64_t l = 26;
    constexpr int64_t lMax = 32;
    ComputationalGraphBuilder G;
    ConstructDynamicValidShapeReshapeCopyInGraph(G, l, lMax);

    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessView(*function), SUCCESS);

    VerifyDynamicValidShapeReshapeCopyInGraph(G, l, lMax);
}

TEST_F(PreGraphTest, TestRemoveRedundantViewMultiReshape)
{
    ComputationalGraphBuilder G;
    // add tensor
    G.AddTensor(DataType::DT_FP16, {4, 6144}, "t1");
    G.AddTensor(DataType::DT_FP16, {4, 1, 32, 192}, "t2");
    G.AddTensor(DataType::DT_FP16, {4, 1, 32, 128}, "t3");
    auto inputTensor2 = G.GetTensor("t2");
    auto inputTensor3 = G.GetTensor("t3");
    inputTensor3->tensor = inputTensor2->tensor;
    inputTensor3->tensor->UpdateRawShape({4, 1, 32, 192});

    G.AddTensor(DataType::DT_FP16, {4, 32, 128}, "t4");

    // add op
    G.AddOp(Opcode::OP_RESHAPE, {"t1"}, {"t2"}, "RESHAPE1");
    G.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "VIEW");
    auto view = G.GetOp("VIEW");
    auto attrA = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0, 0, 0}, MemoryType::MEM_UNKNOWN,
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0, 0, 0})));
    view->SetOpAttribute(attrA);
    G.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "RESHAPE2");
    for (size_t i = 0; i < 32; ++i) {
        std::string tensorName = "t5-" + std::to_string(i);
        G.AddTensor(DataType::DT_FP16, {4, 1, 128}, tensorName);
        std::string copyInName = "COPYIN" + tensorName;
        G.AddOp(Opcode::OP_COPY_IN, {"t4"}, {tensorName}, copyInName);
        auto copyIn = G.GetOp(copyInName);
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
        copyAttr->SetFromOffset(OpImmediate::Specified(std::vector<int64_t>{0, static_cast<int64_t>(i), 0}));
        copyAttr->SetShape(OpImmediate::Specified({4, 1, 128}));
    }

    // set incast and outcast
    G.SetInCast({"t1"});

    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    EXPECT_EQ(passLocal.Run(*function, "", "", 0), SUCCESS);
    // check after pass
    auto opList = function->Operations();
    int64_t viewCnt = 0;
    for (const auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewCnt;
        }
    }
    EXPECT_EQ(viewCnt, 0);
}

inline void CheckProcessReshape(Function* function)
{
    PreGraphProcess passLocal;
    EXPECT_EQ(passLocal.Run(*function, "", "", 0), SUCCESS);
    // check after pass
    auto opList = function->Operations();
    int64_t viewCnt = 0;
    int64_t reshapeCnt = 0;

    for (const auto& op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewCnt;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeCnt++;
        }
    }
    EXPECT_EQ(viewCnt, 1);
    EXPECT_EQ(reshapeCnt, 2); // tmp Reshape Remove
}
TEST_F(PreGraphTest, TestProcessReshape)
{
    ComputationalGraphBuilder G;
    // add tensor
    G.AddTensor(DataType::DT_FP16, {16, 24576}, "t1");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 192}, "t2");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "t3");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "t4");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "t5");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "t6");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "o1");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "o2");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "o3");

    // add op
    G.AddOp(Opcode::OP_RESHAPE, {"t1"}, {"t2"}, "RESHAPE1");
    G.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "VIEW");
    auto viewAttr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0, 0, 0}, MemoryType::MEM_UNKNOWN,
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0, 0, 0})));
    G.GetOp("VIEW")->SetOpAttribute(viewAttr);
    G.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "RESHAPE2");
    G.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t5"}, "COPY_IN1");
    G.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t6"}, "COPY_IN2");
    G.AddOp(Opcode::OP_ABS, {"t5"}, {"o1"}, "ABS1");
    G.AddOp(Opcode::OP_ABS, {"t6"}, {"o2"}, "ABS2");
    G.AddOp(Opcode::OP_ABS, {"t4"}, {"o3"}, "ABS3");

    // set incast and outcast
    G.SetInCast({"t1"});
    G.SetOutCast({"o1", "o2", "o3"});

    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CheckProcessReshape(function);

    // Verify RESHAPE2 exists and has correct shapes
    auto* reshape2Op = G.GetOp("RESHAPE2");
    EXPECT_NE(reshape2Op, nullptr) << "RESHAPE2 should exist";
    auto reshape2Input = reshape2Op->GetIOperands().front();
    EXPECT_EQ(reshape2Input->GetShape(), (std::vector<int64_t>{16, 1, 128, 128}))
        << "RESHAPE2 input shape should be {16, 1, 128, 128}";

    // Verify RESHAPE1 : connect to copyin
    auto* reshape1Op = G.GetOp("RESHAPE1");
    int64_t copyinCnt = 0;
    for (auto consumer : reshape1Op->GetOutputOperand(0)->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_COPY_IN) {
            ++copyinCnt;
        }
    }
    EXPECT_EQ(copyinCnt, 2);
}

TEST_F(PreGraphTest, TestRemoveViewMultiReshapeErrCondition)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 24576}, "t1");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 192}, "t2");
    G.AddTensor(DataType::DT_FP16, {16, 1, 128, 128}, "t3");
    G.AddOp(Opcode::OP_RESHAPE, {"t1"}, {"t2"}, "RESHAPE1");
    G.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "VIEW");

    RemoveRedundantAssemble pass;
    std::vector<std::pair<Operation*, Operation*>> multiReshapeVector;
    multiReshapeVector.push_back({G.GetOp("RESHAPE1"), G.GetOp("VIEW")});

    EXPECT_EQ(pass.RemoveViewMultiReshape(multiReshapeVector), FAILED);

    auto viewOp = G.GetOp("VIEW");
    viewOp->oOperand[0] = nullptr;
    EXPECT_EQ(pass.RemoveViewMultiReshape(multiReshapeVector), FAILED);
}

void CompareOpImmediateVector(const std::vector<OpImmediate>& result, const std::vector<int64_t>& expect)
{
    EXPECT_EQ(result.size(), expect.size());
    for (size_t idx = 0; idx < result.size(); ++idx) {
        EXPECT_EQ(result[idx].Dump(), std::to_string(expect[idx]));
    }
}

TEST_F(PreGraphTest, TestMergeViewToCopyInOffset)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 16}, MemoryType::MEM_DEVICE_DDR, "view_input");
    G.AddTensor(DataType::DT_FP16, {8, 8}, MemoryType::MEM_DEVICE_DDR, "view_output");
    G.AddTensor(DataType::DT_FP16, {4, 4}, MemoryType::MEM_UB, "copy_output");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output"}, "VIEW");
    G.GetOp("VIEW")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{2, 3}));
    G.AddOp(Opcode::OP_COPY_IN, {"view_output"}, {"copy_output"}, "COPYIN");
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN")->GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified({4, 5}));

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessViewToCopyIn(*G.GetFunction()), SUCCESS);

    EXPECT_TRUE(G.GetOp("VIEW")->IsDeleted());
    EXPECT_EQ(G.GetOp("COPYIN")->GetIOperands().front(), G.GetTensor("view_input"));
    CompareOpImmediateVector(copyAttr->GetFromOffset(), {6, 8});
    CompareOpImmediateVector(copyAttr->GetShape(), {4, 4});
    CompareOpImmediateVector(copyAttr->GetRawShape(), {16, 16});
    CompareOpImmediateVector(copyAttr->GetToDynValidShape(), {4, 4});
}

TEST_F(PreGraphTest, TestMergeViewToMultiCopyInOffset)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 16}, MemoryType::MEM_DEVICE_DDR, "view_input");
    G.AddTensor(DataType::DT_FP16, {8, 8}, MemoryType::MEM_DEVICE_DDR, "view_output");
    G.AddTensor(DataType::DT_FP16, {4, 4}, MemoryType::MEM_UB, "copy_output1");
    G.AddTensor(DataType::DT_FP16, {4, 4}, MemoryType::MEM_UB, "copy_output2");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output"}, "VIEW");
    G.GetOp("VIEW")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{2, 3}));
    G.AddOp(Opcode::OP_COPY_IN, {"view_output"}, {"copy_output1"}, "COPYIN1");
    G.AddOp(Opcode::OP_COPY_IN, {"view_output"}, {"copy_output2"}, "COPYIN2");
    auto copyAttr1 = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN1")->GetOpAttribute());
    auto copyAttr2 = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN2")->GetOpAttribute());
    copyAttr1->SetFromOffset(OpImmediate::Specified({4, 5}));
    copyAttr2->SetFromOffset(OpImmediate::Specified({6, 7}));

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessViewToCopyIn(*G.GetFunction()), SUCCESS);

    EXPECT_TRUE(G.GetOp("VIEW")->IsDeleted());
    CompareOpImmediateVector(copyAttr1->GetFromOffset(), {6, 8});
    CompareOpImmediateVector(copyAttr2->GetFromOffset(), {8, 10});
}

TEST_F(PreGraphTest, TestMergeDynamicViewToCopyInOffset)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 16}, MemoryType::MEM_DEVICE_DDR, "view_input");
    G.AddTensor(DataType::DT_FP16, {8, 8}, MemoryType::MEM_DEVICE_DDR, "view_output");
    G.AddTensor(DataType::DT_FP16, {4, 4}, MemoryType::MEM_UB, "copy_output");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output"}, "VIEW");
    auto dynOffset = OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{2, 3}));
    G.GetOp("VIEW")->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_UNKNOWN, dynOffset));
    G.AddOp(Opcode::OP_COPY_IN, {"view_output"}, {"copy_output"}, "COPYIN");
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN")->GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified({4, 5}));

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessViewToCopyIn(*G.GetFunction()), SUCCESS);

    EXPECT_TRUE(G.GetOp("VIEW")->IsDeleted());
    CompareOpImmediateVector(copyAttr->GetFromOffset(), {6, 8});
}

TEST_F(PreGraphTest, TestKeepViewWithNonCopyInConsumer)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 16}, MemoryType::MEM_DEVICE_DDR, "view_input");
    G.AddTensor(DataType::DT_FP16, {8, 8}, MemoryType::MEM_DEVICE_DDR, "view_output");
    G.AddTensor(DataType::DT_FP16, {4, 4}, MemoryType::MEM_UB, "copy_output");
    G.AddTensor(DataType::DT_FP16, {8, 8}, "abs_output");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output"}, "VIEW");
    G.GetOp("VIEW")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{2, 3}));
    G.AddOp(Opcode::OP_COPY_IN, {"view_output"}, {"copy_output"}, "COPYIN");
    G.AddOp(Opcode::OP_ABS, {"view_output"}, {"abs_output"}, "ABS");
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN")->GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified({4, 5}));

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessViewToCopyIn(*G.GetFunction()), SUCCESS);

    EXPECT_FALSE(G.GetOp("VIEW")->IsDeleted());
    EXPECT_EQ(G.GetOp("COPYIN")->GetIOperands().front(), G.GetTensor("view_output"));
    CompareOpImmediateVector(copyAttr->GetFromOffset(), {4, 5});
}

TEST_F(PreGraphTest, TestMergeViewToCopyInWithoutCopyAttr)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 16}, MemoryType::MEM_DEVICE_DDR, "view_input");
    G.AddTensor(DataType::DT_FP16, {8, 8}, MemoryType::MEM_DEVICE_DDR, "view_output");
    G.AddTensor(DataType::DT_FP16, {4, 4}, MemoryType::MEM_UB, "copy_output");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output"}, "VIEW");
    G.GetOp("VIEW")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{2, 3}));
    G.AddOp(Opcode::OP_COPY_IN, {"view_output"}, {"copy_output"}, "COPYIN");
    G.GetOp("COPYIN")->GetOpAttribute() = nullptr;

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessViewToCopyIn(*G.GetFunction()), SUCCESS);

    EXPECT_TRUE(G.GetOp("VIEW")->IsDeleted());
    EXPECT_EQ(G.GetOp("COPYIN")->GetIOperands().front(), G.GetTensor("view_input"));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN")->GetOpAttribute());
    ASSERT_NE(copyAttr, nullptr);
    CompareOpImmediateVector(copyAttr->GetFromOffset(), {2, 3});
    CompareOpImmediateVector(copyAttr->GetShape(), {4, 4});
    CompareOpImmediateVector(copyAttr->GetRawShape(), {16, 16});
    CompareOpImmediateVector(copyAttr->GetToDynValidShape(), {4, 4});
}

TEST_F(PreGraphTest, TestKeepViewWhenCopyInOffsetRankMismatch)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {16, 16}, MemoryType::MEM_DEVICE_DDR, "view_input");
    G.AddTensor(DataType::DT_FP16, {8, 8}, MemoryType::MEM_DEVICE_DDR, "view_output");
    G.AddTensor(DataType::DT_FP16, {8, 8}, MemoryType::MEM_UB, "copy_output");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output"}, "VIEW");
    G.GetOp("VIEW")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{2, 3}));
    G.AddOp(Opcode::OP_COPY_IN, {"view_output"}, {"copy_output"}, "COPYIN");
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN")->GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified({4}));

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessViewToCopyIn(*G.GetFunction()), SUCCESS);

    EXPECT_FALSE(G.GetOp("VIEW")->IsDeleted());
    EXPECT_EQ(G.GetOp("COPYIN")->GetIOperands().front(), G.GetTensor("view_output"));
    CompareOpImmediateVector(copyAttr->GetFromOffset(), {4});
}

TEST_F(PreGraphTest, TestRemoveViewSingleReshapeNormalizesCopyInRawShape)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {1, 64, 128}, "view_input");
    G.AddTensor(DataType::DT_FP16, {1, 64, 128}, "reshape_input");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "reshape_output");
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_UB, "copy_dst");
    G.GetTensor("view_input")->tensor->UpdateRawShape({-1, 64, 128});

    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"reshape_input"}, "VIEW");
    auto viewAttr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0, 0}, MemoryType::MEM_UNKNOWN,
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0, 0})));
    G.GetOp("VIEW")->SetOpAttribute(viewAttr);
    G.AddOp(Opcode::OP_RESHAPE, {"reshape_input"}, {"reshape_output"}, "RESHAPE");
    G.AddOp(Opcode::OP_COPY_IN, {"reshape_output"}, {"copy_dst"}, "COPYIN");
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYIN")->GetOpAttribute());

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.ProcessView(*G.GetFunction()), SUCCESS);
    CompareOpImmediateVector(copyAttr->GetRawShape(), {-1, 128});
    EXPECT_EQ(G.GetTensor("reshape_output")->tensor->GetRawShape(), (std::vector<int64_t>{-1, 128}));
}

TEST_F(PreGraphTest, TestRemoveRedundantAssemble)
{
    ComputationalGraphBuilder G;
    // add tensor
    G.AddTensor(DataType::DT_FP16, {128, 256}, "t2");
    std::vector<std::vector<int64_t>> offsets = {{0, 0}, {0, 128}, {64, 0}, {64, 128}};
    for (size_t i = 0; i < 4; i++) {
        std::string tensorName = "t1" + std::to_string(i);
        std::string copyName = "COPYOUT" + std::to_string(i);
        G.AddTensor(DataType::DT_FP16, {64, 128}, tensorName);
        G.AddOp(Opcode::OP_COPY_OUT, {tensorName}, {"t2"}, copyName);
        auto copy = G.GetOp(copyName);
        auto copyAttr = std::make_shared<CopyOpAttribute>(MemoryType::MEM_L0C, OpImmediate::Specified(offsets[i]),
                                                          OpImmediate::Specified(std::vector<int64_t>{64, 128}),
                                                          OpImmediate::Specified(std::vector<int64_t>{64, 128}));
        copy->SetOpAttribute(copyAttr);
    }

    G.AddTensor(DataType::DT_FP16, {1, 128, 256}, "t3");
    G.AddTensor(DataType::DT_FP16, {3, 128, 256}, MemoryType::MEM_DEVICE_DDR, "t4");
    auto tensor3 = G.GetTensor("t3");
    auto tensor4 = G.GetTensor("t4");
    tensor3->tensor = tensor4->tensor;
    // add op

    G.AddOp(Opcode::OP_RESHAPE, {"t2"}, {"t3"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"t3"}, {"t4"}, "ASSEMBLE");
    std::vector<int64_t> offset = {0, 0, 0};
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{2, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{2, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);
    // set incast and outcast
    G.SetOutCast({"t4"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    EXPECT_EQ(passLocal.Run(*function, "", "", 0), SUCCESS);
}

TEST_F(PreGraphTest, TestRemoveRedundantAssembleNormalizesCopyOutRawShape)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_L0C, "copy_src");
    G.AddTensor(DataType::DT_FP16, {64, 128}, "reshape_input");
    G.AddTensor(DataType::DT_FP16, {1, 64, 128}, "reshape_output");
    G.AddTensor(DataType::DT_FP16, {-1, 64, 128}, MemoryType::MEM_DEVICE_DDR, "assemble_output");

    auto reshapeOutput = G.GetTensor("reshape_output");
    auto assembleOutput = G.GetTensor("assemble_output");
    reshapeOutput->tensor = assembleOutput->tensor;

    G.AddOp(Opcode::OP_COPY_OUT, {"copy_src"}, {"reshape_input"}, "COPYOUT");
    auto copyOut = G.GetOp("COPYOUT");
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{0, 0}),
        OpImmediate::Specified(std::vector<int64_t>{64, 128}), OpImmediate::Specified(std::vector<int64_t>{64, 128}));
    copyOut->SetOpAttribute(copyAttr);

    G.AddOp(Opcode::OP_RESHAPE, {"reshape_input"}, {"reshape_output"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"reshape_output"}, {"assemble_output"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);
    G.SetOutCast({"assemble_output"});

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    CompareOpImmediateVector(copyAttr->GetRawShape(), {-1, 128});
    EXPECT_EQ(G.GetTensor("reshape_input")->tensor->GetRawShape(), (std::vector<int64_t>{-1, 128}));
}

TEST_F(PreGraphTest, TestRemoveRedundantAssembleKeepsReshapeInputOffset)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_L0C, "copy_src");
    G.AddTensor(DataType::DT_FP16, {8, 32}, "reshape_input");
    G.AddTensor(DataType::DT_FP16, {1, 8, 32}, "reshape_output");
    G.AddTensor(DataType::DT_FP16, {8, 8, 32}, MemoryType::MEM_DEVICE_DDR, "assemble_output");

    auto reshapeInput = G.GetTensor("reshape_input");
    auto reshapeOutput = G.GetTensor("reshape_output");
    auto assembleOutput = G.GetTensor("assemble_output");
    reshapeOutput->tensor = assembleOutput->tensor;

    G.AddOp(Opcode::OP_COPY_OUT, {"copy_src"}, {"reshape_input"}, "COPYOUT");
    auto copyOut = G.GetOp("COPYOUT");
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{0, 0}),
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{8, 32}));
    copyOut->SetOpAttribute(copyAttr);

    G.AddOp(Opcode::OP_RESHAPE, {"reshape_input"}, {"reshape_output"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"reshape_output"}, {"assemble_output"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{6, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{6, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);
    G.SetOutCast({"assemble_output"});

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    CompareOpImmediateVector(copyAttr->GetToOffset(), {48, 0});
    EXPECT_EQ(reshapeInput->GetOffset(), (std::vector<int64_t>{0, 0}));
    EXPECT_TRUE(reshapeInput->GetDynOffset().empty());
    EXPECT_EQ(reshapeInput->tensor->GetRawShape(), (std::vector<int64_t>{64, 32}));
    EXPECT_EQ(reshapeOutput->tensor->GetRawShape(), (std::vector<int64_t>{8, 8, 32}));
}

TEST_F(PreGraphTest, TestMergeCompleteViewCopyoutFamily)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_L0C, "view_input");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L0C, "view_output0");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L0C, "view_output1");
    G.AddTensor(DataType::DT_FP16, {-1, -1}, MemoryType::MEM_DEVICE_DDR, "copyout_output");
    G.GetTensor("copyout_output")
        ->UpdateDynValidShape({CreateTestScalarVar("copyout_dim0"), CreateTestScalarVar("copyout_dim1")});
    auto viewInput = G.GetTensor("view_input");
    G.GetTensor("view_output0")->tensor = viewInput->tensor;
    G.GetTensor("view_output0")->UpdateOffset({0, 0});
    G.GetTensor("view_output1")->tensor = viewInput->tensor;
    G.GetTensor("view_output1")->UpdateOffset({0, 64});

    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output0"}, "VIEW0");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output1"}, "VIEW1");
    G.GetOp("VIEW0")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.GetOp("VIEW1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 64}));
    G.AddOp(Opcode::OP_COPY_OUT, {"view_output0"}, {"copyout_output"}, "COPYOUT0");
    G.AddOp(Opcode::OP_COPY_OUT, {"view_output1"}, {"copyout_output"}, "COPYOUT1");
    auto retainedMagic = G.GetOp("COPYOUT0")->GetOpMagic();
    auto copyAttr0 = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYOUT0")->GetOpAttribute());
    auto copyAttr1 = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYOUT1")->GetOpAttribute());
    auto copyoutBase0 = CreateTestScalarVar("copyout_base0");
    auto copyoutBase1 = CreateTestScalarVar("copyout_base1");
    auto copyoutRawDim0 = CreateTestScalarVar("copyout_raw_dim0");
    auto copyoutRawDim1 = CreateTestScalarVar("copyout_raw_dim1");
    copyAttr0->SetToOffset(OpImmediate::Specified(std::vector<SymbolicScalar>{copyoutBase0, copyoutBase1}));
    copyAttr1->SetToOffset(OpImmediate::Specified(std::vector<SymbolicScalar>{copyoutBase0, copyoutBase1 + 64}));
    copyAttr0->SetRawShape(OpImmediate::Specified(std::vector<SymbolicScalar>{copyoutRawDim0, copyoutRawDim1}));
    copyAttr1->SetRawShape(OpImmediate::Specified(std::vector<SymbolicScalar>{copyoutRawDim0, copyoutRawDim1}));
    GmOutOfRangeCheckInfo checkInfo{OpImmediate::Specified(1), OpImmediate::Specified(2), OpImmediate::Specified(3),
                                    GmOutOfRangeCheckInfo::AccessType::WRITE_GM};
    copyAttr0->SetGmOutOfRangeCheck(checkInfo);
    G.SetOutCast({"copyout_output"});

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(G.GetFunction()->Operations().size(), 4);
    EXPECT_EQ(G.GetOp("COPYOUT0")->GetIOperands().front(), G.GetTensor("view_output0"));
    EXPECT_EQ(G.GetOp("COPYOUT1")->GetIOperands().front(), G.GetTensor("view_output1"));

    config::SetPassGlobalConfig(TEST_ENABLE_TRUE_KEY, true);
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);

    ASSERT_EQ(G.GetFunction()->Operations().size(), 1);
    auto* retainedCopyout = G.GetFunction()->GetOpByOpMagic(retainedMagic);
    ASSERT_NE(retainedCopyout, nullptr);
    EXPECT_EQ(retainedCopyout->GetIOperands().front(), viewInput);
    auto retainedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(retainedCopyout->GetOpAttribute());
    ASSERT_EQ(retainedAttr, copyAttr0);
    CompareOpImmediateVector(retainedAttr->GetShape(), {64, 128});
    ASSERT_EQ(retainedAttr->GetRawShape().size(), 2);
    EXPECT_EQ(retainedAttr->GetRawShape()[0].GetSpecifiedValue().Dump(), copyoutRawDim0.Dump());
    EXPECT_EQ(retainedAttr->GetRawShape()[1].GetSpecifiedValue().Dump(), copyoutRawDim1.Dump());
    CompareOpImmediateVector(retainedAttr->GetFromDynValidShape(), {64, 128});
    ASSERT_NE(retainedAttr->GetGmOutOfRangeCheck(), nullptr);
    EXPECT_EQ(retainedAttr->GetGmOutOfRangeCheck()->accessType, GmOutOfRangeCheckInfo::AccessType::WRITE_GM);
}

TEST_F(PreGraphTest, TestKeepViewCopyoutWhenTargetTranslationMismatch)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_L0C, "view_input");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L0C, "view_output0");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L0C, "view_output1");
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_DEVICE_DDR, "copyout_output");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output0"}, "VIEW0");
    G.AddOp(Opcode::OP_VIEW, {"view_input"}, {"view_output1"}, "VIEW1");
    G.GetOp("VIEW0")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    G.GetOp("VIEW1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 64}));
    G.AddOp(Opcode::OP_COPY_OUT, {"view_output0"}, {"copyout_output"}, "COPYOUT0");
    G.AddOp(Opcode::OP_COPY_OUT, {"view_output1"}, {"copyout_output"}, "COPYOUT1");
    auto copyAttr0 = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYOUT0")->GetOpAttribute());
    auto copyAttr1 = std::static_pointer_cast<CopyOpAttribute>(G.GetOp("COPYOUT1")->GetOpAttribute());
    copyAttr0->SetToOffset(OpImmediate::Specified({0, 0}));
    copyAttr1->SetToOffset(OpImmediate::Specified({0, 32}));

    config::SetPassGlobalConfig(TEST_ENABLE_TRUE_KEY, true);
    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(G.GetFunction()->Operations().size(), 4);
    EXPECT_EQ(G.GetOp("COPYOUT0")->GetIOperands().front(), G.GetTensor("view_output0"));
    EXPECT_EQ(G.GetOp("COPYOUT1")->GetIOperands().front(), G.GetTensor("view_output1"));
}

TEST_F(PreGraphTest, TestMergeCompleteCopyinAssembleFamily)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_DEVICE_DDR, "copyin_input");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L1, "copyin_output0");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L1, "copyin_output1");
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_L1, "assemble_output");
    G.AddOp(Opcode::OP_COPY_IN, {"copyin_input"}, {"copyin_output0"}, "COPYIN0");
    G.AddOp(Opcode::OP_COPY_IN, {"copyin_input"}, {"copyin_output1"}, "COPYIN1");
    auto copyinBase0 = CreateTestScalarVar("copyin_base0");
    auto copyinBase1 = CreateTestScalarVar("copyin_base1");
    auto copyAttr0 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(std::vector<SymbolicScalar>{copyinBase0, copyinBase1}), MemoryType::MEM_L1,
        OpImmediate::Specified({64, 64}), OpImmediate::Specified({64, 128}), OpImmediate::Specified({64, 64}));
    auto copyAttr1 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(std::vector<SymbolicScalar>{copyinBase0, copyinBase1 + 64}), MemoryType::MEM_L1,
        OpImmediate::Specified({64, 64}), OpImmediate::Specified({64, 128}), OpImmediate::Specified({64, 64}));
    G.GetOp("COPYIN0")->SetOpAttribute(copyAttr0);
    G.GetOp("COPYIN1")->SetOpAttribute(copyAttr1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"copyin_output0"}, {"assemble_output"}, "ASSEMBLE0");
    G.AddOp(Opcode::OP_ASSEMBLE, {"copyin_output1"}, {"assemble_output"}, "ASSEMBLE1");
    G.GetOp("ASSEMBLE0")
        ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 0}));
    G.GetOp("ASSEMBLE1")
        ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 64}));
    auto retainedMagic = G.GetOp("COPYIN0")->GetOpMagic();
    G.SetOutCast({"assemble_output"});

    config::SetPassGlobalConfig(TEST_ENABLE_TRUE_KEY, true);
    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);

    ASSERT_EQ(G.GetFunction()->Operations().size(), 1);
    auto* retainedCopyin = G.GetFunction()->GetOpByOpMagic(retainedMagic);
    ASSERT_NE(retainedCopyin, nullptr);
    EXPECT_EQ(retainedCopyin->GetOOperands().front(), G.GetTensor("assemble_output"));
    auto retainedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(retainedCopyin->GetOpAttribute());
    ASSERT_EQ(retainedAttr, copyAttr0);
    EXPECT_EQ(retainedAttr->GetFromOffset()[0].Dump(), copyinBase0.Dump());
    EXPECT_EQ(retainedAttr->GetFromOffset()[1].Dump(), copyinBase1.Dump());
    CompareOpImmediateVector(retainedAttr->GetShape(), {64, 128});
    CompareOpImmediateVector(retainedAttr->GetRawShape(), {64, 128});
    CompareOpImmediateVector(retainedAttr->GetToDynValidShape(), {64, 128});
}

TEST_F(PreGraphTest, TestKeepCopyinAssembleWhenRegionsOverlap)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_DEVICE_DDR, "copyin_input");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L1, "copyin_output0");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_L1, "copyin_output1");
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_L1, "assemble_output");
    G.AddOp(Opcode::OP_COPY_IN, {"copyin_input"}, {"copyin_output0"}, "COPYIN0");
    G.AddOp(Opcode::OP_COPY_IN, {"copyin_input"}, {"copyin_output1"}, "COPYIN1");
    G.GetOp("COPYIN0")->SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L1, OpImmediate::Specified({64, 64}),
        OpImmediate::Specified({64, 128}), OpImmediate::Specified({64, 64})));
    G.GetOp("COPYIN1")->SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 32}), MemoryType::MEM_L1, OpImmediate::Specified({64, 64}),
        OpImmediate::Specified({64, 128}), OpImmediate::Specified({64, 64})));
    G.AddOp(Opcode::OP_ASSEMBLE, {"copyin_output0"}, {"assemble_output"}, "ASSEMBLE0");
    G.AddOp(Opcode::OP_ASSEMBLE, {"copyin_output1"}, {"assemble_output"}, "ASSEMBLE1");
    G.GetOp("ASSEMBLE0")
        ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 0}));
    G.GetOp("ASSEMBLE1")
        ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 32}));

    config::SetPassGlobalConfig(TEST_ENABLE_TRUE_KEY, true);
    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    EXPECT_EQ(G.GetFunction()->Operations().size(), 4);
    EXPECT_EQ(G.GetOp("COPYIN0")->GetOOperands().front(), G.GetTensor("copyin_output0"));
    EXPECT_EQ(G.GetOp("COPYIN1")->GetOOperands().front(), G.GetTensor("copyin_output1"));
}

TEST_F(PreGraphTest, TestRemoveRedundantAssembleRefreshesCopyInAfterReshapeInput)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_L0C, "copy_src");
    G.AddTensor(DataType::DT_FP16, {8, 32}, "reshape_input");
    G.AddTensor(DataType::DT_FP16, {1, 8, 32}, "reshape_output");
    G.AddTensor(DataType::DT_FP16, {8, 8, 32}, MemoryType::MEM_DEVICE_DDR, "assemble_output");
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_UB, "copyin_output");

    auto reshapeInput = G.GetTensor("reshape_input");
    auto reshapeOutput = G.GetTensor("reshape_output");
    auto assembleOutput = G.GetTensor("assemble_output");
    reshapeOutput->tensor = assembleOutput->tensor;

    G.AddOp(Opcode::OP_COPY_OUT, {"copy_src"}, {"reshape_input"}, "COPYOUT");
    auto copyOut = G.GetOp("COPYOUT");
    auto copyOutAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{2, 0}),
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyOut->SetOpAttribute(copyOutAttr);

    G.AddOp(Opcode::OP_RESHAPE, {"reshape_input"}, {"reshape_output"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"reshape_output"}, {"assemble_output"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{6, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{6, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    G.AddOp(Opcode::OP_COPY_IN, {"reshape_input"}, {"copyin_output"}, "COPYIN");
    auto copyIn = G.GetOp("COPYIN");
    auto copyInAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(std::vector<int64_t>{3, 0}), MemoryType::MEM_UB,
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyIn->SetOpAttribute(copyInAttr);
    G.SetOutCast({"assemble_output"});

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    CompareOpImmediateVector(copyOutAttr->GetToOffset(), {50, 0});
    EXPECT_EQ(reshapeInput->GetOffset(), (std::vector<int64_t>{0, 0}));
    EXPECT_TRUE(reshapeInput->GetDynOffset().empty());
    CompareOpImmediateVector(copyInAttr->GetFromOffset(), {51, 0});
    CompareOpImmediateVector(copyInAttr->GetRawShape(), {64, 32});
}

TEST_F(PreGraphTest, TestRemoveRedundantAssembleRebuildsCopyInConsumerLink)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_L0C, "copy_src");
    G.AddTensor(DataType::DT_FP16, {8, 32}, "reshape_input");
    G.AddTensor(DataType::DT_FP16, {1, 8, 32}, "reshape_output");
    G.AddTensor(DataType::DT_FP16, {8, 8, 32}, MemoryType::MEM_DEVICE_DDR, "assemble_output");
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_UB, "copyin_output");

    auto reshapeInput = G.GetTensor("reshape_input");
    auto reshapeOutput = G.GetTensor("reshape_output");
    auto assembleOutput = G.GetTensor("assemble_output");
    reshapeOutput->tensor = assembleOutput->tensor;

    G.AddOp(Opcode::OP_COPY_OUT, {"copy_src"}, {"reshape_input"}, "COPYOUT");
    auto copyOut = G.GetOp("COPYOUT");
    auto copyOutAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{2, 0}),
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyOut->SetOpAttribute(copyOutAttr);

    G.AddOp(Opcode::OP_RESHAPE, {"reshape_input"}, {"reshape_output"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"reshape_output"}, {"assemble_output"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{6, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{6, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    G.AddOp(Opcode::OP_COPY_IN, {"reshape_input"}, {"copyin_output"}, "COPYIN");
    auto copyIn = G.GetOp("COPYIN");
    auto copyInAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(std::vector<int64_t>{3, 0}), MemoryType::MEM_UB,
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyIn->SetOpAttribute(copyInAttr);
    reshapeInput->RemoveConsumer(*copyIn);
    G.SetOutCast({"assemble_output"});

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    CompareOpImmediateVector(copyOutAttr->GetToOffset(), {50, 0});
    EXPECT_EQ(reshapeInput->GetOffset(), (std::vector<int64_t>{0, 0}));
    EXPECT_TRUE(reshapeInput->GetDynOffset().empty());
    CompareOpImmediateVector(copyInAttr->GetFromOffset(), {51, 0});
    CompareOpImmediateVector(copyInAttr->GetRawShape(), {64, 32});
}

TEST_F(PreGraphTest, TestRemoveRedundantAssembleUpdatesCopyInOnceWithMultipleCopyOut)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_L0C, "copy_src0");
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_L0C, "copy_src1");
    G.AddTensor(DataType::DT_FP16, {8, 32}, "reshape_input");
    G.AddTensor(DataType::DT_FP16, {1, 8, 32}, "reshape_output");
    G.AddTensor(DataType::DT_FP16, {8, 8, 32}, MemoryType::MEM_DEVICE_DDR, "assemble_output");
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_UB, "copyin_output");

    auto reshapeInput = G.GetTensor("reshape_input");
    auto reshapeOutput = G.GetTensor("reshape_output");
    auto assembleOutput = G.GetTensor("assemble_output");
    reshapeOutput->tensor = assembleOutput->tensor;

    G.AddOp(Opcode::OP_COPY_OUT, {"copy_src0"}, {"reshape_input"}, "COPYOUT0");
    auto copyOut0 = G.GetOp("COPYOUT0");
    auto copyOutAttr0 = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{2, 0}),
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyOut0->SetOpAttribute(copyOutAttr0);

    G.AddOp(Opcode::OP_COPY_OUT, {"copy_src1"}, {"reshape_input"}, "COPYOUT1");
    auto copyOut1 = G.GetOp("COPYOUT1");
    auto copyOutAttr1 = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{2, 0}),
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyOut1->SetOpAttribute(copyOutAttr1);

    G.AddOp(Opcode::OP_RESHAPE, {"reshape_input"}, {"reshape_output"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"reshape_output"}, {"assemble_output"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{6, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{6, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    G.AddOp(Opcode::OP_COPY_IN, {"reshape_input"}, {"copyin_output"}, "COPYIN");
    auto copyIn = G.GetOp("COPYIN");
    auto copyInAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(std::vector<int64_t>{3, 0}), MemoryType::MEM_UB,
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyIn->SetOpAttribute(copyInAttr);
    G.SetOutCast({"assemble_output"});

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    CompareOpImmediateVector(copyOutAttr0->GetToOffset(), {50, 0});
    CompareOpImmediateVector(copyOutAttr1->GetToOffset(), {50, 0});
    EXPECT_EQ(reshapeInput->GetOffset(), (std::vector<int64_t>{0, 0}));
    EXPECT_TRUE(reshapeInput->GetDynOffset().empty());
    CompareOpImmediateVector(copyInAttr->GetFromOffset(), {51, 0});
    CompareOpImmediateVector(copyInAttr->GetRawShape(), {64, 32});
}

TEST_F(PreGraphTest, TestRemoveRedundantAssembleSkipsCopyInWithMismatchedFromOffsetRank)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_L0C, "copy_src");
    G.AddTensor(DataType::DT_FP16, {8, 32}, "reshape_input");
    G.AddTensor(DataType::DT_FP16, {1, 8, 32}, "reshape_output");
    G.AddTensor(DataType::DT_FP16, {8, 8, 32}, MemoryType::MEM_DEVICE_DDR, "assemble_output");
    G.AddTensor(DataType::DT_FP16, {8, 32}, MemoryType::MEM_UB, "copyin_output");

    auto reshapeOutput = G.GetTensor("reshape_output");
    auto assembleOutput = G.GetTensor("assemble_output");
    reshapeOutput->tensor = assembleOutput->tensor;

    G.AddOp(Opcode::OP_COPY_OUT, {"copy_src"}, {"reshape_input"}, "COPYOUT");
    auto copyOut = G.GetOp("COPYOUT");
    auto copyOutAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{2, 0}),
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyOut->SetOpAttribute(copyOutAttr);

    G.AddOp(Opcode::OP_RESHAPE, {"reshape_input"}, {"reshape_output"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"reshape_output"}, {"assemble_output"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{6, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{6, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    G.AddOp(Opcode::OP_COPY_IN, {"reshape_input"}, {"copyin_output"}, "COPYIN");
    auto copyIn = G.GetOp("COPYIN");
    auto copyInAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(std::vector<int64_t>{3}), MemoryType::MEM_UB,
        OpImmediate::Specified(std::vector<int64_t>{8, 32}), OpImmediate::Specified(std::vector<int64_t>{16, 32}));
    copyIn->SetOpAttribute(copyInAttr);
    G.SetOutCast({"assemble_output"});

    RemoveRedundantAssemble pass;
    EXPECT_EQ(pass.DeleteRedundantAssemble(*G.GetFunction()), SUCCESS);
    CompareOpImmediateVector(copyOutAttr->GetToOffset(), {50, 0});
    CompareOpImmediateVector(copyInAttr->GetFromOffset(), {3});
    CompareOpImmediateVector(copyInAttr->GetRawShape(), {16, 32});
}

// vec_in0 - CopyIn - L1_TO_L0A - A_MUL_B[isCube = true] - CopyOut[no isCube] - vec_out
// vec_in1 - CopyIn - L1_TO_L0B |/
// to
// vec_in0 - CopyIn - L1_TO_L0A - A_MUL_B[isCube = true] - CopyOut[isCube = true] - vec_out
// vec_in1 - CopyIn - L1_TO_L0B |/
TEST_F(PreGraphTest, TestAmulbWithIsCubeCopyOut)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    std::vector<int64_t> shape = {16, 16};
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_DEVICE_DDR, "vec_in0");
    auto vec_in0 = G.GetTensor("vec_in0");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_DEVICE_DDR, "vec_in1");
    auto vec_in1 = G.GetTensor("vec_in1");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_L1, "copy_in0");
    auto copy_in0 = G.GetTensor("copy_in0");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_L1, "copy_in1");
    auto copy_in1 = G.GetTensor("copy_in1");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_L0A, "l1_to_l0a");
    auto l1_to_l0a = G.GetTensor("l1_to_l0a");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_L0B, "l1_to_l0b");
    auto l1_to_l0b = G.GetTensor("l1_to_l0b");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_L0C, "a_mul_b");
    auto a_mul_b = G.GetTensor("a_mul_b");
    G.AddTensor(outputAstDtype, shape, MemoryType::MEM_DEVICE_DDR, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in0"}, {"copy_in0"}, "copyin0");
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in1"}, {"copy_in1"}, "copyin1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"copy_in0"}, {"l1_to_l0a"}, "l1tol0a");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"copy_in1"}, {"l1_to_l0b"}, "l1tol0b");
    G.AddOp(Opcode::OP_A_MUL_B, {"l1_to_l0a", "l1_to_l0b"}, {"a_mul_b"}, "amulb");
    auto amulb = G.GetOp("amulb");
    amulb->SetAttribute(MATMUL_NZ_ATTR, 0);
    amulb->SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, NUM10));
    amulb->SetAttribute(A_MUL_B_RELU_ATTR, 1);
    amulb->SetAttribute(A_MUL_B_ACT_M, 1);
    amulb->SetAttribute(A_MUL_B_ACT_K, 1);
    amulb->SetAttribute(A_MUL_B_ACT_N, 1);
    amulb->SetAttribute(OpAttributeKey::isCube, true);
    G.AddOp(Opcode::OP_COPY_OUT, {"a_mul_b"}, {"vec_out"}, "copyout");
    auto copyout = G.GetOp("copyout");
    // set incast and outcast
    G.SetInCast({"vec_in0"});
    G.SetInCast({"vec_in1"});
    G.SetOutCast({"vec_out"});

    // before
    EXPECT_EQ(amulb->HasAttr(OpAttributeKey::isCube), true);
    EXPECT_EQ(amulb->GetBoolAttribute(OpAttributeKey::isCube), true);
    EXPECT_EQ(copyout->HasAttr(OpAttributeKey::isCube), false);
    // after
    Function* function = G.GetFunction();
    PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.Run(*function, "", "", 0);
    preGraph.PostCheck(*function);
    EXPECT_EQ(copyout->HasAttr(OpAttributeKey::isCube), true);
    EXPECT_EQ(copyout->GetBoolAttribute(OpAttributeKey::isCube), true);
}

// vec_in1 - CopyIn - copy_in1 - L1_TO_L0A[16]
//                                             A_MUL_B - l0c - CopyOut - vec_out[16]应该被重置为32
// vec_in2 - CopyIn - copy_in2 - L1_TO_L0B[16]
TEST_F(PreGraphTest, TestAmulbInputDT_FP16)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype1 = DataType::DT_FP16;
    DataType inputAstDtype2 = DataType::DT_FP16;
    // wrong type
    DataType outputAstDtype = DataType::DT_FP16;
    std::vector<int64_t> shape = {16, 16};
    G.AddTensor(inputAstDtype1, shape, MemoryType::MEM_DEVICE_DDR, "vec_in1");
    auto vec_in1 = G.GetTensor("vec_in1");
    G.AddTensor(inputAstDtype2, shape, MemoryType::MEM_DEVICE_DDR, "vec_in2");
    auto vec_in2 = G.GetTensor("vec_in2");
    G.AddTensor(inputAstDtype1, shape, MemoryType::MEM_L1, "copy_in1");
    auto copy_in1 = G.GetTensor("copy_in1");
    G.AddTensor(inputAstDtype2, shape, MemoryType::MEM_L1, "copy_in2");
    auto copy_in2 = G.GetTensor("copy_in2");
    G.AddTensor(inputAstDtype1, shape, MemoryType::MEM_L0A, "l0a");
    auto l0a = G.GetTensor("l0a");
    G.AddTensor(inputAstDtype2, shape, MemoryType::MEM_L0B, "l0b");
    auto l0b = G.GetTensor("l0b");
    G.AddTensor(outputAstDtype, shape, MemoryType::MEM_L0C, "l0c");
    auto l0c = G.GetTensor("l0c");
    G.AddTensor(outputAstDtype, shape, MemoryType::MEM_DEVICE_DDR, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in1"}, {"copy_in1"}, "copyin1");
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in2"}, {"copy_in2"}, "copyin2");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"copy_in1"}, {"l0a"}, "l1_to_l0a");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"copy_in2"}, {"l0b"}, "l1_to_l0b");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0a", "l0b"}, {"l0c"}, "a_mul_b");
    auto aMulb = G.GetOp("a_mul_b");
    aMulb->SetAttribute(MATMUL_NZ_ATTR, 0);
    aMulb->SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, NUM10));
    aMulb->SetAttribute(A_MUL_B_RELU_ATTR, 1);
    aMulb->SetAttribute(A_MUL_B_ACT_M, 1);
    aMulb->SetAttribute(A_MUL_B_ACT_K, 1);
    aMulb->SetAttribute(A_MUL_B_ACT_N, 1);
    G.AddOp(Opcode::OP_COPY_OUT, {"l0c"}, {"vec_out"}, "copyout");

    // set incast and outcast
    G.SetInCast({"vec_in1"});
    G.SetInCast({"vec_in2"});
    G.SetOutCast({"vec_out"});

    // before
    EXPECT_EQ(l0c->tensor->GetDataType() == outputAstDtype, true);
    // after
    Function* function = G.GetFunction();
    PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.Run(*function, "", "", 0);
    preGraph.PostCheck(*function);
    DataType outDtype = supportDtypeMap.at(std::make_pair(inputAstDtype1, inputAstDtype2));
    EXPECT_EQ(l0c->tensor->GetDataType() == outDtype, true);
}

// vec_in - CopyIn - copy_in - TransposeMoveout - vec_out
TEST_F(PreGraphTest, TestTransposeMoveout)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP32;

    std::vector<int64_t> shape = {16, 16};
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_DEVICE_DDR, "vec_in");
    auto vec_in = G.GetTensor("vec_in");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_UB, "copy_in");
    auto copy_in = G.GetTensor("copy_in");
    G.AddTensor(inputAstDtype, shape, MemoryType::MEM_DEVICE_DDR, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->offset = {8, 8};
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in"}, {"copy_in"}, "copyin");
    G.AddOp(Opcode::OP_TRANSPOSE_MOVEOUT, {"copy_in"}, {"vec_out"}, "transpose_moveout");
    auto transpose_moveout = G.GetOp("transpose_moveout");
    // set incast and outcast
    G.SetInCast({"vec_in"});
    G.SetOutCast({"vec_out"});
    // before
    EXPECT_EQ(IsCopyOut(transpose_moveout->GetOpcode()), true);
    // run
    Function* function = G.GetFunction();
    PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.Run(*function, "", "", 0);
    preGraph.PostCheck(*function);
    // after validate transpose_moveout_attr
    auto transpose_moveout_attr = dynamic_cast<CopyOpAttribute*>(transpose_moveout->GetOpAttribute().get());
    EXPECT_EQ(transpose_moveout_attr->from_, MemoryType::MEM_UB);
    EXPECT_EQ(transpose_moveout_attr->GetToOffset().size(), vec_out->GetOffset().size());
    for (size_t i = 0; i < vec_out->GetOffset().size(); i++) {
        EXPECT_EQ(transpose_moveout_attr->GetToOffset()[i].Dump(), std::to_string(vec_out->GetOffset()[i]));
    }
    EXPECT_EQ(transpose_moveout_attr->GetShape().size(), vec_out->GetShape().size());
    for (size_t i = 0; i < vec_out->GetShape().size(); i++) {
        EXPECT_EQ(transpose_moveout_attr->GetShape()[i].Dump(), std::to_string(vec_out->GetShape()[i]));
    }
    EXPECT_EQ(transpose_moveout_attr->GetRawShape().size(), vec_out->tensor->GetDynRawShape().size());
    for (size_t i = 0; i < vec_out->tensor->GetDynRawShape().size(); i++) {
        EXPECT_EQ(transpose_moveout_attr->GetRawShape()[i].Dump(), std::to_string(vec_out->tensor->GetRawShape()[i]));
    }
}

void RunSetTensorBoundary(ComputationalGraphBuilder& G)
{
    // set incast and outcast
    G.SetInCast({"vec_in"});
    G.SetOutCast({"vec_out"});
    Function* function = G.GetFunction();
    function->SetTotalSubGraphCount(NUM3);
    PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.Run(*function, "", "", 0);
    preGraph.PostCheck(*function);
    auto vec_in = G.GetTensor("vec_in");
    auto copy_out = G.GetTensor("copy_out");
    auto reshape_out = G.GetTensor("reshape_out");
    auto vec_out = G.GetTensor("vec_out");
    EXPECT_TRUE(SubgraphUtils::IsBoundary(vec_in));
    EXPECT_TRUE(SubgraphUtils::IsBoundary(copy_out));
    EXPECT_TRUE(SubgraphUtils::IsBoundary(reshape_out));
    EXPECT_TRUE(SubgraphUtils::IsBoundary(vec_out));
}

//        CopyIn[0] - copy_in1 - Exp[0] - e1 - CopyOut[0]
// vec_in                                                 copy_out - Reshape[2] - reshape_out - CopyOut[2] -vec_out
//        COPYIN[1] - copy_in2 - exp[1] - e2 - CopyOut[1]
TEST_F(PreGraphTest, TestSetTensorBoundary)
{
    ComputationalGraphBuilder G;
    // add tensor
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_DEVICE_DDR, "vec_in");
    auto vec_in = G.GetTensor("vec_in");
    G.AddTensor(DataType::DT_FP16, {32, 64}, MemoryType::MEM_UB, "copy_in1");
    auto copy_in1 = G.GetTensor("copy_in1");
    G.AddTensor(DataType::DT_FP16, {32, 64}, MemoryType::MEM_UB, "copy_in2");
    auto copy_in2 = G.GetTensor("copy_in2");
    G.AddTensor(DataType::DT_FP16, {32, 64}, MemoryType::MEM_UB, "e1");
    auto e1 = G.GetTensor("e1");
    G.AddTensor(DataType::DT_FP16, {32, 64}, MemoryType::MEM_UB, "e2");
    auto e2 = G.GetTensor("e2");
    G.AddTensor(DataType::DT_FP16, {64, 64}, MemoryType::MEM_DEVICE_DDR, "copy_out");
    auto copy_out = G.GetTensor("copy_out");
    G.AddTensor(DataType::DT_FP16, {32, 128}, MemoryType::MEM_DEVICE_DDR, "reshape_out");
    auto reshape_out = G.GetTensor("reshape_out");
    G.AddTensor(DataType::DT_FP16, {32, 128}, MemoryType::MEM_DEVICE_DDR, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    // add copyin
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in"}, {"copy_in1"}, "op_copy_in1");
    auto attrCopyIn1 = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_DEVICE_DDR,
                                                         OpImmediate::Specified(copy_in1->GetShape()),
                                                         OpImmediate::Specified(vec_in->tensor->GetRawShape()));
    G.GetOp("op_copy_in1")->SetOpAttribute(attrCopyIn1);
    G.GetOp("op_copy_in1")->UpdateSubgraphID(SUBGRAPHID0);
    G.AddOp(Opcode::OP_COPY_IN, {"vec_in"}, {"copy_in2"}, "op_copy_in2");
    auto attrCopyIn2 = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({32, 0}), MemoryType::MEM_DEVICE_DDR,
                                                         OpImmediate::Specified(copy_in2->GetShape()),
                                                         OpImmediate::Specified(vec_in->tensor->GetRawShape()));
    G.GetOp("op_copy_in2")->SetOpAttribute(attrCopyIn2);
    G.GetOp("op_copy_in2")->UpdateSubgraphID(SUBGRAPHID1);
    // add exp
    G.AddOp(Opcode::OP_EXP, {"copy_in1"}, {"e1"}, "exp1");
    G.GetOp("exp1")->UpdateSubgraphID(SUBGRAPHID0);
    G.AddOp(Opcode::OP_EXP, {"copy_in2"}, {"e2"}, "exp2");
    G.GetOp("exp2")->UpdateSubgraphID(SUBGRAPHID1);
    // add copyout
    G.AddOp(Opcode::OP_COPY_OUT, {"e1"}, {"copy_out"}, "op_copy_out1");
    auto attrCopyOut1 = std::make_shared<CopyOpAttribute>(MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({0, 0}),
                                                          OpImmediate::Specified(copy_out->GetShape()),
                                                          OpImmediate::Specified(copy_out->tensor->GetRawShape()));
    G.GetOp("op_copy_out1")->SetOpAttribute(attrCopyOut1);
    G.GetOp("op_copy_out1")->UpdateSubgraphID(SUBGRAPHID0);
    G.AddOp(Opcode::OP_COPY_OUT, {"e2"}, {"copy_out"}, "op_copy_out2");
    auto attrCopyOut2 = std::make_shared<CopyOpAttribute>(MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({32, 0}),
                                                          OpImmediate::Specified(copy_out->GetShape()),
                                                          OpImmediate::Specified(copy_out->tensor->GetRawShape()));
    G.GetOp("op_copy_out2")->SetOpAttribute(attrCopyOut2);
    G.GetOp("op_copy_out2")->UpdateSubgraphID(SUBGRAPHID1);
    // add reshape and copyout
    G.AddOp(Opcode::OP_RESHAPE, {"copy_out"}, {"reshape_out"}, "op_reshape");
    G.GetOp("op_reshape")->UpdateSubgraphID(SUBGRAPHID2);
    G.AddOp(Opcode::OP_COPY_OUT, {"reshape_out"}, {"vec_out"}, "op_copy_out3");
    G.GetOp("op_copy_out3")->UpdateSubgraphID(SUBGRAPHID2);
    RunSetTensorBoundary(G);
} // namespace tile_fwk

/*
TESTPreGraphReduceReShape
inCast1{4,8,16}->copyin->ubTensor{4,8,16}->copyout->ddrTensor1{4,8,16}->Reshape->ddrTensor2{8,64}->Assemble->outCast{16,64}
                                                                                    inCast2{8,64}->copyout
inCast1{4,8,16}->copyin->ubTensor{4,8,16}->copyout->ddrTensor1{4,8,16}->Reshape->outCast{16,64}
                                                         inCast2{8,64}->copyout->
*/
TEST_F(PreGraphTest, PreGraphReduceReShape)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {4, 8, 16}, "inCast1");
    auto inCast1 = G.GetTensor("inCast1");
    inCast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {8, 64}, "inCast2");
    auto inCast2 = G.GetTensor("inCast2");
    inCast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {4, 8, 16}, "ddrTensor1");
    auto ddrTensor1 = G.GetTensor("ddrTensor1");
    ddrTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {8, 64}, "ddrTensor2");
    auto ddrTensor2 = G.GetTensor("ddrTensor2");
    ddrTensor2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {16, 64}, "outCast");
    auto outCast = G.GetTensor("outCast");
    outCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {4, 8, 16}, "ubTensor");
    auto ubTensor = G.GetTensor("ubTensor");
    ubTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    ddrTensor1->tensor->UpdateRawShape({4, 8, 16});
    ddrTensor2->tensor->UpdateRawShape({8, 64});
    outCast->tensor->UpdateRawShape({16, 64});
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"inCast1"}, {"ubTensor"}, "COPYIN");
    G.AddOp(Opcode::OP_COPY_OUT, {"ubTensor"}, {"ddrTensor1"}, "COPYOUT1");
    G.AddOp(Opcode::OP_RESHAPE, {"ddrTensor1"}, {"ddrTensor2"}, "RESHAPE");
    G.AddOp(Opcode::OP_ASSEMBLE, {"ddrTensor2"}, {"outCast"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0})));
    assemble->SetOpAttribute(assembleAttr);
    G.AddOp(Opcode::OP_COPY_OUT, {"inCast2"}, {"outCast"}, "COPYOUT2");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({0, 64}),
                                                         OpImmediate::Specified(outCast->GetShape()),
                                                         OpImmediate::Specified(outCast->tensor->GetRawShape()));
    G.GetOp("COPYOUT2")->SetOpAttribute(attrCopyOut);
    // set incast and outcast
    G.SetInCast({"inCast1"});
    G.SetInCast({"inCast2"});
    G.SetOutCast({"outCast"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(ddrTensor1->tensor->GetRawShapeSize(), outCast->tensor->GetRawShapeSize());
    EXPECT_EQ(function->Operations().size(), NUM4);
}

/*
PreGraphReduceExpand
inCast1{8,16}->copyin->ubTensor{8,16}->copyout->ddrTensor1{8,16}->Reshape->ddrTensor2{8,2,8}->Assemble->outCast{8,2,16}
                                                                                inCast2{8,2,8}->copyout
inCast1{8,16}->copyin->ubTensor{8,16}->copyout->ddrTensor1{8,16}->Reshape->ddrTensor2{8,2,8}->outCast{8,2,16}
                                                                     inCast2{8,2,8}->copyout
*/
TEST_F(PreGraphTest, PreGraphReduceExpand)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {8, 16}, "inCast1");
    auto inCast1 = G.GetTensor("inCast1");
    inCast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {8, 2, 8}, "inCast2");
    auto inCast2 = G.GetTensor("inCast2");
    inCast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {8, 16}, "ddrTensor1");
    auto ddrTensor1 = G.GetTensor("ddrTensor1");
    ddrTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {8, 2, 8}, "ddrTensor2");
    auto ddrTensor2 = G.GetTensor("ddrTensor2");
    ddrTensor2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {8, 2, 16}, "outCast");
    auto outCast = G.GetTensor("outCast");
    outCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {8, 16}, "ubTensor");
    auto ubTensor = G.GetTensor("ubTensor");
    ubTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    ddrTensor1->tensor->UpdateRawShape({8, 16});
    ddrTensor2->tensor->UpdateRawShape({8, 2, 8});
    outCast->tensor->UpdateRawShape({8, 2, 16});
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"inCast1"}, {"ubTensor"}, "COPYIN");
    G.AddOp(Opcode::OP_COPY_OUT, {"ubTensor"}, {"ddrTensor1"}, "COPYOUT1");
    G.AddOp(Opcode::OP_RESHAPE, {"ddrTensor1"}, {"ddrTensor2"}, "RESHAPE");
    G.AddOp(Opcode::OP_COPY_OUT, {"inCast2"}, {"outCast"}, "COPYOUT2");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({0, 0, 8}),
                                                         OpImmediate::Specified(outCast->GetShape()),
                                                         OpImmediate::Specified(outCast->tensor->GetRawShape()));
    G.GetOp("COPYOUT2")->SetOpAttribute(attrCopyOut);
    G.AddOp(Opcode::OP_ASSEMBLE, {"ddrTensor2"}, {"outCast"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    // set incast and outcast
    G.SetInCast({"inCast1"});
    G.SetInCast({"inCast2"});
    G.SetOutCast({"outCast"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(ddrTensor1->tensor->GetRawShapeSize(), outCast->tensor->GetRawShapeSize());
    EXPECT_EQ(function->Operations().size(), NUM4);
}

/*
PreGraphSingleAssembleMutiConsumerExpand
inCast{16,16}->copyin->ubTensor{16,16}->copyout->ddrTensor1{16,16}->Reshape->ddrTensor2{8,2,16}->Assemble->outCast{8,2,32}
                                                                                                ->copyout
inCast{16,16}->copyin->ubTensor{16,16}->copyout->ddrTensor1{16,16}->Reshape->ddrTensor2{8,2,16}->Assemble->outCast{8,2,32}
                                                                                                ->copyout
*/
TEST_F(PreGraphTest, PreGraphMutiConsumerExpand)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {16, 16}, "inCast");
    auto inCast = G.GetTensor("inCast");
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {16, 16}, "ubTensor");
    auto ubTensor = G.GetTensor("ubTensor");
    ubTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(inputAstDtype, {16, 16}, "ddrTensor1");
    auto ddrTensor1 = G.GetTensor("ddrTensor1");
    ddrTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {8, 2, 16}, "ddrTensor2");
    auto ddrTensor2 = G.GetTensor("ddrTensor2");
    ddrTensor2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {8, 2, 32}, "outCast");
    auto outCast = G.GetTensor("outCast");
    outCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    ddrTensor1->tensor->UpdateRawShape({16, 16});
    ddrTensor2->tensor->UpdateRawShape({8, 2, 32});
    outCast->tensor->UpdateRawShape({8, 2, 32});
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"inCast"}, {"ubTensor"}, "COPYIN");
    G.AddOp(Opcode::OP_COPY_OUT, {"ubTensor"}, {"ddrTensor1"}, "COPYOUT1");
    G.AddOp(Opcode::OP_RESHAPE, {"ddrTensor1"}, {"ddrTensor2"}, "RESHAPE");
    G.AddOp(Opcode::OP_COPY_OUT, {"ddrTensor2"}, {"outCast"}, "COPYOUT2");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({0, 0, 16}),
                                                         OpImmediate::Specified(outCast->GetShape()),
                                                         OpImmediate::Specified(outCast->tensor->GetRawShape()));
    G.GetOp("COPYOUT2")->SetOpAttribute(attrCopyOut);
    G.AddOp(Opcode::OP_ASSEMBLE, {"ddrTensor2"}, {"outCast"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    // set incast and outcast
    G.SetInCast({"inCast"});
    G.SetOutCast({"outCast"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(ddrTensor1->tensor->GetRawShapeSize(), outCast->tensor->GetRawShapeSize());
    EXPECT_EQ(function->Operations().size(), NUM5);
}

/*
PreGraphMutiConsumerReduce
inCast{4,8,16}->copyin->ubTensor{4,8,16}->copyout->ddrTensor1{4,8,16}->Reshape->ddrTensor2{32,16}->Assemble->outCast{32,32}
                                                                                                 ->copyout
inCast{4,8,16}->copyin->ubTensor{4,8,16}->copyout->ddrTensor1{4,8,16}->Reshape->ddrTensor2{32,16}->Assemble->outCast{32,32}
                                                                                                 ->copyout
*/
TEST_F(PreGraphTest, PreGraphMutiConsumerReduce)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {4, 8, 16}, "inCast");
    auto inCast = G.GetTensor("inCast");
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {4, 8, 16}, "ubTensor");
    auto ubTensor = G.GetTensor("ubTensor");
    ubTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(inputAstDtype, {4, 8, 16}, "ddrTensor1");
    auto ddrTensor1 = G.GetTensor("ddrTensor1");
    ddrTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {32, 16}, "ddrTensor2");
    auto ddrTensor2 = G.GetTensor("ddrTensor2");
    ddrTensor2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {32, 32}, "outCast");
    auto outCast = G.GetTensor("outCast");
    outCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    ddrTensor1->tensor->UpdateRawShape({4, 8, 16});
    ddrTensor2->tensor->UpdateRawShape({32, 16});
    outCast->tensor->UpdateRawShape({32, 32});
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"inCast"}, {"ubTensor"}, "COPYIN");
    G.AddOp(Opcode::OP_COPY_OUT, {"ubTensor"}, {"ddrTensor1"}, "COPYOUT1");
    G.AddOp(Opcode::OP_RESHAPE, {"ddrTensor1"}, {"ddrTensor2"}, "RESHAPE");
    G.AddOp(Opcode::OP_COPY_OUT, {"ddrTensor2"}, {"outCast"}, "COPYOUT2");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({0, 16}),
                                                         OpImmediate::Specified(outCast->GetShape()),
                                                         OpImmediate::Specified(outCast->tensor->GetRawShape()));
    G.GetOp("COPYOUT2")->SetOpAttribute(attrCopyOut);
    G.AddOp(Opcode::OP_ASSEMBLE, {"ddrTensor2"}, {"outCast"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    // set incast and outcast
    G.SetInCast({"inCast"});
    G.SetOutCast({"outCast"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(ddrTensor1->tensor->GetRawShapeSize(), outCast->tensor->GetRawShapeSize());
    EXPECT_EQ(function->Operations().size(), NUM5);
}

/*
MutiConsumerDeleteSingleAssemble
inCast{4,8,16}->copyin->ubTensor{4,8,16}->copyout->ddrTensor1{4,8,16}->Reshape->ddrTensor2{32,16}->Assemble->outCast1{32,32}
                                                                                                 ->copyout
->outCast2{32,16}
inCast{4,8,16}->copyin->ubTensor{4,8,16}->copyout->ddrTensor1{4,8,16}->Reshape->ddrTensor2{32,16}->Assemble->outCast1{32,32}
                                                                                                 ->copyout
->outCast2{32,16}
*/
TEST_F(PreGraphTest, MutiConsumerDeleteSingleAssemble)
{
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {4, 8, 16}, "inCast");
    auto inCast = G.GetTensor("inCast");
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {4, 8, 16}, "ubTensor");
    auto ubTensor = G.GetTensor("ubTensor");
    ubTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(inputAstDtype, {4, 8, 16}, "ddrTensor_1");
    auto ddrTensor_1 = G.GetTensor("ddrTensor_1");
    ddrTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {32, 16}, "ddrTensor_2");
    auto ddrTensor_2 = G.GetTensor("ddrTensor_2");
    ddrTensor_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {32, 32}, "outCast_1");
    auto outCast_1 = G.GetTensor("outCast_1");
    outCast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {32, 16}, "outCast_2");
    auto outCast_2 = G.GetTensor("outCast_2");
    outCast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    ddrTensor_1->tensor->UpdateRawShape({4, 8, 16});
    ddrTensor_2->tensor->UpdateRawShape({32, 16});
    outCast_1->tensor->UpdateRawShape({32, 32});
    outCast_2->tensor->UpdateRawShape({32, 16});
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"inCast"}, {"ubTensor"}, "COPYIN");
    G.AddOp(Opcode::OP_COPY_OUT, {"ubTensor"}, {"ddrTensor_1"}, "COPYOUT_1");
    G.AddOp(Opcode::OP_RESHAPE, {"ddrTensor_1"}, {"ddrTensor_2"}, "RESHAPE");
    G.AddOp(Opcode::OP_COPY_OUT, {"ddrTensor_2"}, {"outCast_2"}, "COPYOUT_2");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified({0, 0}),
                                                         OpImmediate::Specified(outCast_2->GetShape()),
                                                         OpImmediate::Specified(outCast_2->tensor->GetRawShape()));
    G.GetOp("COPYOUT_2")->SetOpAttribute(attrCopyOut);
    G.AddOp(Opcode::OP_ASSEMBLE, {"ddrTensor_2"}, {"outCast_1"}, "ASSEMBLE");
    auto assemble = G.GetOp("ASSEMBLE");
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0},
        OpImmediate::ToSpecified(OpImmediate::Specified(std::vector<int64_t>{0, 0})));
    assemble->SetOpAttribute(assembleAttr);

    // set incast and outcast
    G.SetInCast({"inCast"});
    G.SetOutCast({"outCast_1"});
    G.SetOutCast({"outCast_2"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess preGraph;
    auto operationSize = function->Operations().size();
    preGraph.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(function->Operations().size(), operationSize - 1);
}

/*
maybeCycle
incast -> COPYIN -> COPYOUT -> ASSEMBLE ->         midOut -> COPYIN -> COPYOUT -> outcast
                            -> COPYIN -> COPYOUT -/
不删除ASSEMBLE
*/
TEST_F(PreGraphTest, maybeCycle)
{
    ComputationalGraphBuilder G;
    DataType dataType = DataType::DT_FP16;
    // add tensor
    G.AddTensor(dataType, {4, 4}, "inCast");
    auto inCast = G.GetTensor("inCast");
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(dataType, {4, 4}, "copyInOut1");
    auto copyInOut1 = G.GetTensor("copyInOut1");
    copyInOut1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(dataType, {4, 4}, "copyOutOut");
    auto copyOutOut = G.GetTensor("copyOutOut");
    copyOutOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(dataType, {4, 8}, "assembleOut");
    auto assembleOut = G.GetTensor("assembleOut");
    assembleOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(dataType, {4, 4}, "copyInOut2");
    auto copyInOut2 = G.GetTensor("copyInOut2");
    copyInOut2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(dataType, {4, 8}, "copyInOut3");
    auto copyInOut3 = G.GetTensor("copyInOut3");
    copyInOut3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddTensor(dataType, {4, 8}, "outCast");
    auto outCast = G.GetTensor("outCast");
    outCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"inCast"}, {"copyInOut1"}, "COPYIN1");
    G.AddOp(Opcode::OP_COPY_OUT, {"copyInOut1"}, {"copyOutOut"}, "COPYOUT1");
    G.AddOp(Opcode::OP_ASSEMBLE, {"copyOutOut"}, {"assembleOut"}, "ASSEMBLE");
    G.AddOp(Opcode::OP_COPY_IN, {"copyOutOut"}, {"copyInOut2"}, "COPYIN2");
    G.AddOp(Opcode::OP_COPY_OUT, {"copyInOut2"}, {"assembleOut"}, "COPYOUT2");
    G.AddOp(Opcode::OP_COPY_IN, {"assembleOut"}, {"copyInOut3"}, "COPYIN3");
    G.AddOp(Opcode::OP_COPY_OUT, {"copyInOut3"}, {"outCast"}, "COPYOUT3");

    // set incast and outcast
    G.SetInCast({"inCast"});
    G.SetOutCast({"outCast"});
    // run pass
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess preGraph;
    auto opSize = function->Operations().size();
    preGraph.Run(*function, "", "", 0);
    // check after pass
    EXPECT_EQ(function->Operations().size(), opSize);
}
} // namespace tile_fwk
} // namespace npu
#undef private
