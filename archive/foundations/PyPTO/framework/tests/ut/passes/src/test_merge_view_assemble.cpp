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
 * \file test_merge_view_assemble.cpp
 * \brief Unit test for merge_view_assemble pass.
 */

#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_optimization/merge_view_assemble.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/tile_graph_pass/graph_optimization/process_atomic.h"
#include <fstream>
#include <vector>
#include <string>
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
constexpr int NUM5 = 5;
constexpr int NUM6 = 6;

class MergeViewAssembleTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ViewAssembleTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

namespace {
void BuildLargeSharedTensorMergeableChains(ComputationalGraphBuilder& graph, int assembleChainCount, int viewChainCount)
{
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    inputs.reserve(assembleChainCount);
    outputs.reserve(viewChainCount);
    std::vector<int64_t> shape{16, 16};
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, shape, "shared"));
    for (int index = 0; index < assembleChainCount; ++index) {
        std::string id = std::to_string(index);
        std::string input = "assemble_input_" + id;
        std::string producerOut = "assemble_producer_out_" + id;
        std::string mid = "assemble_mid_" + id;
        std::string producerName = "assemble_producer_" + id;
        std::string firstAssembleName = "assemble_first_" + id;
        std::string secondAssembleName = "assemble_second_" + id;
        ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, shape, {input, producerOut, mid}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {input}, {producerOut}, producerName, true));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ASSEMBLE, {producerOut}, {mid}, firstAssembleName, true));
        graph.GetOp(firstAssembleName)
            ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ASSEMBLE, {mid}, {"shared"}, secondAssembleName, true));
        graph.GetOp(secondAssembleName)
            ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}));
        inputs.emplace_back(std::move(input));
    }
    for (int index = 0; index < viewChainCount; ++index) {
        std::string id = std::to_string(index);
        std::string mid = "view_mid_" + id;
        std::string output = "view_output_" + id;
        std::string firstViewName = "view_first_" + id;
        std::string secondViewName = "view_second_" + id;
        ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, shape, {mid, output}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_VIEW, {"shared"}, {mid}, firstViewName, true));
        graph.GetOp(firstViewName)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_VIEW, {mid}, {output}, secondViewName, true));
        graph.GetOp(secondViewName)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 0}));
        outputs.emplace_back(std::move(output));
    }
    ASSERT_TRUE(graph.SetInCast(inputs));
    ASSERT_TRUE(graph.SetOutCast(outputs));
}
} // namespace

static void ExpectTopologicalOrder(Function* function)
{
    auto ops = function->Operations().DuplicatedOpList();
    std::unordered_map<Operation*, size_t> opToIndex;
    opToIndex.reserve(ops.size());
    for (size_t index = 0; index < ops.size(); ++index) {
        opToIndex.emplace(ops[index], index);
    }
    for (auto* op : ops) {
        for (const auto& input : op->GetIOperands()) {
            for (auto* producer : input->GetProducers()) {
                if (producer->BelongTo() == function && producer != op) {
                    EXPECT_LT(opToIndex.at(producer), opToIndex.at(op));
                }
            }
        }
        for (const auto& depend : op->GetDependOperands()) {
            for (auto* producer : depend->GetProducers()) {
                if (producer->BelongTo() == function && producer != op) {
                    EXPECT_LT(opToIndex.at(producer), opToIndex.at(op));
                }
            }
        }
    }
}

TEST_F(MergeViewAssembleTest, MergeViewAssembleKeepsTopologicalOrderForSharedTensorFanout)
{
    constexpr int producerNum = 4;
    constexpr int consumerNum = 5;
    std::vector<int64_t> shape{16, 16};
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, shape, {"input", "shared"}));
    for (int index = 0; index < consumerNum; ++index) {
        auto indexStr = std::to_string(index);
        ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, shape, {"view_mid_" + indexStr, "view_out_" + indexStr}));
        ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"shared"}, {"view_mid_" + indexStr}, "view_mid_" + indexStr, true));
        G.GetOp("view_mid_" + indexStr)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        ASSERT_TRUE(
            G.AddOp(Opcode::OP_VIEW, {"view_mid_" + indexStr}, {"view_out_" + indexStr}, "view_out_" + indexStr, true));
        G.GetOp("view_out_" + indexStr)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    }
    for (int index = 0; index < producerNum; ++index) {
        auto indexStr = std::to_string(index);
        ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "producer_out_" + indexStr));
        ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"input"}, {"producer_out_" + indexStr}, "producer_" + indexStr, true));
        ASSERT_TRUE(
            G.AddOp(Opcode::OP_ASSEMBLE, {"producer_out_" + indexStr}, {"shared"}, "assemble_" + indexStr, true));
        G.GetOp("assemble_" + indexStr)
            ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    }
    std::vector<std::string> outcasts;
    for (int index = 0; index < consumerNum; ++index) {
        outcasts.emplace_back("view_out_" + std::to_string(index));
    }
    ASSERT_TRUE(G.SetInCast({"input"}));
    ASSERT_TRUE(G.SetOutCast(outcasts));

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*G.GetFunction()), SUCCESS);

    ExpectTopologicalOrder(G.GetFunction());
}

TEST_F(MergeViewAssembleTest, TestMergeViewAssemble)
{
    constexpr int32_t tilex = 8;
    constexpr int32_t tiley = 16;
    constexpr int expectedOps = 8;
    constexpr int expected_view = 0;
    constexpr int expected_slice = 4;
    constexpr int expected_add = 2;
    constexpr int expected_assemble = 0;
    constexpr int expected_contract = 2;
    std::vector<int64_t> shape{16, 16};
    Tensor a(DT_FP32, shape, "a");
    Tensor in_tensor(DT_FP32, shape, "in_tensor");
    Tensor out_tensor(DT_FP32, shape, "out_tensor");

    TileShape::Current().SetVecTile(tilex, tiley);

    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ViewAssembleTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                 });

    Function* originFunction = nullptr;
    std::vector<int64_t> originOpmagic;
    FUNCTION("AddFunction")
    {
        out_tensor = Add(in_tensor, a);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "Current function pointer is null";
        auto operations = originFunction->Operations();
        for (const auto& op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/merge_view_assemble.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_AddFunction");

    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currentFunction);
    mergeViewAssemble.RunOnFunction(*currentFunction);
    mergeViewAssemble.PostCheck(*currentFunction);

    // ================== Verify Pass Effect ==================
    auto updated_operations = currentFunction->Operations();
    EXPECT_EQ(updated_operations.size(), expectedOps) << "14 operations should remain";
    int view_count = 0;
    int slice_count = 0;
    int add_count = 0;
    int assemble_count = 0;
    int contract_count = 0;
    for (const auto& op : updated_operations) {
        if (op.GetOpcodeStr() == "VIEW") {
            view_count++;
        } else if (op.GetOpcodeStr() == "ADD") {
            add_count++;
        } else if (op.GetOpcodeStr() == "SLICE") {
            slice_count++;
        } else if (op.GetOpcodeStr() == "ASSEMBLE") {
            assemble_count++;
        } else if (op.GetOpcodeStr() == "CONTRACT") {
            contract_count++;
        }
    }

    EXPECT_EQ(view_count, expected_view) << "0 VIEW operations should remain";
    EXPECT_EQ(slice_count, expected_slice) << "4 SLICE operations should remain";
    EXPECT_EQ(add_count, expected_add) << "2 ADD operations should remain";
    EXPECT_EQ(assemble_count, expected_assemble) << "0 ASSEMBLE operation should remain";
    EXPECT_EQ(contract_count, expected_contract) << "2 CONTRACT operation should remain";

    // Check the offset of the View operation
}

TEST_F(MergeViewAssembleTest, MergeTwoConsecutiveViews)
{
    Program program;
    std::string funcMagicName = "test_function";
    std::string funcRawName = "test_function_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);
    // 创建原始输入tensor
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);
    // 创建第一个VIEW操作，偏移量[1,2]
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{8, 8},
                                                                CreateTestConstIntVector(std::vector<int64_t>{8, 8}));
    auto view1Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2},    // from_offset
                                                       std::vector<SymbolicScalar>{}, // from_dyn_offset
                                                       std::vector<SymbolicScalar>{}  // to_dyn_valid_shape
    );
    auto& view1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {inputTensor}, {midTensor});
    view1Op.SetOpAttribute(view1Attr);

    // 2. 创建第2个VIEW操作，偏移量[3,4]
    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 6}, CreateTestConstIntVector(std::vector<int64_t>{6, 6}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto view2Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{3, 4},    // from_offset
                                                       std::vector<SymbolicScalar>{}, // from_dyn_offset
                                                       std::vector<SymbolicScalar>{}  // to_dyn_valid_shape
    );
    auto& view2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {midTensor}, {outputTensor});
    view2Op.SetOpAttribute(view2Attr);

    // 3. 执行MergeViewAssemble pass
    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // 4. 验证结果
    // 4.1 检查原始VIEW操作是否被标记为删除

    const auto& operations = function->Operations();
    EXPECT_FALSE(operations.Contains(view1Op));
    EXPECT_FALSE(operations.Contains(view2Op));

    // 4.2 检查合并后的VIEW操作
    int viewOpCount = 0;
    Operation* mergedViewOp = nullptr;
    for (auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW && !op.IsDeleted()) {
            viewOpCount++;
            mergedViewOp = &op;
        }
    }

    ASSERT_EQ(viewOpCount, 1) << "只有一个合并后的VIEW操作";
    ASSERT_NE(mergedViewOp, nullptr);

    // 4.3 检查合并后的偏移量是否正确
    auto mergedAttr = dynamic_cast<ViewOpAttribute*>(mergedViewOp->GetOpAttribute().get());
    ASSERT_NE(mergedAttr, nullptr);

    const auto& mergedOffset = mergedAttr->GetFromOffset();
    ASSERT_EQ(mergedOffset.size(), 2);
    EXPECT_EQ(mergedOffset[0], 4) << "第一个维度偏移量应为4";
    EXPECT_EQ(mergedOffset[1], 6) << "第二个维度偏移量应为6";

    // 4.4 检查输入输出tensor是否正确
    ASSERT_EQ(mergedViewOp->GetIOperands().size(), 1);
    ASSERT_EQ(mergedViewOp->GetOOperands().size(), 1);
    EXPECT_EQ(mergedViewOp->GetInputOperand(0), inputTensor);
    EXPECT_EQ(mergedViewOp->GetOutputOperand(0), outputTensor);

    // 4.5 检查中间tensor是否被清理
    bool midTensorExists = false;
    for (const auto& item : GraphUtils::GetAllTensors(*function)) {
        if (item == midTensor) {
            midTensorExists = true;
            break;
        }
    }
    EXPECT_FALSE(midTensorExists) << "中间tensor应该被清理";
}

TEST_F(MergeViewAssembleTest, MergeThreeConsecutiveAssembles)
{
    Program program;
    std::string funcMagicName = "test_function";
    std::string funcRawName = "test_function_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);

    // 1.创建原始输入tensor并设置incast
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);

    // 2.创建三个连续的ASSEMBLE操作
    // 第一个ASSEMBLE：偏移量[1,0]
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{9, 10},
                                                                 CreateTestConstIntVector(std::vector<int64_t>{9, 10}));
    auto assemble1Attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0},   // to_offset
                                                               std::vector<SymbolicScalar>{} // to_dyn_offset
    );
    auto& assemble1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {inputTensor}, {midTensor1});
    assemble1Op.SetOpAttribute(assemble1Attr);

    // 第二个ASSEMBLE：偏移量[0,2]
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{9, 8},
                                                                 CreateTestConstIntVector(std::vector<int64_t>{9, 8}));
    auto assemble2Attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2},   // to_offset
                                                               std::vector<SymbolicScalar>{} // to_dyn_offset
    );
    auto& assemble2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {midTensor1}, {midTensor2});
    assemble2Op.SetOpAttribute(assemble2Attr);

    // 第三个ASSEMBLE：偏移量[3,0]
    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 8}, CreateTestConstIntVector(std::vector<int64_t>{6, 8}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto assemble3Attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{3, 0},   // to_offset
                                                               std::vector<SymbolicScalar>{} // to_dyn_offset
    );
    auto& assemble3Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {midTensor2}, {outputTensor});
    assemble3Op.SetOpAttribute(assemble3Attr);

    // 3.执行MergeViewAssemble pass
    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // 4.验证结果
    // 4.1检查原始ASSEMBLE操作是否被标记为已删除
    const auto& operations = function->Operations();
    EXPECT_FALSE(operations.Contains(assemble1Op));
    EXPECT_FALSE(operations.Contains(assemble2Op));
    EXPECT_FALSE(operations.Contains(assemble3Op));

    // 4.2检查合并后的ASSEMBLE操作
    ASSERT_EQ(operations.size(), 0) << "所有op都应该被删除";

    // 4.3检查中间tensor是否被清理
    bool midTensor1Exists = false;
    bool midTensor2Exists = false;
    for (const auto& item : GraphUtils::GetAllTensors(*function)) {
        if (item == midTensor1) {
            midTensor1Exists = true;
        }
        if (item == midTensor2) {
            midTensor2Exists = true;
        }
    }
    EXPECT_FALSE(midTensor1Exists) << "中间tensor1应该被清理";
    EXPECT_FALSE(midTensor2Exists) << "中间tensor2应该被清理";
}

TEST_F(MergeViewAssembleTest, ViewAssembleChainShouldNotMerge)
{
    // 创建计算图构建器
    ComputationalGraphBuilder G;

    // 创建测试tensor
    std::vector<std::string> tensorNames = {"input", "view1_out", "assemble_out", "view2_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));

    // 创建op链: VIEW1 -> ASSEMBLE -> VIEW2
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"input"}, {"view1_out"}, {"assemble_out"}};
    std::vector<std::vector<std::string>> ooperands{{"view1_out"}, {"assemble_out"}, {"view2_out"}};
    std::vector<std::string> opNames{"VIEW1", "ASSEMBLE", "VIEW2"};

    // 添加op
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 设置输入输出
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"view2_out"}));

    // 获取Function并验证
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    // 为VIEW设置必要的属性
    size_t op_index = 0;
    for (auto& op : function->Operations()) {
        switch (op_index) {
            case 0: {                                                                        // VIEW1
                auto attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0},    // offset
                                                              MemoryType::MEM_UNKNOWN,       // toType
                                                              std::vector<SymbolicScalar>{}, // dynOffset
                                                              std::vector<SymbolicScalar>{}  // dynValidShape
                );
                op.SetOpAttribute(attr);
                break;
            }
            case 2: {                                                                        // VIEW2
                auto attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0},    // offset
                                                              MemoryType::MEM_UNKNOWN,       // toType
                                                              std::vector<SymbolicScalar>{}, // dynOffset
                                                              std::vector<SymbolicScalar>{}  // dynValidShape
                );
                op.SetOpAttribute(attr);
                break;
            }
            default:
                break;
        }
        op_index++;
    }

    // 记录原始op数量
    const size_t originalOpCount = function->Operations().size();
    ASSERT_EQ(originalOpCount, 3);

    // 执行pass
    MergeViewAssemble mva;
    Status status = mva.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // 验证op数量不变
    EXPECT_EQ(function->Operations().size(), originalOpCount);

    // 验证op顺序和类型保持不变
    const auto& ops = function->Operations();
    EXPECT_EQ(ops[0].GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(ops[1].GetOpcode(), Opcode::OP_ASSEMBLE);
    EXPECT_EQ(ops[2].GetOpcode(), Opcode::OP_VIEW);
}

TEST_F(MergeViewAssembleTest, Test2View2Assemble2View2AssembleChain)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames{"input",     "view1_out", "view2_out",     "assemble1_out", "assemble2_out",
                                         "view3_out", "view4_out", "assemble3_out", "assemble4_out", "final_out"};

    std::vector<Opcode> opCodes{Opcode::OP_VIEW,     Opcode::OP_VIEW,     Opcode::OP_ASSEMBLE,
                                Opcode::OP_ASSEMBLE, Opcode::OP_VIEW,     Opcode::OP_VIEW,
                                Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};

    std::vector<std::vector<std::string>> ioperands{{"input"},         {"view1_out"},     {"view2_out"},
                                                    {"assemble1_out"}, {"assemble2_out"}, {"view3_out"},
                                                    {"view4_out"},     {"assemble3_out"}, {"assemble4_out"}};

    std::vector<std::vector<std::string>> ooperands{{"view1_out"},     {"view2_out"},     {"assemble1_out"},
                                                    {"assemble2_out"}, {"view3_out"},     {"view4_out"},
                                                    {"assemble3_out"}, {"assemble4_out"}, {"final_out"}};

    std::vector<std::string> opNames{"view1", "view2",     "assemble1", "assemble2", "view3",
                                     "view4", "assemble3", "assemble4", "abs"};

    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10, 10}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_EQ(G.SetInCast({"input"}), true);
    EXPECT_EQ(G.SetOutCast({"final_out"}), true);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    size_t op_index = 0;
    for (auto& op : function->Operations()) {
        switch (op_index) {
            case 0: { // view1
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{1, 1, 1}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 1: { // view2
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{2, 2, 2}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 2: { // assemble1
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{3, 3, 3},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 3: { // assemble2
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{4, 4, 4},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 4: { // view3
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{5, 5, 5}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 5: { // view4
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{6, 6, 6}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 6: { // assemble3
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{7, 7, 7},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 7: { // assemble4
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{8, 8, 8},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            default:
                // No attribute needed for other operations (e.g. ABS)
                break;
        }
        op_index++;
    }

    MergeViewAssemble pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    const auto& operations = function->Operations();
    int view_count = 0;
    int assemble_count = 0;

    for (const auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count++;
        }
    }

    // Should have 2 merged views and 2 merged assembles
    EXPECT_EQ(view_count, 2);
    EXPECT_EQ(assemble_count, 2);

    // Verify final graph structure
    bool found_structure = false;
    for (const auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_ABS) {
            const auto* abs_input = op.GetIOperands()[0].get();
            ASSERT_NE(abs_input, nullptr);

            const auto* last_assemble = *abs_input->GetProducers().begin();
            ASSERT_NE(last_assemble, nullptr);
            EXPECT_EQ(last_assemble->GetOpcode(), Opcode::OP_ASSEMBLE);

            const auto* last_view = *last_assemble->GetIOperands()[0]->GetProducers().begin();
            ASSERT_NE(last_view, nullptr);
            EXPECT_EQ(last_view->GetOpcode(), Opcode::OP_VIEW);

            const auto* first_assemble = *last_view->GetIOperands()[0]->GetProducers().begin();
            ASSERT_NE(first_assemble, nullptr);
            EXPECT_EQ(first_assemble->GetOpcode(), Opcode::OP_ASSEMBLE);

            const auto* first_view = *first_assemble->GetIOperands()[0]->GetProducers().begin();
            ASSERT_NE(first_view, nullptr);
            EXPECT_EQ(first_view->GetOpcode(), Opcode::OP_VIEW);

            found_structure = true;
            break;
        }
    }
    EXPECT_TRUE(found_structure);
}

TEST_F(MergeViewAssembleTest, TestMixedBranchWithViewAndAssemble)
{
    ComputationalGraphBuilder G;

    // 定义张量 (包含主分支和两个不同类型的子分支)
    std::vector<std::string> tensorNames{"input",
                                         // 主分支
                                         "view1_out", "view2_out", "assemble1_out",
                                         // 分支1 (VIEW分支)
                                         "view3_out", "view4_out", "assemble2_out",
                                         // 分支2 (Assemble分支)
                                         "view5_out", "assemble3_out", "assemble4_out",
                                         // 合并输出
                                         "merged_out", "final_out"};

    // 操作序列 (包含混合类型分支)
    std::vector<Opcode> opCodes{
        // 主分支
        Opcode::OP_VIEW,     // view1
        Opcode::OP_VIEW,     // view2
        Opcode::OP_ASSEMBLE, // assemble1
        // 分支1 (VIEW分支)
        Opcode::OP_VIEW,     // view3
        Opcode::OP_VIEW,     // view4
        Opcode::OP_ASSEMBLE, // assemble2
        // 分支2 (Assemble分支)
        Opcode::OP_ASSEMBLE, // assemble3
        Opcode::OP_VIEW,     // view5
        Opcode::OP_ASSEMBLE, // assemble4
        // 合并
        Opcode::OP_ASSEMBLE, // merge_assemble
        Opcode::OP_ABS       // final
    };

    // 定义输入输出关系
    std::vector<std::vector<std::string>> ioperands{
        {"input"},                          // view1
        {"view1_out"},                      // view2
        {"view2_out"},                      // assemble1
        {"assemble1_out"},                  // view3 (分支1)
        {"view3_out"},                      // view4 (分支1)
        {"view4_out"},                      // assemble2 (分支1)
        {"assemble1_out"},                  // assemble3 (分支2)
        {"assemble3_out"},                  // view5 (分支2)
        {"view5_out"},                      // assemble4 (分支2)
        {"assemble2_out", "assemble4_out"}, // merge_assemble
        {"merged_out"}                      // abs
    };

    std::vector<std::vector<std::string>> ooperands{
        {"view1_out"},     {"view2_out"}, {"assemble1_out"}, {"view3_out"},  {"view4_out"}, {"assemble2_out"},
        {"assemble3_out"}, {"view5_out"}, {"assemble4_out"}, {"merged_out"}, {"final_out"}};

    // 添加操作名称列表
    std::vector<std::string> opNames{
        "view1",          "view2",    "assemble1", // 主分支
        "view3",          "view4",    "assemble2", // 分支1 (VIEW分支)
        "assemble3",      "view5",    "assemble4", // 分支2 (Assemble分支)
        "merge_assemble", "abs_final"              // 合并与最终操作
    };

    // 构建计算图
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {20, 20, 20}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 辅助函数：安全设置属性
    auto set_attr = [&G](const std::string& op_name, auto attr) {
        auto* op = G.GetOp(op_name);
        ASSERT_NE(op, nullptr) << "Operation " << op_name << " not found!";
        op->SetOpAttribute(attr);
    };

    // ------------------------- 主分支属性设置 -------------------------
    // view1: offset=[1,0,0]
    set_attr("view1", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 0, 0}, // offset
                                                        std::vector<SymbolicScalar>{}, // stride (空表示默认)
                                                        std::vector<SymbolicScalar>{} // shape (空表示保持输入形状)
                                                        ));

    // view2: offset=[0,2,0]
    set_attr("view2", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 2, 0}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // assemble1: offset=[1,2,0]
    set_attr("assemble1", std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 2, 0}, // offset
                                                                std::vector<SymbolicScalar>{} // 其他参数（如无则空）
                                                                ));

    // ------------------------- 分支1 (VIEW分支) 属性设置 -------------------------
    // view3: offset=[0,0,3]
    set_attr("view3", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0, 3}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // view4: offset=[4,0,0]
    set_attr("view4", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{4, 0, 0}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // assemble2: offset=[4,0,3]
    set_attr("assemble2",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{4, 0, 3}, std::vector<SymbolicScalar>{}));

    // ------------------------- 分支2 (Assemble分支) 属性设置 -------------------------
    // assemble3: offset=[5,0,0]
    set_attr("assemble3",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{5, 0, 0}, std::vector<SymbolicScalar>{}));

    // view5: offset=[0,6,0]
    set_attr("view5", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 6, 0}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // assemble4: offset=[5,6,0]
    set_attr("assemble4",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{5, 6, 0}, std::vector<SymbolicScalar>{}));

    // ------------------------- 合并操作属性设置 -------------------------
    // merge_assemble: offset=[9,9,9]
    set_attr("merge_assemble",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{9, 9, 9}, std::vector<SymbolicScalar>{}));

    // ABS操作无需属性
    // ------------------------- 设置输入输出 -------------------------
    EXPECT_EQ(G.SetInCast({"input"}), true);
    EXPECT_EQ(G.SetOutCast({"final_out"}), true);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    // 执行pass
    MergeViewAssemble pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // 验证结果
    const auto& operations = function->Operations();

    // 1. 验证操作合并情况
    int view_count = 0;
    int assemble_count = 0;
    for (const auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count++;
        }
    }
    EXPECT_EQ(view_count, 3);     // 应合并为3个VIEW操作
    EXPECT_EQ(assemble_count, 4); // assemble3/4不应被合并

    // 验证final_out的生成路径
    bool found_abs = false;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ABS) {
            // 1. 检查输入数量是否为1
            const auto& abs_inputs = op.GetIOperands();
            ASSERT_EQ(abs_inputs.size(), 1) << "OP_ABS should have exactly 1 input tensor";

            // 2. 获取输入Tensor及其生产者
            const auto* input_tensor = abs_inputs[0].get();
            ASSERT_NE(input_tensor, nullptr) << "Input tensor is null";

            const auto& producers = input_tensor->GetProducers();
            ASSERT_EQ(producers.size(), 2) << "Input tensor should have 2 producers";

            // 3. 验证两个生产者均为ASSEMBLE操作
            int assemble_producer_count = 0;
            for (const auto* producer : producers) {
                ASSERT_NE(producer, nullptr) << "Producer is null";
                if (producer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                    assemble_producer_count++;
                }
            }
            EXPECT_EQ(assemble_producer_count, 2)
                << "Expected 2 ASSEMBLE producers, but got " << assemble_producer_count;

            found_abs = true;
            break;
        }
    }
    EXPECT_TRUE(found_abs) << "OP_ABS operation not found in the graph";
}
void MergeViewL1DataMoveGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> input_cast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> input_cast1_view = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2_view = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> redundant_view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> redundant_view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
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
    auto& head_view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1},
                                                         {input_cast1_view});
    head_view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& head_view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2},
                                                         {input_cast2_view});
    head_view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));

    auto& head_view_op11 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1_view},
                                                          {redundant_view_out1});
    head_view_op11.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& head_view_op12 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2_view},
                                                          {redundant_view_out2});
    head_view_op12.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));

    auto& view_L1_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {redundant_view_out1},
                                                       {op_view_L1_out1});
    std::vector<int> newoffset{0, 0};
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    view_L1_op1.SetOpAttribute(viewAttribute);

    auto& view_L1_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {redundant_view_out2},
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
TEST_F(MergeViewAssembleTest, MergeViewL1DataMove)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "MergeViewL1DataMove",
                                                      "MergeViewL1DataMove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("MergeViewL1DataMove", currFunctionPtr);

    MergeViewL1DataMoveGraph(currFunctionPtr);

    // 验证构图
    int view_count = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        }
    }
    EXPECT_EQ(view_count, 10);

    std::stringstream ssBefore;
    ssBefore << "Before_MergeViewAssemble";

    // Call the pass
    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_before.json");
    mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_after.json");
    mergeViewAssemble.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_MergeViewAssemble";

    // Validate the results
    int view_count_after_pass = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count_after_pass++;
        }
    }
    EXPECT_EQ(view_count_after_pass, NUM6);
}
void MergeViewWithAttr(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> view_in = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in}, {tensor1});
    auto viewAttribute1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute1->SetToType(MemoryType::MEM_UNKNOWN);
    view_op1.SetOpAttribute(viewAttribute1);
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto viewAttribute2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute2->SetToType(MemoryType::MEM_BT);
    view_op2.SetOpAttribute(viewAttribute2);
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor2}, {tensor3});
    auto viewAttribute3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute3->SetToType(MemoryType::MEM_BT);
    view_op3.SetOpAttribute(viewAttribute3);
    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor3}, {tensor4});
    auto viewAttribute4 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute4->SetToType(MemoryType::MEM_UB);
    view_op4.SetOpAttribute(viewAttribute4);
    auto& view_op5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor4}, {view_out});
    auto viewAttribute5 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute5->SetToType(MemoryType::MEM_UB);
    view_op5.SetOpAttribute(viewAttribute5);

    currFunctionPtr->inCasts_.push_back(view_in);
    currFunctionPtr->outCasts_.push_back(view_out);
}
TEST_F(MergeViewAssembleTest, MergeViewWithAttr)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "MergeViewWithAttr", "MergeViewWithAttr",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("MergeViewWithAttr", currFunctionPtr);

    MergeViewWithAttr(currFunctionPtr);

    // 验证构图
    int view_count = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        }
    }
    EXPECT_EQ(view_count, NUM5);

    std::stringstream ssBefore;
    ssBefore << "Before_MergeViewAssemble";

    // Call the pass
    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_before.json");
    mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_after.json");
    mergeViewAssemble.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_MergeViewAssemble";

    // Validate the results
    int view_count_after_pass = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count_after_pass++;
        }
    }
    EXPECT_EQ(view_count_after_pass, NUM2);
}

TEST_F(MergeViewAssembleTest, MergeViewInheritIsGemv)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "MergeViewInheritIsGemv",
                                                      "MergeViewInheritIsGemv", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("MergeViewInheritIsGemv", currFunctionPtr);

    MergeViewWithAttr(currFunctionPtr);
    // 给链尾 VIEW 设置 isGemv 属性（模拟 GEMV 的 DDR→L1 VIEW 信号）
    Operation* lastUbView = nullptr;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            lastUbView = &op;
        }
    }
    ASSERT_NE(lastUbView, nullptr);
    lastUbView->SetAttribute(OpAttributeKey::isGemv, static_cast<int64_t>(1));

    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currFunctionPtr);
    mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    mergeViewAssemble.PostCheck(*currFunctionPtr);

    // 白名单继承：合并后的 VIEW 中至少一个带 isGemv 属性
    bool isGemvInherited = false;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            int64_t isGemv = 0;
            if (op.GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv) && isGemv != 0) {
                isGemvInherited = true;
                break;
            }
        }
    }
    EXPECT_TRUE(isGemvInherited);
}

TEST_F(MergeViewAssembleTest, TestPreCheck)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMergeViewAssemble",
                                                      "TestMergeViewAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {8, 4};
    std::vector<int64_t> shape2 = {1, 8, 4};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {ubTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor}, {});

    currFunctionPtr->inCasts_.push_back(inCast);

    MergeViewAssemble mergeViewAssemble;
    Status status = mergeViewAssemble.PreCheck(*currFunctionPtr);
    EXPECT_EQ(status, FAILED);
}

/*
TestSpan
inCast{16,16}->view->ubTensor1{8,4}->assemble->ubTensor2{8,4}->assemble->outCast{8,4}
inCast{16,16}->view->ubTensor1{8,4}->assemble->outCast{8,4}
*/
TEST_F(MergeViewAssembleTest, TestSpan)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSpan", "TestSpan", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {16, 16};
    std::vector<int64_t> shape2 = {8, 4};
    auto span1 = ir::Span("view", NUM1, 0);
    auto span2 = ir::Span("view", NUM2, 0);
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto& viewOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor1}, span1);
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_UB);
    viewOp.SetOpAttribute(viewAttribute);
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {ubTensor2}, span2);
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->inCasts_.push_back(outCast);
    MergeViewAssemble mergeViewAssemble;
    Status status = mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            EXPECT_EQ(op.GetSpan().BeginLine(), NUM2);
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            EXPECT_EQ(op.GetSpan().BeginLine(), NUM1);
        }
    }
}

TEST_F(MergeViewAssembleTest, ViewChainSameScopeIdsShouldMerge)
{
    Program program;
    std::string funcMagicName = "test_view_same_scope";
    std::string funcRawName = "test_view_same_scope_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);

    // VIEW1(scopeId=1) -> VIEW2(scopeId=1), should merge
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{8, 8},
                                                                CreateTestConstIntVector(std::vector<int64_t>{8, 8}));
    auto view1Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {inputTensor}, {midTensor});
    view1Op.SetOpAttribute(view1Attr);
    view1Op.SetScopeId(1);

    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 6}, CreateTestConstIntVector(std::vector<int64_t>{6, 6}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto view2Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{3, 4}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {midTensor}, {outputTensor});
    view2Op.SetOpAttribute(view2Attr);
    view2Op.SetScopeId(1);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // Verify: 1 merged VIEW with scopeId=1
    int viewCount = 0;
    int mergedScopeId = -1;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW && !op.IsDeleted()) {
            viewCount++;
            mergedScopeId = op.GetScopeId();
        }
    }
    EXPECT_EQ(viewCount, 1);
    EXPECT_EQ(mergedScopeId, 1);
}

TEST_F(MergeViewAssembleTest, ViewChainMixedDefaultAndValidScopeIdShouldMerge)
{
    Program program;
    std::string funcMagicName = "test_view_mixed_scope";
    std::string funcRawName = "test_view_mixed_scope_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);

    // VIEW1(scopeId=-1, default) -> VIEW2(scopeId=1), should merge and inherit scopeId=1
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{8, 8},
                                                                CreateTestConstIntVector(std::vector<int64_t>{8, 8}));
    auto view1Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {inputTensor}, {midTensor});
    view1Op.SetOpAttribute(view1Attr);

    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 6}, CreateTestConstIntVector(std::vector<int64_t>{6, 6}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto view2Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{3, 4}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {midTensor}, {outputTensor});
    view2Op.SetOpAttribute(view2Attr);
    view2Op.SetScopeId(1);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // Verify: 1 merged VIEW with scopeId=1
    int viewCount = 0;
    int mergedScopeId = -2;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW && !op.IsDeleted()) {
            viewCount++;
            mergedScopeId = op.GetScopeId();
        }
    }
    EXPECT_EQ(viewCount, 1);
    EXPECT_EQ(mergedScopeId, 1);
}

TEST_F(MergeViewAssembleTest, AssembleChainDifferentScopeIdsShouldNotMerge)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames = {"input", "view_out", "assemble1_out", "assemble2_out", "final_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, tensorNames));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"view_out"}, {"assemble1_out"}, {"assemble2_out"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"view_out"}, {"assemble1_out"}, {"assemble2_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "assemble1", "assemble2", "abs"};

    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* view1 = G.GetOp("view1");
    view1->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
                                                            std::vector<SymbolicScalar>{}));

    auto* assemble1 = G.GetOp("assemble1");
    assemble1->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{}));
    assemble1->SetScopeId(1);

    auto* assemble2 = G.GetOp("assemble2");
    assemble2->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2}, std::vector<SymbolicScalar>{}));
    assemble2->SetScopeId(2);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // Verify: 2 ASSEMBLEs remain (not merged due to different scopeIds)
    int assembleCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            assembleCount++;
        }
    }
    EXPECT_EQ(assembleCount, 2);
}

TEST_F(MergeViewAssembleTest, MergeAssembleChainShouldInheritAtomicAddAttr)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames = {"input", "view_out", "assemble1_out", "assemble2_out", "final_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, tensorNames));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"view_out"}, {"assemble1_out"}, {"assemble2_out"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"view_out"}, {"assemble1_out"}, {"assemble2_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "assemble1", "assemble2", "abs"};

    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* view1 = G.GetOp("view1");
    ASSERT_NE(view1, nullptr);
    view1->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
                                                            std::vector<SymbolicScalar>{}));

    auto* assemble1 = G.GetOp("assemble1");
    ASSERT_NE(assemble1, nullptr);
    assemble1->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{}));
    assemble1->SetAttribute(RMW_MODE_ATTR_ADD, 1L);

    auto* assemble2 = G.GetOp("assemble2");
    ASSERT_NE(assemble2, nullptr);
    assemble2->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2}, std::vector<SymbolicScalar>{}));

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    int assembleCount = 0;
    Operation* mergedAssemble = nullptr;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            assembleCount++;
            mergedAssemble = &op;
        }
    }
    ASSERT_EQ(assembleCount, 1);
    ASSERT_NE(mergedAssemble, nullptr);
    EXPECT_TRUE(mergedAssemble->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(MergeViewAssembleTest, AssembleChainWithDifferentRmwModeAttrsShouldNotMerge)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames = {"input", "view_out", "assemble1_out", "assemble2_out", "final_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, tensorNames));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"view_out"}, {"assemble1_out"}, {"assemble2_out"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"view_out"}, {"assemble1_out"}, {"assemble2_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "assemble1", "assemble2", "abs"};

    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* view1 = G.GetOp("view1");
    ASSERT_NE(view1, nullptr);
    view1->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
                                                            std::vector<SymbolicScalar>{}));

    auto* assemble1 = G.GetOp("assemble1");
    ASSERT_NE(assemble1, nullptr);
    assemble1->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{}));
    assemble1->SetAttribute(RMW_MODE_ATTR_ADD, 1L);

    auto* assemble2 = G.GetOp("assemble2");
    ASSERT_NE(assemble2, nullptr);
    assemble2->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2}, std::vector<SymbolicScalar>{}));
    assemble2->SetAttribute(RMW_MODE_ATTR_MIN, 1L);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    int assembleCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            assembleCount++;
        }
    }
    EXPECT_EQ(assembleCount, 2);
}

TEST_F(MergeViewAssembleTest, BuilderLargeSharedTensorMergeableChainsPerfGuard)
{
    int assembleChainCount = 4096;
    int viewChainCount = 4096;
    int64_t thresholdMs = 1000;

    ComputationalGraphBuilder graph;
    BuildLargeSharedTensorMergeableChains(graph, assembleChainCount, viewChainCount);
    Function* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    ASSERT_EQ(function->Operations(false).size(), static_cast<size_t>(assembleChainCount * 3 + viewChainCount * 2));

    MergeViewAssemble pass;
    ASSERT_EQ(pass.PreCheck(*function), SUCCESS);
    auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    auto end = std::chrono::steady_clock::now();
    int64_t elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    size_t viewOpCount = 0;
    size_t assembleOpCount = 0;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewOpCount;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleOpCount;
        }
    }
    EXPECT_EQ(viewOpCount, static_cast<size_t>(viewChainCount));
    EXPECT_EQ(assembleOpCount, static_cast<size_t>(assembleChainCount));
    std::cout << "BuilderLargeSharedTensorMergeableChainsPerfGuard elapsed: " << elapsedMs << " ms" << std::endl;
    EXPECT_LT(elapsedMs, thresholdMs);
}

static void SetSimpleViewAttr(Operation* op, const std::vector<int64_t>& offset)
{
    ASSERT_NE(op, nullptr);
    op->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(offset, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{}));
}

static void SetSimpleAssembleAttr(Operation* op, const std::vector<int64_t>& offset)
{
    ASSERT_NE(op, nullptr);
    op->SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset, std::vector<SymbolicScalar>{}));
}

TEST_F(MergeViewAssembleTest, ViewSliceChainShouldMergeToSlice)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_VIEW, {"input"}, {"mid"}, "view", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_SLICE, {"mid"}, {"output"}, "slice", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleViewAttr(G.GetOp("view"), {1, 2});
    SetSimpleViewAttr(G.GetOp("slice"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int sliceCount = 0;
    Operation* mergedSlice = nullptr;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.IsDeleted()) {
            sliceCount++;
            mergedSlice = &op;
        }
        EXPECT_NE(op.GetOpcode(), Opcode::OP_VIEW);
    }
    ASSERT_EQ(sliceCount, 1);
    ASSERT_NE(mergedSlice, nullptr);
    auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(mergedSlice->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->GetFromOffset(), (std::vector<int64_t>{4, 6}));
}

TEST_F(MergeViewAssembleTest, SliceSliceChainShouldNotMerge)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_SLICE, {"input"}, {"mid"}, "slice1", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_SLICE, {"mid"}, {"output"}, "slice2", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleViewAttr(G.GetOp("slice1"), {1, 2});
    SetSimpleViewAttr(G.GetOp("slice2"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int sliceCount = 0;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.IsDeleted()) {
            sliceCount++;
        }
    }
    EXPECT_EQ(sliceCount, 2);
}

TEST_F(MergeViewAssembleTest, AssembleContractChainShouldMergeToContract)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"input"}, {"mid"}, "assemble", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_CONTRACT, {"mid"}, {"output"}, "contract", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleAssembleAttr(G.GetOp("assemble"), {1, 2});
    SetSimpleAssembleAttr(G.GetOp("contract"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int contractCount = 0;
    Operation* mergedContract = nullptr;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT && !op.IsDeleted()) {
            contractCount++;
            mergedContract = &op;
        }
        EXPECT_NE(op.GetOpcode(), Opcode::OP_ASSEMBLE);
    }
    ASSERT_EQ(contractCount, 1);
    ASSERT_NE(mergedContract, nullptr);
    auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(mergedContract->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->GetToOffset(), (std::vector<int64_t>{4, 6}));
}

TEST_F(MergeViewAssembleTest, ContractContractChainShouldNotMerge)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_CONTRACT, {"input"}, {"mid"}, "contract1", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_CONTRACT, {"mid"}, {"output"}, "contract2", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleAssembleAttr(G.GetOp("contract1"), {1, 2});
    SetSimpleAssembleAttr(G.GetOp("contract2"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int contractCount = 0;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT && !op.IsDeleted()) {
            contractCount++;
        }
    }
    EXPECT_EQ(contractCount, 2);
}
} // namespace tile_fwk
} // namespace npu
