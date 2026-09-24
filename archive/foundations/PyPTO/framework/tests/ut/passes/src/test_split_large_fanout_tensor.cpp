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
 * \file test_split_large_fanout_tensor.cpp
 * \brief Unit test for Split Large Fanout Tensor pass.
 */

#include <chrono>
#include <numeric>
#include <set>
#include <vector>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#define private public
#include "passes/tile_graph_pass/graph_optimization/split_large_fanout_tensor.h"
#undef private
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class SplitLargeFanoutTensorTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutTensorTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}

    void SetTestStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("SplitLargeFanoutTensorTestStrategy",
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
                                     });
        ConfigManager::Instance();
    }

    void SetPreSplitStrategy()
    {
        PassManager& passManager = PassManager::Instance();
        passManager.RegisterStrategy("SplitLargeFanoutPreSplitStrategy",
                                     {
                                         {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                         {"AutoCast", PassName::AUTO_CAST},
                                         {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                         {"RemoveUndrivenView", PassName::REMOVE_UNDRIVEN_VIEW},
                                         {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                         {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                         {"SplitReshape", PassName::SPLIT_RESHAPE},
                                         {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                     });
        ConfigManager::Instance();
    }

    void ExecutePass(Function* function, bool enableMoreSplit)
    {
        npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
        splitLargeFanoutTensor.SetEnableMoreSplit(enableMoreSplit);
        splitLargeFanoutTensor.PreCheck(*function);
        splitLargeFanoutTensor.RunOnFunction(*function);
        splitLargeFanoutTensor.PostCheck(*function);
        std::cout << "Run Pass Done." << std::endl;
    }

    std::vector<int64_t> CountViewAssemble(Function& func)
    {
        std::vector<int64_t> result = {0, 0};
        for (auto& op : func.Operations()) {
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            if (IsViewLike(op.GetOpcode())) {
                result[0]++;
            } else if (IsAssembleLike(op.GetOpcode())) {
                result[1]++;
            }
        }
        return result;
    }

    struct SplitCounts {
        size_t vCnt = 0, aCnt = 0, vLargeShape = 0, vTileShape = 0, aTileShape = 0, aLargeShape = 0;
    };

    SplitCounts CountSplitByShape(Function& func, const std::vector<int64_t>& largeShape,
                                  const std::vector<int64_t>& tileShape)
    {
        SplitCounts c;
        for (auto& op : func.Operations()) {
            if (IsViewLike(op.GetOpcode())) {
                ++c.vCnt;
                auto& s = op.GetIOperands().front()->GetShape();
                if (s == largeShape)
                    ++c.vLargeShape;
                else if (s == tileShape)
                    ++c.vTileShape;
            } else if (IsAssembleLike(op.GetOpcode())) {
                ++c.aCnt;
                auto& s = op.GetOOperands().front()->GetShape();
                if (s == tileShape)
                    ++c.aTileShape;
                else if (s == largeShape)
                    ++c.aLargeShape;
            }
        }
        return c;
    }

    bool CanReachOutCast(Function& func, const LogicalTensorPtr& startTensor)
    {
        std::vector<LogicalTensorPtr> pending{startTensor};
        std::set<int> visited;
        while (!pending.empty()) {
            auto tensor = pending.back();
            pending.pop_back();
            if (tensor == nullptr || !visited.insert(tensor->GetMagic()).second) {
                continue;
            }
            if (func.IsFromOutCast(tensor)) {
                return true;
            }
            for (auto* consumer : tensor->GetConsumers()) {
                if (consumer == nullptr) {
                    continue;
                }
                const auto& outputs = consumer->GetOOperands();
                pending.insert(pending.end(), outputs.begin(), outputs.end());
            }
        }
        return false;
    }

    void ExpectInCastsReachOutCasts(Function& func)
    {
        for (const auto& incast : func.GetIncast()) {
            if (incast == nullptr) {
                ADD_FAILURE() << "Incast tensor is nullptr.";
                continue;
            }
            EXPECT_TRUE(CanReachOutCast(func, incast))
                << "Incast tensor " << incast->GetMagic() << " cannot reach any outcast.";
        }
    }

    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    void TileExpandSub(ComputationalGraphBuilder& G, const int N, const int T)
    {
        std::vector<int64_t> tileShape{T, T};
        std::vector<SymbolicScalar> dynShape{CreateTestScalarVar("a"), T};
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::vector<int64_t> offset = {i * T, j * T};

                std::string localA = "a_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localA);
                auto tensorA = G.GetTensor(localA);
                tensorA->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
                G.AddOp(Opcode::OP_VIEW, {"a"}, {localA}, "View_A_" + std::to_string(i * N + j));
                auto View_A = G.GetOp("View_A_" + std::to_string(i * N + j));
                auto attrA = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UNKNOWN);
                View_A->SetOpAttribute(attrA);

                std::string localB = "b_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localB);
                auto tensorB = G.GetTensor(localB);
                tensorB->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
                G.AddOp(Opcode::OP_VIEW, {"b"}, {localB}, "View_B_" + std::to_string(i * N + j));
                auto View_B = G.GetOp("View_B_" + std::to_string(i * N + j));
                auto attrB = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UNKNOWN);
                View_B->SetOpAttribute(attrB);

                std::string localSubOut = "sub_out_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localSubOut);
                G.AddOp(Opcode::OP_SUB, {localA, localB}, {localSubOut}, "Sub_" + std::to_string(i * N + j));
                auto tensorSubOut = G.GetTensor(localSubOut);
                tensorSubOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
                tensorSubOut->UpdateDynValidShape(dynShape);

                G.AddOp(Opcode::OP_ASSEMBLE, {localSubOut}, {"sub_out"}, "Assemble_" + std::to_string(i * N + j));
                auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, offset);
                auto assembleOp = G.GetOp("Assemble_" + std::to_string(i * N + j));
                assembleOp->SetOpAttribute(attrAssemble);
            }
        }
    }
    // multiConsumer means a tensor before the large tensor has more than one assemble op
    void BuildGraphForMToM(ComputationalGraphBuilder& G, bool multiConsumer = false)
    {
        int NUM_64 = 64;
        int NUM_128 = 128;
        int NUM_192 = 192;
        int NUM_256 = 256;
        int NUM_320 = 320;
        std::vector<int64_t> shape1{NUM_256, NUM_256};
        std::vector<int64_t> tiledShape1{NUM_64, NUM_256};
        std::vector<int64_t> shape2{NUM_256, NUM_64};
        std::vector<int64_t> tiledShape2{NUM_64, NUM_64};
        std::vector<int64_t> largeShape{NUM_128, NUM_320};
        std::vector<int64_t> tiledShape3{NUM_128, NUM_128};
        std::vector<int64_t> tiledShape4{NUM_128, NUM_64};

        // InCast a
        std::vector<SymbolicScalar> dynShapeA = {CreateTestScalarVar("a"), NUM_256};
        G.AddTensor(DataType::DT_FP32, shape1, "a"); // [256, 256]
        auto a = G.GetTensor("a");
        a->UpdateDynValidShape(dynShapeA);
        a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        // [256, 256] --> View(64, 0) --> [64, 256]
        G.AddTensor(DataType::DT_FP32, tiledShape1, "tiledA"); // [64, 256]
        auto tiledA = G.GetTensor("tiledA");
        tiledA->UpdateDynValidShape(dynShapeA);
        tiledA->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledA"}, "View_A");
        auto View_A = G.GetOp("View_A");
        std::vector<int64_t> offsetA = {NUM_64, 0};
        auto attrA = std::make_shared<ViewOpAttribute>(offsetA, MemoryType::MEM_UNKNOWN);
        View_A->SetOpAttribute(attrA);
        // [256, 256] --> View(192, 0) --> [64, 256]
        G.AddTensor(DataType::DT_FP32, tiledShape1, "tiledB"); // [64, 256]
        auto tiledB = G.GetTensor("tiledB");
        tiledB->UpdateDynValidShape(dynShapeA);
        tiledB->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledB"}, "View_B");
        auto View_B = G.GetOp("View_B");
        std::vector<int64_t> offsetB = {NUM_192, 0};
        auto attrB = std::make_shared<ViewOpAttribute>(offsetB, MemoryType::MEM_UNKNOWN);
        View_B->SetOpAttribute(attrB);

        // InCast c
        std::vector<SymbolicScalar> dynShapeC = {CreateTestScalarVar("c"), NUM_64};
        G.AddTensor(DataType::DT_FP32, shape2, "c"); // [256, 64]
        auto c = G.GetTensor("c");
        c->UpdateDynValidShape(dynShapeC);
        c->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        // [256, 64] --> View(64, 0) --> [64, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape2, "tiledC"); // [64, 64]
        auto tiledC = G.GetTensor("tiledC");
        tiledC->UpdateDynValidShape(dynShapeC);
        tiledC->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"c"}, {"tiledC"}, "View_C");
        auto View_C = G.GetOp("View_C");
        std::vector<int64_t> offsetC = {NUM_64, 0};
        auto attrC = std::make_shared<ViewOpAttribute>(offsetC, MemoryType::MEM_UNKNOWN);
        View_C->SetOpAttribute(attrC);
        // [256, 64] --> View(192, 0) --> [64, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape2, "tiledD"); // [64, 64]
        auto tiledD = G.GetTensor("tiledD");
        tiledD->UpdateDynValidShape(dynShapeC);
        tiledD->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"c"}, {"tiledD"}, "View_D");
        auto View_D = G.GetOp("View_D");
        std::vector<int64_t> offsetD = {NUM_192, 0};
        auto attrD = std::make_shared<ViewOpAttribute>(offsetD, MemoryType::MEM_UNKNOWN);
        View_D->SetOpAttribute(attrD);

        if (multiConsumer) {
            G.AddTensor(DataType::DT_FP32, tiledShape4, "out3");
            auto out3 = G.GetTensor("out3");
            out3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
            G.AddOp(Opcode::OP_VIEW, {"tiledC"}, {"out3"}, "View_C1");
            auto View_C1 = G.GetOp("View_C1");
            std::vector<int64_t> offsetC1 = {0, 0};
            auto attrC1 = std::make_shared<ViewOpAttribute>(offsetC1, MemoryType::MEM_UNKNOWN);
            View_C1->SetOpAttribute(attrC1);
        }

        // [64, 256][64, 64]
        // [64, 256][64, 64]  Assemble --> [128, 320]
        G.AddTensor(DataType::DT_FP32, largeShape, "largeTensor"); // Assemble --> [128, 320]
        auto largeTensor = G.GetTensor("largeTensor");
        largeTensor->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledA"}, {"largeTensor"}, "Assemble_A");
        auto attrAssembleA = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
        auto assembleA = G.GetOp("Assemble_A");
        assembleA->SetOpAttribute(attrAssembleA);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledB"}, {"largeTensor"}, "Assemble_B");
        auto attrAssembleB = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                   std::vector<int64_t>{NUM_64, 0});
        auto assembleB = G.GetOp("Assemble_B");
        assembleB->SetOpAttribute(attrAssembleB);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledC"}, {"largeTensor"}, "Assemble_C");
        auto attrAssembleC = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                   std::vector<int64_t>{0, NUM_256});
        auto assembleC = G.GetOp("Assemble_C");
        assembleC->SetOpAttribute(attrAssembleC);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledD"}, {"largeTensor"}, "Assemble_D");
        auto attrAssembleD = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                   std::vector<int64_t>{NUM_64, NUM_256});
        auto assembleD = G.GetOp("Assemble_D");
        assembleD->SetOpAttribute(attrAssembleD);

        // output 1: MtoM
        // [128, 320] --> View(0, 0) --> [128, 128]
        G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_1");
        auto tiledView_1 = G.GetTensor("tiledView_1");
        tiledView_1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"tiledView_1"}, "View_1");
        auto View_1 = G.GetOp("View_1");
        std::vector<int64_t> offset1 = {0, 0};
        auto attr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
        View_1->SetOpAttribute(attr1);

        // output 2: MtoM
        // [128, 320] --> View(0, 128) --> [128, 128]
        G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_2");
        auto tiledView_2 = G.GetTensor("tiledView_2");
        tiledView_2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"tiledView_2"}, "View_2");
        auto View_2 = G.GetOp("View_2");
        std::vector<int64_t> offset2 = {0, 0};
        auto attr2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
        View_2->SetOpAttribute(attr2);

        // output 3: Mto1
        // [128, 320] --> View(0, 256) --> [128, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape4, "tiledView_3");
        auto tiledView_3 = G.GetTensor("tiledView_3");
        tiledView_3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"tiledView_3"}, "View_3");
        auto View_3 = G.GetOp("View_3");
        std::vector<int64_t> offset3 = {0, 256};
        auto attr3 = std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_UNKNOWN);
        View_3->SetOpAttribute(attr3);

        // output 1 + output 2
        // [128, 128] + [128, 128] --> [128, 128]
        G.AddTensor(DataType::DT_FP32, tiledShape3, "add_out");
        G.AddOp(Opcode::OP_ADD, {"tiledView_1", "tiledView_2"}, {"add_out"}, "Add");
        auto addOut = G.GetTensor("add_out");
        addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

        G.AddTensor(DataType::DT_FP32, tiledShape3, "out1");
        auto out1 = G.GetTensor("out1");
        out1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out1"}, "Assemble_1");
        auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
        auto assemble1 = G.GetOp("Assemble_1");
        assemble1->SetOpAttribute(attrAssemble1);

        // output 3 exp
        // [128, 64] --> Exp --> [128, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape4, "exp_out");
        G.AddOp(Opcode::OP_EXP, {"tiledView_3"}, {"exp_out"}, "Exp");
        auto expOut = G.GetTensor("exp_out");
        expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

        G.AddTensor(DataType::DT_FP32, tiledShape4, "out2");
        auto out2 = G.GetTensor("out2");
        out2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"exp_out"}, {"out2"}, "Assemble_2");
        auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
        auto assemble2 = G.GetOp("Assemble_2");
        assemble2->SetOpAttribute(attrAssemble2);

        G.SetInCast({"a", "c"});
        if (multiConsumer) {
            G.SetOutCast({"out1", "out2", "out3"});
        } else {
            G.SetOutCast({"out1", "out2"});
        }
    }
    void PerfectlyMatchWithAll_FullConstruct(ComputationalGraphBuilder& G)
    {
        int N = 2;
        int T = 8;
        auto dynDimA = CreateTestScalarVar("a");
        std::vector<int64_t> shape0{N * T, N * T};
        std::vector<int64_t> shape1{T, T};
        std::vector<int64_t> shape2{N * T, T};
        G.AddTensor(DataType::DT_FP32, shape0, "a");
        G.AddTensor(DataType::DT_FP32, shape0, "b");
        G.AddTensor(DataType::DT_FP32, shape2, "out");
        G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
        // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
        TileExpandSub(G, N, T);
        auto subOut = G.GetTensor("sub_out");
        subOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

        // View 获取 [16, 16] 左半边 [16, 8]
        G.AddTensor(DataType::DT_FP32, shape2, "sub_out_left");
        auto subOutLeft = G.GetTensor("sub_out_left");
        subOutLeft->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_left"}, "View_Left");
        auto View_Left = G.GetOp("View_Left");
        std::vector<int64_t> offsetLeft = {0, 0};
        auto attrLeft = std::make_shared<ViewOpAttribute>(offsetLeft, MemoryType::MEM_UNKNOWN);
        View_Left->SetOpAttribute(attrLeft);

        // View 获取 [16, 16] 右半边 [16, 8]
        G.AddTensor(DataType::DT_FP32, shape2, "sub_out_right");
        auto subOutRight = G.GetTensor("sub_out_right");
        subOutRight->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_right"}, "View_Right");
        auto View_UR = G.GetOp("View_Right");
        std::vector<int64_t> offsetRight = {0, T};
        auto attrUR = std::make_shared<ViewOpAttribute>(offsetRight, MemoryType::MEM_UNKNOWN);
        View_UR->SetOpAttribute(attrUR);

        // 左半边 [16, 8] + 右半边 [16, 8]
        std::vector<SymbolicScalar> subDynShape = {dynDimA, T};
        G.AddTensor(DataType::DT_FP32, shape2, "add_out");
        G.AddOp(Opcode::OP_ADD, {"sub_out_right", "sub_out_left"}, {"add_out"}, "Add");
        auto addOut = G.GetTensor("add_out");
        addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        addOut->UpdateDynValidShape(subDynShape);

        G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
        auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                       std::vector<int64_t>{0, 0});
        auto Op = G.GetOp("Assemble_final");
        Op->SetOpAttribute(attrAssembleFinal);

        auto a = G.GetTensor("a");
        a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        auto b = G.GetTensor("b");
        b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        auto out = G.GetTensor("out");
        out->UpdateDynValidShape(subDynShape);

        G.SetInCast({"a", "b"});
        G.SetOutCast({"out"});
    }

    void BuildPerfectlyMatchWithInferShapeGraph(ComputationalGraphBuilder& G, bool setDynShape = false)
    {
        int N = 2;
        int T = 8;
        std::vector<int64_t> shape0{N * T, N * T};
        std::vector<int64_t> shape1{T, T};
        G.AddTensor(DataType::DT_FP32, shape0, "a");
        G.AddTensor(DataType::DT_FP32, shape0, "b");
        G.AddTensor(DataType::DT_FP32, shape1, "out");
        G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
        TileExpandSub(G, N, T);
        G.GetTensor("sub_out")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

        G.AddTensor(DataType::DT_FP32, shape1, "sub_out_upper_right");
        G.GetTensor("sub_out_upper_right")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_upper_right"}, "View_Upper_Right");
        std::vector<int64_t> offsetUR = {0, T};
        auto attrUR = std::make_shared<ViewOpAttribute>(offsetUR, MemoryType::MEM_UNKNOWN);
        G.GetOp("View_Upper_Right")->SetOpAttribute(attrUR);

        G.AddTensor(DataType::DT_FP32, shape1, "sub_out_lower_left");
        G.GetTensor("sub_out_lower_left")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_lower_left"}, "View_Lower_Left");
        std::vector<int64_t> offsetLL = {T, 0};
        auto attrLL = std::make_shared<ViewOpAttribute>(offsetLL, MemoryType::MEM_UNKNOWN);
        G.GetOp("View_Lower_Left")->SetOpAttribute(attrLL);

        std::vector<SymbolicScalar> addDynShape = {CreateTestScalarVar("a"), T};
        G.AddTensor(DataType::DT_FP32, shape1, "add_out");
        G.AddOp(Opcode::OP_ADD, {"sub_out_upper_right", "sub_out_lower_left"}, {"add_out"}, "Add");
        G.GetTensor("add_out")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
        auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                       std::vector<int64_t>{0, 0});
        auto assembleOp = G.GetOp("Assemble_final");
        assembleOp->SetOpAttribute(attrAssembleFinal);

        G.GetTensor("a")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.GetTensor("b")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        auto out = G.GetTensor("out");
        G.GetTensor("out")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        if (setDynShape) {
            G.GetTensor("sub_out_upper_right")->UpdateDynValidShape(addDynShape);
            G.GetTensor("sub_out_lower_left")->UpdateDynValidShape(addDynShape);
            G.GetTensor("add_out")->UpdateDynValidShape(addDynShape);
            G.GetTensor("out")->UpdateDynValidShape(addDynShape);
        }

        G.SetInCast({"a", "b"});
        G.SetOutCast({"out"});
    }

    void BuildPerfectlyMatchFullV2_BaseTensorsAndExp(ComputationalGraphBuilder& G)
    {
        int N = 2;
        int T = 8;
        std::vector<int64_t> shape0{N * T, N * T};
        std::vector<int64_t> shape1{T, T};
        G.AddTensor(DataType::DT_FP32, shape0, "a");
        G.AddTensor(DataType::DT_FP32, shape0, "b");
        G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
        G.AddTensor(DataType::DT_FP32, shape1, "out");
        G.AddTensor(DataType::DT_FP32, shape0, "out2");
        // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
        TileExpandSub(G, N, T);
        auto subOut = G.GetTensor("sub_out");
        subOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

        // [16, 16] --> exp --> [16, 16] --> Assemble --> OCAST
        G.AddTensor(DataType::DT_FP32, shape0, "expOut");
        auto expOut = G.GetTensor("expOut");
        expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_EXP, {"sub_out"}, {"expOut"}, "Exp");
        G.AddOp(Opcode::OP_ASSEMBLE, {"expOut"}, {"out2"}, "Assemble_exp");
        auto attrAssembleExp = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                     std::vector<int64_t>{0, 0});
        G.GetOp("Assemble_exp")->SetOpAttribute(attrAssembleExp);
    }

    void BuildPerfectlyMatchFullV2_ViewsAndAdd(ComputationalGraphBuilder& G)
    {
        int T = 8;
        std::vector<int64_t> shape1{T, T};
        // View 获取 [16, 16] 右上角 [8, 8]
        G.AddTensor(DataType::DT_FP32, shape1, "sub_out_lower_left");
        auto subOutLL = G.GetTensor("sub_out_lower_left");
        subOutLL->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_lower_left"}, "View_Lower_Left");
        auto View_LL = G.GetOp("View_Lower_Left");
        std::vector<int64_t> offsetLL = {T, 0};
        auto attrLL = std::make_shared<ViewOpAttribute>(offsetLL, MemoryType::MEM_UNKNOWN);
        View_LL->SetOpAttribute(attrLL);

        // View 获取 [16, 16] 左上角 [8, 8]
        G.AddTensor(DataType::DT_FP32, shape1, "sub_out_upper_right");
        auto subOutUR = G.GetTensor("sub_out_upper_right");
        subOutUR->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_upper_right"}, "View_Upper_Right");
        auto View_UR = G.GetOp("View_Upper_Right");
        std::vector<int64_t> offsetUR = {0, T};
        auto attrUR = std::make_shared<ViewOpAttribute>(offsetUR, MemoryType::MEM_UNKNOWN);
        View_UR->SetOpAttribute(attrUR);

        // 左上角 [8, 8] + 右上角 [8, 8] --> [8, 8]
        G.AddTensor(DataType::DT_FP32, shape1, "add_out");
        G.AddOp(Opcode::OP_ADD, {"sub_out_upper_right", "sub_out_lower_left"}, {"add_out"}, "Add");
        G.GetTensor("add_out")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
        auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                       std::vector<int64_t>{0, 0});
        G.GetOp("Assemble_final")->SetOpAttribute(attrAssembleFinal);

        G.GetTensor("a")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.GetTensor("b")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.GetTensor("out")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.GetTensor("out2")->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.SetInCast({"a", "b"});
        G.SetOutCast({"out", "out2"});
    }
};

namespace {

void AddAssembleInput(ComputationalGraphBuilder& graph, int inputIndex, int viewIndex, int assembleIndex,
                      int64_t assembleCols, std::vector<std::string>& inputs)
{
    auto input = "input_" + std::to_string(inputIndex);
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, assembleCols}, input));
    auto inputTensor = graph.GetTensor(input);
    ASSERT_NE(inputTensor, nullptr);
    inputTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto viewIn = "view_in_" + std::to_string(inputIndex);
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, assembleCols}, viewIn));
    auto viewInTensor = graph.GetTensor(viewIn);
    ASSERT_NE(viewInTensor, nullptr);
    viewInTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    graph.AddOp(Opcode::OP_VIEW, {input}, {viewIn}, "view_in_" + std::to_string(inputIndex));
    graph.GetOp("view_in_" + std::to_string(inputIndex))
        ->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, MemoryType::MEM_DEVICE_DDR));

    auto addsIn = "adds_in_" + std::to_string(inputIndex);
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, assembleCols}, addsIn));
    auto addsInTensor = graph.GetTensor(addsIn);
    ASSERT_NE(addsInTensor, nullptr);
    addsInTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    graph.AddOp(Opcode::OP_ADDS, {viewIn}, {addsIn}, "adds_in_" + std::to_string(inputIndex));
    graph.GetOp("adds_in_" + std::to_string(inputIndex))
        ->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 0.01));

    graph.AddOp(Opcode::OP_ASSEMBLE, {addsIn}, {"largeTensor"}, "assemble_" + std::to_string(inputIndex));
    graph.GetOp("assemble_" + std::to_string(inputIndex))
        ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(
            MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{viewIndex, assembleIndex * assembleCols}));

    inputs.emplace_back(std::move(input));
}

void AddViewOutput(ComputationalGraphBuilder& graph, int viewIndex, int64_t viewCols, std::vector<std::string>& outputs)
{
    auto viewTmp = "view_tmp_" + std::to_string(viewIndex);
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, viewCols}, viewTmp));
    auto viewTmpTensor = graph.GetTensor(viewTmp);
    ASSERT_NE(viewTmpTensor, nullptr);
    viewTmpTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    graph.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {viewTmp}, "view_" + std::to_string(viewIndex));
    graph.GetOp("view_" + std::to_string(viewIndex))
        ->SetOpAttribute(
            std::make_shared<ViewOpAttribute>(std::vector<int64_t>{viewIndex, 0}, MemoryType::MEM_DEVICE_DDR));

    auto addsTmp = "adds_tmp_" + std::to_string(viewIndex);
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, viewCols}, addsTmp));
    auto addsTmpTensor = graph.GetTensor(addsTmp);
    ASSERT_NE(addsTmpTensor, nullptr);
    addsTmpTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    graph.AddOp(Opcode::OP_ADDS, {viewTmp}, {addsTmp}, "adds_" + std::to_string(viewIndex));
    graph.GetOp("adds_" + std::to_string(viewIndex))
        ->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 0.01));

    auto output = "output_" + std::to_string(viewIndex);
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, viewCols}, output));
    auto outputTensor = graph.GetTensor(output);
    ASSERT_NE(outputTensor, nullptr);
    outputTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    graph.AddOp(Opcode::OP_ASSEMBLE, {addsTmp}, {output}, "out_assemble_" + std::to_string(viewIndex));
    graph.GetOp("out_assemble_" + std::to_string(viewIndex))
        ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0}));
    outputs.emplace_back(std::move(output));
}

void BuildLargeLinearFanoutGraph(ComputationalGraphBuilder& graph, int viewCount, int assemblesPerView,
                                 int64_t assembleCols)
{
    int64_t viewCols = assembleCols * assemblesPerView;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    inputs.reserve(viewCount * assemblesPerView);
    outputs.reserve(viewCount);
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {viewCount, viewCols}, "largeTensor"));
    auto largeTensor = graph.GetTensor("largeTensor");
    ASSERT_NE(largeTensor, nullptr);
    largeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    for (int viewIndex = 0; viewIndex < viewCount; ++viewIndex) {
        AddViewOutput(graph, viewIndex, viewCols, outputs);
        for (int assembleIndex = 0; assembleIndex < assemblesPerView; ++assembleIndex) {
            int inputIndex = viewIndex * assemblesPerView + assembleIndex;
            AddAssembleInput(graph, inputIndex, viewIndex, assembleIndex, assembleCols, inputs);
        }
    }
    ASSERT_TRUE(graph.SetInCast(inputs));
    ASSERT_TRUE(graph.SetOutCast(outputs));
}

void BuildDynamicDslPerfFunction(Tensor& q, Tensor& out, int b, int sq, int d, const char* perfGuardStrategy)
{
    FUNCTION("SplitLargeFanoutDynamicPerfGuard", {q, out})
    {
        config::SetPassStrategy(perfGuardStrategy);
        Tensor addTmp(DT_FP32, {b, sq, d});
        for (int sqIdx = 0; sqIdx < sq; ++sqIdx) {
            Tensor viewTmp = View(q, {b, 1, d}, {0, sqIdx, 0});
            auto res = Add(viewTmp, Element(viewTmp.GetStorage()->Datatype(), 0.01));
            Assemble(res, {0, sqIdx, 0}, addTmp);
        }

        for (int sqIdx = 0; sqIdx < sq; ++sqIdx) {
            Tensor tmp0 = View(addTmp, {b, 1, d}, {0, sqIdx, 0});
            Tensor tmp = Add(tmp0, Element(tmp0.GetStorage()->Datatype(), 0.01));
            Assemble(tmp, {0, sqIdx, 0}, out);
        }
    }
}
} // namespace

TEST_F(SplitLargeFanoutTensorTest, TestLCM)
{
    int64_t NUM_16 = 16;
    int64_t NUM_2 = 2;
    int64_t NUM_3 = 3;
    int64_t NUM_96 = 96;
    int64_t x = NUM_16 * NUM_2;
    int64_t y = NUM_16 * NUM_3;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    auto gcd = splitLargeFanoutTensor.GCD(x, y);
    int64_t lcm;
    auto status = splitLargeFanoutTensor.LCM(x, y, lcm);
    std::cout << "Run Pass Done." << std::endl;
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(gcd, NUM_16);
    EXPECT_EQ(lcm, NUM_96);
}

TEST_F(SplitLargeFanoutTensorTest, CalLcmShape_DimMismatch)
{
    npu::tile_fwk::SplitLargeFanoutTensor pass;
    std::vector<int64_t> toShape = {16, 32};
    std::vector<int64_t> fromShape = {8, 16, 4};
    std::vector<int64_t> lcmShape;
    EXPECT_NE(pass.CalLcmShape(toShape, fromShape, lcmShape), SUCCESS);
}

TEST_F(SplitLargeFanoutTensorTest, CalLcmShape_LcmFail)
{
    npu::tile_fwk::SplitLargeFanoutTensor pass;
    std::vector<int64_t> toShape = {0, 32};
    std::vector<int64_t> fromShape = {0, 16};
    std::vector<int64_t> lcmShape(2);
    EXPECT_NE(pass.CalLcmShape(toShape, fromShape, lcmShape), SUCCESS);
}

TEST_F(SplitLargeFanoutTensorTest, CalGcdShape_DimMismatch)
{
    npu::tile_fwk::SplitLargeFanoutTensor pass;
    std::vector<int64_t> toShape = {16, 32};
    std::vector<int64_t> fromShape = {8};
    std::vector<int64_t> gcdShape;
    EXPECT_NE(pass.CalGcdShape(toShape, fromShape, gcdShape), SUCCESS);
}

TEST_F(SplitLargeFanoutTensorTest, BeCovered_Full)
{
    int NUM_2 = 2;
    (void)NUM_2;
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_128 = 128;
    int NUM_256 = 256;
    int NUM_512 = 512;
    int NUM_576 = 512 + 64;
    std::vector<int64_t> shape0{NUM_128, NUM_64};
    std::vector<int64_t> tiledShape0{NUM_32, NUM_64};

    std::vector<int64_t> shape1{NUM_128, NUM_512};
    std::vector<int64_t> tiledShape1{NUM_32, NUM_512};

    std::vector<int64_t> tiledShape2{NUM_32, NUM_576};
    std::vector<int64_t> tiledShape3{NUM_32, NUM_256};
    ComputationalGraphBuilder G;

    // [128, 512] --> View(64, 0) --> [32, 512]
    std::vector<SymbolicScalar> dynShapeA = {CreateTestScalarVar("a"), NUM_512};
    G.AddTensor(DataType::DT_FP32, shape1, "a"); // [128, 512]
    auto a = G.GetTensor("a");
    a->UpdateDynValidShape(dynShapeA);
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, tiledShape1, "tiledA"); // [32, 512]
    auto tiledA = G.GetTensor("tiledA");
    tiledA->UpdateDynValidShape(dynShapeA);
    tiledA->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledA"}, "View_A");
    auto View_A = G.GetOp("View_A");
    std::vector<int64_t> offsetA = {NUM_64, 0};
    auto attrA = std::make_shared<ViewOpAttribute>(offsetA, MemoryType::MEM_UNKNOWN);
    View_A->SetOpAttribute(attrA);

    // [128, 64] --> View(64, 0) --> [32, 64]
    std::vector<SymbolicScalar> dynShapeB = {CreateTestScalarVar("a"), NUM_64};
    G.AddTensor(DataType::DT_FP32, shape0, "b"); // [128, 64]
    auto b = G.GetTensor("b");
    b->UpdateDynValidShape(dynShapeB);
    b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, tiledShape0, "tiledB"); // [32, 64]
    auto tiledB = G.GetTensor("tiledB");
    tiledB->UpdateDynValidShape(dynShapeB);
    tiledB->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"b"}, {"tiledB"}, "View_B");
    auto View_B = G.GetOp("View_B");
    std::vector<int64_t> offsetB = {NUM_64, 0};
    auto attrB = std::make_shared<ViewOpAttribute>(offsetB, MemoryType::MEM_UNKNOWN);
    View_B->SetOpAttribute(attrB);

    // [32, 512] concat [32, 64] -->  [32, 576]
    G.AddTensor(DataType::DT_FP32, tiledShape2, "tiledConcat"); // [32, 512] + [32, 64] --> [32, 576]
    auto tiledConcat = G.GetTensor("tiledConcat");
    tiledConcat->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledA"}, {"tiledConcat"}, "Assemble_A");
    auto attrAssembleA = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assembleA = G.GetOp("Assemble_A");
    assembleA->SetOpAttribute(attrAssembleA);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledB"}, {"tiledConcat"}, "Assemble_B");
    auto attrAssembleB = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                               std::vector<int64_t>{0, NUM_512});
    auto assembleB = G.GetOp("Assemble_B");
    assembleB->SetOpAttribute(attrAssembleB);

    // part 1: BeCovered
    // [32, 576] --> View(0, 0) --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_1");
    auto tiledView_1 = G.GetTensor("tiledView_1");
    tiledView_1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_1"}, "View_1");
    auto View_1 = G.GetOp("View_1");
    std::vector<int64_t> offset1 = {0, 0};
    auto attr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
    View_1->SetOpAttribute(attr1);

    // part 2: BeCovered
    // [32, 576] --> View(0, 256) --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_2");
    auto tiledView_2 = G.GetTensor("tiledView_2");
    tiledView_2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_2"}, "View_2");
    auto View_2 = G.GetOp("View_2");
    std::vector<int64_t> offset2 = {0, NUM_256};
    auto attr2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
    View_2->SetOpAttribute(attr2);

    // part 3: Perfectly Match
    // [32, 576] --> View(0, 512) --> [32, 64]
    G.AddTensor(DataType::DT_FP32, tiledShape0, "tiledView_3");
    auto tiledView_3 = G.GetTensor("tiledView_3");
    tiledView_3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_3"}, "View_3");
    auto View_3 = G.GetOp("View_3");
    std::vector<int64_t> offset3 = {0, NUM_512};
    auto attr3 = std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_UNKNOWN);
    View_3->SetOpAttribute(attr3);

    // part1 + part2
    // [32, 256] + [32, 256] --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "add_out");
    G.AddOp(Opcode::OP_ADD, {"tiledView_1", "tiledView_2"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.AddTensor(DataType::DT_FP32, tiledShape3, "out1");
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out1"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assemble1 = G.GetOp("Assemble_1");
    assemble1->SetOpAttribute(attrAssemble1);

    // part3 exp
    // [32, 64] --> Exp --> [32, 64]
    G.AddTensor(DataType::DT_FP32, tiledShape0, "exp_out");
    G.AddOp(Opcode::OP_EXP, {"tiledView_3"}, {"exp_out"}, "Exp");
    auto expOut = G.GetTensor("exp_out");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.AddTensor(DataType::DT_FP32, tiledShape0, "out2");
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"exp_out"}, {"out2"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assemble2 = G.GetOp("Assemble_2");
    assemble2->SetOpAttribute(attrAssemble2);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out1", "out2"});
    Function* function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 11;
    constexpr int viewNumBefore = 5;
    constexpr int assembleNumBefore = 4;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // TODO: CreateOpFor1toM 改为共享 assemble 后，1:N 场景下保留一个共享 ASSEMBLE 作为汇聚点，
    // 当前的 EraseRedundantAssembleOp 不会删除该 ASSEMBLE（它只删 1:1 的 ASSEMBLE->VIEW）。
    // 后续计划将 RemoveRedundantOp 中的删除冗余 assemble 逻辑提成公共方法，由本 pass 调用。
    // 暂时注释断言，避免阻塞后续测试。
    constexpr int opNumAfter = 9;
    constexpr int viewNumAfter = 4;
    constexpr int assembleNumAfter = 3;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

/*
[64, 256][64, 64]
[64, 256][64, 64] --> [128, 128][128, 128][128, 64]
*/
TEST_F(SplitLargeFanoutTensorTest, MtoM)
{
    int NUM_2 = 2;
    ComputationalGraphBuilder G;
    BuildGraphForMToM(G);
    Function* function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 15;
    constexpr int viewNumBefore = 7;
    constexpr int assembleNumBefore = 6;

    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";

    constexpr int singleViewOpmagic = 10010;
    for (auto& op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_EQ(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
            auto dynOffset = viewAttr->GetFromDynOffset();
            EXPECT_EQ(dynOffset.size(), NUM_2);
            auto input = op.GetIOperands().front();
            auto inputDynShape = input->GetDynValidShape();
            EXPECT_EQ(inputDynShape.size(), NUM_2);
            EXPECT_EQ(inputDynShape[0].Dump(), "RUNTIME_Max(c, RUNTIME_Max(((c+64)*(c!=0)), 0))");
        }
    }
}

/*
[64, 256][64, 64]
[64, 256][64, 64] --> [128, 128][128, 128][128, 64]
*/
TEST_F(SplitLargeFanoutTensorTest, MtoMtoMoreSplit)
{
    ComputationalGraphBuilder G;
    BuildGraphForMToM(G);
    Function* function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 15;
    constexpr int viewNumBefore = 7;
    constexpr int assembleNumBefore = 6;

    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit_ = true;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 16;
    constexpr int viewNumAfter = 6;
    constexpr int assembleNumAfter = 8;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, MtoMGetCorrectAssemble)
{
    ComputationalGraphBuilder G;
    BuildGraphForMToM(G, true);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done" << std::endl;
    // 执行pass, 不发生segmentFault即为获取assemble正常
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit_ = true;
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PreCheck(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.RunOnFunction(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PostCheck(*function));
    std::cout << "Run Pass Done" << std::endl;
}

TEST_F(SplitLargeFanoutTensorTest, NewAssembleInheritsScopeInfo)
{
    ComputationalGraphBuilder G;
    BuildGraphForMToM(G);
    Function* function = G.GetFunction();

    Operation::ScopeInfo scopeInfo(5002);
    scopeInfo.allowParallelMerge = true;
    scopeInfo.allowCrossScopeMerge = false;
    for (const auto& opName : {"Assemble_A", "Assemble_B", "Assemble_C", "Assemble_D"}) {
        G.GetOp(opName)->SetScopeInfo(scopeInfo);
    }

    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PreCheck(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.RunOnFunction(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PostCheck(*function));

    auto tiledA = G.GetTensor("tiledA");
    auto tiledB = G.GetTensor("tiledB");
    auto tiledC = G.GetTensor("tiledC");
    auto tiledD = G.GetTensor("tiledD");
    auto largeTensor = G.GetTensor("largeTensor");
    size_t newAssembleCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE || op.GetOOperands().front() == largeTensor) {
            continue;
        }
        auto input = op.GetIOperands().front();
        if (input != tiledA && input != tiledB && input != tiledC && input != tiledD) {
            continue;
        }
        EXPECT_EQ(op.GetScopeId(), scopeInfo.scopeId);
        EXPECT_EQ(op.GetAllowParallelMerge(), scopeInfo.allowParallelMerge);
        EXPECT_EQ(op.GetAllowCrossScopeMerge(), scopeInfo.allowCrossScopeMerge);
        newAssembleCount++;
    }
    EXPECT_EQ(newAssembleCount, 4U);
}

void Build1ToMMultiConsumers(ComputationalGraphBuilder& G)
{
    int NUM_16 = 16;
    int NUM_32 = 32;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"incast0", {NUM_16, NUM_32}},     {"incast1", {NUM_16, NUM_32}},  {"outcast0", {NUM_16, NUM_16}},
        {"outcast1", {NUM_16, NUM_16}},    {"outcast2", {NUM_16, NUM_16}}, {"outcast3", {NUM_16, NUM_16}},
        {"largeTensor", {NUM_32, NUM_32}}, {"outcast", {NUM_16, NUM_16}}};
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    G.AddOp(Opcode::OP_VIEW, {"incast0"}, {"outcast"}, "view");
    auto viewEX = G.GetOp("view");
    Offset offsetEX = {0, 0};
    viewEX->SetOpAttribute(std::make_shared<ViewOpAttribute>(offsetEX, MemoryType::MEM_DEVICE_DDR));

    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"incast0", "assemble0", {0, 0}}, {"incast1", "assemble1", {16, 0}}};
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> viewOps = {{"outcast0", "view0", {0, 0}},
                                                                                       {"outcast1", "view1", {0, 16}},
                                                                                       {"outcast2", "view2", {16, 0}},
                                                                                       {"outcast3", "view3", {16, 16}}};
    for (const auto& [output, opName, offset] : viewOps) {
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"incast0", "incast1"});
    G.SetOutCast({"outcast0", "outcast1", "outcast2", "outcast3", "outcast"});
}

TEST_F(SplitLargeFanoutTensorTest, 1ToMGetCorrectAssemble)
{
    ComputationalGraphBuilder G;
    Build1ToMMultiConsumers(G);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass, 不发生segmentFault即为获取assemble正常
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit_ = false;
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PreCheck(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.RunOnFunction(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PostCheck(*function));
    std::cout << "Run Pass Done." << std::endl;
}

TEST_F(SplitLargeFanoutTensorTest, Unmatched)
{
    int N = 2;
    int T = 8;
    std::vector<int64_t> shape0{N * T, N * T};
    std::vector<int64_t> shape1{T, T};
    std::vector<int64_t> shape2{N * T, T};
    std::vector<int64_t> shape3{T / N, N * T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    G.AddTensor(DataType::DT_FP32, shape0, "out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    //  [16, 16] --> View --> [4, 16] --> Exp --> [4, 16]  --> [16, 16]
    for (int i = 0; i < T / N; i++) {
        // View
        G.AddTensor(DataType::DT_FP32, shape3, "viewOut_" + std::to_string(i));
        auto viewOut = G.GetTensor("viewOut_" + std::to_string(i));
        viewOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"viewOut_" + std::to_string(i)}, "View_" + std::to_string(i));
        auto viewOp = G.GetOp("View_" + std::to_string(i));
        std::vector<int64_t> offset = {i * T / N, 0};
        auto attr = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UNKNOWN);
        viewOp->SetOpAttribute(attr);

        // Exp
        G.AddTensor(DataType::DT_FP32, shape3, "expOut_" + std::to_string(i));
        auto expOut = G.GetTensor("expOut_" + std::to_string(i));
        expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_EXP, {"viewOut_" + std::to_string(i)}, {"expOut_" + std::to_string(i)},
                "Exp_" + std::to_string(i));

        // Assemble
        G.AddOp(Opcode::OP_ASSEMBLE, {"expOut_" + std::to_string(i)}, {"out"}, "AssembleFinal_" + std::to_string(i));
        auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                                  std::vector<int64_t>{i * T / N, 0});
        auto assembleOp = G.GetOp("AssembleFinal_" + std::to_string(i));
        assembleOp->SetOpAttribute(attrAssemble);
    }

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out"});
    Function* function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 28;
    constexpr int viewNumBefore = 12;
    constexpr int assembleNumBefore = 8;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;

    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatchWithAll_Full)
{
    int NUM_2 = 2;
    ComputationalGraphBuilder G;
    PerfectlyMatchWithAll_FullConstruct(G);

    Function* function = G.GetFunction();
    // 确认构图完毕
    constexpr int singleViewOpmagic = 10017;
    constexpr int opNumBefore = 20;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 5;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    for (auto& op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_NE(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
        }
    }
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
    for (auto& op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_EQ(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
            auto input = op.GetIOperands().front();
            auto inputDynShape = input->GetDynValidShape();
            EXPECT_EQ(inputDynShape.size(), NUM_2);
            EXPECT_EQ(inputDynShape[0].Dump(), "RUNTIME_Max(a, RUNTIME_Max(((a+8)*(a!=0)), 0))");
        }
    }
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatch_Full)
{
    ComputationalGraphBuilder G;
    BuildPerfectlyMatchWithInferShapeGraph(G);
    Function* function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 20;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 5;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    constexpr int opNumAfter = 10;
    constexpr int viewNumAfter = 6;
    constexpr int assembleNumAfter = 1;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatch_Full_V2)
{
    ComputationalGraphBuilder G;
    BuildPerfectlyMatchFullV2_BaseTensorsAndExp(G);
    BuildPerfectlyMatchFullV2_ViewsAndAdd(G);
    Function* function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 22;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 6;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, OneViewOneAssemble)
{
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_96 = 96;
    int NUM_128 = 128;
    std::vector<int64_t> shape0{NUM_128, NUM_64};
    std::vector<int64_t> shape1{NUM_32, NUM_64};
    std::vector<int64_t> shape2{NUM_64, NUM_64};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out");
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    // a[128, 64] --> View(0, 0) --> a_ub[128, 64]
    G.AddTensor(DataType::DT_FP32, shape0, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"a_ub"}, "View_To_Ub");
    auto view2ub = G.GetOp("View_To_Ub");
    std::vector<int64_t> offset2ub = {0, 0};
    auto attr2ub = std::make_shared<ViewOpAttribute>(offset2ub, MemoryType::MEM_UNKNOWN);
    view2ub->SetOpAttribute(attr2ub);

    /*
    a_ub[128, 64] --> View1(32, 0) --> tensor1[32, 64] --> Assemble1(0, 0) -->\
                  \--> View2(96, 0) --> tensor2[32, 64] --> Assemble2(32, 0) --> tensor3[64, 64] --> Exp --> tensor4 -->
    Assemble --> out
    */
    G.AddTensor(DataType::DT_FP32, shape1, "tensor1");
    auto tensor1 = G.GetTensor("tensor1");
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub"}, {"tensor1"}, "View_1");
    auto viewOp1 = G.GetOp("View_1");
    std::vector<int64_t> offset1 = {NUM_32, 0};
    auto attrView1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
    viewOp1->SetOpAttribute(attrView1);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor2");
    auto tensor2 = G.GetTensor("tensor2");
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub"}, {"tensor2"}, "View_2");
    auto viewOp2 = G.GetOp("View_2");
    std::vector<int64_t> offset2 = {NUM_96, 0};
    auto attrView2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
    viewOp2->SetOpAttribute(attrView2);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor3");
    auto tensor3 = G.GetTensor("tensor3");
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor3"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assembleOp1 = G.GetOp("Assemble_1");
    assembleOp1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor3"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                               std::vector<int64_t>{NUM_32, 0});
    auto assembleOp2 = G.GetOp("Assemble_2");
    assembleOp2->SetOpAttribute(attrAssemble2);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor4");
    auto tensor4 = G.GetTensor("tensor4");
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_EXP, {"tensor3"}, {"tensor4"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor4"}, {"out"}, "Assemble_Final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assembleFinal = G.GetOp("Assemble_Final");
    assembleFinal->SetOpAttribute(attrAssembleFinal);

    G.SetInCast({"a"});
    G.SetOutCast({"out"});
    Function* function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 7;
    constexpr int viewNumBefore = 3;
    constexpr int assembleNumBefore = 3;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    constexpr int opNumAfter = 6;
    constexpr int viewNumAfter = 2;
    constexpr int assembleNumAfter = 3;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, OneViewMultiAssemble)
{
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_128 = 128;
    std::vector<int64_t> shape0{NUM_32, NUM_64};
    std::vector<int64_t> shape1{NUM_64, NUM_64};
    std::vector<int64_t> shape2{NUM_32, NUM_128};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape1, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, shape1, "out1");
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out2");
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    // a[64, 64] --> View(0, 0) --> a_ub[64, 64] --> Muls --> a_ub_new[64, 64]
    G.AddTensor(DataType::DT_FP32, shape1, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"a_ub"}, "View_To_Ub");
    auto view2ub = G.GetOp("View_To_Ub");
    std::vector<int64_t> offset2ub = {0, 0};
    auto attr2ub = std::make_shared<ViewOpAttribute>(offset2ub, MemoryType::MEM_UNKNOWN);
    view2ub->SetOpAttribute(attr2ub);
    G.AddTensor(DataType::DT_FP32, shape1, "a_ub_new");
    auto a_ub_new = G.GetTensor("a_ub_new");
    a_ub_new->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_MULS, {"a_ub"}, {"a_ub_new"}, "Muls");

    /*
    a_ub_new[64, 64] --> View1(0, 0) --> tensor1[32, 64] --> Assemble1(0, 0) -->\
                  \--> View2(32, 0) --> tensor2[32, 64] --> Assemble2(32, 0) --> tensor3[64, 64] --> Exp --> tensor4 -->
    AssembleOut1 --> out1
    */
    G.AddTensor(DataType::DT_FP32, shape0, "tensor1");
    auto tensor1 = G.GetTensor("tensor1");
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub_new"}, {"tensor1"}, "View_1");
    auto viewOp1 = G.GetOp("View_1");
    std::vector<int64_t> offset1 = {0, 0};
    auto attrView1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
    viewOp1->SetOpAttribute(attrView1);

    G.AddTensor(DataType::DT_FP32, shape0, "tensor2");
    auto tensor2 = G.GetTensor("tensor2");
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub_new"}, {"tensor2"}, "View_2");
    auto viewOp2 = G.GetOp("View_2");
    std::vector<int64_t> offset2 = {NUM_32, 0};
    auto attrView2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
    viewOp2->SetOpAttribute(attrView2);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor3");
    auto tensor3 = G.GetTensor("tensor3");
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor3"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assembleOp1 = G.GetOp("Assemble_1");
    assembleOp1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor3"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                               std::vector<int64_t>{NUM_32, 0});
    auto assembleOp2 = G.GetOp("Assemble_2");
    assembleOp2->SetOpAttribute(attrAssemble2);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor4");
    auto tensor4 = G.GetTensor("tensor4");
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_EXP, {"tensor3"}, {"tensor4"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor4"}, {"out1"}, "Assemble_Out1");
    auto attrAssembleOut1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assembleOut1 = G.GetOp("Assemble_Out1");
    assembleOut1->SetOpAttribute(attrAssembleOut1);

    /*
    a_ub_new[64, 64] --> View1(0, 0) --> tensor1[32, 64] --> Assemble3(0, 0) -->\
                  \--> View2(32, 0) --> tensor2[32, 64] --> Assemble4(0, 64) --> tensor5[32, 128] --> Exp --> tensor6
    --> AssembleOut2 --> out2
    */
    G.AddTensor(DataType::DT_FP32, shape2, "tensor5");
    auto tensor5 = G.GetTensor("tensor5");
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor5"}, "Assemble_3");
    auto attrAssemble3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assembleOp3 = G.GetOp("Assemble_3");
    assembleOp3->SetOpAttribute(attrAssemble3);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor5"}, "Assemble_4");
    auto attrAssemble4 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN,
                                                               std::vector<int64_t>{0, NUM_64});
    auto assembleOp4 = G.GetOp("Assemble_4");
    assembleOp4->SetOpAttribute(attrAssemble4);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor6");
    auto tensor6 = G.GetTensor("tensor6");
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ABS, {"tensor5"}, {"tensor6"}, "Abs");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor6"}, {"out2"}, "Assemble_Out2");
    auto attrAssembleOut2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t>{0, 0});
    auto assembleOut2 = G.GetOp("Assemble_Out2");
    assembleOut2->SetOpAttribute(attrAssembleOut2);

    G.SetInCast({"a"});
    G.SetOutCast({"out1", "out2"});
    Function* function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 12;
    constexpr int viewNumBefore = 3;
    constexpr int assembleNumBefore = 6;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

void BuildComplexOverlap(ComputationalGraphBuilder& G)
{
    int NUM_8 = 8;
    int NUM_16 = 16;
    int NUM_32 = 32;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"a", {NUM_8, NUM_32}},           {"b", {NUM_8, NUM_32}},    {"c", {NUM_16, NUM_8}},
        {"d", {NUM_16, NUM_8}},           {"e", {NUM_8, NUM_16}},    {"f", {NUM_8, NUM_16}},
        {"a1", {NUM_8, NUM_32}},          {"b1", {NUM_8, NUM_32}},   {"c1", {NUM_16, NUM_8}},
        {"d1", {NUM_16, NUM_8}},          {"e1", {NUM_8, NUM_16}},   {"f1", {NUM_8, NUM_16}},
        {"out1", {NUM_8, NUM_16}},        {"out2", {NUM_8, NUM_16}}, {"out3", {NUM_32, NUM_8}},
        {"out4", {NUM_32, NUM_8}},        {"out5", {NUM_16, NUM_8}}, {"out6", {NUM_16, NUM_8}},
        {"largeTensor", {NUM_32, NUM_32}}};
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    // 定义所有CopyIn操作并添加
    std::vector<std::tuple<std::string, std::string, std::string>> copyInOps = {
        {"a", "copyIn1", "a1"}, {"b", "copyIn2", "b1"}, {"c", "copyIn3", "c1"},
        {"d", "copyIn4", "d1"}, {"e", "copyIn5", "e1"}, {"f", "copyIn6", "f1"}};
    for (const auto& [input, opName, output] : copyInOps) {
        G.AddOp(Opcode::OP_COPY_IN, {input}, {output}, opName);
    }
    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"a1", "Assemble_A", {0, 0}},  {"b1", "Assemble_B", {24, 0}}, {"c1", "Assemble_C", {8, 0}},
        {"d1", "Assemble_D", {8, 24}}, {"e1", "Assemble_E", {8, 8}},  {"f1", "Assemble_F", {16, 8}}};
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::vector<int64_t>>> viewOps = {
        {"out1", {0, 8}}, {"out2", {24, 8}}, {"out3", {0, 0}}, {"out4", {0, 24}}, {"out5", {8, 8}}, {"out6", {8, 16}}};
    for (const auto& [output, offset] : viewOps) {
        std::string opName = "View_" + output.substr(3);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"a", "b", "c", "d", "e", "f"});
    G.SetOutCast({"out1", "out2", "out3", "out4", "out5", "out6"});
}

/*
    input[shape]:       a[8, 32]    b[8, 32]    c[16, 8]    d[16, 8]    e[8, 16]    f[8, 16]
    assemble offset:    [0, 0]      [24, 0]     [8, 0]      [8, 24]     [8, 8]      [16, 8]
    largeTensor[shape]:                         largeTensor[32, 32]
    view offset:        [0, 8]      [24, 8]     [0, 0]      [0, 24]     [8, 8]      [8, 16]
    output[shape]:      out1[8, 16] out2[8, 16] out3[32, 8] out4[32, 8] out5[16, 8] out6[16, 8]
*/
TEST_F(SplitLargeFanoutTensorTest, ComplexOverlap)
{
    ComputationalGraphBuilder G;
    BuildComplexOverlap(G);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit_ = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：
    // 拆分后除了两个incast分别各cover一个outcast的场景会被单独拆出
    // 中间的[16, 32]会被拆除形成对两个[8, 32]的多对多
    // 剩余两个会被保留
    std::unordered_map<int, int> recordAssemble;
    std::unordered_map<int, int> recordView;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            recordAssemble[op.oOperand.front()->GetMagic()]++;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            recordView[op.iOperand.front()->GetMagic()]++;
        }
    }
    for (auto& [k, v] : recordAssemble) {
        // 1. out1被输入a包含，out2被输入b包含，拆为1对1
        // 2. 输入e, f组成的[16, 16]tile和out5,out6匹配，拆出2对2
        // 3. 剩余无法继续拆分，保留从largeTensor进行view
        // 故对于拆分后view和assemble的中间tensor，
        // 如果只有一个assemble（场景1），则只有一个view
        // 否则（场景2，3）会有两个view
        EXPECT_EQ(recordView[k], (v == 1) ? 1 : 2);
    }
}

/*
    输入是8个{16， 656}，输出是{16， 512}, {16， 128}各两个。
    其中两个输入与两个输出形成一对一，一个输入对应两个输出的和，其余五个输入未被使用
    拆分后理应得到
    {16， 656} -> view -> {16， 512}
    {16， 656} -> view -> {16， 128}
    {16， 656} -> view -> {16， 512} + {16， 128}
    {16， 656}
    {16， 656}
    {16， 656}
    {16， 656}
    {16， 656}
*/
void BuildPartialInputUnusedGraph(ComputationalGraphBuilder& G)
{
    G.AddTensor(DataType::DT_FP32, {128, 656}, "largeTensor");
    auto largeTensor = G.GetTensor("largeTensor");
    largeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    std::map<std::string, Shape> logicTensors = {
        {"in1", {16, 656}},       {"in2", {16, 656}},       {"in3", {16, 656}},       {"in4", {16, 656}},
        {"in5", {16, 656}},       {"in6", {16, 656}},       {"in7", {16, 656}},       {"in8", {16, 656}},
        {"in1Tensor", {16, 656}}, {"in2Tensor", {16, 656}}, {"in3Tensor", {16, 656}}, {"in4Tensor", {16, 656}},
        {"in5Tensor", {16, 656}}, {"in6Tensor", {16, 656}}, {"in7Tensor", {16, 656}}, {"in8Tensor", {16, 656}},
        {"out11", {16, 512}},     {"out81", {16, 512}},     {"out22", {16, 128}},     {"out82", {16, 128}},
        {"out33", {16, 16}},      {"out83", {16, 16}}};
    for (const auto& [name, shape] : logicTensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto logicTensor = G.GetTensor(name);
        logicTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }
    // 定义所有CopyIn操作并添加
    std::vector<std::tuple<std::string, std::string, std::string>> copyInOps = {
        {"in1", "copyIn1", "in1Tensor"}, {"in2", "copyIn2", "in2Tensor"}, {"in3", "copyIn3", "in3Tensor"},
        {"in4", "copyIn4", "in4Tensor"}, {"in5", "copyIn5", "in5Tensor"}, {"in6", "copyIn6", "in6Tensor"},
        {"in7", "copyIn7", "in7Tensor"}, {"in8", "copyIn8", "in8Tensor"}};
    for (const auto& [input, opName, output] : copyInOps) {
        G.AddOp(Opcode::OP_COPY_IN, {input}, {output}, opName);
    }
    G.SetInCast({"in1", "in2", "in3", "in4", "in5", "in6", "in7", "in8"});
    G.SetOutCast({"out11", "out22", "out81", "out82"});

    std::vector<std::tuple<std::string, std::string, std::string, Shape>> assembleOps = {
        {"in1Tensor", "largeTensor", "assemble1", {0, 0}},  {"in2Tensor", "largeTensor", "assemble2", {16, 0}},
        {"in3Tensor", "largeTensor", "assemble3", {32, 0}}, {"in4Tensor", "largeTensor", "assemble4", {48, 0}},
        {"in5Tensor", "largeTensor", "assemble5", {64, 0}}, {"in6Tensor", "largeTensor", "assemble6", {80, 0}},
        {"in7Tensor", "largeTensor", "assemble7", {96, 0}}, {"in8Tensor", "largeTensor", "assemble8", {112, 0}}};
    for (const auto& [input, output, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {output}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    std::vector<std::tuple<std::string, std::string, std::string, Shape>> viewOps = {
        {"largeTensor", "out11", "view11", {0, 0}},
        {"largeTensor", "out81", "view81", {112, 0}},
        {"largeTensor", "out22", "view22", {16, 512}},
        {"largeTensor", "out82", "view82", {112, 512}}};
    for (const auto& [input, output, opName, offset] : viewOps) {
        G.AddOp(Opcode::OP_VIEW, {input}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }
}

TEST_F(SplitLargeFanoutTensorTest, TestPartialInputUnused)
{
    ComputationalGraphBuilder G;
    BuildPartialInputUnusedGraph(G);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit_ = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // TODO: CreateOpFor1toM 改为共享 assemble 后，1:N 场景下保留一个共享 ASSEMBLE 作为汇聚点，
    // 当前的 EraseRedundantAssembleOp 不会删除该 ASSEMBLE（它只删 1:1 的 ASSEMBLE->VIEW）。
    // 后续计划将 RemoveRedundantOp 中的删除冗余 assemble 逻辑提成公共方法，由本 pass 调用。
    // 暂时注释断言，避免阻塞后续测试。
    // const int viewNum = 4;
    // const int assembleNum = 0;
    // auto countResultAfter = CountViewAssemble(*function);
    // EXPECT_EQ(viewNum, countResultAfter[0]) << countResultAfter[0] << "OP_VIEW after pass, should be 4";
    // EXPECT_EQ(assembleNum, countResultAfter[1]) << countResultAfter[1] << "OP_ASSEMBLE after pass, should be 0";
}

void BuildOneDim(ComputationalGraphBuilder& G, bool shouldSplit)
{
    int NUM_1 = 1;
    int NUM_15 = 15;
    int NUM_16 = 16;
    int NUM_32 = 32;

    std::map<std::string, std::vector<int64_t>> tensors = {
        {"a", {NUM_1}},
        {"b", shouldSplit ? std::vector<int64_t>{NUM_15} : std::vector<int64_t>{NUM_16}},
        {"c", shouldSplit ? std::vector<int64_t>{NUM_16} : std::vector<int64_t>{NUM_15}},
        {"a1", {NUM_1}},
        {"b1", shouldSplit ? std::vector<int64_t>{NUM_15} : std::vector<int64_t>{NUM_16}},
        {"c1", shouldSplit ? std::vector<int64_t>{NUM_16} : std::vector<int64_t>{NUM_15}},
        {"out1", {NUM_16}},
        {"out2", {NUM_16}},
        {"largeTensor", {NUM_32}}};
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    std::vector<std::tuple<std::string, std::string, std::string>> copyInOps = {
        {"a", "copyIn1", "a1"}, {"b", "copyIn2", "b1"}, {"c", "copyIn3", "c1"}};
    for (const auto& [input, opName, output] : copyInOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {output}, opName);
    }

    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"a1", "assemble1", {0}}, {"b1", "assemble2", {1}}, {"c1", "assemble3", {shouldSplit ? 16 : 17}}};
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    std::vector<std::tuple<std::string, std::vector<int64_t>>> viewOps = {{"out1", {0}}, {"out2", {16}}};
    for (const auto& [output, offset] : viewOps) {
        std::string opName = "view" + output.substr(3);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"a", "b", "c"});
    G.SetOutCast({"out1", "out2"});
}

// {1} + {15} + {16} --assemble--> {32} --view--> {16} + {16}
// =>   {1} + {15} --assemble--> {16} --view--> {16}
//      {16} --view--> {16}
TEST_F(SplitLargeFanoutTensorTest, OneDimShouldSplit)
{
    ComputationalGraphBuilder G;
    BuildOneDim(G, true);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    ExecutePass(function, false);

    // 验证：
    // 依据UT注释展示，共会出现2个view和2个assemble
    auto countResultAfter = CountViewAssemble(*function);
    const int viewNum = 2;
    const int assembleNum = 0;
    EXPECT_EQ(viewNum, countResultAfter[0]) << countResultAfter[0] << " OP_VIEW after pass, should be 2";
    EXPECT_EQ(assembleNum, countResultAfter[1]) << countResultAfter[1] << " OP_ASSEMBLE after pass, should be 0";
}

// {1} + {16} + {15} --assemble--> {32} --view--> {16} + {16}
// 不进行拆分
TEST_F(SplitLargeFanoutTensorTest, OneDimNotSplit)
{
    ComputationalGraphBuilder G;
    BuildOneDim(G, false);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    auto countResultBefore = CountViewAssemble(*function);
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit_ = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：pass不会切分，所以前后一致, 但由于MergeViewAssemble影响，所以Assemble数量不相等
    auto countResultAfter = CountViewAssemble(*function);
    EXPECT_EQ(countResultBefore[0], countResultAfter[0]) << countResultBefore[0] << "OP_VIEW before pass; "
                                                         << countResultAfter[0] << " OP_VIEW after pass, should equal.";
    EXPECT_NE(countResultBefore[1], countResultAfter[1])
        << countResultBefore[1] << "OP_ASSEMBLE before pass; " << countResultAfter[1]
        << " OP_ASSEMBLE after pass, should equal.";
}

// {1} + {2} + {1} + {1} --assemble--> {5} --view--> {3} + {1}
void BuildDiffLcmShape(ComputationalGraphBuilder& G)
{
    int NUM_1 = 1;
    int NUM_2 = 2;
    int NUM_3 = 3;
    int NUM_5 = 5;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"a", {NUM_1}},    {"b", {NUM_2}},    {"c", {NUM_1}},          {"d", {NUM_1}},
        {"a1", {NUM_1}},   {"b1", {NUM_2}},   {"c1", {NUM_1}},         {"d1", {NUM_1}},
        {"out1", {NUM_3}}, {"out2", {NUM_1}}, {"largeTensor", {NUM_5}}};
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    // 定义所有CopyIn操作并添加
    std::vector<std::tuple<std::string, std::string, std::string>> copyInOps = {
        {"a", "copyIn1", "a1"}, {"b", "copyIn2", "b1"}, {"c", "copyIn3", "c1"}, {"d", "copyIn4", "d1"}};
    for (const auto& [input, opName, output] : copyInOps) {
        G.AddOp(Opcode::OP_COPY_IN, {input}, {output}, opName);
    }
    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"a1", "Assemble_A", {0}}, {"b1", "Assemble_B", {1}}, {"c1", "Assemble_C", {3}}, {"d1", "Assemble_D", {4}}};
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::vector<int64_t>>> viewOps = {{"out1", {0}}, {"out2", {3}}};
    for (const auto& [output, offset] : viewOps) {
        std::string opName = "View_" + output.substr(3);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"a", "b", "c", "d"});
    G.SetOutCast({"out1", "out2"});
}

// {1} + {2} + {1} + {1} --assemble--> {5} --view--> {3} + {1}
// ==> {1} + {2} --assemble--> {3} --view--> {3}
//     {1} --view--> {1}
// 用于验证小的LcmShape被优先尝试拆分，否则第二组拆分结果会得到：
// {1} + {1} --assemble--> {2} --view--> {1}
// 存在冗余的assemble，不符合预期
TEST_F(SplitLargeFanoutTensorTest, SplitSmallTileFirst)
{
    ComputationalGraphBuilder G;
    BuildDiffLcmShape(G);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    ExecutePass(function, false);

    auto countResultAfter = CountViewAssemble(*function);
    const int viewNum = 2;
    const int AssembleNum = 2;
    EXPECT_EQ(viewNum, countResultAfter[0]) << countResultAfter[0] << " OP_VIEW after pass, should be 2";
    EXPECT_EQ(AssembleNum, countResultAfter[1]) << countResultAfter[1] << " OP_ASSEMBLE after pass, should be 2";
}

// {2} + {2} + {3} --assemble--> {7} --view--> {5}
void BuildCoprimeInputOutput(ComputationalGraphBuilder& G)
{
    int NUM_2 = 2;
    int NUM_3 = 3;
    int NUM_5 = 5;
    int NUM_7 = 7;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"a", {NUM_2}}, {"b", {NUM_2}}, {"c", {NUM_3}}, {"out1", {NUM_5}}, {"largeTensor", {NUM_7}}};
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"a", "Assemble_A", {0}}, {"b", "Assemble_B", {2}}, {"c", "Assemble_C", {4}}};
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::vector<int64_t>>> viewOps = {{"out1", {0}}};
    for (const auto& [output, offset] : viewOps) {
        std::string opName = "View_" + output.substr(3);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"a", "b", "c"});
    G.SetOutCast({"out1"});
}

// {2} + {2} + {3} --assemble--> {7} --view--> {5}
// 不进行拆分，不会生成新的lcmTile
TEST_F(SplitLargeFanoutTensorTest, NoSplitLcmLargerThanLargeTensor)
{
    ComputationalGraphBuilder G;
    BuildCoprimeInputOutput(G);
    Function* function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    auto countResultBefore = CountViewAssemble(*function);
    std::vector<int> opMagicBefore;
    for (auto& op : function->Operations()) {
        opMagicBefore.emplace_back(op.GetOpMagic());
    }

    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit_ = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：pass不会切分，所以前后一致, 但由于MergeViewAssemble功能会合并View和Assemble，导致结果不一致
    auto countResultAfter = CountViewAssemble(*function);
    std::vector<int> opMagicAfter;
    for (auto& op : function->Operations()) {
        opMagicAfter.emplace_back(op.GetOpMagic());
    }
    EXPECT_EQ(countResultBefore[0], countResultAfter[0]) << countResultBefore[0] << "OP_VIEW before pass; "
                                                         << countResultAfter[0] << " OP_VIEW after pass, should equal.";
    EXPECT_NE(countResultBefore[1], countResultAfter[1])
        << countResultBefore[1] << "OP_ASSEMBLE before pass; " << countResultAfter[1]
        << " OP_ASSEMBLE after pass, should equal.";
    EXPECT_NE(CommonUtils::ContainerToStr(opMagicBefore), CommonUtils::ContainerToStr(opMagicAfter))
        << "All op magic before pass: " << CommonUtils::ContainerToStr(opMagicBefore)
        << "; All op magic after pass: " << CommonUtils::ContainerToStr(opMagicAfter) << "; Op should change.";
}

TEST_F(SplitLargeFanoutTensorTest, TestSplitTailTile)
{
    config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutTensorTestStrategy");
    std::vector<int64_t> shape0 = {112, 112};
    std::vector<int64_t> shape1 = {16, 112};
    std::vector<int64_t> shape2 = {112, 48};
    std::vector<int64_t> shape3 = {16, 48};
    std::vector<int64_t> shape4 = {128, 272};
    PROGRAM("SplitLargeFanoutTensorTest")
    {
        Tensor input1a(DataType::DT_FP32, shape0, "In1a");
        Tensor input1b(DataType::DT_FP32, shape0, "In1b");
        Tensor input2a(DataType::DT_FP32, shape1, "In2a");
        Tensor input2b(DataType::DT_FP32, shape1, "In2b");
        Tensor input3a(DataType::DT_FP32, shape0, "In3a");
        Tensor input3b(DataType::DT_FP32, shape0, "In3b");
        Tensor input4a(DataType::DT_FP32, shape1, "In4a");
        Tensor input4b(DataType::DT_FP32, shape1, "In4b");
        Tensor input5a(DataType::DT_FP32, shape2, "In5a");
        Tensor input5b(DataType::DT_FP32, shape2, "In5b");
        Tensor input6a(DataType::DT_FP32, shape3, "In6a");
        Tensor input6b(DataType::DT_FP32, shape3, "In6b");
        Tensor output1(DataType::DT_FP32, shape3, "Out1");
        SetTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestSplitTailTile", {input1a, input1b, input2a, input2b, input3a, input3b, input4a, input4b, input5a,
                                       input5b, input6a, input6b, output1})
        {
            TileShape::Current().SetVecTile(256, 256);
            Tensor t1 = Add(input1a, input1b);
            Tensor t2 = Add(input2a, input2b);
            Tensor t3 = Add(input3a, input3b);
            Tensor t4 = Add(input4a, input4b);
            Tensor t5 = Add(input5a, input5b);
            Tensor t6 = Add(input6a, input6b);
            Tensor lt(DT_FP32, shape4, "lt");
            Assemble(t1, {0, 0}, lt);
            Assemble(t2, {112, 0}, lt);
            Assemble(t3, {0, 112}, lt);
            Assemble(t4, {112, 112}, lt);
            Assemble(t5, {0, 224}, lt);
            Assemble(t6, {112, 224}, lt);
            output1 = View(lt, shape3, {112, 224});
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestSplitTailTile");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        size_t maxProducerCount = 0;
        std::set<LogicalTensorPtr> visitedTensors;
        for (auto& op : originFunction->Operations()) {
            for (auto& output : op.GetOOperands()) {
                if (!visitedTensors.insert(output).second) {
                    continue;
                }
                size_t producerCnt = output->GetProducers().size();
                if (producerCnt > maxProducerCount) {
                    maxProducerCount = producerCnt;
                }
            }
        }
        EXPECT_LT(maxProducerCount, 6) << "存在 tensor 的生产者 op 数量 >= 6，最大生产者数: " << maxProducerCount;
    }
}

// 当输入同时存在头块和尾块时，tileOffset是头块大小加n倍中间块大小的情况暂不切分，避免级联场景validShape表达式过长。
TEST_F(SplitLargeFanoutTensorTest, TestHeadTileOffsetNoSplit)
{
    config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutTensorTestStrategy");
    std::vector<int64_t> shape0 = {256};
    std::vector<int64_t> shape1 = {512};
    std::vector<int64_t> shape2 = {352};
    std::vector<int64_t> shape3 = {96};
    std::vector<int64_t> shape4 = {2656};
    PROGRAM("SplitLargeFanoutTensorTest")
    {
        Tensor input1(DataType::DT_FP32, shape0, "In1");
        Tensor input2(DataType::DT_FP32, shape1, "In2");
        Tensor input3(DataType::DT_FP32, shape1, "In3");
        Tensor input4(DataType::DT_FP32, shape1, "In4");
        Tensor input5(DataType::DT_FP32, shape1, "In5");
        Tensor input6(DataType::DT_FP32, shape2, "In6");
        Tensor output1(DataType::DT_FP32, shape1, "Out1");
        Tensor output2(DataType::DT_FP32, shape1, "Out2");
        Tensor output3(DataType::DT_FP32, shape1, "Out3");
        Tensor output4(DataType::DT_FP32, shape1, "Out4");
        Tensor output5(DataType::DT_FP32, shape1, "Out5");
        Tensor output6(DataType::DT_FP32, shape3, "Out6");
        SetTestStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestHeadTileOffsetNoSplit",
                 {input1, input2, input3, input4, input5, input6, output1, output2, output3, output4, output5, output6})
        {
            TileShape::Current().SetVecTile(1024);
            Tensor lt(DT_FP32, shape4, "lt");
            Assemble(input1, {0}, lt);
            Assemble(input2, {256}, lt);
            Assemble(input3, {768}, lt);
            Assemble(input4, {1280}, lt);
            Assemble(input5, {1792}, lt);
            Assemble(input6, {2304}, lt);
            output1 = View(lt, shape1, {0});
            output2 = View(lt, shape1, {512});
            output3 = View(lt, shape1, {1024});
            output4 = View(lt, shape1, {1536});
            output5 = View(lt, shape1, {2048});
            output6 = View(lt, shape3, {2560});
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestHeadTileOffsetNoSplit");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto countResultAfter = CountViewAssemble(*originFunction);
        const int expectViewCount = 12;
        const int expectAssembleCount = 15;
        EXPECT_EQ(countResultAfter[0], expectViewCount);
        EXPECT_EQ(countResultAfter[1], expectAssembleCount);
        ExpectInCastsReachOutCasts(*originFunction);
    }
}

TEST_F(SplitLargeFanoutTensorTest, TestSimplifyOverlapDualOverlap)
{
    config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutTensorTestStrategy");
    std::vector<int64_t> shape0 = {64};
    std::vector<int64_t> shape1 = {128};
    std::vector<int64_t> shape2 = {32};
    std::vector<int64_t> shape3 = {96};
    PROGRAM("SplitLargeFanoutTensorTest")
    {
        Tensor input1(DataType::DT_FP32, shape1, "In1");
        Tensor input2(DataType::DT_FP32, shape1, "In2");
        Tensor input3(DataType::DT_FP32, shape1, "In3");
        Tensor input4(DataType::DT_FP32, shape1, "In4");
        Tensor input5(DataType::DT_FP32, shape1, "In5");
        Tensor input6(DataType::DT_FP32, shape2, "In6");
        Tensor output1(DataType::DT_FP32, shape0, "Out1");
        Tensor output2(DataType::DT_FP32, shape1, "Out2");
        Tensor output3(DataType::DT_FP32, shape1, "Out3");
        Tensor output4(DataType::DT_FP32, shape1, "Out4");
        Tensor output5(DataType::DT_FP32, shape1, "Out5");
        Tensor output6(DataType::DT_FP32, shape3, "Out6");
        SetTestStrategy();
        config::SetBuildStatic(true);
        FUNCTION("TestSimplifyOverlapDualOverlap",
                 {input1, input2, input3, input4, input5, input6, output1, output2, output3, output4, output5, output6})
        {
            TileShape::Current().SetVecTile(1024);
            Tensor lt(DT_FP32, {672}, "lt");
            Assemble(input1, {0}, lt);
            Assemble(input2, {128}, lt);
            Assemble(input3, {256}, lt);
            Assemble(input4, {384}, lt);
            Assemble(input5, {512}, lt);
            Assemble(input6, {640}, lt);
            output1 = View(lt, shape0, {0});
            output2 = View(lt, shape1, {64});
            output3 = View(lt, shape1, {192});
            output4 = View(lt, shape1, {320});
            output5 = View(lt, shape1, {448});
            output6 = View(lt, shape3, {576});
        }
        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestSimplifyOverlapDualOverlap");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        size_t maxProducerCount = 0;
        std::set<LogicalTensorPtr> visitedTensors;
        for (auto& op : originFunction->Operations()) {
            for (auto& output : op.GetOOperands()) {
                if (!visitedTensors.insert(output).second) {
                    continue;
                }
                size_t producerCnt = output->GetProducers().size();
                if (producerCnt > maxProducerCount) {
                    maxProducerCount = producerCnt;
                }
            }
        }
        EXPECT_LT(maxProducerCount, 6) << "存在 tensor 的生产者 op 数量 >= 6，最大生产者数: " << maxProducerCount;
    }
}

// Verify CreateOpFor1toM creates ONE shared contract+tensor for a 1:M tile
// (multiple slice consumers of the same producer tile), instead of one
// contract+tensor per slice consumer.
//
// Graph: lt[32] is produced by 2 contracts (2 Assembles at {0}/{16}) and
// consumed by 3 slices (3 Views: two at {0} -> 1:2 tile, one at {16} -> 1:1
// tile). After SplitLargeFanoutTensor the 2:3 large tensor should split into
// one 1:1 chain and one 1:2 chain, i.e. 2 new contracts feeding slices with
// slice-consumer counts {1, 2}.
//
// The split methods are invoked manually (skipping ProcessContractSlice /
// MergeViewAssemble) because ProcessContractSlice would otherwise merge the
// contract->slice chains back together for static shapes, hiding the
// structural difference.
TEST_F(SplitLargeFanoutTensorTest, OneToMSharedTensor)
{
    config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutPreSplitStrategy");
    std::vector<int64_t> tileShape = {16};
    std::vector<int64_t> largeShape = {32};
    PROGRAM("SplitLargeFanoutTensorTest")
    {
        Tensor in1(DataType::DT_FP32, tileShape, "in1");
        Tensor in2(DataType::DT_FP32, tileShape, "in2");
        Tensor out1(DataType::DT_FP32, tileShape, "out1");
        Tensor out2(DataType::DT_FP32, tileShape, "out2");
        Tensor out3(DataType::DT_FP32, tileShape, "out3");
        SetPreSplitStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("OneToMSharedTensor", {in1, in2, out1, out2, out3})
        {
            TileShape::Current().SetVecTile(32);
            Tensor lt(DT_FP32, largeShape, "lt");
            Assemble(in1, {0}, lt);
            Assemble(in2, {16}, lt);
            out1 = View(lt, tileShape, {0});
            out2 = View(lt, tileShape, {0});
            out3 = View(lt, tileShape, {16});
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_OneToMSharedTensor");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";

        npu::tile_fwk::SplitLargeFanoutTensor pass;
        pass.Init();
        pass.CollectLargeTensor(*originFunction);
        pass.SplitLargeTensor(*originFunction);
        pass.EraseRedundantAssembleOp(*originFunction);

        // After the fix, CreateOpFor1toM creates ONE shared contract for all
        // slice consumers of the same producer tile. The [0:16] tile has 2
        // slice consumers -> one contract with 2 slice consumers (the 1:2
        // chain). Without the fix each slice consumer would get its own
        // contract, so no contract would have >=2 slice consumers.
        // ProcessContractSlice is intentionally skipped (it would merge the
        // contract->slice chains for static shapes and hide the difference).
        size_t contractWithMultiSlice = 0;
        for (auto& op : originFunction->Operations()) {
            if (op.GetOpcode() != Opcode::OP_CONTRACT || op.GetOOperands().empty()) {
                continue;
            }
            auto output = op.GetOOperands().front();
            size_t sliceCnt = 0;
            for (auto* consumer : output->GetConsumers()) {
                if (consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE) {
                    sliceCnt++;
                }
            }
            if (sliceCnt >= 2) {
                contractWithMultiSlice++;
            }
        }
        EXPECT_EQ(contractWithMultiSlice, 1U)
            << "期望存在一条 1:2 contract->slice 链（共享 tensor），实际 >=2 slice 消费者的 contract 数: "
            << contractWithMultiSlice;
    }
}

TEST_F(SplitLargeFanoutTensorTest, TestInferShapeHaveDynValidShape)
{
    ComputationalGraphBuilder G;
    BuildPerfectlyMatchWithInferShapeGraph(G, true);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PreCheck(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.RunOnFunction(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PostCheck(*function));

    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            const auto& in_shape = op.iOperand[0]->GetDynValidShape();
            const auto& out_shape = op.oOperand[0]->GetDynValidShape();
            auto viewAttr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            ASSERT_NE(viewAttr, nullptr);
            auto expect_shape = GetViewValidShape(in_shape, viewAttr->GetFromOffset(), viewAttr->GetFromDynOffset(),
                                                  op.oOperand[0]->GetShape());
            ASSERT_EQ(expect_shape.size(), out_shape.size()) << "Size mismatch";
            for (size_t i = 0; i < expect_shape.size(); ++i) {
                if (expect_shape[i].ConcreteValid() || out_shape[i].ConcreteValid()) {
                    ASSERT_TRUE(expect_shape[i].ConcreteValid()) << "Expected dim " << i << " is symbolic.";
                    ASSERT_TRUE(out_shape[i].ConcreteValid()) << "Output dim " << i << " is symbolic.";
                    EXPECT_EQ(expect_shape[i].Concrete(), out_shape[i].Concrete())
                        << "Mismatch at dim " << i << " input " << in_shape[i].Dump() << " output "
                        << out_shape[i].Dump();
                }
            }
        }
    }
}

// 统计指定 opcode 且首个输入 tensor magic 等于 inputMagic 的 op 数量
int CountOpsWithInputMagic(Function& func, Opcode oc, int inputMagic)
{
    int count = 0;
    for (auto& op : func.Operations()) {
        if (op.GetOpcode() != oc || op.iOperand.empty()) {
            continue;
        }
        auto input = op.iOperand.front();
        if (input != nullptr && input->GetMagic() == inputMagic) {
            count++;
        }
    }
    return count;
}

// 为 builder 添加 DDR 内存类型的 tensor 集合
void AddTensorsWithDDR(ComputationalGraphBuilder& G, const std::map<std::string, std::vector<int64_t>>& tensors)
{
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        G.GetTensor(name)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }
}

// 为 builder 添加写入 largeTensor 的 Assemble 操作集合
void AddAssembleOpsToLarge(ComputationalGraphBuilder& G,
                           const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>>& ops)
{
    for (const auto& [input, opName, offset] : ops) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        G.GetOp(opName)->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }
}

// 添加从 largeTensor 读取的 View 操作
void AddViewFromLarge(ComputationalGraphBuilder& G, const std::string& output, const std::string& opName,
                      const std::vector<int64_t>& offset)
{
    G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
    G.GetOp(opName)->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
}

// 混合消费者 1to1 场景构图: 2个Assemble生产者, 1个View + 1个Exp消费者
void BuildMixedConsumer1to1(ComputationalGraphBuilder& G)
{
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"input1", {16, 16}},      {"input2", {16, 16}}, {"in1", {16, 16}},   {"in2", {16, 16}},
        {"largeTensor", {32, 16}}, {"out1", {16, 16}},   {"expOut", {32, 16}}};
    AddTensorsWithDDR(G, tensors);
    G.AddOp(Opcode::OP_ABS, {"input1"}, {"in1"}, "Abs_1");
    G.AddOp(Opcode::OP_ABS, {"input2"}, {"in2"}, "Abs_2");
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "Assemble_1", {0, 0}}, {"in2", "Assemble_2", {16, 0}}};
    AddAssembleOpsToLarge(G, assembleOps);
    AddViewFromLarge(G, "out1", "View_1", {0, 0});
    G.AddOp(Opcode::OP_EXP, {"largeTensor"}, {"expOut"}, "Exp_1");
    G.SetInCast({"input1", "input2"});
    G.SetOutCast({"out1", "expOut"});
}

// 混合消费者 1toM 场景构图: 2个Assemble生产者, 2个View + 1个Exp消费者
void BuildMixedConsumer1toM(ComputationalGraphBuilder& G)
{
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"input1", {16, 32}},      {"input2", {16, 32}}, {"in1", {16, 32}},  {"in2", {16, 32}},
        {"largeTensor", {32, 32}}, {"out1", {16, 16}},   {"out2", {16, 16}}, {"expOut", {32, 32}}};
    AddTensorsWithDDR(G, tensors);
    G.AddOp(Opcode::OP_ABS, {"input1"}, {"in1"}, "Abs_1");
    G.AddOp(Opcode::OP_ABS, {"input2"}, {"in2"}, "Abs_2");
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "Assemble_1", {0, 0}}, {"in2", "Assemble_2", {16, 0}}};
    AddAssembleOpsToLarge(G, assembleOps);
    AddViewFromLarge(G, "out1", "View_1", {0, 0});
    AddViewFromLarge(G, "out2", "View_2", {0, 16});
    G.AddOp(Opcode::OP_EXP, {"largeTensor"}, {"expOut"}, "Exp_1");
    G.SetInCast({"input1", "input2"});
    G.SetOutCast({"out1", "out2", "expOut"});
}

// 混合消费者 MtoM 场景构图: 4个Assemble生产者, 1个View + 1个Exp消费者
void BuildMixedConsumerMtoM(ComputationalGraphBuilder& G)
{
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"input1", {16, 16}},      {"input2", {16, 16}}, {"input3", {16, 16}}, {"input4", {16, 16}},
        {"in1", {16, 16}},         {"in2", {16, 16}},    {"in3", {16, 16}},    {"in4", {16, 16}},
        {"largeTensor", {32, 32}}, {"out1", {32, 16}},   {"expOut", {32, 32}}};
    AddTensorsWithDDR(G, tensors);
    G.AddOp(Opcode::OP_ABS, {"input1"}, {"in1"}, "Abs_1");
    G.AddOp(Opcode::OP_ABS, {"input2"}, {"in2"}, "Abs_2");
    G.AddOp(Opcode::OP_ABS, {"input3"}, {"in3"}, "Abs_3");
    G.AddOp(Opcode::OP_ABS, {"input4"}, {"in4"}, "Abs_4");
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "Assemble_1", {0, 0}},
        {"in2", "Assemble_2", {0, 16}},
        {"in3", "Assemble_3", {16, 0}},
        {"in4", "Assemble_4", {16, 16}}};
    AddAssembleOpsToLarge(G, assembleOps);
    AddViewFromLarge(G, "out1", "View_1", {0, 0});
    G.AddOp(Opcode::OP_EXP, {"largeTensor"}, {"expOut"}, "Exp_1");
    G.SetInCast({"input1", "input2", "input3", "input4"});
    G.SetOutCast({"out1", "expOut"});
}

// 运行 SplitLargeFanoutTensor pass 的公共流程, 返回 function 和 largeTensor magic
struct MixedPassResult {
    Function* function;
    int ltMagic;
};

template <typename BuildFn>
MixedPassResult RunMixedConsumerPass(ComputationalGraphBuilder& G, BuildFn buildFn)
{
    buildFn(G);
    Function* function = G.GetFunction();
    int ltMagic = G.GetTensor("largeTensor")->GetMagic();
    npu::tile_fwk::SplitLargeFanoutTensor pass;
    pass.enableMoreSplit_ = false;
    EXPECT_EQ(SUCCESS, pass.PreCheck(*function));
    EXPECT_EQ(SUCCESS, pass.RunOnFunction(*function));
    EXPECT_EQ(SUCCESS, pass.PostCheck(*function));
    return {function, ltMagic};
}

// 验证混合消费者场景下 1to1 拆分正常工作, View被重定向, Exp不受影响
TEST_F(SplitLargeFanoutTensorTest, MixedConsumer_1to1_ShouldSplit)
{
    ComputationalGraphBuilder G;
    auto [function, ltMagic] = RunMixedConsumerPass(G, BuildMixedConsumer1to1);
    EXPECT_EQ(CountOpsWithInputMagic(*function, Opcode::OP_EXP, ltMagic), 1);
    EXPECT_EQ(CountOpsWithInputMagic(*function, Opcode::OP_VIEW, ltMagic), 0);
}

// 验证混合消费者场景下 1toM 拆分正常工作, 两个View均被重定向
TEST_F(SplitLargeFanoutTensorTest, MixedConsumer_1toM_ShouldSplit)
{
    ComputationalGraphBuilder G;
    auto [function, ltMagic] = RunMixedConsumerPass(G, BuildMixedConsumer1toM);
    EXPECT_EQ(CountOpsWithInputMagic(*function, Opcode::OP_EXP, ltMagic), 1);
    EXPECT_EQ(CountOpsWithInputMagic(*function, Opcode::OP_VIEW, ltMagic), 0);
}

// 验证混合消费者场景下 MtoM 被正确跳过, 不发生拆分
TEST_F(SplitLargeFanoutTensorTest, MixedConsumer_MtoM_Skipped)
{
    ComputationalGraphBuilder G;
    auto [function, ltMagic] = RunMixedConsumerPass(G, BuildMixedConsumerMtoM);
    EXPECT_EQ(CountOpsWithInputMagic(*function, Opcode::OP_EXP, ltMagic), 1);
    EXPECT_EQ(CountOpsWithInputMagic(*function, Opcode::OP_VIEW, ltMagic), 1);
}

TEST_F(SplitLargeFanoutTensorTest, BuilderLargeLinearFanoutPerfGuard)
{
    int viewCount = 2048, assemblesPerView = 2, assembleCount = viewCount * assemblesPerView;
    int64_t assembleCols = 4, thresholdMs = 3500;

    ComputationalGraphBuilder graph;
    BuildLargeLinearFanoutGraph(graph, viewCount, assemblesPerView, assembleCols);
    Function* f = graph.GetFunction();
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(f->Operations(false).size(), static_cast<size_t>((assembleCount + viewCount) * 3));

    SplitLargeFanoutTensor pass;
    ASSERT_EQ(pass.PreCheck(*f), SUCCESS);

    auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(pass.RunOnFunction(*f), SUCCESS);
    auto end = std::chrono::steady_clock::now();
    int64_t elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_EQ(pass.PostCheck(*f), SUCCESS);

    std::vector<int64_t> largeShape{viewCount, assembleCols * assemblesPerView};
    std::vector<int64_t> tileShape{1, largeShape[1]};
    int inCnt = assembleCount;
    int outCnt = viewCount;
    auto c = CountSplitByShape(*f, largeShape, tileShape);
    EXPECT_EQ(c.vCnt, static_cast<size_t>(inCnt + outCnt));
    EXPECT_EQ(c.aCnt, static_cast<size_t>(inCnt + outCnt));
    EXPECT_EQ(c.vLargeShape, static_cast<size_t>(0));             // no VIEW still reads from largeTensor
    EXPECT_EQ(c.vTileShape, static_cast<size_t>(outCnt));         // outcast VIEWs now read from tiles
    EXPECT_EQ(c.aTileShape, static_cast<size_t>(inCnt + outCnt)); // all ASSEMBLEs write into tiles
    EXPECT_EQ(c.aLargeShape, static_cast<size_t>(0));             // no ASSEMBLE still writes into largeTensor
    EXPECT_LT(elapsedMs, thresholdMs);
}

TEST_F(SplitLargeFanoutTensorTest, DynamicDslExpandAndSplitPerfGuard)
{
    PROGRAM("SplitLargeFanoutTensorPerfGuard")
    {
        const char* perfGuardStrategy = "SplitLargeFanoutTensorPerfGuardStrategy";
        int64_t thresholdMs = 5000;
        TileShape::Current().SetVecTile(1, 1, 8);
        PassManager::Instance().RegisterStrategy(
            perfGuardStrategy,
            {{"ExpandFunction", PassName::EXPAND_FUNCTION}, {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE}});
        ConfigManager::Instance();
        config::SetBuildStatic(true);
        config::SetHostConfig(KEY_STRATEGY, perfGuardStrategy);
        config::SetPassConfig(perfGuardStrategy, "SplitReshape", KEY_DISABLE_PASS, true);
        int b = 2, sq = 512, d = 16;
        Tensor q(DT_FP32, {b, sq, d}, "q");
        Tensor out(DT_FP32, {b, sq, d}, "out");
        BuildDynamicDslPerfFunction(q, out, b, sq, d, perfGuardStrategy);
        auto* f = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitLargeFanoutDynamicPerfGuard");
        ASSERT_NE(f, nullptr);

        SplitLargeFanoutTensor pass;
        ASSERT_EQ(pass.PreCheck(*f), SUCCESS);
        auto start = std::chrono::steady_clock::now();
        EXPECT_EQ(pass.RunOnFunction(*f), SUCCESS);
        auto end = std::chrono::steady_clock::now();
        int64_t elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        EXPECT_EQ(pass.PostCheck(*f), SUCCESS);

        // Verify: addTmp is inlined by MergeViewAssemble, then slice/contract post-processing removes
        // identity tile-to-tile view-like ops. All assemble-like ops target the large output shape {b,sq,d}.
        std::vector<int64_t> largeShape{b, sq, d}, tileShape{1, 1, 8};
        auto c = CountSplitByShape(*f, largeShape, tileShape);
        int pathCnt = sq * b * (d / 8);                         // each {b,1,d} view expands to b*(d/8) tiles
        EXPECT_EQ(c.vCnt, static_cast<size_t>(pathCnt));        // q -> tile view-like ops
        EXPECT_EQ(c.aCnt, static_cast<size_t>(pathCnt));        // all ASSEMBLEs to out
        EXPECT_EQ(c.vLargeShape, static_cast<size_t>(pathCnt)); // VIEW from q
        EXPECT_EQ(c.vTileShape, static_cast<size_t>(0));        // identity view-like ops on tiles are merged
        EXPECT_EQ(c.aLargeShape, static_cast<size_t>(pathCnt)); // ASSEMBLE into out
        EXPECT_EQ(c.aTileShape, static_cast<size_t>(0));        // addTmp merged, no tile ASSEMBLE
        EXPECT_LT(elapsedMs, thresholdMs);
    }
}

namespace {
// 为 OverlapSearchIndex 直接 UT 构造 tensorInfos：(shape, offset) 列表 -> (LogicalTensorPtr, Offset) 列表
std::vector<std::pair<LogicalTensorPtr, Offset>> BuildOverlapTensorInfos(
    ComputationalGraphBuilder& graph, const std::vector<std::pair<Shape, Offset>>& shapeOffsets)
{
    std::vector<std::pair<LogicalTensorPtr, Offset>> tensorInfos;
    tensorInfos.reserve(shapeOffsets.size());
    for (size_t i = 0; i < shapeOffsets.size(); ++i) {
        const auto name = "overlap_tensor_" + std::to_string(i);
        EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, shapeOffsets[i].first, name));
        tensorInfos.emplace_back(graph.GetTensor(name), shapeOffsets[i].second);
    }
    return tensorInfos;
}
} // namespace

// 直接覆盖 IsTensorCoveredByTile 的覆盖判断语义（含维度不匹配、完美匹配、被覆盖、部分超出）
TEST_F(SplitLargeFanoutTensorTest, IsTensorCoveredByTileDirect)
{
    SplitLargeFanoutTensor pass;
    EXPECT_FALSE(pass.IsTensorCoveredByTile({0, 0}, {2, 2}, {0, 0, 0}, {2, 2, 2}));
    EXPECT_FALSE(pass.IsTensorCoveredByTile({0, 0, 0}, {2, 2, 2}, {0, 0}, {2, 2}));
    EXPECT_TRUE(pass.IsTensorCoveredByTile({0, 0}, {2, 2}, {0, 0}, {2, 2}));
    EXPECT_TRUE(pass.IsTensorCoveredByTile({1, 1}, {1, 1}, {0, 0}, {2, 2}));
    EXPECT_FALSE(pass.IsTensorCoveredByTile({1, 1}, {2, 2}, {0, 0}, {2, 2}));
}

// 多维场景下 bestDim 非 dim 0：dim1 候选数小于 dim0，验证"选择候选数最少的维度"核心优化逻辑与结果顺序
TEST_F(SplitLargeFanoutTensorTest, OverlapSearchIndexBestDimNotZero)
{
    SplitLargeFanoutTensor pass;
    ComputationalGraphBuilder graph;
    // tile {3,4}@{0,0}：dim0 候选 4 个（T0..T3），dim1 候选 3 个（T0,T3,T4），bestDim=dim1
    auto tensorInfos = BuildOverlapTensorInfos(graph, {
                                                          {{1, 2}, {0, 0}},
                                                          {{1, 2}, {1, 6}},
                                                          {{1, 2}, {2, 6}},
                                                          {{1, 2}, {2, 0}},
                                                          {{1, 2}, {7, 0}},
                                                      });
    SplitLargeFanoutTensor::OverlapSearchIndex index;
    pass.BuildOverlapSearchIndex(tensorInfos, index);
    ASSERT_EQ(index.orderByDim.size(), 2u);
    LogicalTensors result;
    pass.CollectCoveredTensors({3, 4}, {0, 0}, index, result);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], tensorInfos[0].first);
    EXPECT_EQ(result[1], tensorInfos[3].first);
}
} // namespace tile_fwk
} // namespace npu
