/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_infer_tensor_format.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <initializer_list>

#include "computational_graph_builder.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/tensor_graph_pass/infer_tensor_format.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error_code.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"

using namespace npu::tile_fwk;

class InferTensorFormatTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override { Program::GetInstance().Reset(); }

    int CountOpsByOpcode(Function* func, Opcode opcode)
    {
        int count = 0;
        for (auto& op : func->Operations()) {
            if (op.GetOpcode() == opcode)
                count++;
        }
        return count;
    }

    Status RunInferTensorFormat(Function* func)
    {
        InferTensorFormat pass;
        return pass.Run(*func, "InferTensorFormatTest", "InferTensorFormat");
    }

    void UpdateDynValidShapes(ComputationalGraphBuilder& graph, std::initializer_list<const char*> tensorNames)
    {
        for (auto* name : tensorNames) {
            auto shape = graph.GetTensor(name)->GetShape();
            std::vector<SymbolicScalar> validShape;
            for (auto dim : shape) {
                validShape.emplace_back(dim);
            }
            graph.GetTensor(name)->UpdateDynValidShape(validShape);
        }
    }

    void SetFakeTransFormat(ComputationalGraphBuilder& graph, const std::string& opName, TileOpFormat inFormat,
                            TileOpFormat outFormat)
    {
        auto* fakeTrans = graph.GetOp(opName);
        ASSERT_NE(fakeTrans, nullptr);
        fakeTrans->SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(inFormat));
        fakeTrans->SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR, static_cast<int64_t>(outFormat));
    }
};

// =============================================================================
// 场景 j1: ND 格式透传链
//   图: incast0(ND) → View → Reshape → View → Assemble → local_out
//   预期: 非 outcast 的 Assemble 仍沿用首输入格式，全链路 ND
// =============================================================================
TEST_F(InferTensorFormatTest, NDFormatPassThroughLocalAssemble)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape{16, 32};
    ASSERT_TRUE(
        G.AddTensors(DataType::DT_FP16, shape, {"incast0", "v1_out", "r1_out", "v2_out", "asm_out", "local_out"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast0"}, {"v1_out"}, "view1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_RESHAPE, {"v1_out"}, {"r1_out"}, "reshape1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"r1_out"}, {"v2_out"}, "view2"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"v2_out"}, {"local_out"}, "assemble"));
    ASSERT_TRUE(G.SetInCast({"incast0"}));
    G.GetTensor("incast0")->GetRawTensor()->format = TileOpFormat::TILEOP_ND;

    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(G.GetTensor("incast0")->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(G.GetTensor("v1_out")->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(G.GetTensor("r1_out")->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(G.GetTensor("v2_out")->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(G.GetTensor("local_out")->Format(), TileOpFormat::TILEOP_ND);
}

// =============================================================================
// 场景 j2: 3×(view+abs) → conv2D → sub → assemble
//   图: 三路 ND 输入分别经过 view+abs 进入 conv2D
//       conv2D input[0]要求NC1HWC0, input[1]要求FRACTAL_Z, input[2]要求ND
//       conv_out(NC1HWC0) → FakeTrans(ND) → sub(要求ND) → assemble
//   预期: Path0 ND→NC1HWC0 (OP_NCHW2NC1HWC0×1)
//         Path1 ND→FRACTAL_Z (OP_NCHW2Fractal_Z×1)
//         conv_out NC1HWC0→ND (OP_NC1HWC02NCHW×1)
//         sub_out(ND), incast4(ND) → assemble 输出 ND
// =============================================================================
TEST_F(InferTensorFormatTest, Conv2DWithTransDataInsertion)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d,
                             {"incast0", "incast1", "incast2", "incast3", "incast4", "v0_out", "a0_out", "v1_out",
                              "a1_out", "v2_out", "a2_out", "conv_nd", "sub_out", "asm_out", "outcast"}));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP16, shape5d, "conv_out"));

    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast0"}, {"v0_out"}, "view0"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"v0_out"}, {"a0_out"}, "abs0"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast1"}, {"v1_out"}, "view1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"v1_out"}, {"a1_out"}, "abs1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast2"}, {"v2_out"}, "view2"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"v2_out"}, {"a2_out"}, "abs2"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_CONV2D, {"a0_out", "a1_out", "a2_out"}, {"conv_out"}, "conv"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"conv_out"}, {"conv_nd"}, "conv_to_nd"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_SUB, {"conv_nd", "incast3"}, {"sub_out"}, "sub"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"sub_out"}, {"outcast"}, "assemble0"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"incast4"}, {"outcast"}, "assemble1"));
    ASSERT_TRUE(G.SetInCast({"incast0", "incast1", "incast2", "incast3", "incast4"}));
    ASSERT_TRUE(G.SetOutCast({"outcast"}));
    SetFakeTransFormat(G, "conv_to_nd", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);

    UpdateDynValidShapes(G, {"incast0", "incast1", "incast2", "incast3", "incast4", "v0_out", "a0_out", "v1_out",
                             "a1_out", "v2_out", "a2_out", "conv_out", "conv_nd"});

    G.GetOp("conv")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    G.GetOp("conv_to_nd")->GetTileShapeForSetting().SetVecTile({16, 16, 1, 1, 16});
    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(CountOpsByOpcode(G.GetFunction(), Opcode::OP_NCHW2NC1HWC0), 1);
    EXPECT_EQ(CountOpsByOpcode(G.GetFunction(), Opcode::OP_NCHW2Fractal_Z), 1);
    EXPECT_EQ(CountOpsByOpcode(G.GetFunction(), Opcode::OP_NC1HWC02NCHW), 1);
    EXPECT_EQ(G.GetTensor("conv_out")->Format(), TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_EQ(G.GetTensor("sub_out")->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(G.GetTensor("outcast")->Format(), TileOpFormat::TILEOP_ND);

    // 验证 conv2D 输入 tensor 的 format
    auto* conv = G.GetOp("conv");
    EXPECT_EQ(conv->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_EQ(conv->GetIOperands()[1]->Format(), TileOpFormat::TILEOP_FRACTAL_Z);
    EXPECT_EQ(conv->GetIOperands()[2]->Format(), TileOpFormat::TILEOP_ND);
    auto* sub = G.GetOp("sub");
    ASSERT_NE(sub, nullptr);
    EXPECT_NE(sub->GetIOperands()[0], G.GetTensor("conv_out"));
    EXPECT_NE(sub->GetIOperands()[0], G.GetTensor("conv_nd"));
    EXPECT_EQ(sub->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_ND);
}

// =============================================================================
// 场景 j3: 共享输入分叉 —— t0,t1 同时喂给 conv2D 和 matmul
//   图: 三路 ND→view→t0/t1/t2, t0/t1/t2 → conv2D, t0/t1 → matmul
//       conv_out(NC1HWC0) → FakeTrans(ND) → assemble0, mm_out(ND) → assemble1
//   预期: t0→conv2D[0](NC1HWC0): TransData×1
//         t1→conv2D[1](FRACTAL_Z): TransData×1
//         t0→matmul[0](ND), t1→matmul[1](ND): 匹配, 无转换
//         conv_out NC1HWC0→ND: TransData×1
//         mm_out(ND)→assemble1: outcast 固定 ND, 匹配无转换
//         共 OP_NCHW2NC1HWC0×1 + OP_NCHW2Fractal_Z×1 + OP_NC1HWC02NCHW×1
// =============================================================================
TEST_F(InferTensorFormatTest, SharedInputsConv2DAndMatmul)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    ASSERT_TRUE(
        G.AddTensors(DataType::DT_FP16, shape4d,
                     {"incast0", "incast1", "incast2", "t0", "t1", "t2", "conv_nd", "mm_out", "asm_out", "outcast"}));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP16, shape5d, "conv_out"));

    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast0"}, {"t0"}, "view0"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast1"}, {"t1"}, "view1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast2"}, {"t2"}, "view2"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_CONV2D, {"t0", "t1", "t2"}, {"conv_out"}, "conv"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"t0", "t1"}, {"mm_out"}, "matmul"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"conv_out"}, {"conv_nd"}, "conv_to_nd"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"conv_nd"}, {"outcast"}, "assemble0"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"mm_out"}, {"outcast"}, "assemble1"));
    ASSERT_TRUE(G.SetInCast({"incast0", "incast1", "incast2"}));
    ASSERT_TRUE(G.SetOutCast({"outcast"}));
    SetFakeTransFormat(G, "conv_to_nd", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);

    UpdateDynValidShapes(G, {"incast0", "incast1", "incast2", "t0", "t1", "t2", "conv_out", "conv_nd", "mm_out"});

    G.GetOp("conv")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    G.GetOp("conv_to_nd")->GetTileShapeForSetting().SetVecTile({16, 16, 1, 1, 16});
    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(CountOpsByOpcode(G.GetFunction(), Opcode::OP_NCHW2NC1HWC0), 1);
    EXPECT_EQ(CountOpsByOpcode(G.GetFunction(), Opcode::OP_NCHW2Fractal_Z), 1);
    EXPECT_EQ(CountOpsByOpcode(G.GetFunction(), Opcode::OP_NC1HWC02NCHW), 1);
    EXPECT_EQ(G.GetTensor("conv_out")->Format(), TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_EQ(G.GetTensor("mm_out")->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(G.GetTensor("outcast")->Format(), TileOpFormat::TILEOP_ND);

    // 验证 conv2D 和 matmul 的输入 tensor format 正确连接
    auto* conv = G.GetOp("conv");
    EXPECT_EQ(conv->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_EQ(conv->GetIOperands()[1]->Format(), TileOpFormat::TILEOP_FRACTAL_Z);
    EXPECT_EQ(conv->GetIOperands()[2]->Format(), TileOpFormat::TILEOP_ND);
    auto* mm = G.GetOp("matmul");
    EXPECT_EQ(mm->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(mm->GetIOperands()[1]->Format(), TileOpFormat::TILEOP_ND);
    auto* assemble0 = G.GetOp("assemble0");
    ASSERT_NE(assemble0, nullptr);
    EXPECT_NE(assemble0->GetIOperands()[0], G.GetTensor("conv_out"));
    EXPECT_NE(assemble0->GetIOperands()[0], G.GetTensor("conv_nd"));
    EXPECT_EQ(assemble0->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_ND);
}

// =============================================================================
// 场景 j4: 3 路并行 view → assemble，第一路 NC1HWC0，后两路 ND
//   图: incast0(NC1HWC0)
//       incast1(ND) → view1 → v1_out
//       incast2(ND) → view2 → v2_out
//       incast0(NC1HWC0) → FakeTrans(ND) → assemble0 → outcast
//       v1_out, v2_out → assemble → outcast
//   预期: 输出为 outcast 时固定 ND，不再由首个 assemble 决定
//         incast0(NC1HWC0)→ND 支持 → OP_NC1HWC02NCHW×1
//         v1_out/v2_out 已为 ND，不插入转换
//         输出 ND
// =============================================================================
TEST_F(InferTensorFormatTest, AssembleToOutcastRequiresND)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP16, shape5d, "incast0"));
    ASSERT_TRUE(
        G.AddTensors(DataType::DT_FP16, shape4d, {"incast1", "incast2", "v0_nd", "v1_out", "v2_out", "outcast"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast1"}, {"v1_out"}, "view1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"incast2"}, {"v2_out"}, "view2"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"incast0"}, {"v0_nd"}, "v0_to_nd"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"v0_nd"}, {"outcast"}, "assemble0"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"v1_out"}, {"outcast"}, "assemble1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"v2_out"}, {"outcast"}, "assemble2"));
    ASSERT_TRUE(G.SetInCast({"incast0", "incast1", "incast2"}));
    ASSERT_TRUE(G.SetOutCast({"outcast"}));
    G.GetTensor("incast0")->GetRawTensor()->format = TileOpFormat::TILEOP_NC1HWC0;
    SetFakeTransFormat(G, "v0_to_nd", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);

    G.GetOp("v0_to_nd")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(G.GetTensor("incast0")->Format(), TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_EQ(G.GetTensor("outcast")->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NC1HWC02NCHW), 1);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2NC1HWC0), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2Fractal_Z), 0);

    auto* assemble0 = G.GetOp("assemble0");
    auto* assemble1 = G.GetOp("assemble1");
    auto* assemble2 = G.GetOp("assemble2");
    ASSERT_NE(assemble0, nullptr);
    ASSERT_NE(assemble1, nullptr);
    ASSERT_NE(assemble2, nullptr);
    EXPECT_EQ(assemble0->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_NE(assemble0->GetIOperands()[0], G.GetTensor("v0_nd"));
    EXPECT_EQ(assemble1->GetIOperands()[0], G.GetTensor("v1_out"));
    EXPECT_EQ(assemble2->GetIOperands()[0], G.GetTensor("v2_out"));
}

TEST_F(InferTensorFormatTest, FakeTransInsertsRequiredInputAndOutputTransDataThenReconnectsConsumer)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d, {"incast", "fake_out", "abs_out"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"incast"}, {"fake_out"}, "fake_trans"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"fake_out"}, {"abs_out"}, "abs"));
    ASSERT_TRUE(G.SetInCast({"incast"}));
    ASSERT_TRUE(G.SetOutCast({"abs_out"}));
    UpdateDynValidShapes(G, {"incast", "fake_out", "abs_out"});

    auto* fakeTrans = G.GetOp("fake_trans");
    ASSERT_NE(fakeTrans, nullptr);
    fakeTrans->SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(TileOpFormat::TILEOP_NC1HWC0));
    fakeTrans->SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR, static_cast<int64_t>(TileOpFormat::TILEOP_ND));

    fakeTrans->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    // Phase 2 eliminates the FakeTrans pair (ND→NC1HWC0→ND is identity), so 0 TransData ops.
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_FAKE_TRANS), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2NC1HWC0), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NC1HWC02NCHW), 0);
    auto* abs = G.GetOp("abs");
    ASSERT_NE(abs, nullptr);
    // abs is reconnected directly to incast (same signature as FakeTrans output)
    EXPECT_EQ(abs->GetIOperands()[0], G.GetTensor("incast"));
    EXPECT_EQ(abs->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_ND);
    EXPECT_TRUE(G.GetTensor("fake_out")->GetProducers().empty());
    EXPECT_TRUE(G.GetTensor("fake_out")->GetConsumers().empty());
}

TEST_F(InferTensorFormatTest, FakeTransMissingFormatAttrFails)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d, {"incast", "fake_out"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"incast"}, {"fake_out"}, "fake_trans"));
    ASSERT_TRUE(G.SetInCast({"incast"}));

    auto* fakeTrans = G.GetOp("fake_trans");
    ASSERT_NE(fakeTrans, nullptr);
    fakeTrans->SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(TileOpFormat::TILEOP_NC1HWC0));

    EXPECT_EQ(RunInferTensorFormat(G.GetFunction()), FAILED);
}

TEST_F(InferTensorFormatTest, FakeTransUnsupportedFormatConversionFails)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d, {"incast", "fake_out"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"incast"}, {"fake_out"}, "fake_trans"));
    ASSERT_TRUE(G.SetInCast({"incast"}));

    auto* fakeTrans = G.GetOp("fake_trans");
    ASSERT_NE(fakeTrans, nullptr);
    fakeTrans->SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(TileOpFormat::TILEOP_FRACTAL_Z));
    fakeTrans->SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR, static_cast<int64_t>(TileOpFormat::TILEOP_ND));

    fakeTrans->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    EXPECT_EQ(RunInferTensorFormat(G.GetFunction()), FAILED);
}

// =============================================================================
// 场景 j5: 零输入 op (VEC_DUP) 输出驱动 FAKE_TRANS 推导
//   图: VEC_DUP(零输入) → dup_out(ND) → FAKE_TRANS(ND→NC1HWC0) → fake_out → ASSEMBLE → local_out
//   预期: 无 incast 时 worklist 完全依赖零输入 op 种子循环;
//         VEC_DUP 输出被 seed 后, FAKE_TRANS 输出格式被正确推导为 NC1HWC0,
//         FAKE_TRANS 被物化为 NCHW2NC1HWC0, ASSEMBLE 输入重连到物化结果
// =============================================================================
TEST_F(InferTensorFormatTest, ZeroInputOpOutputFeedsFakeTrans)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP16, shape4d, "dup_out"));
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape5d, {"fake_out", "local_out"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_VEC_DUP, {}, {"dup_out"}, "vec_dup"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"dup_out"}, {"fake_out"}, "fake_trans"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"fake_out"}, {"local_out"}, "assemble"));
    SetFakeTransFormat(G, "fake_trans", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_NC1HWC0);
    UpdateDynValidShapes(G, {"dup_out", "fake_out", "local_out"});

    auto* fakeTrans = G.GetOp("fake_trans");
    ASSERT_NE(fakeTrans, nullptr);
    fakeTrans->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});

    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_FAKE_TRANS), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2NC1HWC0), 1);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NC1HWC02NCHW), 0);
    auto* assemble = G.GetOp("assemble");
    ASSERT_NE(assemble, nullptr);
    EXPECT_NE(assemble->GetIOperands()[0], G.GetTensor("fake_out"));
    EXPECT_EQ(assemble->GetIOperands()[0]->Format(), TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_EQ(G.GetTensor("local_out")->Format(), TileOpFormat::TILEOP_NC1HWC0);
    EXPECT_TRUE(G.GetTensor("fake_out")->GetProducers().empty());
    EXPECT_TRUE(G.GetTensor("fake_out")->GetConsumers().empty());
}

// =============================================================================
// 场景: 多步链式+多 consumer（值编号+并查集核心场景）
//   图: root(NC1HWC0) → FakeTrans_A(NC1HWC0→ND) → FakeTrans_B(ND→FRACTAL_Z) → asm1 → local_out
//                                            └── FakeTrans_C(ND→NC1HWC0) → asm2 → outcast
//   预期: FakeTrans_C 与 root 签名相同 (root, NC1HWC0) → 消除
//         FakeTrans_A (root, ND) 和 FakeTrans_B (root, FRACTAL_Z) 签名唯一 → 保留并物化
// =============================================================================
TEST_F(InferTensorFormatTest, MultiStepChainWithMultiConsumer)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape5d, {"root", "c_out"}));
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d, {"a_out", "b_out", "local_out", "outcast"}));

    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"root"}, {"a_out"}, "fake_trans_a"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"a_out"}, {"b_out"}, "fake_trans_b"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"a_out"}, {"c_out"}, "fake_trans_c"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"b_out"}, {"local_out"}, "asm1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"c_out"}, {"outcast"}, "asm2"));

    ASSERT_TRUE(G.SetInCast({"root"}));
    ASSERT_TRUE(G.SetOutCast({"outcast"}));

    SetFakeTransFormat(G, "fake_trans_a", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);
    SetFakeTransFormat(G, "fake_trans_b", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_FRACTAL_Z);
    SetFakeTransFormat(G, "fake_trans_c", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_NC1HWC0);
    G.GetOp("fake_trans_a")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    G.GetOp("fake_trans_b")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    G.GetOp("fake_trans_c")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});

    G.GetTensor("root")->GetRawTensor()->format = TileOpFormat::TILEOP_NC1HWC0;

    UpdateDynValidShapes(G, {"root", "a_out", "b_out", "c_out", "local_out", "outcast"});

    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    // FakeTrans_C eliminated (same signature as root), FakeTrans_A and FakeTrans_B preserved
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_FAKE_TRANS), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NC1HWC02NCHW), 1);   // FakeTrans_A materialized
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2Fractal_Z), 1); // FakeTrans_B materialized
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2NC1HWC0), 0);   // FakeTrans_C eliminated, not materialized
}

// =============================================================================
// 场景: A→B→C→D→A 长链冗余全部消除
//   图: root(NC1HWC0) → FakeTrans_A(NC1HWC0→ND) → FakeTrans_B(ND→NC1HWC0)
//                     → FakeTrans_C(NC1HWC0→ND) → FakeTrans_D(ND→NC1HWC0) → asm → local_out
//   签名: root=(root,NC1HWC0), A=(root,ND), B=(root,NC1HWC0), C=(root,ND), D=(root,NC1HWC0)
//   预期: B/D 与 root 同签名 → 消除; C 与 A 同签名 → 消除; A 无 live consumer → 消除
//         asm 直连 root, 0 TransData
// =============================================================================
TEST_F(InferTensorFormatTest, LongChainAllEliminated)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape5d, {"root", "b_out", "d_out", "local_out"}));
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d, {"a_out", "c_out"}));

    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"root"}, {"a_out"}, "fake_trans_a"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"a_out"}, {"b_out"}, "fake_trans_b"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"b_out"}, {"c_out"}, "fake_trans_c"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"c_out"}, {"d_out"}, "fake_trans_d"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"d_out"}, {"local_out"}, "asm"));

    ASSERT_TRUE(G.SetInCast({"root"}));

    SetFakeTransFormat(G, "fake_trans_a", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);
    SetFakeTransFormat(G, "fake_trans_b", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_NC1HWC0);
    SetFakeTransFormat(G, "fake_trans_c", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);
    SetFakeTransFormat(G, "fake_trans_d", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_NC1HWC0);

    G.GetTensor("root")->GetRawTensor()->format = TileOpFormat::TILEOP_NC1HWC0;

    UpdateDynValidShapes(G, {"root", "a_out", "b_out", "c_out", "d_out", "local_out"});

    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_FAKE_TRANS), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NC1HWC02NCHW), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2NC1HWC0), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2Fractal_Z), 0);

    auto* asm_op = G.GetOp("asm");
    ASSERT_NE(asm_op, nullptr);
    EXPECT_EQ(asm_op->GetIOperands()[0], G.GetTensor("root"));
}

// =============================================================================
// 场景: A→B→A 其中 A 的输出有其他 consumer
//   图: root(NC1HWC0) → FakeTrans_A(NC1HWC0→ND) → FakeTrans_B(ND→NC1HWC0) → asm1 → outcast1
//                                            └── asm2 → outcast2 (也消费 A 的 ND 输出)
//   签名: root=(root,NC1HWC0), A=(root,ND), B=(root,NC1HWC0)
//   预期: B 与 root 同签名 → 消除; Phase1 为 asm1 插入的新 FakeTrans 与 A 同签名 → 消除
//         A 有 asm2 消费 → 保留并物化
//         OP_NC1HWC02NCHW=1, OP_NCHW2NC1HWC0=0
// =============================================================================
TEST_F(InferTensorFormatTest, PartialChainEliminationWithExtraConsumer)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape5d, {"root", "b_out"}));
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d, {"a_out", "outcast1", "outcast2"}));

    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"root"}, {"a_out"}, "fake_trans_a"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"a_out"}, {"b_out"}, "fake_trans_b"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"b_out"}, {"outcast1"}, "asm1"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"a_out"}, {"outcast2"}, "asm2"));

    ASSERT_TRUE(G.SetInCast({"root"}));
    ASSERT_TRUE(G.SetOutCast({"outcast1", "outcast2"}));

    SetFakeTransFormat(G, "fake_trans_a", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);
    SetFakeTransFormat(G, "fake_trans_b", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_NC1HWC0);
    G.GetOp("fake_trans_a")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});
    G.GetOp("fake_trans_b")->GetTileShapeForSetting().SetVecTile({16, 16, 2, 16});

    G.GetTensor("root")->GetRawTensor()->format = TileOpFormat::TILEOP_NC1HWC0;

    UpdateDynValidShapes(G, {"root", "a_out", "b_out", "outcast1", "outcast2"});

    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_FAKE_TRANS), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NC1HWC02NCHW), 1); // FakeTrans_A materialized
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2NC1HWC0), 0); // FakeTrans_B eliminated
}

// =============================================================================
// 场景: A→B→C→B→D 中间部分冗余消除
//   图: root(NC1HWC0) → FakeTrans_A(NC1HWC0→ND) → FakeTrans_B(ND→NC1HWC0)
//                     → FakeTrans_C(NC1HWC0→ND) → FakeTrans_D(ND→FRACTAL_Z) → asm → local_out
//   签名: root=(root,NC1HWC0), A=(root,ND), B=(root,NC1HWC0), C=(root,ND), D=(root,FRACTAL_Z)
//   预期: B 与 root 同签名 → 消除; C 与 A 同签名 → 消除
//         A 和 D 签名唯一 → 保留并物化
//         OP_NC1HWC02NCHW=1, OP_NCHW2Fractal_Z=1
// =============================================================================
TEST_F(InferTensorFormatTest, MiddleChainPartialElimination)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape4d{2, 32, 14, 14};
    std::vector<int64_t> shape5d{2, 2, 14, 14, 16};
    TileShape::Current().SetVecTile({16, 256, 2, 16});
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape5d, {"root", "b_out"}));
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP16, shape4d, {"a_out", "c_out", "d_out", "local_out"}));

    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"root"}, {"a_out"}, "fake_trans_a"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"a_out"}, {"b_out"}, "fake_trans_b"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"b_out"}, {"c_out"}, "fake_trans_c"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_FAKE_TRANS, {"c_out"}, {"d_out"}, "fake_trans_d"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"d_out"}, {"local_out"}, "asm"));

    ASSERT_TRUE(G.SetInCast({"root"}));

    SetFakeTransFormat(G, "fake_trans_a", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);
    SetFakeTransFormat(G, "fake_trans_b", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_NC1HWC0);
    SetFakeTransFormat(G, "fake_trans_c", TileOpFormat::TILEOP_NC1HWC0, TileOpFormat::TILEOP_ND);
    SetFakeTransFormat(G, "fake_trans_d", TileOpFormat::TILEOP_ND, TileOpFormat::TILEOP_FRACTAL_Z);
    G.GetOp("fake_trans_a")->GetTileShapeForSetting().SetVecTile({16, 256, 2, 16});
    G.GetOp("fake_trans_b")->GetTileShapeForSetting().SetVecTile({16, 256, 2, 16});
    G.GetOp("fake_trans_c")->GetTileShapeForSetting().SetVecTile({16, 256, 2, 16});
    G.GetOp("fake_trans_d")->GetTileShapeForSetting().SetVecTile({16, 256, 2, 16});

    G.GetTensor("root")->GetRawTensor()->format = TileOpFormat::TILEOP_NC1HWC0;

    UpdateDynValidShapes(G, {"root", "a_out", "b_out", "c_out", "d_out", "local_out"});

    Function* func = G.GetFunction();
    ASSERT_EQ(RunInferTensorFormat(func), SUCCESS);

    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_FAKE_TRANS), 0);
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NC1HWC02NCHW), 1);   // FakeTrans_A materialized
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2Fractal_Z), 1); // FakeTrans_D materialized
    EXPECT_EQ(CountOpsByOpcode(func, Opcode::OP_NCHW2NC1HWC0), 0);   // FakeTrans_B/C eliminated
}
