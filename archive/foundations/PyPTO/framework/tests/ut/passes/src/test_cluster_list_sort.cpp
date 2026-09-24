/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cluster_list_sort.cpp
 * \brief Unit test for ClusterListSort.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "symbolic_scalar_test_utils.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#define private public
#include "computational_graph_builder.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/cluster_list_sort.h"

namespace npu::tile_fwk {

class ClusterListSortTest : public ::testing::Test {
public:
    void SetUp() override { Program::GetInstance().Reset(); }
    void TearDown() override {}
};

static bool Before(const std::vector<Operation*>& ops, Operation* a, Operation* b)
{
    size_t ia = ops.size();
    size_t ib = ops.size();
    for (size_t i = 0; i < ops.size(); ++i) {
        if (ops[i] == a) {
            ia = i;
        }
        if (ops[i] == b) {
            ib = i;
        }
    }
    return ia < ib;
}

// 用例 1：单簇直行。copy_in -> add -> copy_out，验证建簇/黏簇/alloc 回插。
TEST_F(ClusterListSortTest, SingleClusterStraightRun)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_DEVICE_DDR};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t3"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "CopyIn", "Add", "CopyOut"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);

    ASSERT_EQ(sorter.clusters_.size(), 1U);
    ASSERT_EQ(sorter.clusters_[0].ops.size(), 3U);
    EXPECT_EQ(sorter.clusters_[0].ops[0], b.GetOp("CopyIn"));
    EXPECT_EQ(sorter.clusters_[0].ops[1], b.GetOp("Add"));
    EXPECT_EQ(sorter.clusters_[0].ops[2], b.GetOp("CopyOut"));
    EXPECT_EQ(sorter.cluster_[b.GetOp("CopyIn")], 0);
    EXPECT_EQ(sorter.cluster_[b.GetOp("Add")], 0);
    EXPECT_EQ(sorter.cluster_[b.GetOp("CopyOut")], 0);
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("CopyIn"), b.GetOp("Add")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("Add"), b.GetOp("CopyOut")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("Alloc2"), b.GetOp("Add")));
}

// 用例 2：菱形分簇 + secondaryId。验证 fan-out/join 断簇、并查集合组。
TEST_F(ClusterListSortTest, DiamondFanOutJoinSecondaryId)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_UB, MEM_UB, MEM_UB, MEM_DEVICE_DDR};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5", "t6"}, {"t7"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t3"}, {"t5"}, {"t6"}, {"t7"}, {"t2"},
                                               {"t3"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "CopyIn",
                                     "Add0",   "Add1",   "Add2",   "Add3",   "CopyOut"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);

    ASSERT_EQ(sorter.clusters_.size(), 4U);
    ASSERT_EQ(sorter.clusters_[0].ops.size(), 2U);
    EXPECT_EQ(sorter.clusters_[0].ops[0], b.GetOp("CopyIn"));
    EXPECT_EQ(sorter.clusters_[0].ops[1], b.GetOp("Add0"));
    ASSERT_EQ(sorter.clusters_[3].ops.size(), 2U);
    EXPECT_EQ(sorter.clusters_[3].ops[0], b.GetOp("Add3"));
    EXPECT_EQ(sorter.clusters_[3].ops[1], b.GetOp("CopyOut"));
    EXPECT_EQ(sorter.clusters_[1].secondaryId, sorter.clusters_[2].secondaryId);
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("CopyIn"), b.GetOp("Add0")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("Add0"), b.GetOp("Add1")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("Add0"), b.GetOp("Add2")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("Add1"), b.GetOp("Add3")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("Add2"), b.GetOp("Add3")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("Add3"), b.GetOp("CopyOut")));
}

// 用例 3a：三方比较换簇。A 首个 op 无条件执行；第二 op net>0，B 头 net 更小 -> 跳转 B。
// 簇 A: copy_inA(t1->t2, 4KB) -> addA(t2->t3, 24KB) -> copy_outA(t3->t4 DDR)
//   basePeak=28KB。copy_inA 无条件执行，addA net=+20KB>0 -> 三方比较。
// 簇 B: copy_inB(t5->t7, 16KB) -> addB(t7->t6, 16KB)，basePeak=32KB。
//   B 头=copy_inB net=+16KB < 20KB -> 胜出，跳转 B。addB net=0 -> 黏簇直行。
// 期望 orderedNormals_ = [copy_inA, copy_inB, addB, addA, copy_outA]。
TEST_F(ClusterListSortTest, ThreeWayChooseSwitchToB)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_DEVICE_DDR, MEM_DEVICE_DDR, MEM_UB, MEM_UB};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    // t2=4KB(32x32), t3=24KB(96x64), t7=16KB(默认64x64), t6=16KB(默认64x64)
    auto t2 = b.GetTensor("t2");
    t2->shape = {32, 32};
    t2->tensor->rawshape = {32, 32};
    auto t3 = b.GetTensor("t3");
    t3->shape = {96, 64};
    t3->tensor->rawshape = {96, 64};

    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT, Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t5"}, {"t7"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t3"}, {"t6"}, {"t7"}, {"t2"}, {"t3"}, {"t4"}, {"t7"}, {"t6"}};
    std::vector<std::string> opNames{"AllocA1", "AllocA2",  "AllocB1", "AllocB2", "CopyInA",
                                     "AddA",    "CopyOutA", "CopyInB", "AddB"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);

    ASSERT_EQ(sorter.clusters_.size(), 2U);
    ASSERT_EQ(sorter.orderedNormals_.size(), 5U);
    EXPECT_EQ(sorter.orderedNormals_[0], b.GetOp("CopyInA")); // A 新簇首 op 无条件执行
    // addA net=+20KB, copy_inB net=+16KB<20KB -> ThreeWayChoose 跳转 copy_inB
    EXPECT_EQ(sorter.orderedNormals_[1], b.GetOp("CopyInB"));  // 胜出执行
    EXPECT_EQ(sorter.orderedNormals_[2], b.GetOp("AddB"));     // 黏簇直行 net=0
    EXPECT_EQ(sorter.orderedNormals_[3], b.GetOp("AddA"));     // breakQ pop 后执行
    EXPECT_EQ(sorter.orderedNormals_[4], b.GetOp("CopyOutA")); // 黏簇直行
}

// 用例 3b：三方比较不换簇。A 第二 op net>0，但 B 头 net 更大 -> A 不跳转，继续执行。
// 簇 A: copy_inA(t1->t2, 4KB) -> addA(t2->t3, 16KB) -> copy_outA(t3->t4 DDR)，basePeak=20KB。
//   addA net=+12KB>0 -> 三方比较。
// 簇 B: copy_inB(t5->t7, 32KB) -> addB(t7->t6, 4KB)，basePeak=36KB。
//   B 头=copy_inB net=+32KB > 12KB -> 不跳转，A 继续。A 完成后才轮到 B。
// 期望 orderedNormals_ = [copy_inA, addA, copy_outA, copy_inB, addB]。
TEST_F(ClusterListSortTest, ThreeWayChooseNoSwitch)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_DEVICE_DDR, MEM_DEVICE_DDR, MEM_UB, MEM_UB};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    // t2=4KB(32x32), t3=16KB(默认64x64), t7=32KB(128x64), t6=4KB(32x32)
    auto t2 = b.GetTensor("t2");
    t2->shape = {32, 32};
    t2->tensor->rawshape = {32, 32};
    auto t7 = b.GetTensor("t7");
    t7->shape = {128, 64};
    t7->tensor->rawshape = {128, 64};
    auto t6 = b.GetTensor("t6");
    t6->shape = {32, 32};
    t6->tensor->rawshape = {32, 32};

    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT, Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t5"}, {"t7"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t3"}, {"t6"}, {"t7"}, {"t2"}, {"t3"}, {"t4"}, {"t7"}, {"t6"}};
    std::vector<std::string> opNames{"AllocA1", "AllocA2",  "AllocB1", "AllocB2", "CopyInA",
                                     "AddA",    "CopyOutA", "CopyInB", "AddB"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);

    ASSERT_EQ(sorter.clusters_.size(), 2U);
    ASSERT_EQ(sorter.orderedNormals_.size(), 5U);
    EXPECT_EQ(sorter.orderedNormals_[0], b.GetOp("CopyInA")); // A 新簇首 op 无条件执行
    // addA net=+12KB, copy_inB net=+32KB>12KB -> 不跳转，A 继续
    EXPECT_EQ(sorter.orderedNormals_[1], b.GetOp("AddA"));
    EXPECT_EQ(sorter.orderedNormals_[2], b.GetOp("CopyOutA")); // 黏簇直行 net=-16KB
    // A 完成后才轮到 B
    EXPECT_EQ(sorter.orderedNormals_[3], b.GetOp("CopyInB")); // B 新簇首 op 无条件执行
    EXPECT_EQ(sorter.orderedNormals_[4], b.GetOp("AddB"));    // 黏簇直行 net=-28KB
}

// 用例 4：alloc 回插（多生产者写不同 memId + 未引用 alloc 放最前）。
// 图：t1(DDR) -> CopyIn1 -> t2(UB, 16KB)
//                                        -> Add -> t4(UB) -> CopyOut -> t5(DDR)
//     t3(DDR) -> CopyIn2 -> t6(UB, 32KB)
// Alloc1->t2, Alloc2->t4, Alloc3->t9(未引用), Alloc4->t6。t2 < t6 使 CopyIn1 簇排在 CopyIn2 前。
TEST_F(ClusterListSortTest, ReinsertAllocsMultiProducerAndLeftover)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t9"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_DEVICE_DDR, MEM_UB, MEM_DEVICE_DDR, MEM_UB, MEM_UB};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    // t2=16KB(64x64 默认), t6=32KB(128x64) -> CopyIn1 簇 basePeak 小排前面
    auto t6 = b.GetTensor("t6");
    t6->shape = {128, 64};
    t6->tensor->rawshape = {128, 64};
    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {}, {"t1"}, {"t3"}, {"t2", "t6"}, {"t4"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t4"}, {"t9"}, {"t6"}, {"t2"}, {"t6"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "CopyIn1", "CopyIn2", "Add", "CopyOut"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);

    // 建簇：CopyIn1(t2 16KB) / CopyIn2(t6 32KB) 各独立成簇，Add+CopyOut 合一簇
    ASSERT_EQ(sorter.clusters_.size(), 3U);
    // 按 ops 内容找到对应簇 id（cluster id 按 normals_ 遍历顺序分配，不按 basePeak）
    auto FindCid = [&](Operation* op) -> int {
        auto it = sorter.cluster_.find(op);
        return it == sorter.cluster_.end() ? -1 : it->second;
    };
    int cidCopyIn1 = FindCid(b.GetOp("CopyIn1"));
    int cidCopyIn2 = FindCid(b.GetOp("CopyIn2"));
    int cidAdd = FindCid(b.GetOp("Add"));
    EXPECT_GE(cidCopyIn1, 0);
    EXPECT_GE(cidCopyIn2, 0);
    EXPECT_GE(cidAdd, 0);
    auto& clCopyIn1 = sorter.clusters_[static_cast<size_t>(cidCopyIn1)];
    auto& clCopyIn2 = sorter.clusters_[static_cast<size_t>(cidCopyIn2)];
    auto& clAddCopyOut = sorter.clusters_[static_cast<size_t>(cidAdd)];
    ASSERT_EQ(clCopyIn1.ops.size(), 1U);
    EXPECT_EQ(clCopyIn1.ops[0], b.GetOp("CopyIn1"));
    ASSERT_EQ(clCopyIn2.ops.size(), 1U);
    EXPECT_EQ(clCopyIn2.ops[0], b.GetOp("CopyIn2"));
    ASSERT_EQ(clAddCopyOut.ops.size(), 2U);
    EXPECT_EQ(clAddCopyOut.ops[0], b.GetOp("Add"));
    // t2(16KB) < t6(32KB) -> CopyIn1 簇 basePeak < CopyIn2 簇 basePeak -> 调度时 CopyIn1 先
    EXPECT_LT(clCopyIn1.basePeak, clCopyIn2.basePeak);

    auto& result = sorter.operations;
    ASSERT_EQ(result.size(), 8U);
    // 未引用 alloc 放最前
    EXPECT_EQ(result[0], b.GetOp("Alloc3"));
    // alloc_t2 在 copy_in1 前
    EXPECT_TRUE(Before(result, b.GetOp("Alloc1"), b.GetOp("CopyIn1")));
    // alloc_t4 在 add 前
    EXPECT_TRUE(Before(result, b.GetOp("Alloc2"), b.GetOp("Add")));
    // copy_in1 在 copy_in2 前（t2 basePeak < t6 basePeak）
    EXPECT_TRUE(Before(result, b.GetOp("CopyIn1"), b.GetOp("CopyIn2")));
}

// 用例 5：空输入边界。
TEST_F(ClusterListSortTest, EmptyInput)
{
    ComputationalGraphBuilder b;
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);
    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);
    EXPECT_TRUE(sorter.operations.empty());
}

// 用例 6：多输出算子（fan-out）断簇。Add 写 2 个 UB tensor → IsChainEnd=true，应为簇末。
TEST_F(ClusterListSortTest, MultiOutputOpBreaksCluster)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_UB, MEM_DEVICE_DDR, MEM_DEVICE_DDR};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t3"}, {"t4"}, {"t2"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "CopyIn", "FanOutAdd", "CopyOut1", "CopyOut2"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);

    auto FindCid = [&](Operation* op) -> int {
        auto it = sorter.cluster_.find(op);
        return it == sorter.cluster_.end() ? -1 : it->second;
    };
    int cidCopyIn = FindCid(b.GetOp("CopyIn"));
    int cidFanOut = FindCid(b.GetOp("FanOutAdd"));
    int cidCopyOut1 = FindCid(b.GetOp("CopyOut1"));
    int cidCopyOut2 = FindCid(b.GetOp("CopyOut2"));
    EXPECT_GE(cidCopyIn, 0);
    EXPECT_GE(cidFanOut, 0);
    EXPECT_GE(cidCopyOut1, 0);
    EXPECT_GE(cidCopyOut2, 0);
    EXPECT_EQ(cidCopyIn, cidFanOut);
    EXPECT_NE(cidCopyIn, cidCopyOut1);
    EXPECT_NE(cidCopyIn, cidCopyOut2);
    auto& clFanOut = sorter.clusters_[static_cast<size_t>(cidFanOut)];
    ASSERT_GE(clFanOut.ops.size(), 2U);
    EXPECT_EQ(clFanOut.ops.front(), b.GetOp("CopyIn"));
    EXPECT_EQ(clFanOut.ops.back(), b.GetOp("FanOutAdd"));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("CopyIn"), b.GetOp("FanOutAdd")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("FanOutAdd"), b.GetOp("CopyOut1")));
    EXPECT_TRUE(Before(sorter.operations, b.GetOp("FanOutAdd"), b.GetOp("CopyOut2")));
}

// 用例 7：环形依赖 → RunScheduleLoop 返回 FAILED。
// 构造正常图后手动设置 unmetPreds_ 模拟环形依赖（所有 op 互相依赖 → 无 ready 簇头）。
TEST_F(ClusterListSortTest, CycleDependencyReturnsFailed)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_DEVICE_DDR};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ins{{}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t3"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "CopyIn", "Add", "CopyOut"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    sorter.InitRunState();
    ASSERT_EQ(sorter.BuildClusters(), SUCCESS);
    sorter.InitUnmetPreds();
    for (auto& [op, cnt] : sorter.unmetPreds_) {
        cnt = 1;
    }
    EXPECT_EQ(sorter.RunScheduleLoop(), FAILED);
}

// 用例 8：breakQ 队头胜出。addA 进 breakQ 后，下一轮 breakQ 队头 net 最小 → 从 breakQ 弹出执行。
// 簇 A: copy_inA(t1->t2, 4KB) -> addA(t2->t3, 32KB) -> copy_outA(t3->t4 DDR)，addA net=+28KB。
// 簇 D: copy_inD(t5->t6, 4KB) -> addD(t6->t7, 64KB)，copy_inD net=+4KB，addD net=+60KB。
// 调度: copy_inA 执行 → addA net>0, next head=copy_inD(4)<28 → addA 进 breakQ, copy_inD 胜出 →
//       addD net>0, breakQ 队头=addA(28)<60 → addA 从 breakQ 胜出执行。
TEST_F(ClusterListSortTest, ThreeWayChooseBreakQHeadWins)
{
    ComputationalGraphBuilder b;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> memTypes{MEM_DEVICE_DDR, MEM_UB, MEM_UB, MEM_DEVICE_DDR, MEM_DEVICE_DDR, MEM_UB, MEM_UB};
    EXPECT_TRUE(b.AddTensors(DT_FP32, {64, 64}, memTypes, tensorNames, 0));
    auto t2 = b.GetTensor("t2");
    t2->shape = {32, 32};
    t2->tensor->rawshape = {32, 32};
    auto t3 = b.GetTensor("t3");
    t3->shape = {128, 64};
    t3->tensor->rawshape = {128, 64};
    auto t6 = b.GetTensor("t6");
    t6->shape = {32, 32};
    t6->tensor->rawshape = {32, 32};
    auto t7 = b.GetTensor("t7");
    t7->shape = {256, 64};
    t7->tensor->rawshape = {256, 64};

    std::vector<Opcode> opcodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT, Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ins{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t5"}, {"t6"}};
    std::vector<std::vector<std::string>> outs{{"t2"}, {"t3"}, {"t6"}, {"t7"}, {"t2"}, {"t3"}, {"t4"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"AllocA1", "AllocA2",  "AllocD1", "AllocD2", "CopyInA",
                                     "AddA",    "CopyOutA", "CopyInD", "AddD"};
    EXPECT_TRUE(b.AddOps(opcodes, ins, outs, opNames, true));
    Function* func = b.GetFunction();
    ASSERT_NE(func, nullptr);

    auto ops = func->Operations().DuplicatedOpList();
    ClusterListSort sorter(ops, *func);
    sorter.state_.Init(ops);
    EXPECT_EQ(sorter.DoSortOps(), SUCCESS);

    ASSERT_EQ(sorter.clusters_.size(), 2U);
    ASSERT_EQ(sorter.orderedNormals_.size(), 5U);
    EXPECT_EQ(sorter.orderedNormals_[0], b.GetOp("CopyInA"));
    EXPECT_EQ(sorter.orderedNormals_[1], b.GetOp("CopyInD"));
    EXPECT_EQ(sorter.orderedNormals_[2], b.GetOp("AddA"));
    EXPECT_EQ(sorter.orderedNormals_[3], b.GetOp("CopyOutA"));
    EXPECT_EQ(sorter.orderedNormals_[4], b.GetOp("AddD"));
}

} // namespace npu::tile_fwk
