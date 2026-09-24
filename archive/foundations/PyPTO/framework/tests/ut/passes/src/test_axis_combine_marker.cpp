/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_axis_combine_marker.cpp
 * \brief Unit test for AxisCombineMarker.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine_marker.h"
#include "computational_graph_builder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;
constexpr size_t K_1 = 1;
constexpr size_t K_2 = 2;
constexpr size_t K_4 = 4;
constexpr size_t K_8 = 8;
constexpr size_t K_16 = 16;
constexpr size_t K_32 = 32;
constexpr size_t K_64 = 64;
constexpr size_t K_128 = 128;

class TestAxisCombineMarker : public ::testing::Test {
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

/*
before:
    copyin
    [8,1]
      |
    copyout
    [8,1]

after:
    Tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, basic_copyin_enable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as ENABLE
    auto t1 = graph.GetTensor("t1");
    auto t2 = graph.GetTensor("t2");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t1), false); // DDR tensor should not be marked
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), true);  // UB tensor with last dim=1 should be enabled
}

/*
before:
    copyin
    [8,16]
      |
    copyout
    [8,16]

after:
    Tensor should be marked as UNKNOWN
*/
TEST_F(TestAxisCombineMarker, basic_copyin_unknown)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as UNKNOWN
    auto t1 = graph.GetTensor("t1");
    auto t2 = graph.GetTensor("t2");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t1), false); // DDR tensor should not be marked
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), false); // UB tensor with last dim != 1 should be unknown
}

/*
before:
    copyin
    [8,15]
      |
    view
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, view_enable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "view", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false); // View output with last dim=1 should be enabled
}

/*
before:
    copyin
    [8,15]
      |
    view
    [8,16]
      |
    copyout
    [8,16]

after:
    Output tensor should be marked as UNKNOWN
*/
TEST_F(TestAxisCombineMarker, view_unknown)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "view", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as UNKNOWN
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false); // View output with last dim != 1 should be unknown
}

/*
before:
    copyin
    [8,1]
      |
    assemble
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, assemble_enable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t2"}, {"t3"}, "assemble", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true); // Assemble output with same shape should be enabled
}

/*
before:
    copyin
    [8,1]
      |
    assemble
    [8,16]
      |
    copyout
    [8,16]

after:
    Both tensors should be marked as DISABLE
*/
TEST_F(TestAxisCombineMarker, assemble_disable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t2"}, {"t3"}, "assemble", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as DISABLE
    auto t2 = graph.GetTensor("t2");
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), false); // Assemble input should be disabled
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false); // Assemble output should be disabled
}

/*
before:
    copyin
    [8,1]
      |
    expand
    [8,1,1]
      |
    copyout
    [8,1,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, expand_non_last_axis)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t2"}, {"t3"}, "expand", true), true);
    auto expand_op = graph.GetOp("expand");
    expand_op->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0}); // Expand non-last axis

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true); // Expand on non-last axis should be enabled
}

/*
before:
    copyin
    [8,1]
      |
    expand
    [8,16]
      |
    copyout
    [8,16]

after:
    Input tensor should be marked as DISABLE, output should be UNKNOWN
*/
TEST_F(TestAxisCombineMarker, expand_last_axis)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t2"}, {"t3"}, "expand", true), true);
    auto expand_op = graph.GetOp("expand");
    expand_op->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{1}); // Expand last axis

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked correctly
    auto t2 = graph.GetTensor("t2");
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), false); // Input should be disabled
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false); // Output should be unknown (not enabled)
}

/*
before:
    copyin
    [8,1,16]
      |
    reduce
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, reduce_last_axis)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 16}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 1}, MemoryType::MEM_DEVICE_DDR, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWMAX_SINGLE, {"t2"}, {"t3"}, "reduce", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"t4"}, "copy_out", true), true);
    auto reduce_op = graph.GetOp("reduce");
    reduce_op->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2); // Reduce last axis

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true); // Reduce on last axis should be enabled
}

/*
before:
    copyin
    [8,1,16]
      |
    reduce
    [8,16]
      |
    copyout
    [8,16]

after:
    Output tensor should be marked as DISABLE
*/
TEST_F(TestAxisCombineMarker, reduce_second_last_axis)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 8, 16}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 8, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWMAX_SINGLE, {"t2"}, {"t3"}, "reduce", true), true);
    auto reduce_op = graph.GetOp("reduce");
    reduce_op->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1); // Reduce second last axis

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as DISABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false); // Reduce on second last axis should be disabled
}

/*
before:
    copyin     copyin
    [8,1]      [8,1]
      \         /
        add
       [8,1]
         |
      copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, elewise_enable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as ENABLE
    auto t5 = graph.GetTensor("t5");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), true); // Elewise output with last dim=1 should be enabled
}

/*
before:
    copyin     copyin
    [8,1]      [8,16]
      \         /
        add
       [8,16]
         |
      copyout
    [8,16]

after:
    Output tensor should be marked as UNKNOWN
*/
TEST_F(TestAxisCombineMarker, elewise_unknown)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as UNKNOWN
    auto t5 = graph.GetTensor("t5");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), false); // Elewise output with last dim != 1 should be unknown
}

/*
before:
    copyin     copyin
    [8,1]      [8,1]
      \         /
        add
       [8,1]
         |
      view
     [8,1]
       |
     copyout
    [8,1]

after:
    All tensors should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, complex_graph_enable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t5"}, {"t6"}, "view", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify all tensors are marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    auto t4 = graph.GetTensor("t4");
    auto t5 = graph.GetTensor("t5");
    auto t6 = graph.GetTensor("t6");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t4), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t6), true);
}

/*
before:
    copyin     copyin
    [8,1]      [8,1]
      \         /
        add
       [8,1]
         |
     assemble
     [8,16]
       |
     copyout
    [8,16]

after:
    Tensors before assemble should be ENABLE, after assemble should be DISABLE
*/
TEST_F(TestAxisCombineMarker, complex_graph_disable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t5"}, {"t6"}, "assemble", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify tensors before assemble are ENABLE, after assemble are DISABLE
    auto t3 = graph.GetTensor("t3");
    auto t4 = graph.GetTensor("t4");
    auto t5 = graph.GetTensor("t5");
    auto t6 = graph.GetTensor("t6");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false); // Should be disabled due to assemble
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t4), false); // Should be disabled due to assemble
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), false); // Should be disabled due to assemble
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t6), false); // Should be disabled
}

/*
before:
    copyin
    [8,1]
      |
     expand
    [8,1,1]
      |
    reduce
    [8,1]
      |
    copyout
    [8,1]

after:
    All tensors should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, expand_reduce_chain)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_DEVICE_DDR, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t2"}, {"t3"}, "expand", true), true);
    auto expand_op = graph.GetOp("expand");
    expand_op->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0}); // Expand non-last axis
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWMAX_SINGLE, {"t3"}, {"t4"}, "reduce", true), true);
    auto reduce_op = graph.GetOp("reduce");
    reduce_op->SetAttribute(OP_ATTR_PREFIX + "AXIS", 0); // Reduce last axis
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t4"}, {"t5"}, "copy_out", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify all tensors are marked as ENABLE
    auto t2 = graph.GetTensor("t2");
    auto t3 = graph.GetTensor("t3");
    auto t4 = graph.GetTensor("t4");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t4), true);
}

/*
before:
    copyin
    [8,1]
      |
    reshape
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as UNKNOWN (reshape is not handled)
*/
TEST_F(TestAxisCombineMarker, unhandled_op)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"t2"}, {"t3"}, "reshape", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // Verify the tensor is marked as UNKNOWN
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false); // Unhandled op should be unknown
}

TEST_F(TestAxisCombineMarker, cast_op)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 32, 32}, MemoryType::MEM_DEVICE_DDR, "gm"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 32, 32}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm"}, {"t1"}, "copy_in", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 32, 1}, MemoryType::MEM_UB, "r1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"t1"}, {"r1"}, "reduce", true), true);
    auto reduceOp = graph.GetOp("reduce");
    reduceOp->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 32, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VEC_DUP, {}, {"t2"}, "vec_dup", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 32, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_DIV, {"r1", "t2"}, {"t3"}, "div", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 32, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"t3"}, {"t4"}, "cast", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 32, 1}, MemoryType::MEM_DEVICE_DDR, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t4"}, {"out"}, "copyout", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    auto t4 = graph.GetTensor("t4");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t4), true);
}

// QA backward case
TEST_F(TestAxisCombineMarker, qaCase)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"t1"}, "copy_in1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"t2"}, "copy_in2", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "b1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRSUM, {"t1", "t2"}, {"b1"}, "pairsum", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t3"), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUMLINE, {"b1"}, {"t3"}, "rowsumline", true), true);
    auto reduceOp = graph.GetOp("rowsumline");
    reduceOp->SetAttribute(OP_ATTR_PREFIX + "AXIS", 0);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"out"}, "copyout", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("t3")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("b1")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("t1")), false);
}

// op not in whitelist case
TEST_F(TestAxisCombineMarker, transpose_op)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 16}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 16}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"t1"}, "copy_in1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1, 2}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1, 2}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"t2"}, "copy_in2", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_TRANSPOSE_VNCHWCONV, {"t2"}, {"t3"}, "transpose", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 16}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t1", "t3"}, {"t4"}, "sub", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t4"}, {"out"}, "copyout", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("t1")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("t3")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("t4")), false);
}

// pad conflict case
TEST_F(TestAxisCombineMarker, pad_conflict)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"t1"}, "copy_in1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 128}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_TRANSPOSE_VNCHWCONV, {"t1"}, {"t2"}, "transpose", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t1", "t2"}, {"t3"}, "sub", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t3"}, {"out"}, "copyout", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("t1")), false);
}

/*
before:
     copyin         copyin        copyin
   [1,2,32]       [1,2,32]      [1,2,1]
       |              |             |
      v1             v2            v3
   [1,2,32]       [1,2,32]      [1,2,1]
       \            /              |
        \          /               |
         pairsum                  /
         [1,2,32]                /
            |                   /
            |                  /
             \                /
               pairsum
               [1,2,32]
                  |
            rowsum_single
               [1,2,1]
               axis=2
                  |
               copyout
               [1,6,1]

after: var尾块场景 - v3[1,2,1]为ENABLE，PAIRSUM输出tail dim=32不触发pair ops检查，
       rowsum_single尾轴reduce(axis=2)为ENABLE，copyout结果为[1,6,1]
*/
TEST_F(TestAxisCombineMarker, var_tail_block)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 32}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 32}, MemoryType::MEM_UB, "v1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"v1"}, "copy_in1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 32}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 32}, MemoryType::MEM_UB, "v2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"v2"}, "copy_in2", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 1}, MemoryType::MEM_DEVICE_DDR, "gm3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 1}, MemoryType::MEM_UB, "v3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm3"}, {"v3"}, "copy_in3", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 32}, MemoryType::MEM_UB, "b1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRSUM, {"v1", "v2"}, {"b1"}, "pairsum1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 32}, MemoryType::MEM_UB, "b2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRSUM, {"b1", "v3"}, {"b2"}, "pairsum2", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 1}, MemoryType::MEM_UB, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"b2"}, {"out"}, "reduce", true), true);
    auto reduce_op = graph.GetOp("reduce");
    reduce_op->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 6, 1}, MemoryType::MEM_DEVICE_DDR, "output"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"out"}, {"output"}, "copy_out", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    // v1, v2: COPY_IN from DDR [1,2,32] → output tail dim 32 → UNKNOWN
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("v1")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("v2")), false);
    // v3: COPY_IN from DDR [1,2,1] → output tail dim 1 → DISABLE
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("v3")), false);
    // b1: PAIRSUM output [1,2,32] tail dim 32 → UNKNOWN
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("b1")), false);
    // b2: PAIRSUM output [1,2,32] tail dim 32 → UNKNOWN (pair ops check not reached)
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("b2")), false);
    // out: ROWSUM_SINGLE axis=2 reduce last axis → ENABLE
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("out")), true);
}

TEST_F(TestAxisCombineMarker, multi_output_elewise)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "gm0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "ub0"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm0"}, {"ub0"}, "copy_in0", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "ub1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"ub1"}, "copy_in1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "ub2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"ub2"}, "copy_in2", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "gm3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "ub3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm3"}, {"ub3"}, "copy_in3", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "pair_val"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "pair_idx"), true);
    EXPECT_EQ(
        graph.AddOp(Opcode::OP_PAIRARGMAX, {"ub0", "ub1", "ub2", "ub3"}, {"pair_val", "pair_idx"}, "pairargmax", true),
        true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "cast_out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_CAST, {"pair_idx"}, {"cast_out"}, "cast", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"cast_out"}, {"out"}, "copy_out", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("ub0")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("ub1")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("ub2")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("ub3")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("pair_val")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("pair_idx")), false);
}

/*
Test var mean 阶段的 PadLocalBuffer 行为
输入 [16,2,1], DIVS(64) -> COPY_OUT -> COPY_IN -> PAIRSUM
  -> ROWSUMLINE(axis=0) -> ROWSUMLINE(axis=1) -> ROWSUM_SINGLE(axis=2) -> mean([1,1,1])
验证 reduce 链DISABLE属性正确传递
*/
TEST_F(TestAxisCombineMarker, axiscombineVarReduceChain)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_DEVICE_DDR, "input"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_UB, "ci0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_UB, "ci1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_UB, "div0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_UB, "div1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_DEVICE_DDR, "sum_gm0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_DEVICE_DDR, "sum_gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_UB, "sum_ci0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_UB, "sum_ci1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 2, 1}, MemoryType::MEM_UB, "pairsum0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 2, 1}, MemoryType::MEM_UB, "rsl0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 8}, MemoryType::MEM_UB, "rsl0_tmp"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1, 1}, MemoryType::MEM_UB, "rsl1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8}, MemoryType::MEM_UB, "rsl1_tmp"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1, 1}, MemoryType::MEM_UB, "rss0"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8}, MemoryType::MEM_UB, "rss0_tmp"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1, 1}, MemoryType::MEM_DEVICE_DDR, "mean_gm"), true);

    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"input"}, {"ci0"}, "ci0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"input"}, {"ci1"}, "ci1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_DIVS, {"ci0"}, {"div0"}, "div0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_DIVS, {"ci1"}, {"div1"}, "div1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"div0"}, {"sum_gm0"}, "co0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"div1"}, {"sum_gm1"}, "co1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"sum_gm0"}, {"sum_ci0"}, "sci0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"sum_gm1"}, {"sum_ci1"}, "sci1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_PAIRSUM, {"sum_ci0", "sum_ci1"}, {"pairsum0"}, "ps0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUMLINE, {"pairsum0"}, {"rsl0", "rsl0_tmp"}, "rsl0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUMLINE, {"rsl0"}, {"rsl1", "rsl1_tmp"}, "rsl1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"rsl1"}, {"rss0", "rss0_tmp"}, "rss0", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"rss0"}, {"mean_gm"}, "mean_co", true), true);

    graph.GetOp("div0")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 64.0));
    graph.GetOp("div1")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 64.0));
    graph.GetOp("rsl0")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 0);
    graph.GetOp("rsl1")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    graph.GetOp("rss0")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2);

    auto* functionPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*functionPtr);

    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("pairsum0")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("rsl0")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("rsl0_tmp")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("rsl1")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("rsl1_tmp")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("rss0")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("rss0_tmp")), false);
}

/*
ROWARGMAXWITHVALUE_SINGLE 多输出对齐场景：
    copyin
    [10,896]
       |
  rowargmaxwithvalue_single  (axis=1, 尾轴 reduce, 3 输出)
   /     |     \
 v[10,1] i[10,1] t[10,896]
          |
         ADDS
        [10,1]
          |
        copyout
*/
void RunMultiOutputAlignAxisReorderStatusTest(const std::vector<int64_t>& gmOutShape, bool expectEnable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {10, 896}, MemoryType::MEM_DEVICE_DDR, "gm_in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {10, 896}, MemoryType::MEM_UB, "ub_in"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_in"}, {"ub_in"}, "copy_in", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {10, 1}, MemoryType::MEM_UB, "argmax_val"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {10, 1}, MemoryType::MEM_UB, "argmax_idx"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {10, 896}, MemoryType::MEM_UB, "argmax_tmp"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, {"ub_in"}, {"argmax_val", "argmax_idx", "argmax_tmp"},
                          "argmax", true),
              true);
    graph.GetOp("argmax")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {10, 1}, MemoryType::MEM_UB, "adds_out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADDS, {"argmax_idx"}, {"adds_out"}, "adds", true), true);
    graph.GetOp("adds")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 0.0));

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, gmOutShape, MemoryType::MEM_DEVICE_DDR, "gm_out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"adds_out"}, {"gm_out"}, "copy_out", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("argmax_val")), expectEnable);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("argmax_idx")), expectEnable);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("adds_out")), expectEnable);
}

TEST_F(TestAxisCombineMarker, multi_output_align_status1) { RunMultiOutputAlignAxisReorderStatusTest({10, 1}, true); }

TEST_F(TestAxisCombineMarker, multi_output_align_status2) { RunMultiOutputAlignAxisReorderStatusTest({-1, -1}, false); }

/*
图结构:
    copyin [8,16]    copyin [8,1]
         |              |
        SUB [8,16]
          |
    rowargmaxwithvalue_single (axis=1, 3 outputs)
    /     |        \
  v[8,1] i[8,1]  t[8,896]
    |       |
   ADDS   ADDS
   [8,1]  [8,1]
     |
   copyout (dynamic shape [-1,-1] triggers rawShape mismatch)
            adds_idx → SUB(另外一路copyin [8,1]) [8,1] → copyout [8,1]
*/
TEST_F(TestAxisCombineMarker, multi_output_align_status3)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_DEVICE_DDR, "gm_in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "ub_in"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_in"}, {"ub_in"}, "copy_in", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "gm_sub1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "ub_sub1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_sub1"}, {"ub_sub1"}, "copy_in_sub1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "sub_in"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"ub_in", "ub_sub1"}, {"sub_in"}, "sub_in", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "val"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "idx"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 896}, MemoryType::MEM_UB, "tmp"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, {"sub_in"}, {"val", "idx", "tmp"}, "argmax", true),
              true);
    graph.GetOp("argmax")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "adds_val"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADDS, {"val"}, {"adds_val"}, "adds_val", true), true);
    graph.GetOp("adds_val")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 0.0));

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "adds_idx"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADDS, {"idx"}, {"adds_idx"}, "adds_idx", true), true);
    graph.GetOp("adds_idx")->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 0.0));

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {-1, -1}, MemoryType::MEM_DEVICE_DDR, "gm_out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"adds_val"}, {"gm_out"}, "copy_out", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "gm_sub2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "ub_sub2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm_sub2"}, {"ub_sub2"}, "copy_in_sub2", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "sub_idx"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"adds_idx", "ub_sub2"}, {"sub_idx"}, "sub_idx", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "gm_out2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"sub_idx"}, {"gm_out2"}, "copy_out2", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);

    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("val")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("idx")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("adds_val")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("adds_idx")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("sub_idx")), false);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(graph.GetTensor("ub_sub2")), false);
}
