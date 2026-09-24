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
 * \file test_add_alloc.cpp
 * \brief Unit test for AddAlloc pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/add_alloc.h"
#include "ut_json/ut_json_tool.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {
class AddAllocTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AddAllocTest, TestAddAlloc)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD,     Opcode::OP_ADD,
                                Opcode::OP_ADD,     Opcode::OP_ADD,     Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"},       {"t2"},       {"t3", "t4"}, {"t3", "t4"},
                                                    {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Add4", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    AddAlloc addalloc;
    Status res = addalloc.AddAndCheckAlloc(*function);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), 13);
}

TEST_F(AddAllocTest, TestAddAllocInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,     Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1"},       {"t2"},       {"t3"},       {"t4"},
                                                    {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    AddAlloc addalloc;
    Status res = addalloc.AddAndCheckAlloc(*function);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), 12);
}

TEST_F(AddAllocTest, TestAddAllocAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
                                Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"},       {"t4"},
                                                    {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Copyin1",   "Copyin2",   "Copyin3", "Copyin4",
                                     "Assemble1", "Assemble2", "Add1",    "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    AddAlloc addalloc;
    Status res = addalloc.AddAndCheckAlloc(*function);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), 13);
}

TEST_F(AddAllocTest, TestAddAllocView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW,
                                Opcode::OP_ADD,     Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    AddAlloc addalloc;
    Status res = addalloc.AddAndCheckAlloc(*function);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), 12);
}

TEST_F(AddAllocTest, TestAddAllocErrorMemId)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW,
                                Opcode::OP_ADD,     Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = -1;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    AddAlloc addalloc;
    Status res = addalloc.AddAndCheckAlloc(*function);
    EXPECT_EQ(res, FAILED);
}

TEST_F(AddAllocTest, TestAddAllocErrorMemorymap)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW,
                                Opcode::OP_ADD,     Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = -1;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    AddAlloc addalloc;
    Status res = addalloc.AddAndCheckAlloc(*function);
    EXPECT_EQ(res, FAILED);
}
} // namespace tile_fwk
} // namespace npu
