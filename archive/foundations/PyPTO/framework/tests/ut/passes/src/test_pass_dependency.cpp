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
 * \file test_pass_dependency.cpp
 * \brief Unit test for PassDependency.
 */

#include <gtest/gtest.h>
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_dependency.h"
#include "interface/configs/config_manager.h"

namespace npu {
namespace tile_fwk {

class TestPassDependency : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(TestPassDependency, TestCheckStrategyDependency)
{
    PassDependency& passDependency = PassDependency::Instance();

    std::vector<PassName> normalPasses = {
        PassName::DUPLICATE_OP,     PassName::SPLIT_LARGE_FANOUT_TENSOR, PassName::SPLIT_RESHAPE,
        PassName::PROCESS_ATOMIC,   PassName::GRAPH_PARTITION,           PassName::REDUCE_COPY_MERGE,
        PassName::N_BUFFER_MERGE,   PassName::L1_COPY_IN_REUSE_MERGE,    PassName::INTRA_SUBGRAPH_ADAPTER,
        PassName::GENERATE_MOVE_OP, PassName::PRE_GRAPH_PROCESS,         PassName::REPLACE_TENSOR,
        PassName::INFER_DYN_SHAPE,  PassName::SUBGRAPH_TO_FUNCTION};
    // GraphPartition缺少前置DuplicateOp
    std::vector<PassName> passesLessPreDependency = {
        PassName::SPLIT_LARGE_FANOUT_TENSOR, PassName::SPLIT_RESHAPE,          PassName::PROCESS_ATOMIC,
        PassName::GRAPH_PARTITION,           PassName::REDUCE_COPY_MERGE,      PassName::N_BUFFER_MERGE,
        PassName::L1_COPY_IN_REUSE_MERGE,    PassName::INTRA_SUBGRAPH_ADAPTER, PassName::GENERATE_MOVE_OP};
    // ProcessAtomic重复
    std::vector<PassName> passesConsecutiveDup = {PassName::DUPLICATE_OP,           PassName::SPLIT_LARGE_FANOUT_TENSOR,
                                                  PassName::SPLIT_RESHAPE,          PassName::PROCESS_ATOMIC,
                                                  PassName::PROCESS_ATOMIC,         PassName::GRAPH_PARTITION,
                                                  PassName::REDUCE_COPY_MERGE,      PassName::N_BUFFER_MERGE,
                                                  PassName::L1_COPY_IN_REUSE_MERGE, PassName::INTRA_SUBGRAPH_ADAPTER,
                                                  PassName::GENERATE_MOVE_OP};
    // GraphPartition前调用L1CopyInReuseMerge
    std::vector<PassName> passesL1CopyBeforeGraphPartition = {
        PassName::DUPLICATE_OP,           PassName::SPLIT_LARGE_FANOUT_TENSOR,
        PassName::SPLIT_RESHAPE,          PassName::PROCESS_ATOMIC,
        PassName::L1_COPY_IN_REUSE_MERGE, PassName::GRAPH_PARTITION,
        PassName::REDUCE_COPY_MERGE,      PassName::N_BUFFER_MERGE,
        PassName::L1_COPY_IN_REUSE_MERGE, PassName::INTRA_SUBGRAPH_ADAPTER,
        PassName::GENERATE_MOVE_OP};

    EXPECT_EQ(passDependency.CheckStrategyDependency("normalPasses", normalPasses), SUCCESS);
    EXPECT_EQ(passDependency.CheckStrategyDependency("passesLessPreDependency", passesLessPreDependency), WARNING);
    EXPECT_EQ(passDependency.CheckStrategyDependency("passesConsecutiveDup", passesConsecutiveDup), WARNING);
    EXPECT_EQ(
        passDependency.CheckStrategyDependency("passesL1CopyBeforeGraphPartition", passesL1CopyBeforeGraphPartition),
        WARNING);
}

TEST_F(TestPassDependency, TestStrategySequenceDependency)
{
    PassDependency& passDependency = PassDependency::Instance();
    // GraphPartition后缺少ReduceConpyMerge
    std::vector<PassName> lessSequenceDependency = {PassName::DUPLICATE_OP,
                                                    PassName::SPLIT_LARGE_FANOUT_TENSOR,
                                                    PassName::SPLIT_RESHAPE,
                                                    PassName::PROCESS_ATOMIC,
                                                    PassName::GRAPH_PARTITION,
                                                    PassName::N_BUFFER_MERGE,
                                                    PassName::L1_COPY_IN_REUSE_MERGE,
                                                    PassName::INTRA_SUBGRAPH_ADAPTER,
                                                    PassName::GENERATE_MOVE_OP,
                                                    PassName::COMMON_OPERATION_ELIMINATE,
                                                    PassName::AXIS_COMBINE};

    EXPECT_EQ(passDependency.CheckStrategyDependency("lessSequenceDependency", lessSequenceDependency), WARNING);
}
} // namespace tile_fwk
} // namespace npu
