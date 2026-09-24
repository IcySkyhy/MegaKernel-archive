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
 * \file test_program_abort_recording.cpp
 * \brief Unit tests for incomplete loop-path recording cleanup (AbandonIncompleteRecording / AbortRecording).
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/inner/tile_shape.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

namespace {
constexpr int64_t kVecTile = 16;
constexpr int64_t kRows = 16;
constexpr int64_t kCols = 64;
constexpr int64_t kChildRows = 16;
constexpr int64_t kChildCols = 16;
} // namespace

class ProgramAbortRecordingTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        TileShape::Current().SetVecTile(kVecTile, kVecTile);
    }

    void TearDown() override
    {
        if (Program::GetInstance().HasIncompleteRecordingState()) {
            Program::GetInstance().AbandonIncompleteRecording();
        }
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(ProgramAbortRecordingTest, CleanStateHasNoIncompleteRecording)
{
    EXPECT_FALSE(Program::GetInstance().HasIncompleteRecordingState());
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 1U);
}

TEST_F(ProgramAbortRecordingTest, AbandonIncompleteRecordingIsNoOpWhenClean)
{
    Program::GetInstance().AbandonIncompleteRecording();
    Program::GetInstance().AbandonIncompleteRecording();
    EXPECT_FALSE(Program::GetInstance().HasIncompleteRecordingState());
}

TEST_F(ProgramAbortRecordingTest, ActiveCheckpointMarksIncompleteState)
{
    auto manager = Program::GetInstance().GetTensorSlotManager();
    manager->Checkpoint();
    EXPECT_TRUE(manager->HasActiveCheckpoints());
    EXPECT_TRUE(Program::GetInstance().HasIncompleteRecordingState());

    Program::GetInstance().AbandonIncompleteRecording();
    EXPECT_FALSE(Program::GetInstance().HasIncompleteRecordingState());
}

TEST_F(ProgramAbortRecordingTest, PartialDynamicLoopLeavesIncompleteState)
{
    std::vector<int64_t> shape{kRows, kCols};
    std::vector<int64_t> childShape{kChildRows, kChildCols};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");

    RecordFunc record("abort_main", {a, b}, {c});
    EXPECT_TRUE(Program::GetInstance().HasIncompleteRecordingState());
    EXPECT_GT(Program::GetInstance().FunctionMapSize(), 1U);

    RecordLoopFunc loop("L0", FunctionType::DYNAMIC_LOOP, "i", LoopRange(4));
    EXPECT_FALSE(Program::GetInstance().GetLoopStack().empty());

    record.AbortRecording();
    EXPECT_FALSE(Program::GetInstance().HasIncompleteRecordingState());
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 1U);
    EXPECT_TRUE(Program::GetInstance().GetLoopStack().empty());
}

TEST_F(ProgramAbortRecordingTest, AbortRecordingIsIdempotent)
{
    std::vector<int64_t> shape{kRows, kCols};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");

    RecordFunc record("abort_twice", {a, b}, {c});
    RecordLoopFunc loop("L0", FunctionType::DYNAMIC_LOOP, "i", LoopRange(2));

    record.AbortRecording();
    record.AbortRecording();
    EXPECT_FALSE(Program::GetInstance().HasIncompleteRecordingState());
}

TEST_F(ProgramAbortRecordingTest, RecordLoopFuncDestructorAfterStackCleared)
{
    {
        RecordLoopFunc loop("L0", FunctionType::DYNAMIC_LOOP, "i", LoopRange(1));
        EXPECT_FALSE(Program::GetInstance().GetLoopStack().empty());
        Program::GetInstance().GetLoopStack().clear();
    }
    EXPECT_TRUE(Program::GetInstance().GetLoopStack().empty());
}

TEST_F(ProgramAbortRecordingTest, AbortRecordingAfterPartialLoopBody)
{
    std::vector<int64_t> shape{kRows, kCols};
    std::vector<int64_t> childShape{kChildRows, kChildCols};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");

    RecordFunc record("abort_mid_loop", {a, b}, {c});
    bool aborted = false;
    LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(4))
    {
        (void)i;
        auto a0 = View(a, childShape, {0, 0});
        auto b0 = View(b, childShape, {0, 0});
        auto c0 = Add(a0, b0);
        Assemble(c0, {0, 0}, c);
        if (!aborted) {
            record.AbortRecording();
            aborted = true;
            break;
        }
    }
    EXPECT_TRUE(aborted);
    EXPECT_FALSE(Program::GetInstance().HasIncompleteRecordingState());
    EXPECT_EQ(Program::GetInstance().FunctionMapSize(), 1U);
}
