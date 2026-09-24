/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "passes/pass_mgr/pass_manager.h"

namespace npu::tile_fwk {

TEST(PassTest, InferMemoryConflictInferOutputWriteConflict)
{
    std::vector<int64_t> shape{-1, 32};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");

    Function* loopFunc = nullptr;
    FUNCTION("MAIN", {a, b, c})
    {
        TileShape::Current().SetVecTile(32, 32);
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(32))
        {
            loopFunc = Program::GetInstance().GetCurrentFunction();
            auto t = Full(1.0, DT_FP32, {32, 32});
            Assemble(t, {idx * 32, 0}, a); // assemble of a not overlap, no conflict
            Assemble(t, {(idx + 1) * 32, 0}, a);

            AtomicRMW(t, {idx * 32, 0}, c, AtomicRMWMode::ADD); // atomic always mark as conflict

            Assemble(t, {idx * 32, 0}, b); // assemble of b overlaped, should be conflict
            Assemble(t, {idx * 32 + 16, 0}, b);
        }
    }
    int cnt = 0;
    PassManager::Instance().RegisterStrategy("InferOutputWriteConflictTestStrategy",
                                             {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(
        PassManager::Instance().RunPass(Program::GetInstance(), *loopFunc, "InferOutputWriteConflictTestStrategy"),
        SUCCESS);
    for (auto out : loopFunc->GetOriginOutcast()) {
        if (out->tensor->GetSymbol() == "a") {
            EXPECT_FALSE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        } else if (out->tensor->GetSymbol() == "b") {
            EXPECT_TRUE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        } else if (out->tensor->GetSymbol() == "c") {
            EXPECT_TRUE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        }
    }
    EXPECT_EQ(cnt, 3);
}

TEST(PassTest, InferMemoryConflictAssembleParallelFalseMarksNormalAndCloneKeepsIt)
{
    std::vector<int64_t> shape{-1, 32};
    Tensor serialOut(DT_FP32, shape, "serial_out");
    Tensor parallelOut(DT_FP32, shape, "parallel_out");

    Function* loopFunc = nullptr;
    FUNCTION("MAIN", {serialOut, parallelOut})
    {
        TileShape::Current().SetVecTile(32, 32);
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(32))
        {
            loopFunc = Program::GetInstance().GetCurrentFunction();
            auto t = Full(1.0, DT_FP32, {32, 32});
            // Cross-loop WAW serial write: parallel=false should mark outcast NORMAL.
            Assemble(t, {idx * 32, 0}, serialOut, false);
            // Default parallel=true should not mark NORMAL.
            Assemble(t, {idx * 32, 0}, parallelOut, true);
        }
    }

    const std::string strategyName = "InferAssembleNormalAttrTestStrategy";
    PassManager::Instance().RegisterStrategy(strategyName, {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(PassManager::Instance().RunPass(Program::GetInstance(), *loopFunc, strategyName), SUCCESS);

    int matched = 0;
    for (auto out : loopFunc->GetOriginOutcast()) {
        if (out->tensor->GetSymbol() == "serial_out") {
            EXPECT_TRUE(out->HasAttr("NORMAL"));
            EXPECT_FALSE(out->HasAttr(OpAttributeKey::writeConflict));
            auto cloned = out->Clone(*loopFunc, true);
            ASSERT_NE(cloned, nullptr);
            EXPECT_TRUE(cloned->HasAttr("NORMAL")) << "NORMAL should be inherited by LogicalTensor::Clone";
            matched++;
        } else if (out->tensor->GetSymbol() == "parallel_out") {
            EXPECT_FALSE(out->HasAttr("NORMAL"));
            matched++;
        }
    }
    EXPECT_EQ(matched, 2);
}
} // namespace npu::tile_fwk
