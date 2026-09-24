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
 * \file test_loop_unroll.cpp
 * \brief Unit test for LoopUnroll pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/interpreter/raw_tensor_data.h"
#define private public
#include "passes/tensor_graph_pass/loop_unroll.h"
#undef private
#include "computational_graph_builder.h"

#include <fstream>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {
class LoopUnrollTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "FunctionUnroll");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
        std::vector<std::string> funcName = {"TENSOR_main"};
        config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
        int s = 32;
        TileShape::Current().SetVecTile(s, s);
        TileShape::Current().SetCubeTile({s, s}, {s, s}, {s, s});
    }
    void TearDown() override {}
};

// Test nested LOOPs
TEST_F(LoopUnrollTest, TestInnerLoopOrder)
{
    ConfigManager::Instance();
    int vecLen = 128;
    int loopNum = 5;
    int tileNum = 4;
    Tensor inputA(DT_FP32, {loopNum, vecLen}, "inputA");
    Tensor inputB(DT_FP32, {tileNum, vecLen}, "inputB");
    Tensor output(DT_FP32, {1, vecLen}, "out");

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("Outer", FunctionType::DYNAMIC_LOOP, i, LoopRange(tileNum))
        {
            Tensor tileB(DT_FP32, {1, vecLen}, "tileB");
            LOOP("Inner", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
            {
                (void)j;
                auto tile = View(inputB, {1, vecLen}, {i, 0});
                tileB = Mul(tile, Element(DataType::DT_FP32, 1.0));
            }

            LOOP("Inner2", FunctionType::DYNAMIC_LOOP, k, LoopRange(loopNum))
            {
                auto tileA = View(inputA, {1, vecLen}, {k, 0});
                tileB = Add(tileA, tileB);
            }

            LOOP("Inner3", FunctionType::DYNAMIC_LOOP, l, LoopRange(1))
            {
                (void)l;
                tileB = Mul(tileB, Element(DataType::DT_FP32, 1.0));
                Assemble(tileB, {i, 0}, output);
            }
        }
    }
}

// Test GetTileShape
TEST_F(LoopUnrollTest, test_only_reshape2)
{
    int b = 1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b * sq;
    std::vector<int64_t> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    FUNCTION("main", {q}, {out})
    {
        Tensor qReshape(DT_FP32, {bSq, d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0, 1, 1), {}, true)
        {
            (void)batchId;
            qReshape = Reshape(q, {bSq, d}, true);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            Tensor q0 = View(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Exp(q0);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }
}

TEST_F(LoopUnrollTest, IsNoOverlapWAWAssembleAttrNull)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {8, 8}, "a");
    G.AddTensor(DataType::DT_FP32, {8, 8}, "b");
    G.AddTensor(DataType::DT_FP32, {8, 8}, "c");
    G.AddOp(Opcode::OP_ASSEMBLE, {"a"}, {"b"}, "assemble1");
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"c"}, "assemble2");
    G.SetInCast({"a"});
    G.SetOutCast({"c"});
    LoopUnroll pass;
    pass.lastWriteMap_[0] = std::make_pair(G.GetTensor("b"), true);
    std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap;
    EXPECT_FALSE(pass.IsNoOverlapWAW(0, G.GetTensor("c"), opDynOffsetMap));
}

// Test IF and nested FUNCTION
TEST_F(LoopUnrollTest, TestLoopIfWithRank)
{
    int s = 32;
    int n = 10;

    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {s, s}, "r0");
    Tensor out(DT_FP32, {s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 0.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int len = 8;
    int three = 3;
    int six = 6;

    FUNCTION("main", {t0, r0}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(len))
        {
            IF(i < three)
            {
                IF(IsLoopEnd(i, len)) { r0 = Add(r0, Element(DataType::DT_FP32, 1.0)); }
                ELSE { r0 = Add(r0, Element(DataType::DT_FP32, 1.0)); }
            }
            ELSE
            {
                IF(i < six) { r0 = Add(r0, Element(DataType::DT_FP32, 0.0)); }
                ELSE
                {
                    Tensor t0v = View(t0, {s, s}, {s * i, 0});
                    r0 = Add(t0v, r0);
                }
            }
        }
        config::SetBuildStatic(true);
        FUNCTION("S1") { out = Add(r0, Element(DataType::DT_FP32, 1.0)); }
    }
}
} // namespace tile_fwk
} // namespace npu
