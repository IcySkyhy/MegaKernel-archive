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
 * \file test_dynamic_pa.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicBasicTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override
    {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }
};

namespace {
// Constants to replace magic numbers
constexpr int LOOP_COUNT = 8;
constexpr int CONDITION_THRESHOLD = 6;
} // namespace

TEST_F(DynamicBasicTest, TestHybridLoopIf2)
{
    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 11.0),
        RawTensorData::CreateConstantTensor<float>(t1, 20.0),
        RawTensorData::CreateConstantTensor<float>(t2, 30.0),
        RawTensorData::CreateConstantTensor<float>(t3, 40.0),
        RawTensorData::CreateConstantTensor<float>(t4, 50.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    // clc
    FUNCTION("main", {t0, t1, t2, t3, t4}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT))
        {
            auto r0 = Add(t0, t1);
            r0 = Mul(r0, t1); // +t0, +t1
            IF(i < CONDITION_THRESHOLD)
            {
                r0 = Sub(r0, t2); // +t2 * 6
            }
            ELSE
            {
                r0 = Sub(r0, t3); // +t3 * 8
            }
            out = Add(r0, t4);
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    // EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.004f));
}

TEST_F(DynamicBasicTest, TestHybridLoopIfWithTernary)
{
    constexpr int LOOP_COUNT_INNER = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {LOOP_COUNT_INNER * s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 0.0),
        RawTensorData::CreateConstantTensor<float>(t1, 20.0),
        RawTensorData::CreateConstantTensor<float>(t2, 30.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    // clc
    FUNCTION("main", {t0, t1, t2}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT_INNER))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);

            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    std::vector<float> golden1(s * s, 20.0f);
    std::vector<float> golden2(s * s, 20.0f);
    std::vector<float> golden3(s * s, 30.0f);
    std::vector<float> golden4(s * s, 30.0f);
    golden1.insert(golden1.end(), golden2.begin(), golden2.end());
    golden1.insert(golden1.end(), golden3.begin(), golden3.end());
    golden1.insert(golden1.end(), golden4.begin(), golden4.end());
    EXPECT_TRUE(resultCmp(golden1, (float*)outs->data(), 0.004f));
}

void TestLoopViewAssemble(const Tensor& t0, const Tensor& t1, const Tensor& blockTable, Tensor& out, int s)
{
    FUNCTION("main", {t0, t1, blockTable}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s))
        {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2 * s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2 * s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul(DataType::DT_FP32, qi, ki, false, true);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}

#if ENABLE_HIDDENLOOP
TEST_F(DynamicBasicTest, HiddenLoopConditionMixed)
{
    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 11.0),
        RawTensorData::CreateConstantTensor<float>(t1, 20.0),
        RawTensorData::CreateConstantTensor<float>(t2, 30.0),
        RawTensorData::CreateConstantTensor<float>(t3, 40.0),
        RawTensorData::CreateConstantTensor<float>(t4, 50.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    FUNCTION("main", {t0, t1, t2, t3, t4}, {out})
    {
        // LOOP("L0", FunctionType::DYNAMIC_LOOP, _, LoopRange(1)) {
        //     (void)_;
        //     LOOP("L01",FunctionType::DYNAMIC_LOOP,idx1,LoopRange(1)){
        //         (void)idx1;
        out = Add(t0, t1);
        // }
        IF(SymbolicScalar(0) < CONDITION_THRESHOLD)
        {
            // LOOP("L02",FunctionType::DYNAMIC_LOOP,idx3,LoopRange(1)){
            //     (void)idx3;
            t3 = Add(t1, t2);
            // }
            LOOP("L03", FunctionType::DYNAMIC_LOOP, idx4, LoopRange(LOOP_COUNT))
            {
                (void)idx4;
                t4 = Sub(t4, t3);
            }
        }
        ELSE
        {
            // LOOP("L04",FunctionType::DYNAMIC_LOOP,idx5,LoopRange(1)){
            //     (void)idx5;
            out = Sub(out, t4);
            // }
        }
        // LOOP("L05",FunctionType::DYNAMIC_LOOP,idx6,LoopRange(1)){
        //     (void)idx6;
        out = Add(t3, t4);
        // }
        // }
    }
    std::vector<float> golden(n * s * m * s, -300.0f);
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

TEST_F(DynamicBasicTest, HiddenLoopConditionMixedMulLoops)
{
    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 11.0),
        RawTensorData::CreateConstantTensor<float>(t1, 20.0),
        RawTensorData::CreateConstantTensor<float>(t2, 30.0),
        RawTensorData::CreateConstantTensor<float>(t3, 40.0),
        RawTensorData::CreateConstantTensor<float>(t4, 50.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });
    Tensor t0_temp;
    FUNCTION("Main", {t0, t1, t2, t3, t4}, {out})
    {
        // LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
        //    (void)i;
        // LOOP("L01", FunctionType::DYNAMIC_LOOP, j, LoopRange(1)) {
        //     (void)j;
        IF(SymbolicScalar(0) < CONDITION_THRESHOLD) { t0_temp = Add(t1, t1); }
        ELSE { t0_temp = Add(t2, t2); }
        t0_temp = Add(t0_temp, Element(DT_FP32, 1.0f));
        // }
        LOOP("L02", FunctionType::DYNAMIC_LOOP, k, LoopRange(2))
        {
            (void)k;
            t3 = Mul(t3, t2);
        }
        // LOOP("L03", FunctionType::DYNAMIC_LOOP, l, LoopRange(1)) {
        //      (void)l;
        out = Sub(t3, t0_temp);
        // }
        LOOP("L04", FunctionType::DYNAMIC_LOOP, h, LoopRange(2))
        {
            (void)h;
            t0_temp = Mul(t0_temp, t2);
        }
        // LOOP("L05", FunctionType::DYNAMIC_LOOP, q, LoopRange(1)) {
        //     (void)q;
        out = Add(out, t0_temp);
        //}
        //}
    }
    std::vector<float> golden(n * s * m * s, 72859.0f); // 显示计算结果
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}
#endif

TEST_F(DynamicBasicTest, TestDD)
{
    SetInterpreterConfig();
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");     // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    std::vector<float> golden(n * s * s, 128.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0), RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData), // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    TestLoopViewAssemble(t0, t1, blockTable, out, s);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestHUB)
{
    int dim0 = 128;
    int tileSizeSmall = 32;
    int tileSizeLarge = 64;

    Tensor t0(DT_FP32, {dim0}, "t0");
    Tensor t1(DT_FP32, {dim0}, "t1");
    Tensor t2(DT_FP32, {dim0}, "t2");
    FUNCTION("main", {t0}, {t2})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx0, LoopRange(1))
        {
            (void)idx0;
            TileShape::Current().SetVecTile(tileSizeSmall);
            auto tmp = Abs(t0);
            TileShape::Current().SetVecTile(tileSizeLarge);
            t1 = Hub(tmp);
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(1))
        {
            (void)idx1;
            TileShape::Current().SetVecTile(tileSizeLarge);
            t2 = Add(t1, t1);
        }
    }

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateConstantTensor<float>(t0, -1.0f)});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(t2, 0.0f),
    });

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(dim0, 2.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestTT)
{
    int s = 64;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [64 * 8, 64]
    Tensor t1(DT_FP32, {n * s, s}, "t1"); // [64 * 8, 64]
    Tensor out(DT_FP32, {n * s, s}, "out");

    std::vector<std::string> funcName = {"TENSOR_main"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    FUNCTION("main", {t0, t1}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(8))
        {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            Tensor t1s = View(t1, {s, s}, {idx * s, 0});
            Tensor o = Add(t0s, t1s);
            Assemble(o, {idx * s, 0}, out);
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * s * s, 3.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestLocalTensor)
{
    int s = 64;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [64 * 8, 64]
    Tensor t1(DT_FP32, {n * s, s}, "t1"); // [64 * 8, 64]
    Tensor out(DT_FP32, {n * s, s}, "out");

    std::vector<std::string> funcName = {"TENSOR_main"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    Tensor t0s;
    FUNCTION("main", {t0, t1}, {out})
    {
        LOOP("loopOut", FunctionType::DYNAMIC_LOOP, loopOut, LoopRange(1))
        {
            Tensor o;
            LOOP("loopMiddle", FunctionType::DYNAMIC_LOOP, loopMiddle, LoopRange(1))
            {
                (void)loopOut;
                (void)loopMiddle;
                LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(n))
                {
                    t0s = View(t0, {s, s}, {idx * s, 0});
                    Tensor t1s = View(t1, {s, s}, {idx * s, 0});
                    o = Add(t0s, t1s);
                    Assemble(o, {idx * s, 0}, out);
                }
            }
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * s * s, 3.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestLocalTempTensor)
{
    int s = 64;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [64 * 8, 64]
    Tensor t1(DT_FP32, {n * s, s}, "t1"); // [64 * 8, 64]
    Tensor out(DT_FP32, {n * s, s}, "out");

    std::vector<std::string> funcName = {"TENSOR_main"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    Tensor t0s;
    FUNCTION("main", {t0, t1}, {out})
    {
        LOOP("loopOut", FunctionType::DYNAMIC_LOOP, loopOut, LoopRange(1))
        {
            LOOP("loopMiddle", FunctionType::DYNAMIC_LOOP, loopMiddle, LoopRange(1))
            {
                (void)loopOut;
                (void)loopMiddle;
                LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(n))
                {
                    Tensor o(DT_FP32, {s, s}, "tempO"); // temp tensor
                    LOOP("LoopLeaf1", FunctionType::DYNAMIC_LOOP, leaf1, LoopRange(1))
                    {
                        (void)leaf1;
                        t0s = View(t0, {s, s}, {idx * s, 0});
                        Tensor t1s = View(t1, {s, s}, {idx * s, 0});
                        o = Add(t0s, t1s);
                    }
                    LOOP("LoopLeaf2", FunctionType::DYNAMIC_LOOP, leaf2, LoopRange(1))
                    {
                        (void)leaf2;
                        o = Add(o, Element(DT_FP32, 0.0f));
                        Assemble(o, {idx * s, 0}, out);
                    }
                }
            }
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * s * s, 3.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestCheckPointRestore)
{
    int s = 16;
    Tensor t;
    Tensor t0(DT_FP32, {s, s}, "t0");

    FUNCTION("main", {t}, {t0})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1))
        {
            IF(idx == 0) { t0 = Full(Element(DT_FP32, 1.0f), DT_FP32, {s, s}); }
            ELSE { t0 = Full(Element(DT_FP32, 2.0f), DT_FP32, {s, s}); }
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1))
        {
            (void)idx;
            t0 = Full(Element(DT_FP32, 1.0f), DT_FP32, {s, s});
        }
    }
    EXPECT_EQ(t0.GetStorage()->tensor->GetRefCount(), 1);
}

TEST_F(DynamicBasicTest, TestSlotId)
{
    int s = 16;
    int id[2] = {0};
    Tensor t(DT_FP32, {s, s}, "t0");
    Tensor out(DT_FP32, {s, s}, "out");

    FUNCTION("main", {t}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1))
        {
            (void)idx;
            Tensor t0(DT_FP32, {s, s}, "t1");
            LOOP("L00", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(1))
            {
                (void)idx1;
                t0 = Add(t, t);
            }
            id[0] = t0.Id();
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1))
        {
            (void)idx;
            Tensor t1(DT_FP32, {s, s}, "t1");
            LOOP("L10", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(1))
            {
                (void)idx1;
                t1 = Add(t, t);
            }
            id[1] = t1.Id();
        }
    }
    EXPECT_NE(id[0], id[1]);
}

TEST_F(DynamicBasicTest, DynamicRawShape)
{
    SetInterpreterConfig();
    int s = 32;
    Tensor t0(DT_FP32, {-1, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor out(DT_FP32, {-1, s}, "out");

    int n = 8;
    Tensor arg0(DT_FP32, {n * s, s});
    Tensor out0(DT_FP32, {n * s, s});

    std::vector<float> golden(n * s * s, 64.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(arg0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out0, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out0, golden),
    });

    FUNCTION("main", {t0, t1}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(GetInputShape(out, 0) / s))
        {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            Tensor t2 = Matrix::Matmul(DataType::DT_FP32, t0s, t1, false, true);
            Assemble(t2, {idx * s, 0}, out);
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, DynamicRawShapeUnalign)
{
    int s = 32;
    Tensor t0(DT_FP32, {-1, s}, "t0"); // [32*8, 32]
    Tensor out(DT_FP32, {-1, s}, "out");

    FUNCTION("main", {t0}, {out})
    {
        auto shape0 = GetInputShape(t0, 0);
        auto t1 = Tensor(t0.GetDataType(), {shape0, s});
        auto loop1 = (shape0 + s - 1) / s;
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop1))
        {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            auto t = Add(t0s, Element(DT_FP32, 3.0));
            Assemble(t, {idx * s, 0}, t1);
        }

        // check t1 use dynshape from t0
        auto loop2 = (GetInputShape(t1, 0) + s - 1) / s;
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop2), {}, true)
        {
            Tensor t1s = View(t1, {s, s}, {idx * s, 0});
            auto t = Sub(t1s, Element(DT_FP32, 1.0));
            Assemble(t, {idx * s, 0}, out);
        }
    }

    int s0 = 200;
    Tensor arg0(DT_FP32, {s0, s});
    Tensor out0(DT_FP32, {s0, s});
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(arg0, 3.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out0, 0.0f),
    });

    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DeviceTensorData argData0{arg0.GetDataType(), nullptr, arg0.GetShape()};
    DeviceTensorData outData0{arg0.GetDataType(), nullptr, arg0.GetShape()};
    std::vector<DeviceTensorData> evalInputs{argData0};
    std::vector<DeviceTensorData> evalOutputs{outData0};
    Evaluator eval{dynAttr->inputSymbolDict, &evalInputs, &evalOutputs};
    EXPECT_EQ(eval.Evaluate(dynAttr->maxDynamicAssembleOutcastMem), s0 * s * BytesOf(arg0.GetDataType()));
}

TEST_F(DynamicBasicTest, TestInplace)
{
    Tensor t0(DT_FP32, {32, 32}, "t0");
    Tensor t1(DT_FP32, {32, 32}, "t1");
    Tensor t2(DT_FP32, {32, 32}, "t2");
    Tensor t3(DT_FP32, {32, 32}, "t3");

    FUNCTION("main", {t0, t1}, {t3}, {{t2, t0}})
    {
        LOOP("l0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            UNUSED(i);
            t3 = Add(t0, t1);
            Assemble(t3, {0, 0}, t2);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(t3, 0.0f),
    });

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(32 * 32, 3.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetInputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestStaticUnderDynDev)
{
    SetInterpreterConfig();
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {n * s, s}, "t1"); // [32, 32]
    Tensor out(DT_FP32, {n * s, s}, "out");
    std::vector<float> golden(n * s, 1.0f);
    FUNCTION("main", {t0, t1}, {out})
    {
        config::SetBuildStatic(true);
        FUNCTION("S0") { out = Sub(t1, t0); }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestStaticLoop)
{
    SetInterpreterConfig();
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {n * s, s}, "t1"); // [32, 32]
    Tensor t2(DT_FP32, {s, s}, "t2");     // [32, 32]
    Tensor out(DT_FP32, {n * s, s}, "out");
    std::vector<float> outGolden(n * s, 4.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0), RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateConstantTensor<float>(t2, 3.0), // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, outGolden),
    });

    FUNCTION("main", {t0, t1, t2}, {out})
    {
        Tensor s0Out;
        config::SetBuildStatic(true);
        FUNCTION("S0") { s0Out = Sub(t1, t0); }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT))
        {
            Tensor t0s = View(s0Out, {s, s}, {i * s, 0});
            Tensor t3 = Add(t0s, t2);
            Assemble(t3, {i * s, 0}, out);
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outGolden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestInnerLoopOrder)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(512, 512);
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    std::vector<std::string> funcName = {"TENSOR_main"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
    int vecLen = 16;
    int loopNum = 4;
    int tileNum = 3;
    Tensor inputA(DT_FP32, {loopNum, vecLen}, "inputA");
    Tensor inputB(DT_FP32, {tileNum, vecLen}, "inputB");
    Tensor output(DT_FP32, {tileNum, vecLen}, "out");

    std::vector<float> inputAData(loopNum * vecLen, 0);
    std::vector<float> inputBData(tileNum * vecLen, 0);
    std::vector<float> golden(tileNum * vecLen, 0);

    readInput<float>(GetGoldenDir() + "/input_a.bin", inputAData);
    readInput<float>(GetGoldenDir() + "/input_b.bin", inputBData);
    readInput(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputB, inputBData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, golden),
    });

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("Outer", FunctionType::DYNAMIC_LOOP, i, LoopRange(tileNum))
        {
            Tensor tileB(DT_FP32, {1, vecLen}, "tileB");
            LOOP("Inner", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
            {
                (void)j;
                auto tile = View(inputB, {1, vecLen}, {i, 0});
                tileB = Mul(tile, Element(DataType::DT_FP32, 2.0));
            }

            LOOP("Inner2", FunctionType::DYNAMIC_LOOP, k, LoopRange(loopNum))
            {
                auto tileA = View(inputA, {1, vecLen}, {k, 0});
                tileB = Add(tileA, tileB);
            }

            LOOP("Inner3", FunctionType::DYNAMIC_LOOP, l, LoopRange(1))
            {
                (void)l;
                tileB = Mul(tileB, Element(DataType::DT_FP32, 3.0));
                Assemble(tileB, {i, 0}, output);
            }
        }
    }

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.005f));
}

TEST_F(DynamicBasicTest, TestDeviceMachineOnModel)
{
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");     // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopViewAssemble(t0, t1, blockTable, out, s);

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0), RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData), // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 25, 5});

    std::cout << "test -> blockdim = 16, aicpunum = 4" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 16, 4});

    std::cout << "test -> blockdim = 9, aicpunum = 4" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 9, 4});

    std::cout << "test -> blockdim = 8, aicpunum = 3" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 8, 3});

    std::cout << "test -> blockdim = 1, aicpunum = 3" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 1, 3});
}

TEST_F(DynamicBasicTest, TestDeviceMachineBlockdimOnBoard)
{
    SetInterpreterConfig();
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");     // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    std::vector<float> golden(n * s * s, 128.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0), RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData), // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    TestLoopViewAssemble(t0, t1, blockTable, out, s);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {true, 15, 4});
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestDeviceMachineBlockdimOnBoard1)
{
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");     // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopViewAssemble(t0, t1, blockTable, out, s);

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0), RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData), // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {true, 7, 3});
    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    std::vector<float> golden(n * s * s, 128.0f);
    EXPECT_TRUE(resultCmp(golden, (float*)outs1->data(), 0.001f));
#endif
}

namespace DynamicTest {

TEST_F(DynamicBasicTest, TestLoopIfWithRank456)
{
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(32, 32); // 设置Tileshape大小为32*32
    std::vector<std::string> funcName = {"TENSOR_main"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);

    int s = 32;
    int n = 10;
    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {s, s}, "r0");
    Tensor out(DT_FP32, {s, s}, "out");
    std::vector<float> golden(s * s, 12.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 0.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    // Direct implementation of the function logic within the test
    FUNCTION("main", {t0, r0}, {out})
    {
        constexpr int LOOP_LENGTH = 10;
        npu::tile_fwk::SymbolicScalar len(LOOP_LENGTH);
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(len))
        {
            IF(i == 0)
            {
                IF(i == len - 1) { r0 = Add(r0, Element(DataType::DT_FP32, 1.0)); }
                ELSE { r0 = Add(r0, Element(DataType::DT_FP32, 2.0)); }
            }
            ELSE
            {
                IF(i == len - 1) { r0 = Add(r0, Element(DataType::DT_FP32, 0.0)); }
                ELSE
                {
                    Tensor t0v = View(t0, {s, s}, {s * i, 0});
                    r0 = Add(t0v, r0);
                }
            }
            out = Add(r0, Element(DataType::DT_FP32, 0.0));
        }

        config::SetBuildStatic(true);
        FUNCTION("S1")
        {
            out = Add(r0, Element(DataType::DT_FP32, 2.0)); // 静态function中增加2.0的偏移量
        }
    }
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestTensorExtract)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = 32;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }
    Tensor output(DT_INT32, {1, s}, "output");

    int row = 3;
    int col = 4;
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", {inputA}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            Tensor t0 = Add(inputA, Element(DT_INT32, (int64_t)2));
            output = TensorExtract(t0, {row, col});
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_EQ(row * n + col + 0x2, *(int32_t*)outputResult->data());
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorData)
{
    SetInterpreterConfig();
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 12.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, outputGolden),
    });

    FUNCTION("main", {inputA, inputC}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            Tensor t0 = Add(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            SymbolicScalar v0 = GetTensorData(t0, {0, 1});          // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            SymbolicScalar v1 = GetTensorData(t0, {0, 2});          // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            auto t2 = View(inputC, {n, n}, {0, v0 * n});
            auto t3 = View(inputC, {n, n}, {0, v1 * n});
            output = Mul(t2, t3);
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataCrossFunction)
{
    SetInterpreterConfig();
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }
    Tensor output(DT_FP32, {n, n}, "output");
    Tensor outsum(DT_INT32, {n, n}, "outsum");

    std::vector<float> outputGolden(n * n, 12.0f);
    std::vector<int> outsumGolden(n * n, 0);
    int d0 = 1 + 2;
    int d1 = 2 + 2;
    int d2 = d0 + d1 + 1;
    outsumGolden[0] = d0;
    outsumGolden[1] = d1;
    outsumGolden[2] = d2;
    outsumGolden[3] = d0 + d1;
    outsumGolden[4] = d0 + d2;
    outsumGolden[5] = d1 + d2;
    outsumGolden[6] = d0 + d1 + d2;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
        RawTensorData::CreateConstantTensor<int32_t>(outsum, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, outputGolden),
        RawTensorData::CreateTensor<int32_t>(outsum, outsumGolden),
    });

    FUNCTION("main", {inputA, inputC}, {output, outsum})
    {
        SymbolicScalar v0;
        SymbolicScalar v1;
        SymbolicScalar v2;
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto t0 = Add(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            v0 = GetTensorData(t0, {0, 1});                       // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            v1 = GetTensorData(t0, {0, 2});                       // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            v2 = v0 + v1 + GetTensorData(inputA, {0, 1});
        }
        LOOP("Step1", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto t2 = View(inputC, {n, n}, {0, v0 * n});
            auto t3 = View(inputC, {n, n}, {0, v1 * n});
            output = Mul(t2, t3);
            SetTensorData(v0, {0, 0}, outsum);
            SetTensorData(v1, {0, 1}, outsum);
            SetTensorData(v2, {0, 2}, outsum);
            SetTensorData(v0 + v1, {0, 3}, outsum);
            SetTensorData(v0 + v2, {0, 4}, outsum);
            SetTensorData(v1 + v2, {0, 5}, outsum);
            SetTensorData(v0 + v1 + v2, {0, 6}, outsum);
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float*)outputResult->data(), 0.001f));
    auto outsumResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(resultCmp(outsumGolden, (int32_t*)outsumResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataUnalign)
{
    SetInterpreterConfig();
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int cnt = 8;
    int n = tiling * 1;
    int m = tiling * cnt;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC1(DT_FP32, {n, m}, "inputC1");
    Tensor inputC2(DT_FP32, {n, m}, "inputC2");
    std::vector<float> inputC1Data(n * m, 0); // 8 x (32 x 32 / 16 x 16)
    std::vector<float> inputC2Data(n * m, 0); // 8 x (32 x 32 / 16 x 16)
    int v0Data = 3;
    int v1Data = 4;
    for (int k = 0; k < cnt; k++) {
        for (int i = 0; i < v0Data; i++) {
            for (int j = 0; j < v1Data; j++) {
                inputC1Data[i * m + k * n + j] = k;
            }
        }
        for (int i = 0; i < v0Data; i++) {
            for (int j = 0; j < v1Data; j++) {
                inputC2Data[i * m + k * n + j] = k + 1;
            }
        }
    }
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 0);
    for (int i = 0; i < v0Data; i++) {
        for (int j = 0; j < v1Data; j++) {
            outputGolden[i * n + j] = 15;
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC1, inputC1Data),
        RawTensorData::CreateTensor<float>(inputC2, inputC2Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, outputGolden),
    });

    FUNCTION("main", {inputA, inputC1, inputC2}, {output})
    {
        SymbolicScalar v0;
        SymbolicScalar v1;
        SymbolicScalar v2;
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto t0 = Add(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            v0 = GetTensorData(t0, {0, 1});                       // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            v1 = GetTensorData(t0, {0, 2});                       // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            v2 = v0 + v1 + GetTensorData(inputA, {0, 1});
        }
        LOOP("Step1", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto t2 = View(inputC1, {n, n}, {v0, v1}, {0, v0Data * n}); // 16 x 16 / 3 x 4, 3
            auto t3 = View(inputC2, {n, n}, {v0, v1}, {0, v1Data * n}); // 16 x 16 / 3 x 4, 5
            output = Mul(t2, t3);                                       // 16 x 16 / 3 x 5, 12
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataExpr)
{
    SetInterpreterConfig();
    int tiling = 32;
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }

    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 35.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, outputGolden),
    });

    FUNCTION("main", {inputA, inputC}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            Tensor t0 = Add(inputA, Element(DT_INT32, (int64_t)2));     // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            SymbolicScalar v0 = GetTensorData(t0, {0, 1});              // t0[0, 1] + 2 -> 0 * n + 1 + 2 -> 3
            SymbolicScalar v1 = GetTensorData(t0, {0, 2});              // t0[0, 2] + 2 -> 0 * n + 2 + 2 -> 4
            SymbolicScalar v2 = GetTensorData(inputA, {0, 1});          // inputA[0, 1] -> 1
            SymbolicScalar v3 = GetTensorData(inputA, {0, 2});          // inputA[0, 2] -> 2
            auto t2 = View(inputC, {n, n}, {0, (v0 + v2 + i / i) * n}); // {0, (3 + 1 + 1) * n} -> {0, 5 * n}
            auto t3 = View(inputC, {n, n}, {0, (v1 + v3 + i / i) * n}); // {0, (4 + 2 + 1) * n} -> {0, 7 * n}
            output = Mul(t2, t3);
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestVectorDup)
{
    SetInterpreterConfig();
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<int32_t> outputGolden(n * n, 50);

    ProgramData::GetInstance().AppendInputs({});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            SymbolicScalar v = 20;
            output = Full(v + 30, DT_INT32, {n, n});
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestTensorInsert)
{
    SetInterpreterConfig();
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_INT32, {n}, "output");
    std::vector<int32_t> outputGolden(n, 20);

    ProgramData::GetInstance().AppendInputs({});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n))
        {
            auto tmp = Full(20, DT_INT32, {1});
            TensorInsert(tmp, {i}, output);
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSetTensorData)
{
    SetInterpreterConfig();
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_INT32, {n}, "output");
    std::vector<int32_t> outputGolden(n, 30);

    ProgramData::GetInstance().AppendInputs({});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) { SetTensorData(30, {i / 2 * 2 + i % 2}, output); }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSetTensorDataExpr)
{
    SetInterpreterConfig();

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor output(DT_INT32, {n, n, n}, "output");
    std::vector<int32_t> outputGolden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        outputGolden[i] = i;
    }

    ProgramData::GetInstance().AppendInputs({});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n))
        {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n))
            {
                for (int k = 0; k < n; k++) {
                    SetTensorData(i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataAndDup)
{
    SetInterpreterConfig();
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor input(DT_INT32, {n, n}, "input");
    std::vector<int32_t> inputData(n * n);
    for (int i = 0; i < n * n; i++) {
        inputData[i] = i;
    }

    int row = 3;
    int col = 4;
    Tensor output(DT_INT32, {n, n}, "output");
    std::vector<int32_t> outputGolden(n * n, (row * n + col) * 2 + 1);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(input, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {input}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto add = Add(input, input);
            auto s = GetTensorData(add, {row, col});
            output = Full(s + 1, DT_INT32, {n, n});
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetAndSetTensorDataExpr)
{
    SetInterpreterConfig();

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    int init = 10;
    Tensor input(DT_INT32, {n, n, n}, "input");
    Tensor output(DT_INT32, {n, n, n}, "output");
    std::vector<int32_t> outputGolden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        outputGolden[i] = init + init + i;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input, init),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FUNCTION("main", {input}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n))
        {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n))
            {
                auto add = Add(input, input);
                for (int k = 0; k < n; k++) {
                    SymbolicScalar s = GetTensorData(add, {i, j, k});
                    SetTensorData(s + i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSelectAttention)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor input(DT_INT32, {n, n}, "input");
    std::vector<int32_t> inputData(n * n);
    for (int i = 0; i < n * n; i++) {
        inputData[i] = i % n; // inputData[x,y] -> y
    }
    Tensor table(DT_INT32, {n, n}, "table");
    std::vector<int32_t> tableData(n * n);
    for (int i = 0; i < n * n; i++) {
        tableData[i] = i % n; // tableData[x,y] -> y
    }
    Tensor c0(DT_FP32, {n, n * n}, "c0");
    std::vector<float> c0Data(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        c0Data[i] = i % (n * n) / n; // c0Data[x, a * n + b] -> a
    }
    Tensor c1(DT_FP32, {n, n * n}, "c1");
    std::vector<float> c1Data(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        c1Data[i] = i % (n * n) / n; // c1Data[x, a * n + b] -> a
    }

    Tensor output(DT_FP32, {n, n}, "output");

    DataType dtype = DT_FP32;
    float outputGoldenCell = 0;
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            outputGoldenCell += i * i;
        }
    }
    std::vector<float> outputGolden(n * n);
    for (int i = 0; i < n * n; i++) {
        outputGolden[i] = outputGoldenCell;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(input, inputData),
        RawTensorData::CreateTensor<int32_t>(table, tableData),
        RawTensorData::CreateTensor<float>(c0, c0Data),
        RawTensorData::CreateTensor<float>(c1, c1Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, outputGolden),
    });

    int topk = 16;
    FUNCTION("main", {input, table, c0, c1}, {output})
    {
        Tensor index;
        LOOP("Idx", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            index = Add(input, input);
            index = Sub(index, input);
        }
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n), {}, true)
        {
            (void)i;
            Tensor r0(dtype, {n, n * n}, "r0");
            Tensor r1(dtype, {n, n * n}, "r1");
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(0, n, topk), {}, true)
            {
                (void)j;
                for (int k = 0; k < topk; k++) {
                    SymbolicScalar s = GetTensorData(index, {i, j + k});       // index[i, j + k] -> j + k
                    SymbolicScalar slcBlockIdx = GetTensorData(table, {i, s}); // table[i, s] -> s
                    auto k0 = View(c0, {n, n}, {0, s * n});
                    auto k1 = View(c1, {n, n}, {0, slcBlockIdx * n});

                    auto k0v = Add(k0, Element(dtype, (float)0));
                    auto k1v = Add(k1, Element(dtype, (float)0));
                    Assemble(k0v, {0, s * n}, r0);
                    Assemble(k1v, {0, slcBlockIdx * n}, r1);
                }
            }
            LOOP("Step2", FunctionType::DYNAMIC_LOOP, j, LoopRange(n), {}, true)
            {
                LOOP("loop1", FunctionType::DYNAMIC_LOOP, _, LoopRange(1), {}, true)
                {
                    (void)_;
                    auto matmul = Matrix::Matmul(DataType::DT_FP32, r0, r1, false, true);
                    auto d1 = Div(matmul, Element(dtype, (float)n));
                    auto d2 = Div(d1, Element(dtype, (float)n));
                    IF(i == 0)
                    {
                        IF(j == 0) { output = d2; }
                        ELSE { output = Add(output, d2); }
                    }
                    ELSE { output = Add(output, d2); }
                }
            }
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataSymbolicValue)
{
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    int n = 4;
    int loopCount = 4;
    int NUM_2 = 2;
    Tensor loopList(DT_INT32, {1, loopCount}, "loopList");
    std::vector<int32_t> loopListData(loopCount);
    for (int k = 0; k < loopCount; k++) {
        loopListData[k] = k + 1;
    }
    Tensor output(DT_INT32, {1, n}, "output");
    std::vector<int32_t> outputGolden(n, 0);
    for (int i = 0; i < n; i++) {
        outputGolden[i] = (i + 1) * NUM_2;
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(loopList, loopListData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });
    FUNCTION("main", {loopList}, {output})
    {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(loopCount))
        {
            Tensor doubleLoopList(DT_INT32, {1, loopCount}, "doubleLoopList");
            doubleLoopList = Add(loopList, loopList);
            SymbolicScalar idxs = GetTensorData(doubleLoopList, {0, i});
            auto result2 = Full(idxs, DT_INT32, {1, 1});
            Assemble(result2, {0, i}, output);
        }
    }
#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t*)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, DuplicateName)
{
    Tensor t0(DT_FP32, {32, 32}, "t0");
    Tensor out(DT_FP32, {64, 64}, "out");

    auto t0Data = RawTensorData::CreateConstantTensor<float>(t0, 1.0f);
    auto outData = RawTensorData::CreateConstantTensor<float>(out, 0.0f);
    auto golden = RawTensorData::CreateConstantTensor<float>(out, 2.0f);

    ProgramData::GetInstance().PrepareData({t0Data}, {outData}, {golden});

    auto dupTile = [&](std::vector<int64_t> offset, bool isAdd) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(2))
        {
            (void)i;
            auto v = isAdd ? Add(t0, t0) : Sub(t0, t0);
            Assemble(v, SymbolicScalar::FromConcrete(offset), out);
        }
    };
    FUNCTION("main", {t0}, {out})
    {
        dupTile({0, 0}, true);
        dupTile({0, 32}, true);
        dupTile({32, 0}, false);
        dupTile({32, 32}, false);
    }
}
} // namespace DynamicTest
