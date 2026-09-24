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
 * \file test_dynamic_reshape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/function.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicReshapeTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override
    {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        RuntimeSetDevice(GetDeviceIdByEnvVar());
    }
};

TEST_F(DynamicReshapeTest, test_only_reshape)
{
    SetInterpreterConfig();

    int b = 1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b * sq;
    std::vector<int64_t> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    b = 1;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});
    std::vector<float> golden(b * sq * d, exp(1.0f) + 1.0f);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q_real, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out_real, 0.001f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out_real, golden),
    });

    FUNCTION("MAIN_FUNC", {q}, {out})
    {
        Tensor bfRes(DT_FP32, qShape, "bfRes");
        LOOP("L0_BF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            TileShape::Current().SetVecTile(1, 64, 64);
            Tensor q0 = View(q, {1, sq, d}, {batchId, 0, 0});
            auto tmp = Exp(q0);
            Assemble(tmp, {batchId, 0, 0}, bfRes);
        }

        Tensor qReshape(DT_FP32, {bSq, d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0, 1, 1), {}, true)
        {
            (void)batchId;
            qReshape = Reshape(bfRes, {bSq, d}, true);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            TileShape::Current().SetVecTile(64, 64);
            Tensor q0 = View(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Add((q0), Element(DataType::DT_FP32, 1.0f));
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeTest, test_only_reshape2)
{
    SetInterpreterConfig();
    std::vector<std::string> funcName = {"TENSOR_MAIN_FUNC"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);

    int b = 2;
    int sq = 32;
    int d = 16;
    int bSq = b * sq;
    std::vector<int64_t> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    std::vector<float> qData(b * sq * d, 0);
    std::vector<float> golden(b * sq * d, 0);

    readInput<float>(GetGoldenDir() + "/q.bin", qData);
    readInput(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, qData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("MAIN_FUNC", {q}, {out})
    {
        Tensor qReshape(DT_FP32, {bSq, d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0, b, 1), {}, true)
        {
            (void)batchId;
            TileShape::Current().SetVecTile(16, 16); // 设置Tileshape大小为16*16
            qReshape = Reshape(q, {bSq, d});
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            TileShape::Current().SetVecTile(16, 16); // 设置Tileshape大小为16*16
            Tensor q0 = View(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Exp(q0);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.005f));
}

TEST_F(DynamicReshapeTest, test_dyn_reshape)
{
    int b = -1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b * sq;
    std::vector<int64_t> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    FUNCTION("MAIN_FUNC", {q}, {out})
    {
        Tensor qReshape(DT_FP32, {GetInputShape(q, 0) * GetInputShape(q, 1), d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0, 1, 1), {}, true)
        {
            (void)batchId;
            qReshape = Reshape(q, {GetInputShape(q, 0) * GetInputShape(q, 1), d}, true);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            TileShape::Current().SetVecTile(64, 64);
            Tensor q0 = View(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Exp(q0);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    b = 1;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q_real, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out_real, 0.001f),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));

    std::vector<float> golden(b * sq * d, exp(1.0f));

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

static void SetupDynReshape2ProgramData(const Tensor& q_real, const Tensor& out_real, const std::vector<float>& golden)
{
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q_real, 1.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out_real, 0.001f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out_real, golden),
    });
}

static void BuildDynReshape2Function(Tensor& q, Tensor& out, int sq, int d)
{
    FUNCTION("MAIN_FUNC", {q}, {out})
    {
        Tensor bfRes(DT_FP32, {GetInputShape(q, 0), sq, d}, "bfRes");
        LOOP("L0_BF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            TileShape::Current().SetVecTile(1, 64, 64);
            Tensor q0 = View(q, {1, sq, d}, {batchId, 0, 0});
            auto tmp = Exp(q0);
            Assemble(tmp, {batchId, 0, 0}, bfRes);
        }

        Tensor qReshape(DT_FP32, {GetInputShape(q, 0) * GetInputShape(q, 1), d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0, 1, 1), {}, true)
        {
            (void)batchId;
            qReshape = Reshape(bfRes, {GetInputShape(q, 0) * GetInputShape(q, 1), d}, true);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            TileShape::Current().SetVecTile(64, 64);
            Tensor q0 = View(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Add((q0), Element(DataType::DT_FP32, 1.0f));
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }
}

static void ExpectDynReshape2AssembleMem(const Tensor& q_real, const Tensor& out_real)
{
    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DeviceTensorData argData0{q_real.GetDataType(), nullptr, q_real.GetShape()};
    DeviceTensorData outData0{out_real.GetDataType(), nullptr, out_real.GetShape()};
    std::vector<DeviceTensorData> evalInputs{argData0};
    std::vector<DeviceTensorData> evalOutputs{outData0};
    Evaluator eval{dynAttr->inputSymbolDict, &evalInputs, &evalOutputs};
    EXPECT_EQ(eval.Evaluate(dynAttr->maxDynamicAssembleOutcastMem), q_real.GetStorage()->GetDataSize());
}

TEST_F(DynamicReshapeTest, test_dyn_reshape2)
{
    SetInterpreterConfig();

    int b = -1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b * sq;
    std::vector<int64_t> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    b = 1;
    Tensor q_real(DT_FP32, {b, sq, d});
    Tensor out_real(DT_FP32, {b * sq, d});
    std::vector<float> golden(b * sq * d, exp(1.0f) + 1.0f);
    SetupDynReshape2ProgramData(q_real, out_real, golden);
    BuildDynReshape2Function(q, out, sq, d);
    ExpectDynReshape2AssembleMem(q_real, out_real);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));
    auto outs = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeTest, test_dyn_reshape1111)
{
    TileShape::Current().SetVecTile(32, 64);

    Tensor A(DT_FP32, {128, 64}, "A");
    Tensor B(DT_FP32, {128, 64}, "B");
    Tensor D(DT_FP32, {256, 64}, "D");

    FUNCTION("MAIN_FUNC", {A, B}, {D})
    {
        LOOP("LOOP_TEST", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(0, 2, 1))
        {
            Tensor C(DT_FP32, {128, 64}, "q");
            auto a0 = View(A, {64, 64}, {loopIdx * 64, 0});
            auto a1 = Add(a0, Element(DataType::DT_FP32, 1.0f));
            Assemble(a1, {0, 0}, C);

            auto b0 = View(B, {64, 64}, {loopIdx * 64, 0});
            auto b1 = Add(b0, Element(DataType::DT_FP32, 1.0f));
            Assemble(b1, {64, 0}, C);

            auto d = Add(C, Element(DataType::DT_FP32, 1.0f));
            Assemble(d, {loopIdx * 128, 0}, D);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(A, 1.0),
        RawTensorData::CreateConstantTensor<float>(B, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(D, 0.001f),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(256 * 64, 3.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

TEST_F(DynamicReshapeTest, test_dyn_reshape22222)
{
    TileShape::Current().SetVecTile(32, 64);

    Tensor A(DT_FP32, {128, 64}, "A");
    Tensor B(DT_FP32, {128, 64}, "B");

    FUNCTION("MAIN_FUNC", {A}, {B})
    {
        LOOP("LOOP_TEST", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(0, 1, 1))
        {
            (void)loopIdx;
            Assemble(A, {0, 0}, B);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(A, 1.0),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(B, 0.001f),

    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(128 * 64, 1.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

/*
 * test infershape case
 */

// test reshape unaligned infershape
TEST_F(DynamicReshapeTest, test_reshape_unalign)
{
    TileShape::Current().SetVecTile(64, 64);

    int b = 2;
    int sq = 64;
    int d = 64;
    std::vector<int64_t> qShape2Dim = {b * sq, d};
    std::vector<int64_t> qShape3Dim = {b, sq, d};

    Tensor q(DT_FP32, qShape2Dim, "q");
    Tensor actSeqs(DT_INT32, {b, 1, 1}, "actual_seq");
    Tensor out(DT_FP32, qShape3Dim, "out");

    FUNCTION("main", {q, actSeqs}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (sq)))
        {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0, 0});

            Tensor q0 = View(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp0 = Reshape(q0, {1, sq, d}, {1, curSeq, d});
            TileShape::Current().SetVecTile(1, 64, 64);
            auto tmp = Exp(tmp0);
            Assemble(tmp, {batchId, 0, 0}, out);
        }
    }

    float inputValue = 2.0f;
    float initValue = 0.5f;

    std::vector<int> actSeqsData(b, 63);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, inputValue),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(b * sq * d, initValue);
    for (int bIdx = 0; bIdx < b; ++bIdx) {
        int offset = bIdx * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[bIdx] * d, exp(inputValue));
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

// test vec + mm diff tile and unaligned  infershape
TEST_F(DynamicReshapeTest, test_assemble_diff_tile)
{
    SetInterpreterConfig();
    TileShape::Current().SetCubeTile({16, 16}, {128, 128}, {128, 128});

    int batch = 2;
    int s1 = 16;
    int s2 = 128;
    int d = 128;

    // (a + b)@c -> out
    Tensor a(DT_FP32, {batch * s1, s2}, "a");
    Tensor b(DT_FP32, {batch * s2, d}, "b");
    Tensor out(DT_FP32, {batch * s1, d}, "out");

    Tensor actSeqs(DT_INT32, {batch}, "actual_seq");

    float inputValue = 1.0f;
    float initValue = 0.5f;
    int acutalValue = 62;
    std::vector<int> actSeqsData(batch, acutalValue);
    std::vector<float> golden(batch * s1 * d, initValue);
    for (int bsIdx = 0; bsIdx < batch * s1; ++bsIdx) {
        int offset = bsIdx * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + acutalValue, 128.0f);
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(a, inputValue),
        RawTensorData::CreateConstantTensor<float>(b, inputValue),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {a, b, actSeqs}, {out})
    {
        LOOP("LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(GetInputShape(a, 0) / s1))
        {
            SymbolicScalar actS2 = GetTensorData(actSeqs, {bIdx});

            Tensor aView = View(a, {s1, s2}, {s1, s2}, {bIdx * s1, 0});
            Tensor bView = View(b, {s2, d}, {s2, actS2}, {bIdx * s2, 0});

            TileShape::Current().SetVecTile(16, 64);
            Tensor aFp16 = Cast(aView, DataType::DT_FP16);
            TileShape::Current().SetVecTile(128, 64);
            Tensor bFp16 = Cast(bView, DataType::DT_FP16);

            auto tmpO = Matrix::Matmul(DataType::DT_FP32, aFp16, bFp16, false, false); // {s1, actS2} @ {actS2, d}
            Assemble(tmpO, {bIdx * s1, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

// test View + Reshape + Assemble 4->2 + op  2batch will wrong
TEST_F(DynamicReshapeTest, test_reshape_dassemble_4_2)
{
    TileShape::Current().SetVecTile(1, 1, 64, 64);

    int b = 2;
    int s = 1;
    int n1 = 64;
    int d = 64;

    // [b,s1,n1,d] -> [b*s1*n1,d]
    Tensor queryOut(DT_FP32, {b, s, n1, d}, "queryOut");
    Tensor qNope(DT_FP32, {b * s * n1, d}, "qNope");
    Tensor qRes(DT_FP32, {b * s * n1, d}, "qRes");

    FUNCTION("main", {queryOut}, {qNope, qRes})
    {
        LOOP("RESHAPE_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true)
        {
            SymbolicScalar bOffset = bIdx * 1;
            LOOP("RESHAPE_LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, s, 1))
            {
                SymbolicScalar sOffset = sIdx * 1;

                Tensor nopeView = View(queryOut, {1, 1, n1, d}, {bOffset, sOffset, 0, 0});
                TileShape::Current().SetVecTile({1, 1, 32, d});
                Tensor nopeRes = Reshape(nopeView, {1 * 1 * n1, d});
                Assemble(nopeRes, {(bOffset * s + sOffset) * n1, 0}, qNope);
            }
        }

        LOOP("Add_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true)
        {
            auto qNopeL = View(qNope, {64, 64}, {bIdx * s * n1, 0});
            TileShape::Current().SetVecTile(64, 64);
            auto qResTmp = Add(qNopeL, Element(DataType::DT_FP32, 1.0));
            Assemble(qResTmp, {bIdx * s * n1, 0}, qRes);
        }
    }

    float inputValue = 2.0f;
    float initValue = 0.5f;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(queryOut, inputValue),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(qNope, initValue),
        RawTensorData::CreateConstantTensor<float>(qRes, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden_qNope(b * s * n1 * d, inputValue);
    std::vector<float> golden_qRes(b * s * n1 * d, inputValue + 1.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputDataList();
    EXPECT_TRUE(resultCmp(golden_qNope, (float*)outs[0]->data(), 0.001f)); // right
    EXPECT_TRUE(resultCmp(golden_qRes, (float*)outs[1]->data(), 0.001f));  // wrong
}

//  dassemble + op + unaligin  Dassemble 不推导 validshape而是使用dst的shape时，后续操作会有问题

/*
 * test copy case
 */

// test View + Reshape + Assemble 2->3
TEST_F(DynamicReshapeTest, test_reshape_dassemble)
{
    TileShape::Current().SetVecTile(64, 64);

    int b = 1;
    int sq = 64;
    int d = 64;
    std::vector<int64_t> qShape2Dim = {b * sq, d};
    std::vector<int64_t> qShape3Dim = {b, sq, d};

    Tensor q(DT_FP32, qShape2Dim, "q");
    Tensor out(DT_FP32, qShape3Dim, "out");

#if 1

    FUNCTION("main", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (sq)))
        {
            Tensor q0 = View(q, {sq, d}, {batchId * sq, 0});
            // auto tmp0 = Mul(q0, Element(DataType::DT_FP32, 1.0));
            auto tmp = Reshape(q0, {1, sq, d});
            TileShape::Current().SetVecTile(1, 64, 64);
            // auto tmp = Mul(tmp, Element(DataType::DT_FP32, 1.0));
            Assemble(tmp, {batchId, 0, 0}, out);
        }
    }
#else

    FUNCTION("main", {q}, {out})
    {
        Tensor q0 = View(q, {sq, d}, {sq, 0});
        auto tmp0 = Mul(q0, Element(DataType::DT_FP32, 1.0));
        auto tmp = Reshape(q0, {1, sq, d});
        TileShape::Current().SetVecTile(1, 64, 64);
        Assemble(tmp, {0, 0, 0}, out);
    }
#endif

    float inputValue = 2.0f;
    float initValue = 0.5f;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(q, inputValue),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden(b * sq * d, inputValue);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float*)outs->data(), 0.001f));
}

// ===================  reshape + op + reshape  ??????
TEST_F(DynamicReshapeTest, test_reshape_op_reshape)
{
    TileShape::Current().SetVecTile(1, 1, 64, 64);

    int b = 2;
    int s = 1;
    int n1 = 64;
    int d = 64;

    // [b,s1,n1,d] -> [b*s1*n1,d]
    Tensor queryOut(DT_FP32, {b, s, n1, d}, "queryOut");
    Tensor qNope(DT_FP32, {b * s, n1, d}, "qNope");

    FUNCTION("main", {queryOut}, {qNope})
    {
        LOOP("RESHAPE_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true)
        {
            SymbolicScalar bOffset = bIdx * 1;
            LOOP("RESHAPE_LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, s, 1))
            {
                SymbolicScalar sOffset = sIdx * 1;
                Tensor nopeView = View(queryOut, {1, 1, n1, d}, {bOffset, sOffset, 0, 0});
                TileShape::Current().SetVecTile({1, 1, 64, 64});
                Tensor tmp0 = Reshape(nopeView, {1 * 1 * n1, d});
                auto tmp1 = Add(tmp0, Element(DataType::DT_FP32, 1.0));
                TileShape::Current().SetVecTile({1, 64, 64});
                auto tmp2 = Reshape(tmp1, {1, n1, d});
                auto nopeRes = Mul(tmp2, Element(DataType::DT_FP32, 1.0));
                Assemble(nopeRes, {(bOffset * s + sOffset), 0, 0}, qNope);
            }
        }
    }

    float inputValue = 2.0f;
    float initValue = 0.5f;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(queryOut, inputValue),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(qNope, initValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> golden_qNope(b * s * n1 * d, inputValue + 2.0f);

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputDataList();
    EXPECT_TRUE(resultCmp(golden_qNope, (float*)outs[0]->data(), 0.001f)); // right
}

TEST_F(DynamicReshapeTest, test_merge)
{
    TileShape::Current().SetVecTile(16, 16);

    int s = 16, d = 32;
    int actD = -1;

    std::vector<int64_t> inputShape = {s, actD};
    std::vector<int64_t> outputShape = {actD};

    Tensor q(DT_FP16, inputShape, "q");
    Tensor out(DT_FP16, {16 * GetInputShape(q, 1)}, "out");

    FUNCTION("main", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, l0Idx, LoopRange(0, (GetInputShape(q, 1) + 31) / 32, 1))
        {
            auto a = View(q, {s, 32}, {16, GetInputShape(q, 1)}, {0, l0Idx * 32});
            auto a1 = Reshape(a, {s * d}, {16 * GetInputShape(q, 1)});
            TileShape::Current().SetVecTile(16 * 16);
            auto a2 = Add(a1, Element(DataType::DT_FP32, 1.0f));
            Assemble(a2, {l0Idx * 32}, out);
        }
    }
    actD = 30;
    inputShape = {s, actD};
    outputShape = {s * actD};
    Tensor q_real(DT_FP16, inputShape, "q");
    Tensor out_real(DT_FP16, outputShape, "out");

    npu::tile_fwk::float16 initInputValue = 2.0f;
    npu::tile_fwk::float16 initOutValue = 0.5f;

    std::vector<npu::tile_fwk::float16> inputValueData;
    for (int i = 0; i < s * actD; i++) {
        inputValueData.push_back(static_cast<npu::tile_fwk::float16>(i));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(q_real, inputValueData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out_real, initOutValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));

    std::vector<npu::tile_fwk::float16> golden(s * actD, initInputValue);
    for (int i = 0; i < s * actD; i++) {
        golden[i] = inputValueData[i] + 1.0f;
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::float16*)outs->data(), 0.001f, 0, 1000, true));
}

TEST_F(DynamicReshapeTest, test_split)
{
    int s = 16, d = 32;
    int actSd = -1;
    int actD = -1;

    std::vector<int64_t> inputShape = {actSd};
    std::vector<int64_t> outputShape = {s, actD};

    Tensor q(DT_FP16, inputShape, "q");
    Tensor out(DT_FP16, {s, GetInputShape(q, 0) / s}, "out");

    FUNCTION("main", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, l0Idx, LoopRange(1))
        {
            (void)l0Idx;
            TileShape::Current().SetVecTile(16 * 16);
            auto a = View(q, {s * d}, {GetInputShape(q, 0)}, {0});
            auto a1 = Reshape(a, {s, d}, {s, GetInputShape(q, 0) / s});
            TileShape::Current().SetVecTile(16, 16);
            auto a2 = Add(a1, Element(a1.GetStorage()->Datatype(), 1.0f));
            Assemble(a2, {0, 0}, out);
        }
    }
    actD = 30;
    inputShape = {s * actD};
    outputShape = {s, actD};
    Tensor q_real(DT_FP16, inputShape, "q");
    Tensor out_real(DT_FP16, outputShape, "out");

    npu::tile_fwk::float16 initInputValue = 2.0f;
    npu::tile_fwk::float16 initOutValue = 0.5f;

    std::vector<npu::tile_fwk::float16> inputValueData;
    for (int i = 0; i < s * actD; i++) {
        inputValueData.push_back(static_cast<npu::tile_fwk::float16>(i));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(q_real, inputValueData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out_real, initOutValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));

    std::vector<npu::tile_fwk::float16> golden(s * actD, initInputValue);
    for (int i = 0; i < s * actD; i++) {
        golden[i] = inputValueData[i] + 1.0f;
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::float16*)outs->data(), 0.001f, 0, 1000, true));
}

TEST_F(DynamicReshapeTest, test_merge_and_split)
{
    TileShape::Current().SetVecTile(16, 16);

    int s = 16, d = 32;
    int actD = -1;

    std::vector<int64_t> inputShape = {s, actD};
    std::vector<int64_t> outputShape = {s, actD};

    Tensor q(DT_FP16, inputShape, "q");
    Tensor out(DT_FP16, outputShape, "out");

    FUNCTION("main", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, l0Idx, LoopRange(0, (GetInputShape(q, 1) + 31) / 32, 1))
        {
            TileShape::Current().SetVecTile(16, 16);
            auto a = View(q, {s, d}, {16, GetInputShape(q, 1)}, {0, l0Idx * 32});
            auto a1 = Reshape(a, {16 * 32}, {16 * GetInputShape(q, 1)});
            TileShape::Current().SetVecTile(16 * 16);
            auto a2 = Add(a1, Element(DataType::DT_FP32, 1.0f));
            auto a3 = Reshape(a2, {{s, d}}, {s, GetInputShape(q, 1)});
            TileShape::Current().SetVecTile(16, 16);
            Assemble(a3, {0, l0Idx * 32}, out);
        }
    }
    actD = 30;
    inputShape = {s, actD};
    outputShape = {s, actD};
    Tensor q_real(DT_FP16, inputShape, "q");
    Tensor out_real(DT_FP16, outputShape, "out");

    npu::tile_fwk::float16 initInputValue = 2.0f;
    npu::tile_fwk::float16 initOutValue = 0.5f;

    std::vector<npu::tile_fwk::float16> inputValueData;
    for (int i = 0; i < s * actD; i++) {
        inputValueData.push_back(static_cast<npu::tile_fwk::float16>(i));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(q_real, inputValueData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out_real, initOutValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));

    std::vector<npu::tile_fwk::float16> golden(s * actD, initInputValue);
    for (int i = 0; i < s * actD; i++) {
        golden[i] = inputValueData[i] + 1.0f;
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::float16*)outs->data(), 0.001f, 0, 1000));
}

TEST_F(DynamicReshapeTest, test_split_and_merge)
{
    TileShape::Current().SetVecTile(16, 16);

    int s = 16, d = 32;
    int actD = -1;

    std::vector<int64_t> inputShape = {-1};
    std::vector<int64_t> outputShape = {-1};

    Tensor q(DT_FP16, inputShape, "q");
    Tensor out(DT_FP16, outputShape, "out");

    FUNCTION("main", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, l0Idx, LoopRange(0, 1, 1))
        {
            TileShape::Current().SetVecTile(16 * 16);
            auto a = View(q, {s * d}, {GetInputShape(q, 0)}, {l0Idx * 32});
            auto a1 = Reshape(a, {s, d}, {s, GetInputShape(q, 0) / s});
            TileShape::Current().SetVecTile(16, 16);
            auto a3 = Add(a1, Element(DataType::DT_FP32, 1.0f));
            auto a4 = Reshape(a3, {s * d}, {GetInputShape(q, 0)});
            TileShape::Current().SetVecTile(16 * 16);
            auto a5 = Add(a4, Element(DataType::DT_FP32, 1.0f));
            Assemble(a5, {0}, out);
        }
    }
    actD = 30;
    inputShape = {s * actD};
    outputShape = {s * actD};
    Tensor q_real(DT_FP16, inputShape, "q");
    Tensor out_real(DT_FP16, outputShape, "out");

    npu::tile_fwk::float16 initInputValue = 2.0f;
    npu::tile_fwk::float16 initOutValue = 0.5f;

    std::vector<npu::tile_fwk::float16> inputValueData;
    for (int i = 0; i < s * actD; i++) {
        inputValueData.push_back(static_cast<npu::tile_fwk::float16>(i));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(q_real, inputValueData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out_real, initOutValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));

    std::vector<npu::tile_fwk::float16> golden(s * actD, initInputValue);
    for (int i = 0; i < s * actD; i++) {
        golden[i] = inputValueData[i] + 2.0f;
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::float16*)outs->data(), 0.001f, 0, 1000));
}

TEST_F(DynamicReshapeTest, test_exchange_dim)
{
    TileShape::Current().SetVecTile(16, 16);

    int s = 16, d = 32;
    int actD = -1;
    std::vector<int64_t> inputShape = {16, -1};
    std::vector<int64_t> outputShape = {-1, 16};

    Tensor q(DT_FP16, inputShape, "q");
    Tensor out(DT_FP16, outputShape, "out");

    FUNCTION("main", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, l0Idx, LoopRange(0, 1, 1))
        {
            TileShape::Current().SetVecTile(16, 16);
            auto a = View(q, {s, d}, {s, GetInputShape(q, 1)}, {0, l0Idx * 32});
            auto a1 = Reshape(a, {d, s}, {GetInputShape(q, 1), s});
            TileShape::Current().SetVecTile(16, 16);
            auto a3 = Add(a1, Element(a1.GetStorage()->Datatype(), 1.0f));
            Assemble(a3, {0, 0}, out);
        }
    }
    actD = 30;
    inputShape = {s, actD};
    outputShape = {actD, s};
    Tensor q_real(DT_FP16, inputShape, "q");
    Tensor out_real(DT_FP16, outputShape, "out");

    npu::tile_fwk::float16 initInputValue = 2.0f;
    npu::tile_fwk::float16 initOutValue = 0.5f;

    std::vector<npu::tile_fwk::float16> inputValueData;
    for (int i = 0; i < s * actD; i++) {
        inputValueData.push_back(static_cast<npu::tile_fwk::float16>(i));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(q_real, inputValueData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out_real, initOutValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));

    std::vector<npu::tile_fwk::float16> golden(s * actD, initInputValue);
    for (int i = 0; i < s * actD; i++) {
        golden[i] = inputValueData[i] + 1.0f;
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::float16*)outs->data(), 0.001f, 0, 1000));
}

TEST_F(DynamicReshapeTest, test_special_reshape)
{
    TileShape::Current().SetVecTile(16, 16);

    int s = 16, d = 32;
    int actD = -1;
    std::vector<int64_t> inputShape = {16, -1};
    std::vector<int64_t> outputShape = {16 * 2, -1};

    Tensor q(DT_FP16, inputShape, "q");
    Tensor out(DT_FP16, {s * 2, GetInputShape(q, 1) / 2}, "out");

    FUNCTION("main", {q}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, l0Idx, LoopRange(0, 1, 1))
        {
            TileShape::Current().SetVecTile(16, 16);
            auto a = View(q, {s, d}, {s, GetInputShape(q, 1)}, {0, l0Idx * 32});
            auto a1 = Reshape(a, {s * 2, d / 2}, {s * 2, GetInputShape(q, 1) / 2}); // (16, 30) --> (32, 15)
            TileShape::Current().SetVecTile(16, 16);
            auto a3 = Add(a1, Element(a1.GetStorage()->Datatype(), 1.0f));
            Assemble(a3, {0, 0}, out);
        }
    }
    actD = 30;
    inputShape = {s, actD};
    outputShape = {s * 2, actD / 2};
    Tensor q_real(DT_FP16, inputShape, "q");
    Tensor out_real(DT_FP16, outputShape, "out");

    npu::tile_fwk::float16 initInputValue = 2.0f;
    npu::tile_fwk::float16 initOutValue = 0.5f;

    std::vector<npu::tile_fwk::float16> inputValueData;
    for (int i = 0; i < s * actD; i++) {
        inputValueData.push_back(static_cast<npu::tile_fwk::float16>(i));
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(q_real, inputValueData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out_real, initOutValue),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                       DeviceLauncherConfig(q_real.GetStorage()->GetDataSize()));

    std::vector<npu::tile_fwk::float16> golden(s * actD, initInputValue);
    for (int i = 0; i < s * actD; i++) {
        golden[i] = inputValueData[i] + 1.0f;
    }

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (npu::tile_fwk::float16*)outs->data(), 0.001f, 0, 1000));
}
