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
 * \file test_codegen_dyn_binary.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include <regex>

#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"
#include "utils/host_log/log_manager.h"

namespace npu::tile_fwk {
class TestCodegenDynBinary : public CodegenTestBase {
public:
    TestCodegenDynBinary()
        : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .setTileTensor = true, .setIdGen = true})
    {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

void TestAddDynBody(const std::string& name, bool isNeedCalcMinForBinaryOperands = false)
{
    auto function = GenMockFuncDyn(name);
    for (auto& subFunc : function->rootFunc_->programs_) {
        for (auto& op : subFunc.second->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ADD && isNeedCalcMinForBinaryOperands) {
                op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
            }
        }
    }
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TileOp::DynTadd_<float, /*DS*/ 1, 1, 64, 64, /*S0*/ 1, 1, 64, 64, /*S1*/ 1, 1, 64, 64>((__ubuf__ float*)UB_S0_E16384+(((RUNTIME_COA_GET_PARAM_OFFSET(2, 46, 0)) * 64) + (RUNTIME_COA_GET_PARAM_OFFSET(2, 46, 1))), (__ubuf__ float*)UB_S0_E16384+(((RUNTIME_COA_GET_PARAM_OFFSET(2, 37, 0)) * 64) + (RUNTIME_COA_GET_PARAM_OFFSET(2, 37, 1))), (__ubuf__ float*)UB_S16384_E32768+(((RUNTIME_COA_GET_PARAM_OFFSET(2, 28, 0)) * 64) + (RUNTIME_COA_GET_PARAM_OFFSET(2, 28, 1))), 1, 1, )!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestCodegenAddDim2) { TestAddDynBody("TestCodegenAddDim2"); }

TEST_F(TestCodegenDynBinary, TestCodegenAddDim2SrcNotSameShape)
{
    TestAddDynBody("TestCodegenAddDim2SrcNotSameShape", true);
}

void TestAddSDynBody(const std::string& testName, float scalar, bool isSupportTileTensor,
                     const std::vector<std::string>& expect)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }

    std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile({64, 64});
    Tensor input_a(DataType::DT_FP32, shape, "A");
    Element value(DataType::DT_FP32, scalar);
    Tensor output(DataType::DT_FP32, shape, "C");
    ConfigManager::Instance();

    std::string funcName = testName;
    FUNCTION(funcName, {input_a, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Add(input_a, value);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    for (auto& e : expect) {
        CheckStringExist(e, res);
    }
}

TEST_F(TestCodegenDynBinary, TestAddSDynamic)
{
    std::vector<std::string> expect = {
        R"!!!(TileOp::DynTadds_<float, /*DS*/ 1, 64, 64, /*S0S*/ 1, 64, 64>((__ubuf__ float*)UB_S0_E16384, (__ubuf__ float*)UB_S0_E16384, (float)1.5f, 1, 1, sym_39_dim_0, sym_39_dim_1);
)!!!"};
    TestAddSDynBody("TestAddsDynamic", 1.5, false, expect);
}

TEST_F(TestCodegenDynBinary, TestAddSTileTensorInfPos)
{
    std::vector<std::string> expect = {
        R"!!!(union {float f; uint32_t u;} float_inf_pos = {.u = 0x7F800000};)!!!",
        R"!!!(TAddS<LastUse2Dim<0, 0>, float>(ubTensor_0, ubTensor_0, float_inf_pos.f);)!!!"};
    TestAddSDynBody("TestAddSTileTensorInfPos", 1.0f / 0.0f, true, expect);
}

TEST_F(TestCodegenDynBinary, TestAddSTileTensorInfNeg)
{
    std::vector<std::string> expect = {
        R"!!!(union {float f; uint32_t u;} float_inf_neg = {.u = 0xFF800000};)!!!",
        R"!!!(TAddS<LastUse2Dim<0, 0>, float>(ubTensor_0, ubTensor_0, float_inf_neg.f);)!!!"};
    TestAddSDynBody("TestAddSTileTensorInfNeg", -1.0f / 0.0f, true, expect);
}

TEST_F(TestCodegenDynBinary, TestAddSTileTensorNAN)
{
    std::vector<std::string> expect = {
        R"!!!(union {float f; uint32_t u;} float_nan = {.u = 0x7FC00000};)!!!",
        R"!!!(TAddS<LastUse2Dim<0, 0>, float>(ubTensor_0, ubTensor_0, float_nan.f);)!!!"};
    TestAddSDynBody("TestAddSTileTensorNAN", 0.0f / 0.0f, true, expect);
}

TEST_F(TestCodegenDynBinary, TestGatherEle)
{
    constexpr const int32_t nRoutedExperts = 256;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({16, 32});
    Tensor inputScores(DT_INT32, outputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    std::string funcName = "GATHER_ELEMET_T";
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            outputTensor = GatherElements(inputTmpScores, inputScores, 1); // [b*s,8]
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TileOp::DynTgatherElement<float, int32_t, 1, 2, 256, 1, 2, 8, 1, 2, 8, 3>((__ubuf__ float*)UB_S2176_E2240, (__ubuf__ float*)UB_S0_E2048, (__ubuf__ int32_t*)UB_S2048_E2112, 1, 1, sym_59_dim_0, sym_59_dim_1);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestGatherEleTileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    constexpr const int32_t nRoutedExperts = 256;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({16, 32});
    Tensor inputScores(DT_INT32, outputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    std::string funcName = "GATHER_ELEMET_TILETENSOR";
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            outputTensor = GatherElements(inputTmpScores, inputScores, 1); // [b*s,8]
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TgatherElement<4>(ubTensor_4, ubTensor_0, ubTensor_2, ubTensor_5);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestAtan2FP32)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    std::vector<int64_t> shape = {32, 32};
    TileShape::Current().SetVecTile({32, 32});
    Tensor input1(DataType::DT_FP32, shape, "input1");
    Tensor input2(DataType::DT_FP32, shape, "input2");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestAtan2FP32";
    FUNCTION(funcName, {input1, input2, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Atan2(input1, input2);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect = R"!!!(TAtan2(ubTensor_4, ubTensor_0, ubTensor_2, ubTensor_5);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, AddUnalignTileTensor)
{
    TileShape::Current().SetVecTile(64, 64);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> inputShape = {b * sq, d};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor input1(DT_FP32, inputShape, "intput1");
    Tensor input2(DT_FP32, inputShape, "intput2");
    Tensor curSeq(DT_INT32, {b, 1}, "curSeq");
    Tensor out(DT_FP32, outShape, "out");

    std::string loopName = "L0_TILETENSOR";
    FUNCTION("main", {input1, input2, curSeq}, {out})
    {
        LOOP(loopName, FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b))
        {
            auto seq = GetTensorData(curSeq, {batchId, 0});
            Tensor intput11 = View(input1, {sq, d}, {seq, d}, {batchId, 0});
            Tensor intput22 = View(input2, {sq, d}, {seq, d}, {batchId, 0});
            auto tmp = Add(intput11, intput22);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    std::vector<int> actSeqsData(b, 100);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input1, 1.0),
        RawTensorData::CreateConstantTensor<float>(input2, 1.0),
        RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + loopName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + loopName + SUB_FUNC_SUFFIX);
#endif

    std::string res = GenCodeByFunction(*function);
#if ENABLE_HIDDENLOOP
    std::string expect = R"!!!(float *UB_S0_E16384_T = (float *)get_imm(0x0); // size: 0x4000
float *UB_S16384_E32768_T = (float *)get_imm(0x4000); // size: 0x4000
uint64_t sym_60_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_60_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
uint64_t sym_61_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_61_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
using GMTileTensorFP32Dim2_1 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_0 = TileTensor<float, LocalLayout2Dim<64, 64>, Hardware::UB>;
SUBKERNEL_PHASE1
UBTileTensorFP32Dim2_0 ubTensor_0((uint64_t)UB_S0_E16384_T, (Shape2Dim(sym_60_dim_0, sym_60_dim_1)));
GMTileTensorFP32Dim2_1 gmTensor_1((__gm__ float*)(RUNTIME_GET_PARAM_ADDR(RUNTIME_param, 1, 10)), DynLayout2Dim(Shape2Dim((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 10, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 10, 1))), Stride2Dim(GET_PARAM_STRIDE_DIM_2((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 10, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 10, 1))))));
TLoad(ubTensor_0, gmTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
UBTileTensorFP32Dim2_0 ubTensor_2((uint64_t)UB_S16384_E32768_T, (Shape2Dim(sym_61_dim_0, sym_61_dim_1)));
GMTileTensorFP32Dim2_1 gmTensor_3((__gm__ float*)(RUNTIME_GET_PARAM_ADDR(RUNTIME_param, 0, 1)), DynLayout2Dim(Shape2Dim((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 1, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 1, 1))), Stride2Dim(GET_PARAM_STRIDE_DIM_2((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 1, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 1, 1))))));
TLoad(ubTensor_2, gmTensor_3, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
TAdd<LastUse3Dim<0, 0, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
GMTileTensorFP32Dim2_1 gmTensor_7((__gm__ float*)(RUNTIME_GET_PARAM_ADDR(RUNTIME_param, 2, 19)), DynLayout2Dim(Shape2Dim((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 19, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 19, 1))), Stride2Dim(GET_PARAM_STRIDE_DIM_2((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 19, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 19, 1))))));
TStoreVec<TStoreConfigVec<pto::AtomicType::AtomicNone>>(gmTensor_7, ubTensor_0, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 1))));
}
)!!!";
#else
    std::string expect = R"!!!(float __ubuf__ *UB_S0_E16384 = (float __ubuf__ *)get_imm(0x0); // size: 0x4000
float *UB_S0_E16384_T = (float *)get_imm(0x0); // size: 0x4000
float __ubuf__ *UB_S16384_E32768 = (float __ubuf__ *)get_imm(0x4000); // size: 0x4000
float *UB_S16384_E32768_T = (float *)get_imm(0x4000); // size: 0x4000
uint64_t sym_13_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 0);
uint64_t sym_13_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 1);
uint64_t sym_76_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_76_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
uint64_t sym_77_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_77_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, LocalLayout2Dim<64, 64>, Hardware::UB>;
SUBKERNEL_PHASE1
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)GET_PARAM_ADDR(param, 1, 10), DynLayout2Dim(Shape2Dim((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 10, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 10, 1))), Stride2Dim(GET_PARAM_STRIDE_DIM_2(param, 1, 10))));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E16384_T, (Shape2Dim(sym_76_dim_0, sym_76_dim_1)));
TLoad(ubTensor_1, gmTensor_2, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
GMTileTensorFP32Dim2_2 gmTensor_4((__gm__ float*)GET_PARAM_ADDR(param, 0, 1), DynLayout2Dim(Shape2Dim((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 1, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 1, 1))), Stride2Dim(GET_PARAM_STRIDE_DIM_2(param, 0, 1))));
UBTileTensorFP32Dim2_1 ubTensor_3((uint64_t)UB_S16384_E32768_T, (Shape2Dim(sym_77_dim_0, sym_77_dim_1)));
TLoad(ubTensor_3, gmTensor_4, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
TAdd<LastUse3Dim<0, 0, 0>>(ubTensor_1, ubTensor_1, ubTensor_3);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
GMTileTensorFP32Dim2_2 gmTensor_8((__gm__ float*)GET_PARAM_ADDR(param, 2, 19), DynLayout2Dim(Shape2Dim((RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 19, 0)), (RUNTIME_COA_GET_PARAM_RAW_SHAPE(2, 19, 1))), Stride2Dim(GET_PARAM_STRIDE_2(param, 2, 19))));
TStoreVec<TStoreConfigVec<pto::AtomicType::AtomicNone>>(gmTensor_8, ubTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 1))));
}
)!!!";
#endif
    const std::regex dynamicSymbolPattern(R"(sym_[0-9]+_dim_)");
    res = std::regex_replace(res, dynamicSymbolPattern, "sym_dynamic_dim_");
    expect = std::regex_replace(expect, dynamicSymbolPattern, "sym_dynamic_dim_");
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestAddTileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    auto function = GenMockFuncDyn("TestAddTileTensor");

    std::vector<int64_t> addShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorA = CreateLogicalTensor(
        {*function, DataType::DT_FP16, MemoryType::MEM_UB, addShape, dynValidShape});
    auto localTensorB = CreateLogicalTensor(
        {*function, DataType::DT_FP16, MemoryType::MEM_UB, addShape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP16, MemoryType::MEM_UB, addShape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_ADD, {localTensorA, localTensorB}, {localOutTensor});
    std::vector<int64_t> initVec(addShape.size(), false);
    op.SetAttribute(OpAttributeKey::lastUse, initVec);

    std::shared_ptr<SymbolManager> symbolManager;
    auto cop = GenOpCloudNPUFromOp(*function, op, symbolManager);
    std::string res = cop.GenOpCode();
    std::string expect = R"!!!(TAdd<LastUse2Dim<0, 0>>(ubTensor_0, ubTensor_0, ubTensor_0);
)!!!";
    EXPECT_EQ(res, expect);

    expect = "UBTileTensorFP16Dim2_0";
    res = cop.QueryTileTensorTypeByIdx(0);
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynBinary, TestDivHighPrecisionFP16)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile({64, 64});
    Tensor input_a(DataType::DT_FP16, shape, "A");
    Tensor input_b(DataType::DT_FP16, shape, "B");
    Tensor output(DataType::DT_FP16, shape, "C");

    std::string funcName = "TestDivHighPrecisionFP16";
    FUNCTION(funcName, {input_a, input_b, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Div(input_a, input_b, PrecisionType::HIGH_PRECISION);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TDiv<pto::DivAlgorithm::HIGH_PRECISION, LastUse3Dim<0, 0, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestDivIntrinsicPrecision)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile({64, 64});
    Tensor input_a(DataType::DT_FP16, shape, "A");
    Tensor input_b(DataType::DT_FP16, shape, "B");
    Tensor output(DataType::DT_FP16, shape, "C");

    std::string funcName = "TestDivIntrinsicPrecision";
    FUNCTION(funcName, {input_a, input_b, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Div(input_a, input_b, PrecisionType::INTRINSIC);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TDiv<pto::DivAlgorithm::DEFAULT, LastUse3Dim<0, 0, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestPowHighPrecision)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile({64, 64});
    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor input_b(DataType::DT_FP32, shape, "B");
    Tensor output(DataType::DT_FP32, shape, "C");

    std::string funcName = "TestPowHighPrecision";
    FUNCTION(funcName, {input_a, input_b, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Pow(input_a, input_b, PrecisionType::HIGH_PRECISION);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TPow<pto::PowAlgorithm::HIGH_PRECISION>(ubTensor_4, ubTensor_0, ubTensor_2, ubTensor_5);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestPowIntrinsicPrecision)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile({64, 64});
    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor input_b(DataType::DT_FP32, shape, "B");
    Tensor output(DataType::DT_FP32, shape, "C");

    std::string funcName = "TestPowIntrinsicPrecision";
    FUNCTION(funcName, {input_a, input_b, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = Pow(input_a, input_b, PrecisionType::INTRINSIC);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    const std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TPow<pto::PowAlgorithm::DEFAULT>(ubTensor_4, ubTensor_0, ubTensor_2, ubTensor_5);
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynBinary, TestAxpyTileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);

    auto function = GenMockFuncDyn("TestAxpyTileTensor");

    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensorA = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorB = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto& op = function->AddOperation(Opcode::OP_AXPY, {localTensorA, localTensorB}, {localOutTensor});
    op.SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 2.0));
    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TAxpy(ubTensor_0, ubTensor_0, (float)2.f);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
