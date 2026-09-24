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
 * \file test_codegen_scalar.cpp
 * \brief Unit test for codegen.
 */

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;
constexpr int DIM4 = 4;
constexpr int VALUE128 = 128;
constexpr float F_127 = 127.0;

class TestCodegenScalar : public CodegenTestBase {
public:
    TestCodegenScalar() : CodegenTestBase({.compileStage = CS_EXECUTE_GRAPH, .buildStatic = true}) {}
};

void TestQuant(std::vector<int64_t>& inputShape)
{
    int shapeDim = inputShape.size();
    std::vector<int64_t> scaleShape(shapeDim, 0);
    for (int i = 0; i < shapeDim; i++) {
        scaleShape[i] = (i == shapeDim - 1) ? 1 : inputShape[i];
    }

    std::vector<int64_t> vecTileShape = {VALUE128, VALUE128};

    // depend on shapeDim
    switch (shapeDim) {
        case DIM2:
            TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
            break;
        case DIM3:
            TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[0], vecTileShape[1]);
            break;
        case DIM4:
            TileShape::Current().SetVecTile(1, 1, vecTileShape[0], vecTileShape[1]);
            break;
        default:
            ASSERT(GenCodeErr::TENSOR_DIM_UNSUPPORTED, shapeDim <= DIM4) << "unsupport dim " << shapeDim << " \n";
            break;
    }

    Tensor input(DataType::DT_FP16, inputShape, "input");
    Tensor output(DataType::DT_INT8, inputShape, "output");
    Tensor scaleDeQuant(DataType::DT_FP32, scaleShape, "scaleDeQuant");

    std::string funcName = "Quant";
    FUNCTION(funcName, {input, output, scaleDeQuant})
    {
        auto res = Quant(input);
        output = std::get<0>(res);
        scaleDeQuant = std::get<1>(res);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenScalar, DISABLED_TestQuant_32_1_7168)
{
    std::vector<int64_t> inputShape = {32, 1, 7168};
    TestQuant(inputShape);
}

TEST_F(TestCodegenScalar, TestScalarOp)
{
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 35};

    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "res");
    std::string funcName = "ScalarAddS";
    FUNCTION(funcName, {input, output})
    {
        auto output_a = ScalarAddS(input, Element(DataType::DT_FP32, F_127), true);
        auto output_b = ScalarSubS(output_a, Element(DataType::DT_FP32, F_127), true);
        auto output_c = ScalarMulS(output_b, Element(DataType::DT_FP32, F_127), true);
        auto output_d = ScalarDivS(output_c, Element(DataType::DT_FP32, F_127), true);
        output = ScalarMaxS(output_d, Element(DataType::DT_FP32, F_127), true);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function);
}

TEST_F(TestCodegenScalar, TestPipeAll)
{
    auto function = GenMockFuncStatic("TestPipeAll");
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto ddrTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, dynValidShape});
    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    Operation& syncOp = function->AddOperation(npu::tile_fwk::Opcode::OP_BAR_ALL, {ddrTensor}, {ubTensor});
    syncOp.syncQueue_ = {PipeType::PIPE_ALL,   PipeType::PIPE_ALL,  CoreType::AIV, CoreType::AIV, -1,
                         AIVCore::UNSPECIFIED, AIVCore::UNSPECIFIED};

    std::string res = GenOpCodeFromOp(*function, syncOp);
    std::string expect = R"!!!(pipe_barrier(PIPE_ALL);
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenScalar, TestAicpuCallOp)
{
    auto function = GenMockFuncStatic("TestAicpuCallOp");
    std::vector<int64_t> shape = {64, 64};
    auto ubTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape});
    Operation& op = function->AddOperation(npu::tile_fwk::Opcode::OP_AICPU_CALL_AIV, {ubTensor}, {});
    op.SetAttribute(OpAttributeKey::aicpuCall, 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect = R"!!!(TileOp::AicpuCall<0,0>(GET_CURRENT_TASKID());
)!!!";

    EXPECT_EQ(res, expect);
}

void TestCrossCoreSyncBody(std::string funcName, Opcode syncOpcode)
{
    auto function = GenMockFuncStatic(funcName);
    std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto& op = function->AddOperation(syncOpcode, {localTensor}, {localOutTensor});

    std::string res = GenOpCodeFromOp(*function, op);
    static const std::map<Opcode, std::string> expectList = {
        {Opcode::OP_CV_SYNC_SRC, R"!!!(set_intra_block(PIPE_S, 0);)!!!"},
        {Opcode::OP_CV_SYNC_DST, R"!!!(wait_intra_block(PIPE_S, 0);)!!!"},
        {Opcode::OP_FFTS_CROSS_CORE_SYNC, R"!!!(ffts_cross_core_sync(PIPE_S, getFFTSMsg(0x2, 0));)!!!"},
        {Opcode::OP_WAIT_FLAG_DEV, R"!!!(wait_flag_dev(PIPE_S, 0);)!!!"},
    };

    std::string expect = expectList.at(syncOpcode);
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenScalar, InjectSyncSet) { TestCrossCoreSyncBody("InjectSyncSet", Opcode::OP_CV_SYNC_SRC); }

TEST_F(TestCodegenScalar, InjectSyncWait) { TestCrossCoreSyncBody("InjectSyncWait", Opcode::OP_CV_SYNC_DST); }

TEST_F(TestCodegenScalar, InjectFFTSCrossCoreSync)
{
    TestCrossCoreSyncBody("InjectFFTSCrossCoreSync", Opcode::OP_FFTS_CROSS_CORE_SYNC);
}

TEST_F(TestCodegenScalar, InjectWaitFlagDev) { TestCrossCoreSyncBody("InjectWaitFlagDev", Opcode::OP_WAIT_FLAG_DEV); }

} // namespace npu::tile_fwk
