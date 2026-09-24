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
 * \file test_codegen_dyn_indexoutcast.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "passes/pass_mgr/pass_manager.h"
#include "codegen/codegen.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"
#include "interface/utils/id_gen.h"

namespace npu::tile_fwk {
class TestCodegenDynIndexOutCast : public CodegenTestBase {
public:
    TestCodegenDynIndexOutCast()
        : CodegenTestBase(
              {.compileStage = CS_EXECUTE_GRAPH, .setTileTensor = true, .tileTensorValue = false, .setIdGen = true})
    {}

    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynIndexOutCast, IndexOutCast)
{
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    int S = 1;
    int S2 = 16;
    int kvLoraRank = 8;
    int qkRopeHeadDim = 8;

    const std::vector<int64_t> shape0 = {S2, kvLoraRank + qkRopeHeadDim}; // [16, 16]
    const std::vector<int64_t> shape1 = {1, S};
    const std::vector<SymbolicScalar> dynValidShape1 = {1, S};
    const std::vector<int64_t> shape2 = {S, kvLoraRank + qkRopeHeadDim}; // [1, 16]
    const std::vector<SymbolicScalar> dynValidShape2 = {S, kvLoraRank + qkRopeHeadDim};

    TileShape::Current().SetVecTile(16, 16);

    Tensor kv_len(DataType::DT_INT64, shape1, "kv_len");
    Tensor past_key_states(DataType::DT_FP32, shape0, "past_key_states");
    Tensor key_states(DataType::DT_FP32, shape2, "key_states"); // [16,16]

    std::string funcName = "ScatterUpdate";
    FUNCTION(funcName, {kv_len, key_states, past_key_states})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            past_key_states = ScatterUpdate(past_key_states, kv_len, key_states, -2);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);

    auto ddrTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape0, "IndexOutCast"});
    auto localTensorSrc0 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape2, dynValidShape2});
    auto localTensorSrc1 = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, shape1, dynValidShape1});

    auto& op = function->AddOperation(Opcode::OP_INDEX_OUTCAST, {localTensorSrc0, localTensorSrc1, ddrTensor},
                                      {ddrTensor});
    op.SetAttribute("axis", 0);
    op.SetAttribute(OpAttributeKey::panzBlockSize, 1);
    std::string cacheMode = "PA_BNSD";
    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
    auto to_offset = OpImmediate::Specified({0, 0});
    auto shapeImme = OpImmediate::Specified({16, 16});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, to_offset, shapeImme, shapeImme));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetOOpAtt(0, 0);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);

    std::string res = GenOpCodeFromOp(*function, op);
    std::string expect =
        R"!!!(TileOp::DynTIndexoutcast<float, float, 1, 1, 16, 1, 1, 16, 1, 1, 1, 0, 1>((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, GET_PARAM_RAWSHAPE_2(param, 0, 0), 0, 0, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynIndexOutCast, TestIndexOutTileTensor)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);

    auto function = GenMockFuncDyn("IndexoutTileTensor");

    std::vector<int64_t> scaterShape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto indexoutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, scaterShape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor(
        {*function, DataType::DT_FP32, MemoryType::MEM_UB, scaterShape, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    indexoutTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    localOutTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    LogicalTensors inputs = {localOutTensor, localOutTensor, localOutTensor};
    LogicalTensors outputs = {indexoutTensor};

    auto& indexoutOp = function->AddOperation(Opcode::OP_INDEX_OUTCAST, inputs, outputs);
    indexoutOp.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    indexoutOp.SetAttribute("axis", 0);
    indexoutOp.SetAttribute(OpAttributeKey::panzBlockSize, 1);
    std::string cacheMode = "PA_BNSD";
    indexoutOp.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
    auto to_offset = OpImmediate::Specified({0, 0});
    auto shapeImme = OpImmediate::Specified(scaterShape);
    indexoutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, to_offset, shapeImme, shapeImme));
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(indexoutOp.GetOpAttribute());
    indexoutOp.SetOOpAtt(0, 0);
    std::string res = GenOpCodeFromOp(*function, indexoutOp, {.isMainBlk = true});
    std::string expect = R"!!!(TIndexOutcast<0, 1>(gmTensor_0, ubTensor_1, ubTensor_1, Coord2Dim(0, 0));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynIndexOutCast, DynIndexOutUnaligned)
{
    TileShape::Current().SetVecTile({32, 32});

    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"DuplicateOp", PassName::DUPLICATE_OP},
                                     {"MergeViewAssemble", PassName::MERGE_VIEW_ASSEMBLE},
                                     {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"RemoveRedundantOp", PassName::REMOVE_REDUNDANT_OP},
                                     {"GenerateMoveOp", PassName::GENERATE_MOVE_OP},
                                 });

    int h = 32;
    int minusTwo = -2;
    Tensor output(DT_INT32, {h, h}, "output");
    Tensor idxs(DT_INT32, {h, h}, "idxs");
    Tensor keyStates(DT_INT32, {h, h}, "keyStates");

    std::string funcName = "DynIndexOutUnaligned";
    FUNCTION(funcName + "Main", {idxs, keyStates}, {output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = ScatterUpdate(output, idxs, keyStates, minusTwo);
        }
    }
#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX);
#endif

    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TileOp::DynTIndexoutcast<int32_t, int32_t, 1, 32, 32, 32, 0, 1>((__gm__ int32_t*)(RUNTIME_GET_PARAM_ADDR(RUNTIME_param, 2, 28)), (__ubuf__ int32_t*)UB_S0_E4096, (__ubuf__ int32_t*)UB_S4096_E8192, 1, 1, sym_76_dim_1, sym_79_dim_0, sym_79_dim_1, 1, 1, GET_PARAM_RAWSHAPE_2(param, 2, 28), 0, 0, (RUNTIME_COA_GET_PARAM_OFFSET(2, 28, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 28, 1)));
)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
