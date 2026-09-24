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
 * \file test_mix_info.cpp
 * \brief Unit test for mix_info module
 */

#include <gtest/gtest.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "machine/host/mix_info.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::mix_info;
using json = nlohmann::json;

class TestMixInfo : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        TileShape::Current().SetVecTile(64, 64);
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestMixInfo, DumpMixInfo_NullFunction)
{
    int result = DumpMixInfo(nullptr);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_ExecuteGraphFunction)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestExecuteGraphFunc";
    FUNCTION(funcName, {input, output})
    {
        auto tmp = Mul(input, input);
        output = Add(tmp, input);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::EXECUTE_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_TileGraphFunction)
{
    std::vector<int64_t> shape = {16, 16};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestTileGraphFunc";
    FUNCTION(funcName, {input, output}) { output = Sub(input, input); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::TILE_GRAPH);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_TensorGraphDynamicType)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestTensorGraphDynamicType";
    FUNCTION(funcName, {input, output}) { output = Add(input, input); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::TENSOR_GRAPH);
    function->SetFunctionType(FunctionType::DYNAMIC);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_LeafFuncWithSyncOps)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestLeafFuncWithSyncOps";
    FUNCTION(funcName, {input, output}) { output = Add(input, input); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::EXECUTE_GRAPH);

    auto leafFuncName = "LeafFuncWithSync";
    auto leafFunc = std::make_shared<Function>(Program::GetInstance(), leafFuncName, leafFuncName, function);
    leafFunc->SetGraphType(GraphType::TENSOR_GRAPH);

    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    leafAttr->mixId = 200;
    leafAttr->coreType = CoreType::AIV;
    leafFunc->SetLeafFuncAttribute(leafAttr);

    auto inCast = std::make_shared<LogicalTensor>(*leafFunc, DataType::DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*leafFunc, DataType::DT_FP32, shape);

    auto& syncSrcOp = leafFunc->AddOperation(Opcode::OP_CV_SYNC_SRC, {}, {});
    syncSrcOp.syncQueue_ = OpSyncQueue(PipeType::PIPE_M, PipeType::PIPE_M, CoreType::AIV, CoreType::AIV, 10,
                                       AIVCore::UNSPECIFIED, AIVCore::UNSPECIFIED);

    auto& syncDstOp = leafFunc->AddOperation(Opcode::OP_CV_SYNC_DST, {}, {});
    syncDstOp.syncQueue_ = OpSyncQueue(PipeType::PIPE_M, PipeType::PIPE_M, CoreType::AIV, CoreType::AIV, 20,
                                       AIVCore::UNSPECIFIED, AIVCore::UNSPECIFIED);

    Program::GetInstance().InsertFuncToFunctionMap(leafFunc->GetMagicName(), leafFunc);

    auto funcInCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
    auto funcOutCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);

    auto& callOp = function->AddOperation(Opcode::OP_CALL, {funcInCast}, {funcOutCast});
    auto callAttr = std::make_shared<CallOpAttribute>();
    callAttr->SetCalleeMagicName(leafFunc->GetMagicName());
    callAttr->wrapId = 10;
    callOp.SetOpAttribute(callAttr);

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}

TEST_F(TestMixInfo, DumpMixInfo_MultipleWrapIdAndMixId)
{
    std::vector<int64_t> shape = {32, 32};

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, shape, "output");

    std::string funcName = "TestMultipleWrapIdAndMixId";
    FUNCTION(funcName, {input, output}) { output = Add(input, input); }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    ASSERT_NE(function, nullptr);

    function->SetGraphType(GraphType::EXECUTE_GRAPH);

    for (int i = 0; i < 3; i++) {
        auto leafFuncName = "LeafFunc_" + std::to_string(i);
        auto leafFunc = std::make_shared<Function>(Program::GetInstance(), leafFuncName, leafFuncName, function);
        leafFunc->SetGraphType(GraphType::TENSOR_GRAPH);

        auto leafAttr = std::make_shared<LeafFuncAttribute>();
        leafAttr->mixId = 300 + i % 2;
        leafAttr->coreType = CoreType::AIV;
        leafFunc->SetLeafFuncAttribute(leafAttr);

        Program::GetInstance().InsertFuncToFunctionMap(leafFunc->GetMagicName(), leafFunc);

        auto funcInCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);
        auto funcOutCast = std::make_shared<LogicalTensor>(*function, DataType::DT_FP32, shape);

        auto& callOp = function->AddOperation(Opcode::OP_CALL, {funcInCast}, {funcOutCast});
        auto callAttr = std::make_shared<CallOpAttribute>();
        callAttr->SetCalleeMagicName(leafFunc->GetMagicName());
        callAttr->wrapId = i * 2;
        callOp.SetOpAttribute(callAttr);
    }

    int result = DumpMixInfo(function);
    EXPECT_EQ(result, 0);
}
