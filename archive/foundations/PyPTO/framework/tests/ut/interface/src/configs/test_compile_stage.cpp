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
 * \file test_config_runmode.cpp
 * \brief
 */
#include <climits>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class TestConfigRunmode : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override { Program::GetInstance().Reset(); }
    void TearDown() override { Program::GetInstance().Reset(); }
};

TEST_F(TestConfigRunmode, COMPILE_STAGE_TENSOR_GRAPH)
{
    const std::vector<int64_t> shape = {4, 4};
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");
    config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
    FUNCTION("ADD", {inputA, inputB, output}) { output = Add(inputA, inputB); }
}

TEST_F(TestConfigRunmode, COMPILE_STAGE_TILE_GRAPH)
{
    const std::vector<int64_t> shape = {4, 4};
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");
    config::SetHostOption(COMPILE_STAGE, CS_TILE_GRAPH);
    FUNCTION("ADD", {inputA, inputB, output}) { output = Add(inputA, inputB); }
}

TEST_F(TestConfigRunmode, COMPILE_STAGE_EXECUTION_GRAPH)
{
    const std::vector<int64_t> shape = {4, 4};
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    FUNCTION("ADD", {inputA, inputB, output}) { output = Add(inputA, inputB); }
}

TEST_F(TestConfigRunmode, COMPILE_STAGE_CODEGEN_INSTRUCTION)
{
    const std::vector<int64_t> shape = {4, 4};
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    FUNCTION("ADD", {inputA, inputB, output}) { output = Add(inputA, inputB); }
}

TEST_F(TestConfigRunmode, COMPILE_STAGE_CODEGEN_BINARY)
{
    const std::vector<int64_t> shape = {4, 4};
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_BINARY);
    FUNCTION("ADD", {inputA, inputB, output}) { output = Add(inputA, inputB); }
}

TEST_F(TestConfigRunmode, COMPILE_VF)
{
    config::SetPassGlobalConfig(KEY_ENABLE_VF, true);
    std::ostringstream oss;
    CodeGenCloudNPU::AppendVFOptions(oss, NPUArch::DAV_3510, false);
    EXPECT_EQ(oss.str().size() > 0, true);
    const std::vector<int64_t> shape = {4, 4};
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "D");
    Tensor inputB(DT_FP32, shape, "E");
    Tensor output(DT_FP32, shape, "F");
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    FUNCTION("ADD", {inputA, inputB, output}) { output = Add(inputA, inputB); }
}
