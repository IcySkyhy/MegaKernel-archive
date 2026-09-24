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
 * \file test_codegen_common.h
 * \brief Unit test for codegen.
 */

#ifndef TEST_CODEGEN_COMMON_H
#define TEST_CODEGEN_COMMON_H

#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/id_gen.h"

namespace npu::tile_fwk {
const std::string SUB_FUNC_SUFFIX = "_Unroll1_PATH0";
const std::string HIDDEN_FUNC_SUFFIX = "_hiddenfunc0";
const constexpr int DummyFuncMagic = 1;

struct CodegenTestConfig {
    bool enableCostModel = false;
    int64_t compileStage = 0;
    bool buildStatic = false;
    bool setTileTensor = false;
    bool tileTensorValue = false;
    bool setIdGen = false;
    bool resetTileTensorOnTearDown = false;
    bool setSocVersion = false;
    std::string socVersionValue = "";
    bool resetSocVersionOnTearDown = false;
};

class CodegenTestBase : public ::testing::Test {
public:
    explicit CodegenTestBase(CodegenTestConfig config = {}) : config_(config) {}

    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, config_.enableCostModel);
        if (config_.compileStage != 0) {
            config::SetHostOption(COMPILE_STAGE, config_.compileStage);
        }
        if (config_.buildStatic) {
            config::SetBuildStatic(true);
        }
        if (config_.setTileTensor) {
            config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, config_.tileTensorValue);
        }
        if (config_.setIdGen) {
            IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        }
        if (config_.setSocVersion) {
            config::SetCodeGenOption<std::string>(PLATFORM_SOC_VERSION, config_.socVersionValue);
        }
    }

    void TearDown() override
    {
        if (config_.resetTileTensorOnTearDown) {
            config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        }
        if (config_.resetSocVersionOnTearDown) {
            config::SetCodeGenOption<std::string>(PLATFORM_SOC_VERSION, "");
        }
    }

protected:
    CodegenTestConfig config_;
};

class CodegenTestLiteNPU : public ::testing::Test {
public:
    CodegenTestLiteNPU(std::string socVersion) { socVersion_ = socVersion; }

    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenOption<std::string>(PLATFORM_SOC_VERSION, socVersion_);
    }

    void TearDown() override { config::SetCodeGenOption<std::string>(PLATFORM_SOC_VERSION, ""); }

private:
    std::string socVersion_;
};

} // namespace npu::tile_fwk

#endif // TEST_CODEGEN_COMMON_H
