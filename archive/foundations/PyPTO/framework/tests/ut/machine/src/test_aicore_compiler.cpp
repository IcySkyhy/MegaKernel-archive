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
 * \file test_aicore_compiler.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <map>
#include <sstream>

#include "machine/compile/aicore_compiler.h"
#include "utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk {
std::string GenSubFuncCall(std::map<uint64_t, Function*>& leafDict, CoreType coreType,
                           dynamic::EncodeDevAscendFunctionParam& param, const std::string& ccePath, uint64_t tilingKey,
                           std::stringstream& src_obj, bool enableSubFunc = true);
}

namespace {
const std::string TEST_TMP_DIR = "/tmp/test_aicore_compiler";
}

class TestAicoreCompiler : public testing::Test {
public:
    void SetUp() override {}
    static void SetUpTestCase() { CreateDir(TEST_TMP_DIR); }

    void TearDown() override {}
    static void TearDownTestCase()
    {
        std::string cmd = "rm -rf " + TEST_TMP_DIR;
        [[maybe_unused]] int ret = system(cmd.c_str());
    }
};

TEST_F(TestAicoreCompiler, CompileAICoreKernel_EmptyCcePath)
{
    std::map<uint64_t, Function*> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::string kernelPath;

    EXPECT_THROW(CompileAICoreKernel(leafDict, param, "", "test_hash", "test_func", kernelPath), npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, CompileAICoreKernel_GenSrcFileFails)
{
    std::map<uint64_t, Function*> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::string kernelPath;
    EXPECT_THROW(
        CompileAICoreKernel(leafDict, param, "/nonexistent_dir/cce_path/", "test_hash", "test_func", kernelPath),
        npu::tile_fwk::Error);
}

TEST_F(TestAicoreCompiler, GenSubFuncCall_EmptyLeafDict)
{
    std::map<uint64_t, Function*> leafDict;
    dynamic::EncodeDevAscendFunctionParam param = {};
    std::stringstream src_obj;

    std::string result = GenSubFuncCall(leafDict, CoreType::AIC, param, TEST_TMP_DIR + "/", 0, src_obj);
    EXPECT_EQ(result, "");
}
