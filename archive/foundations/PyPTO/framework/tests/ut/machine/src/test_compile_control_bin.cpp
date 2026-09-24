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
 * \file test_compile_control_bin.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <nlohmann/json.hpp>

#include "machine/compile/compile_control_bin.h"
#include "utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "machine/utils/machine_utils.h"
#include "tilefwk/pypto_fwk_log.h"

using namespace npu::tile_fwk;
using Json = nlohmann::json;

namespace npu::tile_fwk {
void GenCustomOpInfo(const std::string& funcName, const std::string& controlAicpuPath,
                     const std::string& constrolSoName);
bool GenTilingFunc(const std::string& funcName, const std::string& controlAicpuPath);
bool TieFwkAicpuPreCompile(std::string& preCompileO, std::string& controlAicpuPath);
bool SharedAicpuCompile(const std::string& funcName, const std::string& aicpuDirPath, const std::string& preCompileO);
} // namespace npu::tile_fwk

class TestCompileControlBin : public testing::Test {
public:
    void SetUp() override
    {
        CreateDir(testDir_);
        setenv("LC_ALL", "C", 1);
        setenv("LANG", "C", 1);
    }
    void TearDown() override
    {
        std::string cmd = "rm -rf " + testDir_;
        [[maybe_unused]] int ret = system(cmd.c_str());
    }

public:
    std::string testDir_ = "/tmp/test_compile_control_bin_" + std::to_string(getpid());
};

TEST_F(TestCompileControlBin, GenCustomOpInfo_DumpFileFails)
{
    EXPECT_THROW(GenCustomOpInfo("test_func", "/nonexistent_dir/aicpu", "libtest_control"), npu::tile_fwk::Error);
}

TEST_F(TestCompileControlBin, GenCustomOpInfo_Success)
{
    std::string aicpuPath = testDir_ + "/gen_custom_op";
    CreateDir(aicpuPath);

    GenCustomOpInfo("test_func", aicpuPath, "libtest_control");

    std::string jsonFilePath = aicpuPath + "/libtest_control.json";
    std::ifstream ifs(jsonFilePath);
    EXPECT_TRUE(ifs.is_open());
    if (ifs.is_open()) {
        Json jsonValue;
        ifs >> jsonValue;
        ifs.close();
    }
}

TEST_F(TestCompileControlBin, GenTilingFunc_Success)
{
    std::string controlPath = testDir_ + "/gen_tiling_func";
    CreateDir(controlPath);

    bool result = GenTilingFunc("test_op", controlPath);
    EXPECT_TRUE(result);

    std::string cppFilePath = controlPath + "/control_flow_kernel.cpp";
    std::ifstream ifs(cppFilePath);
    EXPECT_TRUE(ifs.is_open());
    ifs.close();
}

TEST_F(TestCompileControlBin, GenTilingFunc_DumpFileFails)
{
    bool result = GenTilingFunc("test_op", "/nonexistent_dir/aicpu");
    EXPECT_FALSE(result);
}

TEST_F(TestCompileControlBin, TileFwkAiCpuCompile_GenTilingFuncFails)
{
    EXPECT_THROW(TileFwkAiCpuCompile("test_func", "/proc/fake_aicpu_dir"), npu::tile_fwk::Error);
}

TEST_F(TestCompileControlBin, TieFwkAicpuPreCompile_CompileFails)
{
    std::string compileDir = testDir_ + "/precompile_test/";
    CreateDir(compileDir);

    std::string cppFile = compileDir + "dummy.cpp";
    std::ofstream ofs(cppFile);
    ofs << "\"UT: force compile fail\"" << std::endl;
    ofs.close();

    std::string preCompileO;
    std::string dirPath = compileDir;
    EXPECT_THROW(TieFwkAicpuPreCompile(preCompileO, dirPath), npu::tile_fwk::Error);
}

TEST_F(TestCompileControlBin, TieFwkAicpuPreCompile_NoFiles)
{
    std::string emptyDir = testDir_ + "/empty_compile_dir/";
    CreateDir(emptyDir);

    std::string preCompileO;
    std::string dirPath = emptyDir;
    bool result = TieFwkAicpuPreCompile(preCompileO, dirPath);
    EXPECT_TRUE(result);
    EXPECT_TRUE(preCompileO.empty());
}

TEST_F(TestCompileControlBin, SharedAicpuCompile_CompileFails)
{
    std::string aicpuDir = testDir_ + "/shared_compile_test";
    CreateDir(aicpuDir);

    std::string preCompile0 = "/nonexistent_ut_fake.o";
    EXPECT_THROW(SharedAicpuCompile("test_func", aicpuDir, preCompile0), npu::tile_fwk::Error);
}
