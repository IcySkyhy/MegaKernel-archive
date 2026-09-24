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
 * \file test_pass_log.cpp
 * \brief Unit tests for pass log helpers.
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "interface/configs/config_manager.h"
#include "utils/file_utils.h"
#include "interface/program/program.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_log/pass_log.h"

namespace npu {
namespace tile_fwk {

class DummyPass : public Pass {
public:
    DummyPass() : Pass("UTDummyPass") {}

protected:
    Status RunOnFunction(Function& function) override
    {
        (void)function;
        return SUCCESS;
    }
};

class PassLogTest : public ::testing::Test {
public:
    void SetUp() override { Program::GetInstance().Reset(); }
};

static std::string MakeTmpDir(const char* pattern)
{
    std::array<char, 128> dirTemplate = {0};
    std::copy(pattern, pattern + std::strlen(pattern), dirTemplate.data());
    char* tmpDir = mkdtemp(dirTemplate.data());
    EXPECT_NE(tmpDir, nullptr);
    return tmpDir == nullptr ? std::string() : std::string(tmpDir);
}

TEST_F(PassLogTest, EscapeShellArgEscapesSingleQuote)
{
    const std::string input = "abc'def";
    const std::string expected = "'abc'\\''def'";
    EXPECT_EQ(EscapeShellArg(input), expected);
}

TEST_F(PassLogTest, ExtractPassLogByFunctionCoverSystemReturnBranches)
{
    char cwdBuffer[4096] = {0};
    ASSERT_NE(getcwd(cwdBuffer, sizeof(cwdBuffer)), nullptr);
    const std::string oldCwd(cwdBuffer);

    const std::string tmpDir = MakeTmpDir("/tmp/pypto_pass_log_ut_XXXXXX");
    ASSERT_FALSE(tmpDir.empty());
    ASSERT_TRUE(CreateDir(tmpDir + "/tools/scripts", true));
    ASSERT_TRUE(CreateDir(tmpDir + "/out", true));
    ASSERT_EQ(setenv("TILE_FWK_OUTPUT_DIR", (tmpDir + "/out").c_str(), 1), 0);
    ConfigManager::Instance().ResetLog(tmpDir + "/out");

    std::ofstream script(tmpDir + "/tools/scripts/extract_pass_log.py");
    ASSERT_TRUE(script.is_open());
    script << "import os, sys\n";
    script << "ret = int(os.environ.get('PYPTO_PASS_LOG_UT_RET', '0'))\n";
    script << "sys.exit(ret)\n";
    script.close();

    ASSERT_EQ(chdir(tmpDir.c_str()), 0);
    auto function = std::make_shared<Function>(Program::GetInstance(), "TENSOR_PassLog'UT", "PassLogUT", nullptr);

    ASSERT_EQ(setenv("PYPTO_PASS_LOG_UT_RET", "0", 1), 0);
    EXPECT_NO_THROW(ExtractPassLogByFunction(*function, "UTStrategy"));
    ASSERT_EQ(setenv("PYPTO_PASS_LOG_UT_RET", "1", 1), 0);
    EXPECT_NO_THROW(ExtractPassLogByFunction(*function, "UTStrategy"));

    EXPECT_EQ(chdir(oldCwd.c_str()), 0);
    EXPECT_EQ(unsetenv("PYPTO_PASS_LOG_UT_RET"), 0);
    EXPECT_EQ(unsetenv("TILE_FWK_OUTPUT_DIR"), 0);
    EXPECT_EQ(std::system((std::string("rm -rf ") + tmpDir).c_str()), 0);
}

TEST_F(PassLogTest, PassLogUtilDeleteEmptyFolderOnDestruct)
{
    const std::string tmpDir = MakeTmpDir("/tmp/pypto_pass_log_util_ut_XXXXXX");
    ASSERT_FALSE(tmpDir.empty());
    ASSERT_TRUE(CreateDir(tmpDir + "/out"));
    ASSERT_EQ(setenv("TILE_FWK_OUTPUT_DIR", (tmpDir + "/out").c_str(), 1), 0);
    ConfigManager::Instance().ResetLog(tmpDir + "/out");
    ASSERT_TRUE(CreateDir(config::LogTopFolder(), true));

    auto function = std::make_shared<Function>(Program::GetInstance(), "TENSOR_PassLogUtil", "PassLogUtil", nullptr);
    DummyPass pass;
    std::string expectedFolder;

    {
        PassLogUtil util(pass, *function, 9);
        (void)util;
        expectedFolder = pass.GetPassFolder();
    }

    EXPECT_NE(access(expectedFolder.c_str(), F_OK), 0);
    EXPECT_EQ(unsetenv("TILE_FWK_OUTPUT_DIR"), 0);
    EXPECT_EQ(std::system((std::string("rm -rf ") + tmpDir).c_str()), 0);
}

TEST_F(PassLogTest, PassLogUtilKeepFolderWhenNotEmpty)
{
    const std::string tmpDir = MakeTmpDir("/tmp/pypto_pass_log_util_keep_ut_XXXXXX");
    ASSERT_FALSE(tmpDir.empty());
    ASSERT_TRUE(CreateDir(tmpDir + "/out"));
    ASSERT_EQ(setenv("TILE_FWK_OUTPUT_DIR", (tmpDir + "/out").c_str(), 1), 0);
    ConfigManager::Instance().ResetLog(tmpDir + "/out");
    ASSERT_TRUE(CreateDir(config::LogTopFolder(), true));

    auto function = std::make_shared<Function>(Program::GetInstance(), "TENSOR_PassLogKeep", "PassLogKeep", nullptr);
    DummyPass pass;
    std::string expectedFolder;

    {
        PassLogUtil util(pass, *function, 10);
        (void)util;
        expectedFolder = pass.GetPassFolder();
        std::ofstream keepFile(expectedFolder + "/keep.txt");
        ASSERT_TRUE(keepFile.is_open());
        keepFile << "keep";
    }

    EXPECT_EQ(access(expectedFolder.c_str(), F_OK), 0);
    EXPECT_EQ(unsetenv("TILE_FWK_OUTPUT_DIR"), 0);
    EXPECT_EQ(std::system((std::string("rm -rf ") + tmpDir).c_str()), 0);
}

TEST_F(PassLogTest, PassLogUtilCreatesStrategyScopedFolder)
{
    const std::string tmpDir = MakeTmpDir("/tmp/pypto_pass_log_strategy_ut_XXXXXX");
    ASSERT_FALSE(tmpDir.empty());
    ASSERT_TRUE(CreateDir(tmpDir + "/out", true));
    ASSERT_EQ(setenv("TILE_FWK_OUTPUT_DIR", (tmpDir + "/out").c_str(), 1), 0);
    ConfigManager::Instance().ResetLog(tmpDir + "/out");

    auto function = std::make_shared<Function>(Program::GetInstance(), "TENSOR_PassLogStrategy", "PassLogStrategy",
                                               nullptr);
    DummyPass pass;
    const std::string expectedFolder = config::LogTopFolder() +
                                       "/computation_graph/Strategy_00_UTStrategy/Pass_11_UTDummyPass";

    {
        PassLogUtil util(pass, *function, "UTStrategy", 11);
        (void)util;
        std::ofstream keepFile(expectedFolder + "/keep.txt");
        ASSERT_TRUE(keepFile.is_open());
        keepFile << "keep";
    }

    EXPECT_EQ(access(expectedFolder.c_str(), F_OK), 0);
    EXPECT_EQ(unsetenv("TILE_FWK_OUTPUT_DIR"), 0);
    EXPECT_EQ(std::system((std::string("rm -rf ") + tmpDir).c_str()), 0);
}

TEST_F(PassLogTest, PassLogUtilDeletesEmptyStrategyScopedParents)
{
    const std::string tmpDir = MakeTmpDir("/tmp/pypto_pass_log_empty_strategy_ut_XXXXXX");
    ASSERT_FALSE(tmpDir.empty());
    ASSERT_TRUE(CreateDir(tmpDir + "/out", true));
    ASSERT_EQ(setenv("TILE_FWK_OUTPUT_DIR", (tmpDir + "/out").c_str(), 1), 0);
    ConfigManager::Instance().ResetLog(tmpDir + "/out");

    auto function = std::make_shared<Function>(Program::GetInstance(), "TENSOR_PassLogEmptyStrategy",
                                               "PassLogEmptyStrategy", nullptr);
    DummyPass pass;
    const std::string computationGraphFolder = config::LogTopFolder() + "/computation_graph";
    const std::string strategyFolder = computationGraphFolder + "/Strategy_00_UTEmptyStrategy";
    const std::string passFolder = strategyFolder + "/Pass_12_UTDummyPass";

    {
        PassLogUtil util(pass, *function, "UTEmptyStrategy", 12);
        (void)util;
    }

    EXPECT_NE(access(passFolder.c_str(), F_OK), 0);
    EXPECT_NE(access(strategyFolder.c_str(), F_OK), 0);
    EXPECT_EQ(unsetenv("TILE_FWK_OUTPUT_DIR"), 0);
    EXPECT_EQ(std::system((std::string("rm -rf ") + tmpDir).c_str()), 0);
}

TEST_F(PassLogTest, PassRunDeletesEmptyFunctionGraphFolder)
{
    const std::string tmpDir = MakeTmpDir("/tmp/pypto_pass_log_empty_graph_ut_XXXXXX");
    ASSERT_FALSE(tmpDir.empty());
    ASSERT_TRUE(CreateDir(tmpDir + "/out"));
    ASSERT_EQ(setenv("TILE_FWK_OUTPUT_DIR", (tmpDir + "/out").c_str(), 1), 0);
    ConfigManager::Instance().ResetLog(tmpDir + "/out");

    auto function = std::make_shared<Function>(Program::GetInstance(), "TENSOR_PassLogEmptyGraph", "PassLogEmptyGraph",
                                               nullptr);
    DummyPass pass;
    PassConfigs cfg;
    cfg.dumpGraph = true;
    pass.SetPassConfigs(cfg);

    const std::string graphFolder = config::LogTopFolder() + "/computation_graph/Strategy_00_UTEmptyGraph/" +
                                    function->GetMagicName();

    {
        PassLogUtil util(pass, *function, "UTEmptyGraph", 0);
        EXPECT_EQ(pass.Run(*function, "UTEmptyGraph", "UTDummyPass", 0), SUCCESS);
        EXPECT_NE(access(graphFolder.c_str(), F_OK), 0);
    }

    EXPECT_EQ(unsetenv("TILE_FWK_OUTPUT_DIR"), 0);
    EXPECT_EQ(std::system((std::string("rm -rf ") + tmpDir).c_str()), 0);
}

} // namespace tile_fwk
} // namespace npu
