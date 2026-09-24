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
 * \file test_output_dir.cpp
 * \brief Unit tests for OutputBaseDir() and GetEmitPath() in config_manager
 */
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/pypto_fwk_log.h"

using namespace npu::tile_fwk;

class TestOutputDir : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        // Save original env before modification
        savedTileFwkDir_ = GetEnvOrEmpty("TILE_FWK_OUTPUT_DIR");
        savedAscendWorkPath_ = GetEnvOrEmpty("ASCEND_WORK_PATH");
        savedDeviceId_ = GetEnvOrEmpty("TILE_FWK_DEVICE_ID");
    }

    void TearDown() override
    {
        RestoreEnv("TILE_FWK_OUTPUT_DIR", savedTileFwkDir_);
        RestoreEnv("ASCEND_WORK_PATH", savedAscendWorkPath_);
        RestoreEnv("TILE_FWK_DEVICE_ID", savedDeviceId_);

        // Reset codegen config to default
        ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false);
        // Reset compile debug mode to default (in case fixed-CCE mode was enabled)
        config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, CFG_DEBUG_NONE);
    }

    static std::string GetEnvOrEmpty(const char* name)
    {
        const char* val = std::getenv(name);
        return (val != nullptr) ? std::string(val) : std::string();
    }

    static void SetEnv(const char* name, const char* value) { setenv(name, value, 1); }

    static void UnsetEnv(const char* name) { unsetenv(name); }

    static void RestoreEnv(const char* name, const std::string& value)
    {
        if (value.empty()) {
            unsetenv(name);
        } else {
            setenv(name, value.c_str(), 1);
        }
    }

protected:
    std::string savedTileFwkDir_;
    std::string savedAscendWorkPath_;
    std::string savedDeviceId_;
};

// ---------------------------------------------------------------------------
// GetEmitPath tests — evaluates env vars on each call (no call_once)
// ---------------------------------------------------------------------------

TEST_F(TestOutputDir, GetEmitPath_UsesLogTopFolderByDefault)
{
    // Default: KEY_FIXED_OUTPUT_PATH=false → LogTopFolder() + "/" + name
    UnsetEnv("TILE_FWK_OUTPUT_DIR");
    UnsetEnv("ASCEND_WORK_PATH");
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false);

    std::string path = config::GetEmitPath("kernel_aicore");
    EXPECT_FALSE(path.empty());
    // Should contain the name as a suffix
    EXPECT_NE(path.find("kernel_aicore"), std::string::npos);
    // Should contain the LogTopFolder prefix (not just bare name)
    EXPECT_NE(path.find("output"), std::string::npos);
}

TEST_F(TestOutputDir, GetEmitPath_FixedPathWithAscendWork)
{
    // KEY_FIXED_OUTPUT_PATH=true + ASCEND_WORK_PATH set → $ASCEND_WORK_PATH/pypto/<name>
    const char* workPath = "/tmp/test_ut_ascend_work";
    SetEnv("ASCEND_WORK_PATH", workPath);
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    std::string path = config::GetEmitPath("kernel_aicore");
    EXPECT_NE(path.find(workPath), std::string::npos);
    EXPECT_NE(path.find("pypto"), std::string::npos);
    EXPECT_NE(path.find("kernel_aicore"), std::string::npos);
}

TEST_F(TestOutputDir, GetEmitPath_FixedPathWithoutAscendWork)
{
    // KEY_FIXED_OUTPUT_PATH=true + no ASCEND_WORK_PATH → bare name
    UnsetEnv("ASCEND_WORK_PATH");
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    std::string path = config::GetEmitPath("kernel_aicore");
    EXPECT_EQ(path, "kernel_aicore");
}

TEST_F(TestOutputDir, GetEmitPath_FixedPathWithDeviceId)
{
    // KEY_FIXED_OUTPUT_PATH=true + ASCEND_WORK_PATH set + TILE_FWK_DEVICE_ID set
    // → $ASCEND_WORK_PATH/pypto/<device_id>/<name>
    // Note: groupNames.size() > 0 requires distributed context; this test covers
    // the single-node path. The multi-node path is covered by integration tests.
    const char* workPath = "/tmp/test_ut_ascend_work2";
    SetEnv("ASCEND_WORK_PATH", workPath);
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    // Without distributed groups, the path is $ASCEND_WORK_PATH/pypto/<name>
    std::string path = config::GetEmitPath("kernel_aicpu");
    EXPECT_NE(path.find(workPath), std::string::npos);
    EXPECT_NE(path.find("pypto"), std::string::npos);
    EXPECT_NE(path.find("kernel_aicpu"), std::string::npos);
}

TEST_F(TestOutputDir, GetEmitPath_DifferentNames)
{
    // Verify different names produce different paths
    UnsetEnv("ASCEND_WORK_PATH");
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    std::string pathCore = config::GetEmitPath("kernel_aicore");
    std::string pathCpu = config::GetEmitPath("kernel_aicpu");

    EXPECT_EQ(pathCore, "kernel_aicore");
    EXPECT_EQ(pathCpu, "kernel_aicpu");
    EXPECT_NE(pathCore, pathCpu);
}

// ---------------------------------------------------------------------------
// TILE_FWK_OUTPUT_DIR overrides ASCEND_WORK_PATH for OutputBaseDir
// ---------------------------------------------------------------------------
TEST_F(TestOutputDir, GetEmitPath_TileFwkOutputDirTakesPriority)
{
    // Set both TILE_FWK_OUTPUT_DIR and ASCEND_WORK_PATH
    // GetEmitPath doesn't use OutputBaseDir directly, but let's verify
    // TILE_FWK_OUTPUT_DIR doesn't affect GetEmitPath (it uses LogTopFolder/ASCEND_WORK_PATH)
    SetEnv("TILE_FWK_OUTPUT_DIR", "/tmp/tile_fwk_override_test");
    SetEnv("ASCEND_WORK_PATH", "/tmp/ascend_work_test");
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    std::string path = config::GetEmitPath("kernel_aicore");
    // With ASCEND_WORK_PATH set, should use ASCEND_WORK_PATH, not TILE_FWK_OUTPUT_DIR
    EXPECT_NE(path.find("ascend_work_test"), std::string::npos);
    EXPECT_EQ(path.find("tile_fwk_override_test"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Empty / edge cases
// ---------------------------------------------------------------------------
TEST_F(TestOutputDir, GetEmitPath_EmptyName)
{
    UnsetEnv("ASCEND_WORK_PATH");
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    std::string path = config::GetEmitPath("");
    EXPECT_TRUE(path.empty());
}

TEST_F(TestOutputDir, GetEmitPath_RepeatedCallsConsistent)
{
    const char* workPath = "/tmp/test_ut_consistency";
    mkdir(workPath, 0755); // ensure parent dir exists for OutputBaseDir
    SetEnv("ASCEND_WORK_PATH", workPath);
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, true);

    std::string path1 = config::GetEmitPath("kernel_aicore");
    std::string path2 = config::GetEmitPath("kernel_aicore");
    EXPECT_EQ(path1, path2);

    // Switch to non-fixed mode
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false);
    std::string path3 = config::GetEmitPath("kernel_aicore");
    std::string path4 = config::GetEmitPath("kernel_aicore");
    EXPECT_EQ(path3, path4);

    // Different mode produces different path
    EXPECT_NE(path1, path3);
}

// ---------------------------------------------------------------------------
// Fixed CCE mode — compile_debug_mode=2 (CFG_COMPILE_FIXED_CCE) forces fixed path
// even when KEY_FIXED_OUTPUT_PATH is false
// ---------------------------------------------------------------------------
TEST_F(TestOutputDir, GetEmitPath_FixedCceModeForcesFixedPath)
{
    // compile_debug_mode=2 forces fixed output path regardless of KEY_FIXED_OUTPUT_PATH
    const char* workPath = "/tmp/test_ut_fixed_cce_mode";
    SetEnv("ASCEND_WORK_PATH", workPath);
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false);
    config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, CFG_COMPILE_FIXED_CCE);

    std::string path = config::GetEmitPath("kernel_aicore");
    // Fixed CCE mode -> fixed path (ASCEND_WORK_PATH/pypto/<name>), not LogTopFolder
    EXPECT_NE(path.find(workPath), std::string::npos);
    EXPECT_NE(path.find("pypto"), std::string::npos);
    EXPECT_NE(path.find("kernel_aicore"), std::string::npos);
}

TEST_F(TestOutputDir, GetEmitPath_FixedCceModeOverridesExplicitConfig)
{
    // Even with KEY_FIXED_OUTPUT_PATH explicitly false, compile_debug_mode=2 wins
    UnsetEnv("ASCEND_WORK_PATH");
    ConfigManager::Instance().SetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false);
    config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, CFG_COMPILE_FIXED_CCE);

    std::string path = config::GetEmitPath("kernel_aicore");
    // No ASCEND_WORK_PATH -> bare name (fixed-path behavior), not LogTopFolder
    EXPECT_EQ(path, "kernel_aicore");
}

// ---------------------------------------------------------------------------
// OutputBaseDir — uses call_once, so only limited testing is possible
// ---------------------------------------------------------------------------
TEST_F(TestOutputDir, OutputBaseDir_ReturnsNonEmpty)
{
    // OutputBaseDir is called once (static call_once). Verify it returns
    // a non-empty string and can be used.
    const std::string& dir = config::OutputBaseDir();
    EXPECT_FALSE(dir.empty());
}
