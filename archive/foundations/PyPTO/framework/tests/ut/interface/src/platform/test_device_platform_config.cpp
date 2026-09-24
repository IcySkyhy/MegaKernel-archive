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
 * \file test_device_platform_config.cpp
 * \brief Test that device_platform config controls GetMemoryLimit data source
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <nlohmann/json.hpp>
#define private public
#include "platform/parser/platform_parser.h"
#include "platform/parser/simulation_platform/simulation_platform.h"

using namespace npu::tile_fwk;

const std::string version = "version";
const std::string aiCoreSpec = "AICoreSpec";
const std::string ubSize = "ub_size";
const std::string l0cSize = "l0_c_size";
const std::string l1Size = "l1_size";
const std::string l0aSize = "l0_a_size";
const std::string socVersionInfo = "SoC_version";
const std::string npuArchInfo = "NpuArch";

class TestDevicePlatformConfig : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { CleanupTempConfig(); }

private:
    void CleanupTempConfig()
    {
        std::string libPath = GetPyptoLibPath();
        if (!libPath.empty()) {
            std::string configPath = libPath + "/tile_fwk_config.json";
            std::remove(configPath.c_str());
        }
    }
};

// ==================== StringToDpaltform Tests ====================

TEST_F(TestDevicePlatformConfig, StringToDpaltformAllMappings)
{
    EXPECT_EQ(StringToDpaltform("ASCEND_910B1"), DPlatform::ASCEND_910B1);
    EXPECT_EQ(StringToDpaltform("ASCEND_910B2"), DPlatform::ASCEND_910B2);
    EXPECT_EQ(StringToDpaltform("ASCEND_910B3"), DPlatform::ASCEND_910B3);
    EXPECT_EQ(StringToDpaltform("ASCEND_910B4"), DPlatform::ASCEND_910B4);
    EXPECT_EQ(StringToDpaltform("ASCEND_910C4"), DPlatform::ASCEND_910C4);
    EXPECT_EQ(StringToDpaltform("ASCEND_950PR_9579"), DPlatform::ASCEND_950PR_9579);
    EXPECT_EQ(StringToDpaltform("ASCEND_950DT_9581"), DPlatform::ASCEND_950DT_9581);
    EXPECT_EQ(StringToDpaltform("ASCEND_950DT_9581X"), DPlatform::ASCEND_950DT_9581X);
    EXPECT_EQ(StringToDpaltform("ASCEND_950DT_9582"), DPlatform::ASCEND_950DT_9582);
    EXPECT_EQ(StringToDpaltform("ASCEND_950PR_9582"), DPlatform::ASCEND_950PR_9582);
    EXPECT_EQ(StringToDpaltform("ASCEND_950DT_9579"), DPlatform::ASCEND_950DT_9579);
    EXPECT_EQ(StringToDpaltform("KIRIN_9030"), DPlatform::KIRIN_9030);
}

TEST_F(TestDevicePlatformConfig, StringToDpaltformUnknown)
{
    EXPECT_EQ(StringToDpaltform("INVALID_PLATFORM"), DPlatform::UNKNOWN_DEVICE);
    EXPECT_EQ(StringToDpaltform(""), DPlatform::UNKNOWN_DEVICE);
    EXPECT_EQ(StringToDpaltform("ascend_910b2"), DPlatform::UNKNOWN_DEVICE);
}

TEST_F(TestDevicePlatformConfig, ConfigGetDevicePlatformChain)
{
    config::SetPlatformConfig("device_platform", std::string("ASCEND_950PR_9579"));
    DPlatform platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950PR_9579);

    config::SetPlatformConfig("device_platform", std::string("ASCEND_950DT_9581"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9581);

    config::SetPlatformConfig("device_platform", std::string("ASCEND_950DT_9581X"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9581X);

    config::SetPlatformConfig("device_platform", std::string("ASCEND_910B2"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_910B2);

    config::SetPlatformConfig("device_platform", std::string("KIRIN_9030"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::KIRIN_9030);
}

TEST_F(TestDevicePlatformConfig, ConfigGetDevicePlatformUnknown)
{
    config::SetPlatformConfig("device_platform", std::string("INVALID_PLATFORM"));
    DPlatform platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::UNKNOWN_DEVICE);
}

// ==================== SimulationPlatform::GetDevicePlatform Tests ====================

TEST_F(TestDevicePlatformConfig, GetDevicePlatformConfigFileNotFound)
{
    std::string result = SimulationPlatform::GetDevicePlatform();
    EXPECT_EQ(result, "A2A3");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsWithConfigSocVersion)
{
    config::SetPlatformConfig("device_platform", std::string("ASCEND_950PR_9579"));
    DPlatform platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950PR_9579);

    INIParser parser("950PR_957x");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew950DT9581Config)
{
    INIParser parser("950DT_9581");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend950DT_9581");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew950DT9581xConfig)
{
    INIParser parser("950DT_9581x");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend950DT_9581X");
}

TEST_F(TestDevicePlatformConfig, DifferentPlatformDifferentMemory)
{
    INIParser parserA2A3("A2A3");
    INIParser parser950PR("950PR_957x");
    INIParser parserKirin("Kirin9030");

    size_t ubA2A3, ub950PR, ubKirin;
    EXPECT_TRUE(parserA2A3.GetSizeVal(aiCoreSpec, ubSize, ubA2A3));
    EXPECT_TRUE(parser950PR.GetSizeVal(aiCoreSpec, ubSize, ub950PR));
    EXPECT_TRUE(parserKirin.GetSizeVal(aiCoreSpec, ubSize, ubKirin));

    EXPECT_EQ(ubA2A3, 196608UL);
    EXPECT_EQ(ub950PR, 253952UL);
    EXPECT_EQ(ubKirin, 131072UL);

    EXPECT_NE(ubA2A3, ub950PR);
    EXPECT_NE(ubA2A3, ubKirin);

    size_t l0cA2A3, l0c950PR, l0cKirin;
    EXPECT_TRUE(parserA2A3.GetSizeVal(aiCoreSpec, l0cSize, l0cA2A3));
    EXPECT_TRUE(parser950PR.GetSizeVal(aiCoreSpec, l0cSize, l0c950PR));
    EXPECT_TRUE(parserKirin.GetSizeVal(aiCoreSpec, l0cSize, l0cKirin));

    EXPECT_EQ(l0cA2A3, 131072UL);
    EXPECT_EQ(l0c950PR, 262144UL);
    EXPECT_EQ(l0cKirin, 65536UL);
}
