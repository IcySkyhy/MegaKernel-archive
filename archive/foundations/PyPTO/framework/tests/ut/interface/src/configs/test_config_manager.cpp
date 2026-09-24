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
 * \file test_config_manager.cpp
 * \brief
 */
#include <climits>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.cpp"
#include "tilefwk/error_manager.h"

using namespace npu::tile_fwk;

class TestConfigManager : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { ErrorManager::Instance().OutputErrorMessage(true); }
};

TEST_F(TestConfigManager, PassGloablConfig)
{
    {
        auto ret = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 0);
        EXPECT_EQ(ret, 1);
        config::SetPassGlobalConfig(KEY_PASS_THREAD_NUM, 0);
        ret = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 1);
        EXPECT_EQ(ret, 0);
    }

    {
        auto ret = config::GetPassGlobalConfig(KEY_ENABLE_CV_FUSE, true);
        EXPECT_EQ(ret, false);
        config::SetPassGlobalConfig(KEY_ENABLE_CV_FUSE, true);
        ret = config::GetPassGlobalConfig(KEY_ENABLE_CV_FUSE, false);
        EXPECT_EQ(ret, true);
    }
}

TEST_F(TestConfigManager, PassDefaultConfig)
{
    auto ret = config::GetPassDefaultConfig(KEY_PRINT_GRAPH, true);
    EXPECT_EQ(ret, false);
    config::SetPassDefaultConfig(KEY_PRINT_GRAPH, true);
    ret = config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false);
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, PassStrategies2)
{
    {
        auto ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.dumpGraph, false);

        // set default config useful
        config::SetPassDefaultConfig(npu::tile_fwk::KEY_DUMP_GRAPH, true);
        ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.dumpGraph, true);
    }
}

TEST_F(TestConfigManager, PassStrategies3)
{
    // set default config useless
    auto ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
    EXPECT_EQ(ret.expectedValueCheck, false);

    config::SetPassDefaultConfig(KEY_EXPECTED_VALUE_CHECK, true);
    ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
    EXPECT_EQ(ret.expectedValueCheck, true);
}

TEST_F(TestConfigManager, Dump)
{
    auto& cm = ConfigManagerNg::GetInstance();

    cm.BeginScope("scope1", {{"pass.pg_lower_bound", 10L}});
    auto scope1 = cm.CurrentScope();
    cm.EndScope();

    cm.BeginScope("scope2", {{"pass.pg_lower_bound", 20L}});
    {
        auto scope2 = cm.CurrentScope();
        auto lower = AnyCast<int64_t>(scope2->GetAnyConfig("pass.pg_lower_bound"));
        EXPECT_EQ(lower, 20);
    }

    auto scope = cm.CurrentScope();
    auto lower = AnyCast<int64_t>(scope->GetAnyConfig("pass.pg_lower_bound"));
    EXPECT_EQ(lower, 20);
    cm.EndScope();

    cm.BeginScope("scope3", {{"pass.pg_lower_bound", 30L}});
    auto scope3 = cm.CurrentScope();
    cm.SetScope({{"pass.pg_lower_bound", 35L}});
    auto scope4 = cm.CurrentScope();
    cm.EndScope();

    std::cout << cm.GetOptionsTree() << std::endl;
    std::cout << "-- scope3 -- " << std::endl;
    std::cout << scope3->ToString() << std::endl;
}

constexpr const char* ERROR_KEY_WORD = "its value doesn't within the value range";
template <typename T>
bool RangeTest(const std::unordered_map<std::string, std::vector<T>>& input,
               void (*SetFunc)(const std::string&, const T&), std::string group)
{
    for (auto& [key, val] : input) {
        for (auto it : val) {
            T rlv = it;
            try {
                SetFunc(group + "." + key, std::move(rlv));
            } catch (const std::exception& e) {
                std::stringstream ss;
                ss << e.what();
                std::string errStr(ss.str());
                if (errStr.find(ERROR_KEY_WORD) == std::string::npos) {
                    std::cerr << "error exception: " << errStr << std::endl;
                    return false;
                } else {
                    continue;
                }
            }
        }
    }
    return true;
}

TEST_F(TestConfigManager, NormalRuntimeTest)
{
    std::unordered_map<std::string, std::vector<int64_t>> input = {{DEVICE_SCHED_MODE, {0, 1, 2, 3}},
                                                                   {STITCH_FUNCTION_MAX_NUM, {1, 1024}},
                                                                   {CFG_RUN_MODE, {0, 1}},
                                                                   {CFG_VALID_SHAPE_OPTIMIZE, {0, 1}},
                                                                   {DEVICE_SCHED_PARALLELISM, {1, 8}}};
    bool ret = RangeTest<int64_t>(input, &(config::SetOptionsNg), "runtime");
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, AbnormalRuntimeTest)
{
    std::unordered_map<std::string, std::vector<int64_t>> input = {{DEVICE_SCHED_MODE, {-1, 4}},
                                                                   {STITCH_FUNCTION_MAX_NUM, {0, 1025}},
                                                                   {CFG_RUN_MODE, {-1, 2}},
                                                                   {CFG_VALID_SHAPE_OPTIMIZE, {-1, 2}},
                                                                   {DEVICE_SCHED_PARALLELISM, {0, 9}}};
    bool ret = RangeTest<int64_t>(input, &(config::SetOptionsNg), "runtime");
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, NormalPassTest)
{
    std::unordered_map<std::string, std::vector<int64_t>> input = {{SG_PARALLEL_NUM, {0, INT_MAX}},
                                                                   {SG_PG_LOWER_BOUND, {0, INT_MAX}},
                                                                   {MG_COPYIN_UPPER_BOUND, {0, INT_MAX}},
                                                                   {MG_VEC_PARALLEL_LB, {1, 48}},
                                                                   {COPYOUT_RESOLVE_COALESCING, {0, 1000000}}};
    bool ret = RangeTest<int64_t>(input, &(config::SetOptionsNg), "pass");
    EXPECT_EQ(ret, true);

    std::unordered_map<std::string, std::vector<std::map<int64_t, int64_t>>> input2 = {
        {CUBE_L1_REUSE_SETTING, {{{-1, 0}}, {{INT_MAX, INT_MAX}}}},
        {CUBE_NBUFFER_SETTING, {{{-1, 1}}, {{INT_MAX, INT_MAX}}}},
        {VEC_NBUFFER_SETTING, {{{-1, 1}}, {{INT_MAX, INT_MAX}}}}};
    ret = RangeTest<std::map<int64_t, int64_t>>(input2, &(config::SetOptionsNg), "pass");
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, AbnormalPassTest)
{
    int64_t outVal = INT_MAX;
    ++outVal;
    std::unordered_map<std::string, std::vector<int64_t>> input = {{SG_PARALLEL_NUM, {-1, outVal}},
                                                                   {SG_PG_LOWER_BOUND, {-1, outVal}},
                                                                   {MG_COPYIN_UPPER_BOUND, {-1, outVal}},
                                                                   {MG_VEC_PARALLEL_LB, {0, 49}},
                                                                   {COPYOUT_RESOLVE_COALESCING, {-1, 1000001}}};
    bool ret = RangeTest<int64_t>(input, &(config::SetOptionsNg), "pass");
    EXPECT_EQ(ret, true);

    std::unordered_map<std::string, std::vector<std::map<int64_t, int64_t>>> input2 = {
        {CUBE_L1_REUSE_SETTING, {{{-2, 0}}, {{outVal, INT_MAX}}, {{-1, -1}}, {{INT_MAX, outVal}}}},
        {CUBE_NBUFFER_SETTING, {{{-2, 1}}, {{INT_MAX, outVal}}, {{-1, 0}}, {{outVal, INT_MAX}}}},
        {VEC_NBUFFER_SETTING, {{{-2, 1}}, {{INT_MAX, outVal}}, {{-1, 0}}, {{outVal, INT_MAX}}}}};
    ret = RangeTest<std::map<int64_t, int64_t>>(input2, &(config::SetOptionsNg), "pass");
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, GlobalConfig)
{
    std::string res = ConfigManagerNg::GetGlobalConfig<std::string>("platform.device_platform");
    EXPECT_EQ(res, "ASCEND_910B2");

    ConfigManagerNg::SetGlobalConfig("platform.device_platform", "test");
    res = ConfigManagerNg::GetGlobalConfig<std::string>("platform.device_platform");
    EXPECT_EQ(res, "test");

    ConfigManagerNg::SetGlobalConfig("simulation.timeout_threshold", 10);
    long res_int = ConfigManagerNg::GetGlobalConfig<long>("simulation.timeout_threshold");
    EXPECT_EQ(res_int, 10);

    ConfigManagerNg::SetGlobalConfig("codegen.codegen_support_tile_tensor", true);
    bool res_bool = ConfigManagerNg::GetGlobalConfig<bool>("codegen.codegen_support_tile_tensor");
    EXPECT_EQ(res_bool, true);

    // // add code for coverage, python pybind interface
    std::map<std::string, std::any> config_values = {{"simulation.timeout_threshold", 10}};
    ConfigManagerNg::GetInstance().SetGlobalConfig(std::move(config_values), "default", 1);
    ConfigManagerNg::GetInstance().GlobalScope();

    std::map<std::string, std::any> empty_values = {};
    ConfigManagerNg::GetInstance().SetGlobalConfig(std::move(empty_values), "default", 1);

    PrintOptions p = config::GetPrintOptions();
}

TEST_F(TestConfigManager, LoadJson)
{
    nlohmann::json jdata = {
        {"test_label", "field"},
    };
    TypeInfo test;
    test.build_type_infos(jdata, "");
    EXPECT_EQ(test.typeInfos.size(), 0);
    jdata = {{
        "type",
        "none",
    }};
    test.build_type_infos(jdata, "");
    EXPECT_EQ(test.typeInfos.size(), 0);
}

TEST_F(TestConfigManager, JitScopeGuardBasic)
{
    auto& cm = ConfigManagerNg::GetInstance();
    auto scopeBefore = cm.CurrentScope();
    {
        ConfigManagerNg::JitScopeGuard guard("jit_scope", std::map<std::string, std::any>{});
        auto scopeInGuard = cm.CurrentScope();
        EXPECT_NE(scopeInGuard.get(), scopeBefore.get());
        EXPECT_TRUE(scopeInGuard->HasConfig("pass.pg_lower_bound"));
    }
    auto scopeAfter = cm.CurrentScope();
    EXPECT_EQ(scopeAfter.get(), scopeBefore.get());
}

TEST_F(TestConfigManager, IsWithinRangeInvalidKey)
{
    auto& cm = ConfigManagerNg::GetInstance();
    auto scope = cm.CurrentScope();
    std::any value = int64_t(100);
    scope->UpdateValueWithAny("invalid.key.not.in.scope", int64_t(100));
    EXPECT_EQ(cm.IsWithinRange("invalid.key.not.in.schema", value), false);
}

TEST_F(TestConfigManager, InvalidValue)
{
    auto& cm = ConfigManagerNg::GetInstance();
    try {
        cm.SetScope({{"pass.pg_lower_bound", "1"}});
        FAIL() << "Expected exception was not thrown.";
    } catch (const std::exception& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Option 'pass.pg_lower_bound' has invalid type."), std::string::npos);
        EXPECT_NE(msg.find("Expected int64"), std::string::npos);
    }
}
