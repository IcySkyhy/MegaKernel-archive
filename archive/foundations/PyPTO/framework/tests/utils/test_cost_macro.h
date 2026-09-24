/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cost_macro.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>

class TestCostManager {
public:
    static TestCostManager& Instance()
    {
        static TestCostManager instance;
        return instance;
    }

    void RegisterCost(const std::string& suiteName, const std::string& testName, double costSeconds)
    {
        std::string fullName = suiteName + "." + testName;
        costMap_[fullName] = costSeconds;
    }

    // 获取测试用例耗时（无则返回0.0）
    double GetCost(const std::string& fullName) const
    {
        auto it = costMap_.find(fullName);
        return it != costMap_.end() ? it->second : 0.0;
    }

private:
    // 私有构造/析构：禁止外部实例化
    TestCostManager() = default;
    ~TestCostManager() = default;

    // 禁用拷贝/移动：保证单例唯一性
    TestCostManager(const TestCostManager&) = delete;
    TestCostManager& operator=(const TestCostManager&) = delete;

    std::unordered_map<std::string, double> costMap_;
};

#define TEST_WITH_COST(TestCaseName, TestName, CostSeconds)                              \
    static bool g_##TestCaseName##_##TestName##_cost_registered = []() {                 \
        TestCostManager::Instance().RegisterCost(#TestCaseName, #TestName, CostSeconds); \
        return true;                                                                     \
    }();                                                                                 \
    TEST(TestCaseName, TestName)

#define TEST_F_WITH_COST(TestFixtureClass, TestName, CostSeconds)                            \
    static bool g_##TestFixtureClass##_##TestName##_cost_registered = []() {                 \
        TestCostManager::Instance().RegisterCost(#TestFixtureClass, #TestName, CostSeconds); \
        return true;                                                                         \
    }();                                                                                     \
    TEST_F(TestFixtureClass, TestName)

#define TEST_P_WITH_COST(TestFixtureClass, TestName, CostSeconds)                            \
    static bool g_##TestFixtureClass##_##TestName##_cost_registered = []() {                 \
        TestCostManager::Instance().RegisterCost(#TestFixtureClass, #TestName, CostSeconds); \
        return true;                                                                         \
    }();                                                                                     \
    TEST_P(TestFixtureClass, TestName)

inline void ListTestsWithMetadata()
{
    const testing::UnitTest& unitTest = *testing::UnitTest::GetInstance();

    // 遍历所有测试套件
    for (int suiteIdx = 0; suiteIdx < unitTest.total_test_suite_count(); ++suiteIdx) {
        const testing::TestSuite* testSuite = unitTest.GetTestSuite(suiteIdx);
        if (!testSuite) {
            continue;
        }

        std::string suiteName = testSuite->name();
        if (suiteName == "GoogleTestVerification") {
            continue;
        }

        // 遍历测试用例
        for (int testIdx = 0; testIdx < testSuite->total_test_count(); ++testIdx) {
            const testing::TestInfo* testInfo = testSuite->GetTestInfo(testIdx);
            if (!testInfo) {
                continue;
            }

            std::string testName = testInfo->name();
            std::string fullName = suiteName + "." + testName;

            // 仅处理有耗时注册的用例
            double cost = TestCostManager::Instance().GetCost(fullName);
            if (cost > 0.0) {
                std::cout << fullName << "|" << cost << std::endl;
            }
        }
    }
}
