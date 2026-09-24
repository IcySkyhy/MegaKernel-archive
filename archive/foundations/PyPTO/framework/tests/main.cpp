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
 * \file main.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <cstring>
#include <cerrno>
#include <sched.h>
#include <algorithm>
#include "utils/test_cost_macro.h"
#include "interface/configs/config_manager_ng.h"

#if defined(ENABLE_STEST)
#include "adapter/api/runtime_api.h"
bool CheckDeviceConsistency()
{
    /* 获取实际生效的 DeviceId */
    int32_t rtDevId = -1; // -1 表示无效 DeviceId
    int32_t getDeviceResult = npu::tile_fwk::RuntimeGetDevice(&rtDevId);
    if (getDeviceResult != npu::tile_fwk::RT_SUCCESS) {
        std::cout << "Error: Can't get deviceId" << std::endl;
        return false;
    }

    /* 获取环境变量中设置的 DeviceId */
    int32_t envDevId = 0;
    const char* devIdPtr = getenv("TILE_FWK_DEVICE_ID");
    if (devIdPtr != nullptr) {
        envDevId = std::stoi(devIdPtr);
    }

    if (rtDevId != envDevId) {
        std::cout << "Error: rtDevId(" << rtDevId << ") != envDevId(" << envDevId << ")" << std::endl;
        return false;
    }
    return true;
}

#else

bool CheckDeviceConsistency() { return true; }

#endif

class TestExecutionCounter : public testing::EmptyTestEventListener {
public:
    uint64_t executed_count = 0;

    void OnTestStart(const testing::TestInfo&) override { executed_count++; }
};

class CpuAffinityManager {
public:
    // 禁止实例化, 所有功能通过静态函数提供
    CpuAffinityManager() = delete;
    ~CpuAffinityManager() = delete;
    CpuAffinityManager(const CpuAffinityManager&) = delete;
    CpuAffinityManager& operator=(const CpuAffinityManager&) = delete;

    static void GetProcessAffinity(std::vector<int>& cores, bool printDetail = false)
    {
        cores.clear();
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);

        // 调用系统接口获取当前进程亲和性(0表示当前进程PID)
        if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuSet) != 0) {
            std::cerr << "Failed to get process affinity: " << std::strerror(errno) << std::endl;
            return;
        }

        // 将cpu_set_t转换为易读的核心列表
        unsigned int cpuCount = getCpuCoreCount();
        cores = cpuSetToCoreList(cpuSet);

        // 打印结果
        if (!printDetail) {
            return;
        }
        if (cores.empty()) {
            return;
        }
        if (cores.size() == cpuCount) {
            return; // 未单独设置 CPU 亲和性时, 所有核都会被设置到
        }
        int firstCore = cores.front();
        int lastCore = cores.back();
        std::cout << "Note: CPU Affinity, CpuNum=" << cpuCount << ", Cores: " << firstCore << "~" << lastCore
                  << std::endl;
    }

    static bool SetProcessAffinity(const std::vector<int>& cores)
    {
        unsigned int cpuCount = getCpuCoreCount();
        if (cpuCount == 0) {
            return false;
        }

        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        for (int core : cores) {
            if (core < 0 || static_cast<unsigned int>(core) >= cpuCount) {
                std::cerr << "Invalid CPU core ID: " << core << std::endl;
                return false;
            }
            CPU_SET(core, &cpuSet);
        }
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuSet) != 0) {
            std::cerr << "Failed to set process affinity: " << std::strerror(errno) << std::endl;
            return false;
        }
        std::cout << "CPU Num: " << cpuCount << std::endl;
        std::cout << "Process affinity set to cores: ";
        for (size_t i = 0; i < cores.size(); ++i) {
            std::cout << cores[i];
            if (i != cores.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        return true;
    }

    static bool SetProcessAffinityFromEnv(const std::string& envName)
    {
        const char* envStr = std::getenv(envName.c_str());
        if (envStr == nullptr) {
            std::cerr << "Environment variable " << envName << " is not set" << std::endl;
            return false;
        }
        std::string envCoreStr = envStr;

        std::vector<int> targetCores;
        if (!parseAndValidateCores(envCoreStr, targetCores)) {
            return false;
        }

        return SetProcessAffinity(targetCores);
    }

private:
    static unsigned int getCpuCoreCount()
    {
        unsigned int cpuCount = std::thread::hardware_concurrency();
        if (cpuCount == 0) {
            // 兜底: 使用系统调用
            cpuCount = sysconf(_SC_NPROCESSORS_ONLN);
            if (cpuCount == 0) {
                std::cerr << "Failed to get CPU core count" << std::endl;
            }
        }
        return cpuCount;
    }

    static bool stringToInt(const std::string& str, int& outVal)
    {
        try {
            size_t pos;
            outVal = std::stoi(str, &pos);
            return pos == str.length(); // 确保整个字符串都是数字 (避免"12a"这类非法值)
        } catch (const std::invalid_argument&) {
            return false;
        } catch (const std::out_of_range&) {
            return false;
        }
    }

    static std::vector<std::string> splitString(const std::string& str, char delimiter)
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    static bool parseAndValidateCores(const std::string& envValue, std::vector<int>& outCores)
    {
        outCores.clear();
        std::vector<std::string> coreStrList = splitString(envValue, ';');

        for (const std::string& coreStr : coreStrList) {
            int coreId;
            if (!stringToInt(coreStr, coreId)) {
                std::cerr << "Invalid CPU core ID (not a number): " << coreStr << std::endl;
                outCores.clear();
                return false;
            }
            outCores.push_back(coreId);
        }

        return true;
    }

    static std::vector<int> cpuSetToCoreList(const cpu_set_t& cpuSet)
    {
        std::vector<int> coreList;
        unsigned int cpuCount = getCpuCoreCount();
        if (cpuCount == 0) {
            return coreList;
        }

        // 遍历所有CPU核心, 检查是否在掩码中
        for (unsigned int i = 0; i < cpuCount; ++i) {
            if (CPU_ISSET(i, &cpuSet)) {
                coreList.push_back(static_cast<int>(i));
            }
        }
        return coreList;
    }
};

int main(int argc, char** argv)
{
    // 查询 CPU 亲和性设置
    std::vector<int> cores;
    CpuAffinityManager::GetProcessAffinity(cores, true);

    // 特殊参数场景判断
    auto isListMetasFunc = [](const std::string& arg) { return arg == "--gtest_list_tests_with_meta"; };
    bool isListMetas = (std::find_if(argv + 1, argv + argc, isListMetasFunc) != argv + argc);

    auto isListTestsFunc = [](const std::string& arg) { return arg == "--gtest_list_tests"; };
    bool isListTests = (std::find_if(argv + 1, argv + argc, isListTestsFunc) != argv + argc);

    // 初始化 GTest
    testing::InitGoogleTest(&argc, argv);

    if (isListMetas) {
        ListTestsWithMetadata();
        return 0;
    }

    // 总开关：在 root scope 上设置 enable_slice=true（覆盖 tile_fwk_config.json 的 false 默认值）
    // root scope 不受 config::Reset() 影响，所以 UT SetUp 中的 Clear() 不会清除该设置
    // 单个测试用例仍可通过 config::SetPassOption(ENABLE_SLICE, false) 覆盖此设置
    const std::string enableSliceKey = "pass." + std::string(npu::tile_fwk::ENABLE_SLICE);
    npu::tile_fwk::ConfigManagerNg::GlobalScope()->UpdateValue(enableSliceKey, true);

    auto ret = RUN_ALL_TESTS();

    // 后检查
    if (isListTests) {
        return ret;
    }
    ret = CheckDeviceConsistency() ? ret : 1;

    return ret;
}
