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
 * \file test_common.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "cost_model/simulation/emulator/SoftMemory.h"
#include "interface/configs/config_manager.h"
#include "test_common_types.h"

using namespace npu::tile_fwk;
using Json = nlohmann::json;
using namespace std;

template <typename T = float>
static void readInput(std::string filename, vector<T>& inputData)
{
    ifstream input_file(filename, ios::binary);
    if (!input_file) {
        std::cerr << "Failed to open file for writing input data! filename:" << filename;
        ASSERT(false);
    }
    input_file.read((char*)inputData.data(), inputData.size() * sizeof(T));
    input_file.close();
}

template <typename T>
static void writeInput(std::string filename, vector<T> outData)
{
    std::ofstream ascendOutFile(filename, std::ios::out | std::ios::binary);
    if (!ascendOutFile) {
        std::cerr << "Can not open out file!" << std::endl;
    }
    ascendOutFile.write((char*)outData.data(), outData.size() * sizeof(T));
    ascendOutFile.close();
}

template <typename T = float>
static bool resultCmpUnary(const vector<T>& x, const vector<T>& outDataValExp, const vector<T>& outDataValAct,
                           float eps, size_t threshold = 1, bool printAll = false, bool printErr = false)
{
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }
    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto inVal = static_cast<float>(x[eIdx]);
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = (std::abs(expVal) < 0.001 && std::abs(actVal) < 0.001) ? diff : std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);

        auto eErr = (diff > eps && relRatio > eps);
        errCount += eErr ? 1 : 0;

        if ((printAll) || (eErr && printErr)) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", input->" << inVal << ", exp->" << expVal
                      << ", act->" << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
    }
    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto inVal = static_cast<float>(x[eIdx]);
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = (std::abs(expVal) < 0.001 && std::abs(actVal) < 0.001) ? diff : std::abs(diff / expVal);

        auto eErr = (diff > eps && relRatio > eps);
        if (eErr) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", input->" << inVal << ", exp->" << expVal
                      << ", act->" << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename Ts, typename Td>
static bool resultCmpCast(const vector<Ts>& x, const vector<Td>& outDataValExp, const vector<Td>& outDataValAct,
                          float eps, size_t threshold = 1, bool printAll = false, bool printErr = false)
{
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }

    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto inVal = static_cast<float>(x[eIdx]);
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = expVal - actVal;
        auto relRatio = std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);

        auto eErr = (diff > eps && relRatio > eps);
        errCount += eErr ? 1 : 0;

        if ((printAll) || (eErr && printErr)) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", input->" << inVal << ", exp->" << expVal
                      << ", act->" << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
    }
    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);

        auto eErr = (diff > eps && relRatio > eps);
        if (eErr) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename T>
static bool resultCmp4TopK(const std::vector<T>& outDataValExp, const T* outDataValAct, size_t selectedCount,
                           float ratio)
{
    size_t data_size = outDataValExp.size();
    bool precision = true;

    if (data_size != static_cast<size_t>(data_size)) {
        return false;
    }

    std::map<size_t, std::pair<std::vector<T>, std::vector<T>>> part_result_dict;
    std::map<size_t, std::pair<std::vector<T>, std::vector<T>>> all_result_dict;

    for (size_t idx = 0; idx < data_size; ++idx) {
        int32_t expVal = outDataValExp[idx];
        int32_t actVal = outDataValAct[idx];
        size_t part_index = static_cast<size_t>(idx / selectedCount);

        if (expVal != actVal) {
            if (part_result_dict.find(part_index) == part_result_dict.end()) {
                part_result_dict[part_index] = {{}, {}};
            }
            part_result_dict[part_index].first.push_back(expVal);
            part_result_dict[part_index].second.push_back(actVal);
        }

        if (idx % selectedCount == 0) {
            all_result_dict[part_index] = {{}, {}};
        }
        all_result_dict[part_index].first.push_back(expVal);
        all_result_dict[part_index].second.push_back(actVal);
    }

    for (const auto& [idx_index, result_pair] : part_result_dict) {
        (void)idx_index;
        std::vector<T> exp_list = result_pair.first;
        std::vector<T> act_list = result_pair.second;

        std::sort(exp_list.begin(), exp_list.end());
        std::sort(act_list.begin(), act_list.end());

        size_t error_count = 0;
        std::vector<T> error_list;
        for (T tok_id : exp_list) {
            if (std::find(act_list.begin(), act_list.end(), tok_id) == act_list.end()) {
                error_count++;
                error_list.push_back(tok_id);
            }
        }
        if (error_count > size_t(selectedCount * ratio)) {
            precision = false;

            std::cout << "current group idx: " << idx_index << " failed, error info: " << std::endl;
            for (auto expValue : error_list) {
                std::vector<T> exp_list_ori = all_result_dict[idx_index].first;
                std::vector<T> act_list_ori = all_result_dict[idx_index].second;
                auto pos_idx = -1;
                auto it = std::find(exp_list_ori.begin(), exp_list_ori.end(), expValue);
                if (it != exp_list_ori.end()) {
                    pos_idx = std::distance(exp_list_ori.begin(), it);
                }
                std::cout << "err idx: " << pos_idx << ", exp->" << expValue << ", act->" << act_list_ori[pos_idx]
                          << std::endl;
            }
            // break;
        }
    }
    std::cout << "result is "
              << (precision ? "\033[32m"
                              "PASS"
                              "\033[0m" :
                              "\033[31m"
                              "FAILED"
                              "\033[0m")
              << std::endl;
    return precision;
}

template <typename T = float>
static bool resultCmp(const T* outDataValExp, const T* outDataValAct, size_t eSize, float eps, size_t threshold = 0,
                      size_t zeroCountThreshold = 1000, bool printAll = false, bool printErr = false,
                      size_t testNum = 0)
{
    //
    threshold = threshold == 0 ? static_cast<int>(eSize * eps) : threshold;

    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t zeroCount = 0;
    size_t errCount = 0;

    bool rst = true;
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);
        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;
        testNum = testNum - (testNum > 0 ? 1 : 0);

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if ((printAll) || (eErr && printErr) || (testNum > 0)) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
    }

    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    float zeroCountRatio = static_cast<float>(zeroCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio
              << ", act zero count: " << zeroCount << ", act zero threshold: " << zeroCountThreshold
              << ", act zero ratio: " << zeroCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }

    errCount = 0;
    zeroCount = 0;
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if (eErr) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename T = float>
static bool resultCmp(const vector<T>& outDataValExp, const T* outDataValAct, float eps, size_t threshold = 0,
                      size_t zeroCountThreshold = 1000, bool printAll = false, bool printErr = false,
                      size_t testNum = 0)
{
    //
    threshold = threshold == 0 ? static_cast<int>(outDataValExp.size() * eps) : threshold;

    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t zeroCount = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);
        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;
        testNum = testNum - (testNum > 0 ? 1 : 0);

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if ((printAll) || (eErr && printErr) || (testNum > 0)) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
    }

    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    float zeroCountRatio = static_cast<float>(zeroCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio
              << ", act zero count: " << zeroCount << ", act zero threshold: " << zeroCountThreshold
              << ", act zero ratio: " << zeroCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }

    errCount = 0;
    zeroCount = 0;
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if (eErr) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename T = float>
static bool resultCmpPrint(const vector<T>& outDataValExp, const T* outDataValAct, float eps, size_t testNum = 0)
{
    return resultCmp(outDataValExp, outDataValAct, eps, 8, 1000, false, false, testNum);
}

template <typename T = float>
static bool resultCmpAbsDelta(const vector<T>& outDataValExp, const T* outDataValAct, float absDelta,
                              size_t threshold = 0, size_t zeroCountThreshold = 1000, bool printAll = false,
                              bool printErr = false, size_t testNum = 0)
{
    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t zeroCount = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);
        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;
        testNum = testNum - (testNum > 0 ? 1 : 0);

        auto eErr = ((diff > absDelta) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if ((printAll) || (eErr && printErr) || (testNum > 0)) {
            std::cout << "abs diff threshold: " << absDelta << ", idx: " << eIdx << ", exp->" << expVal << ", act->"
                      << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
    }

    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    float zeroCountRatio = static_cast<float>(zeroCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio
              << ", act zero count: " << zeroCount << ", act zero threshold: " << zeroCountThreshold
              << ", act zero ratio: " << zeroCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }

    errCount = 0;
    zeroCount = 0;
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;

        auto eErr = ((diff > absDelta) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if (eErr) {
            std::cout << "abs diff threshold: " << absDelta << ", idx: " << eIdx << ", exp->" << expVal << ", act->"
                      << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename T = float>
static bool resultCmp(const vector<T>& outDataValExp, const vector<T>& outDataValAct, float eps, size_t threshold = 0,
                      size_t zeroCountThreshold = 1000, bool printAll = false, bool printErr = false,
                      size_t testNum = 0)
{
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }
    return resultCmp(outDataValExp, outDataValAct.data(), eps, threshold, zeroCountThreshold, printAll, printErr,
                     testNum);
}

template <class T = float>
void* readToDev(const std::string& path, int size)
{
    size_t bytes = size * sizeof(T);
    std::vector<uint8_t> data(bytes);
    readInput(path, data);

    uint8_t* devPtr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&devPtr, bytes);
    if (devPtr == nullptr) {
        std::cout << "RuntimeMalloc failed" << std::endl;
        devPtr = reinterpret_cast<uint8_t*>(CostModel::SoftMemory::Instance().AllocateData(bytes, data));
        if (devPtr == nullptr) {
            std::cout << "SoftMemory RuntimeMalloc failed" << std::endl;
            return nullptr;
        } else {
            return devPtr;
        }
    }
    RuntimeMemcpy(devPtr, bytes, data.data(), bytes, RtMemcpyKind::HOST_TO_DEVICE);
    return devPtr;
}

[[maybe_unused]] static uint8_t* allocDevAddr(uint64_t size)
{
    uint8_t* devPtr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&devPtr, size);
    if (devPtr == nullptr) {
        std::cout << "allocDevAddr RuntimeMalloc failed" << std::endl;
        std::vector<uint8_t> data(size);
        devPtr = reinterpret_cast<uint8_t*>(CostModel::SoftMemory::Instance().AllocateData(size, data));
        if (devPtr == nullptr) {
            std::cout << "SoftMemory RuntimeMalloc failed" << std::endl;
            return nullptr;
        } else {
            return devPtr;
        }
        return nullptr;
    }

    return devPtr;
}

[[maybe_unused]] static std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    std::string fullName = std::string(testInfo->test_suite_name()) + "." + testInfo->name();
    // 先读取 TILE_FWK_STEST_GOLDEN_PATH 环境变量, 否则使用当前目录
    char* path = getenv("TILE_FWK_STEST_GOLDEN_PATH");
    std::string fullPath;
    if (path == nullptr) {
        fullPath = "./golden";
    } else {
        fullPath = std::string(path);
    }
    fullPath = fullPath + "/" + fullName;
    return fullPath;
}

inline void SetInterpreterConfig()
{
// 通过build_ci.py --enable_interpreter_config使能
#ifdef ENABLE_STEST_INTERPRETER_CONFIG
    // config::SetVerifyOption(KEY_VERIFY_TENSOR_GRAPH, true);
    // config::SetVerifyOption(KEY_VERIFY_PASS, true);
    // config::SetVerifyOption(KEY_VERIFY_EXECUTE_GRAPH, true);
    // config::SetVerifyOption(KEY_VERIFY_CHECK_PRECISION, true);
#endif
}
