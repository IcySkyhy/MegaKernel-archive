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
 * \file test_distributed.h
 * \brief
 */

#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include "test_common.h"
#include "interface/configs/config_manager.h"
#include "distributed_op_test_suite.h"
#include <filesystem>
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::Distributed {

struct OpMetaData {
    explicit OpMetaData(const nlohmann::json& testData, std::string& fileName)
        : testData_(testData), fileName_(fileName)
    {}
    nlohmann::json testData_;
    std::string fileName_;
};

struct DisOpRegister {
    using opFunc = std::function<void(OpTestParam&, const std::string&, std::string& goldenDir)>;
    std::unordered_map<std::string, opFunc> disRegisterMap;

    static DisOpRegister& GetRegister()
    {
        static DisOpRegister disOpRegister;
        return disOpRegister;
    }

    template <typename TFunc>
    void RegisterOp(const std::string& opName, TFunc func)
    {
        disRegisterMap[opName] = [func](OpTestParam& testParam, const std::string& dtype, std::string& goldenDir) {
            if (dtype == "int32") {
                func.template operator()<int32_t>(testParam, goldenDir);
            } else if (dtype == "float16") {
                func.template operator()<float16>(testParam, goldenDir);
            } else if (dtype == "bfloat16") {
                func.template operator()<bfloat16>(testParam, goldenDir);
            } else if (dtype == "float32") {
                func.template operator()<float>(testParam, goldenDir);
            } else {
                FAIL() << "Unsupported dtype: " << dtype;
            }
        };
    }

    void Run(const std::string& opName, OpTestParam& testParam, const std::string& dtype, std::string& goldenDir)
    {
        if (!disRegisterMap.count(opName)) {
            FAIL() << "Unsupported op: " << opName;
        }
        disRegisterMap[opName](testParam, dtype, goldenDir);
    }
};

template <typename T>
std::vector<T> GetOpMetaDataFromFile(const std::filesystem::path& filePath)
{
    std::ifstream jsonFile(filePath);
    if (!jsonFile.is_open()) {
        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "Failed to open Json file for Path: %s",
                         std::filesystem::absolute(filePath).string().c_str());
        return {};
    }
    std::string fileName = filePath.stem().string();
    std::vector<T> testCaseList;
    nlohmann::json jsonData = nlohmann::json::parse(jsonFile);
    for (auto& tc : jsonData.at("test_cases")) {
        testCaseList.emplace_back(tc, fileName);
    }
    if (testCaseList.empty()) {
        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "No test cases found in json for File: %s",
                         std::filesystem::absolute(filePath).string().c_str());
    }
    return testCaseList;
}

template <typename T>
std::vector<T> GetOpMetaDataFromDir(const std::filesystem::path& dirPath)
{
    static std::vector<T> allTestCases;
    std::vector<std::filesystem::path> jsonFiles;
    for (const auto& file : std::filesystem::directory_iterator(dirPath)) {
        if (file.path().extension() != ".json")
            continue;
        jsonFiles.push_back(file.path());
    }
    std::sort(jsonFiles.begin(), jsonFiles.end(), [](const auto& a, const auto& b) {
        auto aLower = a.filename().string();
        auto bLower = b.filename().string();
        std::transform(aLower.begin(), aLower.end(), aLower.begin(), ::tolower);
        std::transform(bLower.begin(), bLower.end(), bLower.begin(), ::tolower);
        return aLower < bLower;
    });
    for (const auto& file : jsonFiles) {
        auto testCases = GetOpMetaDataFromFile<T>(file);
        allTestCases.insert(allTestCases.end(), testCases.begin(), testCases.end());
    }
    return allTestCases;
}

template <typename T>
std::vector<T> GetOpMetaData()
{
    const char* jsonPath = std::getenv("JSON_PATH");
    std::filesystem::path casePath;
    if (jsonPath != nullptr) {
        casePath = std::filesystem::path(jsonPath);
    } else {
        casePath = std::filesystem::path(TEST_CASE_EXE_DIR) / TEST_CASE_RELATIVE_PATH;
    }
    if (!std::filesystem::exists(casePath)) {
        DISTRIBUTED_LOGE(DistributedErrorCode::UNKNOW_ERROR, "JSON path does not exist: %s", casePath.string().c_str());
        return {};
    }
    if (std::filesystem::is_regular_file(casePath)) {
        return GetOpMetaDataFromFile<T>(casePath);
    }
    if (std::filesystem::is_directory(casePath)) {
        return GetOpMetaDataFromDir<T>(casePath);
    }
    return {};
}

std::string GetGoldenDirPath(const nlohmann::json& testData, const std::string& fileName)
{
    std::filesystem::path goldenDir = GetGoldenDir();
    goldenDir = goldenDir.parent_path().parent_path();
    std::string operation = testData["operation"].get<std::string>();
    int32_t caseIndex = testData["case_index"].get<int32_t>();
    std::string caseName = testData["case_name"].get<std::string>();
    goldenDir /= std::filesystem::path(operation) / std::filesystem::path(fileName);
    goldenDir /= std::to_string(caseIndex) + "_" + caseName;
    return goldenDir.string();
}

void GegisterOps();

} // namespace npu::tile_fwk::Distributed
