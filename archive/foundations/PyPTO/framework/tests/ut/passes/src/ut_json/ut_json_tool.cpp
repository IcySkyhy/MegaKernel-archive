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
 * \file ut_json_tool.cpp
 * \brief Tool for pass.
 */

#include "ut_json_tool.h"

#include "utils/file_utils.h"

void DumpJsonFile(Json programJson, std::string jsonFilePath)
{
    std::string filePath = jsonFilePath;
    const auto slashPos = filePath.find_last_of('/');
    if (slashPos != std::string::npos) {
        (void)CreateDir(filePath.substr(0, slashPos), true);
    }

    std::ofstream outFile(filePath);
    if (outFile.is_open()) {
        outFile << programJson;
        outFile.close();
        std::cout << "Successfully saved the json file., file path: " << filePath << std::endl;
    } else {
        std::cerr << "Failed to save the json file" << std::endl;
    }
}

Json LoadJsonFile(std::string jsonFilePath)
{
    std::ifstream inFile(jsonFilePath);
    Json readData;
    inFile >> readData;
    return readData;
}
