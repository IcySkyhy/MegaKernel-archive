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
 * \file ParseArgs.h
 * \brief
 */

#ifndef PARSEARGS_H
#define PARSEARGS_H

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <functional>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

namespace CostModel {
class ParseArgs {
public:
    void RegisterParam(const std::string& key, int& value, const std::string& description)
    {
        params_[key] = [this, &value](const std::string& str) {
            std::istringstream iss(str);
            iss >> value;
        };
        descriptions_[key] = description;
    }

    void RegisterParam(const std::string& key, bool& value, const std::string& description)
    {
        params_[key] = [this, &value](const std::string& str) { value = (str == "true" || str == "1"); };
        descriptions_[key] = description;
    }

    void RegisterParam(const std::string& key, std::string& value, const std::string& description)
    {
        params_[key] = [this, &value](const std::string& str) { value = str; };
        descriptions_[key] = description;
    }

    void RegisterParam(const std::string& key, std::vector<std::string>& values, const std::string& description)
    {
        paramArrays_[key] = &values;
        descriptions_[key] = description;
    }

    void ParseSingleArgs(const std::vector<std::string>& args, size_t& currentIndex)
    {
        auto& index = args[currentIndex];
        if (currentIndex + 1 < args.size()) {
            params_[index](args[currentIndex + 1]);
            ++currentIndex; // 跳过下一个参数
        } else {
            SIMULATION_LOGE(CostModel::ExternalErrorScene::INVALID_CONFIG, "Missing argument for %s",
                            args[currentIndex].c_str());
        }
    }

    void ParseArrays(const std::vector<std::string>& args, size_t& currentIndex)
    {
        auto arrayIt = paramArrays_.find(args[currentIndex]);
        if (arrayIt != paramArrays_.end()) {
            while (currentIndex + 1 < args.size() && args[currentIndex + 1][0] != '-') {
                (*arrayIt->second).push_back(args[currentIndex + 1]);
                ++currentIndex; // 跳过下一个参数
            }
        } else {
            SIMULATION_LOGE(CostModel::ExternalErrorScene::INVALID_CONFIG, "Unknown parameter: %s",
                            args[currentIndex].c_str());
        }
    }

    // 解析参数
    void Parse(const std::vector<std::string>& args)
    {
        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i][0] == '-') {
                auto it = params_.find(args[i]);
                if (it != params_.end()) {
                    ParseSingleArgs(args, i);
                } else {
                    ParseArrays(args, i);
                }
            }
        }
    }

private:
    std::map<std::string, std::function<void(const std::string&)>> params_;
    std::map<std::string, std::vector<std::string>*> paramArrays_;
    std::map<std::string, std::string> descriptions_;
};
} // namespace CostModel
#endif
