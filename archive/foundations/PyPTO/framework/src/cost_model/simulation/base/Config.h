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
 * \file Config.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <string>
#include <regex>
#include "tilefwk/error.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

namespace CostModel {

class Config {
protected:
    std::string prefix;
    std::map<std::string, std::function<void(std::string const&)>> dispatcher;

public:
    std::map<std::string, std::function<std::string()>> recorder;
    Config() = default;
    virtual ~Config() = default;

    virtual void OverrideDefaultConfig(std::vector<std::string>* cfgs) final
    {
        std::regex r{"([\\w.]+)=(\\S+)"};
        std::smatch sm;
        size_t parameterNum = 3;
        for (auto& c : *cfgs) {
            regex_match(c, sm, r);
            ASSERT(CostModel::ExternalErrorScene::INVALID_CONFIG, sm.size() == parameterNum)
                << "[SIMULATION]: "
                << "the config regex size is 3. the format is error: " << c;
            std::string cfgName{sm.str(1)};
            std::string cfgValue{sm.str(2)};
            ParseConfig(cfgName, cfgValue);
        }
    }

    void ParseConfig(std::string const& cfgName, std::string const& cfgValue)
    {
        if (cfgName.substr(0, prefix.size()) == prefix) {
            ASSERT(CostModel::ExternalErrorScene::INVALID_CONFIG, cfgName[prefix.size()] == '.')
                << "[SIMULATION]: "
                << "cfgName format is error: " << cfgName;
            auto it = dispatcher.find(cfgName.substr(prefix.size() + 1));
            if (it != dispatcher.end()) {
                it->second(cfgValue);
            } else {
                SIMULATION_LOGE(CostModel::ExternalErrorScene::INVALID_CONFIG_NAME, "Invalid config name: %s",
                                cfgName.c_str());
            }
        }
    }

    virtual std::string const& ParseString(std::string const& v) final { return v; }

    virtual uint64_t ParseInteger(std::string const& v) final
    {
        int readWidth = 16;
        if (v[0] == '0' && (v[1] == 'x' || v[1] == 'X')) {
            return stoull(v, nullptr, readWidth);
        } else {
            return stoull(v);
        }
    }

    virtual bool ParseBoolean(std::string const& v) final
    {
        if (v == "false") {
            return false;
        } else if (v == "true") {
            return true;
        }
        return false;
    }

    virtual void ParseIntVec(std::string const& v, std::vector<uint64_t>& array) final
    {
        std::size_t pos = 0;
        array.clear();
        std::string vv = v;
        while ((pos = vv.find(':')) != std::string::npos) {
            array.push_back(stoull(vv.substr(0, pos)));
            vv.erase(0, pos + 1);
        }
        array.push_back(stoull(vv.substr(0, pos)));
    }

    virtual void ParseStrVec(std::string const& v, std::vector<std::string>& array) final
    {
        std::size_t pos = 0;
        array.clear();
        std::string vv = v;
        while ((pos = vv.find(',')) != std::string::npos) {
            array.push_back(vv.substr(0, pos));
            vv.erase(0, pos + 1);
        }
        array.push_back(vv.substr(0, pos));
    }

    virtual std::string ParameterToStr(bool parameter) final { return std::to_string(parameter); }

    virtual std::string ParameterToStr(uint64_t parameter) final { return std::to_string(parameter); }

    virtual std::string ParameterToStr(std::vector<uint64_t>& parameter) final
    {
        std::stringstream oss;
        oss << "[";
        for (auto& it : parameter) {
            oss << it << ",";
        }
        oss << "]";
        return oss.str();
    }

    virtual std::string ParameterToStr(const std::string& parameter) final { return parameter; }

    virtual std::string ParameterToStr(std::vector<std::string>& parameter) final
    {
        std::stringstream oss;
        oss << "[";
        for (size_t i = 0; i < parameter.size(); ++i) {
            oss << parameter[i];
            if (i < parameter.size() - 1) {
                oss << ",";
            }
        }
        oss << "]";
        return oss.str();
    }

    virtual std::string DumpParameters() final
    {
        std::stringstream oss;
        oss << "[" << prefix << "]" << std::endl;
        for (auto& it : recorder) {
            oss << it.second() << std::endl;
        }
        oss << std::endl;
        return oss.str();
    }
};

} // namespace CostModel
