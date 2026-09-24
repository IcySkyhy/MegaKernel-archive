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
 * \file bimap.h
 * \brief Bidirectional map template (key <-> string).
 */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

namespace npu {
namespace tile_fwk {

template <typename T>
class BiMap {
public:
    BiMap(const std::initializer_list<std::pair<T, std::string>>& init)
    {
        for (const auto& [i, s] : init) {
            type2strDict[i] = s;
            str2typeDict[s] = i;
        }
    }

    bool Count(T key) const { return type2strDict.count(key); }

    bool Count(const std::string& key) const { return str2typeDict.count(key); }

    const std::string& Find(T key, const std::string& defaultValue = "") const
    {
        if (type2strDict.count(key)) {
            return type2strDict.find(key)->second;
        } else {
            return defaultValue;
        }
    }

    T Find(const std::string& key, T defaultValue) const
    {
        if (str2typeDict.count(key)) {
            return str2typeDict.find(key)->second;
        } else {
            return defaultValue;
        }
    }

private:
    std::unordered_map<T, std::string> type2strDict;
    std::unordered_map<std::string, T> str2typeDict;
};

} // namespace tile_fwk
} // namespace npu
