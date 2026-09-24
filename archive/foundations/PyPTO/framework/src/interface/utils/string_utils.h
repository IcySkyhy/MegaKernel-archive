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
 * \file string_utils.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include "securec.h"
#include "tilefwk/error.h"

namespace npu::tile_fwk {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    for (auto iter = vec.begin(); iter != vec.end(); ++iter) {
        if (iter != vec.begin()) {
            os << ", ";
        }
        os << *iter;
    }
    os << "]";
    return os;
}

class StringUtils {
public:
    static constexpr size_t MAX_DATA_LEN = 0x40000000UL;
    static void Trim(std::string& str)
    {
        if (str.empty()) {
            return;
        }
        size_t startPos = str.find_first_not_of(" \t");
        size_t endPos = str.find_last_not_of(" \t");
        if (startPos == std::string::npos || startPos > endPos) {
            str.clear();
            return;
        }
        str = str.substr(startPos, endPos - startPos + 1);
    }

    static std::vector<std::string> Split(const std::string& str, const std::string& pattern)
    {
        std::vector<std::string> resVec;
        if (str.empty() || pattern.empty()) {
            return resVec;
        }
        std::string strAndPattern = str + pattern;
        size_t pos = strAndPattern.find(pattern);
        while (pos != std::string::npos) {
            resVec.push_back(strAndPattern.substr(0, pos));
            strAndPattern = strAndPattern.substr(pos + pattern.size());
            pos = strAndPattern.find(pattern);
        }
        return resVec;
    }

    static bool StartsWith(const std::string& str, const std::string& prefix)
    {
        if (str.size() < prefix.size())
            return false;
        for (size_t i = 0; i < prefix.size(); i++) {
            if (prefix[i] != str[i]) {
                return false;
            }
        }
        return true;
    }

    static bool EndsWith(const std::string& str, const std::string& suffix)
    {
        if (str.size() < suffix.size())
            return false;
        for (size_t i = 0; i < suffix.size(); i++) {
            if (suffix[suffix.size() - 1 - i] != str[str.size() - 1 - i]) {
                return false;
            }
        }
        return true;
    }

    static std::string BaseName(const char* fname)
    {
        if (auto start = strrchr(fname, '/')) {
            return start + 1;
        }
        return fname;
    }

    static std::string ToLower(const std::string& str)
    {
        std::string res(str);
        for (auto& c : res) {
            c = std::tolower(c);
        }
        return res;
    }

    static std::string ToUpper(const std::string& str)
    {
        std::string res(str);
        for (auto& c : res) {
            c = std::toupper(c);
        }
        return res;
    }

    // memcpy_s内部限制了待拷贝目的buffer的字节数，若大于SECUREC_MEM_MAX_LEN（int32最大值0x7fffffff），会报错返回错误码ERANGE
    static void DataCopy(void* dest, size_t destMax, const void* src, size_t count)
    {
        ASSERT(destMax >= count) << "destMax: " << destMax << ", count: " << count;

        size_t offset = 0;
        while (offset < count) {
            size_t copyLen = std::min(count - offset, MAX_DATA_LEN);
            auto err = memcpy_s(static_cast<char*>(dest) + offset, copyLen, static_cast<const char*>(src) + offset,
                                copyLen);
            ASSERT(err == 0) << "errCode: " << err;
            offset += copyLen;
        }
    }

    // memset_s内部限制了待拷贝目的buffer的字节数，若大于SECUREC_MEM_MAX_LEN（int32最大值0x7fffffff），会报错返回错误码ERANGE
    static void DataSet(void* dest, size_t destMax, int c, size_t count)
    {
        ASSERT(destMax >= count) << "destMax: " << destMax << ", count: " << count;

        size_t offset = 0;
        while (offset < count) {
            size_t copyLen = std::min(count - offset, MAX_DATA_LEN);
            auto err = memset_s(static_cast<char*>(dest) + offset, copyLen, c, copyLen);
            ASSERT(err == 0) << "errCode: " << err;
            offset += copyLen;
        }
    }

    template <typename T>
    static std::string ToString(const std::vector<T>& args)
    {
        std::stringstream ss;
        ss << args;
        return ss.str();
    }

    static void AppendUniqueToken(std::string& acc, const std::string& tok, char sep = ';')
    {
        if (tok.empty()) {
            return;
        }
        size_t pos = 0;
        while (pos <= acc.size()) {
            size_t s = acc.find(sep, pos);
            size_t end = (s == std::string::npos) ? acc.size() : s;
            if (acc.compare(pos, end - pos, tok) == 0) {
                return;
            }
            if (s == std::string::npos) {
                break;
            }
            pos = s + 1;
        }
        if (!acc.empty()) {
            acc.push_back(sep);
        }
        acc.append(tok);
    }
};
} // namespace npu::tile_fwk
