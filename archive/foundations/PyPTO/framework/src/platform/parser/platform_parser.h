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
 * \file platform_parser.h
 * \brief
 */
#ifndef PLATFORM_PARSER_H_
#define PLATFORM_PARSER_H_

#include <map>
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
class INIParser : public PlatformParser {
public:
    INIParser();
    INIParser(const std::string& socVersion);
    ~INIParser() = default;
    bool GetStringVal(const std::string& column, const std::string& key, std::string& val) const override;

private:
    void INIParserInit(const std::string& socVersion);
    bool ReadINIFile(const std::string& filepath);
    bool Initialize(const std::string& iniFilePath);
    std::map<std::string, std::map<std::string, std::string>> data_;
};

class CmdParser : public PlatformParser {
public:
    CmdParser() = default;
    ~CmdParser() = default;
    bool GetStringVal(const std::string& column, const std::string& key, std::string& val) const override;
};

std::string TrimLine(std::string_view s);
} // namespace tile_fwk
} // namespace npu
#endif
