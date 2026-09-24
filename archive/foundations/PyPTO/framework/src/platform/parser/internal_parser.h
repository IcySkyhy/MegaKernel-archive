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
 * \file internal_parser.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <limits>
#include <fstream>
#include <sys/stat.h>
#include "utils/file_utils.h"
#include "tilefwk/data_type.h"

namespace npu {
namespace tile_fwk {
class InternalParser {
public:
    InternalParser(const std::string& archType) : archType_(archType) {}
    bool LoadInternalInfo();
    bool GetDataPath(std::vector<std::pair<MemoryType, MemoryType>>& dataPath);

private:
    std::string archType_;
    std::map<std::string, std::string> data_;
};
} // namespace tile_fwk
} // namespace npu
