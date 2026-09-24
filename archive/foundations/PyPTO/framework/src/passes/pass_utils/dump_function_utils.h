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
 * \file dump_function_utils.h
 * \brief 提供遍历 TileFunction 并调用基类 PrintFunction/DumpFunctionJson 的公共工具
 */

#pragma once

#include <unordered_set>
#include <string>
#include <functional>
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu {
namespace tile_fwk {

class DumpFunctionUtils {
public:
    Status GetTileFunction(Function* function, std::unordered_set<Function*>& tileFunctionSet);

    Status DumpTileFunctionsJson(Function& function, const std::string& logFolder, bool beforeFunction,
                                 std::function<Status(Function&, const std::string&, bool)> dumpFunc);

    Status PrintTileFunctions(Function& function, const std::string& logFolder, bool beforeFunction,
                              std::function<Status(Function&, const std::string&, bool)> printFunc);

private:
    Status GetCallee(const Operation& callop, Function*& callFunc);
};

} // namespace tile_fwk
} // namespace npu
