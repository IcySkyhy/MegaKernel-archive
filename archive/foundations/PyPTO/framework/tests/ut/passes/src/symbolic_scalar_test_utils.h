/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "interface/tensor/irbuilder.h"

namespace npu::tile_fwk {
inline SymbolicScalar CreateTestScalarVar(const std::string& sym)
{
    static std::mutex scalarVarMutex;
    static std::unordered_map<std::string, SymbolicScalar> scalarVars;
    std::lock_guard<std::mutex> lock(scalarVarMutex);

    // 整个 UT 进程内为同一个字面量保留同一个 scalar：这些用例需要稳定的字面量名称身份，
    // 而 IRBuilder::CreateScalarVar 会对重复名称做唯一化处理。
    auto iter = scalarVars.find(sym);
    if (iter != scalarVars.end()) {
        return iter->second;
    }

    IRBuilder builder;
    auto result = scalarVars.emplace(sym, builder.CreateScalarVar(sym));
    return result.first->second;
}

inline std::vector<SymbolicScalar> CreateTestConstIntVector(const std::vector<int64_t>& values)
{
    IRBuilder builder;
    std::vector<SymbolicScalar> result;
    result.reserve(values.size());
    for (auto value : values) {
        result.emplace_back(builder.CreateConstInt(value));
    }
    return result;
}

inline std::vector<SymbolicScalar> CreateTestConstIntVector(std::initializer_list<int64_t> values)
{
    return CreateTestConstIntVector(std::vector<int64_t>(values));
}
} // namespace npu::tile_fwk
