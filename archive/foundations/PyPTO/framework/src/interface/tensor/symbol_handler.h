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
 * \file symbol_handler.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace npu::tile_fwk {

enum class SymbolHandlerId : uint64_t {
    GetInputShapeDimSize,
    GetInputShapeDim,
    GetInputData,
    GetTensorDataInt32Dim1,
    GetTensorDataInt32Dim2,
    GetTensorDataInt32Dim3,
    GetTensorDataInt32Dim4,
    GetViewValidShapeDim,
    IsLoopBegin,
    IsLoopEnd,
    TernaryOP,
    GetHcclRankId,
    BindTensor
};

const std::unordered_map<std::string, SymbolHandlerId> symbolHandlerIndexDict = {
    {"GetInputShapeDimSize", SymbolHandlerId::GetInputShapeDimSize},
    {"GetInputShapeDim", SymbolHandlerId::GetInputShapeDim},
    {"GetInputData", SymbolHandlerId::GetInputData},
    {"GetTensorDataInt32Dim1", SymbolHandlerId::GetTensorDataInt32Dim1},
    {"GetTensorDataInt32Dim2", SymbolHandlerId::GetTensorDataInt32Dim2},
    {"GetTensorDataInt32Dim3", SymbolHandlerId::GetTensorDataInt32Dim3},
    {"GetTensorDataInt32Dim4", SymbolHandlerId::GetTensorDataInt32Dim4},
    {"GetViewValidShapeDim", SymbolHandlerId::GetViewValidShapeDim},
    {"IsLoopBegin", SymbolHandlerId::IsLoopBegin},
    {"IsLoopEnd", SymbolHandlerId::IsLoopEnd},
    {"TernaryOP", SymbolHandlerId::TernaryOP},
    {"GetHcclRankId", SymbolHandlerId::GetHcclRankId},
    {"BindTensor", SymbolHandlerId::BindTensor},
};

struct SymbolHandler {
    SymbolHandler(SymbolHandlerId id, uint64_t index) : handlerId(id), symIndex(index) {}
    SymbolHandlerId handlerId;
    uint64_t symIndex;

    static std::string GetNameByHandlerId(SymbolHandlerId tmpHandlerId)
    {
        for (auto& [name, id] : symbolHandlerIndexDict) {
            if (id == tmpHandlerId) {
                return name;
            }
        }
        return "";
    }
};
} // namespace npu::tile_fwk
