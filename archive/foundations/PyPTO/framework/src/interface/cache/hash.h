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
 * \file hash.h
 * \brief
 */

#pragma once

#include <cstdint>
#include "interface/utils/common.h"
#include "interface/inner/pre_def.h"

namespace npu::tile_fwk {

class FunctionHash : public std::string {
public:
    FunctionHash() : std::string(), hash_(0) {}
    FunctionHash(unsigned long hash) : std::string(std::to_string(hash)), hash_(hash) {}

    FunctionHash(const FunctionHash&) = default;
    bool Empty() const { return std::string::size() == 0; }
    const std::string& Data() const { return *this; }
    uint64_t GetHash() const { return hash_; }

    bool operator==(const FunctionHash& h) { return Data() == h.Data(); }
    FunctionHash& operator=(const FunctionHash&) = default;

private:
    unsigned long hash_{0};
};
} // namespace npu::tile_fwk

template <>
struct std::hash<npu::tile_fwk::FunctionHash> {
    std::size_t operator()(const npu::tile_fwk::FunctionHash& h) const { return std::hash<std::string>()(h.Data()); }
};
