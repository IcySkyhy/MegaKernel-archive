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
#include <memory>

#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {

inline uint32_t CountOpcode(const std::shared_ptr<Function>& currFunctionPtr, Opcode opcode)
{
    uint32_t count = 0u;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == opcode) {
            ++count;
        }
    }
    return count;
}

inline const Operation* FindSingleOp(const std::shared_ptr<Function>& currFunctionPtr, Opcode opcode)
{
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == opcode) {
            return &op;
        }
    }
    return nullptr;
}

} // namespace tile_fwk
} // namespace npu
