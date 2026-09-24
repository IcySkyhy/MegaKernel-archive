/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file checker_utils.cpp
 * \brief
 */

#include "checker_utils.h"

namespace npu {
namespace tile_fwk {
bool OpChecker::CalcTypeChecker::check(Operation* op) const
{
    if (conditions.empty())
        return true;
    OpCalcType currentType = OpcodeManager::Inst().GetOpCalcType(op->GetOpcode());
    return std::find(conditions.begin(), conditions.end(), currentType) != conditions.end();
}

bool OpChecker::CoreTypeChecker::check(Operation* op) const
{
    if (conditions.empty())
        return true;
    OpCoreType currentType = OpcodeManager::Inst().GetCoreType(op->GetOpcode());
    return std::find(conditions.begin(), conditions.end(), currentType) != conditions.end();
}

bool OpChecker::InputMemTypeChecker::check(Operation* op) const
{
    if (conditions.empty())
        return true;
    const std::vector<MemoryType>& currentType = OpcodeManager::Inst().GetInputsMemType(op->GetOpcode());
    return std::any_of(currentType.begin(), currentType.end(), [this](const MemoryType& memType) {
        return std::find(conditions.begin(), conditions.end(), memType) != conditions.end();
    });
}

bool OpChecker::OutputMemTypeChecker::check(Operation* op) const
{
    if (conditions.empty())
        return true;
    const std::vector<MemoryType>& currentType = OpcodeManager::Inst().GetOutputsMemType(op->GetOpcode());
    return std::any_of(currentType.begin(), currentType.end(), [this](const MemoryType& memType) {
        return std::find(conditions.begin(), conditions.end(), memType) != conditions.end();
    });
}
} // namespace tile_fwk
} // namespace npu
