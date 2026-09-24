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
 * \file assign_memory_type_checker.h
 * \brief
 */

#ifndef ASSIGN_MEMORY_TYPE_CHECKER_H
#define ASSIGN_MEMORY_TYPE_CHECKER_H

#include <queue>
#include <unordered_set>
#include "checker.h"
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/data_type.h"

namespace npu {
namespace tile_fwk {
class AssignMemoryTypeChecker : Checker {
public:
    Status DoPreCheck(Function& function) override;
    Status DoPostCheck(Function& function) override;

private:
    void CheckPattern(Operation* operation, std::queue<std::pair<Operation*, int>>& opQueue, int depth,
                      std::unordered_set<Operation*>& visited);
    Status CheckAmulBInputProducers(Operation& operation);
    Status CheckTensorNotMemUnknown(Function& function);
    Status CheckMoveOpReachable(Function& function);

    static const std::unordered_set<Opcode> kValidProducerOpcodes;
    static const std::unordered_set<MemoryType> kValidViewToTypes;
};
} // namespace tile_fwk
} // namespace npu
#endif // ASSIGN_MEMORY_TYPE_CHECKER_H
