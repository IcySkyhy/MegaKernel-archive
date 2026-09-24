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
 * \file common_operation_eliminate_checker.cpp
 * \brief
 */

#include "common_operation_eliminate_checker.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "CommonOperationEliminate"

namespace npu {
namespace tile_fwk {
Status CommonOperationEliminateChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for CommonOperationEliminate.");

    for (auto& op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_SHMEM_LOAD) {
            continue;
        }
        if (op->GetOpAttribute() != nullptr) {
            size_t fromOffsetSize = -1;
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op->GetOpAttribute().get())) {
                auto& fromOffset = viewOpAttribute->GetFromOffset();
                fromOffsetSize = fromOffset.size();
            } else if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(op->GetOpAttribute().get())) {
                if (copyOpAttribute->IsCopyOut()) {
                    continue;
                }
                auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
                (void)memType;
                fromOffsetSize = fromOffset.size();
            } else {
                continue;
            }
            auto& ioperands = op->GetIOperands();
            const int opMagic = op->GetOpMagic();

            if (ioperands.size() != 1) {
                APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                                  "View or Copy_In Operation %d with not one input operand.", opMagic);
                return FAILED;
            }
            if (ioperands.front()->offset.size() != fromOffsetSize) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_SHAPE_MISMATCH, Elements::Operation,
                                  "View or Copy_In Operation %d with mismatch input offset shape.", opMagic);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
