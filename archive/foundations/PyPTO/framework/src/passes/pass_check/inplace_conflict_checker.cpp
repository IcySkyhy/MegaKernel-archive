/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_outcast_checker.cpp
 * \brief
 */

#include <utility>
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "inplace_conflict_checker.h"

#define MODULE_NAME "InplaceConflictChecker"

namespace npu {
namespace tile_fwk {

constexpr size_t DST_TILE_INPUT_PARAM_INDEX = 2;

/*
    pypto.scatter_update()约束需要原地写法x = pypto.scatter_update(x, -2, y, z)操作结果会写回到x
    如果x同时作为其它OP的输入，无法保证读取到的是scatter_update更新后的值
*/
Status InplaceConflictChecker::CheckIndexOutcastDisorderedCoverage(Function& function)
{
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    for (const auto& tensor : allTensors) {
        if (tensor->GetConsumers().size() <= 1) {
            continue;
        }
        for (const auto& consumerOp : tensor->GetConsumers()) {
            if (consumerOp->GetOpcode() != Opcode::OP_INDEX_OUTCAST) {
                continue;
            }
            if (consumerOp->GetIOperands().size() <= DST_TILE_INPUT_PARAM_INDEX ||
                consumerOp->GetIOperands()[DST_TILE_INPUT_PARAM_INDEX]->GetMagic() != tensor->GetMagic()) {
                continue;
            }
            APASS_LOG_WARN_F(Elements::Tensor,
                             "Tensor[%d] is the dst of OP_INDEX_OUTCAST, it can't be input of other OP.",
                             tensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

/*
    tensor -> view/reshape -> 其他原地操作
   场景，如果tensor还作为了其他Op的输入，则无法确认其他Op从tensor中取到的值是被原地操作更新前还是跟新后的值
*/
Status InplaceConflictChecker::CheckInplaceOperationConflict(Function& function)
{
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    for (const auto& tensor : allTensors) {
        if (tensor->GetConsumers().size() <= 1) {
            continue;
        }
        std::unordered_set<Opcode> checkOpSet = {Opcode::OP_RESHAPE, Opcode::OP_VIEW};
        for (const auto& consumerOp : tensor->GetConsumers()) {
            if (checkOpSet.count(consumerOp->GetOpcode()) == 0) {
                continue;
            }
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "Tensor[%d] is the input of multiple operations, and contains inplace operation, the precision may "
                "be abnormal.",
                tensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
