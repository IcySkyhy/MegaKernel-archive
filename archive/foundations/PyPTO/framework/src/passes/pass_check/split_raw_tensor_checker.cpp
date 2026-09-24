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
 * \file split_raw_tensor_checker.cpp
 * \brief
 */

#include "split_raw_tensor_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"

#define MODULE_NAME "SplitRawTensor"

namespace npu {
namespace tile_fwk {
Status SplitRawTensorChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start PostCheck for SplitRawTensor.");
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    for (const auto& logicalTensor : allTensors) {
        if (function.IsFromOutCast(logicalTensor) || function.IsFromInCast(logicalTensor)) {
            continue;
        }
        if ((logicalTensor->GetShape() != logicalTensor->tensor->GetRawShape())) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d]'s shape(%s) is differ from its rawShape(%s).",
                              logicalTensor->GetMagic(), CommonUtils::ContainerToStr(logicalTensor->GetShape()).c_str(),
                              CommonUtils::ContainerToStr(logicalTensor->tensor->GetRawShape()).c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "End PostCheck for SplitRawTensor.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
