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
 * \file split_reshape_checker.cpp
 * \brief
 */

#include "split_reshape_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SplitReshape"

namespace npu {
namespace tile_fwk {
Status SplitReshapeChecker::DoDefaultEnabledPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start Precheck for SplitReshape.");
    if (CheckAssembleOverlap(function) == FAILED) {
        APASS_LOG_WARN_F(Elements::Function, "Precheck of SplitReshape failed since overlaps of assemble inputs.");
    }
    APASS_LOG_INFO_F(Elements::Function, "End Precheck for SplitReshape.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
