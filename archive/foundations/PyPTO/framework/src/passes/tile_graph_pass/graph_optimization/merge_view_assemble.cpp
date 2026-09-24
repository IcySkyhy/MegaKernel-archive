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
 * \file merge_view_assemble.cpp
 * \brief Implementation of view and assemble operation merging pass
 */

#include "merge_view_assemble.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_check/merge_view_assemble_checker.h"

#define MODULE_NAME "MergeViewAssemble"

namespace npu::tile_fwk {
Status MergeViewAssemble::PreCheck(Function& function)
{
    MergeViewAssembleChecker checker;
    return checker.DoPreCheck(function);
}

Status MergeViewAssemble::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start MergeViewAssemble.");
    Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
        return status;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End MergeViewAssemble.");
    return SUCCESS;
}
} // namespace npu::tile_fwk
