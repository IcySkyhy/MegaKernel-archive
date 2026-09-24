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
 * \file common_operation_eliminate.cpp
 * \brief
 */

#include "common_operation_eliminate.h"
#include "passes/pass_check/common_operation_eliminate_checker.h"
#include "passes/pass_utils/common_operation_eliminate_utils.h"

namespace npu::tile_fwk {
Status CommonOperationEliminate::RunOnFunction(Function& function)
{
    return CommonOperationEliminateUtils::EliminateCommonOperation(function);
}

Status CommonOperationEliminate::PreCheck(Function& function)
{
    CommonOperationEliminateChecker checker;
    return checker.DoPreCheck(function);
}
} // namespace npu::tile_fwk
