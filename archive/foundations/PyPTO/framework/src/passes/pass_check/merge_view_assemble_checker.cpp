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
 * \file merge_veiw_assemble_checker.cpp
 * \brief
 */

#include "merge_view_assemble_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "MergeViewAssemble"

namespace npu {
namespace tile_fwk {
Status MergeViewAssembleChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PreCheck for MergeViewAssemble.");
    return CheckCompleteness(function);
}
} // namespace tile_fwk
} // namespace npu
