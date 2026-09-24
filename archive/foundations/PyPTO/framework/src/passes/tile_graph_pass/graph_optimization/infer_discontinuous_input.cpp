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
 * \file infer_memory_conflict.cpp
 * \brief
 */

#include "infer_discontinuous_input.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_check/infer_discontinuous_input_checker.h"
#include "passes/pass_utils/infer_discontinuous_input_utils.h"

namespace npu {
namespace tile_fwk {
Status InferDiscontinuousInput::RunOnFunction(Function& function)
{
    InferDiscontinuousInputUtils utils;
    return utils.Process(function, true);
}

Status InferDiscontinuousInput::PostCheck(Function& function)
{
    InferDisContinuousInputChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace tile_fwk
} // namespace npu
