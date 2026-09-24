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
 * \file permute.h
 * \brief Permute operation header file
 */

#pragma once
#include <string>
#include <vector>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

void PermuteOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                  const std::vector<LogicalTensorPtr>& oOperand);
std::vector<int64_t> PermuteResultShape(const std::vector<int64_t>& inputShape, const std::vector<int>& perm);
bool IsIdentityPermutation(const std::vector<int>& perm);
void NormalizePermutation(std::vector<int>& perm, int shapeSize);
void ValidatePermutation(const std::vector<int>& perm, int shapeSize);
Tensor TensorPermuteOperation(Function& function, LogicalTensorPtr self, const std::vector<int>& perm);
Tensor TensorElementPermuteOperation(Function& function, LogicalTensorPtr self, const std::vector<int>& perm);

} // namespace npu::tile_fwk
