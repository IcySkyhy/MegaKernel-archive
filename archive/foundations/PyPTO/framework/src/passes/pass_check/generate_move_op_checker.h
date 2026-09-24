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
 * \file generate_move_op_checker.h
 * \brief
 */

#ifndef GENERATE_MOVE_OP_CHECKER_H
#define GENERATE_MOVE_OP_CHECKER_H

#include "checker.h"
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class GenerateMoveOpChecker : Checker {
public:
    Status DoPreCheck(Function& function) override;
    Status DoPostCheck(Function& function) override;

private:
    bool CheckViewOutTensorMemType(const Operation& op) const;
    bool ValidAssembleOp(const Operation& op) const;
    bool ValidConvertOp(const Operation& op) const;
    bool ValidViewOp(const Operation& op) const;
};
} // namespace tile_fwk
} // namespace npu
#endif // GENERATE_MOVE_OP_CHECKER_H
