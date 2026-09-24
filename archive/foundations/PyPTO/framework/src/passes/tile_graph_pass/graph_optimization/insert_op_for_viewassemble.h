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
 * \file insert_op_for_viewassemble.h
 * \brief
 */

#pragma once

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"

#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"

#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class InsertOpForViewAssemble : public Pass {
public:
    InsertOpForViewAssemble() : Pass("InsertOpForViewAssemble") {}
    ~InsertOpForViewAssemble() override = default;

private:
    Status RunOnFunction(Function& function) override;
    Status JudgedViewAssemble(Function& function);
    Status InsertCopy(Function& function, Operation* viewOp, Operation* assOp);
    bool NeedInsertCopy(LogicalTensorPtr& assembleOut);
    void InsertViewAssemble(Function& function, Operation* viewOp, Operation* assembleOp);

    std::unordered_set<LogicalTensorPtr> assembleOutSet_;
    std::unordered_set<LogicalTensorPtr> notProcessOut_;
    std::vector<std::pair<Operation*, Operation*>> recordOpPair_;
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu
