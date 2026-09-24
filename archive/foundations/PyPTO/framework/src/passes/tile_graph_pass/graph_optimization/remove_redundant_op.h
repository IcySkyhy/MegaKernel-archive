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
 * \file remove_redundant_op.h
 * \brief
 */

#ifndef REMOVE_REDUNDANT_OP_H
#define REMOVE_REDUNDANT_OP_H

#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"

#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"

#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
const std::unordered_set<Opcode> matchOpcodeWithDynshape = {Opcode::OP_VIEW, Opcode::OP_SLICE, Opcode::OP_EXPAND};
const std::unordered_set<Opcode> matchOpcodeWithoutDynshape = {Opcode::OP_REGISTER_COPY, Opcode::OP_ASSEMBLE,
                                                               Opcode::OP_CONTRACT};
class RemoveRedundantOp : public Pass {
public:
    RemoveRedundantOp() : Pass("RemoveRedundantOp") {}
    ~RemoveRedundantOp() override = default;

private:
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status RemoveDummyOp(Function& function);
    Status ProcessViewAssemble(Function& function);
    Status ProcessReshape(Function& function);
    Status RemoveDummyOps(Function& function);
    bool ProcessRedundantOpWithDynShape(Operation& op) const;
    bool ProcessRedundantOpWithoutDynShape(Operation& op) const;

    bool operationUpdated;
    uint32_t iterTime;
    std::vector<Operation*> newOps_;
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu
#endif // REMOVE_REDUNDANT_OP_H
