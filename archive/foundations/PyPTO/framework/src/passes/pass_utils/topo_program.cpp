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
 * \file topo_program.cpp
 * \brief
 */

#include "topo_program.h"

#include "interface/operation/op_infer_shape_impl.h"

namespace npu {
namespace tile_fwk {
bool NeedInferShape(const Operation* op)
{
    if (op->GetOOperands().empty()) {
        return false;
    }
    if (op->GetOpcode() != Opcode::OP_ASSEMBLE) {
        for (const auto& output : op->GetOOperands()) {
            if (output->GetDynValidShape().empty()) {
                return true;
            }
        }
        return false;
    }
    return true;
}

void TopoProgramUtils::TopoProgram(const std::vector<Operation*>& opList, bool isParamIndex)
{
    for (auto* op : opList) {
        if (isParamIndex) {
            if (NeedInferShape(op)) {
                InferShapeRegistry::GetInstance().CallInferShapeFunc(op);
            }
            continue;
        }
        InferShapeRegistry::GetInstance().CallInferShapeFunc(op);
    }
}
} // namespace tile_fwk
} // namespace npu
