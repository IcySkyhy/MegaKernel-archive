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
 * \file op_infer_shape_impl.h
 * \brief
 */

#ifndef DEVICE_INFER_SHAPE_H
#define DEVICE_INFER_SHAPE_H
#include <unordered_map>
#include <functional>
#include "opcode.h"
#include "operation.h"

namespace npu::tile_fwk {
// params: operation*, validshape, symshapes
using FuncType = std::function<void(Operation* op, std::vector<std::vector<SymbolicScalar>>&)>;
class InferShapeRegistry {
private:
    InferShapeRegistry() = default;
    ~InferShapeRegistry() = default;

public:
    static InferShapeRegistry& GetInstance()
    {
        static InferShapeRegistry instance;
        return instance;
    }

    // 注册默认函数
    void RegisterInferShapeFunc(const Opcode opcode, FuncType func) { inferShapeFuncs_[opcode] = func; }

    // 调用场景对应的函数
    void CallInferShapeFunc(Operation* op)
    {
        const Opcode opcode = op->GetOpcode();
        std::vector<std::vector<SymbolicScalar>> outValidShapes;
        auto it = inferShapeFuncs_.find(opcode);
        if (it != inferShapeFuncs_.end()) {
            it->second(op, outValidShapes);
        } else {
            PASS_LOGW("Infer shape failed, opcode [%s] doesn't support infer shape.", op->GetOpcodeStr().c_str());
            // 如果op infershape未注册，那么validshape设置成shape
            for (auto output : op->GetOOperands()) {
                auto immShape = OpImmediate::Specified(output->GetShape());
                if (output->GetDynValidShape().empty()) {
                    std::vector<SymbolicScalar> validShape;
                    for (auto immDim : immShape) {
                        validShape.push_back(immDim.GetSpecifiedValue());
                    }
                    outValidShapes.push_back(validShape);
                } else {
                    outValidShapes.push_back(output->GetDynValidShape());
                }
            }
        }
        // 设置属性
        for (size_t i = 0; i < op->GetOOperands().size(); ++i) {
            if (op->GetOOperands()[i]->GetDynValidShape().empty() || op->GetOpcode() == Opcode::OP_COPY_OUT ||
                op->GetOpcode() == Opcode::OP_ASSEMBLE) {
                op->GetOOperands()[i]->UpdateDynValidShape(outValidShapes[i]);
            }
        }
    }

private:
    std::unordered_map<Opcode, FuncType> inferShapeFuncs_;
};

#define REGISTER_INFER_SHAPE_FUNC(OpCoreStr, OpType, FuncName)                                                \
    class OpCoreStr##Register {                                                                               \
    public:                                                                                                   \
        OpCoreStr##Register() { InferShapeRegistry::GetInstance().RegisterInferShapeFunc(OpType, FuncName); } \
    };                                                                                                        \
    static OpCoreStr##Register OpCoreStr##_register
} // namespace npu::tile_fwk
#endif // DEVICE_INFER_SHAPE_H
