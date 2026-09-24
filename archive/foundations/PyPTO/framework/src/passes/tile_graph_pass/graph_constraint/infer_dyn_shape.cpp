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
 * \file infer_dyn_shape.cpp
 * \brief
 */

#include <queue>
#include "interface/function/function.h"
#include "infer_dyn_shape.h"
#include "passes/pass_check/infer_dyn_shape_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/infer_shape_utils.h"

#define MODULE_NAME "InferDynShape"

namespace npu {
namespace tile_fwk {
Status InferDynShape::PostCheck(Function& function)
{
    InferDynShapeChecker checker;
    return checker.DoPostCheck(function);
}

Status InferDynShape::RunOnFunction(Function& function)
{
    // 遍历每一个op，调用对应的infershape函数
    // 遍历顺序，按照入度解依赖
    APASS_LOG_INFO_F(Elements::Function, "===> Start InferDynShape.");
    // InferShape 会把所有 shape 统一 normalize 成动态表达式 (即便原本是静态),
    // 之后下游 (如 OoOSchedule 的 dualdst 融合) 比较 validShape 时静态信息已丢失。
    // 在这一步之前先把 OP_L0C_COPY_UB 的 UB 输出 validShape 快照到 op 属性,
    // 供 dualdst_fuse 在融合候选判定阶段优先使用。本函数与 InferShape 完全解耦,
    // 不影响原有逻辑;若 op 输出 validShape 含动态成分则直接跳过 (回退 dyn 比较)。
    RecordStaticValidShapeOnL0CCopyUB(function);
    if (InferShapeUtils::InferShape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Dump: %s", function.Dump().c_str());
    APASS_LOG_INFO_F(Elements::Function, "===> End InferDynShape.");
    return SUCCESS;
}

// 在 InferDynShape::RunOnFunction 中, InferShape 转换前调用一次。
// 仅处理 OP_L0C_COPY_UB: 把其唯一输出 (UB tensor) 的 validShape (此时仍是静态)
// 以 vector<int64_t> 形式写到 op 属性 OpAttributeKey::staticValidShape。
// 任何一维含动态成分时整体跳过, 让下游回退到 GetDynValidShape。
void InferDynShape::RecordStaticValidShapeOnL0CCopyUB(Function& function)
{
    // 不强制排序: 仅做 op 属性快照, 与遍历顺序无关, 避免对 InferShape 的 op 顺序产生副作用。
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_L0C_COPY_UB)
            continue;
        if (op.GetOOperands().empty())
            continue;
        auto ubOut = op.GetOutputOperand(0);
        if (ubOut == nullptr)
            continue;
        const auto& valid = ubOut->GetDynValidShape();
        if (valid.empty())
            continue;
        std::vector<int64_t> staticVals;
        staticVals.reserve(valid.size());
        bool allConcrete = true;
        for (const auto& s : valid) {
            if (!s.ConcreteValid()) {
                allConcrete = false;
                break;
            }
            staticVals.push_back(s.Concrete());
        }
        if (!allConcrete)
            continue;
        op.SetAttribute(OpAttributeKey::staticValidShape, staticVals);
    }
}
} // namespace tile_fwk
} // namespace npu
