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
 * \file function_clone.cpp
 * \brief
 */

#include "function_clone.h"

namespace npu {
namespace tile_fwk {
void FunctionClone::ProcessOperations(const InternalComponentInfo& component)
{
    // 获取原始Mix子图的所有op（按原始顺序）
    auto originalOps = originalMixFunc->Operations(false).DuplicatedOpList();
    // 按原始顺序筛选属于当前component的op
    for (auto* originalOp : originalOps) {
        if (originalOp->IsNOP()) {
            continue;
        }
        // 检查这个op是否属于当前component
        bool belongsToComponent = false;
        for (auto* compOp : component.operations) {
            if (compOp == originalOp) {
                belongsToComponent = true;
                break;
            }
        }
        if (belongsToComponent) {
            std::shared_ptr<Operation> opPtr = originalOp->shared_from_this();
            programOps.push_back(opPtr);
        }
    }
}

void FunctionClone::CopyInferParamIndexInfo()
{
    // 获取原Mix子图的完整符号表
    const auto& originalDynParamTable = originalMixFunc->GetDynParamTable();

    // 使用InsertDynParam方法逐个复制dynParam
    for (const auto& [dim, info] : originalDynParamTable) {
        DynParamInfo copiedInfo = info;
        cloneFunc->InsertDynParam(dim, copiedInfo);
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Copied %zu dyn param entries to function: %s", originalDynParamTable.size(),
                      cloneFunc->GetRawName().c_str());
}

Function* FunctionClone::CloneFunctionByComponent(const InternalComponentInfo& component, uint64_t newProgramID,
                                                  size_t idx)
{
    // 创建新的function名称
    std::string leafName = originalMixFunc->GetRawName() + "_leaf" + std::to_string(idx);
    APASS_LOG_DEBUG_F(Elements::Function, "Add leafFunction %s", leafName.c_str());
    // 手动创建function对象
    auto funcMagicName = leafName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
    cloneFunc = std::make_shared<Function>(Program::GetInstance(), funcMagicName, leafName, &rootFunc);
    // 设置function类型
    cloneFunc->SetFunctionType(originalMixFunc->GetFunctionType());
    cloneFunc->SetGraphType(originalMixFunc->GetGraphType());
    if (cloneFunc->GetGraphType() != GraphType::BLOCK_GRAPH) {
        APASS_LOG_ERROR_F(Elements::Function, "WRONG GRAPH TYPE FOR CLONE FUNCTION: %s", funcMagicName.c_str());
    }

    ProcessOperations(component);
    // 验证顺序正确性
    APASS_LOG_DEBUG_F(Elements::Function, "Leaf function %s has %zu ops in original order", leafName.c_str(),
                      programOps.size());
    cloneFunc->SetProgramOp(programOps);
    // 创建并设置LeafFuncAttribute
    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    // 设置aivCore属性
    leafAttr->aivCore = component.aivCore;
    cloneFunc->SetLeafFuncAttribute(leafAttr);
    cloneFunc->UpdateBelongToThis();
    cloneFunc->SetProgramId(newProgramID);
    // 复制参数配置
    cloneFunc->paramConfigs_ = originalMixFunc->paramConfigs_;
    APASS_LOG_DEBUG_F(Elements::Function, "Called UpdateBelongToThis for new function: %s", leafName.c_str());
    CopyInferParamIndexInfo();
    // 设置每个新建leaf function继承originalMixFuncisUnderDynamicFunction属性
    bool isUnderDynamicFunction = originalMixFunc->IsUnderDynamicFunction();
    APASS_LOG_DEBUG_F(Elements::Function, "Original mix function isUnderDynamicFunction: %s for programID=%d",
                      isUnderDynamicFunction ? "true" : "false", originalMixFunc->GetProgramId());
    cloneFunc->SetUnderDynamicFunction(isUnderDynamicFunction);
    APASS_LOG_DEBUG_F(Elements::Function, "Set isUnderDynamicFunction=%s for leaf function programID=%d",
                      isUnderDynamicFunction ? "true" : "false", cloneFunc->GetProgramId());

    auto* resultFunc = cloneFunc.get();
    return resultFunc;
}
} // namespace tile_fwk
} // namespace npu
