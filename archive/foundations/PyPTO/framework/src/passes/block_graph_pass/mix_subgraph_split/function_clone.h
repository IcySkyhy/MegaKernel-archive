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
 * \file function_clone.h
 * \brief
 */

#include "mix_subgraph_split_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/utils/id_gen.h"

namespace npu::tile_fwk {
class FunctionClone { // 类名
public:
    FunctionClone(Function& rootFunc_, Function* originalMixFunc_)
        : rootFunc(rootFunc_), originalMixFunc(originalMixFunc_)
    {}
    ~FunctionClone() {}

    Function* CloneFunctionByComponent(const InternalComponentInfo& component, uint64_t newProgramID, size_t idx = 0);
    void CopyInferParamIndexInfo();
    void ProcessOperations(const InternalComponentInfo& component);

    std::vector<std::shared_ptr<Operation>> programOps;
    Function& rootFunc;
    Function* originalMixFunc;
    std::shared_ptr<Function> cloneFunc;
};

} // namespace npu::tile_fwk
