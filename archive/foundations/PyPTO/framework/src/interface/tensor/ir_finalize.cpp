/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir_finalize.h"

#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {

namespace {

void CollectCompileQueueFunctions(Function* func, std::vector<Function*>& out, std::unordered_set<Function*>& visited)
{
    if (func == nullptr || !visited.insert(func).second) {
        return;
    }

    if (func->HasCallOperation()) {
        for (auto* callee : func->GetCalleeFunctionList()) {
            CollectCompileQueueFunctions(callee, out, visited);
        }
    }
    if (!func->Operations(false).IsEmpty()) {
        out.push_back(func);
    }
}

void SubmitCompileTask(Function* dynFunc)
{
    auto& program = Program::GetInstance();
    program.functionSequence_.clear();

    std::vector<Function*> compileFuncs;
    std::unordered_set<Function*> visited;
    CollectCompileQueueFunctions(dynFunc, compileFuncs, visited);

    FE_LOGI("FinalizeDynamicFunction SubmitCompileTask queue_size=%zu", compileFuncs.size());
    for (auto* func : compileFuncs) {
        FE_LOGI("FinalizeDynamicFunction enqueue compile task: %s ops=%zu", func->GetMagicName().c_str(),
                func->Operations(false).size());
        program.RefillCompileQueue(func);
    }

    if (compileFuncs.empty()) {
        FE_LOGW("FinalizeDynamicFunction: no leaf functions to compile, skip UpdateCompileTask");
        return;
    }

    program.UpdateCompileTask();
}

} // namespace

void FinalizeDynamicFunction(Function* dynFunc)
{
    if (dynFunc == nullptr) {
        FE_LOGW("FinalizeDynamicFunction: dynFunc is null");
        return;
    }

    if (config::GetVerifyOption<bool>(KEY_ENABLE_PASS_VERIFY)) {
        Program::GetInstance().VerifyTensorGraph();
    }
    SubmitCompileTask(dynFunc);
}

} // namespace npu::tile_fwk
