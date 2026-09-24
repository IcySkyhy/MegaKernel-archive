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
 * \file machine_task.h
 * \brief
 */

#pragma once

#ifndef MACHINE_TASK_H
#define MACHINE_TASK_H

#include <list>
#include <cstdint>
#include <unistd.h>
#include <memory>
#include <iostream>
#include "interface/function/function.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
inline int64_t CalcShapeSizeFunc(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto& i : shape) {
        size *= i;
    }
    return size;
}

class MachineTask {
public:
    MachineTask(uint64_t taskId, Function* function) : taskId_(taskId), function_(function) {}

    uint64_t GetTaskId() const { return taskId_; }
    Function* GetFunction() const { return function_; }
    void SetFunction(Function* func) { function_ = func; }
    void SetError(std::string msg) { error = std::move(msg); }
    const std::string& Error() { return error; }
    int GetFunctionIndex() const { return function_index_; }
    void SetFunctionIndex(int idx) { function_index_ = idx; }

private:
    uint64_t taskId_;
    Function* function_;
    std::string error;
    int function_index_{0}; // 1-based index for compiler monitor progress (k/N)
};
} // namespace npu::tile_fwk
#endif // MACHINE_TASK_H
