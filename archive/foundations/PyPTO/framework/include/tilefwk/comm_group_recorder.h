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
 * \file comm_group_recorder.h
 * \brief
 */

#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>
#include "tilefwk/error.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
namespace Distributed {

class CommGroupRecorder {
public:
    static CommGroupRecorder& GetInstance();

    CommGroupRecorder(const CommGroupRecorder&) = delete;
    CommGroupRecorder& operator=(const CommGroupRecorder&) = delete;

    // 注册组，返回对应的 groupIndex（自动去重）
    uint32_t Input(const std::string& hcclGroupName);

    // 获取所有 groupName 的列表（按 index 顺序）
    const std::vector<std::string>& Output() const;

    std::string PrintString(std::vector<std::string>& commGroups);

private:
    CommGroupRecorder() = default;
    ~CommGroupRecorder() = default;

    std::unordered_map<std::string, uint32_t> name2Index_;
    std::vector<std::string> index2Name_;
};
} // namespace Distributed
} // namespace npu::tile_fwk
