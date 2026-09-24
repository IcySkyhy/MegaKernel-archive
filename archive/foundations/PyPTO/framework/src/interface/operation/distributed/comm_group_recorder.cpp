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

#include "tilefwk/comm_group_recorder.h"

namespace npu::tile_fwk {
namespace Distributed {
CommGroupRecorder& CommGroupRecorder::GetInstance()
{
    static CommGroupRecorder instance_;
    return instance_;
}

uint32_t CommGroupRecorder::Input(const std::string& hcclGroupName)
{
    auto it = name2Index_.find(hcclGroupName);
    if (it != name2Index_.end()) {
        return it->second; // 已存在，返回现有 index
    }

    // 新组：记录映射关系
    uint32_t newIndex = index2Name_.size();
    ASSERT(DistributedErrorCode::INVALID_GROUP_INDEX, newIndex < HCCL_GROUP_NUM)
        << "Failed to create communication group '" << hcclGroupName << "': maximum number of groups ("
        << HCCL_GROUP_NUM << ") exceeded. Current index: " << newIndex
        << ", existing groups: " << PrintString(index2Name_);

    index2Name_.push_back(hcclGroupName);
    name2Index_[hcclGroupName] = newIndex;
    return newIndex;
}

// 获取所有 groupName 的列表（按 index 顺序）
const std::vector<std::string>& CommGroupRecorder::Output() const { return index2Name_; }

std::string CommGroupRecorder::PrintString(std::vector<std::string>& commGroups)
{
    std::ostringstream oss;
    oss << "distributed comm groups: [";
    for (size_t i = 0; i < commGroups.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << commGroups[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace Distributed
} // namespace npu::tile_fwk
