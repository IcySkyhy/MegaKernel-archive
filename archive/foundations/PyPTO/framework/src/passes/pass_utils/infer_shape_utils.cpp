/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_shape_utils.cpp
 * \brief 公共的 InferShape 方法实现
 */

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "infer_shape_utils.h"

namespace npu {
namespace tile_fwk {
Status InferShapeUtils::InferShape(Function& function, const std::vector<Operation*>& targetOps)
{
    auto sortedOperations = function.Operations(true, SortOperationsMode::LIGHTWEIGHT);
    if (targetOps.empty()) {
        for (auto& op : sortedOperations) {
            InferShapeRegistry::GetInstance().CallInferShapeFunc(&op);
        }
        return SUCCESS;
    }

    std::unordered_map<const Operation*, size_t> opToIndex;
    opToIndex.reserve(sortedOperations.size());
    for (size_t index = 0; index < sortedOperations.size(); ++index) {
        opToIndex.emplace(&sortedOperations[index], index);
    }

    std::vector<Operation*> opList;
    std::unordered_set<Operation*> targetOpSet;
    opList.reserve(targetOps.size());
    for (const auto op : targetOps) {
        if (op != nullptr && opToIndex.find(op) != opToIndex.end() && targetOpSet.insert(op).second) {
            opList.push_back(op);
        }
    }

    std::sort(opList.begin(), opList.end(), [&opToIndex](const Operation* lhs, const Operation* rhs) {
        return opToIndex.at(lhs) < opToIndex.at(rhs);
    });
    for (auto* op : opList) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(op);
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
