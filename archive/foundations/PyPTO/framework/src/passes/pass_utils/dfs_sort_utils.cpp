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
 * \file dfs_sort_utils.cpp
 * \brief
 */

#include "passes/pass_utils/dfs_sort_utils.h"

namespace npu {
namespace tile_fwk {
void DFSVisit(int preColor, const std::vector<std::vector<int>>& inColor, const std::vector<std::vector<int>>& outColor,
              std::unordered_set<int>& visited, std::unordered_map<int, int>& dfsColorOrder)
{
    std::vector<int> visitStack{preColor};
    std::unordered_set<int> pushedStackColorSet;
    while (visitStack.size() > 0) {
        int currColor = visitStack.back();
        if (visited.count(currColor) > 0) {
            visitStack.pop_back();
            continue;
        }
        bool allVisited = true;
        for (int pred : inColor[currColor]) {
            if (visited.count(pred) == 0) {
                visitStack.push_back(pred);
                pushedStackColorSet.insert(pred);
                allVisited = false;
                break;
            }
        }
        if (!allVisited) {
            continue;
        }
        int currColorNum = dfsColorOrder.size();
        dfsColorOrder[currColor] = currColorNum;
        visited.insert(currColor);
        visitStack.pop_back();
        auto s = outColor[currColor];
        for (auto it = s.rbegin(); it != s.rend(); ++it) {
            auto succ = *it;
            if (visited.count(succ) == 0 && pushedStackColorSet.count(succ) == 0) {
                visitStack.push_back(succ);
                pushedStackColorSet.insert(succ);
            }
        }
    }
}

void DFSSortUtils::DFSSortColor(const int color, const std::vector<std::vector<int>>& inColor,
                                const std::vector<std::vector<int>>& outColor,
                                std::unordered_map<int, int>& dfsColorOrder)
{
    std::unordered_set<int> visited;
    for (int preColor = 0; preColor < color; preColor++) {
        if (visited.count(preColor) > 0) {
            continue;
        }
        DFSVisit(preColor, inColor, outColor, visited, dfsColorOrder);
    }
}

} // namespace tile_fwk
} // namespace npu
