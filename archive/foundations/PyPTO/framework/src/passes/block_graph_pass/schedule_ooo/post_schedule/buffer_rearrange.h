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
 * \file buffer_rearrange.h
 * \brief
 */

#ifndef PASS_BUFFER_REARRANGE_H
#define PASS_BUFFER_REARRANGE_H
#include <vector>
#include <climits>
#include <unordered_map>
#include "interface/operation/operation.h"
#include "passes/block_graph_pass/schedule_ooo/common/buffer_pool.h"
#include "passes/pass_log/pass_log.h"
#include <iostream>
using namespace std;

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif

#define MODULE_NAME "OoOSchedule"

namespace npu {
namespace tile_fwk {

struct RearrangeScheme {
    size_t cost = INT_MAX;
    size_t start;            //  整理区间起始地址
    size_t end;              //  整理区间结束地址
    std::vector<int> memIds; //  需要整理的内存id列表
    std::unordered_map<int, size_t> moveFrom;
    std::unordered_map<int, size_t> moveTo;
    std::unordered_map<int, size_t> memSizeMap;
    std::vector<std::pair<int, size_t>> orderedMoveTo;

    void PrintScheme()
    {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Memory Rearange Scheme,  Span : [%lu, %lu], Cost : %lu", start, end, cost);
        for (auto memId : memIds) {
            if (moveFrom[memId] != moveTo[memId]) {
                APASS_LOG_DEBUG_F(Elements::Tensor,
                                  "    |--- MemId : %d, Ori Span : [%lu, %lu], Size : %lu, Move From %lu to %lu", memId,
                                  moveFrom[memId], moveFrom[memId] + memSizeMap[memId], memSizeMap[memId],
                                  moveFrom[memId], moveTo[memId]);
            } else {
                APASS_LOG_DEBUG_F(Elements::Tensor, "    |--- MemId : %d, Ori Span : [%lu, %lu], Size : %lu", memId,
                                  moveFrom[memId], moveFrom[memId] + memSizeMap[memId], memSizeMap[memId]);
            }
        }
    }
};

RearrangeScheme GetRearrangeScheme(BufferPool& bufferManager, size_t sizeNeeded);
} // namespace tile_fwk
} // namespace npu

#endif // PASS_BUFFER_REARRANGE_H
