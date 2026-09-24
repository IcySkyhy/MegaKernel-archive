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
 * \file memory_path_utils.h
 * \brief
 */

#ifndef TILE_FWK_MEMORY_PATH_UTILS_H
#define TILE_FWK_MEMORY_PATH_UTILS_H

#include <functional>

#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk {

class MemoryPathUtils {
public:
    using OutputRequirementResolver = std::function<MemoryType(const LogicalTensorPtr&)>;

    static bool IsSpecialDirectMemoryPath(MemoryType from, MemoryType to);

    static bool IsDifferentKnownRequirement(MemoryType requirement, MemoryType targetType);

    static bool ShouldUseDdrForSpecialPath(bool hasParallelDifferentRequirement, MemoryType from, MemoryType to);

    // 在判断直接搬运路径冲突前，解析 view/assemble 语义 op 背后的有效消费者需求。
    // OP_VIEW/OP_SLICE: 通过 ViewOpAttribute.GetTo() 获取目的地类型。
    // OP_ASSEMBLE/OP_CONTRACT: 若处于特殊直连路径且 output 有下游消费者，推断 targetType；
    //                          无下游消费者（如 outcast）则保留 output MemoryTypeOriginal。
    // resolver 的语义与阶段相关：AssignMemoryType 阶段可递归穿透 view 消费者，
    // ConvertInserter 阶段应基于 tensorTobeMap 中已经规划好的直接需求判断。
    static MemoryType ResolveEffectiveConsumerRequirement(Operation* consumerOp, MemoryType directRequirement,
                                                          MemoryType targetType,
                                                          const OutputRequirementResolver& resolveOutputRequirement);
};

} // namespace npu::tile_fwk

#endif // TILE_FWK_MEMORY_PATH_UTILS_H
