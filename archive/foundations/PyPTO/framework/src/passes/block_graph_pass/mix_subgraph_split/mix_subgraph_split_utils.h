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
 * \file mix_dependency_analyzer.h
 * \brief 用于提供所需的接口
 */

#ifndef MIX_SUBGRAPH_SPLIT_UTILS_H
#define MIX_SUBGRAPH_SPLIT_UTILS_H

#include <set>
#include <vector>
#include <unordered_map>
#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "MixSubgraphSplit"

namespace npu {
namespace tile_fwk {
enum class ComponentType {
    UNKNOWN = 0,
    C_SCOPE = 1, // C类型scope
    V_SCOPE = 2, // V类型scope
};

int GetStartIndex(const std::vector<Operation*>& opList, Operation* startOp);

// Mix子图内部独立子图的信息
struct InternalComponentInfo {
    int internalSubgraphID;             // mix子图内部的子图ID(cube/vector组件ID)
    std::vector<Operation*> operations; // 包含的op
    std::string suffix;
    AIVCore aivCore;
    ComponentType componentType;

    InternalComponentInfo(int id, const std::string& suf = "")
        : internalSubgraphID(id), suffix(suf), aivCore(AIVCore::UNSPECIFIED), componentType(ComponentType::UNKNOWN)
    {}

    InternalComponentInfo(int id, const std::string& suf, AIVCore aiv)
        : internalSubgraphID(id), suffix(suf), aivCore(aiv), componentType(ComponentType::UNKNOWN)
    {}

    InternalComponentInfo(int id, const std::string& suf, AIVCore aiv, ComponentType compType)
        : internalSubgraphID(id), suffix(suf), aivCore(aiv), componentType(compType)
    {}
};

// 内部依赖信息结构
struct InternalDependencyInfo {
    int srcComp;                  // 源scope索引
    int dstComp;                  // 目标scope索引
    LogicalTensorPtr dummyTensor; // 用于表示依赖的dummy tensor
    uint64_t dummyTensorMagic;    // dummy tensor的magic值
    bool isSameType;              // 是否是同类型scope依赖
    ComponentType compType;       // scope类型（C或V）

    InternalDependencyInfo(int src, int dst, ComponentType type)
        : srcComp(src), dstComp(dst), dummyTensor(nullptr), dummyTensorMagic(0), isSameType(true), compType(type)
    {}
};
} // namespace tile_fwk
} // namespace npu

#endif // MIX_SUBGRAPH_SPLIT_UTILS_H
