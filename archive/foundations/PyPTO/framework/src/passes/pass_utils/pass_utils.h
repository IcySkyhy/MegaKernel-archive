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
 * \file pass_utils.h
 * \brief
 */

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/function/function.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

static const std::string RMW_MODE_ATTR_ADD = OP_ATTR_PREFIX + "atomic_add";
static const std::string RMW_MODE_ATTR_MIN = OP_ATTR_PREFIX + "atomic_min";
static const std::string RMW_MODE_ATTR_MAX = OP_ATTR_PREFIX + "atomic_max";

inline std::map<std::string, std::set<int>> BuildLabelToColorsMap(const OperationsViewer& opOriList)
{
    std::map<std::string, std::set<int>> labelToColors;
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        const std::string& label = opOriList[i].GetSemanticLabelStr();
        if (!label.empty()) {
            labelToColors[label].insert(opOriList[i].GetSubgraphID());
        }
    }
    return labelToColors;
}

class FunctionUtils {
public:
    static void RelinkOperationInput(Operation* op, const size_t inputIndex, const Operation* targetOp,
                                     const size_t outputIndex);

    static bool IsContinuous(const std::vector<std::shared_ptr<LogicalTensor>>& tensors);

    static NodeType GetNodeType(const LogicalTensor& tensor, const Function& function);

    static Status InferOutcastWriteConflict(Function& function);

    /// Warns when assemble outputs may lose non-immediate dynValidShape propagation.
    /// This scans assemble ops in the function and only emits warnings; it does not modify the graph.
    static void WarnAssembleDynValidShapeRisk(Function& function);
};

class CommonUtils {
public:
    template <typename Container>
    static std::string ContainerToStr(const Container& container, const std::string& delimiter = ", ")
    {
        if (container.empty()) {
            return "{}";
        }
        std::ostringstream oss;
        oss << "{";
        auto it = container.begin();
        oss << *it;
        std::for_each(std::next(it), container.end(),
                      [&oss, &delimiter](const auto& elem) { oss << delimiter << elem; });
        oss << "}";
        return oss.str();
    }

    // 判断 Tensor 的 shape 是否存在-1
    static bool ContainsNegativeOne(const Shape& shape)
    {
        return std::any_of(shape.begin(), shape.end(), [](int64_t val) { return val == -1; });
    }

    // Number of Elements, 用来计算给定（tensor的）shape的总元素数量
    static int64_t Numel(const Shape& shape)
    {
        if (shape.empty())
            return 0;
        int64_t numel = 1;
        for (int64_t num : shape) {
            numel *= num;
        }
        return numel;
    }

    static std::unordered_map<MemoryType, int64_t> GetLocalMemorySize();

    // 安全计算 shape 的乘积，检测溢出
    // 返回 pair<结果, 是否溢出>
    // 如果溢出则返回 <0, true>
    // 如果存在动态轴 -1，则返回 <-1, false> 表示无法精确计算但不溢出；
    static std::pair<int64_t, bool> SafeMultiplyShape(const Shape& shape)
    {
        if (shape.empty()) {
            return {0, false};
        }
        int64_t result = 1;
        for (int64_t dim : shape) {
            if (dim == -1) {
                return {-1, false};
            }
            if (result != 0 && dim != 0) {
                if (result > INT64_MAX / dim) {
                    return {0, true};
                }
            }
            result *= dim;
        }
        return {result, false};
    }

    static int GetTensorSubgraphID(const LogicalTensorPtr& tensor);
    static int GetTensorSubgraphID(const LogicalTensor* tensor);

    static std::vector<SymbolicScalar> CreateConstIntVector(const std::vector<int64_t>& values);
};
} // namespace npu::tile_fwk
