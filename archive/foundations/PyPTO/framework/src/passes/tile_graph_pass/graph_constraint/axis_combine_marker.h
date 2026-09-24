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
 * \file axis_combine_marker.h
 * \brief
 */

#ifndef AXIS_COMBINE_MARKER_H
#define AXIS_COMBINE_MARKER_H

#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
namespace npu::tile_fwk {
enum class AxisReorderStatus {
    UNKNOWN = 0, // 不确定合轴优化
    ENABLE,      // 明确可以支持合轴优化
    DISABLE      // 尾轴为1，但是不支持合轴优化的场景
};
class AxisCombineMarker {
public:
    AxisCombineMarker() = default;
    ~AxisCombineMarker() = default;
    bool IsTensorEnableAxisCombine(LogicalTensorPtr tensor) const;
    void Run(Function& function);

private:
    void Init(Function& function);
    std::vector<Operation*> opList_;
    void UpdateOpACEnableForward(size_t opIdx);
    void ForwardVisit();
    void DisableNoneWhiteListTensor(Operation* op);
    // 并查集相关方法
    LogicalTensorPtr Find(LogicalTensorPtr x);
    void Union(LogicalTensorPtr x, LogicalTensorPtr y);
    bool IsEligibleUnionOutput(Operation* op, OpCalcType calcType, LogicalTensorPtr outputTensor) const;
    void UnionOutputWithInputs(Operation* op, OpCalcType calcType, LogicalTensorPtr outputTensor);
    void UnionMultiOutputTensors(Operation* op);
    void BuildUnionFind();
    void ResolveGroupStatus();
    std::unordered_map<LogicalTensorPtr, AxisReorderStatus> tensorStatus_;
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> parent_;
    std::unordered_map<LogicalTensorPtr, int> rank_;
};
} // namespace npu::tile_fwk
#endif
