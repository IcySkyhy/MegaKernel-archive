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
 * \file axis_combine.h
 * \brief
 */

#ifndef AXIS_COMBINE_H
#define AXIS_COMBINE_H

#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_utils.h"
#include "axis_combine_marker.h"

namespace npu::tile_fwk {

class AxisCombine : public Pass {
public:
    AxisCombine() : Pass("AxisCombine") {}
    ~AxisCombine() override = default;

    Status RunOnFunction(Function& function) override;
    Status Process(Function& function);

private:
    Status AlignBroadCastOpInputs(Function& function, Operation& op);
    LogicalTensorPtr CreateAlignedTensor(const LogicalTensorPtr& srcTensor, const std::vector<int64_t>& alignedShape);
    bool enableBrcb_{true};
    AxisCombineMarker axisCombineMarker;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // AXIS_COMBINE_H
