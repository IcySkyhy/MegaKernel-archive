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
 * \file tensor_transformation.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
void Expand(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
            const std::vector<LogicalTensorPtr>& other, const LogicalTensorPtr& result);
void ExpandWithResultValidShape(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                                const LogicalTensorPtr& result, const std::vector<SymbolicScalar> resultValidShape);
Tensor TensorFullOperation(Function& function, const Element& src, const SymbolicScalar& dynValue, DataType dtype,
                           const std::vector<int64_t>& dstShape, const std::vector<SymbolicScalar>& validShape);

enum class CastOpType {
    CAST,
    // FLOOR,
    // ROUND,
};

template <CastOpType T>
Opcode GetCastOpName()
{
#define CASE(X)         \
    case CastOpType::X: \
        return Opcode::OP_##X
    switch (T) {
        CASE(CAST);
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown cast op type";
    }
#undef CASE
}

template <CastOpType T>
LogicalTensorPtr TensorCastOperation(Function& function, LogicalTensorPtr self, const DataType& dstDataType,
                                     const CastMode& mode = CAST_NONE,
                                     const SaturationMode& satmode = SaturationMode::OFF)
{
    auto result = std::make_shared<LogicalTensor>(function, dstDataType, self->shape, self->dynValidShape_);
    auto& op = function.AddOperation(GetCastOpName<T>(), {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
    op.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(satmode));
    return result;
}

} // namespace npu::tile_fwk
