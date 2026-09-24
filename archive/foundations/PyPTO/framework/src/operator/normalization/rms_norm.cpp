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
 * \file rms_norm.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Tensor RmsNorm(const Tensor& operand)
{
    constexpr float epsilon = 1e-6f;

    auto fp32Operand = Cast(operand, DataType::DT_FP32);
    // y = x^2 / n
    auto y = Mul(fp32Operand, fp32Operand);
    y = Mul(y, Element(DataType::DT_FP32, 1.0f / operand.GetShape()[operand.GetShape().size() - 1]));

    // ReduceSum(x^2 / n) + Eps
    y = Sum(y, -1, true);
    y = Add(y, Element(DataType::DT_FP32, epsilon));

    // sqrt rstd
    y = Sqrt(y);
    Element src(DataType::DT_FP32, 1.0f);
    auto ones = Full(src, DT_FP32, y.GetShape());
    y = Div(ones, y);
    return Cast(Mul(fp32Operand, y), operand.GetStorage()->Datatype());
}

Tensor RmsNorm(const Tensor& operand, const Tensor& gamma, float epsilon)
{
    auto fp32Operand = Cast(operand, DataType::DT_FP32);
    int size = operand.GetShape().size();
    std::vector<int64_t> shape(size, 1);
    shape[size - 1] = gamma.GetShape()[0];
    auto gammaCast = Reshape(gamma, shape);
    auto gammaOperand = Cast(gammaCast, DataType::DT_FP32);
    // y = x^2 / n
    auto y = Mul(fp32Operand, fp32Operand);
    y = Mul(y, Element(DataType::DT_FP32, 1.0f / operand.GetShape()[operand.GetShape().size() - 1]));

    // ReduceSum(x^2 / n) + Eps
    y = Sum(y, -1, true);
    y = Add(y, Element(DataType::DT_FP32, epsilon));

    // sqrt rstd
    y = Sqrt(y);
    Element src(DataType::DT_FP32, 1.0f);
    auto ones = Full(src, DT_FP32, y.GetShape());
    y = Div(ones, y);
    y = Mul(fp32Operand, y);
    y = Mul(gammaOperand, y);
    y = Cast(y, operand.GetStorage()->Datatype());

    return y;
}

} // namespace npu::tile_fwk
