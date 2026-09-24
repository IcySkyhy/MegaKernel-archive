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
 * \file sigmoid.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
constexpr float F_1 = 1.0;
constexpr float F_NEGA_1 = -1.0;

Tensor Sigmoid(Tensor& input)
{
    // 1/(1+exp(-x))
    auto dtype = input.GetStorage()->Datatype();
    if (dtype != DT_FP32) {
        input = Cast(input, DataType::DT_FP32);
    }
    auto expRes = Exp(Mul(input, Element(DataType::DT_FP32, F_NEGA_1)));
    auto res = Add(expRes, Element(DataType::DT_FP32, F_1));
    Element src(DataType::DT_FP32, 1.0f);
    auto ones = Full(src, DataType::DT_FP32, res.GetShape());
    res = Div(ones, res);
    if (dtype != DT_FP32) {
        res = Cast(res, dtype);
    }
    return res;
}

} // namespace npu::tile_fwk
