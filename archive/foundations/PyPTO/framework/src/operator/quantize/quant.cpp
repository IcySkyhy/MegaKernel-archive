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
 * \file quant.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
constexpr float F_1 = 1.0;
constexpr float F_127 = 127.0;
constexpr float F_255 = 255.0;
constexpr float F_1E_12 = 1e-12f;

std::tuple<Tensor, Tensor> Quant(const Tensor& input, bool isSymmetry, bool hasSmoothFactor, const Tensor& smoothFactor)
{
    auto inputFp32 = Cast(input, DataType::DT_FP32, CAST_NONE);
    if (hasSmoothFactor) {
        inputFp32 = Mul(inputFp32, smoothFactor);
    }
    // perToken
    if (isSymmetry) {
        auto absRes = Abs(inputFp32);
        auto maxValue = Amax(absRes, -1, true);
        auto scaleQuant = ScalarDivS(maxValue, Element(DataType::DT_FP32, F_127), true);
        auto outFp32 = Mul(inputFp32, scaleQuant);
        auto outInt32 = Cast(outFp32, DataType::DT_INT32, CAST_RINT);
        auto outHalf = Cast(outInt32, DataType::DT_FP16, CAST_ROUND);
        auto outInt8 = Cast(outHalf, DataType::DT_INT8, CAST_TRUNC, SaturationMode::ON);
        auto scaleDeQuant = ScalarDivS(scaleQuant, Element(DataType::DT_FP32, F_1), true);
        return std::tie(outInt8, scaleDeQuant);
    } else {
        // 优先级低
        auto maxValue = Amax(inputFp32, -1, true);
        auto minValue = Amin(inputFp32, -1, true);
        auto scaleDeQuant = ScalarMaxS(ScalarDivS(ScalarSub(maxValue, minValue), Element(DataType::DT_FP32, F_255)),
                                       Element(DataType::DT_FP32, F_1E_12));
        auto offset = ScalarSubS(ScalarDiv(maxValue, scaleDeQuant), Element(DataType::DT_FP32, F_127), true);
        auto scaleQuant = ScalarDivS(scaleDeQuant, Element(DataType::DT_FP32, F_1), true);
        auto outFp32 = Mul(inputFp32, scaleQuant);
        auto outInt32 = Cast(outFp32, DataType::DT_INT32, CAST_RINT);
        auto outHalf = Cast(outInt32, DataType::DT_FP16, CAST_ROUND);
        auto outInt8 = Cast(outHalf, DataType::DT_INT8, CAST_TRUNC, SaturationMode::ON);
        return std::tie(outInt8, scaleDeQuant);
    }
}
} // namespace npu::tile_fwk

namespace npu::tile_fwk::Matrix {
Tensor QuantMM(const Tensor& operand1, const Tensor& operand2, const Tensor& dequantScaleW)
{
    auto quantA = Quant(operand1);
    auto quantizedA = std::get<0>(quantA);
    auto dequantScaleA = std::get<1>(quantA);
    Tensor res;
    if (operand1.GetShape().size() == NUM_VALUE_2) {
        res = Matmul(DataType::DT_INT32, quantizedA, operand2, false, false);
    } else if (operand1.GetShape().size() == NUM_VALUE_3) {
        res = BatchMatmul(DataType::DT_INT32, quantizedA, operand2);
    } else {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, operand1.GetShape().size() <= NUM_VALUE_3)
            << "QuantMM only supports 2D or 3D tensors.";
    }
    res = Cast(res, DataType::DT_FP32);
    res = Mul(res, dequantScaleA);
    res = Mul(res, dequantScaleW);
    res = Cast(res, DataType::DT_BF16, CAST_RINT);
    return res;
}
} // namespace npu::tile_fwk::Matrix
