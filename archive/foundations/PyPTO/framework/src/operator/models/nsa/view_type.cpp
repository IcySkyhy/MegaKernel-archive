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
 * \file view_type.cpp
 * \brief
 */

#include "view_type.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void ViewTypeFunc(const Tensor& x, Tensor& result, DataType dstDtype)
{
    FUNCTION("ViewTypeFunc", {x}, {result})
    {
        int m = x.GetShape()[0];
        int k = x.GetShape()[1];
        int n = x.GetShape()[2];
        int tileM = m / 4;
        SymbolicScalar mLoop = m / tileM;

        LOOP("LOOP_L0_nIdx_view_type", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mLoop, 1))
        {
            SymbolicScalar mOffset = mIdx * tileM;
            TileShape::Current().SetVecTile({tileM, k, n});
            auto xView = View(x, {tileM, k, n}, {mOffset, 0, 0});
            TileShape::Current().SetVecTile({tileM, k, n});
            auto resultView = View(xView, dstDtype);
            auto resultRes = resultView;

            if (dstDtype == DT_FP32) {
                resultRes = Add(resultView, Element(dstDtype, float(0)));
            }
            Assemble(resultRes, {mOffset, 0, 0}, result);
        }
    }
}

std::tuple<Tensor, Tensor> MyPrologQuant(const Tensor& input)
{
    config::SetSemanticLabel("Prolog-Quant");
    constexpr const float s8_max_value = 127.0f;
    constexpr const float s8_one_value = 1.0f;
    auto inputFp32 = Cast(input, DataType::DT_FP32, CAST_NONE);

    auto absRes = Abs(inputFp32);
    auto maxValue = Amax(absRes, -1, true);
    auto temp127 = Full(Element(DT_FP32, s8_max_value), DT_FP32, maxValue.GetShape());

    auto scaleQuant = Div(temp127, maxValue);
    auto outFp32 = Mul(inputFp32, scaleQuant);
    auto outInt32 = Cast(outFp32, DataType::DT_INT32, CAST_RINT);
    auto outHalf = Cast(outInt32, DataType::DT_FP16, CAST_ROUND);
    auto outInt8 = Cast(outHalf, DataType::DT_INT8, CAST_TRUNC);
    auto temp1 = Full(Element(DT_FP32, s8_one_value), DT_FP32, scaleQuant.GetShape());
    auto scaleDeQuant = Div(temp1, scaleQuant);
    return std::tie(outInt8, scaleDeQuant);
}

void ViewTypeQuantTestFunc(const Tensor& x, Tensor& result)
{
    FUNCTION("ViewTypeQuantTestFunc", {x}, {result})
    {
        int m = x.GetShape()[0];
        int k = x.GetShape()[1];
        int n = x.GetShape()[2];
        int tileM = m / 2;
        SymbolicScalar mLoop = m / tileM;
        LOOP("LOOP_L0_mIdx_view_type_quant", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mLoop, 1))
        {
            SymbolicScalar mOffset = mIdx * tileM;
            TileShape::Current().SetVecTile({tileM, k, n});
            auto xView = View(x, {tileM, k, n}, {mOffset, 0, 0});
            TileShape::Current().SetVecTile({tileM, k, n});
            auto xReshape = Reshape(xView, {tileM, k, 4, n / 4});
            TileShape::Current().SetVecTile({tileM, k, 4, n / 4});
            std::tuple<Tensor, Tensor> xQuant = MyPrologQuant(xReshape);
            auto outInt8 = std::get<0>(xQuant);
            auto scaleDeQuant = std::get<1>(xQuant);

            TileShape::Current().SetVecTile({tileM, k, 4, n / 4});
            auto outInt8Reshape = Reshape(outInt8, {tileM, k, n});
            TileShape::Current().SetVecTile({tileM, k, 8});
            auto scaleDeQuantReshape = Reshape(scaleDeQuant, {tileM, k, 4});
            TileShape::Current().SetVecTile({tileM, k, 8});
            auto scaleQuantView = View(scaleDeQuantReshape, DT_INT8);
            TileShape::Current().SetVecTile({tileM, k, 32});
            auto scaleQuantViewRes = Reshape(scaleQuantView, {tileM, k, 16});
            TileShape::Current().SetVecTile({tileM, k, n});
            auto combinedRes = Cat({outInt8Reshape, scaleQuantViewRes}, -1);
            Assemble(combinedRes, {mOffset, 0, 0}, result);
        }
    }
}

void ViewTypeDequantTestFunc(const Tensor& x, Tensor& result)
{
    FUNCTION("ViewTypeDequantTestFunc", {x}, {result})
    {
        int m = x.GetShape()[0];
        int k = x.GetShape()[1];

        int tileM = m / 8;
        SymbolicScalar mLoop = m / tileM;

        LOOP("LOOP_L0_nIdx_view_type_dequant", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mLoop, 1))
        {
            SymbolicScalar mOffset = mIdx * tileM;

            TileShape::Current().SetVecTile({tileM, k, 128});
            auto ropeView = View(x, {tileM, k, 128}, {mOffset, 0, 512});
            TileShape::Current().SetVecTile({tileM, k, 64});
            auto ropeRes = View(ropeView, DT_BF16);

            TileShape::Current().SetVecTile({tileM, k, 32});
            auto nopeView = View(x, {tileM, k, 512}, {mOffset, 0, 0});
            auto nopeCast = Cast(nopeView, DT_FP16);
            auto nopeCastCast = Cast(nopeCast, DT_FP32);
            auto nopeRehape = Reshape(nopeCastCast, {tileM, k, 4, 128});

            TileShape::Current().SetVecTile({tileM, k, 32});
            auto scaleView = View(x, {tileM, k, 16}, {mOffset, 0, 640});
            TileShape::Current().SetVecTile({tileM, k, 32});
            auto scaleRes = View(scaleView, DT_FP32);
            TileShape::Current().SetVecTile({tileM, k, 8});
            auto scaleReshape = Reshape(scaleRes, {tileM, k, 4, 1});

            TileShape::Current().SetVecTile({tileM, k, 4, 16});
            auto cacheNope = Mul(nopeRehape, scaleReshape);
            TileShape::Current().SetVecTile({tileM, k, 4, 16});
            auto cacheNopeBf16 = Cast(cacheNope, DT_BF16);
            TileShape::Current().SetVecTile({tileM, k, 4, 16});
            auto cacheNopeReshape = Reshape(cacheNopeBf16, {tileM, k, 512});
            TileShape::Current().SetVecTile({tileM, k, 64});
            auto cache = Cat({cacheNopeReshape, ropeRes}, -1);

            Assemble(cache, {mOffset, 0, 0}, result);
        }
    }
}

} // namespace npu::tile_fwk
