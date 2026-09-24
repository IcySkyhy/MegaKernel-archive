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
 * \file calc.h
 * \brief
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tilefwk/data_type.h"
#include "tilefwk/element.h"
#include "raw_tensor_data.h"
#include "tilefwk/error_code.h"
#include "calculator/calc_api.h"

namespace npu::tile_fwk::calc {

CalcOps* GetCalcOps();

inline bool IsVerifyEnabled() { return GetCalcOps() != nullptr; }

inline TensorData Trans(LogicalTensorDataPtr data)
{
    TensorData calcData;
    if (data != nullptr) {
        RawTensorDataPtr raw = data->GetData();
        calcData.dataPtr = raw->data();
        calcData.rawShape = raw->GetShape();
        calcData.shape = data->GetShape();
        calcData.stride = raw->GetStride();
        calcData.storageOffset = data->GetStorageOffset();
        calcData.dtype = raw->GetDataType();
        calcData.isAxisCombine = data->IsAxisCombine();
    }
    return calcData;
}

inline std::vector<TensorData> TransVec(std::vector<LogicalTensorDataPtr> datas)
{
    std::vector<TensorData> result;
    result.reserve(datas.size());

    for (const auto& data : datas) {
        result.push_back(Trans(data));
    }
    return result;
}

inline void Random(LogicalTensorDataPtr out) { GetCalcOps()->Random(Trans(out)); }
inline bool AllClose(LogicalTensorDataPtr self, LogicalTensorDataPtr other, double atol = 1e-8, double rtol = 1e-5)
{
    return GetCalcOps()->AllClose(Trans(self), Trans(other), atol, rtol);
}
inline void Cast(LogicalTensorDataPtr out, LogicalTensorDataPtr self, CastMode mode = CAST_NONE)
{
    GetCalcOps()->Cast(Trans(out), Trans(self), mode);
}
inline void QuantPreCompute(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr scalePtr,
                            uint64_t scale, int relu)
{
    CalcOps* ops = GetCalcOps();
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ops != nullptr);
    if (scalePtr == nullptr) {
        ops->QuantPreCompute(Trans(out), Trans(self), nullptr, scale, relu);
    } else {
        auto scaleData = Trans(scalePtr);
        ops->QuantPreCompute(Trans(out), Trans(self), &scaleData, scale, relu);
    }
}
inline void Sin(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Sin(Trans(out), Trans(self)); }
inline void Cos(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Cos(Trans(out), Trans(self)); }
inline void Erf(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Erf(Trans(out), Trans(self)); }
inline void Exp(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Exp(Trans(out), Trans(self)); }
inline void Exp2(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Exp2(Trans(out), Trans(self)); }
inline void Expm1(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Expm1(Trans(out), Trans(self)); }
inline void Sinh(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Sinh(Trans(out), Trans(self)); }
inline void Cosh(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Cosh(Trans(out), Trans(self)); }
inline void Erfc(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Erfc(Trans(out), Trans(self)); }
inline void Asin(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Asin(Trans(out), Trans(self)); }
inline void Acos(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Acos(Trans(out), Trans(self)); }
inline void ASinh(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->ASinh(Trans(out), Trans(self)); }
inline void ACosh(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->ACosh(Trans(out), Trans(self)); }
inline void Atanh(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Atanh(Trans(out), Trans(self)); }
inline void Neg(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Neg(Trans(out), Trans(self)); }
inline void Pack(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Pack(Trans(out), Trans(self)); }
inline void UnPack(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->UnPack(Trans(out), Trans(self));
}
inline void Round(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int decimals)
{
    GetCalcOps()->Round(Trans(out), Trans(self), decimals);
}
inline void Rsqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Rsqrt(Trans(out), Trans(self)); }
inline void Sqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Sqrt(Trans(out), Trans(self)); }
inline void Ceil(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Ceil(Trans(out), Trans(self)); }
inline void Floor(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Floor(Trans(out), Trans(self)); }
inline void Trunc(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Trunc(Trans(out), Trans(self)); }
inline void Sign(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Sign(Trans(out), Trans(self)); }
inline void Signbit(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->Signbit(Trans(out), Trans(self));
}
inline void Tanh(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Tanh(Trans(out), Trans(self)); }
inline void Tan(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Tan(Trans(out), Trans(self)); }
inline void Reciprocal(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->Reciprocal(Trans(out), Trans(self));
}
inline void Relu(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Relu(Trans(out), Trans(self)); }
inline void Pad(LogicalTensorDataPtr out, LogicalTensorDataPtr input, const Element& padValue)
{
    GetCalcOps()->Pad(Trans(out), Trans(input), padValue);
}
inline void FillPad(LogicalTensorDataPtr out, LogicalTensorDataPtr input, const Element& padValue)
{
    GetCalcOps()->FillPad(Trans(out), Trans(input), padValue);
}
inline void BitwiseNot(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->BitwiseNot(Trans(out), Trans(self));
}
inline void Abs(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Abs(Trans(out), Trans(self)); }
inline void Brcb(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Brcb(Trans(out), Trans(self)); }
inline void WhereTT(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input,
                    LogicalTensorDataPtr other)
{
    GetCalcOps()->WhereTT(Trans(out), Trans(condition), Trans(input), Trans(other));
}
inline void WhereTS(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, LogicalTensorDataPtr input,
                    const Element& other)
{
    GetCalcOps()->WhereTS(Trans(out), Trans(condition), Trans(input), other);
}
inline void WhereST(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element& input,
                    LogicalTensorDataPtr other)
{
    GetCalcOps()->WhereST(Trans(out), Trans(condition), input, Trans(other));
}
inline void WhereSS(LogicalTensorDataPtr out, LogicalTensorDataPtr condition, const Element& input,
                    const Element& other)
{
    GetCalcOps()->WhereSS(Trans(out), Trans(condition), input, other);
}
inline void LReLU(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& alpha)
{
    GetCalcOps()->LReLU(Trans(out), Trans(self), alpha);
}
inline void Ln(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Ln(Trans(out), Trans(self)); }
inline void IsFinite(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->IsFinite(Trans(out), Trans(self));
}
inline void IsNan(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->IsNan(Trans(out), Trans(self)); }
inline void Log1p(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Log1p(Trans(out), Trans(self)); }
inline void LogicalNot(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->LogicalNot(Trans(out), Trans(self));
}
inline void Range(LogicalTensorDataPtr out, const Element& start, const Element& end, const Element& step)
{
    GetCalcOps()->Range(Trans(out), start, end, step);
}
inline void Uniform(LogicalTensorDataPtr out, const Element& key, const Element& counter0, const Element& counter1,
                    const Element& rounds, DataType dtype = DT_FP32)
{
    GetCalcOps()->Uniform(Trans(out), key, counter0, counter1, rounds, dtype);
}

inline void Compare(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
                    CmpOperationType operation, CmpModeType mode)
{
    GetCalcOps()->Compare(Trans(out), Trans(self), Trans(other), operation, mode);
}
inline void Cmps(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, CmpOperationType operation,
                 CmpModeType mode)
{
    GetCalcOps()->Cmps(Trans(out), Trans(self), scalar, operation, mode);
}
inline void Hypot(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Hypot(Trans(out), Trans(self), Trans(other));
}
inline void PReLU(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr weight)
{
    GetCalcOps()->PReLU(Trans(out), Trans(self), Trans(weight));
}
inline void LogicalAnd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->LogicalAnd(Trans(out), Trans(self), Trans(other));
}
inline void QuantMX(LogicalTensorDataPtr out, LogicalTensorDataPtr exp, LogicalTensorDataPtr max,
                    LogicalTensorDataPtr scaling, LogicalTensorDataPtr self, bool performanceMode, int64_t mode = 1,
                    int64_t axis = -1)
{
    GetCalcOps()->QuantMX(Trans(out), Trans(exp), Trans(max), Trans(scaling), Trans(self), performanceMode, mode, axis);
}

inline void Interleave(LogicalTensorDataPtr out0, LogicalTensorDataPtr out1, LogicalTensorDataPtr self,
                       LogicalTensorDataPtr other)
{
    GetCalcOps()->Interleave(Trans(out0), Trans(out1), Trans(self), Trans(other));
}

inline void DeInterleave(LogicalTensorDataPtr out0, LogicalTensorDataPtr out1, LogicalTensorDataPtr self,
                         LogicalTensorDataPtr other)
{
    GetCalcOps()->DeInterleave(Trans(out0), Trans(out1), Trans(self), Trans(other));
}

inline void DeInterleaveSingle(LogicalTensorDataPtr out0, LogicalTensorDataPtr out1, LogicalTensorDataPtr self)
{
    GetCalcOps()->DeInterleaveSingle(Trans(out0), Trans(out1), Trans(self));
}

inline void AddS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->AddS(Trans(out), Trans(self), scalar, reverse);
}
inline void SubS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->SubS(Trans(out), Trans(self), scalar, reverse);
}
inline void MulS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->MulS(Trans(out), Trans(self), scalar, reverse);
}
inline void DivS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->DivS(Trans(out), Trans(self), scalar, reverse);
}
inline void FloorDivS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->FloorDivS(Trans(out), Trans(self), scalar, reverse);
}
inline void FmodS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->FmodS(Trans(out), Trans(self), scalar, reverse);
}
inline void GcdS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar)
{
    GetCalcOps()->GcdS(Trans(out), Trans(self), scalar);
}
inline void RemainderS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->RemainderS(Trans(out), Trans(self), scalar, reverse);
}
inline void RemainderRS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = true)
{
    GetCalcOps()->RemainderRS(Trans(out), Trans(self), scalar, reverse);
}
inline void PowS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->PowS(Trans(out), Trans(self), scalar, reverse);
}
inline void BitwiseAndS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar,
                        bool reverse = false)
{
    GetCalcOps()->BitwiseAndS(Trans(out), Trans(self), scalar, reverse);
}
inline void BitwiseOrS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar, bool reverse = false)
{
    GetCalcOps()->BitwiseOrS(Trans(out), Trans(self), scalar, reverse);
}
inline void BitwiseXorS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar,
                        bool reverse = false)
{
    GetCalcOps()->BitwiseXorS(Trans(out), Trans(self), scalar, reverse);
}
inline void Add(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Add(Trans(out), Trans(self), Trans(other));
}
inline void Sub(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Sub(Trans(out), Trans(self), Trans(other));
}
inline void Mul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Mul(Trans(out), Trans(self), Trans(other));
}
inline void Div(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Div(Trans(out), Trans(self), Trans(other));
}
inline void FloorDiv(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->FloorDiv(Trans(out), Trans(self), Trans(other));
}
inline void Fmod(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Fmod(Trans(out), Trans(self), Trans(other));
}
inline void Remainder(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Remainder(Trans(out), Trans(self), Trans(other));
}
inline void Pow(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Pow(Trans(out), Trans(self), Trans(other));
}
inline void Gcd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Gcd(Trans(out), Trans(self), Trans(other));
}
inline void Min(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Min(Trans(out), Trans(self), Trans(other));
}
inline void Max(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->Max(Trans(out), Trans(self), Trans(other));
}
inline void MinS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar)
{
    GetCalcOps()->MinS(Trans(out), Trans(self), scalar);
}
inline void MaxS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar)
{
    GetCalcOps()->MaxS(Trans(out), Trans(self), scalar);
}
inline void BitwiseRightShift(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->BitwiseRightShift(Trans(out), Trans(self), Trans(other));
}
inline void BitwiseLeftShift(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->BitwiseLeftShift(Trans(out), Trans(self), Trans(other));
}
inline void BitwiseRightShiftS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar)
{
    GetCalcOps()->BitwiseRightShiftS(Trans(out), Trans(self), scalar);
}
inline void BitwiseLeftShiftS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element& scalar)
{
    GetCalcOps()->BitwiseLeftShiftS(Trans(out), Trans(self), scalar);
}
inline void SBitwiseRightShift(LogicalTensorDataPtr out, const Element& scalar, LogicalTensorDataPtr other)
{
    GetCalcOps()->SBitwiseRightShift(Trans(out), scalar, Trans(other));
}
inline void SBitwiseLeftShift(LogicalTensorDataPtr out, const Element& scalar, LogicalTensorDataPtr other)
{
    GetCalcOps()->SBitwiseLeftShift(Trans(out), scalar, Trans(other));
}
inline void BitwiseAnd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->BitwiseAnd(Trans(out), Trans(self), Trans(other));
}
inline void BitwiseOr(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->BitwiseOr(Trans(out), Trans(self), Trans(other));
}
inline void BitwiseXor(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->BitwiseXor(Trans(out), Trans(self), Trans(other));
}
inline void ExpandExpDif(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->ExpandExpDif(Trans(out), Trans(self), Trans(other));
}
inline void CopySign(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->CopySign(Trans(out), Trans(self), Trans(other));
}
/* used by reducc op, if shape are not same, need masked */
inline void PairSum(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->PairSum(Trans(out), Trans(self), Trans(other));
}
inline void PairMax(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->PairMax(Trans(out), Trans(self), Trans(other));
}
inline void PairMin(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->PairMin(Trans(out), Trans(self), Trans(other));
}
inline void PairProd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other)
{
    GetCalcOps()->PairProd(Trans(out), Trans(self), Trans(other));
}
inline void RowSumExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowSumExpand(Trans(out), Trans(self), dim);
}
inline void RowMinExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowMinExpand(Trans(out), Trans(self), dim);
}
inline void RowMaxExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowMaxExpand(Trans(out), Trans(self), dim);
}
inline void RowSumSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowSumSingle(Trans(out), Trans(self), dim);
}
inline void RowMinSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowMinSingle(Trans(out), Trans(self), dim);
}
inline void RowMinLine(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowMinLine(Trans(out), Trans(self), dim);
}
inline void RowMaxSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowMaxSingle(Trans(out), Trans(self), dim);
}
inline void RowMaxLine(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowMaxLine(Trans(out), Trans(self), dim);
}
inline void RowProdSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowProdSingle(Trans(out), Trans(self), dim);
}
inline void RowProdLine(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowProdLine(Trans(out), Trans(self), dim);
}
inline void RowArgMaxSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowArgMaxSingle(Trans(out), Trans(self), dim);
}
inline void RowArgMinSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowArgMinSingle(Trans(out), Trans(self), dim);
}
inline void RowArgMaxWithValueSingle(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex,
                                     LogicalTensorDataPtr outTemp, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowArgMaxWithValueSingle(Trans(outValue), Trans(outIndex), Trans(outTemp), Trans(self), dim);
}
inline void RowArgMinWithValueSingle(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex,
                                     LogicalTensorDataPtr outTemp, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowArgMinWithValueSingle(Trans(outValue), Trans(outIndex), Trans(outTemp), Trans(self), dim);
}
inline void RowArgMaxWithValueLine(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex,
                                   LogicalTensorDataPtr outTemp, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowArgMaxWithValueLine(Trans(outValue), Trans(outIndex), Trans(outTemp), Trans(self), dim);
}
inline void RowArgMinWithValueLine(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex,
                                   LogicalTensorDataPtr outTemp, LogicalTensorDataPtr self, int dim)
{
    GetCalcOps()->RowArgMinWithValueLine(Trans(outValue), Trans(outIndex), Trans(outTemp), Trans(self), dim);
}
inline void PairArgMax(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex, LogicalTensorDataPtr value1,
                       LogicalTensorDataPtr index1, LogicalTensorDataPtr value2, LogicalTensorDataPtr index2)
{
    GetCalcOps()->PairArgMax(Trans(outValue), Trans(outIndex), Trans(value1), Trans(index1), Trans(value2),
                             Trans(index2));
}
inline void PairArgMin(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex, LogicalTensorDataPtr value1,
                       LogicalTensorDataPtr index1, LogicalTensorDataPtr value2, LogicalTensorDataPtr index2)
{
    GetCalcOps()->PairArgMin(Trans(outValue), Trans(outIndex), Trans(value1), Trans(index1), Trans(value2),
                             Trans(index2));
}
inline void OneHot(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int numClasses)
{
    GetCalcOps()->OneHot(Trans(out), Trans(self), numClasses);
}
inline void ExpandS(LogicalTensorDataPtr out, const Element& scalar) { GetCalcOps()->ExpandS(Trans(out), scalar); }
inline void Expand(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->Expand(Trans(out), Trans(self));
}
inline void GatherElements(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices,
                           int axis)
{
    GetCalcOps()->GatherElements(Trans(out), Trans(params), Trans(indices), axis);
}
inline void GatherMask(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int patternMode)
{
    GetCalcOps()->GatherMask(Trans(out), Trans(self), patternMode);
}
inline void IndexAdd(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr src,
                     LogicalTensorDataPtr indices, int axis, const Element& alpha = Element(DT_FP32, 1.0))
{
    GetCalcOps()->IndexAdd(Trans(out), Trans(self), Trans(src), Trans(indices), axis, alpha);
}
inline void TriU(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int diagonal)
{
    GetCalcOps()->TriU(Trans(out), Trans(in), diagonal);
}
inline void TriL(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int diagonal)
{
    GetCalcOps()->TriL(Trans(out), Trans(in), diagonal);
}
inline void CumSum(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int axis)
{
    GetCalcOps()->CumSum(Trans(out), Trans(in), axis);
}
inline void CumProd(LogicalTensorDataPtr out, LogicalTensorDataPtr in, int axis)
{
    GetCalcOps()->CumProd(Trans(out), Trans(in), axis);
}
inline void IndexPut(LogicalTensorDataPtr out, LogicalTensorDataPtr self, std::vector<LogicalTensorDataPtr> indices,
                     LogicalTensorDataPtr values, bool accumulate = false)
{
    GetCalcOps()->IndexPut(Trans(out), Trans(self), TransVec(indices), Trans(values), accumulate);
}
inline void Atan(LogicalTensorDataPtr out, LogicalTensorDataPtr self) { GetCalcOps()->Atan(Trans(out), Trans(self)); }
inline void Atan2(LogicalTensorDataPtr out, LogicalTensorDataPtr y, LogicalTensorDataPtr x)
{
    GetCalcOps()->Atan2(Trans(out), Trans(y), Trans(x));
}
inline void Reshape(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->Reshape(Trans(out), Trans(self));
}
inline void Permute(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const std::vector<int64_t>& dim)
{
    GetCalcOps()->Permute(Trans(out), Trans(self), dim);
}
inline void Transpose(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t dim0, int64_t dim1)
{
    GetCalcOps()->Transpose(Trans(out), Trans(self), dim0, dim1);
}

inline void ReduceAcc(LogicalTensorDataPtr out, const std::vector<LogicalTensorDataPtr>& tdatas)
{
    GetCalcOps()->ReduceAcc(Trans(out), TransVec(tdatas));
}

inline void Copy(LogicalTensorDataPtr out, LogicalTensorDataPtr self, bool trans = false, bool isMX = false)
{
    GetCalcOps()->Copy(Trans(out), Trans(self), trans, isMX);
}
inline void ScatterUpdate(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
                          LogicalTensorDataPtr dst, int axis = -2, std::string cacheMode = "BSND", int blockSize = 1)
{
    GetCalcOps()->ScatterUpdate(Trans(out), Trans(self), Trans(index), Trans(dst), axis, cacheMode, blockSize);
}
inline void ScatterElement(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
                           const Element& src, int axis, int reduce)
{
    GetCalcOps()->ScatterElement(Trans(out), Trans(self), Trans(index), src, axis, reduce);
}
inline void Scatter(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr index,
                    LogicalTensorDataPtr src, int axis, int reduce)
{
    GetCalcOps()->Scatter(Trans(out), Trans(self), Trans(index), Trans(src), axis, reduce);
}
inline void BitSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, bool descending, int64_t offset)
{
    GetCalcOps()->BitSort(Trans(out), Trans(self), axis, descending, offset);
}
inline void TiledMrgSort(LogicalTensorDataPtr out, LogicalTensorDataPtr src1, LogicalTensorDataPtr src2,
                         LogicalTensorDataPtr src3, LogicalTensorDataPtr src4, int validBit, int kvalue)
{
    GetCalcOps()->TiledMrgSort(Trans(out), Trans(src1), Trans(src2), Trans(src3), Trans(src4), validBit, kvalue);
}
inline void Gather(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices, int64_t axis)
{
    GetCalcOps()->Gather(Trans(out), Trans(params), Trans(indices), axis);
}
inline void GatherINUB(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices,
                       LogicalTensorDataPtr pageTable, int64_t blockSize, int64_t axis)
{
    GetCalcOps()->GatherINUB(Trans(out), Trans(params), Trans(indices), Trans(pageTable), blockSize, axis);
}
inline void GatherInL1(LogicalTensorDataPtr out, LogicalTensorDataPtr params, LogicalTensorDataPtr indices,
                       LogicalTensorDataPtr pageTable, int64_t blockSize)
{
    CalcOps* ops = GetCalcOps();
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ops != nullptr);
    ops->GatherInL1(Trans(out), Trans(params), Trans(indices), Trans(pageTable), blockSize);
}

inline void Extract(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int mod, bool descending)
{
    GetCalcOps()->Extract(Trans(out), Trans(self), mod, descending);
}

inline void MrgSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t axis, int64_t k)
{
    GetCalcOps()->MrgSort(Trans(out), Trans(self), axis, k);
}

inline void TopK(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outIndex, LogicalTensorDataPtr self, int k,
                 int axis, bool descending)
{
    GetCalcOps()->TopK(Trans(outValue), Trans(outIndex), Trans(self), k, axis, descending);
}

inline void TopkSort(LogicalTensorDataPtr outValue, LogicalTensorDataPtr outTemp, LogicalTensorDataPtr self,
                     int startIndex)
{
    GetCalcOps()->TopkSort(Trans(outValue), Trans(outTemp), Trans(self), startIndex);
}

inline void TopkMerge(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int mergeSize)
{
    GetCalcOps()->TopkMerge(Trans(out), Trans(self), mergeSize);
}

inline void TopkExtract(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int k, bool isIndex)
{
    GetCalcOps()->TopkExtract(Trans(out), Trans(self), k, isIndex);
}

inline void TwoTileMrgSort(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->TwoTileMrgSort(Trans(out), Trans(self));
}

inline void Sort(LogicalTensorDataPtr value, LogicalTensorDataPtr index, LogicalTensorDataPtr self, int64_t axis,
                 bool descending)
{
    GetCalcOps()->Sort(Trans(value), Trans(index), Trans(self), axis, descending);
}

// Quantize
inline void Quantize(LogicalTensorDataPtr out, LogicalTensorDataPtr input, LogicalTensorDataPtr scale,
                     LogicalTensorDataPtr zeroPoints)
{
    TensorData scaleData = Trans(scale);
    if (zeroPoints == nullptr) {
        TensorData emptyZeroPoints = {nullptr, {}, {}, {}, 0, DataType::DT_FP32, false};
        GetCalcOps()->Quantize(Trans(out), Trans(input), scaleData, emptyZeroPoints);
    } else {
        GetCalcOps()->Quantize(Trans(out), Trans(input), scaleData, Trans(zeroPoints));
    }
}

// Dequantize
inline void Dequantize(LogicalTensorDataPtr out, LogicalTensorDataPtr input, LogicalTensorDataPtr scale,
                       LogicalTensorDataPtr zeroPoints)
{
    TensorData scaleData = Trans(scale);
    if (zeroPoints == nullptr) {
        TensorData emptyZeroPoints = {nullptr, {}, {}, {}, 0, DataType::DT_FP32, false};
        GetCalcOps()->Dequantize(Trans(out), Trans(input), scaleData, emptyZeroPoints);
    } else {
        GetCalcOps()->Dequantize(Trans(out), Trans(input), scaleData, Trans(zeroPoints));
    }
}

// matmul
inline void FormatNZ2ND(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->FormatNZ2ND(Trans(out), Trans(self));
}
inline void FormatND2NZ(LogicalTensorDataPtr out, LogicalTensorDataPtr self)
{
    GetCalcOps()->FormatND2NZ(Trans(out), Trans(self));
}

inline void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
                   MatMulParam param = {false, false, false, false, 0, 0, 0, nullptr, nullptr, nullptr, nullptr})
{
    CalcOps* ops = GetCalcOps();
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ops != nullptr);
    ops->MatMul(Trans(out), Trans(self), Trans(other), nullptr, param);
}

inline void AccMatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other,
                      LogicalTensorDataPtr acc = nullptr,
                      MatMulParam param = {false, false, false, false, 0, 0, 0, nullptr, nullptr, nullptr, nullptr})
{
    CalcOps* ops = GetCalcOps();
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ops != nullptr);
    if (acc == nullptr) {
        ops->MatMul(Trans(out), Trans(self), Trans(other), nullptr, param);
    } else {
        auto accData = Trans(acc);
        ops->MatMul(Trans(out), Trans(self), Trans(other), &accData, param);
    }
}
} // namespace npu::tile_fwk::calc
