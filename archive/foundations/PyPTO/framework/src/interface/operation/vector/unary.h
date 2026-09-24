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
 * \file unary.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

enum class UnaryOpType {
    EXP,
    RSQRT,
    RELU,
    SQRT,
    CEIL,
    FLOOR,
    TRUNC,
    RECIPROCAL,
    DUPLICATE,
    ABS,
    LN,
    HUB,
    BITWISENOT,
    SIGN,
    SIGNBIT,
    TANH,
    ISFINITE,
    ISNAN,
    ATAN,
    SINH,
    COSH,
    ATANH,
    SIN,
    COS,
    ERFC,
    PACK,
    UNPACK,
    ASIN,
    ACOS,
    ERF,
    ASINH,
    ACOSH,
};

template <UnaryOpType T>
std::string GetUnaryOpName()
{
    switch (T) {
        case UnaryOpType::EXP:
            return "EXP";
        case UnaryOpType::RSQRT:
            return "RSQRT";
        case UnaryOpType::RELU:
            return "RELU";
        case UnaryOpType::SQRT:
            return "SQRT";
        case UnaryOpType::CEIL:
            return "CEIL";
        case UnaryOpType::FLOOR:
            return "FLOOR";
        case UnaryOpType::TRUNC:
            return "TRUNC";
        case UnaryOpType::RECIPROCAL:
            return "RECIPROCAL";
        case UnaryOpType::DUPLICATE:
            return "DUPLICATE";
        case UnaryOpType::ABS:
            return "ABS";
        case UnaryOpType::LN:
            return "LN";
        case UnaryOpType::ISFINITE:
            return "ISFINITE";
        case UnaryOpType::ISNAN:
            return "ISNAN";
        case UnaryOpType::ATAN:
            return "ATAN";
        case UnaryOpType::HUB:
            return "HUB";
        case UnaryOpType::BITWISENOT:
            return "BITWISENOT";
        case UnaryOpType::SIGN:
            return "SIGN";
        case UnaryOpType::SIGNBIT:
            return "SIGNBIT";
        case UnaryOpType::SINH:
            return "SINH";
        case UnaryOpType::COSH:
            return "COSH";
        case UnaryOpType::ATANH:
            return "ATANH";
        case UnaryOpType::SIN:
            return "SIN";
        case UnaryOpType::COS:
            return "COS";
        case UnaryOpType::ERFC:
            return "ERFC";
        case UnaryOpType::PACK:
            return "PACK";
        case UnaryOpType::UNPACK:
            return "UNPACK";
        case UnaryOpType::ASINH:
            return "ASINH";
        case UnaryOpType::ACOSH:
            return "ACOSH";
        case UnaryOpType::TANH:
            return "TANH";
        case UnaryOpType::ASIN:
            return "ASIN";
        case UnaryOpType::ACOS:
            return "ACOS";
        case UnaryOpType::ERF:
            return "ERF";
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown unary op type";
            return "";
    }
}

template <UnaryOpType T>
Opcode GetUnaryOpNameCode()
{
#define CASE(X)          \
    case UnaryOpType::X: \
        return Opcode::OP_##X
    switch (T) {
        CASE(EXP);
        CASE(RSQRT);
        CASE(RELU);
        CASE(SQRT);
        CASE(CEIL);
        CASE(FLOOR);
        CASE(TRUNC);
        CASE(RECIPROCAL);
        CASE(DUPLICATE);
        CASE(ABS);
        CASE(LN);
        CASE(ISFINITE);
        CASE(ISNAN);
        CASE(ATAN);
        CASE(HUB);
        CASE(BITWISENOT);
        CASE(SIGN);
        CASE(SIGNBIT);
        CASE(ERF);
        CASE(SINH);
        CASE(COSH);
        CASE(ATANH);
        CASE(SIN);
        CASE(COS);
        CASE(ERFC);
        CASE(PACK);
        CASE(UNPACK);
        CASE(ASINH);
        CASE(ACOSH);
        CASE(TANH);
        CASE(ASIN);
        CASE(ACOS);
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown unary op type";
    }
#undef CASE
}

void UnaryOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                const std::vector<LogicalTensorPtr>& oOperand);

template <UnaryOpType T>
std::pair<LogicalTensorPtr, Operation*> TensorUnaryOperationWithOp(Function& function, LogicalTensorPtr operand,
                                                                   std::optional<DataType> datatype = std::nullopt)
{
    auto opName = GetUnaryOpName<T>();
    CheckTensorDimRange(operand, MIN_TENSOR_DIM, MAX_TENSOR_DIM, opName);
    CheckTensorShapeSize(operand, opName);
    datatype = datatype.value_or(operand->tensor->datatype);
    auto result = std::make_shared<LogicalTensor>(function, *datatype, operand->shape, operand->GetDynValidShape(),
                                                  operand->Format());
    Operation* op = &function.AddOperation(GetUnaryOpNameCode<T>(), {operand}, {result});
    return {result, op};
}

template <UnaryOpType T>
LogicalTensorPtr TensorUnaryOperation(Function& function, LogicalTensorPtr operand,
                                      std::optional<DataType> datatype = std::nullopt)
{
    return TensorUnaryOperationWithOp<T>(function, operand, datatype).first;
}

} // namespace npu::tile_fwk
