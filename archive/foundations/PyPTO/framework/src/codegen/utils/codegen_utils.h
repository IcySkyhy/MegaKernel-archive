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
 * \file codegen_utils.h
 * \brief
 */

#ifndef CODEGEN_UTILS_H
#define CODEGEN_UTILS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <variant>
#include <map>

#include "codegen/codegen_common.h"
#include "tilefwk/error_code.h"
#include "tilefwk/data_type.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/inner/pre_def.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"

namespace npu::tile_fwk {
constexpr int COMMENT_PREFIX_LENGTH = 2;

template <typename T>
inline void FillVecWithDummyInHead(std::vector<T>& input, unsigned padNum, T dummy)
{
    for (unsigned i = 0; i < padNum; ++i) {
        input.insert(input.begin(), dummy);
    }
}

// only recogonize /* as comment prefix
inline bool StartWithComment(const std::string& str)
{
    return str.size() >= COMMENT_PREFIX_LENGTH && str[0] == '/' && str[1] == '*';
}
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::string> ToStringHelper(const T& value)
{
    return std::to_string(value);
}
inline std::string ToStringHelper(const std::string& value) { return value; }
template <typename... Ts>
std::string ToStringHelper(const std::variant<Ts...>& value)
{
    return std::visit([](const auto& arg) { return ToStringHelper(arg); }, value);
}

inline std::string ToStringHelper(const SymbolicScalar& value)
{
    return SymbolicExpressionTable::BuildExpression(value);
}

template <typename T = std::string>
std::string JoinString(const std::vector<T>& params, const std::string& sep)
{
    std::ostringstream oss;

    for (size_t i = 0; i < params.size(); ++i) {
        std::string current = ToStringHelper(params[i]);
        if (current.empty()) {
            continue;
        }
        if (i > 0) {
            bool useEmptySep{false};
            if constexpr (std::is_same_v<T, std::string>) {
                useEmptySep = StartWithComment(params[i - 1]);
            }
            oss << (useEmptySep ? " " : sep);
        }
        oss << current;
    }

    return oss.str();
}

template <typename T = std::string>
std::string PrintParams(const std::pair<std::string, std::string>& delimiter, const std::vector<T>& params,
                        const std::string& conj)
{
    std::ostringstream oss;
    oss << delimiter.first << JoinString<T>(params, conj) << delimiter.second;
    return oss.str();
}

template <typename T = std::string>
std::string WrapParamByParentheses(const std::vector<T>& params)
{
    return PrintParams(DELIMITER_PARENTHESES, params, CONN_COMMA);
}

template <typename T = std::string>
std::string WrapParamByAngleBrackets(const std::vector<T>& params)
{
    return PrintParams(DELIMITER_ANGLE_BRACKETS, params, CONN_COMMA);
}

std::vector<int64_t> NormalizeShape(const std::vector<int64_t>& shapeVec, unsigned dim);

std::vector<int> NormalizeExpandAxes(const std::vector<int64_t>& expandAxes, unsigned originalDimSize,
                                     unsigned targetDimSize);

std::string FormatFloat(const std::variant<int64_t, uint64_t, double>& v, DataType dtype = DataType::DT_FP32,
                        int precision = 9);

std::string FormatScalarLiteral(const Element& scalar);

std::string GetTypeForB16B32(const DataType& dtype);

inline std::string GetPipeId(PipeType queue)
{
    auto res = PIPE_ID.find(queue);
    ASSERT(GenCodeErr::PIPE_ID_NOT_FOUND, res != PIPE_ID.end()) << "can not find pipe id: " << ToUnderlying(queue);
    return res->second;
}

inline std::string GetTileOpName(Opcode opCode)
{
    const auto& opCfg = OpcodeManager::Inst().GetTileOpCfg(opCode);
    return opCfg.tileOpCode_;
}

std::string GetAddrTypeByOperandType(OperandType type);

int64_t CalcLinearOffset(const std::vector<int64_t>& shape, const std::vector<int64_t>& offset);

template <typename T>
void FillParamWithInput(std::vector<std::string>& paramList, const std::vector<T>& input, int start, int count)
{
    for (int i = start; i < count; ++i) {
        paramList.emplace_back(ToStringHelper(input[i]));
    }
}

void PrintIndent(std::ostringstream& os, int scopeLevel);

struct FloatSpecVal {
    DataType dtype; // fp32 or fp16
    double value;

    bool operator<(const FloatSpecVal& other) const
    {
        if (dtype != other.dtype) {
            return ToUnderlying(dtype) < ToUnderlying(other.dtype);
        }
        // NaN 比较特殊处理
        if (std::isnan(value) && std::isnan(other.value))
            return false;
        if (std::isnan(value))
            return true;
        if (std::isnan(other.value))
            return false;
        return value < other.value;
    }

    std::string GetFsVarName() const
    {
        std::string fsType = std::isinf(value) ? (std::signbit(value) ? "inf_neg" : "inf_pos") : "nan";
        std::string fsVarName = std::string(DataType2CCEStr(dtype)) + "_" + fsType;
        return fsVarName;
    }

    std::string GetFsValueStr() const
    {
        static const std::map<std::pair<DataType, bool>, std::string> infMap = {
            {{DataType::DT_FP16, false}, FP16_INF_POS}, {{DataType::DT_FP16, true}, FP16_INF_NEG},
            {{DataType::DT_FP32, false}, FP32_INF_POS}, {{DataType::DT_FP32, true}, FP32_INF_NEG},
            {{DataType::DT_BF16, false}, BF16_INF_POS}, {{DataType::DT_BF16, true}, BF16_INF_NEG},
        };
        static const std::map<DataType, std::string> nanMap = {
            {DataType::DT_FP16, FP16_NAN},
            {DataType::DT_FP32, FP32_NAN},
            {DataType::DT_BF16, BF16_NAN},
        };

        if (std::isinf(value)) {
            auto key = std::make_pair(dtype, std::signbit(value));
            auto iter = infMap.find(key);
            if (iter != infMap.end()) {
                return iter->second;
            }
        } else if (std::isnan(value)) {
            auto iter = nanMap.find(dtype);
            if (iter != nanMap.end()) {
                return iter->second;
            }
        }
        ASSERT(GenCodeErr::DATA_TYPE_UNSUPPORTED, false)
            << "FloatSpecVal not found, dtype: " << ToUnderlying(dtype) << ", value: " << value;
        return "";
    }
};

std::string CopyInModeToString(Matrix::CopyInMode copyMode);

std::string CopyOutModeToString(Matrix::CopyOutMode copyMode);

std::string PaddingModeToString(Matrix::PaddingMode paddingMode);

std::string CopyModeToString(Matrix::CopyMode copyMode);

unsigned GetCGThreadNum();

using SubstMap = std::map<std::string, std::string>;

std::string StringSubstitute(std::string const& in, SubstMap const& subst);

} // namespace npu::tile_fwk
#endif
