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
 * \file codegen_utils.cpp
 * \brief
 */

#include "codegen_utils.h"

#include <cstring>
#include <algorithm>
#include <thread>
#include <unistd.h>

#include "interface/configs/config_manager.h"
#include "interface/inner/element.h"
#include "codegen/codegen_common.h"

namespace npu::tile_fwk {
constexpr const int8_t CORE_NUM_MULTIPLE = 2;

std::vector<int64_t> NormalizeShape(const std::vector<int64_t>& shapeVec, unsigned dim)
{
    std::vector<int64_t> normalizedVec(dim, 1);
    for (size_t i = 0; i < shapeVec.size(); i++) {
        ASSERT(OperErr::TENSOR_DIM_EXCEEDED, i < dim) << "exceed dimension limit!";
        normalizedVec[i] = shapeVec[shapeVec.size() - 1 - i];
    }
    std::reverse(normalizedVec.begin(), normalizedVec.end());
    return normalizedVec;
}

std::vector<int> NormalizeExpandAxes(const std::vector<int64_t>& expandAxes, unsigned originalDimSize,
                                     unsigned targetDimSize)
{
    std::vector<int> normalizedAxesList;
    for (auto axis : expandAxes) {
        bool isValidAxis = ((axis >= 0) && (axis <= (originalDimSize - 1)));
        ASSERT(OperErr::ATTRIBUTE_INVALID, isValidAxis) << "unsupported expand axis: " << axis;
        normalizedAxesList.push_back(static_cast<int>(axis) + targetDimSize - originalDimSize);
    }
    return normalizedAxesList;
}

std::string FormatFloat(const std::variant<int64_t, uint64_t, double>& v, DataType dtype, int precision)
{
    // 定义处理函数
    auto apply = [&](auto&& val) -> std::string {
        std::ostringstream oss;
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, double>) {
            if ((dtype == DataType::DT_FP32 || dtype == DataType::DT_FP16 || dtype == DataType::DT_BF16) &&
                (std::isinf(val) || std::isnan(val))) {
                FloatSpecVal fsv = {dtype, val};
                oss << fsv.GetFsVarName() << ".f";
                return oss.str();
            }
        }
        oss << std::setprecision(precision) << val;
        std::string s = oss.str();
        // A C/C++ float literal with f suffix needs a decimal point or exponent.
        if (s.find('.') == std::string::npos && s.find_first_of("eE") == std::string::npos) {
            s += ".f";
        } else {
            s += "f";
        }
        return s;
    };

    return std::visit(apply, v);
}

std::string FormatScalarLiteral(const Element& scalar)
{
    if (scalar.IsFloat()) {
        return FormatFloat(scalar.Cast<float>(), scalar.GetDataType());
    }
    if (scalar.IsUnsigned() || scalar.IsSigned()) {
        return std::visit([](const auto& val) -> std::string { return std::to_string(val); }, scalar.GetVariantData());
    }
    return "";
}

std::string GetTypeForB16B32(const DataType& dtype)
{
    if (BytesOf(dtype) == K_BYTES_OF16_BIT) {
        return "uint16_t";
    }
    if (BytesOf(dtype) == K_BYTES_OF32_BIT) {
        return "uint32_t";
    }
    ASSERT(GenCodeErr::DATA_TYPE_UNSUPPORTED, false) << "can not support dtype: " << DataType2String(dtype);
    return {};
}

std::string GetAddrTypeByOperandType(OperandType type)
{
    auto iter = OPERAND_TYPE_TO_ADDR_TYPE.find(type);
    if (iter != OPERAND_TYPE_TO_ADDR_TYPE.end()) {
        return iter->second;
    }
    ASSERT(OperErr::OPERAND_TYPE_UNSUPPORTED, false) << "cannot support current OperandType " << type;
    return "";
}

std::string CopyInModeToString(Matrix::CopyInMode copyMode)
{
    switch (copyMode) {
        case Matrix::CopyInMode::ND2ND:
            return "CopyInMode::ND2ND";
        case Matrix::CopyInMode::DN2NZ:
            return "CopyInMode::DN2NZ";
        case Matrix::CopyInMode::NZ2NZ:
            return "CopyInMode::NZ2NZ";
        case Matrix::CopyInMode::ND2NZ:
            return "CopyInMode::ND2NZ";
        default:
            return "CopyInMode::ND2NZ";
    }
}

std::string CopyOutModeToString(Matrix::CopyOutMode copyMode)
{
    switch (copyMode) {
        case Matrix::CopyOutMode::NZ2ND:
            return "CopyOutMode::NZ2ND";
        case Matrix::CopyOutMode::NZ2NZ:
            return "CopyOutMode::NZ2NZ";
        case Matrix::CopyOutMode::ND2ND:
            return "CopyOutMode::ND2ND";
        case Matrix::CopyOutMode::NZ2DN:
            return "CopyOutMode::NZ2DN";
        default:
            return "CopyOutMode::NZ2ND";
    }
}

std::string PaddingModeToString(Matrix::PaddingMode paddingMode)
{
    switch (paddingMode) {
        case Matrix::PaddingMode::NORMAL_PADDING_MODE:
            return "PaddingMode::NORMAL_PADDING_MODE";
        case Matrix::PaddingMode::MX_PADDING_MODE:
            return "PaddingMode::MX_PADDING_MODE";
        default:
            return "PaddingMode::NORMAL_PADDING_MODE";
    }
}

std::string CopyModeToString(Matrix::CopyMode copyMode)
{
    switch (copyMode) {
        case Matrix::CopyMode::EXTRACT:
            return "CopyMode::EXTRACT";
        case Matrix::CopyMode::INSERT:
            return "CopyMode::INSERT";
        case Matrix::CopyMode::MOVE:
            return "CopyMode::MOVE";
        default:
            return "CopyMode::UNKNOWN";
    }
}

int64_t CalcLinearOffset(const std::vector<int64_t>& shape, const std::vector<int64_t>& offset)
{
    if (shape.empty() || offset.empty() || shape.size() != offset.size()) {
        CODEGEN_LOGE(GenCodeErr::TENSOR_SHAPE_INVALID, "Invalid Input! shape: %s, offset: %s",
                     IntVecToStr(shape).c_str(), IntVecToStr(offset).c_str());
        return 0;
    }

    int64_t resOffset{0};
    int64_t base = 1;
    for (int i = static_cast<int>(offset.size()) - 1; i >= 0; i--) {
        resOffset += offset[i] * base;
        base *= shape[i];
    }

    return resOffset;
}

void PrintIndent(std::ostringstream& os, int scopeLevel)
{
    for (int i = 0; i < scopeLevel; i++) {
        os << "  ";
    }
}

unsigned GetCGThreadNum()
{
    // kernel名已按任务身份确定性编号，固定CCE场景不必再退化为单线程
    unsigned threadNum = ConfigManager::Instance().GetCodeGenConfig(KEY_PARALLEL_COMPILE, 1u);
    unsigned cpuCores = std::thread::hardware_concurrency();
    if (cpuCores != 0 && threadNum > cpuCores * CORE_NUM_MULTIPLE) {
        return cpuCores * CORE_NUM_MULTIPLE;
    }
    return threadNum;
}

std::string StringSubstitute(std::string const& in, SubstMap const& subst)
{
    const char* tokenHead = "${";
    const char* tokenTail = "}$";
    constexpr size_t tokenSepLen = 2;

    std::ostringstream out;
    size_t pos = 0;
    for (;;) {
        size_t substPos = in.find(tokenHead, pos);
        size_t endPos = in.find(tokenTail, substPos);
        if (endPos == std::string::npos) {
            break;
        }

        out.write(&*in.begin() + pos, substPos - pos);

        substPos += tokenSepLen;
        auto substIter = subst.find(in.substr(substPos, endPos - substPos));
        if (substIter == subst.end()) {
            throw std::runtime_error("undefined substitution");
        }

        out << substIter->second;
        pos = endPos + tokenSepLen;
    }
    out << in.substr(pos, std::string::npos);
    return out.str();
}

} // namespace npu::tile_fwk
