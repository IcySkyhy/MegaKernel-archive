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
 * \file test_data_prepare.h
 * \brief Data preparation manager for test utilities
 */

#pragma once

#include <string>
#include <vector>
#include "test_common.h"
#include "interface/interpreter/raw_tensor_data.h"

using namespace npu::tile_fwk;

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName)
{
    auto shape = tensor.GetShape();
    int64_t capacity = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        capacity = capacity * shape[i];
    }
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::shared_ptr<RawTensorData> LoadTensorData(const Shape& shape, DataType dType, std::string fileName)
{
    int64_t capacity = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        capacity = capacity * shape[i];
    }
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensorData<T>(shape, dType, values);
}

template <typename T>
static std::shared_ptr<RawTensorData> LoadTensorData(const Tensor& t, std::string fileName)
{
    return LoadTensorData<T>(t.GetShape(), t.GetDataType(), fileName);
}

template <typename T>
static std::vector<T> GetGoldenVec(std::vector<int64_t> shape, std::string fileName)
{
    int64_t capacity = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        capacity = capacity * shape[i];
    }
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

struct TensorWithData {
    Tensor tensor;
    RawTensorDataPtr dataPtr = nullptr;
};

struct QuantTensorWithData {
    bool isQuant = false;
    bool isSmooth = false;
    std::vector<int64_t> scaleShape;
    std::vector<int64_t> smoothShape;
    std::string scaleTensorName;
    std::string smoothTensorName;
    std::string scaleDataPath;
    std::string smoothDataPath;
    TensorWithData scale;
    TensorWithData smooth;

    QuantTensorWithData(bool isQuantT, bool isSmoothT, std::vector<int64_t> scaleShapeT,
                        std::vector<int64_t> smoothShapeT, std::string scaleName, std::string smoothName,
                        std::string scalePath, std::string smoothPath)
        : isQuant(isQuantT),
          isSmooth(isSmoothT),
          scaleShape(scaleShapeT),
          smoothShape(smoothShapeT),
          scaleTensorName(scaleName),
          smoothTensorName(smoothName),
          scaleDataPath(scalePath),
          smoothDataPath(smoothPath)
    {}
};

inline void CreateQuantTensorAndData(QuantTensorWithData& quant)
{
    if (quant.isQuant) {
        Tensor quantTmp(DT_FP32, quant.scaleShape, quant.scaleTensorName);
        quant.scale.tensor = quantTmp;
        quant.scale.dataPtr = LoadTensorData<float>(quant.scale.tensor, quant.scaleDataPath);

        if (quant.isSmooth) {
            Tensor smoothTmp(DT_FP32, quant.smoothShape, quant.smoothTensorName);
            quant.smooth.tensor = smoothTmp;
            quant.smooth.dataPtr = LoadTensorData<float>(quant.smooth.tensor, quant.smoothDataPath);
        }
    }
}

template <typename T>
TensorWithData CreateTensorAndData(const std::vector<int64_t>& shape, DataType dType, std::string name,
                                   TileOpFormat format, std::string binPath, const std::vector<int>& dynamicAxises = {})
{
    std::vector<int64_t> dynamicShape = shape;
    for (int axis : dynamicAxises) {
        ASSERT(axis >= 0 && (size_t)axis < dynamicShape.size());
        dynamicShape[axis] = -1;
    }

    Tensor dynamicT(dType, dynamicShape, name, format);
    RawTensorDataPtr data = LoadTensorData<T>(shape, dType, binPath);
    return TensorWithData{dynamicT, data};
}

template <typename T>
TensorWithData CreateTensorAndData(const std::vector<int64_t>& shape, DataType dType, std::string name,
                                   std::string binPath, const std::vector<int>& dynamicAxises = {})
{
    return CreateTensorAndData<T>(shape, dType, name, TileOpFormat::TILEOP_ND, binPath, dynamicAxises);
}

template <typename T>
TensorWithData CreateDynamicOutputTensor(const std::vector<int64_t>& shape, DataType dType, std::string name,
                                         std::string binPath, const std::vector<SymbolicScalar>& dynamicShape = {})
{
    ASSERT(dynamicShape.size() == 0 || dynamicShape.size() == shape.size());

    if (dynamicShape.empty()) {
        Tensor dynamicT(dType, shape, name);
        RawTensorDataPtr data = LoadTensorData<T>(shape, dType, binPath);
        return TensorWithData{dynamicT, data};
    }

    Tensor dynamicT(dType, dynamicShape, name);
    RawTensorDataPtr data = LoadTensorData<T>(shape, dType, binPath);
    return TensorWithData{dynamicT, data};
}

template <typename T>
TensorWithData CreateConstantDynamicOutputTensor(const std::vector<int64_t>& shape, DataType dType, std::string name,
                                                 T value, const std::vector<SymbolicScalar>& dynamicShape = {})
{
    ASSERT(dynamicShape.size() == 0 || dynamicShape.size() == shape.size());

    if (dynamicShape.empty()) {
        Tensor dynamicT(dType, shape, name);
        RawTensorDataPtr data = RawTensorData::CreateConstantTensorData<T>(shape, dType, value);
        return TensorWithData{dynamicT, data};
    }

    Tensor dynamicT(dType, dynamicShape, name);
    RawTensorDataPtr data = RawTensorData::CreateConstantTensorData<T>(shape, dType, value);
    return TensorWithData{dynamicT, data};
}

template <typename T>
TensorWithData CreateConstantTensorAndData(const std::vector<int64_t>& shape, DataType dType, std::string name, T value,
                                           const std::vector<int>& dynamicAxises = {})
{
    std::vector<int64_t> dynamicShape = shape;
    for (int axis : dynamicAxises) {
        ASSERT(axis >= 0 && (size_t)axis < dynamicShape.size());
        dynamicShape[axis] = -1;
    }

    Tensor dynamicT(dType, dynamicShape, name);
    RawTensorDataPtr data = RawTensorData::CreateConstantTensorData<T>(shape, dType, value);
    return TensorWithData{dynamicT, data};
}
