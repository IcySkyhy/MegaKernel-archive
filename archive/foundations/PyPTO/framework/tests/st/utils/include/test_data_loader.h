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
 * \file test_data_loader.h
 * \brief
 */

#pragma once

#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <fstream>
#include <variant>

#include "tilefwk/tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"

#include "test_dev_func_runner.h"
#include "test_data_prepare.h"

#include <iomanip>

// 定义标量值类型
using Scalar = std::variant<int, float, double, bool, std::string>;
constexpr int COLUMN_WIDTH = 20;

namespace {
std::string ShapeToString(const Shape& shape)
{
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0)
            oss << ", ";
        oss << shape[i];
    }
    oss << ")";
    return oss.str();
}

void PrintTableRow(int index, const Tensor& tensor)
{
    std::string name = tensor.GetName();
    std::string shapeStr = ShapeToString(tensor.GetShape());
    std::string typeStr = DataType2String(tensor.GetDataType());
    std::string formatStr = tensor.GetStorage()->Format() == TileOpFormat::TILEOP_ND ? "TILEOP_ND" : "TILEOP_NZ";
    std::cout << "| " << std::setw(COLUMN_WIDTH / 2) << std::left << index << " | " << std::setw(COLUMN_WIDTH)
              << std::left << name << " | " << std::setw(COLUMN_WIDTH) << std::left << shapeStr << " | "
              << std::setw(COLUMN_WIDTH / 2) << std::left << typeStr << " | " << std::setw(COLUMN_WIDTH / 2)
              << std::left << formatStr << " |" << std::endl;
}

void PrintTableDivider()
{
    std::cout << "+------------+----------------------+----------------------+------------+------------+" << std::endl;
}

Scalar ConvertJsonToScalar(const nlohmann::json& value)
{
    if (value.is_number_integer()) {
        return value.get<int>();
    } else if (value.is_number_float()) {
        return value.get<double>();
    } else if (value.is_boolean()) {
        return value.get<bool>();
    } else if (value.is_string()) {
        return value.get<std::string>();
    } else {
        throw std::invalid_argument("Unsupported JSON type for Scalar");
    }
}

bool IsValidWeightFormat(int value)
{
    return value == static_cast<int>(TileOpFormat::TILEOP_ND) || value == static_cast<int>(TileOpFormat::TILEOP_NZ);
    // 如有新的合法格式可以继续添加
}
} // namespace

class TestDataLoader {
public:
    TestDataLoader(const std::string& path) : configPath(path)
    {
        // 读取JSON
        std::ifstream configFile(this->configPath);
        if (!configFile.is_open()) {
            throw std::runtime_error("Failed to open config file: " + this->configPath);
        }

        // JSON内容存入meta
        try {
            meta = nlohmann::json::parse(configFile);
        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
        }

        LoadParameters();
        LoadInputTensors();
        LoadGoldenTensors();
    }

    // 获取数据函数
    auto GetParams() { return params; }
    auto Param(const std::string& name) { return params.at(name); }
    Tensor& InputTensor(const std::string& name) { return inputTensors.at(name); }
    Tensor& OutputTensor(const std::string& name) { return outputTensors.at(name); }
    const std::vector<std::reference_wrapper<const Tensor>>& GetInputTensorList() const { return inputTensorList; }
    const std::vector<std::reference_wrapper<const Tensor>>& GetOutputTensorList() const { return outputTensorList; }
    std::vector<RawTensorDataPtr> GetInputDataList() { return inputDataList; }
    std::vector<RawTensorDataPtr> GetOutputDataList() { return outputDataList; }
    RawTensorDataPtr GoldenData(const std::string& name) { return goldens.at(name); }
    int GetInputNameToIdx(const std::string& name) { return inputNameToIdx.at(name); }
    int GetOutputNameToIdx(const std::string& name) { return outputNameToIdx.at(name); }

    Tensor& InputTensorCheck(const std::string& name, const DataType dtype, const Shape& shape,
                             const TileOpFormat format = TileOpFormat::TILEOP_ND)
    {
        // 用户传入需要的input tensor的DataType,Shape,weightFormat进行校验，正确返回tensor&

        // 检查inoutTensors是否存在name
        auto it = this->inputTensors.find(name);
        if (it == this->inputTensors.end()) {
            throw std::runtime_error("Tensor " + name + " not found in input tensors.");
        }

        // 获取当前的tensor的DataType,Shape,opFormat
        const DataType curDType = it->second.GetDataType();
        const Shape& curShape = it->second.GetShape();
        const TileOpFormat curOpFormat = it->second.GetStorage()->Format();

        // 校验DataType
        if (dtype != curDType) {
            throw std::runtime_error("Data type mismatch for inputTensor " + name);
        }

        // 校验Shape
        if (shape.size() != curShape.size()) {
            throw std::runtime_error("Shape size mismatch for inputTensor " + name);
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (curShape[i] != -1 && shape[i] != curShape[i]) {
                throw std::runtime_error("Shape mismatch for inputTensor " + name);
            }
        }

        if (format != curOpFormat) {
            throw std::runtime_error("Weight format mismatch for inputTensor " + name);
        }

        return inputTensors.at(name);
    }

    Tensor& OutputTensorCheck(const std::string& name, const DataType dtype, const Shape& shape,
                              const TileOpFormat format = TileOpFormat::TILEOP_ND)
    {
        // 用户传入需要的output tensor的DataType,Shape,weightFormat进行校验，正确返回tensor&

        // 检查outputTensors是否存在name
        auto it = this->outputTensors.find(name);
        if (it == this->outputTensors.end()) {
            throw std::runtime_error("Tensor " + name + " not found in output tensors.");
        }

        // 获取当前的tensor的DataType,Shape,opFormat
        const DataType curDType = it->second.GetDataType();
        const Shape& curShape = it->second.GetShape();
        const TileOpFormat curOpFormat = it->second.GetStorage()->Format();

        // 校验DataType
        if (dtype != curDType) {
            throw std::runtime_error("Data type mismatch for outputTensor " + name);
        }

        // 校验Shape
        if (shape.size() != curShape.size()) {
            throw std::runtime_error("Shape size mismatch for outputTensor " + name);
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (curShape[i] != -1 && shape[i] != curShape[i]) {
                throw std::runtime_error("Shape mismatch for outputTensor " + name);
            }
        }

        if (format != curOpFormat) {
            throw std::runtime_error("Weight format mismatch for outputTensor " + name);
        }

        return outputTensors.at(name);
    }

    RawTensorDataPtr GoldenDataCheck(const std::string& name, const DataType dtype, const Shape& shape)
    {
        // 用户传入需要的goldens的DataType,Shape进行校验，正确返回tensor&

        // 检查RawTensorDataPtr是否存在
        auto it = this->goldens.find(name);
        if (it == this->goldens.end()) {
            throw std::runtime_error("Golden tensor " + name + " not found.");
        }

        // 获取当前的tensor的DataType,Shape
        const DataType curDType = it->second->GetDataType();
        const Shape& curShape = it->second->GetShape();

        // 校验DataType
        if (dtype != curDType) {
            throw std::runtime_error("Data type mismatch for Golden " + name);
        }

        // 校验Shape
        if (shape.size() != curShape.size()) {
            throw std::runtime_error("Shape size mismatch for Golden " + name);
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != curShape[i]) {
                throw std::runtime_error("Shape mismatch for Golden " + name);
            }
        }

        return goldens.at(name);
    }

    void Dump()
    {
        int index = 0;
        std::cout << "Input tensors: ";
        PrintTableDivider();
        std::cout << "| " << std::setw(COLUMN_WIDTH / 2) << std::left << "Index"
                  << " | " << std::setw(COLUMN_WIDTH) << std::left << "Name"
                  << " | " << std::setw(COLUMN_WIDTH) << std::left << "Shape"
                  << " | " << std::setw(COLUMN_WIDTH / 2) << std::left << "Datatype"
                  << " | " << std::setw(COLUMN_WIDTH / 2) << std::left << "OpFormat"
                  << " |" << std::endl;
        ;
        PrintTableDivider();
        for (const auto& tensor : inputTensorList) {
            PrintTableRow(index, tensor);
            index++;
        }
        PrintTableDivider();

        index = 0;
        std::cout << "Output tensors: ";
        PrintTableDivider();
        std::cout << "| " << std::setw(COLUMN_WIDTH / 2) << std::left << "Index"
                  << " | " << std::setw(COLUMN_WIDTH) << std::left << "Name"
                  << " | " << std::setw(COLUMN_WIDTH) << std::left << "Shape"
                  << " | " << std::setw(COLUMN_WIDTH / 2) << std::left << "Datatype"
                  << " | " << std::setw(COLUMN_WIDTH / 2) << std::left << "OpFormat"
                  << " |" << std::endl;
        PrintTableDivider();
        for (const auto& tensor : outputTensorList) {
            PrintTableRow(index, tensor);
            index++;
        }
        PrintTableDivider();
    }

    // 创建 Tensor 函数
    static std::pair<Tensor, RawTensorDataPtr> CreateTensor(const std::string& name, const std::string& dtype,
                                                            const Shape& shape, const std::string& fileName,
                                                            const TileOpFormat format = TileOpFormat::TILEOP_ND)
    {
        Tensor t(CostModel::ToDataType(const_cast<string&>(dtype)), shape, name, format);
        auto dataPtr = CreateHelper(dtype, t, fileName);
        return std::make_pair(t, dataPtr);
    }

    static RawTensorDataPtr CreateHelper(const std::string& dtype, const Tensor& tensor, const std::string& fileName)
    {
        static const std::unordered_map<std::string, std::function<RawTensorDataPtr(const Tensor&, const std::string&)>>
            creators = {
                {"DT_INT8", TestDataLoader::CreateTensorData<int8_t>},
                {"DT_INT32", TestDataLoader::CreateTensorData<int32_t>},
                {"DT_INT64", TestDataLoader::CreateTensorData<int64_t>},
                {"DT_FP16", TestDataLoader::CreateTensorData<npu::tile_fwk::float16>},
                {"DT_FP32", TestDataLoader::CreateTensorData<float>},
                {"DT_BF16", TestDataLoader::CreateTensorData<npu::tile_fwk::bfloat16>},
            };

        auto it = creators.find(dtype);
        if (it != creators.end()) {
            return it->second(tensor, fileName);
        }
        std::cerr << "Unsupported type : " << dtype;
        return nullptr;
    }

    template <typename T>
    static RawTensorDataPtr CreateTensorData(const Tensor& tensor, const std::string& fileName);

    // 设置动态 shape
    Tensor& SetInputDynAxis(const std::string& name, const std::vector<int>& dynAxises)
    {
        auto& tensor = this->inputTensors.at(name);
        Shape dynamicShape = tensor.GetShape();
        for (int axis : dynAxises) {
            ASSERT(axis >= 0 && (size_t)axis < dynamicShape.size());
            dynamicShape[axis] = -1;
        }
        Tensor dynamicT(tensor.GetDataType(), dynamicShape, tensor.GetName(), tensor.GetStorage()->Format());
        auto [it, is_inserted] = this->inputTensors.insert_or_assign(name, dynamicT);
        ASSERT(is_inserted);
        this->inputTensorList[this->inputNameToIdx.at(name)] = std::cref(it->second);

        return it->second;
    }

    Tensor& SetOutputDynAxis(const std::string& name, const std::vector<SymbolicScalar>& dynShape)
    {
        auto& tensor = this->outputTensors.at(name);
        ASSERT(tensor.GetShape().size() == dynShape.size());
        Tensor dynamicT(tensor.GetDataType(), dynShape, tensor.GetName(), tensor.GetStorage()->Format());
        auto [it, is_inserted] = this->outputTensors.insert_or_assign(name, dynamicT);
        ASSERT(is_inserted);
        this->outputTensorList[this->outputNameToIdx.at(name)] = std::cref(it->second);

        return it->second;
    }

private:
    // 加载数据函数
    void LoadParameters()
    {
        // 检查meta中是否存在"parameters"对象
        if (!meta.contains("parameters") || !meta["parameters"].is_object()) {
            throw std::runtime_error("Config file does not contain a valid 'parameters' object.");
        }

        const auto& paramsJson = meta["parameters"];
        for (const auto& [key, value] : paramsJson.items()) {
            params[key] = ConvertJsonToScalar(value);
        }
    }

    void LoadInputTensors()
    {
        // 检查meta中是否存在"inputs"对象
        if (!meta.contains("inputs") || !meta["inputs"].is_object()) {
            throw std::runtime_error("Config file does not contain a valid 'inputs' object.");
        }

        const auto& inputsJson = meta["inputs"];
        size_t index = 0;
        for (const auto& [tensorName, tensorConfig] : inputsJson.items()) {
            // 检查是否存在"tensorConfig"对象
            if (!tensorConfig.is_object()) {
                throw std::runtime_error("Invalid tensor configuration for tensor: " + tensorName);
            }

            // 检查bin格式
            std::string binFile = tensorConfig["bin_file"];
            if (binFile.empty()) {
                throw std::runtime_error("Missing 'bin_file' in tensor configuration: " + tensorName);
            }
            binFile = GetFullPath(binFile);

            // 检查shape格式
            Shape shape;
            if (tensorConfig.contains("shape") && tensorConfig["shape"].is_array()) {
                shape = tensorConfig["shape"].get<Shape>();
            } else {
                throw std::runtime_error("Missing or invalid 'shape' in tensor configuration: " + tensorName);
            }

            // 检查TileOpFormat格式
            TileOpFormat opFormat;
            if (tensorConfig.contains("opFormat") && tensorConfig["opFormat"].is_number_integer()) {
                int value = tensorConfig["opFormat"].get<int>();
                if (IsValidWeightFormat(value)) {
                    opFormat = static_cast<TileOpFormat>(value);
                } else {
                    throw std::runtime_error("Undefined 'opFormat' value in tensor configuration: '" + tensorName +
                                             "': " + std::to_string(value));
                }
            } else {
                throw std::runtime_error("Missing or invalid 'opFormat' in tensor configuration: " + tensorName);
            }

            // 检查Dtype
            std::string dtypeStr;
            if (tensorConfig.contains("dtype") && tensorConfig["dtype"].is_string()) {
                dtypeStr = tensorConfig["dtype"];
            }

            auto [tensor, dataPtr] = CreateTensor(tensorName, dtypeStr, shape, binFile, opFormat);

            auto [it, is_inserted] = inputTensors.emplace(tensorName, tensor);
            ASSERT(is_inserted);
            inputTensorList.push_back(std::cref(it->second));
            this->inputDataList.push_back(dataPtr);
            this->inputNameToIdx.emplace(tensorName, index++);
        }
    }

    void LoadGoldenTensors()
    {
        size_t index = 0;
        for (auto& [name, value] : this->meta["golden_outputs"].items()) {
            auto binFile = value["bin_file"].get<std::string>();
            auto shape = value["shape"].get<Shape>();
            auto dtype = value["dtype"].get<std::string>();
            binFile = GetFullPath(binFile);

            auto [tensor, dataPtr] = CreateTensor(name, dtype, shape, binFile);
            (void)tensor;
            this->goldens.emplace(name, dataPtr);

            auto [output, outputData] = CreateTensor(name, dtype, shape, std::string());
            auto [it, is_inserted] = this->outputTensors.emplace(name, output);
            ASSERT(is_inserted);
            this->outputTensorList.push_back(std::cref(it->second));
            this->outputDataList.push_back(outputData);
            this->outputNameToIdx.emplace(name, index++);
        }
    }

    std::string GetFullPath(const std::string& relativePath)
    {
        size_t pos = this->configPath.find_last_of('/');
        ASSERT(pos != std::string::npos);
        std::string fullPath = this->configPath.substr(0, pos + 1) + relativePath;
        return fullPath;
    }

    nlohmann::json meta;
    std::string configPath;
    std::unordered_map<std::string, Scalar> params;
    std::unordered_map<std::string, Tensor> inputTensors;  // 输入 Tensor
    std::unordered_map<std::string, Tensor> outputTensors; // 接受计算结果 Tensor
    std::unordered_map<std::string, RawTensorDataPtr> goldens;

    std::unordered_map<std::string, int> inputNameToIdx;               // 用于查找 tensor 对应的数据
    std::vector<std::reference_wrapper<const Tensor>> inputTensorList; // 保证 TensorList 和 DataList 顺序一致
    std::vector<RawTensorDataPtr> inputDataList;
    std::unordered_map<std::string, int> outputNameToIdx; // 用于查找 tensor 对应的数据
    std::vector<std::reference_wrapper<const Tensor>> outputTensorList;
    std::vector<RawTensorDataPtr> outputDataList;
};

template <typename T>
RawTensorDataPtr TestDataLoader::CreateTensorData(const Tensor& tensor, const std::string& fileName)
{
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    if (!fileName.empty()) {
        readInput<T>(fileName, values);
    }
    return RawTensorData::CreateTensor<T>(tensor, values);
}
