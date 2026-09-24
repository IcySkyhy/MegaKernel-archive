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
 * \file test_operation.h
 * \brief
 */

#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <functional>

#include "test_suite_stest_ops.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"
#include "interface/tensor/float.h"
#include "cost_model/simulation/cost_model_launcher.h"

namespace tile_fwk {
namespace test_operation {
constexpr int perchannel = 2;

struct OpFuncArgs {
    std::unordered_map<size_t, size_t> inplaceInfo;
};

inline SymbolicScalar CeilDivSymbolicScalar(SymbolicScalar a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

using OpFunc = std::function<void(const std::vector<Tensor>&, std::vector<Tensor>&, const OpFuncArgs*)>;

struct TestCaseDesc {
    std::vector<Tensor> inputTensors;
    std::vector<Tensor> outputTensors;
    std::vector<std::string> inputPaths;
    std::vector<std::string> goldenPaths;
    const OpFuncArgs* args;
    OpFunc opFunc;
    bool onBoard{true};
};

struct MatmulTestCaseParam {
    bool transA = false;
    bool transB = false;
    bool isAMatrixNz = false;
    bool isBMatrixNz = false;
    bool isCMatrixNz = false;
    DataType outDtype = DT_FP32;
    bool enableKSplit = false;
    float scaleValue = 0.0f;
    int reluTypeInt = 0;
    bool hasScale = false;
    bool hasBias = false;
    bool l0c2l1IsTrans = false;
    bool l0c2l1AsLeftMatrix = false;
    bool l0c2l1IsNz = false;
    bool l0c2l1TmpIsTrans = false;
    bool enable_l0c2l1 = false;
};

class TestExecutor {
public:
    static void setGMNotClear() { gmClearFlag = false; }
    static void runTest(const TestCaseDesc& testCase)
    {
        init();
        verifyOpResults(testCase);
    }

private:
    static inline bool gmClearFlag = true;
    static void init() {}

    static bool IsFp4PackedDtype(DataType dtype)
    {
        return dtype == DataType::DT_FP4_E2M1X2 || dtype == DataType::DT_FP4_E1M2X2;
    }

    static size_t GetTensorStorageElementCount(const Tensor& tensor)
    {
        size_t elementCount = 1;
        const auto& shape = tensor.GetShape();
        for (int dim : shape) {
            elementCount *= dim;
        }
        if (!shape.empty() && IsFp4PackedDtype(tensor.GetDataType())) {
            elementCount = (elementCount / shape.back()) * ((shape.back() + 1) / 2);
        }
        return elementCount;
    }

    static void verifyOpResults(const TestCaseDesc& testCase)
    {
        // 设置输入数据
        std::vector<RawTensorDataPtr> inputs;
        ASSERT_EQ(testCase.inputTensors.size(), testCase.inputPaths.size());
        for (size_t i = 0; i < testCase.inputTensors.size(); ++i) {
            if (testCase.inputTensors[i].GetStorage() == nullptr) {
                inputs.push_back(nullptr);
                continue;
            }
            const size_t elementCount = GetTensorStorageElementCount(testCase.inputTensors[i]);
            std::vector<uint8_t> input(elementCount * BytesOf(testCase.inputTensors[i].GetDataType()), 0);
            readInput<uint8_t>(testCase.inputPaths[i], input);
            inputs.push_back(RawTensorData::CreateTensor(testCase.inputTensors[i], input));
        }
        ProgramData::GetInstance().AppendInputs({inputs});

        // 设置输出Tensor
        std::vector<RawTensorDataPtr> outputs;
        for (size_t i = 0; i < testCase.outputTensors.size(); i++) {
            if (testCase.args->inplaceInfo.find(i) != testCase.args->inplaceInfo.end()) {
                outputs.push_back(inputs[testCase.args->inplaceInfo.at(i)]);
            } else {
                if (gmClearFlag) {
                    outputs.push_back(RawTensorData::CreateTensorZero(testCase.outputTensors[i]));
                } else {
                    switch (testCase.outputTensors[i].GetDataType()) {
                        case DataType::DT_FP32:
                            outputs.push_back(
                                RawTensorData::CreateConstantTensor<float>(testCase.outputTensors[i], 1.0));
                            break;
                        case DataType::DT_INT32:
                            outputs.push_back(
                                RawTensorData::CreateConstantTensor<int32_t>(testCase.outputTensors[i], 1));
                            break;
                        default:
                            ASSERT_TRUE(false) << "no support dtype " << testCase.outputTensors[i].GetDataType();
                            break;
                    }
                }
            }
        }
        ProgramData::GetInstance().AppendOutputs({outputs});

        std::vector<Tensor> nonConstOutputs = testCase.outputTensors;
        testCase.opFunc(testCase.inputTensors, nonConstOutputs, testCase.args);

        if (testCase.onBoard) {
            DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
        } else {
            config::SetRuntimeOption(CFG_RUN_MODE, CFG_RUN_MODE_SIM);
            CostModelLauncher::CostModelRunOnce(Program::GetInstance().GetLastFunction());
        }

        ASSERT_EQ(testCase.goldenPaths.size(), testCase.outputTensors.size());
        readGoldenCmpType(testCase);
    }

    static void readGoldenCmpType(const TestCaseDesc& testCase)
    {
        for (size_t i = 0; i < testCase.outputTensors.size(); ++i) {
            if (readGoldenCmpFloatType(testCase, i) || readGoldenCmpIntType(testCase, i) ||
                readGoldenCmpUintType(testCase, i) || readGoldenCmpPackedType(testCase, i)) {
                continue;
            }
            ASSERT_TRUE(false) << "no support dtype " << testCase.outputTensors[i].GetDataType();
        }
    }

    static bool readGoldenCmpFloatType(const TestCaseDesc& testCase, size_t i)
    {
        auto& tensor = testCase.outputTensors[i];
        switch (tensor.GetDataType()) {
            case DataType::DT_FP32:
                readGoldenCmp<float>(tensor, testCase.goldenPaths[i], i, 0.005f);
                return true;
            case DataType::DT_FP16:
                readGoldenCmp<npu::tile_fwk::float16>(tensor, testCase.goldenPaths[i], i, 0.005f);
                return true;
            case DataType::DT_BF16:
                readGoldenCmp<npu::tile_fwk::bfloat16>(tensor, testCase.goldenPaths[i], i, 0.005f);
                return true;
            default:
                return false;
        }
    }

    static bool readGoldenCmpIntType(const TestCaseDesc& testCase, size_t i)
    {
        auto& tensor = testCase.outputTensors[i];
        switch (tensor.GetDataType()) {
            case DataType::DT_INT8:
                readGoldenCmp<int8_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            case DataType::DT_INT16:
                readGoldenCmp<int16_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            case DataType::DT_INT32:
                readGoldenCmp<int32_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            case DataType::DT_INT64:
                readGoldenCmp<int64_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            default:
                return false;
        }
    }

    static bool readGoldenCmpUintType(const TestCaseDesc& testCase, size_t i)
    {
        auto& tensor = testCase.outputTensors[i];
        switch (tensor.GetDataType()) {
            case DataType::DT_BOOL:
            case DataType::DT_UINT8:
                readGoldenCmp<uint8_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            case DataType::DT_UINT16:
                readGoldenCmp<uint16_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            case DataType::DT_UINT32:
                readGoldenCmp<uint32_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            case DataType::DT_UINT64:
                readGoldenCmp<uint64_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            default:
                return false;
        }
    }

    static bool readGoldenCmpPackedType(const TestCaseDesc& testCase, size_t i)
    {
        auto& tensor = testCase.outputTensors[i];
        switch (tensor.GetDataType()) {
            case DataType::DT_HF8:
            case DataType::DT_FP8E4M3:
            case DataType::DT_FP8E5M2:
            case DataType::DT_FP8E8M0:
            case DataType::DT_FP4_E2M1X2:
            case DataType::DT_FP4_E1M2X2:
                readGoldenCmp<uint8_t>(tensor, testCase.goldenPaths[i], i, 0);
                return true;
            default:
                return false;
        }
    }

    template <typename T>
    static void readGoldenCmp(const Tensor& tensor, const std::string& goldenPath, size_t index, T tolerance)
    {
        const size_t elementCount = GetTensorStorageElementCount(tensor);
        std::vector<T> goldenOutput(elementCount, 0);
        readInput<T>(goldenPath, goldenOutput);
        auto actualData = ProgramData::GetInstance().GetOutputData(index);
        const T* actual = (T*)actualData->data();
        int ret = resultCmp(goldenOutput, actual, tolerance);
        EXPECT_EQ(ret, true);
    }
};

class TestFlowVerifier {
public:
    static void runTest(const TestCaseDesc& testCase)
    {
        init();
        verifyOpResults(testCase);
    }

private:
    static void init()
    {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    }

    static void verifyOpResults(const TestCaseDesc& testCase)
    {
        // 设置输出Tensor
        std::vector<RawTensorDataPtr> outputs;
        for (const auto& tensor : testCase.outputTensors) {
            outputs.push_back(RawTensorData::CreateTensorZero(tensor));
        }
        ProgramData::GetInstance().AppendOutputs({outputs});

        // 设置输入数据
        ASSERT_EQ(testCase.inputTensors.size(), testCase.inputPaths.size());
        std::vector<RawTensorDataPtr> inputs;
        for (size_t i = 0; i < testCase.inputTensors.size(); ++i) {
            if (testCase.inputTensors[i].GetStorage() == nullptr) {
                inputs.push_back(nullptr);
                continue;
            }
            size_t elementCount = 1;
            for (int dim : testCase.inputTensors[i].GetShape()) {
                elementCount *= dim;
            }
            std::vector<uint8_t> inputValue(elementCount * BytesOf(testCase.inputTensors[i].GetDataType()), 0);
            readInput<uint8_t>(testCase.inputPaths[i], inputValue);
            inputs.push_back(RawTensorData::CreateTensor(testCase.inputTensors[i], inputValue));
        }
        ProgramData::GetInstance().AppendInputs({inputs});

        ASSERT_EQ(testCase.goldenPaths.size(), testCase.outputTensors.size());
        appendGoldenType(testCase);

        std::vector<Tensor> nonConstOutputs = testCase.outputTensors;
        testCase.opFunc(testCase.inputTensors, nonConstOutputs, testCase.args);
    }

    static void appendGoldenType(const TestCaseDesc& testCase)
    {
        for (size_t i = 0; i < testCase.outputTensors.size(); ++i) {
            auto& tensor = testCase.outputTensors[i];
            switch (tensor.GetDataType()) {
                case DataType::DT_FP32:
                    appendGolden<float>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_FP16:
                    appendGolden<npu::tile_fwk::float16>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_BF16:
                    appendGolden<npu::tile_fwk::bfloat16>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_INT8:
                    appendGolden<int8_t>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_INT16:
                    appendGolden<int16_t>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_INT32:
                    appendGolden<int32_t>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_INT64:
                    appendGolden<int64_t>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_UINT8:
                    appendGolden<uint8_t>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_UINT16:
                    appendGolden<uint16_t>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_UINT32:
                    appendGolden<uint32_t>(tensor, testCase.goldenPaths[i]);
                    break;
                case DataType::DT_UINT64:
                    appendGolden<uint64_t>(tensor, testCase.goldenPaths[i]);
                    break;
                default:
                    ASSERT_TRUE(false) << "no support dtype " << tensor.GetDataType();
                    break;
            }
        }
    }

    template <typename T>
    static void appendGolden(const Tensor& tensor, const std::string& goldenPath)
    {
        size_t elementCount = 1;
        for (int dim : tensor.GetShape()) {
            elementCount *= dim;
        }
        std::vector<T> goldenOutput(elementCount, 0);
        readInput<T>(goldenPath, goldenOutput);
        ProgramData::GetInstance().AppendGoldens({
            npu::tile_fwk::RawTensorData::CreateTensor<T>(tensor, goldenOutput),
        });
    }
};

static DataType GetDataType(const std::string& name)
{
    static const std::map<std::string, DataType> name_to_dtype = {
        {"int4", DataType::DT_INT4},         {"int8", DataType::DT_INT8},
        {"int16", DataType::DT_INT16},       {"int32", DataType::DT_INT32},
        {"int64", DataType::DT_INT64},       {"fp8", DataType::DT_FP8},
        {"fp16", DataType::DT_FP16},         {"fp32", DataType::DT_FP32},
        {"bf16", DataType::DT_BF16},         {"hf4", DataType::DT_HF4},
        {"hf8", DataType::DT_HF8},           {"uint8", DataType::DT_UINT8},
        {"uint16", DataType::DT_UINT16},     {"uint32", DataType::DT_UINT32},
        {"uint64", DataType::DT_UINT64},     {"bool", DataType::DT_BOOL},
        {"double", DataType::DT_DOUBLE},     {"fp8e4m3", DataType::DT_FP8E4M3},
        {"fp8e5m2", DataType::DT_FP8E5M2},   {"fp8e8m0", DataType::DT_FP8E8M0},
        {"fp4_e2m1", DataType::DT_FP4_E2M1}, {"fp4_e2m1x2", DataType::DT_FP4_E2M1X2},
        {"fp4_e1m2", DataType::DT_FP4_E1M2}, {"fp4_e1m2x2", DataType::DT_FP4_E1M2X2},
    };
    if (name_to_dtype.find(name) == name_to_dtype.end()) {
        MATMUL_LOGW("Not support type %s yet, return fp32 as default.", name.c_str());
        return DataType::DT_FP32;
    }
    return name_to_dtype.at(name);
}

static std::vector<Tensor> GetTensors(const nlohmann::json& json_data, bool is_input = true)
{
    std::cout << "Create Tensors For " << json_data << std::endl;
    std::vector<Tensor> tensors;
    auto key = is_input ? "input_tensors" : "output_tensors";
    for (const auto& tensor_config : json_data.at(key)) {
        auto shape = tensor_config.at("shape").get<std::vector<int64_t>>();
        auto dtype = GetDataType(tensor_config.at("dtype").get<std::string>());
        auto name = tensor_config.at("name").get<std::string>();
        tensors.push_back(Tensor(dtype, shape, name));
    }
    return tensors;
}

[[maybe_unused]] static std::vector<Tensor> GetInputTensors(const nlohmann::json& json_data)
{
    return GetTensors(json_data, true);
}

[[maybe_unused]] static std::vector<Tensor> GetMatmulTensors(const nlohmann::json& json_data, const std::string key)
{
    std::cout << "Create Matmul Tensors For " << json_data << std::endl;
    std::vector<Tensor> tensors;
    for (const auto& tensor_config : json_data.at(key)) {
        auto shape = tensor_config.at("shape").get<std::vector<int64_t>>();
        auto dtype = GetDataType(tensor_config.at("dtype").get<std::string>());
        auto name = tensor_config.at("name").get<std::string>();
        auto format = tensor_config.at("format").get<std::string>();
        if (format == "ND") {
            std::cout << "Create ND Tensors" << std::endl;
            tensors.push_back(Tensor(dtype, shape, name));
        } else {
            std::cout << "Create NZ Tensors" << std::endl;
            tensors.push_back(Tensor(dtype, shape, name, TileOpFormat::TILEOP_NZ));
        }
    }
    return tensors;
}

[[maybe_unused]] static Tensor GetParamTensor(const nlohmann::json& json_data, const std::string key)
{
    std::cout << "Create Param Tensors For " << json_data << std::endl;
    if (json_data.at("params").find(key) == json_data.at("params").end()) {
        return Tensor();
    }
    const auto& tensor_config = json_data.at("params").at(key);
    auto format = tensor_config.at("format").get<std::string>();
    auto name = tensor_config.at("name").get<std::string>();
    auto dtype = GetDataType(tensor_config.at("dtype").get<std::string>());
    auto shape = tensor_config.at("shape").get<std::vector<int64_t>>();
    if (format == "NZ") {
        std::cout << "Create NZ Tensors" << std::endl;
        return Tensor(dtype, shape, name, TileOpFormat::TILEOP_NZ);
    } else {
        std::cout << "Create ND Tensors" << std::endl;
        return Tensor(dtype, shape, name);
    }
}

[[maybe_unused]] static std::vector<Tensor> GetOutputTensors(const nlohmann::json& json_data)
{
    return GetTensors(json_data, false);
}

template <typename T>
T GetValueByName(const nlohmann::json& json_data, const std::string& name)
{
    nlohmann::json data = json_data;
    if (json_data.find(name) == json_data.end()) {
        data = json_data.at("params");
    }
    ASSERT(data.find(name) != data.end()) << "failed to load " << name << " in " << json_data << "!";
    return data.at(name).get<T>();
}

template <typename T>
T GetValueByNameWithKey(const nlohmann::json& json_data, const std::string& name, const std::string& key)
{
    nlohmann::json data = json_data;
    if (json_data.find(name) == json_data.end()) {
        data = json_data.at("params").at(key);
    }
    ASSERT(data.find(name) != data.end()) << "failed to load " << name << " in " << json_data << "!";
    return data.at(name).get<T>();
}

template <typename T1, typename T2>
T2 GetMapValByName(const std::map<T1, T2>& map_data, const T1& name)
{
    auto it = map_data.find(name);
    if (it != map_data.end()) {
        return it->second;
    }
    ASSERT(0) << "failed to get map val: " << name;
    return T2(0);
}

[[maybe_unused]] static std::vector<int64_t> GetViewShape(const nlohmann::json& json_data)
{
    return GetValueByName<std::vector<int64_t>>(json_data, "view_shape");
}

[[maybe_unused]] static std::vector<int64_t> GetTileShape(const nlohmann::json& json_data)
{
    return GetValueByName<std::vector<int64_t>>(json_data, "tile_shape");
}

[[maybe_unused]] static int GetFuncId(const nlohmann::json& json_data)
{
    return GetValueByName<int>(json_data, "func_id");
}

[[maybe_unused]] static std::vector<std::vector<int64_t>> GetMatmulTileShape(const nlohmann::json& json_data)
{
    std::vector<std::vector<int64_t>> tileShape;
    for (const auto& shape : json_data["tile_shape"]) {
        std::vector<int64_t> tile;
        for (const auto& num : shape) {
            tile.push_back(num);
        }
        tileShape.push_back(tile);
    }
    return tileShape;
}

[[maybe_unused]] static Element GetElementByType(DataType dataType, nlohmann::json test_data, const std::string& name)
{
    if (dataType == DT_FP32 || dataType == DT_BF16 || dataType == DT_FP16) {
        {
            Element element(dataType, GetValueByName<float>(test_data, name));
            return element;
        }
    } else if (dataType == DT_INT8) {
        Element element(dataType, GetValueByName<int8_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT16) {
        Element element(dataType, GetValueByName<int16_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT32) {
        Element element(dataType, GetValueByName<int32_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT64) {
        Element element(dataType, GetValueByName<int64_t>(test_data, name));
        return element;
    } else {
        std::string errorMessage = "Unsupported DataType " + std::string(DataType2String(dataType));
        throw std::invalid_argument(errorMessage.c_str());
    }
}

[[maybe_unused]] static MatmulTestCaseParam GetMatmulParam(const nlohmann::json& json_data)
{
    MatmulTestCaseParam param;
    param.transA = json_data.at("input_tensors")[0].at("need_trans");
    param.transB = json_data.at("input_tensors")[1].at("need_trans");
    param.isAMatrixNz = json_data.at("input_tensors")[0].at("format") == "NZ";
    param.isBMatrixNz = json_data.at("input_tensors")[1].at("format") == "NZ";
    param.isCMatrixNz = json_data.at("output_tensors")[0].at("format") == "NZ";
    param.outDtype = GetDataType(json_data.at("output_tensors")[0].at("dtype"));
    if (json_data.at("params").find("enableKSplit") != json_data.at("params").end()) {
        param.enableKSplit = GetValueByName<bool>(json_data, "enableKSplit");
    }
    if (json_data.at("params").find("relu_type") != json_data.at("params").end()) {
        param.reluTypeInt = GetValueByName<int>(json_data, "relu_type");
    }
    if (json_data.at("params").find("scale_value") != json_data.at("params").end()) {
        param.scaleValue = GetValueByName<float>(json_data, "scale_value");
    }
    if (json_data.at("params").find("quant_type") != json_data.at("params").end() &&
        GetValueByName<int>(json_data, "quant_type") == perchannel) {
        param.hasScale = true;
    }
    if (json_data.at("params").find("bias_info") != json_data.at("params").end() &&
        GetValueByName<std::string>(json_data, "bias_info") != "") {
        param.hasBias = true;
    }
    if (json_data.at("params").find("l0c2l1_params") != json_data.at("params").end()) {
        if (json_data.at("params").at("l0c2l1_params").find("is_l0c2l1_trans") !=
            json_data.at("params").at("l0c2l1_params").end()) {
            param.l0c2l1IsTrans = GetValueByNameWithKey<bool>(json_data, "is_l0c2l1_trans", "l0c2l1_params");
            param.enable_l0c2l1 = true;
        }
        if (json_data.at("params").at("l0c2l1_params").find("is_as_left_matrix") !=
            json_data.at("params").at("l0c2l1_params").end()) {
            param.l0c2l1AsLeftMatrix = GetValueByNameWithKey<bool>(json_data, "is_as_left_matrix", "l0c2l1_params");
        }
    }

    return param;
}

template <typename T, size_t func_offset = 2>
std::vector<T> GetOpMetaData(const std::vector<OpFunc>& opFuncs, const std::string& op)
{
    auto case_file = "../../../framework/tests/st/operation/test_case/" + op + "_st_test_cases.json";
    std::ifstream json_file(case_file);
    if (!json_file.is_open()) {
        MATMUL_LOGI("Not find any input data for %s.", case_file.c_str());
        return {};
    }
    nlohmann::json json_data = nlohmann::json::parse(json_file);
    std::vector<T> test_case_list;
    for (const auto& test_case : json_data.at("test_cases")) {
        if (test_case.at("operation") != op) {
            continue;
        }
        auto func_id = GetFuncId(test_case);
        if (func_id < 0 || static_cast<size_t>(func_id) >= opFuncs.size()) {
            if (GetViewShape(test_case).size() < func_offset) { // cut function start from 2 dim
                func_id = 0;
            } else {
                func_id = GetViewShape(test_case).size() - func_offset;
            }
        }
        test_case_list.push_back(T(opFuncs[func_id], test_case));
    }
    return test_case_list;
}

template <typename T>
TestCaseDesc CreateTestCaseDesc(const T& param, const OpFuncArgs* args)
{
    TestCaseDesc testCase;
    auto test_data = param.test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    testCase.args = args;
    testCase.opFunc = param.opFunc_;
    std::transform(testCase.inputTensors.begin(), testCase.inputTensors.end(), std::back_inserter(testCase.inputPaths),
                   [](const auto& tensor) { return GetGoldenDir() + "/" + tensor.GetStorage()->Symbol() + ".bin"; });
    std::transform(testCase.outputTensors.begin(), testCase.outputTensors.end(),
                   std::back_inserter(testCase.goldenPaths),
                   [](const auto& tensor) { return GetGoldenDir() + "/" + tensor.GetStorage()->Symbol() + ".bin"; });
    auto params_dict = test_data.at("params");
    testCase.onBoard = params_dict.find("on_board") == params_dict.end() || GetValueByName<bool>(test_data, "on_board");
    return testCase;
}
} // namespace test_operation
} // namespace tile_fwk
