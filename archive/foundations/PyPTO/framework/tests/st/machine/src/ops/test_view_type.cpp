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
 * \file test_view_type.cpp
 * \brief
 */
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"
#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "tilefwk/tensor.h"
#include "operator/models/nsa/view_type.h"

using namespace npu::tile_fwk;

class ViewType : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName)
{
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> GetGoldenVec(std::vector<int64_t> shape, std::string fileName)
{
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename InputT, typename OutputT>
void ViewTypeEntry(const std::vector<int64_t>& mkn)
{
    int64_t m = mkn[0];
    int64_t k = mkn[1];
    int64_t n = mkn[2];

    DataType originDtype = GetAstDtype<InputT>();
    DataType dstDtype = GetAstDtype<OutputT>();
    float factor = (float)BytesOf(originDtype) / (float)BytesOf(dstDtype);

    std::vector<int64_t> xShape = {m, k, n};
    std::vector<int64_t> resultShape = {m, k, int(n * factor)};

    Tensor x(originDtype, xShape, "x");
    Tensor result(dstDtype, resultShape, "result");

    auto goldenData = GetGoldenVec<OutputT>(resultShape, "/result.bin");
    auto xData = CreateTensorData<InputT>(x, "/x.bin");

    auto resultData = RawTensorData::CreateConstantTensor<OutputT>(result, 0.0);

    std::vector<RawTensorDataPtr> inputDataList = {xData};
    std::vector<RawTensorDataPtr> outputDataList = {resultData};

    ViewTypeFunc(x, result, dstDtype);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "====== result ======" << std::endl;
    EXPECT_TRUE(resultCmp<OutputT>(goldenData, (OutputT*)resultData->data(), 0.005f));
#endif
}

template <typename InputT, typename OutputT>
void ViewTypeQuantTestEntry(const std::vector<int64_t>& mkn)
{
    int64_t m = mkn[0];
    int64_t k = mkn[1];
    int64_t n = mkn[2];

    DataType originDtype = GetAstDtype<InputT>();
    DataType dstDtype = GetAstDtype<OutputT>();

    std::vector<int64_t> xShape = {m, k, n};
    std::vector<int64_t> resultShape = {m, k, n + 16};

    Tensor x(originDtype, xShape, "x");
    Tensor result(dstDtype, resultShape, "result");

    auto goldenData = GetGoldenVec<OutputT>(resultShape, "/result.bin");
    auto xData = CreateTensorData<InputT>(x, "/x.bin");

    auto resultData = RawTensorData::CreateConstantTensor<OutputT>(result, 0.0);

    std::vector<RawTensorDataPtr> inputDataList = {xData};
    std::vector<RawTensorDataPtr> outputDataList = {resultData};

    ViewTypeQuantTestFunc(x, result);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "====== result ======" << std::endl;
    EXPECT_TRUE(resultCmp<OutputT>(goldenData, (OutputT*)resultData->data(), 0.005f));
#endif
}

template <typename InputT, typename OutputT>
void ViewTypeDequantTestEntry(const std::vector<int64_t>& mkn)
{
    int64_t m = mkn[0];
    int64_t k = mkn[1];
    int64_t n = mkn[2];

    DataType originDtype = GetAstDtype<InputT>();
    DataType dstDtype = GetAstDtype<OutputT>();

    std::vector<int64_t> xShape = {m, k, n};
    std::vector<int64_t> resultShape = {m, k, 512 + 64};

    Tensor x(originDtype, xShape, "x");
    Tensor result(dstDtype, resultShape, "result");

    auto goldenData = GetGoldenVec<OutputT>(resultShape, "/result.bin");
    auto xData = CreateTensorData<InputT>(x, "/x.bin");

    auto resultData = RawTensorData::CreateConstantTensor<OutputT>(result, 0.0);

    std::vector<RawTensorDataPtr> inputDataList = {xData};
    std::vector<RawTensorDataPtr> outputDataList = {resultData};

    ViewTypeDequantTestFunc(x, result);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "====== result ======" << std::endl;
    EXPECT_TRUE(resultCmp<OutputT>(goldenData, (OutputT*)resultData->data(), 0.005f));
#endif
}

TEST_F(ViewType, int8_2_float32)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<int8_t, float>(mkn);
}

TEST_F(ViewType, int8_2_bfloat16)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<int8_t, bfloat16>(mkn);
}

TEST_F(ViewType, int8_2_float16)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<int8_t, float16>(mkn);
}

TEST_F(ViewType, float32_2_int8)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<float, int8_t>(mkn);
}

TEST_F(ViewType, bfloat16_2_int8)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<bfloat16, int8_t>(mkn);
}

TEST_F(ViewType, float16_2_int8)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<float16, int8_t>(mkn);
}

TEST_F(ViewType, float16_2_float32)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<float16, float>(mkn);
}

TEST_F(ViewType, float32_2_float16)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<float, float16>(mkn);
}

TEST_F(ViewType, bfloat16_2_float32)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<bfloat16, float>(mkn);
}

TEST_F(ViewType, float32_2_bfloat16)
{
    std::vector<int64_t> mkn = {4, 32, 1024};
    ViewTypeEntry<float, bfloat16>(mkn);
}

TEST_F(ViewType, quant_test_bf16_2_int8)
{
    std::vector<int64_t> mkn = {64, 1, 512};
    ViewTypeQuantTestEntry<bfloat16, int8_t>(mkn);
}

TEST_F(ViewType, dequant_test_bf16_2_int8)
{
    std::vector<int64_t> mkn = {2048, 1, 656};
    ViewTypeDequantTestEntry<int8_t, bfloat16>(mkn);
}
