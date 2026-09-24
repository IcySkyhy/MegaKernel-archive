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
 * \file test_interp_calc_matmul_sort.cpp
 * \brief Interpreter calc unit tests (matmul/reduce/sort/quantize).
 */

#include "test_interp_calc_utils.h"

namespace npu::tile_fwk {
namespace {

void RunMatMulBt(float goldenVal, int kStep)
{
    auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
    auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
    auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
    auto golden = makeTensorData(DT_FP32, {8, 8}, goldenVal);
    MatMulParam param{};
    param.aTrans = false;
    param.bTrans = true;
    param.kStep = kStep;
    calc::MatMul(out, self, other, param);
    ASSERT_ALLCLOSE(out, golden);
}

void RunAccMatMulBt(float goldenVal, int kStep)
{
    auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
    auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
    auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
    auto golden = makeTensorData(DT_FP32, {8, 8}, goldenVal);
    MatMulParam param{};
    param.aTrans = false;
    param.bTrans = true;
    param.kStep = kStep;
    calc::AccMatMul(out, self, other, out, param);
    ASSERT_ALLCLOSE(out, golden);
}

template <typename CalcFn>
void RunArgReduceWithValue(const std::vector<float>& gvdata, const std::vector<int32_t>& gidata, CalcFn calcFn)
{
    std::vector<float> sdata = {3.0f, 1.0f, 4.0f, 2.0f};
    auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
    auto outValue = makeTensorData(DT_FP32, {2, 1}, 0.0f);
    auto outIndex = makeTensorData(DT_INT32, {2, 1}, 0);
    auto outTemp = makeTensorData(DT_FP32, {2, 2}, 0.0f);
    auto goldenValue = makeTensorData(DT_FP32, {2, 1}, gvdata);
    auto goldenIndex = makeTensorData(DT_INT32, {2, 1}, gidata);
    calcFn(outValue, outIndex, outTemp, self, -1);
    ASSERT_ALLCLOSE(outValue, goldenValue);
    ASSERT_ALLCLOSE(outIndex, goldenIndex);
}

void RunMiscViewTest()
{
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            auto v = out->View({4, 4}, {i * 4, j * 4});
            calc::ExpandS(v, Element(DT_FP32, i * 2.0f + j));
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            auto v = out->View({4, 4}, {i * 4, j * 4});
            auto g = makeTensorData(DT_FP32, {4, 4}, i * 2.0f + j);
            ASSERT(calc::AllClose(v, g)) << v << "\n" << g;
        }
    }
}

} // namespace

TEST_F(TorchAdaptorTest, MatMul)
{
    {
        // matmul
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        MatMulParam param{};
        param.aTrans = false;
        param.bTrans = false;
        param.kStep = 4;
        calc::MatMul(out, self, other, param);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, MatMulBt)
{
    RunMatMulBt(16.0f, 0);
    RunMatMulBt(16.0f, 4);
}

TEST_F(TorchAdaptorTest, MatMulAcc)
{
    {
        // matmul acc
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        calc::AccMatMul(out, self, other, out);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        MatMulParam param{};
        param.aTrans = false;
        param.bTrans = false;
        param.kStep = 4;
        calc::AccMatMul(out, self, other, out, param);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, MatMulAccBt)
{
    RunAccMatMulBt(17.0f, 0);
    RunAccMatMulBt(17.0f, 4);
}

TEST_F(TorchAdaptorTest, MatMulFp16)
{
    {
        // matmul fp16 @ fp16 -> fp32
        auto self = makeTensorData(DT_FP16, {8, 16}, float16(1.0));
        auto other = makeTensorData(DT_FP16, {16, 8}, float16(1.0));
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul fp16 @ fp16 -> fp16
        auto self = makeTensorData(DT_FP16, {8, 16}, float16(1.0));
        auto other = makeTensorData(DT_FP16, {16, 8}, float16(1.0));
        auto out = makeTensorData(DT_FP16, {8, 8}, float16(1.0f));
        auto golden = makeTensorData(DT_FP16, {8, 8}, float16(16.0f));
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, ReduceSum)
{
    {
        // sum expand
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 16.0f);
        calc::RowSumExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sum
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 1}, 16.0f);
        calc::RowSumSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, ReduceMin)
{
    {
        // min expand
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {1.0, 1.0, 4.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::RowMinExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // minsingle
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {1.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 1}, gdata);
        calc::RowMinSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // minline
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {1, 16}, 1.0f);
        calc::RowMinLine(out, self, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, ReduceMax)
{
    {
        // max expand
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {2.0, 2.0, 5.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::RowMaxExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // maxsingle
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {2.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 1}, gdata);
        calc::RowMaxSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // maxline
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {1, 16}, 1.0f);
        calc::RowMaxLine(out, self, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, ReduceProd)
{
    {
        // prodsingle
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 1}, 1.0f);
        calc::RowProdSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // prodline
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {1, 16}, 1.0f);
        calc::RowProdLine(out, self, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, ReduceAcc)
{
    // reduce acc
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    calc::ReduceAcc(out, {self, self, self, self});
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ArgReduceSingle)
{
    {
        std::vector<float> sdata = {1.0f, 5.0f, 3.0f, 2.0f, 8.0f, 4.0f, 7.0f, 6.0f};
        std::vector<int32_t> gdata = {1, 0};
        auto self = makeTensorData(DT_FP32, {2, 4}, sdata);
        auto out = makeTensorData(DT_INT32, {2, 1}, 0);
        auto golden = makeTensorData(DT_INT32, {2, 1}, gdata);
        calc::RowArgMaxSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        std::vector<float> sdata = {1.0f, 5.0f, 3.0f, 2.0f, 8.0f, 4.0f, 7.0f, 6.0f};
        std::vector<int32_t> gdata = {0, 1};
        auto self = makeTensorData(DT_FP32, {2, 4}, sdata);
        auto out = makeTensorData(DT_INT32, {2, 1}, 0);
        auto golden = makeTensorData(DT_INT32, {2, 1}, gdata);
        calc::RowArgMinSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, ArgReduceWithValueSingle)
{
    RunArgReduceWithValue({3.0f, 4.0f}, {0, 0}, calc::RowArgMaxWithValueSingle);
    RunArgReduceWithValue({1.0f, 2.0f}, {1, 1}, calc::RowArgMinWithValueSingle);
}

TEST_F(TorchAdaptorTest, ArgReduceWithValueLine)
{
    RunArgReduceWithValue({3.0f, 4.0f}, {0, 0}, calc::RowArgMaxWithValueLine);
    RunArgReduceWithValue({1.0f, 2.0f}, {1, 1}, calc::RowArgMinWithValueLine);
}

TEST_F(TorchAdaptorTest, PairArgReduce)
{
    {
        std::vector<float> v1data = {5.0f, 2.0f};
        std::vector<int32_t> i1data = {10, 20};
        std::vector<float> v2data = {3.0f, 8.0f};
        std::vector<int32_t> i2data = {30, 40};
        std::vector<float> gvdata = {5.0f, 8.0f};
        std::vector<int32_t> gidata = {10, 40};
        auto value1 = makeTensorData(DT_FP32, {2, 1}, v1data);
        auto index1 = makeTensorData(DT_INT32, {2, 1}, i1data);
        auto value2 = makeTensorData(DT_FP32, {2, 1}, v2data);
        auto index2 = makeTensorData(DT_INT32, {2, 1}, i2data);
        auto outValue = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto outIndex = makeTensorData(DT_INT32, {2, 1}, std::vector<int32_t>{0, 0});
        auto goldenValue = makeTensorData(DT_FP32, {2, 1}, gvdata);
        auto goldenIndex = makeTensorData(DT_INT32, {2, 1}, gidata);
        calc::PairArgMax(outValue, outIndex, value1, index1, value2, index2);
        ASSERT_ALLCLOSE(outValue, goldenValue);
        ASSERT_ALLCLOSE(outIndex, goldenIndex);
    }
    {
        std::vector<float> v1data = {5.0f, 2.0f};
        std::vector<int32_t> i1data = {10, 20};
        std::vector<float> v2data = {3.0f, 8.0f};
        std::vector<int32_t> i2data = {30, 40};
        std::vector<float> gvdata = {3.0f, 2.0f};
        std::vector<int32_t> gidata = {30, 20};
        auto value1 = makeTensorData(DT_FP32, {2, 1}, v1data);
        auto index1 = makeTensorData(DT_INT32, {2, 1}, i1data);
        auto value2 = makeTensorData(DT_FP32, {2, 1}, v2data);
        auto index2 = makeTensorData(DT_INT32, {2, 1}, i2data);
        auto outValue = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto outIndex = makeTensorData(DT_INT32, {2, 1}, std::vector<int32_t>{0, 0});
        auto goldenValue = makeTensorData(DT_FP32, {2, 1}, gvdata);
        auto goldenIndex = makeTensorData(DT_INT32, {2, 1}, gidata);
        calc::PairArgMin(outValue, outIndex, value1, index1, value2, index2);
        ASSERT_ALLCLOSE(outValue, goldenValue);
        ASSERT_ALLCLOSE(outIndex, goldenIndex);
    }
}

TEST_F(TorchAdaptorTest, Permute3D)
{
    {
        std::vector<float> sdata = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::vector<float> gdata = {1,  2,  3,  4,  13, 14, 15, 16, 5,  6,  7,  8,
                                    17, 18, 19, 20, 9,  10, 11, 12, 21, 22, 23, 24};
        auto self = makeTensorData(DT_FP32, {2, 3, 4}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 3, 4}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 3, 4}, gdata);
        calc::Permute(out, self, {1, 0, 2});
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        std::vector<float> sdata = {1, 2, 3, 4, 5, 6};
        std::vector<float> gdata = {1, 4, 2, 5, 3, 6};
        auto self = makeTensorData(DT_FP32, {2, 3}, sdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Permute(out, self, {1, 0});
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, Misc)
{
    {
        // reshape
        std::vector<float> gdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, gdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Reshape(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // transpose
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, sdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Transpose(out, self, -1, -2);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // copy
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {3, 2}, gdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Copy(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // copy trans
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, sdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Copy(out, self, true);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, MiscView) { RunMiscViewTest(); }

TEST_F(TorchAdaptorTest, Pad)
{
    // Test 2D pad with constant value 0.0
    std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> gdata = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
    auto out = makeTensorData(DT_FP32, {3, 4}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {3, 4}, gdata);
    calc::Pad(out, self, Element(DT_FP32, 0.0f));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, FillPad2D)
{
    std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> gdata = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
    auto out = makeTensorData(DT_FP32, {3, 4}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {3, 4}, gdata);
    calc::FillPad(out, self, Element(DT_FP32, 0.0f));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, FillPad1D)
{
    std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> gdata = {1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto self = makeTensorData(DT_FP32, {4}, sdata);
    auto out = makeTensorData(DT_FP32, {12}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {12}, gdata);
    calc::FillPad(out, self, Element(DT_FP32, 0.0f));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, BitSortDescending)
{
    // 降序
    std::vector<float> sdata = {
        0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
        16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
        0.0,   10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0,  100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
        160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0};
    std::vector<float> gdata = {
        31.0,  31.0, 30.0,  30.0, 29.0,  29.0, 28.0,  28.0, 27.0,  27.0, 26.0,  26.0, 25.0,  25.0, 24.0,  24.0,
        23.0,  23.0, 22.0,  22.0, 21.0,  21.0, 20.0,  20.0, 19.0,  19.0, 18.0,  18.0, 17.0,  17.0, 16.0,  16.0,
        15.0,  15.0, 14.0,  14.0, 13.0,  13.0, 12.0,  12.0, 11.0,  11.0, 10.0,  10.0, 9.0,   9.0,  8.0,   8.0,
        7.0,   7.0,  6.0,   6.0,  5.0,   5.0,  4.0,   4.0,  3.0,   3.0,  2.0,   2.0,  1.0,   1.0,  0.0,   0.0,
        310.0, 31.0, 300.0, 30.0, 290.0, 29.0, 280.0, 28.0, 270.0, 27.0, 260.0, 26.0, 250.0, 25.0, 240.0, 24.0,
        230.0, 23.0, 220.0, 22.0, 210.0, 21.0, 200.0, 20.0, 190.0, 19.0, 180.0, 18.0, 170.0, 17.0, 160.0, 16.0,
        150.0, 15.0, 140.0, 14.0, 130.0, 13.0, 120.0, 12.0, 110.0, 11.0, 100.0, 10.0, 90.0,  9.0,  80.0,  8.0,
        70.0,  7.0,  60.0,  6.0,  50.0,  5.0,  40.0,  4.0,  30.0,  3.0,  20.0,  2.0,  10.0,  1.0,  0.0,   0.0};
    auto self = makeTensorData(DT_FP32, {2, 32}, sdata);
    auto out = makeTensorData(DT_FP32, {2, 64}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {2, 64}, gdata);
    calc::BitSort(out, self, -1, true, 0);
    ASSERT_ALLCLOSE(out->View({2, 64}, {0, 0}), golden);
}

TEST_F(TorchAdaptorTest, BitSortAscending)
{
    // 升序
    std::vector<float> sdata = {
        0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
        16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
        0.0,   10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0,  100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
        160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0};
    std::vector<float> gdata = {
        0.0,    0.0,  -1.0,   1.0,  -2.0,   2.0,  -3.0,   3.0,  -4.0,   4.0,  -5.0,   5.0,  -6.0,   6.0,  -7.0,   7.0,
        -8.0,   8.0,  -9.0,   9.0,  -10.0,  10.0, -11.0,  11.0, -12.0,  12.0, -13.0,  13.0, -14.0,  14.0, -15.0,  15.0,
        -16.0,  16.0, -17.0,  17.0, -18.0,  18.0, -19.0,  19.0, -20.0,  20.0, -21.0,  21.0, -22.0,  22.0, -23.0,  23.0,
        -24.0,  24.0, -25.0,  25.0, -26.0,  26.0, -27.0,  27.0, -28.0,  28.0, -29.0,  29.0, -30.0,  30.0, -31.0,  31.0,
        0.0,    0.0,  -10.0,  1.0,  -20.0,  2.0,  -30.0,  3.0,  -40.0,  4.0,  -50.0,  5.0,  -60.0,  6.0,  -70.0,  7.0,
        -80.0,  8.0,  -90.0,  9.0,  -100.0, 10.0, -110.0, 11.0, -120.0, 12.0, -130.0, 13.0, -140.0, 14.0, -150.0, 15.0,
        -160.0, 16.0, -170.0, 17.0, -180.0, 18.0, -190.0, 19.0, -200.0, 20.0, -210.0, 21.0, -220.0, 22.0, -230.0, 23.0,
        -240.0, 24.0, -250.0, 25.0, -260.0, 26.0, -270.0, 27.0, -280.0, 28.0, -290.0, 29.0, -300.0, 30.0, -310.0, 31.0};
    auto self = makeTensorData(DT_FP32, {2, 32}, sdata);
    auto out = makeTensorData(DT_FP32, {2, 64}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {2, 64}, gdata);
    calc::BitSort(out, self, -1, false, 0);
    ASSERT_ALLCLOSE(out->View({2, 64}, {0, 0}), golden);
}

TEST_F(TorchAdaptorTest, TopKTieIndices)
{
    // Equal values: smaller index first
    std::vector<float> sdata = {5.0f, 4.0f, 5.0f, 3.0f, 5.0f, 2.0f};
    std::vector<float> gvalues = {5.0f, 5.0f, 5.0f};
    std::vector<int32_t> gindices = {0, 2, 4};
    auto self = makeTensorData(DT_FP32, {1, 6}, sdata);
    auto outValue = makeTensorData(DT_FP32, {1, 3}, 0.0f);
    auto outIndex = makeTensorData(DT_INT32, {1, 3}, 0);
    auto goldenValue = makeTensorData(DT_FP32, {1, 3}, gvalues);
    auto goldenIndex = makeTensorData(DT_INT32, {1, 3}, gindices);
    calc::TopK(outValue, outIndex, self, 3, -1, true);
    ASSERT_ALLCLOSE(outValue, goldenValue);
    ASSERT_ALLCLOSE(outIndex, goldenIndex);
}

TEST_F(TorchAdaptorTest, TopkDescending)
{
    // 降序
    std::vector<float> sdata1 = {31.0, 31.0, 30.0, 30.0, 29.0, 29.0, 28.0, 28.0, 27.0, 27.0, 26.0, 26.0, 25.0,
                                 25.0, 24.0, 24.0, 23.0, 23.0, 22.0, 22.0, 21.0, 21.0, 20.0, 20.0, 19.0, 19.0,
                                 18.0, 18.0, 17.0, 17.0, 16.0, 16.0, 15.0, 15.0, 14.0, 14.0, 13.0, 13.0, 12.0,
                                 12.0, 11.0, 11.0, 10.0, 10.0, 9.0,  9.0,  8.0,  8.0,  7.0,  7.0,  6.0,  6.0,
                                 5.0,  5.0,  4.0,  4.0,  3.0,  3.0,  2.0,  2.0,  1.0,  1.0,  0.0,  0.0};
    std::vector<float> sdata2 = {
        310.0, 31.0, 300.0, 30.0, 290.0, 29.0, 280.0, 28.0, 270.0, 27.0, 260.0, 26.0, 250.0, 25.0, 240.0, 24.0,
        230.0, 23.0, 220.0, 22.0, 210.0, 21.0, 200.0, 20.0, 190.0, 19.0, 180.0, 18.0, 170.0, 17.0, 160.0, 16.0,
        150.0, 15.0, 140.0, 14.0, 130.0, 13.0, 120.0, 12.0, 110.0, 11.0, 100.0, 10.0, 90.0,  9.0,  80.0,  8.0,
        70.0,  7.0,  60.0,  6.0,  50.0,  5.0,  40.0,  4.0,  30.0,  3.0,  20.0,  2.0,  10.0,  1.0,  0.0,   0.0};
    const int64_t TOPK_VAL = 32;
    std::vector<float> gdata = sdata1;
    gdata.insert(gdata.end(), sdata2.begin(), sdata2.end());
    auto golden = makeTensorData(DT_FP32, {2, 64}, gdata);
    std::vector<float> sdata = sdata1;
    sdata.insert(sdata.end(), sdata2.begin(), sdata2.end());
    auto self = makeTensorData(DT_FP32, {2, 64}, sdata);
    auto out = makeTensorData(DT_FP32, {2, 64}, 0.0f);

    calc::MrgSort(out, self, -1, TOPK_VAL);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, TiledMrgSortOp)
{
    std::vector<float> sdata1 = {0.0, 0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0, -6.0, 6.0, -7.0, 7.0};
    std::vector<float> sdata2 = {-0.5,  8.0,  -10.0, 9.0,  -20.0, 10.0, -30.0, 11.0,
                                 -40.0, 12.0, -50.0, 13.0, -60.0, 14.0, -70.0, 15.0};
    std::vector<float> gdata = {0.0, 0.0, -0.5, 8.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0, -6.0, 6.0};
    auto golden = makeTensorData(DT_FP32, {1, 16}, gdata);
    auto src1 = makeTensorData(DT_FP32, {1, 16}, sdata1);
    auto src2 = makeTensorData(DT_FP32, {1, 16}, sdata2);
    auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
    int validBit = 2;
    int kvalue = 8;
    calc::TiledMrgSort(out, src1, src2, src2, src2, validBit, kvalue);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, MrgSortPadToK)
{
    std::vector<float> sdata = {5.0f, 0.0f};
    auto self = makeTensorData(DT_FP32, {1, 2}, sdata);
    auto out = makeTensorData(DT_FP32, {1, 64}, 0.0f);
    constexpr float negInf = -std::numeric_limits<float>::infinity();
    std::vector<float> gdata(64, negInf);
    gdata[0] = 5.0f;
    gdata[1] = 0.0f;
    auto golden = makeTensorData(DT_FP32, {1, 64}, gdata);
    calc::MrgSort(out, self, -1, 32);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, MrgSortEmptyInput)
{
    auto self = makeTensorData(DT_FP32, {1, 0}, std::vector<float>{});
    auto out = makeTensorData(DT_FP32, {1, 64}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 64}, -std::numeric_limits<float>::infinity());
    calc::MrgSort(out, self, -1, 32);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, TiledMrgSortPadToK)
{
    std::vector<float> sdata1 = {5.0f, 0.0f, 3.0f, 1.0f};
    std::vector<float> sdata2 = {9.0f, 2.0f, 1.0f, 3.0f};
    auto src1 = makeTensorData(DT_FP32, {1, 4}, sdata1);
    auto src2 = makeTensorData(DT_FP32, {1, 4}, sdata2);
    auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
    constexpr float negInf = -std::numeric_limits<float>::infinity();
    std::vector<float> gdata = {9.0f,   2.0f,   5.0f,   0.0f,   3.0f,   1.0f,   1.0f,   3.0f,
                                negInf, negInf, negInf, negInf, negInf, negInf, negInf, negInf};
    auto golden = makeTensorData(DT_FP32, {1, 16}, gdata);
    calc::TiledMrgSort(out, src1, src2, src2, src2, 2, 8);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, TiledMrgSortEmptyInput)
{
    auto src1 = makeTensorData(DT_FP32, {1, 0}, std::vector<float>{});
    auto src2 = makeTensorData(DT_FP32, {1, 0}, std::vector<float>{});
    auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 16}, -std::numeric_limits<float>::infinity());
    calc::TiledMrgSort(out, src1, src2, src2, src2, 2, 8);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ExtractDescending)
{
    // 降序
    std::vector<float> sdata = {
        31,  31, 30,  30, 29,  29, 28,  28, 27,  27, 26,  26, 25,  25, 24,  24, 23,  23, 22,  22, 21,  21,
        20,  20, 19,  19, 18,  18, 17,  17, 16,  16, 15,  15, 14,  14, 13,  13, 12,  12, 11,  11, 10,  10,
        9,   9,  8,   8,  7,   7,  6,   6,  5,   5,  4,   4,  3,   3,  2,   2,  1,   1,  0,   0,  310, 31,
        300, 30, 290, 29, 280, 28, 270, 27, 260, 26, 250, 25, 240, 24, 230, 23, 220, 22, 210, 21, 200, 20,
        190, 19, 180, 18, 170, 17, 160, 16, 150, 15, 140, 14, 130, 13, 120, 12, 110, 11, 100, 10, 90,  9,
        80,  8,  70,  7,  60,  6,  50,  5,  40,  4,  30,  3,  20,  2,  10,  1,  0,   0};
    std::vector<float> gdata0 = {
        31.0,  30.0,  29.0,  28.0,  27.0,  26.0,  25.0,  24.0,  23.0,  22.0,  21.0,  20.0,  19.0,  18.0,  17.0,  16.0,
        15.0,  14.0,  13.0,  12.0,  11.0,  10.0,  9.0,   8.0,   7.0,   6.0,   5.0,   4.0,   3.0,   2.0,   1.0,   0.0,
        310.0, 300.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 200.0, 190.0, 180.0, 170.0, 160.0,
        150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 90.0,  80.0,  70.0,  60.0,  50.0,  40.0,  30.0,  20.0,  10.0,  0.0};
    std::vector<float> gdata1 = {31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
                                 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0,  8.0,  7.0,  6.0,
                                 5.0,  4.0,  3.0,  2.0,  1.0,  0.0,  31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0,
                                 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0,
                                 11.0, 10.0, 9.0,  8.0,  7.0,  6.0,  5.0,  4.0,  3.0,  2.0,  1.0,  0.0};
    auto self = makeTensorData(DT_FP32, {2, 64}, sdata);
    auto out0 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto out1 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto golden0 = makeTensorData(DT_FP32, {2, 32}, gdata0);
    auto golden1 = makeTensorData(DT_FP32, {2, 32}, gdata1);

    calc::Extract(out0, self, 0, true);
    calc::Extract(out1, self, 1, true);
    ASSERT_ALLCLOSE(out0, golden0);
    ASSERT_ALLCLOSE(out1, golden1);
}

TEST_F(TorchAdaptorTest, ExtractAscending)
{
    // 升序
    std::vector<float> sdata = {
        0,    0,  -1,   1,  -2,   2,  -3,   3,  -4,   4,  -5,   5,  -6,   6,  -7,   7,  -8,   8,  -9,   9,  -10,  10,
        -11,  11, -12,  12, -13,  13, -14,  14, -15,  15, -16,  16, -17,  17, -18,  18, -19,  19, -20,  20, -21,  21,
        -22,  22, -23,  23, -24,  24, -25,  25, -26,  26, -27,  27, -28,  28, -29,  29, -30,  30, -31,  31, 0,    0,
        -10,  1,  -20,  2,  -30,  3,  -40,  4,  -50,  5,  -60,  6,  -70,  7,  -80,  8,  -90,  9,  -100, 10, -110, 11,
        -120, 12, -130, 13, -140, 14, -150, 15, -160, 16, -170, 17, -180, 18, -190, 19, -200, 20, -210, 21, -220, 22,
        -230, 23, -240, 24, -250, 25, -260, 26, -270, 27, -280, 28, -290, 29, -300, 30, -310, 31};
    std::vector<float> gdata0 = {
        0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
        16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
        0.0,   10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0,  100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
        160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0};
    std::vector<float> gdata1 = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
                                 7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                                 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0};
    auto self = makeTensorData(DT_FP32, {2, 64}, sdata);
    auto out0 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto out1 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto golden0 = makeTensorData(DT_FP32, {2, 32}, gdata0);
    auto golden1 = makeTensorData(DT_FP32, {2, 32}, gdata1);

    calc::Extract(out0, self, 0, false);
    calc::Extract(out1, self, 1, false);
    ASSERT_ALLCLOSE(out0, golden0);
    ASSERT_ALLCLOSE(out1, golden1);
}

TEST_F(TorchAdaptorTest, TwoTileMrgSort)
{
    std::vector<float> sdata = {15.0, 15.0, 14.0, 14.0, 13.0, 13.0, 12.0, 12.0, 11.0, 11.0, 10.0, 10.0, 9.0,
                                9.0,  8.0,  8.0,  7.0,  7.0,  6.0,  6.0,  5.0,  5.0,  4.0,  4.0,  3.0,  3.0,
                                2.0,  2.0,  1.0,  1.0,  0.0,  0.0,  31.0, 31.0, 30.0, 30.0, 29.0, 29.0, 28.0,
                                28.0, 27.0, 27.0, 26.0, 26.0, 25.0, 25.0, 24.0, 24.0, 23.0, 23.0, 22.0, 22.0,
                                21.0, 21.0, 20.0, 20.0, 19.0, 19.0, 18.0, 18.0, 17.0, 17.0, 16.0, 16.0};
    std::vector<float> gdata = {31.0, 31.0, 30.0, 30.0, 29.0, 29.0, 28.0, 28.0, 27.0, 27.0, 26.0, 26.0, 25.0,
                                25.0, 24.0, 24.0, 23.0, 23.0, 22.0, 22.0, 21.0, 21.0, 20.0, 20.0, 19.0, 19.0,
                                18.0, 18.0, 17.0, 17.0, 16.0, 16.0, 15.0, 15.0, 14.0, 14.0, 13.0, 13.0, 12.0,
                                12.0, 11.0, 11.0, 10.0, 10.0, 9.0,  9.0,  8.0,  8.0,  7.0,  7.0,  6.0,  6.0,
                                5.0,  5.0,  4.0,  4.0,  3.0,  3.0,  2.0,  2.0,  1.0,  1.0,  0.0,  0.0};
    auto self = makeTensorData(DT_FP32, {1, 64}, sdata);
    auto golden = makeTensorData(DT_FP32, {1, 64}, gdata);
    auto out = makeTensorData(DT_FP32, {1, 64}, 0.0f);

    calc::TwoTileMrgSort(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, SortUB)
{
    std::vector<float> sdata = {3.0, 7.0, 1.0, 5.0, 9.0, 2.0, 8.0, 4.0};
    std::vector<float> gdata0 = {9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    std::vector<int> gdata1 = {4, 6, 1, 3, 7, 0, 5, 2};
    auto self = makeTensorData(DT_FP32, {1, 8}, sdata);
    auto goldenValue = makeTensorData(DT_FP32, {1, 8}, gdata0);
    auto goldenIndex = makeTensorData(DT_INT32, {1, 8}, gdata1);
    auto outValue = makeTensorData(DT_FP32, {1, 8}, 0.0f);
    auto outIndex = makeTensorData(DT_INT32, {1, 8}, 0);

    calc::Sort(outValue, outIndex, self, 1, true);
    ASSERT_ALLCLOSE(outValue, goldenValue);
    ASSERT_ALLCLOSE(outIndex, goldenIndex);
}

TEST_F(TorchAdaptorTest, TopkSort)
{
    // Test TopkSort with 8-element input
    // Input: small array for easy verification
    std::vector<float> sdata = {3.0, 7.0, 1.0, 5.0, 9.0, 2.0, 8.0, 4.0};

    auto self = makeTensorData(DT_FP32, {1, 8}, sdata);
    auto outValue = makeTensorData(DT_FP32, {1, 64}, 0.0f); // Output padded to 32*2
    auto outTemp = makeTensorData(DT_FP32, {1, 64}, 0.0f);

    calc::TopkSort(outValue, outTemp, self, 0);

    // Expected: pack format [v0, i0, v1, i1, ...] with 32 elements (8 real + 24 padding)
    // Values sorted descending within the 32-element group, indices from 0-7
    // The exact golden output depends on the implementation
    // We verify by extracting top values and checking they're in descending order
    auto extractedValues = makeTensorData(DT_FP32, {1, 8}, 0.0f);
    calc::TopkExtract(extractedValues, outValue->View({1, 64}, {0, 0}), 8, false);

    // Top 8 values should include all original values (9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    std::vector<float> expectedTop = {9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    auto goldenTop = makeTensorData(DT_FP32, {1, 8}, expectedTop);
    ASSERT_ALLCLOSE(extractedValues, goldenTop);
}

TEST_F(TorchAdaptorTest, TopkSortLargeInput)
{
    // Test TopkSort with 32-element aligned input
    std::vector<float> sdata = {31.0, 15.0, 27.0, 8.0,  19.0, 3.0,  23.0, 11.0, 7.0, 28.0, 16.0,
                                2.0,  24.0, 9.0,  30.0, 14.0, 22.0, 5.0,  18.0, 1.0, 26.0, 10.0,
                                29.0, 13.0, 6.0,  20.0, 12.0, 25.0, 4.0,  21.0, 0.0, 17.0};

    auto self = makeTensorData(DT_FP32, {1, 32}, sdata);
    auto outValue = makeTensorData(DT_FP32, {1, 64}, 0.0f);
    auto outTemp = makeTensorData(DT_FP32, {1, 64}, 0.0f);

    calc::TopkSort(outValue, outTemp, self, 0);

    // Verify by extracting top-8 values
    auto extractedValues = makeTensorData(DT_FP32, {1, 8}, 0.0f);
    calc::TopkExtract(extractedValues, outValue->View({1, 64}, {0, 0}), 8, false);

    std::vector<float> expectedTop8 = {31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0};
    auto goldenTop8 = makeTensorData(DT_FP32, {1, 8}, expectedTop8);
    ASSERT_ALLCLOSE(extractedValues, goldenTop8);
}

TEST_F(TorchAdaptorTest, TopkMerge)
{
    // Test TopkMerge with pre-sorted pack array
    // Pack format: [v0, i0, v1, i1, v2, i2, ...]
    std::vector<float> packData = {
        // First 8 packs (sorted descending)
        30.0, 0.0, 28.0, 1.0, 26.0, 2.0, 24.0, 3.0, 22.0, 4.0, 20.0, 5.0, 18.0, 6.0, 16.0, 7.0,
        // Second 8 packs (sorted descending)
        31.0, 8.0, 29.0, 9.0, 27.0, 10.0, 25.0, 11.0, 23.0, 12.0, 21.0, 13.0, 19.0, 14.0, 17.0, 15.0};

    auto self = makeTensorData(DT_FP32, {1, 32}, packData);
    auto out = makeTensorData(DT_FP32, {1, 32}, 0.0f);

    // mergeSize = 8 means every 8 packs are already sorted
    calc::TopkMerge(out, self, 8);

    // Extract top 8 values to verify proper merging
    auto extractedValues = makeTensorData(DT_FP32, {1, 8}, 0.0f);
    calc::TopkExtract(extractedValues, out, 8, false);

    // Top 8 should be: 31, 30, 29, 28, 27, 26, 25, 24
    std::vector<float> expectedTop = {31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0};
    auto goldenTop = makeTensorData(DT_FP32, {1, 8}, expectedTop);
    ASSERT_ALLCLOSE(extractedValues, goldenTop);
}

TEST_F(TorchAdaptorTest, TopkExtractValues)
{
    // Test TopkExtract for value extraction (isIndex=false)
    std::vector<float> packData = {// Pack format: [v0, i0, v1, i1, ...]
                                   // Values sorted in descending order
                                   100.0, 5.0,  95.0, 12.0, 90.0, 3.0,  85.0, 18.0, 80.0, 7.0,  75.0,
                                   21.0,  70.0, 1.0,  65.0, 14.0, 60.0, 9.0,  55.0, 25.0, 50.0, 2.0,
                                   45.0,  16.0, 40.0, 11.0, 35.0, 28.0, 30.0, 4.0,  25.0, 19.0};

    auto self = makeTensorData(DT_FP32, {1, 32}, packData);
    auto out = makeTensorData(DT_FP32, {1, 8}, 0.0f);

    // Extract top 8 values
    calc::TopkExtract(out, self, 8, false);

    std::vector<float> expectedValues = {100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0};
    auto golden = makeTensorData(DT_FP32, {1, 8}, expectedValues);

    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, TopkExtractIndices)
{
    // Test TopkExtract for index extraction (isIndex=true)
    std::vector<float> packData = {// Pack format: [v0, i0, v1, i1, ...]
                                   100.0, 5.0,  95.0, 12.0, 90.0, 3.0,  85.0, 18.0, 80.0, 7.0,  75.0,
                                   21.0,  70.0, 1.0,  65.0, 14.0, 60.0, 9.0,  55.0, 25.0, 50.0, 2.0,
                                   45.0,  16.0, 40.0, 11.0, 35.0, 28.0, 30.0, 4.0,  25.0, 19.0};

    auto self = makeTensorData(DT_FP32, {1, 32}, packData);
    auto out = makeTensorData(DT_INT32, {1, 8}, 0);

    // Extract top 8 indices
    calc::TopkExtract(out, self, 8, true);

    std::vector<int> expectedIndices = {5, 12, 3, 18, 7, 21, 1, 14};
    auto golden = makeTensorData(DT_INT32, {1, 8}, expectedIndices);

    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, GatherINUB)
{
    // params shape: [num_buffer_tokens, hidden_dim] = [8, 4]
    std::vector<float> paramsData = {
        0.0f,  1.0f,  2.0f,  3.0f,  // row 0
        10.0f, 11.0f, 12.0f, 13.0f, // row 1
        20.0f, 21.0f, 22.0f, 23.0f, // row 2
        30.0f, 31.0f, 32.0f, 33.0f, // row 3
        40.0f, 41.0f, 42.0f, 43.0f, // row 4
        50.0f, 51.0f, 52.0f, 53.0f, // row 5
        60.0f, 61.0f, 62.0f, 63.0f, // row 6
        70.0f, 71.0f, 72.0f, 73.0f  // row 7
    };
    // logical indices [0, 2, 5, 3], blockSize=2:
    // pageTable=[2,0,1] => logical->physical mapping:
    // 0->4, 2->0, 5->3, 3->1
    std::vector<int64_t> indicesData = {0, 2, 5, 3};
    std::vector<int64_t> pageTableData = {2, 0, 1};
    std::vector<float> goldenData = {
        40.0f, 41.0f, 42.0f, 43.0f, // row 4
        0.0f,  1.0f,  2.0f,  3.0f,  // row 0
        30.0f, 31.0f, 32.0f, 33.0f, // row 3
        10.0f, 11.0f, 12.0f, 13.0f  // row 1
    };

    auto params = makeTensorData(DT_FP32, {8, 4}, paramsData);
    auto indices = makeTensorData(DT_INT64, {1, 4}, indicesData);
    auto pageTable = makeTensorData(DT_INT64, {1, 3}, pageTableData);
    auto out = makeTensorData(DT_FP32, {4, 4}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {4, 4}, goldenData);

    calc::GatherINUB(out, params, indices, pageTable, 2, 0);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Print)
{
    auto t0 = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    std::cout << t0->ToString() << std::endl;
    auto t1 = makeTensorData(DT_FP32, {4, 4, 4}, 4.0f);
    std::cout << t1->ToString() << std::endl;
    auto t2 = makeTensorData(DT_FP32, {4, 4, 1024, 512}, 4.0f);
    std::cout << t2->ToString() << std::endl;
    auto t3 = makeTensorData(DT_FP32, {4, 128}, 4.0f);
    std::cout << t3->ToString() << std::endl;
}

TEST_F(TorchAdaptorTest, NDNZ)
{
    for (auto m : {32, 33, 48}) {
        for (auto n : {32, 33, 48}) {
            int padm = alignup(m, 16);
            int padn = alignup(n, 8);
            std::vector<int> data(2 * m * n);
            std::iota(data.begin(), data.end(), 0);
            auto t0 = makeTensorData(DT_INT32, {2, m, n}, data);
            auto nzout = makeTensorData(DT_INT32, {2, padm, padn}, 0);
            auto ndzout = makeTensorData(DT_INT32, {2, m, n}, 0);
            auto golden = makeTensorData(DT_INT32, {2, m, n}, data);

            calc::FormatND2NZ(nzout, t0);
            calc::FormatNZ2ND(ndzout, nzout);
            ASSERT_ALLCLOSE(ndzout, golden);
        }
    }
}

TEST_F(TorchAdaptorTest, QuantizeSymmetricToInt8)
{
    // Symmetric quantization: FP32 -> INT8
    // Formula: output = clamp(round(input * scale), -128, 127)
    // Input: [2.0, 4.0, 6.0, 8.0], scale: 0.5
    // Expected: round([1.0, 2.0, 3.0, 4.0]) = [1, 2, 3, 4]
    std::vector<float> inputData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> scaleData = {0.5f};
    std::vector<int8_t> goldenData = {1, 2, 3, 4};

    auto input = makeTensorData(DT_FP32, {2, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto out = makeTensorData(DT_INT8, {2, 2}, static_cast<int8_t>(0));
    auto golden = makeTensorData(DT_INT8, {2, 2}, goldenData);

    calc::Quantize(out, input, scale, nullptr);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, QuantizeAsymmetricToUInt8)
{
    // Asymmetric quantization: FP32 -> UINT8
    // Formula: output = clamp(round(input * scale + zero_points), 0, 255)
    // Input: [2.0, 4.0, 6.0, 8.0], scale: 0.5, zero_points: 128
    // Expected: round([1.0, 2.0, 3.0, 4.0]) + 128 = [129, 130, 131, 132]
    std::vector<float> inputData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> scaleData = {0.5f};
    std::vector<int32_t> zeroPointsData = {128};
    std::vector<uint8_t> goldenData = {129, 130, 131, 132};

    auto input = makeTensorData(DT_FP32, {2, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto zeroPoints = makeTensorData(DT_INT32, {1}, zeroPointsData);
    auto out = makeTensorData(DT_UINT8, {2, 2}, static_cast<uint8_t>(0));
    auto golden = makeTensorData(DT_UINT8, {2, 2}, goldenData);

    calc::Quantize(out, input, scale, zeroPoints);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, QuantizeSymmetricClamp)
{
    // Test clamping for symmetric quantization (INT8 range: -128 to 127)
    // Input: [300.0, -300.0], scale: 1.0
    // Expected: clamp([300, -300], -128, 127) = [127, -128]
    std::vector<float> inputData = {300.0f, -300.0f};
    std::vector<float> scaleData = {1.0f};
    std::vector<int8_t> goldenData = {127, -128};

    auto input = makeTensorData(DT_FP32, {1, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto out = makeTensorData(DT_INT8, {1, 2}, static_cast<int8_t>(0));
    auto golden = makeTensorData(DT_INT8, {1, 2}, goldenData);

    calc::Quantize(out, input, scale, nullptr);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, QuantizeAsymmetricClamp)
{
    // Test clamping for asymmetric quantization (UINT8 range: 0 to 255)
    // Input: [100.0, -100.0], scale: 2.0, zero_points: 128
    // Expected: clamp([200, -200] + 128, 0, 255) = clamp([328, -72], 0, 255) = [255, 0]
    std::vector<float> inputData = {100.0f, -100.0f};
    std::vector<float> scaleData = {2.0f};
    std::vector<int32_t> zeroPointsData = {128};
    std::vector<uint8_t> goldenData = {255, 0};

    auto input = makeTensorData(DT_FP32, {1, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto zeroPoints = makeTensorData(DT_INT32, {1}, zeroPointsData);
    auto out = makeTensorData(DT_UINT8, {1, 2}, static_cast<uint8_t>(0));
    auto golden = makeTensorData(DT_UINT8, {1, 2}, goldenData);

    calc::Quantize(out, input, scale, zeroPoints);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, DequantizeInt8ToFP32)
{
    // Dequantize: INT8 -> FP32
    // Formula: output = input * scale
    // Input: [1, 2, 3, 4], scale: 2.0
    // Expected: [2.0, 4.0, 6.0, 8.0]
    std::vector<int8_t> inputData = {1, 2, 3, 4};
    std::vector<float> scaleData = {2.0f};
    std::vector<float> goldenData = {2.0f, 4.0f, 6.0f, 8.0f};

    auto input = makeTensorData(DT_INT8, {2, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto zeroPoints = makeTensorData(DT_INT32, {1}, 0);
    auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {2, 2}, goldenData);

    calc::Dequantize(out, input, scale, zeroPoints);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, DequantizeInt16ToFP32)
{
    // Dequantize: INT16 -> FP32
    // Input: [10, 20, 30, 40], scale: 0.5
    // Expected: [5.0, 10.0, 15.0, 20.0]
    std::vector<int16_t> inputData = {10, 20, 30, 40};
    std::vector<float> scaleData = {0.5f};
    std::vector<float> goldenData = {5.0f, 10.0f, 15.0f, 20.0f};

    auto input = makeTensorData(DT_INT16, {2, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto zeroPoints = makeTensorData(DT_INT32, {1}, 0);
    auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {2, 2}, goldenData);

    calc::Dequantize(out, input, scale, zeroPoints);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, DequantizeAsymmetric)
{
    // Dequantize with zero_points: INT8 -> FP32
    // Formula: output = input * scale - zero_points
    // Input: [30, 31], scale: 0.5, zero_points: 128
    // Expected: [30 * 0.5 - 128, 31 * 0.5 - 128] = [15 - 128, 15.5 - 128] = [-113.0, -112.5]
    std::vector<int8_t> inputData = {30, 31};
    std::vector<float> scaleData = {0.5f};
    std::vector<int32_t> zeroPointsData = {128};
    std::vector<float> goldenData = {-113.0f, -112.5f};

    auto input = makeTensorData(DT_INT8, {1, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto zeroPoints = makeTensorData(DT_INT32, {1}, zeroPointsData);
    auto out = makeTensorData(DT_FP32, {1, 2}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 2}, goldenData);

    calc::Dequantize(out, input, scale, zeroPoints);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, QuantizeDequantizeRoundTrip)
{
    // Quantize-Dequantize round trip test
    // Input: [2.0, 4.0, 6.0, 8.0], scale: 1.0
    // Quantize (symmetric): [2, 4, 6, 8]
    // Dequantize: [2.0, 4.0, 6.0, 8.0]
    std::vector<float> inputData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> scaleData = {1.0f};
    std::vector<float> goldenData = {2.0f, 4.0f, 6.0f, 8.0f};

    auto input = makeTensorData(DT_FP32, {2, 2}, inputData);
    auto scale = makeTensorData(DT_FP32, {1}, scaleData);
    auto quantized = makeTensorData(DT_INT8, {2, 2}, static_cast<int8_t>(0));
    auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {2, 2}, goldenData);

    auto zeroPoints = makeTensorData(DT_INT32, {1}, 0);

    calc::Quantize(quantized, input, scale, nullptr);
    calc::Dequantize(out, quantized, scale, zeroPoints);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, QuantMXFp32)
{
    constexpr int64_t cols = 64;
    std::vector<float> inputData(cols, 1.0f);
    std::vector<uint8_t> quantData(cols, 0x78);
    std::vector<uint8_t> expData = {119, 119};
    std::vector<float> maxData = {1.0f, 1.0f};
    std::vector<float> scalingData = {256.0f, 256.0f, 256.0f, 256.0f};

    auto input = makeTensorData(DT_FP32, {1, cols}, inputData);
    auto out = makeTensorData(DT_FP8E4M3, {1, cols}, static_cast<uint8_t>(0));
    auto exp = makeTensorData(DT_FP8E8M0, {1, 2}, static_cast<uint8_t>(0));
    auto max = makeTensorData(DT_FP32, {1, 2}, 0.0f);
    auto scaling = makeTensorData(DT_FP32, {1, 4}, 0.0f);
    auto quantGolden = makeTensorData(DT_FP8E4M3, {1, cols}, quantData);
    auto expGolden = makeTensorData(DT_FP8E8M0, {1, 2}, expData);
    auto maxGolden = makeTensorData(DT_FP32, {1, 2}, maxData);
    auto scalingGolden = makeTensorData(DT_FP32, {1, 4}, scalingData);

    calc::QuantMX(out, exp, max, scaling, input, false);
    ASSERT_ALLCLOSE(out, quantGolden);
    ASSERT_ALLCLOSE(exp, expGolden);
    ASSERT_ALLCLOSE(max, maxGolden);
    ASSERT_ALLCLOSE(scaling, scalingGolden);
}

TEST_F(TorchAdaptorTest, Erfc)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 0.1573f);
    calc::Erfc(out, self);
    ASSERT_ALLCLOSE(out, golden);
}
} // namespace npu::tile_fwk
