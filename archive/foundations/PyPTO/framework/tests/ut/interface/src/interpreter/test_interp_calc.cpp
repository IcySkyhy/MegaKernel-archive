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
 * \file test_interp_calc.cpp
 * \brief Interpreter calc unit tests (unary/binary/where).
 */

#include "test_interp_calc_utils.h"

namespace npu::tile_fwk {

TEST_F(TorchAdaptorTest, LogicalNot)
{
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_BOOL, {16, 16}, true);
        auto golden = makeTensorData(DT_BOOL, {16, 16}, false);
        calc::LogicalNot(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(4.0f));
        auto out = makeTensorData(DT_BOOL, {16, 16}, true);
        auto golden = makeTensorData(DT_BOOL, {16, 16}, false);
        calc::LogicalNot(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, LogicalAnd)
{
    auto self = makeTensorData(DT_BOOL, {16, 16}, true);
    auto other = makeTensorData(DT_BOOL, {16, 16}, true);
    auto out = makeTensorData(DT_BOOL, {16, 16}, false);
    auto golden = makeTensorData(DT_BOOL, {16, 16}, true);
    calc::LogicalAnd(out, self, other);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Range)
{
    std::vector<float> gdata = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    auto out = makeTensorData(DT_FP32, {7}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {7}, gdata);
    calc::Range(out, Element(DT_FP32, 1.1f), Element(DT_INT32, 8), Element(DT_FP32, 1.1f));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Uniform)
{
    uint64_t key = 12345678901234;
    uint64_t counter0 = 0;
    uint64_t counter1 = 0;
    auto out = makeTensorData(DT_FP32, {16}, 0.0f);
    calc::Uniform(out, Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1),
                  Element(DT_UINT16, static_cast<uint16_t>(10)), DT_FP32);
}

TEST_F(TorchAdaptorTest, Uniform_FP16)
{
    uint64_t key = 12345678901234;
    uint64_t counter0 = 0;
    uint64_t counter1 = 0;
    auto out = makeTensorData(DT_FP16, {16}, float16(0.0));
    calc::Uniform(out, Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1),
                  Element(DT_UINT16, static_cast<uint16_t>(10)), DT_FP16);
}

TEST_F(TorchAdaptorTest, Uniform_BF16)
{
    uint64_t key = 12345678901234;
    uint64_t counter0 = 0;
    uint64_t counter1 = 0;
    auto out = makeTensorData(DT_BF16, {16}, static_cast<bfloat16>(0.0f));
    calc::Uniform(out, Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1),
                  Element(DT_UINT16, static_cast<uint16_t>(10)), DT_BF16);
}

TEST_F(TorchAdaptorTest, Uniform_Rounds7)
{
    uint64_t key = 12345678901234;
    uint64_t counter0 = 0;
    uint64_t counter1 = 0;
    auto out = makeTensorData(DT_FP32, {16}, 0.0f);
    calc::Uniform(out, Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1),
                  Element(DT_UINT16, static_cast<uint16_t>(7)), DT_FP32);
}

TEST_F(TorchAdaptorTest, Uniform_FP16_Rounds7)
{
    uint64_t key = 12345678901234;
    uint64_t counter0 = 0;
    uint64_t counter1 = 0;
    auto out = makeTensorData(DT_FP16, {16}, float16(0.0));
    calc::Uniform(out, Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1),
                  Element(DT_UINT16, static_cast<uint16_t>(7)), DT_FP16);
}

TEST_F(TorchAdaptorTest, Uniform_BF16_Rounds7)
{
    uint64_t key = 12345678901234;
    uint64_t counter0 = 0;
    uint64_t counter1 = 0;
    auto out = makeTensorData(DT_BF16, {16}, static_cast<bfloat16>(0.0f));
    calc::Uniform(out, Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1),
                  Element(DT_UINT16, static_cast<uint16_t>(7)), DT_BF16);
}

TEST_F(TorchAdaptorTest, Uniform_LargeShape)
{
    uint64_t key = 9876543210;
    uint64_t counter0 = 100;
    uint64_t counter1 = 200;
    auto out = makeTensorData(DT_FP32, {64}, 0.0f);
    calc::Uniform(out, Element(DT_UINT64, key), Element(DT_UINT64, counter0), Element(DT_UINT64, counter1),
                  Element(DT_UINT16, static_cast<uint16_t>(10)), DT_FP32);
}

TEST_F(TorchAdaptorTest, Exp2)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, std::exp2(2.0f));
    calc::Exp2(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Erf)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, std::erf(1.0f));
    calc::Erf(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Sin)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 0.0f); // sin(0) = 0
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, std::sin(0.0f));
    calc::Sin(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Cos)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 0.0f); // cos(0) = 1
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, std::cos(0.0f)); // cos(0) = 1
    calc::Cos(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Round)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.1f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    calc::Round(out, self, 0);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Compare)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto out = makeTensorData(DT_BOOL, {16, 16}, false);
    auto golden_true = makeTensorData(DT_BOOL, {16, 16}, true);
    auto golden_false = makeTensorData(DT_BOOL, {16, 16}, false);

    struct {
        CmpOperationType type;
        CmpModeType mode;
        bool expect;
    } cases[] = {
        {CmpOperationType::EQ, CmpModeType::BOOL, true},  {CmpOperationType::NE, CmpModeType::BOOL, false},
        {CmpOperationType::LT, CmpModeType::BOOL, false}, {CmpOperationType::LE, CmpModeType::BOOL, true},
        {CmpOperationType::GT, CmpModeType::BOOL, false}, {CmpOperationType::GE, CmpModeType::BOOL, true},
    };
    for (const auto& test : cases) {
        calc::Compare(out, self, other, test.type, test.mode);
        if (test.expect) {
            ASSERT_ALLCLOSE(out, golden_true);
        } else {
            ASSERT_ALLCLOSE(out, golden_false);
        }
    }
}

TEST_F(TorchAdaptorTest, CompareBit)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto out = makeTensorData(DT_UINT8, {16, 2}, false);
    auto golden_1 = makeTensorData(DT_UINT8, {16, 2}, (uint8_t)0xFF);
    auto golden_0 = makeTensorData(DT_UINT8, {16, 2}, (uint8_t)0);

    struct {
        CmpOperationType type;
        CmpModeType mode;
        bool expect;
    } cases[] = {
        {CmpOperationType::EQ, CmpModeType::BIT, true},  {CmpOperationType::NE, CmpModeType::BIT, false},
        {CmpOperationType::LT, CmpModeType::BIT, false}, {CmpOperationType::LE, CmpModeType::BIT, true},
        {CmpOperationType::GT, CmpModeType::BIT, false}, {CmpOperationType::GE, CmpModeType::BIT, true},
    };
    for (const auto& test : cases) {
        calc::Compare(out, self, other, test.type, test.mode);
        if (test.expect) {
            ASSERT_ALLCLOSE(out, golden_1);
        } else {
            ASSERT_ALLCLOSE(out, golden_0);
        }
    }
}

TEST_F(TorchAdaptorTest, Cmps)
{
    auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto elem = Element(DT_FP32, 4.0f);
    auto out = makeTensorData(DT_BOOL, {16, 16}, false);
    auto golden_true = makeTensorData(DT_BOOL, {16, 16}, true);
    auto golden_false = makeTensorData(DT_BOOL, {16, 16}, false);
    struct {
        CmpOperationType type;
        CmpModeType mode;
        bool expect;
    } cases[] = {
        {CmpOperationType::EQ, CmpModeType::BOOL, true},  {CmpOperationType::NE, CmpModeType::BOOL, false},
        {CmpOperationType::LT, CmpModeType::BOOL, false}, {CmpOperationType::LE, CmpModeType::BOOL, true},
        {CmpOperationType::GT, CmpModeType::BOOL, false}, {CmpOperationType::GE, CmpModeType::BOOL, true},
    };
    for (const auto& test : cases) {
        calc::Cmps(out, self, elem, test.type, test.mode);
        if (test.expect) {
            ASSERT_ALLCLOSE(out, golden_true);
        } else {
            ASSERT_ALLCLOSE(out, golden_false);
        }
    }
}
TEST_F(TorchAdaptorTest, Ceil)
{
    // ceil
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.1f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
    calc::Ceil(out, self);
    ASSERT_ALLCLOSE(out, golden);
}
TEST_F(TorchAdaptorTest, Floor)
{
    // floor
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.1f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    calc::Floor(out, self);
    ASSERT_ALLCLOSE(out, golden);
}
TEST_F(TorchAdaptorTest, Trunc)
{
    // trunc
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.1f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    calc::Trunc(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Log1p)
{
    // ceil
    auto self = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    calc::Log1p(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, UnaryOps)
{
    {
        // rsqrt
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 0.5f);
        calc::Rsqrt(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sqrt
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Sqrt(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // atan
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, (float)(3.14159265358979323 / 4));
        calc::Atan(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // pack
        auto self = makeTensorData(DT_INT32, {16, 16}, 0x01010101);
        auto out = makeTensorData(DT_UINT8, {1024}, static_cast<uint8_t>(0x0));
        auto golden = makeTensorData(DT_UINT8, {1024}, static_cast<uint8_t>(0x01));
        calc::Pack(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // unpack
        auto self = makeTensorData(DT_UINT8, {16}, static_cast<uint8_t>(0x01));
        auto out = makeTensorData(DT_INT32, {4}, 0);
        auto golden = makeTensorData(DT_INT32, {4}, 0x01010101);
        calc::UnPack(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // reciprocal
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 0.25f);
        calc::Reciprocal(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // relu
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::Relu(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sign
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        calc::Sign(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // signbit positive
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_BOOL, {16, 16}, false);
        auto golden = makeTensorData(DT_BOOL, {16, 16}, false);
        calc::Signbit(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // signbit negative
        auto self = makeTensorData(DT_FP32, {16, 16}, -4.0f);
        auto out = makeTensorData(DT_BOOL, {16, 16}, false);
        auto golden = makeTensorData(DT_BOOL, {16, 16}, true);
        calc::Signbit(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // tanh
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::tanh(1.0f));
        calc::Tanh(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // tan
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::tan(1.0f));
        calc::Tan(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // abs
        auto self = makeTensorData(DT_FP32, {16, 16}, -4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::Abs(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // ln
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::log(2.0f));
        calc::Ln(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // exp
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::exp(2.0f));
        calc::Exp(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expm1
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::exp(2.0f) - 1);
        calc::Expm1(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sinh
        auto self = makeTensorData(DT_FP32, {16, 16}, 0.5f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::sinh(0.5f));
        calc::Sinh(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cosh
        auto self = makeTensorData(DT_FP32, {16, 16}, 0.5f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::cosh(0.5f));
        calc::Cosh(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // asinh
        auto self = makeTensorData(DT_FP32, {16, 16}, 0.5f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::asinh(0.5f));
        calc::ASinh(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // acosh
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.5f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::acosh(1.5f));
        calc::ACosh(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // atanh
        auto self = makeTensorData(DT_FP32, {16, 16}, 0.5f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::atanh(0.5f));
        calc::Atanh(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // neg
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, -2.0f);
        calc::Neg(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::Cast(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast
        auto self = makeTensorData(DT_FP32, {16, 16}, 1e8f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(static_cast<int>(1e8f)));
        calc::Cast(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand scalar
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::ExpandS(out, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand broadcast
        auto self = makeTensorData(DT_FP32, {16, 1}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Expand(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand broadcast and cast
        auto self = makeTensorData(DT_FP32, {16, 1}, 2.0f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::Expand(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast (torch modes) - integer targets with ties/non-ties
        // values: +2.5, -2.5, +2.4, -2.4, +2.6, -2.6
        std::vector<float> vals = {2.5f, -2.5f, 2.4f, -2.4f, 2.6f, -2.6f};
        auto self = makeTensorData(DT_FP32, {6, 1}, vals);

        // default (integer dst): torch.round behavior
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -2, 2, -2, 3, -3};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self);
            ASSERT_ALLCLOSE(out, golden);
        }
        // explicit CAST_ROUND
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -2, 2, -2, 3, -3};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_ROUND);
            ASSERT_ALLCLOSE(out, golden);
        }
        // CAST_FLOOR
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -3, 2, -3, 2, -3};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_FLOOR);
            ASSERT_ALLCLOSE(out, golden);
        }
        // CAST_CEIL
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {3, -2, 3, -2, 3, -2};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_CEIL);
            ASSERT_ALLCLOSE(out, golden);
        }
        // CAST_TRUNC
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -2, 2, -2, 2, -2};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_TRUNC);
            ASSERT_ALLCLOSE(out, golden);
        }
        // float targets: pass-through
        {
            auto out = makeTensorData(DT_FP32, {6, 1}, 0.0f);
            auto golden = makeTensorData(DT_FP32, {6, 1}, vals);
            calc::Cast(out, self);
            ASSERT_ALLCLOSE(out, golden);
        }
        // brcb 2D: [4,1] -> [4,3]
        {
            std::vector<float> sdata = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> gdata = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f};
            auto self_brcb = makeTensorData(DT_FP32, {4, 1}, sdata);
            auto out = makeTensorData(DT_FP32, {4, 3}, 0.0f);
            auto golden = makeTensorData(DT_FP32, {4, 3}, gdata);
            calc::Brcb(out, self_brcb);
            ASSERT_ALLCLOSE(out, golden);
        }
        // brcb 3D: [2,2,1] -> [2,2,3]
        {
            std::vector<float> sdata = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> gdata = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f};
            auto self_brcb = makeTensorData(DT_FP32, {2, 2, 1}, sdata);
            auto out = makeTensorData(DT_FP32, {2, 2, 3}, 0.0f);
            auto golden = makeTensorData(DT_FP32, {2, 2, 3}, gdata);
            calc::Brcb(out, self_brcb);
            ASSERT_ALLCLOSE(out, golden);
        }
    }
    {
        // bitwisenot
        auto input = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(-5));
        calc::BitwiseNot(out, input);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BinaryOps)
{
    {
        // add
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sub
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::Sub(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::Sub(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // mul
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::Mul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::Mul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // div
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::Div(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::Div(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        std::vector<int32_t> sdata = {5, -5, 7, -7};
        std::vector<int32_t> odata = {2, 2, -3, -3};
        auto self = makeTensorData(DT_INT32, {2, 2}, sdata);
        auto other = makeTensorData(DT_INT32, {2, 2}, odata);
        auto out = makeTensorData(DT_INT32, {2, 2}, 0);
        std::vector<int32_t> gdata = {2, -3, -3, 2};
        auto golden = makeTensorData(DT_INT32, {2, 2}, gdata);
        calc::FloorDiv(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto out = makeTensorData(DT_BOOL, {16, 16}, true);
        auto golden = makeTensorData(DT_BOOL, {16, 16}, true);
        calc::IsFinite(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // atan2
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, (float)(3.14159265358979323 / 4));
        calc::Atan2(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // pow
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::Pow(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // hypot
        auto self = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Hypot(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // prelu
        auto self = makeTensorData(DT_FP32, {16, 16}, -2.0f);
        auto weight = makeTensorData(DT_FP32, {16}, 0.25f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, -0.5f);
        calc::PReLU(out, self, weight);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // fmod
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        calc::Fmod(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // fmod broadcast
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 1}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        calc::Fmod(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // add broadcast
        auto self = makeTensorData(DT_FP32, {1, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 1}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // elementwise max
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> odata = {2.0, 2.0, 3.0, 5.0};
        std::vector<float> gdata = {2.0, 2.0, 5.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto other = makeTensorData(DT_FP32, {2, 2}, odata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::Max(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 6.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 6.0f);
        calc::Max(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // elementwise min
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> odata = {2.0, 2.0, 3.0, 5.0};
        std::vector<float> gdata = {1.0, 2.0, 3.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto other = makeTensorData(DT_FP32, {2, 2}, odata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::Min(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Min(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scalar min
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
        std::vector<float> gdata = {1.0, 2.0, 2.0, 2.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::MinS(out, self, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scalar max
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
        std::vector<float> gdata = {2.0, 2.0, 3.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::MaxS(out, self, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter update 2dim
        std::vector<float> sdata = {
            1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
        };
        std::vector<float> gdata = {
            0.0f, 0.0f, 0.0f, 16.0f, 17.0f, 18.0f, 0.0f,  0.0f,  0.0f,  1.0f,  2.0f,  3.0f,  7.0f, 8.0f, 9.0f,
            4.0f, 5.0f, 6.0f, 0.0f,  0.0f,  0.0f,  13.0f, 14.0f, 15.0f, 10.0f, 11.0f, 12.0f, 0.0f, 0.0f, 0.0f,
        };
        std::vector<int64_t> idata = {3, 5, 4, 8, 7, 1};
        auto self = makeTensorData(DT_FP32, {6, 3}, sdata);
        auto dst = makeTensorData(DT_FP32, {10, 3}, 0.0f);
        auto out = makeTensorData(DT_FP32, {10, 3}, 0.0f);
        auto index = makeTensorData(DT_INT64, {2, 3}, idata);
        auto golden = makeTensorData(DT_FP32, {10, 3}, gdata);
        calc::ScatterUpdate(out, self, index, dst);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter update 4dim
        std::vector<float> sdata = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                                    3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f};
        std::vector<float> gdata = {0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f};
        std::vector<int64_t> idata = {3, 5, 4, 1};
        auto self = makeTensorData(DT_FP32, {2, 2, 1, 4}, sdata);
        auto dst = makeTensorData(DT_FP32, {3, 2, 1, 4}, 0.0f);
        auto out = makeTensorData(DT_FP32, {3, 2, 1, 4}, 0.0f);
        auto index = makeTensorData(DT_INT64, {2, 2}, idata);
        auto golden = makeTensorData(DT_FP32, {3, 2, 1, 4}, gdata);
        calc::ScatterUpdate(out, self, index, dst, -2, "PA_BSND", 2); // blocksize设置为2
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter replace
        std::vector<float> selfData = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        };
        std::vector<int64_t> indicesData = {
            1,
            0,
            1,
            1,
        };
        std::vector<float> gdata = {
            1.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f,
        };
        auto src = Element(DT_FP32, 2.0f);
        auto self = makeTensorData(DT_FP32, {2, 5}, selfData);
        auto indices = makeTensorData(DT_INT64, {1, 4}, indicesData);
        auto out = makeTensorData(DT_FP32, {2, 5}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 5}, gdata);
        calc::ScatterElement(out, self, indices, src, 0, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter add
        std::vector<float> selfData = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        };
        std::vector<int64_t> indicesData = {
            1,
            0,
            1,
            1,
        };
        std::vector<float> gdata = {
            1.0f, 3.0f, 1.0f, 1.0f, 1.0f, 3.0f, 1.0f, 3.0f, 3.0f, 1.0f,
        };
        auto src = Element(DT_FP32, 2.0f);
        auto self = makeTensorData(DT_FP32, {2, 5}, selfData);
        auto indices = makeTensorData(DT_INT64, {1, 4}, indicesData);
        auto out = makeTensorData(DT_FP32, {2, 5}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 5}, gdata);
        calc::ScatterElement(out, self, indices, src, 0, 1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter tensor replace
        std::vector<float> selfData = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        };
        std::vector<int64_t> indicesData = {
            1,
            0,
            1,
            1,
        };
        std::vector<float> srcData = {
            10,
            11,
            12,
            13,
        };
        std::vector<float> gdata = {
            1.0f, 11.0f, 1.0f, 1.0f, 1.0f, 10.0f, 1.0f, 12.0f, 13.0f, 1.0f,
        };
        auto self = makeTensorData(DT_FP32, {2, 5}, selfData);
        auto indices = makeTensorData(DT_INT64, {1, 4}, indicesData);
        auto src = makeTensorData(DT_FP32, {1, 4}, srcData);
        auto out = makeTensorData(DT_FP32, {2, 5}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 5}, gdata);
        calc::Scatter(out, self, indices, src, 0, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiseand
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        calc::BitwiseAnd(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiseor
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(5));
        calc::BitwiseOr(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwisexor
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(5));
        calc::BitwiseXor(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // copysign
        std::vector<float> s0data = {1.0f, -1.0f, 1.0f, -1.0f};
        std::vector<float> s1data = {1.0f, 2.0f, -3.0f, 4.0f};
        std::vector<float> gdata = {1.0f, 1.0f, -1.0f, 1.0f};
        auto self = makeTensorData(DT_FP32, {1, 4}, s0data);
        auto other = makeTensorData(DT_FP32, {1, 4}, s1data);
        auto out = makeTensorData(DT_FP32, {1, 4}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {1, 4}, gdata);
        calc::CopySign(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BinaryOpsS)
{
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::AddS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::SubS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::MulS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::DivS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, -3.0f);
        calc::SubS(out, self, elem, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 0.4f);
        calc::DivS(out, self, elem, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        std::vector<int32_t> sdata = {5, -5, 7, -7};
        auto self = makeTensorData(DT_INT32, {2, 2}, sdata);
        auto elem = Element(DT_INT32, 2);
        auto out = makeTensorData(DT_INT32, {2, 2}, 0);
        std::vector<int32_t> gdata = {2, -3, 3, -4};
        auto golden = makeTensorData(DT_INT32, {2, 2}, gdata);
        calc::FloorDivS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 0.01f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::LReLU(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        calc::FmodS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(5));
        auto elem = Element(DT_INT16, 2);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        calc::BitwiseAndS(out, self, elem, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(5));
        auto elem = Element(DT_INT16, 2);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(7));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(7));
        calc::BitwiseOrS(out, self, elem, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(5));
        auto elem = Element(DT_INT16, 2);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(7));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(7));
        calc::BitwiseXorS(out, self, elem, true);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BitwiseShift)
{
    {
        // bitwiserightshift
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::BitwiseRightShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiseleftshift
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(8));
        calc::BitwiseLeftShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiserightshifts
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = Element(DT_INT16, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::BitwiseRightShiftS(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiseleftshifts
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = Element(DT_INT16, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(8));
        calc::BitwiseLeftShiftS(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sbitwiserightshift
        auto self = Element(DT_INT16, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::SBitwiseRightShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sbitwiseleftshift
        auto self = Element(DT_INT16, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(8));
        calc::SBitwiseLeftShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, Where)
{
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = makeTensorData(DT_FP32, {16, 16}, 6.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereTT(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = makeTensorData(DT_FP32, {16, 16}, 6.0f);
        auto other = Element(DT_FP32, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereTS(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = Element(DT_FP32, 6.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereST(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = Element(DT_FP32, 6.0f);
        auto other = Element(DT_FP32, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereSS(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(0.0f));
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(6.0f));
        auto other = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(4.0f));
        auto golden = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(4.0f));
        calc::WhereTT(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BinaryPairOps)
{
    int n = 16, p = 5;
    {
        auto self = makeTensorData(DT_FP32, {n, n}, 4.0f);
        auto other = makeTensorData(DT_FP32, {n, p}, 3.0f);
        auto out = makeTensorData(DT_FP32, {n, n}, 0.0f);
        auto golden = makePartialGolden(n, p, 7.0, 4.0);
        calc::PairSum(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {n, p}, 4.0f);
        auto other = makeTensorData(DT_FP32, {n, n}, 3.0f);
        auto out = makeTensorData(DT_FP32, {n, n}, 0.0f);
        auto golden = makePartialGolden(n, p, 7.0, 3.0);
        calc::PairSum(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {n, n}, 4.0f);
        auto other = makeTensorData(DT_FP32, {n, p}, 3.0f);
        auto out = makeTensorData(DT_FP32, {n, n}, 0.0f);
        auto golden = makePartialGolden(n, p, 3.0, 4.0);
        calc::PairMin(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {n, n}, 4.0f);
        auto other = makeTensorData(DT_FP32, {n, p}, 3.0f);
        auto out = makeTensorData(DT_FP32, {n, n}, 0.0f);
        auto golden = makePartialGolden(n, p, 12.0, 4.0);
        calc::PairProd(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

} // namespace npu::tile_fwk
