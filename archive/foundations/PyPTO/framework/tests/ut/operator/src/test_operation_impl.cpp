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
 * \file test_operation_impl.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
using namespace npu::tile_fwk;

class OperationImplTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST_F(OperationImplTest, test_CumSum_dim2_1)
{
    int axis = 1;
    TileShape::Current().SetVecTile(9, 9);
    Tensor input(DT_FP32, {13, 8}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(OperationImplTest, test_CumSum_dim2_0)
{
    int axis = 0;
    TileShape::Current().SetVecTile(4, 3);
    Tensor input(DT_INT16, {11, 7}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(OperationImplTest, test_CumSum_dim1)
{
    int axis = 0;
    TileShape::Current().SetVecTile(5);
    Tensor input(DT_INT32, {13}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(OperationImplTest, test_CumSum_dim3)
{
    int axis = 0;
    TileShape::Current().SetVecTile(4, 5, 3);
    Tensor input(DT_FP32, {8, 8, 8}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(OperationImplTest, test_CumSum_dim4)
{
    int axis = 0;
    TileShape::Current().SetVecTile(4, 5, 5, 3);
    Tensor input(DT_FP32, {7, 7, 7, 7}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(OperationImplTest, test_IsFinite_fp32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor input(DT_FP32, {11, 32}, "input");
    Tensor result;
    FUNCTION("TestIsFinite") { result = IsFinite(input); }
}

TEST_F(OperationImplTest, test_IsFinite_int32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor input(DT_INT32, {11, 32}, "input");
    Tensor result;
    FUNCTION("TestIsFinite") { result = IsFinite(input); }
}

TEST_F(OperationImplTest, test_CeilDivs_int32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor self(DT_INT32, {11, 32}, "self");
    Element other(DT_INT32, 2);
    Tensor result;
    FUNCTION("TestCeilDiv") { result = CeilDiv(self, other); }
}

TEST_F(OperationImplTest, test_CeilDiv_int32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor self(DT_INT32, {11, 32}, "self");
    Tensor other(DT_INT32, {11, 32}, "other");
    Tensor result;
    FUNCTION("TestCeilDiv") { result = CeilDiv(self, other); }
}

TEST_F(OperationImplTest, test_Compare_BOOL)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    Tensor operand2(DT_FP32, {8, 8}, "operand2");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BOOL); }
}

TEST_F(OperationImplTest, test_Sort)
{
    TileShape::Current().SetVecTile({6, 64});
    Tensor operand1(DT_FP32, {6, 64}, "operand1");
    Tensor sort32Result;
    Tensor mrgsortResult;
    FUNCTION("TestSort")
    {
        sort32Result = Sort32(operand1, 0);
        mrgsortResult = MrgSort(sort32Result, 32);
    }
}

TEST_F(OperationImplTest, test_Compare_BIT)
{
    TileShape::Current().SetVecTile({8, 8});
    Tensor operand1(DT_FP16, {16, 16}, "operand1");
    Tensor operand2(DT_FP16, {16, 16}, "operand2");
    std::vector<int64_t> dstShape = {16, 2};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BIT); }
}

TEST_F(OperationImplTest, test_Cmps_BOOL)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    float scalar = 10.0;
    Element operand2(DT_FP32, scalar);
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BOOL); }
}

TEST_F(OperationImplTest, test_Cmps_BIT)
{
    TileShape::Current().SetVecTile({8, 8});
    Tensor operand1(DT_FP16, {16, 16}, "operand1");
    float scalar = 10.0;
    Element operand2(DT_FP16, scalar);
    std::vector<int64_t> dstShape = {16, 2};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BIT); }
}

TEST_F(OperationImplTest, test_Hypot_FP32)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    Tensor operand2(DT_FP32, {8, 8}, "operand2");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestHypot") { result = Hypot(operand1, operand2); }
}

TEST_F(OperationImplTest, test_Hypot_FP16)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP16, {8, 8}, "operand1");
    Tensor operand2(DT_FP16, {8, 8}, "operand2");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestHypot") { result = Hypot(operand1, operand2); }
}

TEST_F(OperationImplTest, test_PReLU_FP32)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    Tensor weight(DT_FP32, {8}, "weight");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestPReLU") { result = PReLU(operand1, weight); }
}

TEST_F(OperationImplTest, test_PReLU_FP16)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP16, {8, 8}, "operand1");
    Tensor weight(DT_FP16, {8}, "weight");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestPReLU") { result = PReLU(operand1, weight); }
}

TEST_F(OperationImplTest, Test_IndexAddUB_BF16)
{
    float scalar = 1.2f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 16});
    Tensor self(DT_BF16, {10, 16}, "operand0");
    Tensor src(DT_BF16, {8, 16}, "operand1");
    Tensor index(DT_INT32, {8}, "operand2");
    Element alpha(DT_BF16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") { result = IndexAddUB(self, src, index, axis, alpha); }
}

TEST_F(OperationImplTest, Test_IndexAddUB_INT16)
{
    int scalar = 2;
    int axis = 1;

    TileShape::Current().SetVecTile({8, 16});
    Tensor self(DT_INT16, {10, 5}, "operand0");
    Tensor src(DT_INT16, {10, 2}, "operand1");
    Tensor index(DT_INT64, {2}, "operand2");
    Element alpha(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") { result = IndexAddUB(self, src, index, axis, alpha); }
}

TEST_F(OperationImplTest, Test_IndexAdd_FP32)
{
    float scalar = 1.2f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 8, 16});
    Tensor self(DT_FP32, {10, 10, 16}, "operand0");
    Tensor src(DT_FP32, {15, 10, 16}, "operand1");
    Tensor index(DT_INT32, {15}, "operand2");
    Element alpha(DT_FP32, scalar);
    FUNCTION("TestIndxAdd") { IndexAdd_(self, src, index, axis, alpha); }
}

TEST_F(OperationImplTest, Test_IndexAddUB_FP16)
{
    float scalar = 1.0f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 8, 8, 16});
    Tensor self(DT_FP16, {10, 10, 10, 16}, "operand0");
    Tensor src(DT_FP16, {8, 10, 10, 16}, "operand1");
    Tensor index(DT_INT64, {8}, "operand2");
    Element alpha(DT_FP16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") { result = IndexAddUB(self, src, index, axis, alpha); }
}

void TestPow(DataType selfType, DataType otherType, DataType resultType)
{
    std::vector<int64_t> shape = {32, 32};
    PROGRAM("POW")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(selfType, shape, "input_a");
        Tensor input_b(otherType, shape, "input_b");
        auto output = Tensor(resultType, shape, "res");
        FUNCTION("POW_FUC") { output = Pow(input_a, input_b); }
    }
}

TEST_F(OperationImplTest, Test_Pow)
{
    TestPow(DataType::DT_FP32, DataType::DT_FP32, DataType::DT_FP32);
    TestPow(DataType::DT_FP16, DataType::DT_FP16, DataType::DT_FP16);
    TestPow(DataType::DT_BF16, DataType::DT_FP16, DataType::DT_FP32);
    TestPow(DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32);
    TestPow(DataType::DT_INT32, DataType::DT_FP32, DataType::DT_FP32);
    TestPow(DataType::DT_FP32, DataType::DT_INT32, DataType::DT_FP32);
    TestPow(DataType::DT_INT32, DataType::DT_INT32, DataType::DT_INT32);
}

TEST_F(OperationImplTest, Test_Pow_FP32_Broadcast)
{
    PROGRAM("POW")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(DataType::DT_FP32, {1, 32}, "input_a");
        Tensor input_b(DataType::DT_FP32, {32, 32}, "input_b");
        auto output = Tensor(DataType::DT_FP32, {32, 32}, "res");
        FUNCTION("POW_FUC") { output = Pow(input_a, input_b); }
    }
}

void TestPows(DataType dataType, double exponent)
{
    std::vector<int64_t> shape = {32, 32};
    PROGRAM("POWS")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(dataType, shape, "input");
        auto output = Tensor(dataType, shape, "res");
        FUNCTION("POWS_FUC") { output = Pow(input_a, Element(dataType, exponent)); }
    }
}

TEST_F(OperationImplTest, Test_Pows_0)
{
    constexpr double EXP0 = 0;
    TestPows(DataType::DT_FP32, EXP0);
    constexpr double EXP_1_5 = -1.5;
    TestPows(DataType::DT_FP32, EXP_1_5);
    constexpr double EXP1_5 = 1.5;
    TestPows(DataType::DT_FP16, EXP1_5);
    constexpr double EXP2 = 2;
    TestPows(DataType::DT_FP32, EXP2);
    constexpr double EXP3 = 3;
    TestPows(DataType::DT_FP32, EXP3);
}

TEST_F(OperationImplTest, Test_Pows_1)
{
    constexpr double EXP1_5 = 1.5;
    std::vector<int64_t> shape = {32, 32};
    PROGRAM("POWS")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(DataType::DT_INT32, shape, "input");
        auto output = Tensor(DataType::DT_FP32, shape, "res");
        FUNCTION("POWS_FUC") { output = Pow(input_a, Element(DataType::DT_FP32, EXP1_5)); }
    }
}

TEST_F(OperationImplTest, Test_LogicalNot_BF16)
{
    PROGRAM("LogicalNot")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("LogicalNot_BF16") { output = LogicalNot(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Expm1_FP16)
{
    PROGRAM("Expm1")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Expm1_FP16") { output = Expm1(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Expm1_FP32)
{
    PROGRAM("Expm1")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Expm1_FP32") { output = Expm1(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Sinh_FP16)
{
    PROGRAM("Sinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sinh_FP16") { output = Sinh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Sinh_FP32)
{
    PROGRAM("Sinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sinh_FP32") { output = Sinh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Cosh_FP16)
{
    PROGRAM("Cosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Cosh_FP16") { output = Cosh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Cosh_FP32)
{
    PROGRAM("Cosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Cosh_FP32") { output = Cosh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_ASinh_FP16)
{
    PROGRAM("ASinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ASinh_FP16") { output = ASinh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_ASinh_FP32)
{
    PROGRAM("ASinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ASinh_FP32") { output = ASinh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_ACosh_FP16)
{
    PROGRAM("ACosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ACosh_FP16") { output = ACosh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_ACosh_FP32)
{
    PROGRAM("ACosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ACosh_FP32") { output = ACosh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Atanh_FP16)
{
    PROGRAM("Atanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Atanh_FP16") { output = Atanh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Atanh_FP32)
{
    PROGRAM("Atanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Atanh_FP32") { output = Atanh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Atanh_BF16)
{
    PROGRAM("Atanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "operand1");
        auto output = Tensor(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Atanh_BF16") { output = Atanh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Sign_FP16)
{
    PROGRAM("Sign")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sign_FP16") { output = Sign(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Sign_FP32)
{
    PROGRAM("Sign")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sign_FP32") { output = Sign(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Sign_INT16)
{
    PROGRAM("Sign")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_INT16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sign_INT16") { output = Sign(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Signbit_FP16)
{
    PROGRAM("Signbit")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Signbit_FP16") { output = Signbit(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Signbit_FP32)
{
    PROGRAM("Signbit")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Signbit_FP32") { output = Signbit(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Tan_FP16)
{
    PROGRAM("Tan")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tan_FP16") { output = Tan(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Tan_FP32)
{
    PROGRAM("Tan")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tan_FP32") { output = Tan(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Tan_BF16)
{
    PROGRAM("Tan")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        auto output = Tensor(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tan_BF16") { output = Tan(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Tanh_FP16)
{
    PROGRAM("Tanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tanh_FP16") { output = Tanh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Tanh_FP32)
{
    PROGRAM("Tanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tanh_FP32") { output = Tanh(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Log1p_FP16)
{
    PROGRAM("Log1p")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Log1p_FP16") { output = Log1p(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Log1p_FP32)
{
    PROGRAM("Log1p")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Log1p_FP32") { output = Log1p(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Log1p_BF16)
{
    PROGRAM("Log1p")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        auto output = Tensor(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Log1p_BF16") { output = Log1p(input_a); }
    }
}

TEST_F(OperationImplTest, Test_WhereTT_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        Tensor input_b(DT_BF16, shape, "B");
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, input_a, input_b); }
    }
}

TEST_F(OperationImplTest, Test_WhereTS_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        float scalar = 10.0;
        Element operand2(DT_BF16, scalar);
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, input_a, operand2); }
    }
}

TEST_F(OperationImplTest, Test_WhereSS_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        float scalar = 10.0;
        Element operand1(DT_BF16, scalar);
        Element operand2(DT_BF16, scalar);
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, operand1, operand2); }
    }
}

TEST_F(OperationImplTest, Test_WhereST_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        float scalar = 10.0;
        Element operand1(DT_BF16, scalar);
        Tensor input_b(DT_BF16, shape, "B");
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, operand1, input_b); }
    }
}

template <DataType inputType, DataType outputType, bool IsANZ = false, bool IsBNZ = false, bool isTransB = false>
void TestNZFormatBatch(int bs, int m, int k, int n)
{
    std::vector<int64_t> batch_shape_a = {bs * m, k};
    auto nLen = isTransB ? bs * n : bs * k;
    auto kLen = isTransB ? k : n;
    std::vector<int64_t> batch_shape_b = {nLen, kLen};
    std::vector<int64_t> batch_shape_c = {bs * m, n};
    PROGRAM("BATCHMATMUL")
    {
        config::Reset();
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        auto afmt = IsANZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor matA(inputType, batch_shape_a, "MatA", afmt);
        Tensor matB(inputType, batch_shape_b, "MatB", bfmt);
        Tensor matC(outputType, batch_shape_c, "MatC");
        std::vector<Tensor> matrixVec;
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC})
        {
            std::vector<std::pair<Tensor, std::vector<int64_t>>> assembleVec;
            for (size_t index = 0; index < (size_t)bs; ++index) {
                auto inputA = View(matA, {m, k}, {(int)index * m, 0});
                auto inputB = isTransB ? View(matB, {n, k}, {(int)index * n, 0}) :
                                         View(matB, {k, n}, {(int)index * k, 0});
                TileShape::Current().SetMatrixSize({m, k, n});
                auto outTensor = npu::tile_fwk::Matrix::Matmul(outputType, inputA, inputB, false, isTransB);
                std::vector<int64_t> pairSecond = {(int)index * m, 0};
                auto pair = std::make_pair(outTensor, pairSecond);
                assembleVec.emplace_back(pair);
            }
            matC = Assemble(assembleVec);
        }
    }
}

TEST_F(OperationImplTest, test_Range_FP16)
{
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_FP16, startValue);
    Element end(DT_FP16, endValue);
    Element step(DT_FP16, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(OperationImplTest, test_Range_BF16)
{
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_BF16, startValue);
    Element end(DT_BF16, endValue);
    Element step(DT_BF16, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(OperationImplTest, test_Range_FP32)
{
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_FP32, startValue);
    Element end(DT_FP32, endValue);
    Element step(DT_FP32, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(OperationImplTest, test_Range_INT32)
{
    int startValue = 1;
    int endValue = 10;
    int stepValue = 3;
    int NUM_Eight = 8;
    Element start(DT_INT32, startValue);
    Element end(DT_INT32, endValue);
    Element step(DT_INT32, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(OperationImplTest, Test_Uniform_UINT32)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_UINT32")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_FP32);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Uniform_FP16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_FP16")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_FP16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Uniform_BF16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_BF16")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_BF16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Uniform_Rounds7)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_Rounds7")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(7)),
                             DT_FP32);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Uniform_FP16_Rounds7)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_FP16_Rounds7")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(7)),
                             DT_FP16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Uniform_BF16_Rounds7)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_BF16_Rounds7")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(7)),
                             DT_BF16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Uniform_LargeShape)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {256};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 9876543210;
        uint64_t counter0 = 100;
        uint64_t counter1 = 200;
        Tensor output(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_LargeShape")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_FP32);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Exp2_FP16)
{
    PROGRAM("Exp2")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Exp2_FP16") { output = Exp2(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Exp2_FP32)
{
    PROGRAM("Exp2")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Exp2_FP32") { output = Exp2(input_a); }
    }
}

TEST_F(OperationImplTest, Test_Round_FP16)
{
    PROGRAM("Round")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Round_FP16") { output = Round(input_a, 1); }
    }
}

TEST_F(OperationImplTest, Test_Round_FP32)
{
    PROGRAM("Round")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Round_FP32") { output = Round(input_a, 1); }
    }
}

TEST_F(OperationImplTest, test_Rsqrt_FP16)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP16, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRsqrt") { result = Rsqrt(operand1); }
}

TEST_F(OperationImplTest, test_Rsqrt_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRsqrt") { result = Rsqrt(operand1); }
}

TEST_F(OperationImplTest, test_Ceil_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestCeil") { result = Ceil(operand1); }
}

TEST_F(OperationImplTest, test_Floor_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestFloor") { result = Floor(operand1); }
}

TEST_F(OperationImplTest, test_Trunc_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestTrunc") { result = Trunc(operand1); }
}

TEST_F(OperationImplTest, test_FloorDiv_int32)
{
    TileShape::Current().SetVecTile(16, 32);
    Tensor self(DT_INT32, {64, 64}, "self");
    Tensor other(DT_INT32, {64, 64}, "other");
    Tensor result;
    FUNCTION("TestFloorDiv") { result = FloorDiv(self, other); }
}

TEST_F(OperationImplTest, test_FloorDivs_int32)
{
    TileShape::Current().SetVecTile(16, 32);
    Tensor self(DT_INT32, {64, 64}, "self");
    Element other(DT_INT32, 2);
    Tensor result;
    FUNCTION("TestFloorDivs") { result = FloorDiv(self, other); }
}

TEST_F(OperationImplTest, test_Reciprocal_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestReciprocal") { result = Reciprocal(operand1); }
}

TEST_F(OperationImplTest, test_Relu_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRelu") { result = Relu(operand1); }
}

TEST_F(OperationImplTest, TestIndexPut_)
{
    constexpr int TILE_SHAPE = 8;
    TileShape::Current().SetVecTile(TILE_SHAPE);
    Shape shapeSelf({128, 8, 8});
    Shape shapeValues({128, 8});
    Shape shapeIndices({128});
    Tensor self(DT_INT32, shapeSelf, "self");
    Tensor values(DT_INT32, shapeValues, "values");
    Tensor indices0(DT_INT32, shapeIndices, "indices0");
    Tensor indices1(DT_INT32, shapeIndices, "indices1");
    std::vector<Tensor> indices{indices0, indices1};
    bool accumulate = false;
    Tensor result;
    FUNCTION("TestIndexPut_") { IndexPut_(self, indices, values, accumulate); }
}

TEST_F(OperationImplTest, test_Expand_8_1_to_8_8)
{
    TileShape::Current().SetVecTile({4, 4});

    Tensor operand1(DT_FP32, {8, 1}, "operand1");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestExpand") { result = Expand(operand1, dstShape); }
}

TEST_F(OperationImplTest, test_Expand_8_1_to_8_8_dyn)
{
    TileShape::Current().SetVecTile({3, 3});

    Tensor operand1(DT_FP32, {8, 1}, "operand1");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestExpand") { result = Expand(operand1, dstShape); }
}

TEST_F(OperationImplTest, test_Clip_FP16)
{
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP16, {8, 16, 16}, "src");
    Element min(DT_FP16, minValue);
    Element max(DT_FP16, maxValue);

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(OperationImplTest, test_Clip_FP32_VS)
{
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Element min(DT_FP32, minValue);
    Element max(DT_FP32, maxValue);

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(OperationImplTest, test_Clip_FP16_VS)
{
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP16, {8, 16, 16}, "src");
    Element min(DT_FP16, minValue);
    Element max(DT_FP16, maxValue);

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(OperationImplTest, test_Clip_FP32_VV)
{
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Tensor min(DT_FP32, {8, 16, 16}, "min");
    Tensor max(DT_FP32, {8, 16, 16}, "max");

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(OperationImplTest, test_Clip_FP32_VV_BRC)
{
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Tensor min(DT_FP32, {8, 1, 16}, "min");
    Tensor max(DT_FP32, {1, 16, 16}, "max");

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(OperationImplTest, Test_Amax)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand(DT_FP32, {16, 16}, "operand");
    Tensor result;
    FUNCTION("TestAmax") { result = Amax(operand, -1, true); }
}

TEST_F(OperationImplTest, Test_Amax_NegativeAxisOutOfRange)
{
    Tensor operand(DT_INT16, {4, 5120}, "operand");
    Tensor result;
    auto runAmax = [&]() {
        FUNCTION("TestAmaxNegativeAxisOutOfRange")
        {
            auto viewTensor = View(operand, {4, 138}, {4, 138}, {0, 0});
            TileShape::Current().SetVecTile(4, 16);
            result = Amax(viewTensor, -3, true);
        }
    };
    EXPECT_THROW(runAmax(), Error);
}

TEST_F(OperationImplTest, Test_Amin)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand(DT_FP32, {16, 16}, "operand");
    Tensor result;
    FUNCTION("TestAmin") { result = Amin(operand, -1, true); }
}

TEST_F(OperationImplTest, Test_Amin_NegativeAxisOutOfRange)
{
    Tensor operand(DT_INT16, {4, 5120}, "operand");
    Tensor result;
    auto runAmin = [&]() {
        FUNCTION("TestAminNegativeAxisOutOfRange")
        {
            auto viewTensor = View(operand, {4, 138}, {4, 138}, {0, 0});
            TileShape::Current().SetVecTile(4, 16);
            result = Amin(viewTensor, -3, true);
        }
    };
    EXPECT_THROW(runAmin(), Error);
}

TEST_F(OperationImplTest, test_Atan)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input(DT_FP16, {16, 16}, "input");
    Tensor result;
    FUNCTION("TestAtan") { result = Atan(input); }
}

TEST_F(OperationImplTest, test_Atan2)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input1(DT_FP16, {16, 16}, "input1");
    Tensor input2(DT_FP16, {16, 16}, "input2");
    Tensor result;
    FUNCTION("TestAtan2") { result = Atan2(input1, input2); }
}

TEST_F(OperationImplTest, test_Gather)
{
    TileShape::Current().SetVecTile(8, 8, 8);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT32, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGather") { result = Gather(operand1, operand2, -1); }
}

TEST_F(OperationImplTest, test_GatherMask_1)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGatherMask") { result = GatherMask(operand1, 1); }
}

TEST_F(OperationImplTest, test_GatherMask_3)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGatherMask") { result = GatherMask(operand1, 3); }
}

TEST_F(OperationImplTest, test_GatherMask_ViewSplitLastAxis_ExpectError)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 32}, "operand1");
    Tensor result;
    auto runGatherMask = [&]() {
        FUNCTION("TestGatherMaskViewSplit")
        {
            auto viewTensor = View(operand1, {8, 16}, {8, 16}, {0, 0});
            result = GatherMask(viewTensor, 1);
        }
    };
    EXPECT_THROW(runGatherMask(), Error);
}

TEST_F(OperationImplTest, test_GatherMask_ViewSplitLastAxis_Mode3_ExpectError)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 32}, "operand1");
    Tensor result;
    auto runGatherMask = [&]() {
        FUNCTION("TestGatherMaskViewSplitMode3")
        {
            auto viewTensor = View(operand1, {8, 16}, {8, 16}, {0, 0});
            result = GatherMask(viewTensor, 3);
        }
    };
    EXPECT_THROW(runGatherMask(), Error);
}

TEST_F(OperationImplTest, test_GatherMask_ViewSplitLastAxis_4D_ExpectError)
{
    TileShape::Current().SetVecTile(2, 3, 5, 16);
    Tensor operand1(DT_FP32, {4, 8, 8, 32}, "operand1");
    Tensor result;
    auto runGatherMask = [&]() {
        FUNCTION("TestGatherMaskViewSplit4D")
        {
            auto viewTensor = View(operand1, {4, 8, 8, 16}, {4, 8, 8, 16}, {0, 0, 0, 0});
            result = GatherMask(viewTensor, 2);
        }
    };
    EXPECT_THROW(runGatherMask(), Error);
}

TEST_F(OperationImplTest, test_GatherMask_ViewEqualInputShape_Pass)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGatherMaskViewEqual")
    {
        auto viewTensor = View(operand1, {8, 16}, {8, 16}, {0, 0});
        result = GatherMask(viewTensor, 1);
    }
}

TEST_F(OperationImplTest, test_Scatter_FP16)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT64, {2, 16}, "operand2");
    Element operand3(DT_FP16, 1.0);
    Tensor result;
    FUNCTION("TestScatter") { result = Scatter(operand1, operand2, operand3, 0); }
}

TEST_F(OperationImplTest, test_ScatterTensor_FP16)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT64, {2, 16}, "operand2");
    Tensor operand3(DT_FP16, {2, 16}, "operand3");
    Tensor result;
    FUNCTION("TestScatter") { result = Scatter(operand1, operand2, operand3, 0); }
}

TEST_F(OperationImplTest, test_Var_FP16)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestVar") { result = Var(operand1); }
}

TEST_F(OperationImplTest, test_Where)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor condition(DT_UINT8, {8, 2}, "condition");
    Tensor input(DT_FP32, {8, 16}, "input");
    Tensor other(DT_FP32, {8, 16}, "other");
    Tensor result;
    FUNCTION("TestWhere") { result = Where(condition, input, other); }
}

TEST_F(OperationImplTest, test_Add_Brcb)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 1}, "input0");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    FUNCTION("TestAddBrcb") { result = Add(input0, input1); }
}

TEST_F(OperationImplTest, test_Fmod)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 16}, "input1");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, false);
    FUNCTION("TestFmod") { result = Fmod(input0, input1); }
}

TEST_F(OperationImplTest, test_Fmod_Brcb)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 1}, "input1");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    FUNCTION("TestFmodBrcb") { result = Fmod(input0, input1); }
}

TEST_F(OperationImplTest, test_FmodS)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor input0(DT_FP32, {8, 8}, "input0");
    float scalar = 10.0;
    Element input1(DT_FP32, scalar);
    Tensor result;
    FUNCTION("TestFmodS") { result = Fmod(input0, input1); }
}

TEST_F(OperationImplTest, test_LReLU)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor input0(DT_FP32, {8, 8}, "input0");
    float scalar = 0.01f;
    Element input1(DT_FP32, scalar);
    Tensor result;
    FUNCTION("TestLReLU") { result = LReLU(input0, input1); }
}

TEST_F(OperationImplTest, Test_TopK_01)
{
    std::vector<int64_t> inputShape = {1, 16384};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(OperationImplTest, Test_TopK_02)
{
    std::vector<int64_t> inputShape = {1, 24576};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(OperationImplTest, Test_TopK_03)
{
    std::vector<int64_t> inputShape = {1, 49152};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(OperationImplTest, Test_TopK_04)
{
    std::vector<int64_t> inputShape = {1, 40960};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(OperationImplTest, Test_ArgSort_01)
{
    std::vector<int64_t> inputShape = {16, 128};
    std::vector<int64_t> outputShape = {16, 128};
    TileShape::Current().SetVecTile({4, 32});
    Tensor input_a(DT_FP32, inputShape, "A");
    Tensor output(DT_INT32, outputShape, "res");
    FUNCTION("ArgSort_T") { output = ArgSort(input_a, 1, true); }
}

TEST_F(OperationImplTest, Test_ArgSort_02)
{
    std::vector<int64_t> inputShape = {1, 9000};
    std::vector<int64_t> outputShape = {1, 9000};
    TileShape::Current().SetVecTile({1, 5024});
    Tensor input_a(DT_FP32, inputShape, "A");
    Tensor output(DT_INT32, outputShape, "res");
    FUNCTION("ArgSort_T") { output = ArgSort(input_a, 1, false); }
}

TEST_F(OperationImplTest, Test_BitwiseRightShift)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(OperationImplTest, Test_BitwiseRightShift_brc)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {1, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(OperationImplTest, Test_BitwiseRightShifts)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    int scalar = 1;
    Element other(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(OperationImplTest, Test_SBitwiseRightShift)
{
    TileShape::Current().SetVecTile({16, 16});
    int scalar = 1;
    Element self(DT_INT16, scalar);
    Tensor other(DT_INT16, {16, 16}, "self");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(OperationImplTest, Test_BitwiseLeftShift)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(OperationImplTest, Test_BitwiseLeftShift_brc)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {1, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(OperationImplTest, Test_BitwiseLeftShifts)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    int scalar = 1;
    Element other(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(OperationImplTest, Test_SBitwiseLeftShift)
{
    TileShape::Current().SetVecTile({16, 16});
    int scalar = 1;
    Element self(DT_INT16, scalar);
    Tensor other(DT_INT16, {16, 16}, "self");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(OperationImplTest, Test_CopySign)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_FP32, {16, 16}, "self");
    Tensor other(DT_FP32, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = CopySign(self, other); }
}

TEST_F(OperationImplTest, Test_CopySign_int)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT32, {16, 16}, "self");
    Tensor other(DT_INT32, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = CopySign(self, other); }
}

TEST_F(OperationImplTest, Test_Conv2d_FP16_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_FP32_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP32, {1, 8, 2, 64}, "fmap");
    Tensor weight(DT_FP32, {32, 8, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_BF16_Groups_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_BF16, {1, 32, 2, 64}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 2);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv1d_FP16_Bias_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP16, {1, 32, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 32, 3}, "weight");
    Tensor bias(DT_FP16,
                {
                    32,
                },
                "bias");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    convExtendParam.biasTensor = bias;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_FP16_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_FP32_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP32, {1, 8, 2, 64}, "fmap");
    Tensor weight(DT_FP32, {32, 8, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_BF16_Groups_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_BF16, {1, 32, 2, 64}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 2);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv1d_FP16_Bias_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP16, {1, 32, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 32, 3}, "weight");
    Tensor bias(DT_FP16,
                {
                    32,
                },
                "bias");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    convExtendParam.biasTensor = bias;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_FP16_SmallTile_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_FP16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {2, 2}, {0, 0, 0, 0}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_BF16_Stride2_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 32, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 32);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_BF16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {2, 2}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv1d_FP32_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP32, {1, 8, 32}, "fmap");
    Tensor weight(DT_FP32, {16, 8, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {2}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv3d_FP16_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 4, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 2, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 2, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1},
                                           convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_FP16_SmallTile_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_FP16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {2, 2}, {0, 0, 0, 0}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv2d_BF16_Stride2_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 32, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 32);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_BF16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {2, 2}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv1d_FP32_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP32, {1, 8, 32}, "fmap");
    Tensor weight(DT_FP32, {16, 8, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {2}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, Test_Conv3d_FP16_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 4, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 2, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 2, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1},
                                           convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplTest, test_Pad_1D)
{
    TileShape::Current().SetVecTile(8);
    Tensor input(DT_FP32, {10}, "input");
    Tensor result;
    FUNCTION("TestPad1D") { result = Pad(input, {0, 36}, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(OperationImplTest, test_Pad_2D)
{
    TileShape::Current().SetVecTile(4, 4);
    Tensor input(DT_FP32, {6, 6}, "input");
    Tensor result;
    FUNCTION("TestPad2D") { result = Pad(input, {0, 12, 0, 12}, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(OperationImplTest, Test_Matmul_Bias)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {128, 256}, "matA");
    Tensor matB(DT_BF16, {256, 128}, "matB");
    Tensor matBias(DT_FP32, {1, 128}, "biasA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestMatmulBias")
    {
        result = npu::tile_fwk::Matrix::Matmul(DT_FP32, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(OperationImplTest, Test_MatmulMX_Bias)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_FP8E5M2, {128, 256}, "matA");
    Tensor matB(DT_FP8E5M2, {256, 128}, "matB");
    Tensor scaleA(DT_FP8E8M0, {128, 4, 2}, "scaleA");
    Tensor scaleB(DT_FP8E8M0, {4, 128, 2}, "scaleB");
    Tensor matBias(DT_BF16, {1, 128}, "biasA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestMatmulMXBias")
    {
        result = npu::tile_fwk::Matrix::MatmulMX(DT_FP32, matA, scaleA, matB, scaleB, extendParam, false, false, false,
                                                 false, false);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(OperationImplTest, Test_BatchMatmul_Bias_3D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {2, 128, 256}, "matA");
    Tensor matB(DT_BF16, {2, 256, 128}, "matB");
    Tensor matBias(DT_FP32, {1, 128}, "biasA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulBiasWith3Dim")
    {
        result = npu::tile_fwk::Matrix::BatchMatmul(DT_FP32, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(OperationImplTest, Test_BatchMatmul_Bias_4D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {1, 2, 128, 256}, "matA");
    Tensor matB(DT_BF16, {2, 2, 256, 128}, "matB");
    Tensor matBias(DT_FP32, {1, 128}, "biasA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulBiasWith4Dim")
    {
        result = npu::tile_fwk::Matrix::BatchMatmul(DT_FP32, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(OperationImplTest, Test_BatchMatmul_fixpipe_4D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_INT8, {1, 2, 128, 256}, "matA");
    Tensor matB(DT_INT8, {2, 2, 256, 128}, "matB");
    Tensor matfixpipe(DT_UINT64, {1, 128}, "fixpipeA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.scaleTensor = matfixpipe;
    FUNCTION("TestBatchMatmulFixpipeWith4Dim")
    {
        result = npu::tile_fwk::Matrix::BatchMatmul(DT_FP16, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(OperationImplTest, Test_BatchMatmul_fixpipe_3D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_INT8, {2, 128, 256}, "matA");
    Tensor matB(DT_INT8, {2, 256, 128}, "matB");
    Tensor matfixpipe(DT_UINT64, {1, 128}, "fixpipeA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.scaleTensor = matfixpipe;
    FUNCTION("TestBatchMatmulFixpipeWith3Dim")
    {
        result = npu::tile_fwk::Matrix::BatchMatmul(DT_FP16, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(OperationImplTest, Test_BatchMatmulMX_Bias_3D)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_FP8E5M2, {2, 128, 256}, "matA");
    Tensor matB(DT_FP8E5M2, {2, 256, 128}, "matB");
    Tensor scaleA(DT_FP8E8M0, {128, 4, 2}, "scaleA");
    Tensor scaleB(DT_FP8E8M0, {4, 128, 2}, "scaleB");
    Tensor matBias(DT_BF16, {1, 128}, "biasA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulMXBias3D")
    {
        result = npu::tile_fwk::Matrix::BatchMatmulMX(DT_FP32, matA, scaleA, matB, scaleB, extendParam, false, false,
                                                      false, false, false);
    }
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(OperationImplTest, Test_BatchMatmulMX_Bias_4D)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_FP8E5M2, {2, 2, 128, 256}, "matA");
    Tensor matB(DT_FP8E5M2, {2, 2, 256, 128}, "matB");
    Tensor scaleA(DT_FP8E8M0, {128, 4, 2}, "scaleA");
    Tensor scaleB(DT_FP8E8M0, {4, 128, 2}, "scaleB");
    Tensor matBias(DT_BF16, {1, 128}, "biasA");
    Tensor result;
    npu::tile_fwk::Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulMXBias4D")
    {
        result = npu::tile_fwk::Matrix::BatchMatmulMX(DT_FP32, matA, scaleA, matB, scaleB, extendParam, false, false,
                                                      false, false, false);
    }
    Platform::Instance().ReloadMemoryPaths("2201");
}

TEST_F(OperationImplTest, Test_BatchMatmulGCC_3D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128}, true);
    Tensor matA(DT_FP16, {2, 128, 256}, "matA");
    Tensor matB(DT_FP16, {2, 256, 128}, "matB");
    Tensor result;
    FUNCTION("TestBatchMatmulGCC3D")
    {
        result = npu::tile_fwk::Matrix::BatchMatmul(DT_FP32, matA, matB, false, false, false);
    }
}

TEST_F(OperationImplTest, Test_BatchMatmulGCC_4D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128}, true);
    Tensor matA(DT_INT8, {2, 2, 128, 256}, "matA");
    Tensor matB(DT_INT8, {2, 2, 256, 128}, "matB");
    Tensor result;
    FUNCTION("TestBatchMatmulGCC4D")
    {
        result = npu::tile_fwk::Matrix::BatchMatmul(DT_INT32, matA, matB, false, false, false);
    }
}

TEST_F(OperationImplTest, test_FillPad_1D)
{
    TileShape::Current().SetVecTile(8);
    Tensor input(DT_FP32, {10}, "input");
    Tensor result;
    FUNCTION("TestFillPad1D") { result = FillPad(input, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(OperationImplTest, test_FillPad_2D)
{
    TileShape::Current().SetVecTile(4, 4);
    Tensor input(DT_FP32, {6, 6}, "input");
    Tensor result;
    FUNCTION("TestFillPad2D") { result = FillPad(input, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(OperationImplTest, Test_Permute_3D_FP32)
{
    TileShape::Current().SetVecTile(4, 8, 16);
    Tensor input(DT_FP32, {4, 8, 16}, "input");
    Tensor result;
    FUNCTION("TestPermute3D") { result = Permute(input, {1, 0, 2}); }
}

TEST_F(OperationImplTest, Test_Permute_3D_FP16)
{
    TileShape::Current().SetVecTile(4, 8, 16);
    Tensor input(DT_FP16, {4, 8, 16}, "input");
    Tensor result;
    FUNCTION("TestPermute3D_FP16") { result = Permute(input, {0, 2, 1}); }
}

TEST_F(OperationImplTest, Test_Permute_4D_FP32)
{
    TileShape::Current().SetVecTile(4, 4, 8, 8);
    Tensor input(DT_FP32, {4, 4, 8, 8}, "input");
    Tensor result;
    FUNCTION("TestPermute4D") { result = Permute(input, {0, 2, 3, 1}); }
}

TEST_F(OperationImplTest, Test_Permute_5D_FP32)
{
    TileShape::Current().SetVecTile(2, 2, 4, 4, 4);
    Tensor input(DT_FP32, {2, 2, 4, 4, 4}, "input");
    Tensor result;
    FUNCTION("TestPermute5D") { result = Permute(input, {4, 3, 2, 1, 0}); }
}

TEST_F(OperationImplTest, Test_Permute_Identity)
{
    TileShape::Current().SetVecTile(4, 8);
    Tensor input(DT_FP32, {4, 8}, "input");
    Tensor result;
    FUNCTION("TestPermuteIdentity") { result = Permute(input, {0, 1}); }
}

TEST_F(OperationImplTest, Test_Permute_1D)
{
    TileShape::Current().SetVecTile(32);
    Tensor input(DT_FP32, {32}, "input");
    Tensor result;
    FUNCTION("TestPermute1D") { result = Permute(input, {0}); }
}

TEST_F(OperationImplTest, Test_Matmul_SFA)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {128, 576}, "matA");
    Tensor matB(DT_BF16, {576, 2048}, "matB");
    Tensor result;
    FUNCTION("TestMatmulSFA") { result = npu::tile_fwk::Matrix::Matmul(DT_FP32, matA, matB, false, false, false); }
}

TEST_F(OperationImplTest, Test_Matmul_SFA_T)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {576, 128}, "matA");
    Tensor matB(DT_BF16, {576, 2048}, "matB");
    Tensor result;
    FUNCTION("TestMatmulSFAT") { result = npu::tile_fwk::Matrix::Matmul(DT_FP32, matA, matB, true, false, false); }
}

TEST_F(OperationImplTest, test_Erfc_fp16)
{
    TileShape::Current().SetVecTile(8, 128);
    Tensor input(DT_FP16, {8, 4608}, "input");
    Tensor result;
    FUNCTION("TestErfc") { result = Erfc(input); }
}

TEST_F(OperationImplTest, test_Erfc_fp32)
{
    TileShape::Current().SetVecTile(8, 128);
    Tensor input(DT_FP32, {8, 4608}, "input");
    Tensor result;
    FUNCTION("TestErfc") { result = Erfc(input); }
}
