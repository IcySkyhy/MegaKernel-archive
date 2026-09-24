/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_tensor_ops.cpp
 * \brief Coverage tests for tensor_ops type deduction via OpRegistry::Create
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// MakeShapeTuple is just an alias for MakeOffsetsTuple in this context
static ExprPtr MakeShapeTuple(std::vector<int64_t> dims) { return MakeOffsetsTuple(dims); }

// ============================================================================
// elementwise.cpp: tensor.add (binary), tensor.add_scalar (tensor+scalar)
// ============================================================================

class TensorOpsElemwiseTest : public testing::Test {};

TEST_F(TensorOpsElemwiseTest, TensorAdd_TwoTensors_ReturnsBroadcastedType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP16);
    auto rhs = MakeTensorVar("b", {4, 8}, DataType::FP16);
    auto call = reg.Create("tensor.add", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorAdd_Broadcasting_ReturnsLargerShape)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto rhs = MakeTensorVar("b", {8}, DataType::FP32);
    auto call = reg.Create("tensor.add", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorAdd_DtypePromotion)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {16}, DataType::INT32);
    auto rhs = MakeTensorVar("b", {16}, DataType::FP32);
    auto call = reg.Create("tensor.add", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(TensorOpsElemwiseTest, TensorAdd_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("tensor.add", {MakeTensorVar("a", {16}, DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(TensorOpsElemwiseTest, TensorAdd_NonTensorArg_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("tensor.add",
                                  {MakeScalarVar("s", DataType::FP32), MakeTensorVar("b", {16}, DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(TensorOpsElemwiseTest, TensorAddScalar_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP16);
    auto rhs = MakeScalarVar("s", DataType::FP16);
    auto call = reg.Create("tensor.add_scalar", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorAddScalar_NonScalarSecond_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("tensor.add_scalar",
                         {MakeTensorVar("a", {16}, DataType::FP32), MakeTensorVar("b", {16}, DataType::FP32)}, Sp()),
        npu::tile_fwk::Error);
}

// Representative coverage for other elementwise binary ops (sub, mul, div, maximum).
// All use DeduceTensorOpElementwiseBinaryType: validate 2 tensor args, broadcast shapes, promote dtypes.

TEST_F(TensorOpsElemwiseTest, TensorSub_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP16);
    auto rhs = MakeTensorVar("b", {4, 8}, DataType::FP16);
    auto call = reg.Create("tensor.sub", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorMul_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto rhs = MakeTensorVar("b", {4, 8}, DataType::FP32);
    auto call = reg.Create("tensor.mul", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorDiv_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto rhs = MakeTensorVar("b", {4, 8}, DataType::FP32);
    auto call = reg.Create("tensor.div", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorMaximum_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP16);
    auto rhs = MakeTensorVar("b", {4, 8}, DataType::FP16);
    auto call = reg.Create("tensor.maximum", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

// Representative coverage for elementwise scalar ops (sub_scalar, mul_scalar, div_scalar).
// All use DeduceTensorOpElementwiseScalarType: validate tensor+scalar args, preserve tensor shape.

TEST_F(TensorOpsElemwiseTest, TensorSubScalar_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP16);
    auto rhs = MakeScalarVar("s", DataType::FP16);
    auto call = reg.Create("tensor.sub_scalar", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorMulScalar_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto rhs = MakeScalarVar("s", DataType::FP32);
    auto call = reg.Create("tensor.mul_scalar", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsElemwiseTest, TensorDivScalar_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto lhs = MakeTensorVar("a", {4, 8}, DataType::FP32);
    auto rhs = MakeScalarVar("s", DataType::FP32);
    auto call = reg.Create("tensor.div_scalar", {lhs, rhs}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

// ============================================================================
// unary.cpp: tensor.exp, tensor.cast
// ============================================================================

class TensorOpsUnaryTest : public testing::Test {};

TEST_F(TensorOpsUnaryTest, TensorExp_FloatInput_PreservesDtype)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("tensor.exp", {MakeTensorVar("t", {16, 32}, DataType::FP16)}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(TensorOpsUnaryTest, TensorExp_IntInput_PromotesToFP32)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("tensor.exp", {MakeTensorVar("t", {16}, DataType::INT32)}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(TensorOpsUnaryTest, TensorExp_NonTensor_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("tensor.exp", {MakeScalarVar("s", DataType::FP32)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(TensorOpsUnaryTest, TensorCast_ChangesDtype)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_type", DataType::FP16}};
    auto call = reg.Create("tensor.cast", {MakeTensorVar("t", {8, 16}, DataType::FP32)}, kwargs, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsUnaryTest, TensorCast_MissingTargetType_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("tensor.cast", {MakeTensorVar("t", {8}, DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// matmul.cpp: tensor.matmul
// ============================================================================

class TensorOpsMatmulTest : public testing::Test {};

TEST_F(TensorOpsMatmulTest, Matmul_2Dx2D_ReturnsCorrectShape)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"a_trans", false}, {"b_trans", false}};
    auto call = reg.Create("tensor.matmul",
                           {MakeTensorVar("a", {16, 32}, DataType::FP16), MakeTensorVar("b", {32, 64}, DataType::FP16)},
                           kwargs, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsMatmulTest, Matmul_WithTranspose_ReturnsCorrectShape)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"a_trans", true}, {"b_trans", false}};
    auto call = reg.Create("tensor.matmul",
                           {MakeTensorVar("a", {32, 16}, DataType::FP16), MakeTensorVar("b", {32, 64}, DataType::FP16)},
                           kwargs, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsMatmulTest, Matmul_VectorDot_ReturnsScalarTensor)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"a_trans", false}, {"b_trans", false}};
    auto call = reg.Create("tensor.matmul",
                           {MakeTensorVar("a", {32}, DataType::FP32), MakeTensorVar("b", {32}, DataType::FP32)}, kwargs,
                           Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 0u);
}

TEST_F(TensorOpsMatmulTest, Matmul_Batched3D_ReturnsBatchedShape)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"a_trans", false}, {"b_trans", false}};
    auto call = reg.Create(
        "tensor.matmul",
        {MakeTensorVar("a", {4, 16, 32}, DataType::FP16), MakeTensorVar("b", {4, 32, 64}, DataType::FP16)}, kwargs,
        Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 3u);
}

TEST_F(TensorOpsMatmulTest, Matmul_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"a_trans", false}, {"b_trans", false}};
    EXPECT_THROW((void)reg.Create("tensor.matmul", {MakeTensorVar("a", {16, 32}, DataType::FP16)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// tensor.create, tensor.view, tensor.assemble
// ============================================================================

class TensorOpsMemoryTest : public testing::Test {};

TEST_F(TensorOpsMemoryTest, TensorCreate_ReturnsTensorWithShape)
{
    auto& reg = OpRegistry::GetInstance();
    auto shape = MakeShapeTuple({16, 32});
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", DataType::FP16}};
    auto call = reg.Create("tensor.create", {shape}, kwargs, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsMemoryTest, TensorCreate_MissingDtype_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto shape = MakeShapeTuple({16});
    EXPECT_THROW((void)reg.Create("tensor.create", {shape}, Sp()), npu::tile_fwk::Error);
}

TEST_F(TensorOpsMemoryTest, TensorView_ReturnsTensorWithNewShape)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("t", {64, 128}, DataType::FP32);
    auto shape = MakeShapeTuple({32, 64});
    auto offset = MakeShapeTuple({0, 0});
    auto call = reg.Create("tensor.view", {tensor, shape, offset}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsMemoryTest, TensorView_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW(
        (void)reg.Create("tensor.view", {MakeTensorVar("t", {64}, DataType::FP32), MakeShapeTuple({32})}, Sp()),
        npu::tile_fwk::Error);
}

TEST_F(TensorOpsMemoryTest, TensorAssemble_ReturnsTargetType)
{
    auto& reg = OpRegistry::GetInstance();
    auto target = MakeTensorVar("target", {64, 128}, DataType::FP16);
    auto source = MakeTensorVar("source", {16, 32}, DataType::FP16);
    auto offset = MakeShapeTuple({0, 0});
    auto call = reg.Create("tensor.assemble", {target, source, offset}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// ============================================================================
// reduction.cpp: tensor.row_max, tensor.row_sum
// ============================================================================

class TensorOpsReductionTest : public testing::Test {};

TEST_F(TensorOpsReductionTest, TensorRowMax_KeepDim_ReturnsReducedShape)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"axis", int(-1)}, {"keep_dim", true}};
    auto call = reg.Create("tensor.row_max", {MakeTensorVar("t", {16, 32}, DataType::FP16)}, kwargs, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 2u);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(TensorOpsReductionTest, TensorRowSum_NoKeepDim_RemovesDim)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"axis", int(1)}, {"keep_dim", false}};
    auto call = reg.Create("tensor.row_sum", {MakeTensorVar("t", {16, 32}, DataType::FP32)}, kwargs, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 1u);
}

TEST_F(TensorOpsReductionTest, TensorRowMax_1D_NoKeepDim_ReturnsScalar)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"axis", int(0)}, {"keep_dim", false}};
    auto call = reg.Create("tensor.row_max", {MakeTensorVar("t", {64}, DataType::FP32)}, kwargs, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(TensorOpsReductionTest, TensorRowMax_NonTensor_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"axis", int(0)}, {"keep_dim", true}};
    EXPECT_THROW((void)reg.Create("tensor.row_max", {MakeScalarVar("s", DataType::FP32)}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// transform.cpp: tensor.reshape, tensor.transpose
// ============================================================================

class TensorOpsTransformTest : public testing::Test {};

TEST_F(TensorOpsTransformTest, TensorReshape_ReturnsNewShape)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("t", {16, 32}, DataType::FP16);
    auto shape = MakeShapeTuple({8, 64});
    auto call = reg.Create("tensor.reshape", {tensor, shape}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsTransformTest, TensorReshape_IncompatibleSize_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("t", {16, 32}, DataType::FP16);
    auto shape = MakeShapeTuple({7, 7});
    EXPECT_THROW((void)reg.Create("tensor.reshape", {tensor, shape}, Sp()), npu::tile_fwk::Error);
}

TEST_F(TensorOpsTransformTest, TensorTranspose_SwapsAxes)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("t", {16, 32}, DataType::FP16);
    auto axis1 = std::make_shared<ConstInt>(int64_t(0), DataType::INT64, Sp());
    auto axis2 = std::make_shared<ConstInt>(int64_t(1), DataType::INT64, Sp());
    auto call = reg.Create("tensor.transpose", {tensor, axis1, axis2}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(TensorOpsTransformTest, TensorTranspose_NegativeAxis)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("t", {4, 16, 32}, DataType::FP32);
    auto axis1 = std::make_shared<ConstInt>(int64_t(0), DataType::INT64, Sp());
    auto axis2 = std::make_shared<ConstInt>(int64_t(-1), DataType::INT64, Sp());
    auto call = reg.Create("tensor.transpose", {tensor, axis1, axis2}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->shape_.size(), 3u);
}

TEST_F(TensorOpsTransformTest, TensorTranspose_SameAxis_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("t", {16, 32}, DataType::FP16);
    auto axis1 = std::make_shared<ConstInt>(int64_t(0), DataType::INT64, Sp());
    auto axis2 = std::make_shared<ConstInt>(int64_t(0), DataType::INT64, Sp());
    EXPECT_THROW((void)reg.Create("tensor.transpose", {tensor, axis1, axis2}, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
