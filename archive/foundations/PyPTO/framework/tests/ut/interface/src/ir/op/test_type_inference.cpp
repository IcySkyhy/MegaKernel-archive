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
 * \file test_type_inference.cpp
 * \brief Coverage tests for type_inference.cpp utility functions
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

static ExprPtr MakeDim(int64_t val) { return std::make_shared<ConstInt>(val, DataType::INT64, Sp()); }

// ============================================================================
// BroadcastShapes
// ============================================================================

class TypeInferenceTest : public testing::Test {};

TEST_F(TypeInferenceTest, BroadcastShapes_BothEmpty_ReturnsEmpty)
{
    auto result = BroadcastShapes({}, {});
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.shape.empty());
}

TEST_F(TypeInferenceTest, BroadcastShapes_OneEmpty_ReturnsOther)
{
    std::vector<ExprPtr> shape = {MakeDim(16), MakeDim(32)};
    auto result = BroadcastShapes(shape, {});
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.shape.size(), 2u);
}

TEST_F(TypeInferenceTest, BroadcastShapes_SameShape_ReturnsSame)
{
    std::vector<ExprPtr> shape = {MakeDim(4), MakeDim(8)};
    auto result = BroadcastShapes(shape, shape);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.shape.size(), 2u);
}

TEST_F(TypeInferenceTest, BroadcastShapes_DimOne_Broadcasts)
{
    std::vector<ExprPtr> s1 = {MakeDim(4), MakeDim(1)};
    std::vector<ExprPtr> s2 = {MakeDim(4), MakeDim(8)};
    auto result = BroadcastShapes(s1, s2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.shape.size(), 2u);
    auto last_dim = As<ConstInt>(result.shape[1]);
    ASSERT_NE(last_dim, nullptr);
    EXPECT_EQ(last_dim->value_, 8);
}

TEST_F(TypeInferenceTest, BroadcastShapes_DifferentRank_PadsWithOnes)
{
    std::vector<ExprPtr> s1 = {MakeDim(4), MakeDim(8)};
    std::vector<ExprPtr> s2 = {MakeDim(8)};
    auto result = BroadcastShapes(s1, s2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.shape.size(), 2u);
}

TEST_F(TypeInferenceTest, BroadcastShapes_Incompatible_Fails)
{
    std::vector<ExprPtr> s1 = {MakeDim(3)};
    std::vector<ExprPtr> s2 = {MakeDim(5)};
    auto result = BroadcastShapes(s1, s2);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

// ============================================================================
// PromoteDataTypes
// ============================================================================

TEST_F(TypeInferenceTest, PromoteDataTypes_SameType_ReturnsSame)
{
    auto result = PromoteDataTypes(DataType::FP32, DataType::FP32);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::FP32);
}

TEST_F(TypeInferenceTest, PromoteDataTypes_FloatOverInt)
{
    auto result = PromoteDataTypes(DataType::FP32, DataType::INT32);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::FP32);
}

TEST_F(TypeInferenceTest, PromoteDataTypes_IntOverFloat_Reversed)
{
    auto result = PromoteDataTypes(DataType::INT32, DataType::FP16);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::FP16);
}

TEST_F(TypeInferenceTest, PromoteDataTypes_LargerBitsWins)
{
    auto result = PromoteDataTypes(DataType::FP16, DataType::FP32);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::FP32);
}

TEST_F(TypeInferenceTest, PromoteDataTypes_SameSizeIntPrefersSigned)
{
    auto result = PromoteDataTypes(DataType::UINT32, DataType::INT32);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::INT32);
}

// ============================================================================
// CheckTypeCompatibility
// ============================================================================

TEST_F(TypeInferenceTest, CheckTypeCompatibility_BothScalar_True)
{
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::FP32);
    EXPECT_TRUE(CheckTypeCompatibility(t1, t2));
}

TEST_F(TypeInferenceTest, CheckTypeCompatibility_BothTensor_True)
{
    auto t1 = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeDim(16)}, DataType::FP16);
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeDim(32)}, DataType::FP32);
    EXPECT_TRUE(CheckTypeCompatibility(t1, t2));
}

TEST_F(TypeInferenceTest, CheckTypeCompatibility_BothTile_True)
{
    auto t1 = std::make_shared<TileType>(std::vector<ExprPtr>{MakeDim(16)}, DataType::FP16);
    auto t2 = std::make_shared<TileType>(std::vector<ExprPtr>{MakeDim(32)}, DataType::FP32);
    EXPECT_TRUE(CheckTypeCompatibility(t1, t2));
}

TEST_F(TypeInferenceTest, CheckTypeCompatibility_ScalarVsTensor_False)
{
    auto t1 = std::make_shared<ScalarType>(DataType::FP32);
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeDim(16)}, DataType::FP32);
    EXPECT_FALSE(CheckTypeCompatibility(t1, t2));
}

// ============================================================================
// ExtractDataType
// ============================================================================

TEST_F(TypeInferenceTest, ExtractDataType_Scalar)
{
    auto t = std::make_shared<ScalarType>(DataType::INT64);
    auto result = ExtractDataType(t);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::INT64);
}

TEST_F(TypeInferenceTest, ExtractDataType_Tensor)
{
    auto t = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeDim(8)}, DataType::FP16);
    auto result = ExtractDataType(t);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::FP16);
}

TEST_F(TypeInferenceTest, ExtractDataType_Tile)
{
    auto t = std::make_shared<TileType>(std::vector<ExprPtr>{MakeDim(32)}, DataType::FP32);
    auto result = ExtractDataType(t);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, DataType::FP32);
}

TEST_F(TypeInferenceTest, ExtractDataType_Unknown_ReturnsNullopt)
{
    auto t = GetUnknownType();
    auto result = ExtractDataType(t);
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// ExtractShape
// ============================================================================

TEST_F(TypeInferenceTest, ExtractShape_Tensor_ReturnsShape)
{
    auto t = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeDim(4), MakeDim(8)}, DataType::FP32);
    auto result = ExtractShape(t);
    EXPECT_EQ(result.size(), 2u);
}

TEST_F(TypeInferenceTest, ExtractShape_Tile_ReturnsShape)
{
    auto t = std::make_shared<TileType>(std::vector<ExprPtr>{MakeDim(16), MakeDim(32)}, DataType::FP16);
    auto result = ExtractShape(t);
    EXPECT_EQ(result.size(), 2u);
}

TEST_F(TypeInferenceTest, ExtractShape_Scalar_ReturnsEmpty)
{
    auto t = std::make_shared<ScalarType>(DataType::INT32);
    auto result = ExtractShape(t);
    EXPECT_TRUE(result.empty());
}

// ============================================================================
// GetConstantDimension, ComputeStaticShapeProduct, DimensionsEqual
// ============================================================================

TEST_F(TypeInferenceTest, GetConstantDimension_ConstInt_ReturnsValue)
{
    auto dim = MakeDim(42);
    auto result = GetConstantDimension(dim);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

TEST_F(TypeInferenceTest, GetConstantDimension_Var_ReturnsNullopt)
{
    auto dim = MakeScalarVar("n", DataType::INT64);
    auto result = GetConstantDimension(dim);
    EXPECT_FALSE(result.has_value());
}

TEST_F(TypeInferenceTest, ComputeStaticShapeProduct_AllConst)
{
    std::vector<ExprPtr> shape = {MakeDim(4), MakeDim(8), MakeDim(2)};
    auto result = ComputeStaticShapeProduct(shape);
    EXPECT_EQ(result, 64);
}

TEST_F(TypeInferenceTest, ComputeStaticShapeProduct_WithSymbolic_ReturnsMinus1)
{
    std::vector<ExprPtr> shape = {MakeDim(4), MakeScalarVar("n", DataType::INT64)};
    auto result = ComputeStaticShapeProduct(shape);
    EXPECT_EQ(result, -1);
}

TEST_F(TypeInferenceTest, DimensionsEqual_SameConst_True)
{
    auto d1 = MakeDim(16);
    auto d2 = MakeDim(16);
    EXPECT_TRUE(DimensionsEqual(d1, d2));
}

TEST_F(TypeInferenceTest, DimensionsEqual_DifferentConst_False)
{
    auto d1 = MakeDim(16);
    auto d2 = MakeDim(32);
    EXPECT_FALSE(DimensionsEqual(d1, d2));
}

TEST_F(TypeInferenceTest, DimensionsEqual_SamePtr_True)
{
    auto d = MakeDim(16);
    EXPECT_TRUE(DimensionsEqual(d, d));
}

// ============================================================================
// IsBroadcastable
// ============================================================================

TEST_F(TypeInferenceTest, IsBroadcastable_Equal_True) { EXPECT_TRUE(IsBroadcastable(MakeDim(8), MakeDim(8))); }

TEST_F(TypeInferenceTest, IsBroadcastable_SourceOne_True) { EXPECT_TRUE(IsBroadcastable(MakeDim(1), MakeDim(64))); }

TEST_F(TypeInferenceTest, IsBroadcastable_TargetOne_True) { EXPECT_TRUE(IsBroadcastable(MakeDim(64), MakeDim(1))); }

TEST_F(TypeInferenceTest, IsBroadcastable_Incompatible_False) { EXPECT_FALSE(IsBroadcastable(MakeDim(3), MakeDim(5))); }

// ============================================================================
// RequireTileScalarArgs
// ============================================================================

TEST_F(TypeInferenceTest, RequireTileScalarArgs_Valid_ReturnsTypes)
{
    auto tile = MakeTileVar("t", {16, 32}, DataType::FP16);
    auto scalar = MakeScalarVar("s", DataType::FP16);
    auto result = RequireTileScalarArgs({tile, scalar}, "test_op", 2);
    EXPECT_NE(result.tile_type, nullptr);
    EXPECT_NE(result.scalar_type, nullptr);
    EXPECT_EQ(result.tile_type->dtype_, DataType::FP16);
}

TEST_F(TypeInferenceTest, RequireTileScalarArgs_WrongArgCount_Throws)
{
    auto tile = MakeTileVar("t", {16}, DataType::FP16);
    EXPECT_THROW((void)RequireTileScalarArgs({tile}, "test_op", 2), npu::tile_fwk::Error);
}

TEST_F(TypeInferenceTest, RequireTileScalarArgs_NonTileFirst_Throws)
{
    auto s1 = MakeScalarVar("a", DataType::FP32);
    auto s2 = MakeScalarVar("b", DataType::FP32);
    EXPECT_THROW((void)RequireTileScalarArgs({s1, s2}, "test_op", 2), npu::tile_fwk::Error);
}

TEST_F(TypeInferenceTest, RequireTileScalarArgs_NonScalarSecond_Throws)
{
    auto tile1 = MakeTileVar("t1", {16}, DataType::FP16);
    auto tile2 = MakeTileVar("t2", {16}, DataType::FP16);
    EXPECT_THROW((void)RequireTileScalarArgs({tile1, tile2}, "test_op", 2), npu::tile_fwk::Error);
}

// ============================================================================
// DeduceBlockOutTileType
// ============================================================================

TEST_F(TypeInferenceTest, DeduceBlockOutTileType_Valid_ReturnsFirstTile)
{
    auto tile1 = MakeTileVar("a", {16, 32}, DataType::FP16);
    auto tile2 = MakeTileVar("b", {16, 32}, DataType::FP16);
    auto out = MakeTileVar("out", {16, 32}, DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs;
    auto result = DeduceBlockOutTileType({out, tile1, tile2}, kwargs, "test_op", 3);
    auto rt = As<TileType>(result);
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(TypeInferenceTest, DeduceBlockOutTileType_WrongArgCount_Throws)
{
    auto tile = MakeTileVar("a", {16}, DataType::FP16);
    std::vector<std::pair<std::string, std::any>> kwargs;
    EXPECT_THROW((void)DeduceBlockOutTileType({tile}, kwargs, "test_op", 2), npu::tile_fwk::Error);
}

TEST_F(TypeInferenceTest, DeduceBlockOutTileType_FirstNotTile_Throws)
{
    auto tile = MakeTileVar("a", {16}, DataType::FP16);
    auto scalar = MakeScalarVar("s", DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs;
    EXPECT_THROW((void)DeduceBlockOutTileType({scalar, tile}, kwargs, "test_op", 2), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
