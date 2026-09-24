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
 * \file test_type.cpp
 * \brief Unit tests for IR type system
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/scalar_expr.h"
#include "ir/memref.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// Type Base Class Tests
// ============================================================================

TEST(IRTypeTest, TestTypeBasic)
{
    // Test base Type class via UnknownType (Type is abstract with pure virtual GetKind)
    auto type = std::make_shared<UnknownType>();
    ASSERT_NE(type, nullptr);
    ASSERT_EQ(type->TypeName(), "UnknownType");
}

// ============================================================================
// UnknownType Tests
// ============================================================================

TEST(IRTypeTest, TestUnknownTypeConstructor)
{
    // Test UnknownType construction
    auto unknownType = std::make_shared<UnknownType>();
    ASSERT_NE(unknownType, nullptr);
    ASSERT_EQ(unknownType->TypeName(), "UnknownType");
}

TEST(IRTypeTest, TestGetUnknownTypeSingleton)
{
    // Test GetUnknownType returns singleton
    auto type1 = GetUnknownType();
    auto type2 = GetUnknownType();
    ASSERT_EQ(type1, type2); // Should be same instance
    ASSERT_EQ(type1->TypeName(), "UnknownType");
}

// ============================================================================
// ScalarType Tests
// ============================================================================

TEST(IRTypeTest, TestScalarTypeInt32)
{
    // Test ScalarType with INT32
    auto scalarType = std::make_shared<ScalarType>(DataType::INT32);
    ASSERT_NE(scalarType, nullptr);
    ASSERT_EQ(scalarType->TypeName(), "ScalarType");
    ASSERT_EQ(scalarType->dtype_, DataType::INT32);
}

TEST(IRTypeTest, TestScalarTypeFloat32)
{
    // Test ScalarType with FLOAT32
    auto scalarType = std::make_shared<ScalarType>(DataType::FP32);
    ASSERT_NE(scalarType, nullptr);
    ASSERT_EQ(scalarType->dtype_, DataType::FP32);
}

TEST(IRTypeTest, TestScalarTypeBool)
{
    // Test ScalarType with BOOL
    auto scalarType = std::make_shared<ScalarType>(DataType::BOOL);
    ASSERT_NE(scalarType, nullptr);
    ASSERT_EQ(scalarType->dtype_, DataType::BOOL);
}

TEST(IRTypeTest, TestScalarTypeVariousDtypes)
{
    // Test ScalarType with various data types
    std::vector<DataType> dtypes = {DataType::INT8,  DataType::INT16,  DataType::INT32,  DataType::INT64,
                                    DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64,
                                    DataType::FP16,  DataType::FP32,   DataType::BOOL};

    for (const auto& dtype : dtypes) {
        auto scalarType = std::make_shared<ScalarType>(dtype);
        ASSERT_NE(scalarType, nullptr);
        ASSERT_EQ(scalarType->dtype_, dtype);
    }
}

// ============================================================================
// TileView Tests
// ============================================================================

TEST(IRTypeTest, TestTileViewDefaultConstructor)
{
    // Test TileView default construction
    TileView tileView;
    ASSERT_TRUE(tileView.validShape.empty());
    ASSERT_TRUE(tileView.stride.empty());
    ASSERT_EQ(tileView.startOffset, nullptr);
}

TEST(IRTypeTest, TestTileViewWithParameters)
{
    // Test TileView with parameters
    auto const1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    auto const2 = std::make_shared<ConstInt>(32, DataType::INT32, Span::Unknown());
    auto const3 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto offset = std::make_shared<ConstInt>(0, DataType::INT32, Span::Unknown());

    std::vector<ExprPtr> validShape = {const1, const2};
    std::vector<ExprPtr> stride = {const3, const3};

    TileView tileView(validShape, stride, offset);
    ASSERT_EQ(tileView.validShape.size(), 2);
    ASSERT_EQ(tileView.stride.size(), 2);
    ASSERT_NE(tileView.startOffset, nullptr);
}

// ============================================================================
// ShapedType Tests
// ============================================================================

TEST(IRTypeTest, TestShapedTypeWithoutMemRef)
{
    // Test ShapedType without memory reference
    auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2};

    auto shapedType = std::make_shared<ShapedType>(DataType::FP32, shape);
    ASSERT_NE(shapedType, nullptr);
    ASSERT_EQ(shapedType->TypeName(), "ShapedType");
    ASSERT_EQ(shapedType->dtype_, DataType::FP32);
    ASSERT_EQ(shapedType->shape_.size(), 2);
    ASSERT_FALSE(shapedType->memref_.has_value());
}

TEST(IRTypeTest, TestShapedTypeWithMemRef)
{
    // Test ShapedType with memory reference
    auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1};

    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::Vec, addr, 1024);

    auto shapedType = std::make_shared<ShapedType>(DataType::INT32, shape, memref);
    ASSERT_NE(shapedType, nullptr);
    ASSERT_TRUE(shapedType->memref_.has_value());
    ASSERT_EQ((*shapedType->memref_)->memorySpace_, MemorySpace::Vec);
    ASSERT_EQ((*shapedType->memref_)->size_, 1024);
}

TEST(IRTypeTest, TestShapedTypeWithVecInt)
{
    // Test ShapedType with VecInt
    std::vector<int64_t> shape = {10, 10};

    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::Vec, addr, 1024);

    auto shapedType = std::make_shared<ShapedType>(DataType::INT32, shape, memref);
    ASSERT_NE(shapedType, nullptr);
    ASSERT_TRUE(shapedType->memref_.has_value());
    ASSERT_EQ((*shapedType->memref_)->memorySpace_, MemorySpace::Vec);
    ASSERT_EQ((*shapedType->memref_)->size_, 1024);
}

// ============================================================================
// TensorType Tests
// ============================================================================

TEST(IRTypeTest, TestTensorTypeBasic)
{
    // Test basic TensorType construction
    auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2};

    auto tensorType = std::make_shared<TensorType>(shape, DataType::FP32);
    ASSERT_NE(tensorType, nullptr);
    ASSERT_EQ(tensorType->TypeName(), "TensorType");
    ASSERT_EQ(tensorType->dtype_, DataType::FP32);
    ASSERT_EQ(tensorType->shape_.size(), 2);
}

TEST(IRTypeTest, TestTensorType1D)
{
    // Test 1D TensorType
    auto dim = std::make_shared<ConstInt>(100, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim};

    auto tensorType = std::make_shared<TensorType>(shape, DataType::INT32);
    ASSERT_NE(tensorType, nullptr);
    ASSERT_EQ(tensorType->shape_.size(), 1);
}

TEST(IRTypeTest, TestTensorType3D)
{
    // Test 3D TensorType
    auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::Unknown());
    auto dim3 = std::make_shared<ConstInt>(30, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2, dim3};

    auto tensorType = std::make_shared<TensorType>(shape, DataType::FP16);
    ASSERT_NE(tensorType, nullptr);
    ASSERT_EQ(tensorType->shape_.size(), 3);
    ASSERT_EQ(tensorType->dtype_, DataType::FP16);
}

TEST(IRTypeTest, TestTensorTypeWithMemRef)
{
    // Test TensorType with memory reference
    auto dim = std::make_shared<ConstInt>(100, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim};

    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::DDR, addr, 400);

    auto tensorType = std::make_shared<TensorType>(shape, DataType::INT32, memref);
    ASSERT_NE(tensorType, nullptr);
    ASSERT_TRUE(tensorType->memref_.has_value());
    ASSERT_EQ((*tensorType->memref_)->memorySpace_, MemorySpace::DDR);
}

// ============================================================================
// TileType Tests
// ============================================================================

TEST(IRTypeTest, TestTileType1D)
{
    // Test 1D TileType
    auto dim = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim};

    auto tileType = std::make_shared<TileType>(shape, DataType::FP32);
    ASSERT_NE(tileType, nullptr);
    ASSERT_EQ(tileType->TypeName(), "TileType");
    ASSERT_EQ(tileType->shape_.size(), 1);
    ASSERT_FALSE(tileType->tileView_.has_value());
}

TEST(IRTypeTest, TestTileType2D)
{
    // Test 2D TileType (maximum allowed dimensions)
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2};

    auto tileType = std::make_shared<TileType>(shape, DataType::FP16);
    ASSERT_NE(tileType, nullptr);
    ASSERT_EQ(tileType->shape_.size(), 2);
    ASSERT_EQ(tileType->dtype_, DataType::FP16);
}

TEST(IRTypeTest, TestTileTypeWithMemRef)
{
    // Test TileType with memory reference
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2};

    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::Left, addr, 512);

    auto tileType = std::make_shared<TileType>(shape, DataType::FP32, memref);
    ASSERT_NE(tileType, nullptr);
    ASSERT_TRUE(tileType->memref_.has_value());
    ASSERT_EQ((*tileType->memref_)->memorySpace_, MemorySpace::Left);
}

TEST(IRTypeTest, TestTileTypeWithTileView)
{
    // Test TileType with tile view
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2};

    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRefPtr memref = std::make_shared<MemRef>(MemorySpace::Acc, addr, 512);

    auto validDim1 = std::make_shared<ConstInt>(8, DataType::INT32, Span::Unknown());
    auto validDim2 = std::make_shared<ConstInt>(8, DataType::INT32, Span::Unknown());
    auto stride1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto stride2 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto offset = std::make_shared<ConstInt>(0, DataType::INT32, Span::Unknown());

    TileView tileView({validDim1, validDim2}, {stride1, stride2}, offset);

    auto tileType = std::make_shared<TileType>(shape, DataType::FP32, memref, tileView);
    ASSERT_NE(tileType, nullptr);
    ASSERT_TRUE(tileType->tileView_.has_value());
    ASSERT_EQ(tileType->tileView_->validShape.size(), 2);
    ASSERT_EQ(tileType->tileView_->stride.size(), 2);
}

TEST(IRTypeTest, TestTileTypeInvalidDimensions)
{
    // Test that TileType with more than 2 dimensions can be created
    // (dimension limit is now enforced at code generation level, not type level)
    auto dim1 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    auto dim3 = std::make_shared<ConstInt>(16, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2, dim3};

    auto tileType = std::make_shared<TileType>(shape, DataType::FP32);
    ASSERT_NE(tileType, nullptr);
    ASSERT_EQ(tileType->shape_.size(), 3);
}

// ============================================================================
// TupleType Tests
// ============================================================================

TEST(IRTypeTest, TestTupleTypeEmpty)
{
    // Test empty TupleType
    std::vector<TypePtr> types;
    auto tupleType = std::make_shared<TupleType>(types);
    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->TypeName(), "TupleType");
    ASSERT_TRUE(tupleType->types_.empty());
}

TEST(IRTypeTest, TestTupleTypeSingleElement)
{
    // Test TupleType with single element
    auto scalarType = std::make_shared<ScalarType>(DataType::INT32);
    std::vector<TypePtr> types = {scalarType};

    auto tupleType = std::make_shared<TupleType>(types);
    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->types_.size(), 1);
}

TEST(IRTypeTest, TestTupleTypeMultipleElements)
{
    // Test TupleType with multiple elements
    auto scalarType = std::make_shared<ScalarType>(DataType::INT32);
    auto dim = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim};
    auto tensorType = std::make_shared<TensorType>(shape, DataType::FP32);

    std::vector<TypePtr> types = {scalarType, tensorType};
    auto tupleType = std::make_shared<TupleType>(types);
    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->types_.size(), 2);
}

TEST(IRTypeTest, TestTupleTypeNested)
{
    // Test nested TupleType
    auto scalarType1 = std::make_shared<ScalarType>(DataType::INT32);
    auto scalarType2 = std::make_shared<ScalarType>(DataType::FP32);

    std::vector<TypePtr> innerTypes = {scalarType1, scalarType2};
    auto innerTuple = std::make_shared<TupleType>(innerTypes);

    std::vector<TypePtr> outerTypes = {innerTuple, scalarType1};
    auto outerTuple = std::make_shared<TupleType>(outerTypes);

    ASSERT_NE(outerTuple, nullptr);
    ASSERT_EQ(outerTuple->types_.size(), 2);
}

TEST(IRTypeTest, TestTupleTypeWithTensorAndScalar)
{
    // Test TupleType with mixed tensor and scalar types
    auto scalarType = std::make_shared<ScalarType>(DataType::BOOL);

    auto dim1 = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    auto dim2 = std::make_shared<ConstInt>(20, DataType::INT32, Span::Unknown());
    std::vector<ExprPtr> shape = {dim1, dim2};
    auto tensorType = std::make_shared<TensorType>(shape, DataType::FP32);

    std::vector<TypePtr> types = {tensorType, scalarType};
    auto tupleType = std::make_shared<TupleType>(types);

    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->types_.size(), 2);
    ASSERT_EQ(tupleType->types_[0]->TypeName(), "TensorType");
    ASSERT_EQ(tupleType->types_[1]->TypeName(), "ScalarType");
}

// ============================================================================
// Type Polymorphism Tests
// ============================================================================

TEST(IRTypeTest, TestTypePolymorphism)
{
    // Test that derived types can be used as base Type pointers
    TypePtr type1 = std::make_shared<UnknownType>();
    TypePtr type2 = std::make_shared<ScalarType>(DataType::INT32);
    TypePtr type3 = std::make_shared<TupleType>(std::vector<TypePtr>{});

    ASSERT_EQ(type1->TypeName(), "UnknownType");
    ASSERT_EQ(type2->TypeName(), "ScalarType");
    ASSERT_EQ(type3->TypeName(), "TupleType");
}

TEST(IRTypeTest, TestTypeDynamicCast)
{
    // Test dynamic casting of types
    TypePtr baseType = std::make_shared<ScalarType>(DataType::FP32);

    auto scalarType = std::dynamic_pointer_cast<const ScalarType>(baseType);
    ASSERT_NE(scalarType, nullptr);
    ASSERT_EQ(scalarType->dtype_, DataType::FP32);

    auto tensorType = std::dynamic_pointer_cast<const TensorType>(baseType);
    ASSERT_EQ(tensorType, nullptr); // Should fail
}

} // namespace ir
} // namespace pypto
