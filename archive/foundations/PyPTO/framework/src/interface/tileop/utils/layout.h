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
 * \file layout.h
 * \brief
 */

#ifndef TILEOP_UTILS_LAYOUT_H
#define TILEOP_UTILS_LAYOUT_H

#include "tuple.h"
#include "../tileop_common.h"

constexpr size_t DIM_1ST = 0;
constexpr size_t DIM_2ND = 1;
constexpr size_t DIM_3RD = 2;
constexpr size_t DIM_4TH = 3;
constexpr size_t DIM_5TH = 4;
constexpr size_t DIM_6TH = 5;
constexpr size_t MAX_DIMS = 5;

namespace TileOp {
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord, const ShapeType& shape, const StrideType& stride);

template <typename... Shapes>
using Shape = Std::tuple<Shapes...>;

template <typename... Strides>
using Stride = Std::tuple<Strides...>;

template <typename... TileShapes>
using TileShape = Std::tuple<TileShapes...>;

template <typename... Coords>
using Coord = Std::tuple<Coords...>;

template <typename... LastUses>
using LastUse = Std::tuple<LastUses...>;

template <typename Tuple>
constexpr bool IsConstantTuple = Std::IsIntegralConstantV<typename Std::tuple_element<0, Tuple>::type>;

template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr TileShape<Ts...> MakeTileShape(const Ts&... t)
{
    return {t...};
}

template <typename Tuple, size_t index, size_t expect_size = Std::tuple_size<Tuple>::value, size_t default_value = 1>
__aicore__ inline constexpr size_t GetTupleElement()
{
    static_assert(index < expect_size, "The index of tuple is out of range.");
    constexpr auto size = Std::tuple_size<Tuple>::value;
    if constexpr (size < expect_size && index < (expect_size - size)) {
        return default_value;
    } else {
        return Std::tuple_element<index + size - expect_size, Tuple>::type::value;
    }
}

template <typename Tuple, size_t index, size_t expect_size = Std::tuple_size<Tuple>::value, size_t default_value = 1>
__aicore__ inline constexpr size_t GetTupleElement(const Tuple& t)
{
    static_assert(index < expect_size, "The index of tuple is out of range.");
    constexpr auto size = Std::tuple_size<Tuple>::value;
    if constexpr (size < expect_size && index < (expect_size - size)) {
        return default_value;
    } else {
        return Std::get<index + size - expect_size>(t);
    }
}

template <typename ShapeType, typename StrideType, typename TileShapeType>
struct Layout : private Std::tuple<ShapeType, StrideType, TileShapeType> {
    using Shape = ShapeType;
    using Stride = StrideType;
    using TileShape = TileShapeType;
    __aicore__ inline constexpr Layout(const ShapeType& shape = {}, const StrideType& stride = {},
                                       const TileShapeType& tileShape = {})
        : Std::tuple<ShapeType, StrideType, TileShapeType>(shape, stride, tileShape)
    {
        static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType> && Std::is_tuple_v<TileShapeType>,
                      "Shape, Stride or TileShape is not tuple!");
    }

    __aicore__ inline constexpr decltype(auto) layout() { return *this; }

    __aicore__ inline constexpr decltype(auto) layout() const { return *this; }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape()
    {
        return GetValue<0, I...>(static_cast<Std::tuple<ShapeType, StrideType, TileShapeType>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape() const
    {
        return GetValue<0, I...>(static_cast<const Std::tuple<ShapeType, StrideType, TileShapeType>&>(*this));
    }

    template <size_t index, size_t expect_size = Std::tuple_size<ShapeType>::value>
    __aicore__ inline constexpr decltype(auto) GetShapeDim() const
    {
        if constexpr (IsConstantTuple<ShapeType> == true) {
            return GetTupleElement<ShapeType, index, expect_size, 1>();
        } else {
            return GetTupleElement<ShapeType, index, expect_size, 1>(GetShape());
        }
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride()
    {
        return GetValue<1, I...>(static_cast<Std::tuple<ShapeType, StrideType, TileShapeType>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride() const
    {
        return GetValue<1, I...>(static_cast<const Std::tuple<ShapeType, StrideType, TileShapeType>&>(*this));
    }

    template <size_t index, size_t expect_size = Std::tuple_size<StrideType>::value>
    __aicore__ inline constexpr decltype(auto) GetStrideDim() const
    {
        if constexpr (IsConstantTuple<StrideType> == true) {
            return GetTupleElement<StrideType, index, expect_size, 0>();
        } else {
            return GetTupleElement<StrideType, index, expect_size, 0>(GetStride());
        }
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetTileShape()
    {
        return GetValue<2, I...>(static_cast<Std::tuple<ShapeType, StrideType, TileShapeType>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetTileShape() const
    {
        return GetValue<2, I...>(static_cast<const Std::tuple<ShapeType, StrideType, TileShapeType>&>(*this));
    }

    template <size_t index, size_t expect_size = Std::tuple_size<TileShapeType>::value>
    __aicore__ inline constexpr decltype(auto) GetTileShapeDim() const
    {
        if constexpr (IsConstantTuple<TileShapeType> == true) {
            return GetTupleElement<TileShapeType, index, expect_size, 0>();
        } else {
            return GetTupleElement<TileShapeType, index, expect_size, 0>(GetTileShape());
        }
    }

    __aicore__ inline static constexpr auto IsStaticLayout() { return IsConstantTuple<ShapeType> == true; }

    template <typename CoordType>
    __aicore__ inline constexpr auto operator()(const CoordType& coord) const
    {
        return Crd2Idx(coord, GetShape(), GetStride());
    }

    template <typename Tuple, size_t expect_size = Std::tuple_size<Tuple>::value>
    __aicore__ inline constexpr decltype(auto) GetGmOffset(const Tuple& coordinate) const
    {
        auto s0 = GetStrideDim<DIM_1ST, expect_size>();
        auto s1 = GetStrideDim<DIM_2ND, expect_size>();
        auto s2 = GetStrideDim<DIM_3RD, expect_size>();
        auto s3 = GetStrideDim<DIM_4TH, expect_size>();
        auto s4 = GetStrideDim<DIM_5TH, expect_size>();
        auto c0 = GetTupleElement<Tuple, DIM_1ST, expect_size, 0>(coordinate);
        auto c1 = GetTupleElement<Tuple, DIM_2ND, expect_size, 0>(coordinate);
        auto c2 = GetTupleElement<Tuple, DIM_3RD, expect_size, 0>(coordinate);
        auto c3 = GetTupleElement<Tuple, DIM_4TH, expect_size, 0>(coordinate);
        auto c4 = GetTupleElement<Tuple, DIM_5TH, expect_size, 0>(coordinate);
        return s4 * c4 + s3 * c3 + s2 * c2 + s1 * c1 + s0 * c0;
    }

private:
    template <size_t index, size_t I, size_t... Is, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
    {
        auto tupleEle = Std::get<index>(t);
        return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
    }

    template <size_t index, size_t I, size_t... Is, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t) const
    {
        auto tupleEle = Std::get<index>(t);
        return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
    }

    template <size_t index, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
    {
        return Std::get<index>(t);
    }

    template <size_t index, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t) const
    {
        return Std::get<index>(t);
    }
};

template <typename ShapeType, typename StrideType, typename TileShapeType>
__aicore__ inline constexpr auto MakeLayout(const ShapeType& shape, const StrideType& stride,
                                            const TileShapeType& tileShape)
{
    return Layout<ShapeType, StrideType, TileShapeType>(shape, stride, tileShape);
}

template <typename T>
struct is_layout : Std::false_type {};

template <typename ShapeType, typename StrideType, typename TileShapeType>
struct is_layout<Layout<ShapeType, StrideType, TileShapeType>> : Std::true_type {};

template <typename T>
constexpr bool is_layout_v = is_layout<T>::value;

template <typename StrideType>
__aicore__ inline constexpr auto GetOuterStride()
{
    return Std::tuple_element<0, StrideType>::type::value;
}

template <typename T, size_t index, size_t expect_size = Std::tuple_size<typename T::Shape>::value>
__aicore__ inline constexpr size_t GetTensorShapeDim()
{
    return GetTupleElement<typename T::Shape, index, expect_size, 1>();
}

template <typename T, size_t index, size_t expect_size = Std::tuple_size<typename T::Stride>::value>
__aicore__ inline constexpr size_t GetTensorStrideDim()
{
    return GetTupleElement<typename T::Stride, index, expect_size, 0>();
}

template <typename T, size_t index, size_t expect_size = Std::tuple_size<typename T::TileShape>::value>
__aicore__ inline constexpr size_t GetTensorTileShapeDim()
{
    return GetTupleElement<typename T::TileShape, index, expect_size, 1>();
}

template <int leftAxis, int rightAxis, typename Shape>
__aicore__ inline constexpr size_t GetAnyAxisMergeResult()
{
    constexpr auto n0 = []() constexpr {
        if constexpr (leftAxis <= 1 && 1 <= rightAxis) {
            return Std::tuple_element<DIM_1ST, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n1 = []() constexpr {
        if constexpr (leftAxis <= 2 && 2 <= rightAxis) {
            return Std::tuple_element<DIM_2ND, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n2 = []() constexpr {
        if constexpr (leftAxis <= 3 && 3 <= rightAxis) {
            return Std::tuple_element<DIM_3RD, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n3 = []() constexpr {
        if constexpr (leftAxis <= 4 && 4 <= rightAxis) {
            return Std::tuple_element<DIM_4TH, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n4 = []() constexpr {
        if constexpr (leftAxis <= 5 && 5 <= rightAxis) {
            return Std::tuple_element<DIM_5TH, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    return n0 * n1 * n2 * n3 * n4;
}

template <size_t shapeSize, typename Shape>
__aicore__ inline constexpr size_t GetNonFirstAxisMergeResult()
{
    return GetAnyAxisMergeResult<2, shapeSize, Shape>();
}

template <size_t shapeSize, typename Shape>
__aicore__ inline constexpr size_t GetOutterAxisMergeResult()
{
    constexpr size_t expectSize = 5;
    constexpr auto n0 = GetTupleElement<Shape, DIM_1ST, expectSize, 1>();
    constexpr auto n1 = GetTupleElement<Shape, DIM_2ND, expectSize, 1>();
    constexpr auto n2 = GetTupleElement<Shape, DIM_3RD, expectSize, 1>();
    constexpr auto n3 = GetTupleElement<Shape, DIM_4TH, expectSize, 1>();
    return n0 * n1 * n2 * n3;
}

template <size_t shapeSize, typename Shape>
__aicore__ inline constexpr size_t GetAllAxisProduct()
{
    constexpr size_t expectSize = 5;
    constexpr auto n0 = GetTupleElement<Shape, DIM_1ST, expectSize, 1>();
    constexpr auto n1 = GetTupleElement<Shape, DIM_2ND, expectSize, 1>();
    constexpr auto n2 = GetTupleElement<Shape, DIM_3RD, expectSize, 1>();
    constexpr auto n3 = GetTupleElement<Shape, DIM_4TH, expectSize, 1>();
    constexpr auto n4 = GetTupleElement<Shape, DIM_5TH, expectSize, 1>();
    return n0 * n1 * n2 * n3 * n4;
}

template <int... BrcOperands>
__aicore__ inline constexpr bool HasBrcOperand()
{
    if constexpr (sizeof...(BrcOperands) == 0) {
        return false;
    }
    return ((BrcOperands != 0) || ...);
}

template <typename T0>
__aicore__ inline constexpr bool JudgeValidShapeEqualTileShape()
{
    if constexpr (T0::IsStaticLayout()) {
        constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
        if constexpr (shapeSize == 1) {
            return true;
        }
        constexpr auto outterStride = GetOuterStride<typename T0::Stride>();
        constexpr auto nonFirstAxis = GetNonFirstAxisMergeResult<shapeSize, typename T0::Shape>();
        if constexpr (outterStride == nonFirstAxis) {
            return true;
        }
        return false;
    }
    return false;
}

template <typename T0>
__aicore__ constexpr bool IsConstContinous()
{
    return JudgeValidShapeEqualTileShape<T0>();
}

template <typename T0, typename T1, typename... Args>
__aicore__ constexpr bool IsConstContinous()
{
    if constexpr (!JudgeValidShapeEqualTileShape<T0>()) {
        return false;
    }
    return IsConstContinous<T1, Args...>();
}
} // namespace TileOp

// common shape
using Shape1Dim = TileOp::Shape<size_t>;
using Shape2Dim = TileOp::Shape<size_t, size_t>;
using Shape3Dim = TileOp::Shape<size_t, size_t, size_t>;
using Shape4Dim = TileOp::Shape<size_t, size_t, size_t, size_t>;
using Shape5Dim = TileOp::Shape<size_t, size_t, size_t, size_t, size_t>;
using Shape6Dim = TileOp::Shape<size_t, size_t, size_t, size_t, size_t, size_t>;

// common stride
using Stride1Dim = TileOp::Stride<size_t>;
using Stride2Dim = TileOp::Stride<size_t, size_t>;
using Stride3Dim = TileOp::Stride<size_t, size_t, size_t>;
using Stride4Dim = TileOp::Stride<size_t, size_t, size_t, size_t>;
using Stride5Dim = TileOp::Stride<size_t, size_t, size_t, size_t, size_t>;
using Stride6Dim = TileOp::Stride<size_t, size_t, size_t, size_t, size_t, size_t>;

// common coord
using Coord1Dim = TileOp::Coord<size_t>;
using Coord2Dim = TileOp::Coord<size_t, size_t>;
using Coord3Dim = TileOp::Coord<size_t, size_t, size_t>;
using Coord4Dim = TileOp::Coord<size_t, size_t, size_t, size_t>;
using Coord5Dim = TileOp::Coord<size_t, size_t, size_t, size_t, size_t>;
using Coord6Dim = TileOp::Coord<size_t, size_t, size_t, size_t, size_t, size_t>;

// common dynamic layouts
using DynLayout1Dim = TileOp::Layout<Shape1Dim, Stride1Dim, TileOp::TileShape<size_t>>;
using DynLayout2Dim = TileOp::Layout<Shape2Dim, Stride2Dim, TileOp::TileShape<size_t, size_t>>;
using DynLayout3Dim = TileOp::Layout<Shape3Dim, Stride3Dim, TileOp::TileShape<size_t, size_t, size_t>>;
using DynLayout4Dim = TileOp::Layout<Shape4Dim, Stride4Dim, TileOp::TileShape<size_t, size_t, size_t, size_t>>;
using DynLayout5Dim = TileOp::Layout<Shape5Dim, Stride5Dim, TileOp::TileShape<size_t, size_t, size_t, size_t, size_t>>;
using DynLayout6Dim = TileOp::Layout<Shape6Dim, Stride6Dim,
                                     TileOp::TileShape<size_t, size_t, size_t, size_t, size_t, size_t>>;

// common lastuse
template <size_t TileW = 0>
using LastUse1Dim = TileOp::LastUse<Std::Int<TileW>>;

template <size_t TileH = 0, size_t TileW = 0>
using LastUse2Dim = TileOp::LastUse<Std::Int<TileH>, Std::Int<TileW>>;

template <size_t TileD = 0, size_t TileH = 0, size_t TileW = 0>
using LastUse3Dim = TileOp::LastUse<Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>;

template <size_t TileN = 0, size_t TileD = 0, size_t TileH = 0, size_t TileW = 0>
using LastUse4Dim = TileOp::LastUse<Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>;

template <size_t TileS = 0, size_t TileN = 0, size_t TileD = 0, size_t TileH = 0, size_t TileW = 0>
using LastUse5Dim = TileOp::LastUse<Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>,
                                    Std::Int<TileW>>;

template <size_t TileB = 0, size_t TileS = 0, size_t TileN = 0, size_t TileD = 0, size_t TileH = 0, size_t TileW = 0>
using LastUse6Dim = TileOp::LastUse<Std::Int<TileB>, Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>,
                                    Std::Int<TileW>>;

// common Local layouts
template <size_t TileW>
using LocalLayout1Dim = TileOp::Layout<Shape1Dim, TileOp::Stride<Std::Int<1>>, TileOp::TileShape<Std::Int<TileW>>>;

template <size_t TileH, size_t TileW>
using LocalLayout2Dim = TileOp::Layout<Shape2Dim, TileOp::Stride<Std::Int<TileW>, Std::Int<1>>,
                                       TileOp::TileShape<Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t TileD, size_t TileH, size_t TileW>
using LocalLayout3Dim = TileOp::Layout<Shape3Dim, TileOp::Stride<Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
                                       TileOp::TileShape<Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t TileN, size_t TileD, size_t TileH, size_t TileW>
using LocalLayout4Dim = TileOp::Layout<
    Shape4Dim, TileOp::Stride<Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t TileS, size_t TileN, size_t TileD, size_t TileH, size_t TileW>
using LocalLayout5Dim = TileOp::Layout<
    Shape5Dim,
    TileOp::Stride<Std::Int<TileN * TileD * TileH * TileW>, Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>,
                   Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t TileB, size_t TileS, size_t TileN, size_t TileD, size_t TileH, size_t TileW>
using LocalLayout6Dim = TileOp::Layout<
    Shape6Dim,
    TileOp::Stride<Std::Int<TileS * TileN * TileD * TileH * TileW>, Std::Int<TileN * TileD * TileH * TileW>,
                   Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileB>, Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>,
                      Std::Int<TileW>>>;

// common static layouts
template <size_t W, size_t TileW>
using StaticLayout1Dim = TileOp::Layout<TileOp::Shape<Std::Int<W>>, TileOp::Stride<Std::Int<1>>,
                                        TileOp::TileShape<Std::Int<TileW>>>;

template <size_t H, size_t W, size_t TileH, size_t TileW>
using StaticLayout2Dim = TileOp::Layout<TileOp::Shape<Std::Int<H>, Std::Int<W>>,
                                        TileOp::Stride<Std::Int<TileW>, Std::Int<1>>,
                                        TileOp::TileShape<Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t D, size_t H, size_t W, size_t TileD, size_t TileH, size_t TileW>
using StaticLayout3Dim = TileOp::Layout<TileOp::Shape<Std::Int<D>, Std::Int<H>, Std::Int<W>>,
                                        TileOp::Stride<Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
                                        TileOp::TileShape<Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t N, size_t D, size_t H, size_t W, size_t TileN, size_t TileD, size_t TileH, size_t TileW>
using StaticLayout4Dim = TileOp::Layout<
    TileOp::Shape<Std::Int<N>, Std::Int<D>, Std::Int<H>, Std::Int<W>>,
    TileOp::Stride<Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t S, size_t N, size_t D, size_t H, size_t W, size_t TileS, size_t TileN, size_t TileD, size_t TileH,
          size_t TileW>
using StaticLayout5Dim = TileOp::Layout<
    TileOp::Shape<Std::Int<S>, Std::Int<N>, Std::Int<D>, Std::Int<H>, Std::Int<W>>,
    TileOp::Stride<Std::Int<TileN * TileD * TileH * TileW>, Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>,
                   Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t B, size_t S, size_t N, size_t D, size_t H, size_t W, size_t TileB, size_t TileS, size_t TileN,
          size_t TileD, size_t TileH, size_t TileW>
using StaticLayout6Dim = TileOp::Layout<
    TileOp::Shape<Std::Int<B>, Std::Int<S>, Std::Int<N>, Std::Int<D>, Std::Int<H>, Std::Int<W>>,
    TileOp::Stride<Std::Int<TileS * TileN * TileD * TileH * TileW>, Std::Int<TileN * TileD * TileH * TileW>,
                   Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileB>, Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>,
                      Std::Int<TileW>>>;

// used to loop dim of tensor
#ifdef __DAV_V310
using LoopVar = uint16_t;
#else // A2 A3
using LoopVar = size_t;
#endif
#endif // TILEOP_UTILS_LAYOUT_H
