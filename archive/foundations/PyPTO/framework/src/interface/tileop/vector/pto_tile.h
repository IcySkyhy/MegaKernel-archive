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
 * \file pto_tile.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_PTO_TILE__H
#define TILEOP_TILE_OPERATOR_PTO_TILE__H
#include <cstddef>
#include <type_traits>

#include "utils/layout.h"
#include "utils/tile_tensor.h"

#ifdef __DAV_V220
#define PTO_WITH_LAST_USE(OP, ...) OP
#else
#define PTO_WITH_LAST_USE(OP, ...) [[pto::last_use(__VA_ARGS__)]] OP
#endif

template <typename Tuple, size_t index, size_t default_value = 1, bool use_default = false>
__aicore__ inline constexpr size_t GetTupleElementWithDefaultOverride(const Tuple& t)
{
    static_assert(index < MAX_DIMS, "The index of tuple is out of range.");
    constexpr auto size = Std::tuple_size<Tuple>::value;
    if constexpr (use_default || (size < MAX_DIMS && index < (MAX_DIMS - size))) {
        return default_value;
    } else {
        return Std::get<index + size - MAX_DIMS>(t);
    }
}

namespace TileOp {
template <typename DType>
inline constexpr bool IsPackedFp4Type =
#if defined PTO_NPU_ARCH_A5
    std::is_same_v<DType, float4_e1m2x2_t> || std::is_same_v<DType, float4_e2m1x2_t> ||
    std::is_same_v<DType, __gm__ float4_e1m2x2_t> || std::is_same_v<DType, __gm__ float4_e2m1x2_t>;
#else
    false;
#endif

template <typename DType, typename Offset>
__aicore__ inline constexpr auto GetPackedElementOffset(Offset offset)
{
    if constexpr (IsPackedFp4Type<DType>) {
        return offset >> 1;
    } else {
        return offset;
    }
}

template <typename DType, typename Offset>
__aicore__ inline constexpr auto GetPackedByteOffset(Offset offset)
{
    if constexpr (IsPackedFp4Type<DType>) {
        return offset >> 1;
    } else {
        return offset * sizeof(DType);
    }
}

template <typename DType, typename Offset>
__aicore__ inline __gm__ DType* GetPackedGmAddr(__gm__ DType* addr, Offset offset)
{
    if constexpr (IsPackedFp4Type<DType>) {
        return (__gm__ DType*)((__gm__ uint8_t*)addr + (offset >> 1));
    } else {
        return addr + offset;
    }
}
} // namespace TileOp

template <typename T, typename Shape, typename Stride, bool need_mask = false>
class PtoGlobal {
public:
    using Dtype = std::conditional_t<std::is_same_v<typename T::Type, __gm__ bool>, __gm__ uint8_t, typename T::Type>;
    using Type = pto::GlobalTensor<Dtype, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;

    __aicore__ inline PtoGlobal(__gm__ typename T::Type* addr, const Shape& shape, const Stride& stride)
        : data_((__gm__ Dtype*)(addr),
                pto::Shape(GetTupleElementWithDefaultOverride<Shape, DIM_1ST, 1, need_mask>(shape),
                           GetTupleElementWithDefaultOverride<Shape, DIM_2ND, 1, need_mask>(shape),
                           GetTupleElementWithDefaultOverride<Shape, DIM_3RD, 1, need_mask>(shape),
                           GetTupleElementWithDefaultOverride<Shape, DIM_4TH>(shape),
                           GetTupleElementWithDefaultOverride<Shape, DIM_5TH>(shape)),
                pto::Stride(GetTupleElementWithDefaultOverride<Stride, DIM_1ST, 0, need_mask>(stride),
                            GetTupleElementWithDefaultOverride<Stride, DIM_2ND, 0, need_mask>(stride),
                            GetTupleElementWithDefaultOverride<Stride, DIM_3RD, 0, need_mask>(stride),
                            GetTupleElementWithDefaultOverride<Stride, DIM_4TH, 0>(stride),
                            GetTupleElementWithDefaultOverride<Stride, DIM_5TH, 0>(stride)))
    {}

    __aicore__ inline PtoGlobal(const Shape& shape, const Stride& stride) : PtoGlobal(0x0, shape, stride) {}

    __aicore__ inline void Assign(__gm__ typename T::Type* addr) { pto::TASSIGN(data_, (__gm__ Dtype*)addr); }

    inline Type& Data() { return data_; }

private:
    Type data_;
};

template <typename T>
__aicore__ inline size_t GenTileOffset(const T& tensor, const TileOffset& offsets)
{
    const auto layout = tensor.GetLayout();
    size_t offset = Std::get<DIM_1ST>(offsets) * layout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    offset += Std::get<DIM_2ND>(offsets) * layout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    offset += Std::get<DIM_3RD>(offsets) * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    return offset;
}

template <typename T>
__aicore__ inline size_t GenTileOffset(const T& tensor, const TileOffset4Dim& offsets)
{
    const auto layout = tensor.GetLayout();
    size_t offset = Std::get<DIM_1ST>(offsets) * layout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    offset += Std::get<DIM_2ND>(offsets) * layout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    offset += Std::get<DIM_3RD>(offsets) * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    offset += Std::get<DIM_4TH>(offsets) * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    return offset;
}

template <typename T, bool Mergeable = false>
__aicore__ inline constexpr size_t GetMergedAxisIfNeed()
{
    if constexpr (Mergeable) {
        constexpr auto size = Std::tuple_size<typename T::TileShape>::value;
        return TileOp::GetOutterAxisMergeResult<size, typename T::TileShape>();
    } else {
        return TileOp::GetTensorTileShapeDim<T, DIM_4TH, MAX_DIMS>();
    }
}

template <typename T, bool Mergeable = false>
__aicore__ inline constexpr int GetValidHeight()
{
    if constexpr (Mergeable) {
        constexpr auto size = Std::tuple_size<typename T::Shape>::value;
        return TileOp::GetOutterAxisMergeResult<size, typename T::Shape>();
    } else if constexpr (T::IsStaticLayout()) {
        return TileOp::GetTensorShapeDim<T, DIM_4TH, MAX_DIMS>();
    } else {
        return -1;
    }
}

template <typename T>
__aicore__ inline constexpr size_t GetAllAxisTileProduct()
{
    constexpr auto size = Std::tuple_size<typename T::TileShape>::value;
    return TileOp::GetAllAxisProduct<size, typename T::TileShape>();
}

template <typename T>
__aicore__ inline constexpr size_t GetAllAxisValidProduct()
{
    constexpr auto size = Std::tuple_size<typename T::Shape>::value;
    return TileOp::GetAllAxisProduct<size, typename T::Shape>();
}

template <typename Dst, typename Src>
__aicore__ inline constexpr bool IsElementwiseDstLayoutCoveredByOperand()
{
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<Dst, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<Dst, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<Src, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<Src, DIM_5TH, MAX_DIMS>();
    constexpr bool rowContinuous = (dstTileH == 1 ||
                                    TileOp::GetTensorStrideDim<Dst, DIM_4TH, MAX_DIMS>() == dstTileW) &&
                                   (srcTileH == 1 ||
                                    TileOp::GetTensorStrideDim<Src, DIM_4TH, MAX_DIMS>() == srcTileW) &&
                                   TileOp::GetTensorStrideDim<Dst, DIM_5TH, MAX_DIMS>() == 1 &&
                                   TileOp::GetTensorStrideDim<Src, DIM_5TH, MAX_DIMS>() == 1;
    // Outer tile dimensions describe local buffer capacity. They may differ even when the runtime execution shapes
    // are identical, so only static tensor shapes can be used for an outer-dimension coverage check.
    if constexpr (Dst::IsStaticLayout() && Src::IsStaticLayout()) {
        constexpr bool validShapeCovered = TileOp::GetTensorShapeDim<Dst, DIM_1ST, MAX_DIMS>() <=
                                               TileOp::GetTensorShapeDim<Src, DIM_1ST, MAX_DIMS>() &&
                                           TileOp::GetTensorShapeDim<Dst, DIM_2ND, MAX_DIMS>() <=
                                               TileOp::GetTensorShapeDim<Src, DIM_2ND, MAX_DIMS>() &&
                                           TileOp::GetTensorShapeDim<Dst, DIM_3RD, MAX_DIMS>() <=
                                               TileOp::GetTensorShapeDim<Src, DIM_3RD, MAX_DIMS>() &&
                                           TileOp::GetTensorShapeDim<Dst, DIM_4TH, MAX_DIMS>() <=
                                               TileOp::GetTensorShapeDim<Src, DIM_4TH, MAX_DIMS>() &&
                                           TileOp::GetTensorShapeDim<Dst, DIM_5TH, MAX_DIMS>() <=
                                               TileOp::GetTensorShapeDim<Src, DIM_5TH, MAX_DIMS>();
        return rowContinuous && validShapeCovered;
    }
    return rowContinuous;
}

template <typename T>
__aicore__ inline constexpr int GetValidWidth()
{
    if constexpr (T::IsStaticLayout()) {
        return TileOp::GetTensorShapeDim<T, DIM_5TH, MAX_DIMS>();
    } else {
        return -1;
    }
}

template <typename T, bool Mergeable, bool MergeAll = true>
struct PtoTileDimConfig {
    static constexpr auto tileH = TileOp::GetTensorTileShapeDim<T, DIM_4TH, MAX_DIMS>();
    static constexpr auto tileW = TileOp::GetTensorTileShapeDim<T, DIM_5TH, MAX_DIMS>();
    static constexpr auto validH = GetValidHeight<T>();
    static constexpr auto validW = GetValidWidth<T>();
};

template <typename T>
struct PtoTileDimConfig<T, true, true> {
    static constexpr auto tileH = size_t(1);
    static constexpr auto tileW = GetAllAxisTileProduct<T>();
    static constexpr auto validH = 1;
    static constexpr auto validW = GetAllAxisValidProduct<T>();
};

template <typename T>
struct PtoTileDimConfig<T, true, false> {
    static constexpr auto tileH = GetMergedAxisIfNeed<T, true>();
    static constexpr auto tileW = TileOp::GetTensorTileShapeDim<T, DIM_5TH, MAX_DIMS>();
    static constexpr auto validH = GetValidHeight<T, true>();
    static constexpr auto validW = GetValidWidth<T>();
};

template <typename T, pto::BLayout Layout = pto::BLayout::RowMajor, bool Mergeable = false,
          typename DtypeOverride = void, bool MergeAll = true>
class PtoTile {
private:
    static constexpr auto size = Std::tuple_size<typename T::Shape>::value;
    static constexpr auto tileH = PtoTileDimConfig<T, Mergeable, MergeAll>::tileH;
    static constexpr auto tileW = PtoTileDimConfig<T, Mergeable, MergeAll>::tileW;
    static constexpr auto validH = PtoTileDimConfig<T, Mergeable, MergeAll>::validH;
    static constexpr auto validW = PtoTileDimConfig<T, Mergeable, MergeAll>::validW;

public:
    using DefaultDtype = std::conditional_t<std::is_same_v<typename T::Type, bool>, uint8_t, typename T::Type>;
    using Dtype = std::conditional_t<std::is_void_v<DtypeOverride>, DefaultDtype, DtypeOverride>;
    using Type = pto::Tile<pto::TileType::Vec, Dtype, tileH, tileW, Layout, validH, validW>;

    __aicore__ inline PtoTile() : data_()
    {
        static_assert(T::IsStaticLayout(), "Only valild for static layout tile tensor.");
    }

    __aicore__ inline PtoTile(const uint64_t& addr) : PtoTile() { pto::TASSIGN(data_, addr); }

    __aicore__ inline PtoTile(const int& h, const int& w)
    {
        if constexpr (!T::IsStaticLayout()) {
            Type tile(h, w);
            data_ = tile;
        }
    }

    __aicore__ inline PtoTile(const int& h, const int& w, const uint64_t addr) : PtoTile(h, w)
    {
        pto::TASSIGN(data_, addr);
    }

    __aicore__ inline PtoTile(const T& tensor)
        : PtoTile(tensor.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                  tensor.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>())
    {}

    __aicore__ inline Type& Data() { return data_; }

    __aicore__ inline const Type& Data() const { return data_; }

    __aicore__ inline void Assign(uint64_t addr) { pto::TASSIGN(data_, addr); }

    __aicore__ inline void Assign(uint64_t addr, uint64_t element_cnt)
    {
        pto::TASSIGN(data_, addr + TileOp::GetPackedByteOffset<typename T::Type>(element_cnt));
    }

    __aicore__ inline void Assign(T& tensor) { Assign((uint64_t)(tensor.GetAddr())); }

    __aicore__ inline void Assign(T& tensor, const TileOffset& offsets)
    {
        auto byteOffset = TileOp::GetPackedByteOffset<typename T::Type>(GenTileOffset(tensor, offsets));
        pto::TASSIGN(data_, (uint64_t)(tensor.GetAddr() + byteOffset));
    }

private:
    Type data_;
};

template <typename Dst, typename Src, typename DtypeOverride = void>
struct ElementwiseOperandExecConfig {
    static constexpr bool dstLayoutCovered = IsElementwiseDstLayoutCoveredByOperand<Dst, Src>();
    static_assert(dstLayoutCovered,
                  "The source layout must cover the destination execution layout for an elementwise operation.");
    static constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<Dst, DIM_4TH, MAX_DIMS>();
    static constexpr auto operandTileH = TileOp::GetTensorTileShapeDim<Src, DIM_4TH, MAX_DIMS>();
    static constexpr auto tileH = dstTileH < operandTileH ? dstTileH : operandTileH;
    static constexpr auto tileW = TileOp::GetTensorTileShapeDim<Src, DIM_5TH, MAX_DIMS>();
    static constexpr auto validH = GetValidHeight<Dst>();
    static constexpr auto validW = GetValidWidth<Dst>();
    using OperandDtype = std::conditional_t<std::is_void_v<DtypeOverride>, typename PtoTile<Src>::Dtype, DtypeOverride>;
    using OperandTile = pto::Tile<pto::TileType::Vec, OperandDtype, tileH, tileW, pto::BLayout::RowMajor, validH,
                                  validW>;
};

template <typename DtypeOverride = void, typename Dst, typename Src>
__aicore__ inline auto MakeElementwiseOperandExecTile(Dst dst, Src)
{
    using OperandExecTile = typename ElementwiseOperandExecConfig<Dst, Src, DtypeOverride>::OperandTile;
    if constexpr (Dst::IsStaticLayout()) {
        OperandExecTile operandExecTile;
        return operandExecTile;
    } else {
        OperandExecTile operandExecTile(dst.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                                        dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>());
        return operandExecTile;
    }
}

template <size_t index, size_t expectSize = MAX_DIMS, typename Dst, typename Src>
__aicore__ inline auto GetElementwiseOperandExecShapeDim(Dst dst, Src)
{
    static_assert(ElementwiseOperandExecConfig<Dst, Src>::dstLayoutCovered);
    return dst.GetLayout().template GetShapeDim<index, expectSize>();
}

template <typename OperandTile, typename Operand>
__aicore__ inline void AssignElementwiseOperandExecTile(OperandTile& operandTile, Operand operand,
                                                        const TileOffset& offsets)
{
    auto operandByteOffset = TileOp::GetPackedByteOffset<typename Operand::Type>(GenTileOffset(operand, offsets));
    pto::TASSIGN(operandTile, (uint64_t)(operand.GetAddr() + operandByteOffset));
}
#endif // TILEOP_TILE_OPERATOR_PTO_TILE__H
