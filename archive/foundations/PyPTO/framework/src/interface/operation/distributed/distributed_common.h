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
 * \file distributed_common.h
 * \brief
 */

#ifndef DISTRIBUTED_COMMON_H
#define DISTRIBUTED_COMMON_H

#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "distributed_expand.h"
#include "tilefwk/comm_group_recorder.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
namespace Distributed {
constexpr int32_t DIST_HEAD_SHAPE = 0;
constexpr int32_t DIST_HEAD_COUNT = 1;
constexpr int32_t DIST_TAIL_SHAPE = 2;
constexpr int32_t DIST_INDEX_ZERO = 0;
constexpr int32_t DIST_INDEX_ONE = 1;
constexpr int32_t DIST_INDEX_TWO = 2;
constexpr uint16_t COPY_BLOCK_BYTE_SIZE = 32;
constexpr uint16_t SAME_ADDR_BYTE_SIZE = 512;
constexpr uint64_t SHMEM_SIZE_ALIGN = 512;
constexpr int32_t ROUTED_EXPET_NUM = 160;
constexpr int32_t FFN_TILE_SIZE = 8;
constexpr size_t MIN_TILE_SHAPE_DIM = 2;
constexpr int32_t AIV_NUM = 4;
constexpr int32_t RECEIVE_CNT_OUT_ROW = 1024;
constexpr int32_t RECEIVE_CNT_OUT_COL = 512;
constexpr int32_t SHMEM_SIGNAL_STRIDE = 8;
constexpr int32_t MAX_SHMEM_TILE_DIMS = 4;
constexpr int32_t MAX_GROUP_NAME_LENGTH = 128;
constexpr uint64_t MOE_INPUT_DIM = 2;
constexpr int32_t MOE_HIDDEN_SIZE = 5120;
constexpr int32_t MOE_BATCH_SIZE = 8;
constexpr int32_t MOE_TOPK = 8;
constexpr int32_t MOE_ASSIST_INFO_COL = 3;
enum class TileIndex : size_t { HEAD_SHAPE, HEAD_NUM, TAIL_SHAPE };

enum class AllReduceType {
    ONE_SHOT,
    TWO_SHOT,
};

inline std::string AtomicTypeToString(AtomicType type)
{
    switch (type) {
        case AtomicType::SET:
            return "TileOp::Distributed::AtomicType::SET";
        case AtomicType::ADD:
            return "TileOp::Distributed::AtomicType::ADD";
        default:
            return "";
    }
}

inline std::string OpTypeToString(OpType type)
{
    switch (type) {
        case OpType::EQ:
            return "OpType::EQ";
        case OpType::NE:
            return "OpType::NE";
        case OpType::LT:
            return "OpType::LT";
        case OpType::LE:
            return "OpType::LE";
        case OpType::GT:
            return "OpType::GT";
        case OpType::GE:
            return "OpType::GE";
        default:
            return "";
    }
}

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;

template <typename T>
typename std::enable_if<!is_iterable_v<T>, std::string>::type ToString(T value)
{
    if constexpr (std::is_same_v<T, std::string>) {
        return value;
    } else if constexpr (std::is_convertible_v<T, std::string>) {
        return std::string(value);
    } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(value);
    } else if constexpr (std::is_same_v<T, AtomicType>) {
        return AtomicTypeToString(value);
    } else if constexpr (std::is_same_v<T, DataType>) {
        return DataType2String(value);
    } else if constexpr (std::is_same_v<T, Opcode>) {
        return OpcodeManager::Inst().GetOpcodeStr(value);
    } else if constexpr (std::is_same_v<T, OpType>) {
        return OpTypeToString(value);
    } else {
        return "";
    }
}

template <typename Container>
typename std::enable_if<is_iterable_v<Container>, std::string>::type ToString(const Container& c)
{
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& item : c) {
        if (!first) {
            oss << ", ";
        }
        oss << ToString(item);
        first = false;
    }
    oss << "]";
    return oss.str();
}

struct ShmemPutAttr {
    std::string group;
    Shape copyBufferShape;
    AtomicType atomicType = AtomicType::SET;
    SymbolicScalar ownerRank;
};

struct ShmemGetAttr {
    Shape copyBufferShape;
    AtomicType atomicType = AtomicType::SET;
    SymbolicScalar ownerRank;
    std::string group;
};

struct ShmemSignalAttr {
    int64_t signalValue = 1;
    int32_t signalStride = SHMEM_SIGNAL_STRIDE;
    std::vector<int64_t> tileShape;
    AtomicType atomicType = AtomicType::SET;
    bool notifyAll{false};
    int64_t worldSize{0};
    std::vector<int64_t> viewshapes;
    int64_t viewTileNum{0};
    int64_t totalTileNum{0};
    SymbolicScalar ownerRank;
    std::string group;
};

struct ShmemWaitUntilAttr {
    int32_t expectedSum = 0;
    int32_t signalStride = SHMEM_SIGNAL_STRIDE;
    bool resetSignal = false;
    std::vector<int64_t> tileShape;
    std::vector<int64_t> viewshapes;
    std::vector<int64_t> viewTileStrides;
    std::vector<int64_t> viewIndexStrides;
    int64_t viewTileNum{0};
    int64_t totalTileNum{0};
    SymbolicScalar ownerRank;
    std::string group;
};

struct ShmemSetAttr {
    std::string group;
    bool isSetData{true};
    Shape setBufferShape;
    SymbolicScalar ownerRank;
};

inline int GetTotalTileNum(const std::array<int, MAX_DIST_DIM_SIZE>& tile)
{
    return tile[static_cast<size_t>(TileIndex::HEAD_NUM)] +
           static_cast<int>(tile[static_cast<size_t>(TileIndex::TAIL_SHAPE)] != 0);
}

inline std::tuple<int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> GetTotalTileNum(
    const VecTile& tileShape, const ShmemTensor& src)
{
    Shape rawShape = ((Operation*)src.signalOp)->GetOOperands()[0]->tensor->rawshape;
    Shape dataShape = src.signal.GetShape();

    ASSERT(DistributedErrorCode::INVALID_TENSOR_DIM, tileShape.size() >= MIN_TILE_SHAPE_DIM)
        << "Invalid dimensional: "
        << " tileShape dim must >= " << MIN_TILE_SHAPE_DIM << ", but got dimensional=" << tileShape.size();
    ASSERT(DistributedErrorCode::INVALID_TENSOR_DIM, dataShape.size() == (tileShape.size() + 1))
        << "Invalid dimensional: "
        << " shape parameter dim must = tileShape dim, but got shape parameter dim=" << (dataShape.size() - 1)
        << ", tileShape dim=" << tileShape.size();

    size_t vecTileDim = tileShape.size();
    size_t startDim = dataShape.size() - vecTileDim;

    for (size_t i = 0; i < vecTileDim; ++i) {
        size_t curDim = startDim + i;
        ASSERT(DistributedErrorCode::INVALID_TENSOR_DIM, rawShape[curDim] % dataShape[curDim] == 0)
            << "signal of shmem tensor shape[" << i << "]=" << rawShape[curDim]
            << " must be divisible by shape parameter[" << i << "]=" << dataShape[curDim];
    }

    std::vector<int64_t> viewshapes(vecTileDim);
    std::vector<int64_t> dimTileNums(vecTileDim);
    std::vector<int64_t> viewTileStrides(vecTileDim);
    std::vector<int64_t> viewIndexStrides(vecTileDim);

    int64_t viewTileNum = 1;
    int64_t crossViewNum = 1;

    viewTileStrides[0] = 1;
    viewIndexStrides[0] = 1;

    for (size_t i = 0; i < vecTileDim; ++i) {
        size_t curDim = startDim + i;
        viewshapes[i] = dataShape[curDim];
        int64_t totalShape = dataShape[curDim];
        int64_t tileShapeVal = tileShape[i];

        dimTileNums[i] = totalShape / tileShapeVal + (totalShape % tileShapeVal == 0 ? 0 : 1);
        viewTileNum *= dimTileNums[i];
        crossViewNum *= (rawShape[curDim] / dataShape[curDim]);

        if (i > 0) {
            viewTileStrides[i] = viewTileStrides[i - 1] * dimTileNums[i - 1];
            viewIndexStrides[i] = viewIndexStrides[i - 1] * (rawShape[curDim - 1] / dataShape[curDim - 1]);
        }
    }

    int64_t totalTileNum = viewTileNum * crossViewNum;

    return {totalTileNum, viewTileNum, viewshapes, viewTileStrides, viewIndexStrides};
}
} // namespace Distributed
} // namespace npu::tile_fwk

#endif
