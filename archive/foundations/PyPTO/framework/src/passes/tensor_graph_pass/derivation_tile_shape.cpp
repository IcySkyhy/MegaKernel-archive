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
 * \file derivation_tile_shape.cpp
 * \brief
 */

#include "derivation_tile_shape.h"
#include <queue>
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "DerivationTileShape"

namespace npu {
namespace tile_fwk {
static std::string GetStr(const std::vector<int64_t>& vec)
{
    std::string ret;
    for (size_t i = 0; i < vec.size(); ++i) {
        ret += std::to_string(vec[i]);
        if (i < vec.size() - 1) {
            ret += ", ";
        }
    }
    return "{" + ret + "}";
}

static std::string GetShapeStr(const std::vector<ShapeStatus>& statusVec)
{
    std::string ret;
    for (size_t i = 0; i < statusVec.size(); ++i) {
        ret += std::to_string(statusVec[i].size);
        if (i < statusVec.size() - 1) {
            ret += ", ";
        }
    }
    return "{" + ret + "}";
}

static std::string GetTileStr(const std::vector<ShapeStatus>& statusVec)
{
    std::string ret;
    for (size_t i = 0; i < statusVec.size(); ++i) {
        ret += std::to_string(statusVec[i].tileSize);
        if (i < statusVec.size() - 1) {
            ret += ", ";
        }
    }
    return "{" + ret + "}";
}

static void InitShapeStatus(const std::vector<int64_t>& inShape, const std::vector<int64_t>& tileShape,
                            std::vector<ShapeStatus>& inStatus)
{
    size_t i = 0UL;
    while (i < inShape.size()) {
        inStatus[i].size = inShape[i];
        inStatus[i].tileSize = tileShape[i];
        i++;
    }
}

static std::vector<int64_t> ShapeToStride(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> stride;
    stride.resize(shape.size());
    stride[shape.size() - 1] = 1;
    for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
        stride[k] = stride[k + 1] * shape[k + 1];
    }
    return stride;
}

static void InitShapeStatusStride(const std::vector<int64_t>& shape, std::vector<ShapeStatus>& shapeStatus)
{
    int64_t currentStride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        shapeStatus[i].stride = currentStride;
        /* 更新下一个维度的stride（前一个维度） */
        currentStride *= shape[i];
    }
}

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
}

static int64_t DotProduct(const std::vector<int64_t>& subIndex, const std::vector<int64_t>& stride)
{
    return std::inner_product(subIndex.begin(), subIndex.end(), stride.begin(), 0);
}

static bool CanNotAlignShape(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape, int64_t inSize,
                             int64_t outSize)
{
    if ((inSize == 0) || (outSize == 0)) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "InSize or outSize is 0, cannot calculate the align of the input%s and the output%s",
                         GetStr(inShape).c_str(), GetStr(outShape).c_str());
        return true;
    }
    /* 两个size不能互相整除，无法推导aligned shape */
    auto ret = (inSize % outSize != 0) && (outSize % inSize != 0);
    if (ret) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Non-segmentable axis, cannot calculate the align of the input%s and the output%s",
                         GetStr(inShape).c_str(), GetStr(outShape).c_str());
    }
    return ret;
}

static void ProcessSplitAndMerge(size_t& a, size_t o, int64_t shapeSize, ShapeStatus& inStatus,
                                 std::vector<int64_t>& alignedShape, std::vector<ShapeStatus>& alignedStatus)
{
    /* input split, output merge */
    inStatus.axisType = AXIS_SPLIT;
    inStatus.transformAxisIndex.push_back(a++);

    alignedShape.push_back(shapeSize);
    alignedStatus.emplace_back(shapeSize, 0, AXIS_MERGE, std::vector<size_t>{o}, 0);
}

static void ProcessSplitAndKeep(size_t& a, size_t o, int64_t shapeSize, ShapeStatus& inStatus,
                                std::vector<int64_t>& alignedShape, std::vector<ShapeStatus>& alignedStatus)
{
    /* input split, output keep */
    inStatus.axisType = AXIS_SPLIT;
    inStatus.transformAxisIndex.push_back(a++);

    alignedShape.push_back(shapeSize);
    alignedStatus.emplace_back(shapeSize, 0, AXIS_KEEP, std::vector<size_t>{o}, 0);
}

static void ProcessKeepAndMerge(size_t& a, size_t o, int64_t shapeSize, ShapeStatus& inStatus,
                                std::vector<int64_t>& alignedShape, std::vector<ShapeStatus>& alignedStatus)
{
    /* input keep, output merge */
    inStatus.axisType = AXIS_KEEP;
    inStatus.transformAxisIndex.push_back(a++);

    alignedShape.push_back(shapeSize);
    alignedStatus.emplace_back(shapeSize, 0, AXIS_MERGE, std::vector<size_t>{o}, 0);
}

static void ProcessKeepAndKeep(size_t& a, size_t o, int64_t shapeSize, ShapeStatus& inStatus,
                               std::vector<int64_t>& alignedShape, std::vector<ShapeStatus>& alignedStatus)
{
    /* input keep, output keep */
    inStatus.axisType = AXIS_KEEP;
    inStatus.transformAxisIndex.push_back(a++);

    alignedShape.push_back(shapeSize);
    alignedStatus.emplace_back(shapeSize, 0, AXIS_KEEP, std::vector<size_t>{o}, 0);
}

static Status HandleInputProductCase(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                                     AlignContext& ctx, std::vector<ShapeStatus>& inStatus,
                                     std::vector<int64_t>& alignedShape, std::vector<ShapeStatus>& alignedStatus)
{
    if (CanNotAlignShape(inShape, outShape, ctx.iprod, outShape[ctx.o])) {
        return WARNING;
    }
    /* 涉及outputshape中的切轴 */
    if (outShape[ctx.o] > ctx.iprod) {
        ProcessSplitAndMerge(ctx.a, ctx.o, ctx.iprod, inStatus[ctx.i], alignedShape, alignedStatus);
        ctx.oprod = outShape[ctx.o] / ctx.iprod;
        ctx.iprod = 1;
        ctx.i++;
        return SUCCESS;
    }
    /* inputshape多次切轴，o继续后移 */
    ProcessSplitAndKeep(ctx.a, ctx.o, outShape[ctx.o], inStatus[ctx.i], alignedShape, alignedStatus);
    if (outShape[ctx.o] == ctx.iprod) {
        ctx.i++;
    }
    ctx.iprod = ctx.iprod / outShape[ctx.o];
    ctx.o++;
    return SUCCESS;
}

static Status HandleOutputProductCase(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                                      AlignContext& ctx, std::vector<ShapeStatus>& inStatus,
                                      std::vector<int64_t>& alignedShape, std::vector<ShapeStatus>& alignedStatus)
{
    if (CanNotAlignShape(inShape, outShape, ctx.oprod, inShape[ctx.i])) {
        return WARNING;
    }
    if (inShape[ctx.i] > ctx.oprod) {
        ProcessSplitAndMerge(ctx.a, ctx.o, ctx.oprod, inStatus[ctx.i], alignedShape, alignedStatus);
        ctx.iprod = inShape[ctx.i] / ctx.oprod;
        ctx.oprod = 1;
        ctx.o++;
        return SUCCESS;
    }
    /* outshape多次切轴，i继续后移 */
    ProcessKeepAndMerge(ctx.a, ctx.o, inShape[ctx.i], inStatus[ctx.i], alignedShape, alignedStatus);
    if (inShape[ctx.i] == ctx.oprod) {
        ctx.o++;
    }
    ctx.oprod = ctx.oprod / inShape[ctx.i];
    ctx.i++;
    return SUCCESS;
}

static Status HandleNormalCase(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                               AlignContext& ctx, std::vector<ShapeStatus>& inStatus,
                               std::vector<int64_t>& alignedShape, std::vector<ShapeStatus>& alignedStatus)
{
    if (CanNotAlignShape(inShape, outShape, inShape[ctx.i], outShape[ctx.o])) {
        return WARNING;
    }
    if (outShape[ctx.o] > inShape[ctx.i]) {
        /* alignshape 合轴 */
        ProcessKeepAndMerge(ctx.a, ctx.o, inShape[ctx.i], inStatus[ctx.i], alignedShape, alignedStatus);
        ctx.oprod = outShape[ctx.o] / inShape[ctx.i];
        ctx.i++;
        return SUCCESS;
    } else if (outShape[ctx.o] < inShape[ctx.i]) {
        ProcessSplitAndKeep(ctx.a, ctx.o, outShape[ctx.o], inStatus[ctx.i], alignedShape, alignedStatus);
        ctx.iprod = inShape[ctx.i] / outShape[ctx.o];
        ctx.o++;
        return SUCCESS;
    }

    ProcessKeepAndKeep(ctx.a, ctx.o, outShape[ctx.o], inStatus[ctx.i], alignedShape, alignedStatus);
    ctx.o++;
    ctx.i++;
    return SUCCESS;
}

static Status DerivationAlignShape(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                                   std::vector<ShapeStatus>& inStatus, std::vector<int64_t>& alignedShape,
                                   std::vector<ShapeStatus>& alignedStatus)
{
    AlignContext ctx;
    while (ctx.i < inShape.size() || ctx.o < outShape.size()) {
        if (ctx.iprod != 1) {
            /* 处理input切轴 */
            if (HandleInputProductCase(inShape, outShape, ctx, inStatus, alignedShape, alignedStatus) != SUCCESS) {
                return WARNING;
            }
            continue;
        }
        if (ctx.oprod != 1) {
            /* 处理output切轴 */
            if (HandleOutputProductCase(inShape, outShape, ctx, inStatus, alignedShape, alignedStatus) != SUCCESS) {
                return WARNING;
            }
            continue;
        }
        if (ctx.i >= inShape.size()) {
            /* 处理输出尾轴为1的场景 */
            while (ctx.o < outShape.size() && outShape[ctx.o] == 1) {
                ProcessSplitAndKeep(ctx.a, ctx.o, outShape[ctx.o], inStatus[inShape.size() - 1], alignedShape,
                                    alignedStatus);
                ctx.o++;
            }
            continue;
        }
        if (ctx.o >= outShape.size()) {
            /* 处理输入尾轴为1的场景 */
            while (ctx.i < inShape.size() && inShape[ctx.i] == 1) {
                ProcessKeepAndKeep(ctx.a, (outShape.size() - 1), inShape[ctx.i], inStatus[ctx.i], alignedShape,
                                   alignedStatus);
                ctx.i++;
            }
            continue;
        }

        if (HandleNormalCase(inShape, outShape, ctx, inStatus, alignedShape, alignedStatus) != SUCCESS) {
            return WARNING;
        }
    }
    return SUCCESS;
}

static bool CheckMemoryCondition(const std::vector<size_t>& indexes, std::vector<ShapeStatus>& shapeStatus)
{
    /* 涉及拆轴的tile size需要满足从低维度开始展开 */
    for (size_t i = 0; i < indexes.size() - 1; i++) {
        auto cur = indexes[i];
        auto next = indexes[i + 1];
        /* tile size可能会大于shape size */
        if (shapeStatus[cur].tileSize != 1 && shapeStatus[next].tileSize < shapeStatus[next].size) {
            return false;
        }
    }
    return true;
}

static Status HandleSplitLargeTileShape(std::vector<ShapeStatus>& inStatus, std::vector<size_t> alignedIndexes,
                                        int64_t& tileShape, std::vector<ShapeStatus>& alignedStatus)
{
    int64_t tempShape = tileShape;
    for (auto i = alignedIndexes.begin(); i != alignedIndexes.end(); ++i) {
        auto& alignStatus = alignedStatus[*i];
        int64_t alignedShape = alignStatus.size;
        if (tileShape == 1) {
            alignStatus.tileSize = 1;
            continue;
        }
        if (i == (alignedIndexes.end() - 1)) {
            if (tempShape < alignedShape) {
                APASS_LOG_WARN_F(Elements::Operation,
                                 "Split last large tile shape fail, Tensor Shape:%s, TileShape:%s, AlignedShape:%s",
                                 GetShapeStr(inStatus).c_str(), GetTileStr(inStatus).c_str(),
                                 GetShapeStr(alignedStatus).c_str());
                return WARNING;
            }
            alignStatus.tileSize = tempShape;
        } else {
            if (tempShape % alignedShape != 0) {
                APASS_LOG_WARN_F(
                    Elements::Operation, "Split large tile shape fail, Tensor Shape:%s, TileShape:%s, AlignedShape:%s",
                    GetShapeStr(inStatus).c_str(), GetTileStr(inStatus).c_str(), GetShapeStr(alignedStatus).c_str());
                return WARNING;
            }
            alignStatus.tileSize = alignedShape;
            tempShape = tempShape / alignedShape;
        }
    }
    return SUCCESS;
}

static Status HandleSplitTileShape(std::vector<size_t> alignedIndexes, int64_t& tileShape,
                                   std::vector<ShapeStatus>& alignedStatus)
{
    for (auto i = alignedIndexes.rbegin(); i != alignedIndexes.rend(); ++i) {
        auto& alignStatus = alignedStatus[*i];
        int64_t alignedShape = alignStatus.size;
        if (tileShape > alignedShape) {
            alignStatus.tileSize = alignedShape;
            tileShape = tileShape / alignedShape;
        } else {
            alignStatus.tileSize = tileShape;
            tileShape = 1;
        }
    }
    return SUCCESS;
}

static Status DerivationAlignShapeTile(std::vector<ShapeStatus>& inStatus, std::vector<ShapeStatus>& alignedStatus)
{
    if (inStatus.size() == 0 || alignedStatus.size() == 0) {
        APASS_LOG_WARN_F(Elements::Tensor, "The shape status is empty.");
        return WARNING;
    }

    for (auto& shapeStatus : inStatus) {
        auto alignedIndexes = shapeStatus.transformAxisIndex;
        auto tileShape = shapeStatus.tileSize;

        if (shapeStatus.axisType == AXIS_KEEP) {
            /* 不涉及拆轴的tile size保持不变 */
            auto index = alignedIndexes[0];
            alignedStatus[index].tileSize = tileShape;
            continue;
        }
        /* shapeStatus.axisType只可能为keep或者split */
        if (shapeStatus.axisType != AXIS_SPLIT) {
            APASS_LOG_WARN_F(Elements::Tensor, "The axisType property of inStatus is invalid.");
            return WARNING;
        }

        /* 拆轴的集合对应的alignedshape状态处理 */
        if (tileShape > shapeStatus.size) {
            if (HandleSplitLargeTileShape(inStatus, alignedIndexes, tileShape, alignedStatus) != SUCCESS) {
                return WARNING;
            }
        } else if (HandleSplitTileShape(alignedIndexes, tileShape, alignedStatus) != SUCCESS) {
            return WARNING;
        }

        if (!CheckMemoryCondition(alignedIndexes, alignedStatus)) {
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "The memory Layout can not be mapped: Tensor Shape:%s, TileShape:%s, AlignedShape:%s, Align "
                "TileShape:%s",
                GetShapeStr(inStatus).c_str(), GetTileStr(inStatus).c_str(), GetShapeStr(alignedStatus).c_str(),
                GetTileStr(alignedStatus).c_str());
            return WARNING;
        }
    }
    return SUCCESS;
}

static Status DerivationOutShapeTileWithAlign(std::vector<ShapeStatus>& alignedStatus,
                                              const std::vector<int64_t>& outShape, std::vector<int64_t>& tileShape)
{
    for (auto& alignStatus : alignedStatus) {
        auto o = alignStatus.transformAxisIndex[0];
        /* alignStatus.axisType只可能为keep或者merge */
        if (alignStatus.axisType != AXIS_KEEP && alignStatus.axisType != AXIS_MERGE) {
            APASS_LOG_WARN_F(Elements::Tensor, "The axisType property of alignedStatus is invalid.");
            return WARNING;
        }
        tileShape[o] = tileShape[o] * alignStatus.tileSize;
    }

    /* 检查推导出的alignshape是否满足outshape的约束 */
    std::vector<size_t> indexes;
    size_t symbol = 0;
    for (size_t i = 0; i < alignedStatus.size(); i++) {
        if (alignedStatus[i].axisType != AXIS_MERGE) {
            continue;
        }

        auto o = alignedStatus[i].transformAxisIndex[0];
        if (indexes.empty()) {
            symbol = o;
            indexes.push_back(i);
        } else if (symbol == o) {
            indexes.push_back(i);
        } else {
            if (!CheckMemoryCondition(indexes, alignedStatus)) {
                APASS_LOG_WARN_F(
                    Elements::Tensor,
                    "The memory Layout can not be mapped: AlignedShape%s, OutputShape%s, alignTileShape%s, tileShape%s",
                    GetShapeStr(alignedStatus).c_str(), GetStr(outShape).c_str(), GetTileStr(alignedStatus).c_str(),
                    GetStr(tileShape).c_str());
                return WARNING;
            }
            indexes.clear();
            continue;
        }

        if (indexes.size() > 1) {
            if (!CheckMemoryCondition(indexes, alignedStatus)) {
                APASS_LOG_WARN_F(
                    Elements::Tensor,
                    "The memory Layout can not be mapped: AlignedShape%s, OutputShape%s, alignTileShape%s, tileShape%s",
                    GetShapeStr(alignedStatus).c_str(), GetStr(outShape).c_str(), GetTileStr(alignedStatus).c_str(),
                    GetStr(tileShape).c_str());
                return WARNING;
            }
        }
    }
    return SUCCESS;
}

static void TiledReshape(const int dimIdx, const std::vector<int64_t>& inShape, const std::vector<int64_t>& tileShape,
                         std::vector<int64_t> actTileShape, std::vector<int64_t> actOffset, int64_t& tileCnt,
                         std::vector<std::vector<int64_t>>& allActTileShape,
                         std::vector<std::vector<int64_t>>& allActOffset)
{
    if (static_cast<size_t>(dimIdx) == inShape.size()) {
        tileCnt++;
        APASS_LOG_DEBUG_F(Elements::Tensor, "tileCnt:%ld, actOffset%s, actTileShape%s", tileCnt,
                          GetStr(actOffset).c_str(), GetStr(actTileShape).c_str());
        allActTileShape.push_back(actTileShape);
        allActOffset.push_back(actOffset);
        return;
    }

    for (auto i = 0; i < inShape[dimIdx]; i += tileShape[dimIdx]) {
        actTileShape[dimIdx] = std::min(inShape[dimIdx] - i, tileShape[dimIdx]);
        actOffset[dimIdx] = i;
        TiledReshape(dimIdx + 1, inShape, tileShape, actTileShape, actOffset, tileCnt, allActTileShape, allActOffset);
    }
}

static int64_t TiledReshape(const std::vector<int64_t>& inShape, const std::vector<int64_t>& tileShape,
                            std::vector<std::vector<int64_t>>& allActTileShape,
                            std::vector<std::vector<int64_t>>& allActOffset)
{
    std::vector<int64_t> actOffset(inShape.size(), 0);
    std::vector<int64_t> actTileShape(inShape.size(), 1);
    int64_t tileCnt = 0;
    TiledReshape(0, inShape, tileShape, actTileShape, actOffset, tileCnt, allActTileShape, allActOffset);
    return tileCnt;
}

static int64_t calcTileSubDistance(int64_t currSub, const std::vector<int64_t>& currTile, const Stride currStride,
                                   std::vector<int64_t> actOffset, Stride tensorStride)
{
    /* 计算第currSub个元素，在tile块上的坐标，例如对于5*4*3为shape的张量，currSub=0对应[0, 0, 1], currSub=12对应[1, 0,
     * 0] */
    std::vector<int64_t> currTileSub(currTile.size(), 0);
    int64_t tempSize = currSub;
    size_t k = 0;
    for (k = 0; k < currTile.size(); k++) {
        currTileSub[k] = tempSize / currStride[k];
        tempSize = tempSize % currStride[k];
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "currSub:%ld, currTileSub%s in tile", currSub, GetStr(currTileSub).c_str());
    /* 将tile块上的坐标，转移到整个shape上的坐标，进行坐标的偏移 */
    for (k = 0; k < currTile.size(); k++) {
        currTileSub[k] = currTileSub[k] + actOffset[k];
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "currSub:%ld, currTileSub%s in tensor", currSub, GetStr(currTileSub).c_str());

    /* 根据在shape上的坐标，计算距离Tensor起点的位置，即使用stride来计算线性的距离 */
    return DotProduct(currTileSub, tensorStride);
}

static std::vector<int64_t> GetTileCntShape(const std::vector<int64_t>& shape, const std::vector<int64_t>& tileShape)
{
    std::vector<int64_t> tileCntShape(shape.size());
    for (size_t k = 0; k < shape.size(); ++k) {
        tileCntShape[k] = shape[k] / tileShape[k];
        tileCntShape[k] = tileCntShape[k] > 0 ? tileCntShape[k] : 1; // tile shape大于shape时，tileCnt为1
    }

    APASS_LOG_DEBUG_F(Elements::Tensor, "shape:%s, tileShape:%s, tileCntShape:%s", GetStr(shape).c_str(),
                      GetStr(tileShape).c_str(), GetStr(tileCntShape).c_str());
    return tileCntShape;
}

static bool IsVertex(const std::vector<int64_t>& shape, int64_t i)
{
    /* 将一维索引转换为多维坐标 */
    std::vector<int64_t> coordinates(shape.size());
    int64_t tempSize = i;
    for (int64_t k = shape.size() - 1; k >= 0; --k) {
        coordinates[k] = tempSize % shape[k];
        tempSize = tempSize / shape[k];
    }

    for (size_t k = 0; k < shape.size(); ++k) {
        /* 检查是否是顶点（所有维度上都位于边界） */
        if ((coordinates[k] != 0) && (coordinates[k] != shape[k] - 1)) {
            return false;
        }
    }
    return true;
}

static Status CheckTileShape(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                             const std::vector<int64_t>& inTile, const std::vector<int64_t>& outTile)
{
    /* 将输入展开 */
    std::vector<std::vector<int64_t>> inActOffset;
    std::vector<std::vector<int64_t>> inActTileShape;
    int64_t inTileCnt = TiledReshape(inShape, inTile, inActTileShape, inActOffset);
    /* 将输出展开 */
    std::vector<std::vector<int64_t>> outActOffset;
    std::vector<std::vector<int64_t>> outActTileShape;
    int64_t ouTileCnt = TiledReshape(outShape, outTile, outActTileShape, outActOffset);
    if (inTileCnt != ouTileCnt || inTileCnt <= 0 || ouTileCnt <= 0) {
        /* 检查tile展开后的个数是否相等 */
        APASS_LOG_WARN_F(Elements::Operation, "inTileCnt:%ld is not same as ouTileCnt:%ld", inTileCnt, ouTileCnt);
        return WARNING;
    }

    /* 计算输入输出的stride */
    Stride inStride = ShapeToStride(inShape);
    Stride outStride = ShapeToStride(outShape);
    auto inTileCntShape = GetTileCntShape(inShape, inTile);
    auto outTileCntShape = GetTileCntShape(outShape, outTile);
    APASS_LOG_DEBUG_F(Elements::Operation, "inStride:%s, outStride:%s, inTileCntShape:%s, outTileCntShape:%s",
                      GetStr(inStride).c_str(), GetStr(outStride).c_str(), GetStr(inTileCntShape).c_str(),
                      GetStr(outTileCntShape).c_str());

    for (int64_t i = 0; i < inTileCnt; ++i) {
        if (!IsVertex(inTileCntShape, i) && !IsVertex(outTileCntShape, i)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "inTileCntShape:%s, outTileCntShape:%s, i=%ld",
                              GetStr(inTileCntShape).c_str(), GetStr(outTileCntShape).c_str(), i);
            continue;
        }

        /* 检查对应tile块的size是否相等，尤其是尾块 */
        int64_t inTileSize = GetShapeSize(inActTileShape[i]);
        int64_t outTileSize = GetShapeSize(outActTileShape[i]);
        if (inTileSize != outTileSize) {
            APASS_LOG_WARN_F(Elements::Tensor, "tileCnt:%ld, inTileSize:%ld is not equal to outTileSize:%ld", i,
                             inTileSize, outTileSize);
            return WARNING;
        }

        /* 检查输入输出的每个tile块的内存排布是否相等 */
        /* 计算每个tile块对应的stride */
        Stride inTileStride = ShapeToStride(inActTileShape[i]);
        Stride outTileStride = ShapeToStride(outActTileShape[i]);
        /* 遍历获取每个tile块的每一个内存 */
        for (int64_t j = 0; j < inTileSize; j++) {
            /* 根据在tile块上的坐标，计算距离Tensor起点的位置，使用stride来计算线性的距离 */
            int64_t inDistance = calcTileSubDistance(j, inTile, inTileStride, inActOffset[i], inStride);
            int64_t outDistance = calcTileSubDistance(j, outTile, outTileStride, outActOffset[i], outStride);
            /* 如果输入输出上的对应元素距离起点的距离不相等，说明tile异常 */
            if (inDistance != outDistance) {
                APASS_LOG_WARN_F(Elements::Tensor, "Cnt:%ld %ld, inDistance:%ld is not same as outDistance:%ld", i, j,
                                 inDistance, outDistance);
                return WARNING;
            }
        }
    }
    return SUCCESS;
}

static bool ValidShape(const std::vector<int64_t>& shape)
{
    if (std::any_of(shape.begin(), shape.end(), [](int64_t x) { return x <= 0; })) {
        return false;
    }
    return true;
}

Status DerivationTileShape::DerivationReshapeTileShape(Operation* op, const Shape& inShape, const Shape& outShape,
                                                       const std::vector<int64_t>& inTileShape,
                                                       std::vector<int64_t>& outTileShape)
{
    if (op->GetOpcode() != Opcode::OP_RESHAPE) {
        return WARNING;
    }
    if (!ValidShape(inShape) || !ValidShape(outShape) || !ValidShape(inTileShape) ||
        (inShape.size() != inTileShape.size()) || (GetShapeSize(inShape) != GetShapeSize(outShape))) {
        APASS_LOG_WARN_F(Elements::Operation, "Op: %d has invalid shape, inShape%s, outShape%s, inTile%s",
                         op->GetOpMagic(), GetStr(inShape).c_str(), GetStr(outShape).c_str(),
                         GetStr(inTileShape).c_str());
        return WARNING;
    }
    /*
     * 示例1： inShape = [8,2,3], tileShape = [2, 1, 3], outShape = [2, 8, 3, 1]
     * alignShape = [2, 4, 2, 3, 1], newTileShape = [1, 2, 3, 1]
     * 示例2： inShape = [7, 8], tileShape = [3, 2], outShape = [7, 2, 2, 2]
     * alignShape = [7, 2, 2, 2], newTileShape = [3, 1, 1, 2]
     */
    std::vector<ShapeStatus> inStatus(inShape.size()); // 记录input shape和aligned shape之间关系
    InitShapeStatus(inShape, inTileShape, inStatus);
    InitShapeStatusStride(inShape, inStatus);

    std::vector<int64_t> alignedShape;
    std::vector<ShapeStatus> alignedStatus; // 记录alignend shape和output shape之间关系
    /* 推导align shape和对应切轴和合轴操作 */
    if (DerivationAlignShape(inShape, outShape, inStatus, alignedShape, alignedStatus) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "Op: %d derivation alignedShape failed, inShape%s, outShape%s, inTile%s",
                         op->GetOpMagic(), GetStr(inShape).c_str(), GetStr(outShape).c_str(),
                         GetStr(inTileShape).c_str());
        return WARNING;
    }

    /* 推导align shape对应的tile shape */
    InitShapeStatusStride(alignedShape, alignedStatus);
    if (DerivationAlignShapeTile(inStatus, alignedStatus) != SUCCESS) {
        APASS_LOG_WARN_F(
            Elements::Operation, "Op: %d derivation aligned tileshape failed, inShape%s, outShape%s, inTile%s",
            op->GetOpMagic(), GetStr(inShape).c_str(), GetStr(outShape).c_str(), GetStr(inTileShape).c_str());
        return WARNING;
    }

    /* 推导输出Tensor的tile shape */
    std::vector<int64_t> newTileShape(outShape.size(), 1);
    if (DerivationOutShapeTileWithAlign(alignedStatus, outShape, newTileShape) != SUCCESS) {
        APASS_LOG_WARN_F(
            Elements::Operation,
            "Op: %d derivation out tileshape with align failed, inShape%s, outShape%s, inTile%s, alignedShape%s",
            op->GetOpMagic(), GetStr(inShape).c_str(), GetStr(outShape).c_str(), GetStr(inTileShape).c_str(),
            GetStr(alignedShape).c_str());
        return WARNING;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Op: %d, inShape%s, alignShape%s, outShape%s, inTile%s, alignTile%s, outTile%s", op->GetOpMagic(),
                     GetStr(inShape).c_str(), GetStr(alignedShape).c_str(), GetStr(outShape).c_str(),
                     GetStr(inTileShape).c_str(), GetTileStr(alignedStatus).c_str(), GetStr(newTileShape).c_str());

    /* 检查输入输出切分tile shape */
    if (CheckTileShape(inShape, outShape, inTileShape, newTileShape) != SUCCESS) {
        APASS_LOG_WARN_F(
            Elements::Operation,
            "Op: %d check tileshape failed, inShape%s, alignShape%s, outShape%s, inTile%s, alignTile%s, outTile%s",
            op->GetOpMagic(), GetStr(inShape).c_str(), GetStr(alignedShape).c_str(), GetStr(outShape).c_str(),
            GetTileStr(inStatus).c_str(), GetTileStr(alignedStatus).c_str(), GetStr(newTileShape).c_str());
        return WARNING;
    }

    outTileShape = newTileShape;
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
