/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_vec_tile_inference.cpp
 * \brief Infer VecTileShape for fmap/weight/out Transdata from ConvTileShape.
 */

#include "conv_vec_tile_inference.h"
#include "interface/operation/operation.h"
#include "interface/utils/common.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

int64_t ConvAlignB(int64_t a, int64_t b);

namespace {
constexpr int64_t NCHW_C_IDX = 1;
constexpr int64_t NCHW_H_IDX = 2;
constexpr int64_t NCHW_W_IDX = 3;
constexpr int64_t NCDHW_D_IDX = 2;
constexpr uint32_t PAD_STRIDE_H = 0;
constexpr uint32_t PAD_STRIDE_W = 1;
constexpr uint32_t PAD_STRIDE_D = 2;
constexpr int64_t MKN_N_VAL = 16;

inline size_t GetUbLimit()
{
    size_t ubSize = Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB);
    return ubSize / 4;
}

inline int64_t VecTileBytes(const std::vector<int64_t>& tile, int64_t dtypeSize)
{
    int64_t total = dtypeSize;
    for (int64_t v : tile) {
        total *= v;
    }
    return total;
}

inline void GrowDim(std::vector<int64_t>& tile, size_t dimIdx, int64_t step, int64_t maxVal, int64_t dtypeSize,
                    size_t ubLimit)
{
    while (tile[dimIdx] + step <= maxVal) {
        std::vector<int64_t> tmp = tile;
        tmp[dimIdx] += step;
        if (VecTileBytes(tmp, dtypeSize) > static_cast<int64_t>(ubLimit)) {
            break;
        }
        tile = tmp;
    }
}

struct ConvInferContext {
    Conv::TileL1Info tileL1Info;
    Conv::TileL0Info tileL0Info;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> oriFmapShape;
    std::vector<int64_t> oriWeightShape;
    int64_t groups{1};
    bool isConv3D{false};
    bool isConv1D{false};
};

ConvInferContext ExtractContext(const Operation& convOp)
{
    ConvInferContext ctx;
    const auto& convTile = convOp.GetTileShape().GetConvTile();
    ctx.tileL1Info = convTile.tileL1Info;
    ctx.tileL0Info = convTile.tileL0Info;
    ctx.isConv3D = convOp.HasAttr("op_attr_is_conv3d") ? convOp.GetBoolAttribute("op_attr_is_conv3d") : false;
    ctx.strides = convOp.HasAttr("op_attr_strides") ?
                      convOp.GetVectorIntAttribute("op_attr_strides") :
                      (ctx.isConv3D ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{1, 1});
    ctx.dilations = convOp.HasAttr("op_attr_dilations") ?
                        convOp.GetVectorIntAttribute("op_attr_dilations") :
                        (ctx.isConv3D ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{1, 1});
    ctx.groups = convOp.HasAttr("op_attr_groups") ? convOp.GetIntAttribute("op_attr_groups") : 1;
    ctx.oriFmapShape = convOp.GetVectorIntAttribute("op_attr_ori_fmap_shape");
    ctx.oriWeightShape = convOp.GetVectorIntAttribute("op_attr_ori_weight_shape");
    ctx.isConv1D = (ctx.oriFmapShape.size() == 3);
    return ctx;
}

VecTile InferFmapVecTile(const ConvInferContext& ctx, int64_t dtypeSize, size_t ubLimit)
{
    int64_t c0 = ALIGN_SIZE_32 / dtypeSize;
    bool isConv3D = ctx.isConv3D;
    bool isConv1D = ctx.isConv1D;

    int64_t kh = isConv1D ? 1 : ctx.oriWeightShape[NCHW_H_IDX];
    int64_t kw = ctx.oriWeightShape[isConv1D ? NCHW_H_IDX : NCHW_W_IDX];
    int64_t strideH = isConv1D ? 1 : ctx.strides[PAD_STRIDE_H];
    int64_t strideW = ctx.strides[isConv1D ? PAD_STRIDE_H : PAD_STRIDE_W];
    int64_t dilationH = isConv1D ? 1 : ctx.dilations[PAD_STRIDE_H];
    int64_t dilationW = ctx.dilations[isConv1D ? PAD_STRIDE_H : PAD_STRIDE_W];

    int64_t tileHout = ctx.tileL1Info.tileHout;
    int64_t tileWout = ctx.tileL1Info.tileWout;
    int64_t hinL1 = (tileHout - 1) * strideH + (kh - 1) * dilationH + 1;
    int64_t winL1 = ConvAlignB((tileWout - 1) * strideW + (kw - 1) * dilationW + 1, c0);

    int64_t cinPerGroup = ctx.oriFmapShape[NCHW_C_IDX] / ctx.groups;
    int64_t cinMax = std::min(cinPerGroup, ctx.tileL1Info.tileCinFmap);

    VecTile vecTile;
    if (isConv3D) {
        int64_t kd = ctx.oriWeightShape[NCDHW_D_IDX];
        int64_t strideD = ctx.strides[PAD_STRIDE_D];
        int64_t dilationD = ctx.dilations[PAD_STRIDE_D];
        int64_t tileDout = 1;
        int64_t dinL1 = (tileDout - 1) * strideD + (kd - 1) * dilationD + 1;
        vecTile.tile = {1, c0, 1, 1, c0};
        GrowDim(vecTile.tile, 4, c0, winL1, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 3, 1, hinL1, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 2, 1, dinL1, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 1, c0, cinMax, dtypeSize, ubLimit);
    } else {
        vecTile.tile = {1, c0, 1, c0};
        GrowDim(vecTile.tile, 3, c0, winL1, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 2, 1, hinL1, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 1, c0, cinMax, dtypeSize, ubLimit);
    }
    return vecTile;
}

VecTile InferWeightVecTile(const ConvInferContext& ctx, int64_t dtypeSize, size_t ubLimit)
{
    int64_t c0 = ALIGN_SIZE_32 / dtypeSize;
    int64_t n0 = MKN_N_VAL;
    bool isConv3D = ctx.isConv3D;
    bool isConv1D = ctx.isConv1D;

    int64_t kw = ctx.oriWeightShape[isConv1D ? NCHW_H_IDX : NCHW_W_IDX];
    int64_t coutPerGroup = ctx.oriWeightShape[0] / ctx.groups;
    int64_t nMax = std::min(coutPerGroup, ctx.tileL1Info.tileN);

    VecTile vecTile;
    if (isConv3D) {
        vecTile.tile = {n0, c0, 1, 1, c0};
        GrowDim(vecTile.tile, 4, c0, kw, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 0, n0, nMax, dtypeSize, ubLimit);
    } else {
        vecTile.tile = {n0, c0, 1, c0};
        GrowDim(vecTile.tile, 3, c0, kw, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 0, n0, nMax, dtypeSize, ubLimit);
    }
    return vecTile;
}

VecTile InferOutVecTile(const ConvInferContext& ctx, int64_t dtypeSize, size_t ubLimit)
{
    int64_t c0 = ALIGN_SIZE_32 / dtypeSize;
    bool isConv3D = ctx.isConv3D;
    int64_t tileH = ctx.tileL0Info.tileH;
    int64_t tileW = ctx.tileL0Info.tileW;
    int64_t tileN1 = ctx.tileL0Info.tileN / c0;

    VecTile vecTile;
    if (isConv3D) {
        vecTile.tile = {1, 1, 1, 1, 1, c0};
        GrowDim(vecTile.tile, 4, 1, tileW, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 3, 1, tileH, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 2, 1, tileN1, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 1, 1, 1, dtypeSize, ubLimit);
    } else {
        vecTile.tile = {1, 1, 1, 1, c0};
        GrowDim(vecTile.tile, 3, 1, tileW, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 2, 1, tileH, dtypeSize, ubLimit);
        GrowDim(vecTile.tile, 1, 1, tileN1, dtypeSize, ubLimit);
    }
    return vecTile;
}
} // namespace

ConvVecTileShapes InferAllVecTiles(const ConvInferContext& ctx, DataType dtype)
{
    ConvVecTileShapes result;
    if (!ctx.tileL1Info.tileWout || ctx.oriFmapShape.empty() || ctx.oriWeightShape.empty()) {
        return result;
    }

    int64_t dtypeSize = static_cast<int64_t>(BytesOf(dtype));
    size_t ubLimit = GetUbLimit();

    result.fmapVecTile = InferFmapVecTile(ctx, dtypeSize, ubLimit);
    result.weightVecTile = InferWeightVecTile(ctx, dtypeSize, ubLimit);
    result.outVecTile = InferOutVecTile(ctx, dtypeSize, ubLimit);
    return result;
}

ConvVecTileShapes InferConvVecTileShapes(const Operation& convOp, DataType dtype)
{
    return InferAllVecTiles(ExtractContext(convOp), dtype);
}

ConvVecTileShapes InferConvVecTileShapes(const ConvTile& convTile, DataType dtype,
                                         const std::vector<int64_t>& oriFmapShape,
                                         const std::vector<int64_t>& oriWeightShape, bool isConv3D, bool isConv1D,
                                         int64_t groups)
{
    ConvInferContext ctx;
    ctx.tileL1Info = convTile.tileL1Info;
    ctx.tileL0Info = convTile.tileL0Info;
    ctx.strides = isConv3D ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{1, 1};
    ctx.dilations = isConv3D ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{1, 1};
    ctx.oriFmapShape = oriFmapShape;
    ctx.oriWeightShape = oriWeightShape;
    ctx.groups = groups;
    ctx.isConv3D = isConv3D;
    ctx.isConv1D = isConv1D;
    return InferAllVecTiles(ctx, dtype);
}

VecTile SelectConvVecTile(const ConvVecTileShapes& vecTiles, TileOpFormat targetFormat)
{
    if (targetFormat == TileOpFormat::TILEOP_ND) {
        CONV_LOGI("Transdata vector tile shape for output after infer is: %s",
                  IntVecToStr(vecTiles.outVecTile.tile).c_str());
        return vecTiles.outVecTile;
    } else if (targetFormat == TileOpFormat::TILEOP_NC1HWC0 || targetFormat == TileOpFormat::TILEOP_NDC1HWC0) {
        CONV_LOGI("Transdata vector tile shape for fmap after infer is: %s",
                  IntVecToStr(vecTiles.fmapVecTile.tile).c_str());
        return vecTiles.fmapVecTile;
    } else {
        CONV_LOGI("Transdata vector tile shape for weight after infer is: %s",
                  IntVecToStr(vecTiles.weightVecTile.tile).c_str());
        return vecTiles.weightVecTile;
    }
}

VecTile GetReshapeVecTile(const VecTile& srcVecTile, bool isConv1D)
{
    VecTile result = srcVecTile;
    if (result.tile.size() == 5) {
        result.tile[1] *= result.tile[4];
        result.tile.pop_back();
    }
    if (isConv1D && result.tile.size() == 4) {
        result.tile.erase(result.tile.begin() + 2);
    }
    return result;
}

} // namespace Conv
} // namespace tile_fwk
} // namespace npu
