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
 * \file quantization.cpp
 * \brief Quantization operation implementation for INT8 symmetric and asymmetric quantization
 */

#include "interface/utils/operator_tracer.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/vector/tensor_transformation.h"

#include <unordered_set>

namespace npu::tile_fwk {

namespace {
constexpr size_t QUANT_MX_MIN_RANK = 1;
constexpr size_t QUANT_MX_MAX_RANK = 4;
constexpr int64_t QUANT_MX_GROUP_COLS = 32;
constexpr int64_t QUANT_MX_SCALE_GROUP_COLS = 64;
constexpr int64_t QUANT_MX_SCALE_PAIR_SIZE = 2;
constexpr int64_t QUANT_MX_TILE_ALIGN_BYTES = 256;
const std::unordered_set<DataType> QUANT_MX_SUPPORTED_INPUT_TYPES = {DataType::DT_FP16, DataType::DT_BF16,
                                                                     DataType::DT_FP32};
const std::unordered_set<DataType> QUANT_MX_SUPPORTED_OUTPUT_TYPES = {DataType::DT_FP8E4M3, DataType::DT_FP4_E2M1X2};
const std::vector<NPUArch> QUANT_MX_SUPPORTED_ARCHITECTURES = {NPUArch::DAV_3510};

int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, divisor != 0) << "CeilDiv divisor must not be zero.";
    return (dividend + divisor - 1) / divisor;
}

void CheckQuantMXDtype(DataType quantDtype)
{
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          QUANT_MX_SUPPORTED_OUTPUT_TYPES.find(quantDtype) != QUANT_MX_SUPPORTED_OUTPUT_TYPES.end())
        << "QuantMX currently only supports DT_FP8E4M3 and DT_FP4_E2M1X2 output. Current quant dtype: "
        << DataType2String(quantDtype);
}

void CheckQuantMXDtypeCombination(DataType inputDtype, DataType quantDtype)
{
    if (quantDtype == DataType::DT_FP8E4M3) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
              inputDtype == DataType::DT_FP32 || inputDtype == DataType::DT_FP16 || inputDtype == DataType::DT_BF16)
            << "QuantMX DT_FP8E4M3 output only supports DT_FP32, DT_FP16, and DT_BF16 input.";
        return;
    }
    if (quantDtype == DataType::DT_FP4_E2M1X2) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
              inputDtype == DataType::DT_FP16 || inputDtype == DataType::DT_BF16)
            << "QuantMX DT_FP4_E2M1X2 output only supports DT_FP16 and DT_BF16 input.";
    }
}

void CheckQuantMXMode(DequantScaleRoundingMode mode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          mode == DequantScaleRoundingMode::ROUND_DOWN || mode == DequantScaleRoundingMode::ROUND_UP)
        << "QuantMX currently only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes.";
}

void CheckQuantMXPerformanceMode(int64_t performanceMode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, performanceMode == 0 || performanceMode == 1)
        << "QuantMX performance mode must be 0 or 1. Current performance mode: " << performanceMode;
}

DequantScaleRoundingMode GetQuantMXMode(const Operation& op)
{
    int64_t modeValue = 0;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantMode, modeValue))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantMode;
    return static_cast<DequantScaleRoundingMode>(modeValue);
}

int64_t GetQuantMXAxis(const Operation& op)
{
    int64_t axis = 0;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantAxis, axis))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantAxis;
    return axis;
}

int64_t GetQuantMXPerformanceMode(const Operation& op)
{
    int64_t performanceMode = 0;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantPerformanceMode, performanceMode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantPerformanceMode;
    return performanceMode;
}

int64_t NormalizeQuantMXAxis(int64_t axis, size_t rank)
{
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < static_cast<int64_t>(rank))
        << "QuantMX axis is out of range. Current axis: " << axis << ", input rank: " << rank;
    return axis;
}

void CheckQuantMXAxis(int64_t axis, size_t rank)
{
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, rank);
    const int64_t lastAxis = static_cast<int64_t>(rank) - 1;
    const int64_t dnAxis = static_cast<int64_t>(rank) - 2;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, normalizedAxis == lastAxis || (rank >= 2 && normalizedAxis == dnAxis))
        << "QuantMX currently only supports the last axis and second-last axis. Current axis: " << axis
        << ", input rank: " << rank;
}

bool IsQuantMXDnAxis(int64_t normalizedAxis, size_t rank)
{
    return rank >= 2 && normalizedAxis == static_cast<int64_t>(rank) - 2;
}

void CheckQuantMXInput(const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode, int64_t axis,
                       bool performanceMode)
{
    const auto inputDtype = input.GetDataType();
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          QUANT_MX_SUPPORTED_INPUT_TYPES.find(inputDtype) != QUANT_MX_SUPPORTED_INPUT_TYPES.end())
        << "QuantMX currently only supports DT_FP16, DT_BF16, and DT_FP32 input.";
    CheckQuantMXDtype(quantDtype);
    CheckQuantMXDtypeCombination(inputDtype, quantDtype);
    CheckQuantMXMode(mode);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "QuantMX only supports TILEOP_ND input.";
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
          QUANT_MX_MIN_RANK <= input.GetShape().size() && input.GetShape().size() <= QUANT_MX_MAX_RANK)
        << "QuantMX only supports 1D to 4D input.";
    const auto& inputShape = input.GetShape();
    CheckQuantMXAxis(axis, inputShape.size());
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, inputShape.size());
    if (IsQuantMXDnAxis(normalizedAxis, inputShape.size())) {
        const int64_t dnDim = inputShape[normalizedAxis];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, dnDim % QUANT_MX_SCALE_GROUP_COLS == 0)
            << "QuantMX axis=-2 requires the second-last dimension to be 64-aligned. Current dim: " << dnDim;
        if (quantDtype == DT_FP4_E2M1X2) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, inputShape.back() % QUANT_MX_SCALE_GROUP_COLS == 0)
                << "QuantMX FP4 axis=-2 requires view shape's last dim to be 64-aligned. Current dim: "
                << inputShape.back();
        }
        return;
    }
    if (!performanceMode) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, inputShape.back() % QUANT_MX_SCALE_GROUP_COLS == 0)
            << "QuantMX non-performance mode requires input last dim to be a multiple of 64. Current last dim: "
            << inputShape.back();
    }
    if (performanceMode) {
        const int64_t lastDimBytes = inputShape.back() * BytesOf(inputDtype);
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
            << "QuantMX performance mode requires view shape's last dim to be 256-byte aligned. Current last dim "
               "bytes: "
            << lastDimBytes;
    }
}

void CheckQuantMXPerformanceTileShape(const LogicalTensorPtr& input, const VecTile& vecTile, int64_t axis,
                                      int64_t performanceMode)
{
    if (performanceMode == 0) {
        return;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "QuantMX performance mode tile shape rank must match input rank.";
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, input->GetShape().size());
    if (IsQuantMXDnAxis(normalizedAxis, input->GetShape().size())) {
        return;
    }
    const int64_t lastTileDim = vecTile[vecTile.size() - 1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastTileDim == input->GetShape().back())
        << "QuantMX performance mode requires tile shape last dim to be the same as input last dim. Current tile "
           "last dim: "
        << lastTileDim << ", input last dim: " << input->GetShape().back();
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedShape(const std::vector<int64_t>& inputShape)
{
    if (inputShape.size() == 1) {
        return {CeilDiv(inputShape[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedShape;
    groupedShape.reserve(inputShape.size() - 1);
    for (size_t i = 0; i + 2 < inputShape.size(); ++i) {
        groupedShape.push_back(inputShape[i]);
    }
    groupedShape.push_back(inputShape[inputShape.size() - 2] * CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS));
    return groupedShape;
}

std::vector<int64_t> BuildQuantMXScalingShape(const std::vector<int64_t>& groupedShape, DataType inputDtype)
{
    auto scalingShape = groupedShape;
    if (inputDtype == DataType::DT_FP32) {
        scalingShape.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    }
    return scalingShape;
}

std::vector<int64_t> BuildQuantMXExpStorageShape(const std::vector<int64_t>& inputShape)
{
    auto expStorageShape = inputShape;
    expStorageShape.back() = CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS);
    return expStorageShape;
}

std::vector<int64_t> BuildQuantMXDnExpShape(const std::vector<int64_t>& inputShape)
{
    auto expShape = inputShape;
    expShape[expShape.size() - 2] /= QUANT_MX_SCALE_GROUP_COLS;
    expShape.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    return expShape;
}

std::vector<int64_t> BuildQuantMXDnScratchShape(const std::vector<int64_t>& inputShape)
{
    auto scratchShape = inputShape;
    scratchShape[scratchShape.size() - 2] /= QUANT_MX_GROUP_COLS;
    return scratchShape;
}

std::vector<int64_t> BuildQuantMXPerformanceVecTile(const std::vector<int64_t>& inputVecTile)
{
    if (inputVecTile.size() == 1) {
        return {CeilDiv(inputVecTile[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedVecTile;
    groupedVecTile.reserve(inputVecTile.size() - 1);
    for (size_t i = 0; i + 2 < inputVecTile.size(); ++i) {
        groupedVecTile.push_back(inputVecTile[i]);
    }
    groupedVecTile.push_back(inputVecTile[inputVecTile.size() - 2] * CeilDiv(inputVecTile.back(), QUANT_MX_GROUP_COLS));
    return groupedVecTile;
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedOffset(const std::vector<int64_t>& inputOffset,
                                                          const std::vector<int64_t>& inputShape,
                                                          const std::vector<int64_t>& inputTileShape)
{
    if (inputOffset.size() == 1) {
        return {inputOffset[0] / QUANT_MX_GROUP_COLS};
    }

    std::vector<int64_t> groupedOffset;
    groupedOffset.reserve(inputOffset.size() - 1);
    for (size_t i = 0; i + 2 < inputOffset.size(); ++i) {
        groupedOffset.push_back(inputOffset[i]);
    }
    const int64_t groupCols = CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS);
    const int64_t tileRows = inputTileShape[inputTileShape.size() - 2];
    groupedOffset.push_back(inputOffset[inputOffset.size() - 2] * groupCols +
                            tileRows * (inputOffset.back() / QUANT_MX_GROUP_COLS));
    return groupedOffset;
}

std::vector<int64_t> BuildQuantMXScalingOffset(const std::vector<int64_t>& groupedOffset, DataType inputDtype)
{
    auto scalingOffset = groupedOffset;
    if (inputDtype == DataType::DT_FP32) {
        scalingOffset.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    }
    return scalingOffset;
}

std::vector<SymbolicScalar> BuildQuantMXPerformanceGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    if (inputValidShape.size() == 1) {
        return {(inputValidShape[0] + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS};
    }

    std::vector<SymbolicScalar> groupedValidShape;
    groupedValidShape.reserve(inputValidShape.size() - 1);
    for (size_t i = 0; i + 2 < inputValidShape.size(); ++i) {
        groupedValidShape.push_back(inputValidShape[i]);
    }
    groupedValidShape.push_back(inputValidShape[inputValidShape.size() - 2] *
                                ((inputValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS));
    return groupedValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXScalingValidShape(const std::vector<SymbolicScalar>& groupedValidShape,
                                                          DataType inputDtype)
{
    auto scalingValidShape = groupedValidShape;
    if (inputDtype == DataType::DT_FP32) {
        scalingValidShape.back() = scalingValidShape.back() * QUANT_MX_SCALE_PAIR_SIZE;
    }
    return scalingValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXExpStorageValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto expStorageValidShape = inputValidShape;
    expStorageValidShape.back() = (expStorageValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS;
    return expStorageValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXDnExpValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto expValidShape = inputValidShape;
    expValidShape[expValidShape.size() - 2] = expValidShape[expValidShape.size() - 2] / QUANT_MX_SCALE_GROUP_COLS;
    expValidShape.back() = expValidShape.back() * QUANT_MX_SCALE_PAIR_SIZE;
    return expValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXDnScratchValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scratchValidShape = inputValidShape;
    scratchValidShape[scratchValidShape.size() - 2] = scratchValidShape[scratchValidShape.size() - 2] /
                                                      QUANT_MX_GROUP_COLS;
    return scratchValidShape;
}

std::vector<int64_t> BuildQuantMXScaleShape(const std::vector<int64_t>& inputShape)
{
    auto scaleShape = inputShape;
    scaleShape.back() = CeilDiv(scaleShape.back(), QUANT_MX_SCALE_GROUP_COLS);
    scaleShape.push_back(QUANT_MX_SCALE_PAIR_SIZE);
    return scaleShape;
}

std::vector<int64_t> BuildQuantMXDnScaleShape(const std::vector<int64_t>& inputShape)
{
    auto scaleShape = inputShape;
    scaleShape[scaleShape.size() - 2] /= QUANT_MX_SCALE_GROUP_COLS;
    scaleShape.push_back(QUANT_MX_SCALE_PAIR_SIZE);
    return scaleShape;
}

std::vector<SymbolicScalar> BuildQuantMXScaleValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scaleValidShape = inputValidShape;
    scaleValidShape.back() = (scaleValidShape.back() + QUANT_MX_SCALE_GROUP_COLS - 1) / QUANT_MX_SCALE_GROUP_COLS;
    scaleValidShape.push_back(SymbolicScalar(QUANT_MX_SCALE_PAIR_SIZE));
    return scaleValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXDnScaleValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scaleValidShape = inputValidShape;
    scaleValidShape[scaleValidShape.size() - 2] = scaleValidShape[scaleValidShape.size() - 2] /
                                                  QUANT_MX_SCALE_GROUP_COLS;
    scaleValidShape.push_back(SymbolicScalar(QUANT_MX_SCALE_PAIR_SIZE));
    return scaleValidShape;
}

void CheckQuantMXTileShape(const LogicalTensorPtr& input, const VecTile& vecTile, DataType quantDtype, int64_t axis,
                           int64_t performanceMode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "QuantMX tile shape rank must match input rank.";
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, input->GetShape().size());
    if (IsQuantMXDnAxis(normalizedAxis, input->GetShape().size())) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[normalizedAxis] > 0)
            << "QuantMX axis=-2 tile shape second-last dim must be positive.";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[normalizedAxis] % QUANT_MX_SCALE_GROUP_COLS == 0)
            << "QuantMX axis=-2 tile shape second-last dim must be 64-aligned. Current dim: "
            << vecTile[normalizedAxis];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] > 0)
            << "QuantMX axis=-2 tile shape last dim must be positive.";
        if (quantDtype == DT_FP4_E2M1X2) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] % QUANT_MX_SCALE_GROUP_COLS == 0)
                << "QuantMX FP4 axis=-2 requires tile shape's last dim to be 64-aligned. Current dim: "
                << vecTile[vecTile.size() - 1];
        }
        return;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] > 0)
        << "QuantMX tile shape last dim must be positive.";

    if (performanceMode == 0) {
        return;
    }
    const int64_t lastDimBytes = vecTile[vecTile.size() - 1] * BytesOf(input->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX performance mode requires tile shape's last dim to be 256-byte aligned. Current last dim bytes: "
        << lastDimBytes;
}

struct QuantMXTileContext {
    Function& function;
    Input& input;
    const LogicalTensorPtr& dst;
    const LogicalTensorPtr& exp;
    const LogicalTensorPtr& maxScratch;
    const LogicalTensorPtr& scalingScratch;
    DequantScaleRoundingMode mode;
    int64_t axis;
    int64_t performanceMode;
};

struct QuantMXTileParams {
    std::vector<int64_t> expShape;
    std::vector<int64_t> expOffset;
    std::vector<int64_t> groupedShape;
    std::vector<int64_t> groupedOffset;
    std::vector<int64_t> scalingShape;
    std::vector<int64_t> scalingOffset;
};

void CheckQuantMXTileAlignment(const QuantMXTileContext& ctx)
{
    const int64_t normalizedAxis = NormalizeQuantMXAxis(ctx.axis, ctx.input.tensor.GetShape().size());
    if (IsQuantMXDnAxis(normalizedAxis, ctx.input.tensor.GetShape().size())) {
        return;
    }
    if (ctx.performanceMode == 0) {
        return;
    }
    const int64_t lastDimBytes = ctx.input.tileInfo.shape.back() * BytesOf(ctx.input.tensor.GetDataType());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX performance mode requires tile width to be 256-byte aligned. Current last dim bytes: "
        << lastDimBytes;
}

QuantMXTileParams BuildQuantMXTileParams(const Input& input, int64_t performanceMode)
{
    QuantMXTileParams params;
    params.groupedShape = BuildQuantMXPerformanceGroupedShape(input.tileInfo.shape);
    params.groupedOffset = BuildQuantMXPerformanceGroupedOffset(input.tileInfo.offset, input.tensor.GetShape(),
                                                                input.tileInfo.shape);
    params.scalingShape = BuildQuantMXScalingShape(params.groupedShape, input.tensor.GetDataType());
    params.scalingOffset = BuildQuantMXScalingOffset(params.groupedOffset, input.tensor.GetDataType());
    if (performanceMode == 0) {
        params.expShape = input.tileInfo.shape;
        params.expShape.back() = CeilDiv(params.expShape.back(), QUANT_MX_GROUP_COLS);
        params.expOffset = input.tileInfo.offset;
        params.expOffset.back() /= QUANT_MX_GROUP_COLS;
        return params;
    }
    params.expShape = params.groupedShape;
    params.expOffset = params.groupedOffset;
    return params;
}

QuantMXTileParams BuildQuantMXDnTileParams(const Input& input)
{
    QuantMXTileParams params;
    params.expShape = input.tileInfo.shape;
    params.expShape[params.expShape.size() - 2] /= QUANT_MX_SCALE_GROUP_COLS;
    params.expShape.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    params.expOffset = input.tileInfo.offset;
    params.expOffset[params.expOffset.size() - 2] /= QUANT_MX_SCALE_GROUP_COLS;
    params.expOffset.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    params.groupedShape = input.tileInfo.shape;
    params.groupedShape[params.groupedShape.size() - 2] /= QUANT_MX_GROUP_COLS;
    params.groupedOffset = input.tileInfo.offset;
    params.groupedOffset[params.groupedOffset.size() - 2] /= QUANT_MX_GROUP_COLS;
    params.scalingShape = params.groupedShape;
    params.scalingOffset = params.groupedOffset;
    return params;
}

void EmitQuantMXTile(const QuantMXTileContext& ctx, const QuantMXTileParams& params)
{
    auto storage = ctx.input.tensor.GetStorage();
    auto srcTile = storage->View(ctx.function, ctx.input.tileInfo.shape, ctx.input.tileInfo.offset);
    auto dstTile = ctx.dst->View(ctx.function, ctx.input.tileInfo.shape, ctx.input.tileInfo.offset);
    auto expTile = ctx.exp->View(ctx.function, params.expShape, params.expOffset);
    auto maxTile = ctx.maxScratch->View(ctx.function, params.groupedShape, params.groupedOffset);
    auto scalingTile = ctx.scalingScratch->View(ctx.function, params.scalingShape, params.scalingOffset);
    auto& tiledOp = ctx.function.AddOperation(Opcode::OP_QUANT_MX, {srcTile}, {dstTile, expTile, maxTile, scalingTile});
    tiledOp.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(ctx.mode));
    tiledOp.SetAttribute(OpAttributeKey::mxQuantAxis, ctx.axis);
    tiledOp.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, ctx.performanceMode);
}

void TiledQuantMXOperationImpl(const QuantMXTileContext& ctx, const TileShape& tileShape, size_t cur)
{
    if (cur == ctx.input.tensor.GetShape().size()) {
        CheckQuantMXTileAlignment(ctx);
        const int64_t normalizedAxis = NormalizeQuantMXAxis(ctx.axis, ctx.input.tensor.GetShape().size());
        const auto params = IsQuantMXDnAxis(normalizedAxis, ctx.input.tensor.GetShape().size()) ?
                                BuildQuantMXDnTileParams(ctx.input) :
                                BuildQuantMXTileParams(ctx.input, ctx.performanceMode);
        EmitQuantMXTile(ctx, params);
        return;
    }

    const auto& vecTile = tileShape.GetVecTile();
    int64_t step = std::max<int64_t>(1, std::min<int64_t>(vecTile[cur], ctx.input.tensor.GetShape()[cur]));
    for (int64_t i = 0; i < ctx.input.tensor.GetShape()[cur]; i += step) {
        ctx.input.tileInfo.shape[cur] = std::min(ctx.input.tensor.GetShape()[cur] - i, step);
        ctx.input.tileInfo.offset[cur] = i;
        TiledQuantMXOperationImpl(ctx, tileShape, cur + 1);
    }
}

void TiledQuantMXOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                           const LogicalTensorPtr& dst, const LogicalTensorPtr& exp, const LogicalTensorPtr& maxScratch,
                           const LogicalTensorPtr& scalingScratch, DequantScaleRoundingMode mode, int64_t axis,
                           int64_t performanceMode)
{
    QuantMXTileContext ctx{function, input, dst, exp, maxScratch, scalingScratch, mode, axis, performanceMode};
    TiledQuantMXOperationImpl(ctx, tileShape, cur);
}

void QuantMXTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                     const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "QuantMX expects 1 input tensor.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 4) << "QuantMX expects 4 output tensors.";

    const auto& src = iOperand[0];
    const auto& dst = oOperand[0];
    const auto& exp = oOperand[1];
    const auto& maxScratch = oOperand[2];
    const auto& scalingScratch = oOperand[3];
    CheckQuantMXDtype(dst->Datatype());
    auto mode = GetQuantMXMode(op);
    auto axis = GetQuantMXAxis(op);
    auto performanceMode = GetQuantMXPerformanceMode(op);
    CheckQuantMXMode(mode);
    CheckQuantMXPerformanceMode(performanceMode);
    CheckQuantMXAxis(axis, src->GetShape().size());
    CheckQuantMXTileShape(src, tileShape.GetVecTile(), dst->Datatype(), axis, performanceMode);
    CheckQuantMXPerformanceTileShape(src, tileShape.GetVecTile(), axis, performanceMode);
    TileInfo inputTileInfo(src->shape.size(), src->offset.size());
    auto input = Input{Tensor(src), inputTileInfo};
    TiledQuantMXOperation(function, tileShape, 0, input, dst, exp, maxScratch, scalingScratch, mode, axis,
                          performanceMode);
}
} // namespace

// =============================================================================
// Symmetric Quantization (FP32 -> INT8)
// =============================================================================

void TiledQuantizeSymmetric(Function& function, const TileShape& tileShape, size_t cur, Input& srcInput,
                            Input& scaleInput, Input& dstInput, int64_t axis, uint32_t workspaceSize)
{
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape,
                                                              scaleInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        Operation* op = nullptr;
        if (workspaceSize == 0) {
            op = &function.AddOperation(Opcode::OP_QUANTIZE_SYM, {srcTile, scaleTile}, {dstTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_INT32,
                                                                         std::vector<int64_t>{workspaceSize});
            op = &function.AddOperation(Opcode::OP_QUANTIZE_SYM, {srcTile, scaleTile}, {dstTile, workspace});
        }
        op->SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            // If scale's dimension is 1 (broadcast dimension), use modulo which gives 0
            // Otherwise use the same index as src/dst
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        TiledQuantizeSymmetric(function, tileShape, cur + 1, srcInput, scaleInput, dstInput, axis, workspaceSize);
    }
}

void TiledQuantizeSymmetric(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
                            const LogicalTensorPtr& scale, const LogicalTensorPtr& dst, int64_t axis,
                            uint32_t workspaceSize)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == src->offset.size())
        << "Source shape size and offset size should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dst->shape.size() == dst->offset.size())
        << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeSymmetric(function, tileShape, 0, srcInput, scaleInput, dstInput, axis, workspaceSize);
}

// =============================================================================
// Asymmetric Quantization (FP32 -> UINT8)
// =============================================================================

void TiledQuantizeAsymmetric(Function& function, const TileShape& tileShape, size_t cur, Input& srcInput,
                             Input& scaleInput, Input& offsetInput, Input& dstInput, int64_t axis,
                             uint32_t workspaceSize)
{
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape,
                                                              scaleInput.tileInfo.offset);
        auto offsetTile = offsetInput.tensor.GetStorage()->View(function, offsetInput.tileInfo.shape,
                                                                offsetInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        Operation* op = nullptr;
        if (workspaceSize == 0) {
            op = &function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {srcTile, scaleTile, offsetTile}, {dstTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_INT32,
                                                                         std::vector<int64_t>{workspaceSize});
            op = &function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {srcTile, scaleTile, offsetTile},
                                        {dstTile, workspace});
        }
        op->SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        // Update offset tile info - offset has same shape as scale
        if (cur < offsetInput.tensor.GetShape().size()) {
            int64_t offsetIdx = i % offsetInput.tensor.GetShape()[cur];
            offsetInput.tileInfo.shape[cur] = std::min(offsetInput.tensor.GetShape()[cur] - offsetIdx, vecTile[cur]);
            offsetInput.tileInfo.offset[cur] = offsetIdx;
        }

        TiledQuantizeAsymmetric(function, tileShape, cur + 1, srcInput, scaleInput, offsetInput, dstInput, axis,
                                workspaceSize);
    }
}

void TiledQuantizeAsymmetric(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
                             const LogicalTensorPtr& scale, const LogicalTensorPtr& offset, const LogicalTensorPtr& dst,
                             int64_t axis, uint32_t workspaceSize)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == src->offset.size())
        << "Source shape size and offset size should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dst->shape.size() == dst->offset.size())
        << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo offsetTileInfo(offset->shape.size(), offset->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto offsetInput = Input{Tensor(offset), offsetTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeAsymmetric(function, tileShape, 0, srcInput, scaleInput, offsetInput, dstInput, axis, workspaceSize);
}

// =============================================================================
// Tensor-level Quantization Operations
// =============================================================================

LogicalTensorPtr TensorQuantizeSymmetricOperation(Function& function, const LogicalTensorPtr& src,
                                                  const LogicalTensorPtr& scale, int64_t axis)
{
    // Output is INT8 for symmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT8, src->shape, src->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_QUANTIZE_SYM, {src, scale}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    return result;
}

LogicalTensorPtr TensorQuantizeAsymmetricOperation(Function& function, const LogicalTensorPtr& src,
                                                   const LogicalTensorPtr& scale, const LogicalTensorPtr& offset,
                                                   int64_t axis)
{
    // Output is UINT8 for asymmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_UINT8, src->shape, src->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    return result;
}

// =============================================================================
// Public Quantize API
// =============================================================================

Tensor Quantize(const Tensor& input, const Tensor& scale, DataType dtype, int axis, const Tensor& zeroPoints)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Quantize");
    CheckTensorFormat(scale.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Quantize");
    if (zeroPoints.GetStorage() != nullptr) {
        CheckTensorFormat(zeroPoints.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Quantize");
    }

    // Validate input dimensions
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
          input.GetShape().size() >= SHAPE_DIM1 && input.GetShape().size() <= SHAPE_DIM5)
        << "The shape.size() only support 1~5";

    // Handle 1D input: reshape to [1, n] and process as 2D
    // scale/zeroPoints remain 1D (no reshape needed)
    bool is1DInput = (input.GetShape().size() == 1);

    // Validate: 1D input only supports axis=-1
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, !(is1DInput && axis == -2))
        << "1D input only supports axis=-1, axis=-2 is not supported";

    Tensor processedInput = input;
    Tensor originalInputStorage = input;
    VecTile originalVecTile;

    if (is1DInput) {
        // Store original VecTile and extend it for 2D processing
        originalVecTile = TileShape::Current().GetVecTile();
        if (!originalVecTile.tile.empty()) {
            VecTile extendedVecTile = originalVecTile;
            // Extend 1D tile to 2D by inserting 1 at the front
            extendedVecTile.tile.insert(extendedVecTile.tile.begin(), 1);
            TileShape::Current().SetVecTile(extendedVecTile);
        }

        // Reshape 1D input to [1, n]
        int64_t n = input.GetShape()[0];
        std::vector<int64_t> newShape = {1, n};

        // Extend validShape from 1D to 2D
        auto originalValidShape = input.GetStorage()->GetDynValidShape();
        std::vector<SymbolicScalar> extendedValidShape;
        if (!originalValidShape.empty()) {
            // originalValidShape is [n], extend to [1, n]
            extendedValidShape.push_back(SymbolicScalar(1));
            extendedValidShape.push_back(originalValidShape[0]);
        }

        processedInput = Reshape(input, newShape, extendedValidShape);
    }

    // Validate data types
    std::vector<DataType> SUPPORT_INPUT_DATATYPES = {DataType::DT_FP32};
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          std::find(SUPPORT_INPUT_DATATYPES.begin(), SUPPORT_INPUT_DATATYPES.end(), input.GetDataType()) !=
              SUPPORT_INPUT_DATATYPES.end())
        << "The input datatype is not supported";

    // Normalize axis to negative indexing
    int ndim = static_cast<int>(processedInput.GetShape().size());
    int normalizedAxis = axis;
    if (axis >= 0) {
        normalizedAxis = axis - ndim;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, normalizedAxis == -1 || normalizedAxis == -2)
        << "Only axis=-1 (per-row) and axis=-2 (per-column) are supported";

    // Determine quantization type based on zeroPoints presence
    bool isAsymmetric = (zeroPoints.GetStorage() != nullptr);

    // For axis=-2 (per-column quantization), use Transpose to swap last two dimensions
    // Strategy: transpose input and scale -> quantize with axis=-1 -> transpose output back
    if (normalizedAxis == -2) {
        // Swap last two dimensions: [..., H, W] -> [..., W, H]
        int lastDim = ndim - 1;       // -1 in positive index
        int secondLastDim = ndim - 2; // -2 in positive index

        // Transpose input: [..., H, W] -> [..., W, H]
        Tensor transposedInput = Transpose(processedInput, {secondLastDim, lastDim});
        // [TQuant] get tmp Tile for Tquant
        VecTile oriVectile = TileShape::Current().GetVecTile();
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        std::swap(tmpVectile[secondLastDim], tmpVectile[lastDim]);
        TileShape::Current().SetVecTile(tmpVectile);
        // [TQuant] get tmp validShape
        auto tmpValidShape = processedInput.GetStorage()->dynValidShape_;
        std::swap(tmpValidShape[secondLastDim], tmpValidShape[lastDim]);
        transposedInput.GetStorage()->UpdateDynValidShape(tmpValidShape);

        // init result
        Tensor quantizedResult;

        if (isAsymmetric) {
            // Asymmetric quantization with axis=-1
            CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype == DataType::DT_UINT8)
                << "Asymmetric quantization output type should be UINT8";
            quantizedResult = CALL(QuantizeAsymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
                                   transposedInput.GetStorage(), scale.GetStorage(), zeroPoints.GetStorage(), -1);
        } else {
            // Symmetric quantization with axis=-1
            CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype == DataType::DT_INT8)
                << "Symmetric quantization output type should be INT8";
            quantizedResult = CALL(QuantizeSymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
                                   transposedInput.GetStorage(), scale.GetStorage(), -1);
        }

        // [Transpose] set tmp validShape
        quantizedResult.GetStorage()->UpdateDynValidShape(tmpValidShape);
        // [Transpose] set tmp VecTile
        TileShape::Current().SetVecTile(tmpVectile);
        // output back: [..., W, H] -> [..., H, W]
        Tensor result = Transpose(quantizedResult, {secondLastDim, lastDim});
        // [Tstore] get origin ValidShape
        result.GetStorage()->UpdateDynValidShape(processedInput.GetStorage()->dynValidShape_);
        // [Tstore] get origin TileShape
        TileShape::Current().SetVecTile(oriVectile);

        // If input was 1D, reshape output back to 1D and restore original VecTile
        if (is1DInput) {
            int64_t n = originalInputStorage.GetShape()[0];
            std::vector<int64_t> originalShape = {n};
            auto originalValidShape = originalInputStorage.GetStorage()->GetDynValidShape();
            result = Reshape(result, originalShape, originalValidShape);
            // Restore original VecTile
            if (!originalVecTile.tile.empty()) {
                TileShape::Current().SetVecTile(originalVecTile);
            }
        }
        return result;
    }

    // axis=-1 case: direct quantization without transpose
    Tensor result;
    if (isAsymmetric) {
        // Asymmetric quantization: FP32 -> UINT8
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype == DataType::DT_UINT8)
            << "Asymmetric quantization output type should be UINT8";
        result = CALL(QuantizeAsymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
                      processedInput.GetStorage(), scale.GetStorage(), zeroPoints.GetStorage(), normalizedAxis);
    } else {
        // Symmetric quantization: FP32 -> INT8
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype == DataType::DT_INT8)
            << "Symmetric quantization output type should be INT8";
        result = CALL(QuantizeSymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
                      processedInput.GetStorage(), scale.GetStorage(), normalizedAxis);
    }

    // If input was 1D, reshape output back to 1D and restore original VecTile
    if (is1DInput) {
        int64_t n = originalInputStorage.GetShape()[0];
        std::vector<int64_t> originalShape = {n};
        auto originalValidShape = originalInputStorage.GetStorage()->GetDynValidShape();
        result = Reshape(result, originalShape, originalValidShape);
        // Restore original VecTile
        if (!originalVecTile.tile.empty()) {
            TileShape::Current().SetVecTile(originalVecTile);
        }
    }
    return result;
}

// =============================================================================
// Tile Function Registration
// =============================================================================

void QuantizeSymmetricOperationTileFunc(Function& function, const TileShape& tileShape,
                                        const std::vector<LogicalTensorPtr>& iOperand,
                                        const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");

    // Calculate workspace size: same size as src (with int32_t type)
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    // tmpbuf: same size as src, with int32_t type
    int64_t tmpRows = (dim >= 2) ? shape.tile[dim - 2] : 1;
    int64_t tmpCols = (dim >= 1) ? shape.tile[dim - 1] : 1;

    // tmpbuf need 32-byte alignment
    constexpr int64_t alignElements = 8; // 8 * 4 = 32 bytes
    tmpCols = (tmpCols + alignElements - 1) / alignElements * alignElements;
    tmpRows = (tmpRows + alignElements - 1) / alignElements * alignElements;

    // workspaceSize is element count, not bytes (LogicalTensor constructor takes shape)
    uint32_t workspaceSize = tmpRows * tmpCols;

    TiledQuantizeSymmetric(function, tileShape, iOperand[0], iOperand[1], oOperand[0], axis, workspaceSize);
}

void QuantizeAsymmetricOperationTileFunc(Function& function, const TileShape& tileShape,
                                         const std::vector<LogicalTensorPtr>& iOperand,
                                         const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");

    // Calculate workspace size: same size as src (with int32_t type)
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    // tmpbuf: same size as src, with int32_t type
    int64_t tmpRows = (dim >= 2) ? shape.tile[dim - 2] : 1;
    int64_t tmpCols = (dim >= 1) ? shape.tile[dim - 1] : 1;

    // tmpbuf need 32-byte alignment
    constexpr int64_t alignElements = 8; // 8 * 4 = 32 bytes
    tmpCols = (tmpCols + alignElements - 1) / alignElements * alignElements;
    tmpRows = (tmpRows + alignElements - 1) / alignElements * alignElements;

    // workspaceSize is element count, not bytes (LogicalTensor constructor takes shape)
    uint32_t workspaceSize = tmpRows * tmpCols;

    TiledQuantizeAsymmetric(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis,
                            workspaceSize);
}

REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_SYM, Opcode::OP_QUANTIZE_SYM, QuantizeSymmetricOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_ASYM, Opcode::OP_QUANTIZE_ASYM, QuantizeAsymmetricOperationTileFunc);

// =============================================================================
// Dequantization Operations (INT8/INT16 -> FP32)
// TDequant always requires 4 params: dst, src, scale, offset (symmetric: offset=0)
// =============================================================================

void TiledDequantize(Function& function, const TileShape& tileShape, size_t cur, Input& srcInput, Input& scaleInput,
                     Input& offsetInput, Input& dstInput, int64_t axis)
{
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape,
                                                              scaleInput.tileInfo.offset);
        auto offsetTile = offsetInput.tensor.GetStorage()->View(function, offsetInput.tileInfo.shape,
                                                                offsetInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        auto& op = function.AddOperation(Opcode::OP_DEQUANTIZE, {srcTile, scaleTile, offsetTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        if (cur < scaleInput.tensor.GetShape().size()) {
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        if (cur < offsetInput.tensor.GetShape().size()) {
            int64_t offsetIdx = i % offsetInput.tensor.GetShape()[cur];
            offsetInput.tileInfo.shape[cur] = std::min(offsetInput.tensor.GetShape()[cur] - offsetIdx, vecTile[cur]);
            offsetInput.tileInfo.offset[cur] = offsetIdx;
        }

        TiledDequantize(function, tileShape, cur + 1, srcInput, scaleInput, offsetInput, dstInput, axis);
    }
}

void TiledDequantize(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
                     const LogicalTensorPtr& scale, const LogicalTensorPtr& offset, const LogicalTensorPtr& dst,
                     int64_t axis)
{
    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo offsetTileInfo(offset->shape.size(), offset->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto offsetInput = Input{Tensor(offset), offsetTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledDequantize(function, tileShape, 0, srcInput, scaleInput, offsetInput, dstInput, axis);
}

LogicalTensorPtr TensorDequantizeOperation(Function& function, const LogicalTensorPtr& src,
                                           const LogicalTensorPtr& scale, const LogicalTensorPtr& offset, int64_t axis)
{
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, src->shape, src->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_DEQUANTIZE, {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    return result;
}

// Helper: create zero tensor for symmetric dequantization
static LogicalTensorPtr CreateZeroOffsetTensor(Function& function, const LogicalTensorPtr& scale)
{
    Element zeroVal(DataType::DT_FP32, (int64_t)0);

    Tensor zeroTensor = TensorFullOperation(function, zeroVal, SymbolicScalar(), DataType::DT_FP32, scale->shape,
                                            scale->GetDynValidShape());

    return zeroTensor.GetStorage();
}

// Public Dequantize API
Tensor Dequantize(const Tensor& input, const Tensor& scale, DataType otype, int axis, const Tensor& zeroPoints)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Dequantize");
    CheckTensorFormat(scale.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Dequantize");
    if (zeroPoints.GetStorage() != nullptr) {
        CheckTensorFormat(zeroPoints.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Dequantize");
    }

    // Validate input dimensions
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
          input.GetShape().size() >= SHAPE_DIM1 && input.GetShape().size() <= SHAPE_DIM5)
        << "The shape.size() only support 1~5";

    // Handle 1D input: reshape to [1, n] and process as 2D
    // scale/zeroPoints remain 1D (no reshape needed)
    bool is1DInput = (input.GetShape().size() == 1);

    // Validate: 1D input only supports axis=-1
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, !(is1DInput && axis == -2))
        << "1D input only supports axis=-1, axis=-2 is not supported";

    Tensor processedInput = input;
    Tensor originalInputStorage = input;
    VecTile originalVecTile;

    if (is1DInput) {
        // Store original VecTile and extend it for 2D processing
        originalVecTile = TileShape::Current().GetVecTile();
        if (!originalVecTile.tile.empty()) {
            VecTile extendedVecTile = originalVecTile;
            // Extend 1D tile to 2D by inserting 1 at the front
            extendedVecTile.tile.insert(extendedVecTile.tile.begin(), 1);
            TileShape::Current().SetVecTile(extendedVecTile);
        }

        // Reshape 1D input to [1, n]
        int64_t n = input.GetShape()[0];
        std::vector<int64_t> newShape = {1, n};

        // Extend validShape from 1D to 2D
        auto originalValidShape = input.GetStorage()->GetDynValidShape();
        std::vector<SymbolicScalar> extendedValidShape;
        if (!originalValidShape.empty()) {
            // originalValidShape is [n], extend to [1, n]
            extendedValidShape.push_back(SymbolicScalar(1));
            extendedValidShape.push_back(originalValidShape[0]);
        }

        processedInput = Reshape(input, newShape, extendedValidShape);
    }

    // Validate input data type: INT8 or INT16
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          input.GetDataType() == DataType::DT_INT8 || input.GetDataType() == DataType::DT_INT16)
        << "Dequantize input dtype must be INT8 or INT16, but got dtype=" << static_cast<int>(input.GetDataType());

    // Validate output type
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, otype == DataType::DT_FP32)
        << "Dequantize output type must be FP32, but got dtype=" << static_cast<int>(otype);

    // Normalize axis to negative indexing and validate
    int ndim = static_cast<int>(processedInput.GetShape().size());
    int normalizedAxis = axis;
    if (axis >= 0) {
        normalizedAxis = axis - ndim;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, normalizedAxis == -1 || normalizedAxis == -2)
        << "Dequantize axis must be -1 (per-row) or -2 (per-column), but got axis=" << axis
        << " (normalized=" << normalizedAxis << ")";

    // Determine if symmetric or asymmetric
    bool isAsymmetric = (zeroPoints.GetStorage() != nullptr);

    // For axis=-2, use Transpose (consistent with Quantize implementation)
    if (normalizedAxis == -2) {
        int lastDim = ndim - 1;
        int secondLastDim = ndim - 2;

        // Transpose input: [..., H, W] -> [..., W, H]
        Tensor transposedInput = Transpose(processedInput, {secondLastDim, lastDim});
        // [TDequant] get tmp Tile for TDequant
        VecTile oriVectile = TileShape::Current().GetVecTile();
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        std::swap(tmpVectile[secondLastDim], tmpVectile[lastDim]);
        TileShape::Current().SetVecTile(tmpVectile);
        // [TDequant] get tmp validShape
        auto tmpValidShape = processedInput.GetStorage()->dynValidShape_;
        std::swap(tmpValidShape[secondLastDim], tmpValidShape[lastDim]);
        transposedInput.GetStorage()->UpdateDynValidShape(tmpValidShape);

        // init result
        Tensor dequantizedResult;

        if (isAsymmetric) {
            // scale and zeroPoints are NOT transposed (consistent with Quantize)
            dequantizedResult = TensorDequantizeOperation(*Program::GetInstance().GetCurrentFunction(),
                                                          transposedInput.GetStorage(), scale.GetStorage(),
                                                          zeroPoints.GetStorage(), -1);
        } else {
            // Symmetric: create zero offset tensor
            auto zeroOffset = CreateZeroOffsetTensor(*Program::GetInstance().GetCurrentFunction(), scale.GetStorage());
            dequantizedResult = TensorDequantizeOperation(*Program::GetInstance().GetCurrentFunction(),
                                                          transposedInput.GetStorage(), scale.GetStorage(), zeroOffset,
                                                          -1);
        }

        // [Transpose] set tmp validShape
        dequantizedResult.GetStorage()->UpdateDynValidShape(tmpValidShape);
        // [Transpose] set tmp VecTile
        TileShape::Current().SetVecTile(tmpVectile);
        // output back: [..., W, H] -> [..., H, W]
        Tensor result = Transpose(dequantizedResult, {secondLastDim, lastDim});
        // [Tstore] get origin ValidShape
        result.GetStorage()->UpdateDynValidShape(processedInput.GetStorage()->dynValidShape_);
        // [Tstore] get origin TileShape
        TileShape::Current().SetVecTile(oriVectile);

        // If input was 1D, reshape output back to 1D and restore original VecTile
        if (is1DInput) {
            int64_t n = originalInputStorage.GetShape()[0];
            std::vector<int64_t> originalShape = {n};
            auto originalValidShape = originalInputStorage.GetStorage()->GetDynValidShape();
            result = Reshape(result, originalShape, originalValidShape);
            // Restore original VecTile
            if (!originalVecTile.tile.empty()) {
                TileShape::Current().SetVecTile(originalVecTile);
            }
        }
        return result;
    }

    // axis=-1 case
    Tensor result;
    if (isAsymmetric) {
        result = CALL(DequantizeOperation, *Program::GetInstance().GetCurrentFunction(), processedInput.GetStorage(),
                      scale.GetStorage(), zeroPoints.GetStorage(), normalizedAxis);
    } else {
        // Symmetric: create zero offset tensor
        auto zeroOffset = CreateZeroOffsetTensor(*Program::GetInstance().GetCurrentFunction(), scale.GetStorage());
        result = CALL(DequantizeOperation, *Program::GetInstance().GetCurrentFunction(), processedInput.GetStorage(),
                      scale.GetStorage(), zeroOffset, normalizedAxis);
    }

    // If input was 1D, reshape output back to 1D and restore original VecTile
    if (is1DInput) {
        int64_t n = originalInputStorage.GetShape()[0];
        std::vector<int64_t> originalShape = {n};
        auto originalValidShape = originalInputStorage.GetStorage()->GetDynValidShape();
        result = Reshape(result, originalShape, originalValidShape);
        // Restore original VecTile
        if (!originalVecTile.tile.empty()) {
            TileShape::Current().SetVecTile(originalVecTile);
        }
    }
    return result;
}

Tensor DequantizeSymmetric(const Tensor& src, const Tensor& scale, int64_t axis)
{
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DequantizeSymmetric");
    CheckTensorFormat(scale.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DequantizeSymmetric");

    return Dequantize(src, scale, DataType::DT_FP32, axis, Tensor());
}

Tensor DequantizeAsymmetric(const Tensor& src, const Tensor& scale, const Tensor& zeroPoints, int64_t axis)
{
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DequantizeAsymmetric");
    CheckTensorFormat(scale.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DequantizeAsymmetric");
    CheckTensorFormat(zeroPoints.GetStorage(), {TileOpFormat::TILEOP_NZ}, "DequantizeAsymmetric");

    return Dequantize(src, scale, DataType::DT_FP32, axis, zeroPoints);
}

std::tuple<Tensor, Tensor> QuantMX(const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode,
                                   int64_t axis, bool performanceMode)
{
    DECLARE_TRACER();
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "QuantMX");

    CheckSupportedNPUArch(QUANT_MX_SUPPORTED_ARCHITECTURES, "QuantMX");
    CheckQuantMXPerformanceMode(static_cast<int64_t>(performanceMode));
    CheckQuantMXInput(input, quantDtype, mode, axis, performanceMode);
    const auto oldVecTile = TileShape::Current().GetVecTile();
    const auto& inputShape = input.GetShape();
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, inputShape.size());
    const bool isDnAxis = IsQuantMXDnAxis(normalizedAxis, inputShape.size());
    if (!oldVecTile.tile.empty() && isDnAxis && quantDtype == DT_FP4_E2M1X2) {
        CheckQuantMXTileShape(input.GetStorage(), oldVecTile, quantDtype, normalizedAxis,
                              static_cast<int64_t>(performanceMode));
    }
    if (performanceMode && !oldVecTile.tile.empty()) {
        CheckQuantMXPerformanceTileShape(input.GetStorage(), oldVecTile, normalizedAxis,
                                         static_cast<int64_t>(performanceMode));
    }

    const std::vector<int64_t> groupedShape = isDnAxis ? BuildQuantMXDnScratchShape(inputShape) :
                                                         BuildQuantMXPerformanceGroupedShape(inputShape);
    const std::vector<int64_t> expShape = isDnAxis ? BuildQuantMXDnExpShape(inputShape) :
                                                     (performanceMode ? groupedShape :
                                                                        BuildQuantMXExpStorageShape(inputShape));
    const std::vector<int64_t> scaleShape = isDnAxis ? BuildQuantMXDnScaleShape(inputShape) :
                                                       BuildQuantMXScaleShape(inputShape);

    const auto scratchDtype = input.GetDataType();
    const auto maxScratchShape = groupedShape;
    const auto scalingShape = isDnAxis ? groupedShape : BuildQuantMXScalingShape(groupedShape, scratchDtype);
    auto quantized = Tensor(quantDtype, inputShape, "", TileOpFormat::TILEOP_ND);
    auto exp = Tensor(DataType::DT_FP8E8M0, expShape, "", TileOpFormat::TILEOP_ND);
    auto maxScratch = Tensor(scratchDtype, maxScratchShape, "", TileOpFormat::TILEOP_ND);
    auto scalingScratch = Tensor(scratchDtype, scalingShape, "", TileOpFormat::TILEOP_ND);

    std::vector<SymbolicScalar> scaleValidShape;
    const auto& inputValidShape = input.GetStorage()->GetDynValidShape();
    if (!inputValidShape.empty()) {
        quantized.GetStorage()->UpdateDynValidShape(inputValidShape);
        const auto groupedValidShape = isDnAxis ? BuildQuantMXDnScratchValidShape(inputValidShape) :
                                                  BuildQuantMXPerformanceGroupedValidShape(inputValidShape);
        const auto expValidShape = isDnAxis ? BuildQuantMXDnExpValidShape(inputValidShape) :
                                              (performanceMode ? groupedValidShape :
                                                                 BuildQuantMXExpStorageValidShape(inputValidShape));
        exp.GetStorage()->UpdateDynValidShape(expValidShape);
        maxScratch.GetStorage()->UpdateDynValidShape(groupedValidShape);
        const auto scalingValidShape = isDnAxis ? groupedValidShape :
                                                  BuildQuantMXScalingValidShape(groupedValidShape, scratchDtype);
        scalingScratch.GetStorage()->UpdateDynValidShape(scalingValidShape);
        scaleValidShape = isDnAxis ? BuildQuantMXDnScaleValidShape(inputValidShape) :
                                     BuildQuantMXScaleValidShape(inputValidShape);
    }

    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_QUANT_MX, {input.GetStorage()},
        {quantized.GetStorage(), exp.GetStorage(), maxScratch.GetStorage(), scalingScratch.GetStorage()});
    op.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(mode));
    op.SetAttribute(OpAttributeKey::mxQuantAxis, normalizedAxis);
    op.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, static_cast<int64_t>(performanceMode ? 1 : 0));
    if (!isDnAxis && performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(BuildQuantMXPerformanceVecTile(oldVecTile.tile));
    }
    auto scale = Reshape(exp, scaleShape, scaleValidShape);
    if (!isDnAxis && performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(oldVecTile);
    }
    return std::tie(quantized, scale);
}

// Tile Function Registration
void DequantizeOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledDequantize(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis);
}

REGISTER_OPERATION_TILED_FUNC(OP_DEQUANTIZE, Opcode::OP_DEQUANTIZE, DequantizeOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(QuantMX, Opcode::OP_QUANT_MX, QuantMXTileFunc);

} // namespace npu::tile_fwk
