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
 * \file indexing.cpp
 * \brief
 */

#include <climits>
#include <limits>
#include <cmath>
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/operation_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

constexpr float FP16_MAX = 65504.0f;

namespace {
bool IsFp8DataType(DataType dtype) { return dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2 || dtype == DT_FP8E8M0; }

void CheckFp8ArchSupport(const Tensor& tensor, const std::string& opName)
{
    if (!IsFp8DataType(tensor.GetDataType())) {
        return;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510)
        << opName << ": DT_FP8E4M3, DT_FP8E5M2 and DT_FP8E8M0 are only supported on DAV_3510 architecture.";
}
} // namespace

struct IndexAddPara {
    const LogicalTensorPtr& selfInput;
    const LogicalTensorPtr& srcInput;
    const LogicalTensorPtr& indicesInput;
    const LogicalTensorPtr& dstTensor;
    const int axis;
    const Element& alpha;
};

struct IndexAddTileInfoPara {
    TileInfo selfTileInfo;
    TileInfo srcTileInfo;
    TileInfo indicesTileInfo;
    TileInfo dstTileInfo;
};

Shape GetTempShape(Shape shape, size_t axis)
{
    Shape newShape(shape.size(), 1);
    for (size_t i = axis + 1; i < shape.size(); ++i) {
        newShape[i] = shape[i];
    }
    auto alignSize = BLOCK_SIZE / BytesOf(DT_BF16);
    newShape[shape.size() - 1] = (newShape[shape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    return newShape;
}

// IndexAdd in UB
void IndexAddUBExpandFunc(Function& function, const IndexAddPara& indexaddPara, IndexAddTileInfoPara& indexaddTileInfo)
{
    const LogicalTensorPtr& selfInput = indexaddPara.selfInput;
    const LogicalTensorPtr& srcInput = indexaddPara.srcInput;
    const LogicalTensorPtr& indicesInput = indexaddPara.indicesInput;
    const LogicalTensorPtr& dstTensor = indexaddPara.dstTensor;
    const int axis = indexaddPara.axis;
    const Element& alpha = indexaddPara.alpha;

    auto dstTile = dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape, indexaddTileInfo.dstTileInfo.offset);
    auto selfTile = selfInput->View(function, indexaddTileInfo.selfTileInfo.shape,
                                    indexaddTileInfo.selfTileInfo.offset);
    auto srcTile = srcInput->View(function, indexaddTileInfo.srcTileInfo.shape, indexaddTileInfo.srcTileInfo.offset);
    indexaddTileInfo.indicesTileInfo.offset = {
        indexaddTileInfo.srcTileInfo.offset[axis]}; // 按照srcShape所在的axis轴切分
    indexaddTileInfo.indicesTileInfo.shape = {indexaddTileInfo.srcTileInfo.shape[axis]};
    auto indexTile = indicesInput->View(function, indexaddTileInfo.indicesTileInfo.shape,
                                        indexaddTileInfo.indicesTileInfo.offset);
    Shape tempShape(dstTile->GetShape().size(), 1);
    auto alignSize = BLOCK_SIZE / BytesOf(DT_BF16);
    tempShape[dstTile->GetShape().size() - 1] = (tempShape[dstTile->GetShape().size() - 1] + alignSize - 1) /
                                                alignSize * alignSize;
    auto tempBuffer = std::make_shared<LogicalTensor>(function, DT_BF16, tempShape);

    if (selfTile->Datatype() == DT_BF16 || (selfTile->Datatype() == DT_FP16 && indexTile->Datatype() == DT_INT64 &&
                                            (std::abs(alpha.Cast<float>() - 1) < 1e-6f))) {
        // vector和scalar均不支持BF16直接计算; alpha=1,且index类型为int64时逻辑不一样
        LogicalTensorPtr selfConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, selfTile->GetShape());
        Operation& castSelfOp = function.AddOperation(Opcode::OP_CAST, {selfTile}, {selfConvertedTile});
        selfConvertedTile->UpdateDynValidShape(selfTile->GetDynValidShape());
        castSelfOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr srcConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, srcTile->GetShape());
        Operation& castSrcOp = function.AddOperation(Opcode::OP_CAST, {srcTile}, {srcConvertedTile});
        srcConvertedTile->UpdateDynValidShape(srcTile->GetDynValidShape());
        castSrcOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr dstConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, dstTile->GetShape());
        tempBuffer = std::make_shared<LogicalTensor>(function, DT_BF16, GetTempShape(dstTile->GetShape(), axis));
        auto& op = function.AddOperation(Opcode::OP_INDEX_ADD_UB, {selfConvertedTile, srcConvertedTile, indexTile},
                                         {dstConvertedTile, tempBuffer});
        dstConvertedTile->UpdateDynValidShape(dstTile->GetDynValidShape());
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
        Operation& castDstOp = function.AddOperation(Opcode::OP_CAST, {dstConvertedTile}, {dstTile});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_RINT);
    } else {
        auto& op = function.AddOperation(Opcode::OP_INDEX_ADD_UB, {selfTile, srcTile, indexTile},
                                         {dstTile, tempBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
    }
}

void InnerTiledIndexAddUB(size_t cur, Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara,
                          IndexAddTileInfoPara& indexaddTileInfo)
{
    if (cur == indexaddPara.dstTensor->shape.size()) {
        IndexAddUBExpandFunc(function, indexaddPara, indexaddTileInfo);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    // axis 维度不参与切分，也不循环
    if (static_cast<int>(cur) == indexaddPara.axis) {
        indexaddTileInfo.dstTileInfo.offset[cur] = 0;
        indexaddTileInfo.dstTileInfo.shape[cur] = indexaddPara.dstTensor->GetShape()[cur];
        indexaddTileInfo.selfTileInfo.offset[cur] = 0;
        indexaddTileInfo.selfTileInfo.shape[cur] = indexaddPara.selfInput->GetShape()[cur];
        indexaddTileInfo.srcTileInfo.offset[cur] = 0;
        indexaddTileInfo.srcTileInfo.shape[cur] = indexaddPara.srcInput->GetShape()[cur];
        InnerTiledIndexAddUB(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo);
        return;
    }

    // 非 axis 维度正常切分
    for (int64_t i = 0; i < indexaddPara.srcInput->GetShape()[cur]; i += tmpTile) {
        indexaddTileInfo.dstTileInfo.offset[cur] = i;
        indexaddTileInfo.dstTileInfo.shape[cur] = std::min(indexaddPara.dstTensor->GetShape()[cur] - i, tmpTile);

        indexaddTileInfo.selfTileInfo.offset[cur] = i;
        indexaddTileInfo.selfTileInfo.shape[cur] = std::min(indexaddPara.selfInput->GetShape()[cur] - i, tmpTile);

        indexaddTileInfo.srcTileInfo.offset[cur] = i;
        indexaddTileInfo.srcTileInfo.shape[cur] = std::min(indexaddPara.srcInput->GetShape()[cur] - i, tmpTile);

        InnerTiledIndexAddUB(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo);
    }
}

void TiledIndexAddUB(Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.selfInput->GetShape().size() == indexaddPara.selfInput->GetOffset().size())
        << "The size of indexaddPara selfinput shape and selfinput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.srcInput->GetShape().size() == indexaddPara.srcInput->GetOffset().size())
        << "The size of indexaddPara srcInput shape and srcInput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.indicesInput->GetShape().size() == indexaddPara.indicesInput->GetOffset().size())
        << "The size of indexaddPara indicesInput shape and indicesInput offset should be equal";

    IndexAddTileInfoPara indexaddTileInfo{
        TileInfo(indexaddPara.selfInput->GetShape().size(), indexaddPara.selfInput->GetOffset().size()),
        TileInfo(indexaddPara.srcInput->GetShape().size(), indexaddPara.srcInput->GetOffset().size()),
        TileInfo(indexaddPara.indicesInput->GetShape().size(), indexaddPara.indicesInput->GetOffset().size()),
        TileInfo(indexaddPara.dstTensor->GetShape().size(), indexaddPara.dstTensor->GetOffset().size())};
    InnerTiledIndexAddUB(0, function, tileShape, indexaddPara, indexaddTileInfo);
}

void TensorIndexAddUB(Function& function, const IndexAddPara& indexaddPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_INDEX_ADD_UB,
                                           {indexaddPara.selfInput, indexaddPara.srcInput, indexaddPara.indicesInput},
                                           {indexaddPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", indexaddPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
}

bool CheckAlphaOverflow(Element alpha, DataType dtype)
{
    double value = alpha.Cast<double>();
    if (std::isnan(value) || std::isinf(value))
        return true;
    switch (dtype) {
        case DT_INT8:
            return value < std::numeric_limits<int8_t>::min() || value > std::numeric_limits<int8_t>::max();
        case DT_INT16:
            return value < std::numeric_limits<int16_t>::min() || value > std::numeric_limits<int16_t>::max();
        case DT_INT32:
            return value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max();
        case DT_FP16:
            return std::abs(value) > FP16_MAX;
        case DT_BF16:
            return std::abs(value) > std::numeric_limits<float>::max();
        case DT_FP32:
            return std::abs(value) > std::numeric_limits<float>::max();
        default:
            return false;
    }
}

void CheckIndexAddParamsInvalid(const Tensor& self, const Tensor& src, const Tensor& indices, const int axis,
                                const Element& alpha, const Opcode& opCode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          axis < static_cast<int>(self.GetShape().size()) && axis >= -static_cast<int>(self.GetShape().size()))
        << "axis out of range of shape size";

    CheckTensorDimRange(self.GetStorage(), 1, 5, "INDEXADD");
    CheckTensorDimRange(indices.GetStorage(), 1, 1, "INDEXADD");
    CheckTensorShapeSize(self.GetStorage(), "INDEXADD");
    CheckTensorShapeSize(src.GetStorage(), "INDEXADD");
    CheckTensorShapeSize(indices.GetStorage(), "INDEXADD");
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), src.GetStorage()};
    CheckTensorsDimConsistency(tensors, "INDEXADD");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src.GetShape()[axis] == indices.GetShape()[0])
        << "src shape[axis] and indices[0] must equal";
    for (size_t i = 0; i < self.GetShape().size(); ++i) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, src.GetShape()[i] == self.GetShape()[i])
            << "src shape and self shape should be equal";
    }

    // 检查数据类型和格式
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    if (opCode == Opcode::OP_INDEX_ADD) {
        supportedTypes.insert(DT_INT8);
    }
    CheckTensorDataType(self.GetStorage(), supportedTypes, "INDEXADD");
    CheckTensorsDataTypeConsistency(self.GetStorage(), src.GetStorage(), "INDEXADD");
    CheckTensorsFormatConsistency(self.GetStorage(), src.GetStorage(), "INDEXADD");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, "INDEXADD");

    // 检验 alpha 溢出
    if (CheckAlphaOverflow(alpha, self.GetDataType())) {
        CHECK(VectorErrorCode::ERR_RUNTIME_LOGIC, false)
            << "Value cannot be converted to type " << DataType2String(self.GetDataType()) << " without overflow!";
    }
}

Tensor IndexAddUB(const Tensor& self, const Tensor& src, const Tensor& indices, int axis, const Element& alpha)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAddUB");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAddUB");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAddUB");

    CheckAxisRange(self, axis);
    CheckIndexAddParamsInvalid(self, src, indices, axis, alpha, Opcode::OP_INDEX_ADD_UB);
    DataType selfDataType = self.GetDataType();
    Element alpha_ = Element(selfDataType, alpha.Cast<float>());
    Tensor result(selfDataType, self.GetShape());
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    CALL(IndexAddUB, *Program::GetInstance().GetCurrentFunction(),
         {self.GetStorage(), src.GetStorage(), indices.GetStorage(), result.GetStorage(), axis, alpha_});
    return result;
}

// IndexAdd in GM
void IndexAddExpandFunc(Function& function, const IndexAddPara& indexaddPara, IndexAddTileInfoPara& indexaddTileInfo,
                        const LogicalTensorPtr& cachedDstTile = nullptr,
                        const LogicalTensorPtr& cachedSelfTile = nullptr)
{
    const LogicalTensorPtr& selfInput = indexaddPara.selfInput;
    const LogicalTensorPtr& srcInput = indexaddPara.srcInput;
    const LogicalTensorPtr& indicesInput = indexaddPara.indicesInput;
    const LogicalTensorPtr& dstTensor = indexaddPara.dstTensor;
    const int axis = indexaddPara.axis;

    auto selfTile = cachedSelfTile ? cachedSelfTile :
                                     selfInput->View(function, indexaddTileInfo.selfTileInfo.shape,
                                                     indexaddTileInfo.selfTileInfo.offset);
    auto dstTile = cachedDstTile ? cachedDstTile :
                                   dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape,
                                                   indexaddTileInfo.dstTileInfo.offset);
    auto srcTile = srcInput->View(function, indexaddTileInfo.srcTileInfo.shape, indexaddTileInfo.srcTileInfo.offset);
    indexaddTileInfo.indicesTileInfo.offset = {indexaddTileInfo.srcTileInfo.offset[axis]};
    indexaddTileInfo.indicesTileInfo.shape = {indexaddTileInfo.srcTileInfo.shape[axis]};
    auto indexTile = indicesInput->View(function, indexaddTileInfo.indicesTileInfo.shape,
                                        indexaddTileInfo.indicesTileInfo.offset);
    Shape tmpShape(2, 1);
    auto alignSize = BLOCK_SIZE / BytesOf(srcTile->Datatype());
    tmpShape[1] = AlignUp(srcTile->GetShape()[srcTile->GetShape().size() - 1], alignSize);
    auto tmpTile = std::make_shared<LogicalTensor>(function, srcTile->Datatype(), tmpShape);

    auto& op = function.AddOperation(Opcode::OP_INDEX_ADD, {selfTile, srcTile, indexTile}, {dstTile, tmpTile});
    op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
}

using TileCache = std::unordered_map<int64_t, std::pair<LogicalTensorPtr, LogicalTensorPtr>>;

void InnerTiledIndexAdd(size_t cur, Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara,
                        IndexAddTileInfoPara& indexaddTileInfo, TileCache& tileCache, int64_t encodeKey = 0)
{
    if (cur == indexaddPara.dstTensor->shape.size()) {
        auto it = tileCache.find(encodeKey);
        if (it == tileCache.end()) {
            auto selfTile = indexaddPara.selfInput->View(function, indexaddTileInfo.selfTileInfo.shape,
                                                         indexaddTileInfo.selfTileInfo.offset);
            auto dstTile = indexaddPara.dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape,
                                                        indexaddTileInfo.dstTileInfo.offset);
            it = tileCache.emplace(encodeKey, std::make_pair(dstTile, selfTile)).first;
        }
        // 调用缓存的dstTile创建子图
        IndexAddExpandFunc(function, indexaddPara, indexaddTileInfo, it->second.first, it->second.second);
        return;
    }
    const auto& vecTile = tileShape.GetVecTile();
    int64_t tileStep = vecTile[cur];
    const auto& srcShape = indexaddPara.srcInput->GetShape();
    const auto& dstShape = indexaddPara.dstTensor->GetShape();
    int64_t numTilesInCurDim = (srcShape[cur] + tileStep - 1) / tileStep;
    if (static_cast<int>(cur) == indexaddPara.axis) {
        // self和dst都在GM上，在axis轴不切分
        indexaddTileInfo.dstTileInfo.offset[cur] = 0;
        indexaddTileInfo.dstTileInfo.shape[cur] = dstShape[cur];
        indexaddTileInfo.selfTileInfo.offset[cur] = 0;
        indexaddTileInfo.selfTileInfo.shape[cur] = dstShape[cur];
        for (int i = 0; i < srcShape[cur]; i += tileStep) {
            indexaddTileInfo.srcTileInfo.offset[cur] = i;
            indexaddTileInfo.srcTileInfo.shape[cur] = std::min(srcShape[cur] - i, tileStep);
            // axis维度不参与编码，使用同一个encodeKey
            InnerTiledIndexAdd(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo, tileCache, encodeKey);
        }
    } else {
        // 非 axis 维度，dst、self、src都切块
        int64_t tileIndex = 0; // 当前维度块索引
        for (int i = 0; i < srcShape[cur]; i += tileStep) {
            indexaddTileInfo.dstTileInfo.offset[cur] = i;
            indexaddTileInfo.dstTileInfo.shape[cur] = std::min(dstShape[cur] - i, tileStep);
            indexaddTileInfo.selfTileInfo.offset[cur] = i;
            indexaddTileInfo.selfTileInfo.shape[cur] = std::min(dstShape[cur] - i, tileStep);
            indexaddTileInfo.srcTileInfo.offset[cur] = i;
            indexaddTileInfo.srcTileInfo.shape[cur] = std::min(srcShape[cur] - i, tileStep);
            // 使用混合基数编码
            int64_t newKey = encodeKey * numTilesInCurDim + tileIndex;
            tileIndex++;
            InnerTiledIndexAdd(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo, tileCache, newKey);
        }
    }
}

void TiledIndexAdd(Function& function, const TileShape& tileShape, const IndexAddPara& indexaddPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.selfInput->GetShape().size() == indexaddPara.selfInput->GetOffset().size())
        << "The size of indexaddPara selfinput shape and selfinput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.srcInput->GetShape().size() == indexaddPara.srcInput->GetOffset().size())
        << "The size of indexaddPara srcInput shape and srcInput offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          indexaddPara.indicesInput->GetShape().size() == indexaddPara.indicesInput->GetOffset().size())
        << "The size of indexaddPara indicesInput shape and indicesInput offset should be equal";

    IndexAddTileInfoPara indexaddTileInfo{
        TileInfo(indexaddPara.selfInput->GetShape().size(), indexaddPara.selfInput->GetOffset().size()),
        TileInfo(indexaddPara.srcInput->GetShape().size(), indexaddPara.srcInput->GetOffset().size()),
        TileInfo(indexaddPara.indicesInput->GetShape().size(), indexaddPara.indicesInput->GetOffset().size()),
        TileInfo(indexaddPara.dstTensor->GetShape().size(), indexaddPara.dstTensor->GetOffset().size())};
    TileCache tileCache;
    InnerTiledIndexAdd(0, function, tileShape, indexaddPara, indexaddTileInfo, tileCache);
}

void TensorIndexAdd(Function& function, const IndexAddPara& indexaddPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_INDEX_ADD,
                                           {indexaddPara.selfInput, indexaddPara.srcInput, indexaddPara.indicesInput},
                                           {indexaddPara.dstTensor});
    op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", indexaddPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
}

void IndexAdd_(Tensor& self, const Tensor& src, const Tensor& indices, int axis, const Element& alpha)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAdd_");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAdd_");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexAdd_");

    CheckAxisRange(self, axis);
    CheckIndexAddParamsInvalid(self, src, indices, axis, alpha, Opcode::OP_INDEX_ADD);
    DataType selfDataType = self.GetDataType();
    Element castedAlpha = Element(selfDataType, alpha.Cast<float>());
    Tensor result(selfDataType, self.GetShape());
    CALL(IndexAdd, *Program::GetInstance().GetCurrentFunction(),
         {self.GetStorage(), src.GetStorage(), indices.GetStorage(), result.GetStorage(), axis, castedAlpha});
    self = result;
}

void TiledGatherOperation(Function& function, const TileShape& tileShape, size_t cur, Input& paramsInput,
                          Input& indicesInput, int axis, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile = paramsInput.tensor.GetStorage()->View(function, paramsInput.tileInfo.shape,
                                                                paramsInput.tileInfo.offset);
        auto indicesTile = indicesInput.tensor.GetStorage()->View(function, indicesInput.tileInfo.shape,
                                                                  indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (function.IsStatic()) {
            auto& op = function.AddOperation(Opcode::OP_GATHER_FROM_UB, {paramsTile, indicesTile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        } else {
            auto& op = function.AddOperation(Opcode::OP_GATHER, {paramsTile, indicesTile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        }

        return;
    }

    // 按照resultShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur < static_cast<size_t>(axis)) {
            // 在result中gather轴的外层轴
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor.GetShape()[cur];
            paramsInput.tileInfo.shape[cur] = std::min(
                paramsInput.tensor.GetShape()[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
        } else if (cur >= static_cast<size_t>(axis) &&
                   (cur < static_cast<size_t>(axis) + indicesInput.tensor.GetShape().size())) {
            // 当前属于indices的gather轴
            // params[axis]不切
            paramsInput.tileInfo.offset[axis] = 0;
            paramsInput.tileInfo.shape[axis] = paramsInput.tensor.GetShape()[axis];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur - axis] = i % indicesInput.tensor.GetShape()[cur - axis];
            indicesInput.tileInfo.shape[cur - axis] = std::min(
                indicesInput.tensor.GetShape()[cur - axis] - indicesInput.tileInfo.offset[cur - axis], tmpTile);
        } else {
            // 在result中gather轴的内层轴
            int paramHighAxis = cur - indicesInput.tensor.GetShape().size() + 1;
            paramsInput.tileInfo.offset[paramHighAxis] = i % paramsInput.tensor.GetShape()[paramHighAxis];
            paramsInput.tileInfo.shape[paramHighAxis] = std::min(
                paramsInput.tensor.GetShape()[paramHighAxis] - paramsInput.tileInfo.offset[paramHighAxis], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherOperation(function, tileShape, cur + 1, paramsInput, indicesInput, axis, result, resultTileInfo);
    }
}

std::vector<int64_t> GatherOperationResultShape(LogicalTensorPtr params, LogicalTensorPtr indices, int axis)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, params->shape.size() == params->offset.size())
        << "The size of params shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices->shape.size() == indices->offset.size())
        << "The size of indices shape and offset should be equal";
    int paramsRank = params->shape.size();
    if (axis < 0) {
        axis = axis + paramsRank;
    }
    // result shape: params.shape[:aixs] + indices.shape + params.shape[axis+1:]
    std::vector<int64_t> resultShape = params->shape;
    resultShape.erase(resultShape.begin() + axis);
    resultShape.insert(resultShape.begin() + axis, indices->shape.begin(), indices->shape.end());

    return resultShape;
}

void TiledGatherOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& params,
                          const LogicalTensorPtr& indices, int axis, const LogicalTensorPtr& result)
{
    // Check Operands Valid
    std::vector<int64_t> expectedShape = GatherOperationResultShape(params, indices, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == expectedShape.size())
        << "The size of result shape and expectedShape should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == result->offset.size())
        << "The size of result shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, params->shape.size() == params->offset.size())
        << "The size of params shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices->shape.size() == indices->offset.size())
        << "The size of indices shape and offset should be equal";

    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, result->shape.size() <= NUM_VALUE_5)
        << "Not support shape size of result greater than 5";
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, indices->shape.size() <= NUM_VALUE_2)
        << "Not support shape size of indices greater than 2";
    if (axis < 0) {
        axis += params->shape.size();
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < static_cast<int>(params->shape.size()))
        << "The axis should be greater than or equal to 0 and less than shape size of params";
    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TensorGatherOperation(Function& function, const LogicalTensorPtr& params,
                                       const LogicalTensorPtr& indices, int axis)
{
    const auto& paramsDynShape = params->GetDynValidShape();
    const auto& indicesDynShape = indices->GetDynValidShape();
    const int paramsRank = paramsDynShape.size();
    if (axis < 0) {
        axis += paramsRank;
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < paramsRank)
            << "The configuration of the axis is incorrect";
    }
    std::vector<int64_t> resultShape = GatherOperationResultShape(params, indices, axis);
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), resultShape);
    std::vector<SymbolicScalar> outValidShape = paramsDynShape;
    outValidShape.erase(outValidShape.begin() + axis);
    outValidShape.insert(outValidShape.begin() + axis, indicesDynShape.begin(), indicesDynShape.end());
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_GATHER, {params, indices}, {result}, {outValidShape});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

void TensorGatherMask(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& result,
                      const uint8_t& patternMode)
{
    if (patternMode != 0) {
        auto& op = function.AddOperation(Opcode::OP_GATHER_MASK_BUILDIN, {self}, {result});
        op.SetAttribute(OP_ATTR_PREFIX + "patternMode", patternMode);
        return;
    }
}

void CheckGatherParamsInvalid(const Tensor& params, const Tensor& indices, int axis, const std::string& opName)
{
    static const std::unordered_set<DataType> a2a3Types = {DT_FP32, DT_FP16,   DT_BF16,   DT_INT32, DT_INT16,
                                                           DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8};
    static const std::unordered_set<DataType> a5Types = {DT_FP32,    DT_FP16,    DT_BF16,   DT_INT32, DT_INT16,
                                                         DT_INT8,    DT_UINT32,  DT_UINT16, DT_UINT8, DT_BOOL,
                                                         DT_FP8E4M3, DT_FP8E5M2, DT_FP8E8M0};
    const auto& supportedTypes = GetSupportedDataTypesByArch(a2a3Types, a5Types);
    CheckTensorDataType(params.GetStorage(), supportedTypes, opName);
    CheckFp8ArchSupport(params, opName);
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, opName);
    CheckTensorDimRange(params.GetStorage(), 1, 4, opName);
    CheckTensorDimRange(indices.GetStorage(), 1, 2, opName);
    CheckTensorShapeSize(params.GetStorage(), opName);
    CheckTensorShapeSize(indices.GetStorage(), opName);
    CheckAxisRange(params, axis);
    CheckTensorsFormatConsistency(params.GetStorage(), indices.GetStorage(), opName);
}

Tensor Gather(const Tensor& params, const Tensor& indices, int axis)
{
    DECLARE_TRACER();
    CheckTensorFormat(params.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gather");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Gather");

    CheckGatherParamsInvalid(params, indices, axis, "GATHER");
    RETURN_CALL(GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
                indices.GetStorage(), axis);
}

Tensor TensorIndex(const Tensor& params, const Tensor& indices)
{
    DECLARE_TRACER();
    CheckGatherParamsInvalid(params, indices, 0, "TENSORINDEX");
    // TensorIndex默认按0轴进行gather
    RETURN_CALL(GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
                indices.GetStorage(), 0);
}

void TiledGatherElementOperation(Function& function, const TileShape& tileShape, size_t cur, Input& paramsInput,
                                 Input& indicesInput, int axis, const LogicalTensorPtr& result,
                                 TileInfo& resultTileInfo)
{
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile = paramsInput.tensor.GetStorage()->View(function, paramsInput.tileInfo.shape,
                                                                paramsInput.tileInfo.offset);
        auto indicesTile = indicesInput.tensor.GetStorage()->View(function, indicesInput.tileInfo.shape,
                                                                  indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        Shape tmpShape({indicesTile->GetShape()[indicesTile->GetShape().size() - 1]});
        tmpShape[0] = 2 * AlignUp(tmpShape[0], BLOCK_SIZE / BytesOf(indicesTile->Datatype()));
        auto tmpBuffer = std::make_shared<LogicalTensor>(function, indicesTile->Datatype(), tmpShape);
        auto& op = function.AddOperation(Opcode::OP_GATHER_ELEMENT, {paramsTile, indicesTile}, {resultTile, tmpBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    // 按照resultShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur == static_cast<size_t>(axis)) {
            // params[axis]不切
            paramsInput.tileInfo.offset[cur] = 0;
            paramsInput.tileInfo.shape[cur] = paramsInput.tensor.GetShape()[cur];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor.GetShape()[cur];
            indicesInput.tileInfo.shape[cur] = std::min(
                indicesInput.tensor.GetShape()[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        } else {
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor.GetShape()[cur];
            paramsInput.tileInfo.shape[cur] = std::min(
                paramsInput.tensor.GetShape()[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor.GetShape()[cur];
            indicesInput.tileInfo.shape[cur] = std::min(
                indicesInput.tensor.GetShape()[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherElementOperation(function, tileShape, cur + 1, paramsInput, indicesInput, axis, result,
                                    resultTileInfo);
    }
}

void TiledGatherElementOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& params,
                                 const LogicalTensorPtr& indices, int axis, const LogicalTensorPtr& result)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == result->offset.size())
        << "The size of result shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, params->shape.size() == params->offset.size())
        << "The size of params shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices->shape.size() == indices->offset.size())
        << "The size of indices shape and offset should be equal";

    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherElementOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TensorGatherElementOperation(Function& function, const LogicalTensorPtr& params,
                                              const LogicalTensorPtr& indices, int axis)
{
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), indices->shape);
    std::vector<std::vector<SymbolicScalar>> outValidShape;
    outValidShape.push_back(indices->GetDynValidShape());
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_GATHER_ELEMENT, {params, indices}, {result},
                                           outValidShape);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

Tensor GatherElements(const Tensor& params, const Tensor& indices, int axis)
{
    DECLARE_TRACER();
    CheckTensorFormat(params.GetStorage(), {TileOpFormat::TILEOP_NZ}, "GatherElements");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "GatherElements");

    std::vector<LogicalTensorPtr> tensors = {params.GetStorage(), indices.GetStorage()};
    CheckTensorsDimConsistency(tensors, "GATHERELEMENTS");
    CheckAxisRange(params, axis); // 支持负轴
    for (size_t i = 0; i < params.GetShape().size(); ++i) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= params.GetShape()[i])
            << "The shape of params and indices should be equal";
    }
    static const std::unordered_set<DataType> a2a3Types = {DT_FP32,   DT_FP16,  DT_BF16,  DT_INT32,
                                                           DT_UINT32, DT_INT16, DT_UINT16};
    static const std::unordered_set<DataType> a5Types = {DT_FP32,   DT_FP16,  DT_BF16,  DT_INT32,
                                                         DT_UINT32, DT_INT16, DT_UINT16};
    const auto& supportedTypes = GetSupportedDataTypesByArch(a2a3Types, a5Types);
    CheckTensorDataType(params.GetStorage(), supportedTypes, "GATHERELEMENTS");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, "GATHERELEMENTS");
    CheckTensorDimRange(params.GetStorage(), 1, 5, "GATHERELEMENTS");
    CheckTensorShapeSize(params.GetStorage(), "GATHERELEMENTS");
    CheckTensorShapeSize(indices.GetStorage(), "GATHERELEMENTS");
    CheckTensorsFormatConsistency(params.GetStorage(), indices.GetStorage(), "GATHERELEMENTS");

    RETURN_CALL(GatherElementOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
                indices.GetStorage(), axis);
}

struct ScatterElementSPara {
    const LogicalTensorPtr& dstTensor;
    const LogicalTensorPtr& srcInput;
    const LogicalTensorPtr& idxInput;
    const Element& scalar;
    const int axis;
    const int scatterMode;
};

struct ScatterElementSTileInfoPara {
    TileInfo srcTileInfo;
    TileInfo idxTileInfo;
    TileInfo dstTileInfo;
};

void InnerTiledScatterElementS(size_t cur, Function& function, const TileShape& tileShape,
                               const ScatterElementSPara& scatterPara, ScatterElementSTileInfoPara& scatterTileInfo)
{
    const LogicalTensorPtr& dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr& srcInput = scatterPara.srcInput;
    const LogicalTensorPtr& idxInput = scatterPara.idxInput;
    const Element& scalar = scatterPara.scalar;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto srcTile = srcInput->View(function, scatterTileInfo.srcTileInfo.shape, scatterTileInfo.srcTileInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxTileInfo.shape, scatterTileInfo.idxTileInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstTileInfo.shape, scatterTileInfo.dstTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_SCATTER_ELEMENT, {srcTile, idxTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, scalar);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= dstTensor->shape[axis])
        << "The axis is not supported for tile splitting";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= idxInput->shape[axis])
        << "The axis is not supported for tile splitting";
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = std::max(dstTensor->shape[axis], idxInput->shape[axis]);
    }
    for (int i = 0; i < idxInput->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            scatterTileInfo.idxTileInfo.offset[cur] = 0;
            scatterTileInfo.idxTileInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.dstTileInfo.offset[cur] = 0;
            scatterTileInfo.dstTileInfo.shape[cur] = dstTensor->shape[cur];
            scatterTileInfo.srcTileInfo.offset[cur] = 0;
            scatterTileInfo.srcTileInfo.shape[cur] = srcInput->shape[cur];
        } else {
            scatterTileInfo.idxTileInfo.offset[cur] = i % idxInput->shape[cur];
            scatterTileInfo.idxTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.dstTileInfo.offset[cur] = i;
            scatterTileInfo.dstTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.srcTileInfo.offset[cur] = i;
            scatterTileInfo.srcTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
        }
        InnerTiledScatterElementS(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatterElementS(Function& function, const TileShape& tileShape, const ScatterElementSPara& scatterPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The size of srcInput shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The size of idxInput shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The size of dst shape and offset should be equal";

    ScatterElementSTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
    };
    InnerTiledScatterElementS(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatterElementS(Function& function, const ScatterElementSPara& scatterPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_SCATTER_ELEMENT,
                                           {scatterPara.srcInput, scatterPara.idxInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, scatterPara.scalar);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

static void CheckScatterElementSParamsInvalid(const Tensor& self, const Tensor& indices, int axis,
                                              const ScatterMode reduce)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SCATTER");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, "SCATTER");
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), indices.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), indices.GetStorage(), "SCATTER");
    CheckAxisRange(self, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, reduce <= ScatterMode::UNKNOWN)
        << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= self.GetShape()[i])
            << "The shape of indices and self should be equal";
    }
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SCATTER");
    CheckTensorShapeSize(self.GetStorage(), "SCATTER");
    CheckTensorShapeSize(indices.GetStorage(), "SCATTER");
}

Tensor Scatter(const Tensor& self, const Tensor& indices, const Element& src, int axis, ScatterMode reduce)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");

    CheckScatterElementSParamsInvalid(self, indices, axis < 0 ? self.GetShape().size() + axis : axis, reduce);
    DataType orgDtype = self.GetDataType();
    auto operandCast = Tensor(DataType::DT_FP32, self.GetShape());
    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = self;
    }
    axis = axis < 0 ? operandCast.GetShape().size() + axis : axis;
    Tensor result(operandCast.GetStorage()->Datatype(), operandCast.GetShape());
    result.GetStorage()->UpdateDynValidShape(operandCast.GetStorage()->GetDynValidShape());
    CALL(ScatterElementS, *Program::GetInstance().GetCurrentFunction(),
         {result.GetStorage(), operandCast.GetStorage(), indices.GetStorage(), src, axis, static_cast<int>(reduce)});

    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    orgDtype, CastMode::CAST_RINT);
    }
    return result;
}

struct ScatterPara {
    const LogicalTensorPtr& dstTensor;
    const LogicalTensorPtr& selfInput;
    const LogicalTensorPtr& idxInput;
    const LogicalTensorPtr& srcInput;
    const int axis;
    const int scatterMode;
};

struct ScatterTileInfoPara {
    TileInfo srcInfo;
    TileInfo idxInfo;
    TileInfo dstInfo;
    TileInfo selfInfo;
};

void InnerTiledScatter(size_t cur, Function& function, const TileShape& tileShape, const ScatterPara& scatterPara,
                       ScatterTileInfoPara& scatterTileInfo)
{
    const LogicalTensorPtr& dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr& selfInput = scatterPara.selfInput;
    const LogicalTensorPtr& idxInput = scatterPara.idxInput;
    const LogicalTensorPtr& srcInput = scatterPara.srcInput;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto selfTile = selfInput->View(function, scatterTileInfo.selfInfo.shape, scatterTileInfo.selfInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxInfo.shape, scatterTileInfo.idxInfo.offset);
        auto srcTile = srcInput->View(function, scatterTileInfo.srcInfo.shape, scatterTileInfo.srcInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstInfo.shape, scatterTileInfo.dstInfo.offset);
        Shape tmpShape({idxTile->GetShape()[idxTile->GetShape().size() - 1]});
        auto tmpBuffer = std::make_shared<LogicalTensor>(function, idxTile->Datatype(), tmpShape);
        auto& op = function.AddOperation(Opcode::OP_SCATTER, {selfTile, idxTile, srcTile}, {dstTile, tmpBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= dstTensor->shape[axis])
        << "The axis is not supported for tile splitting";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= idxInput->shape[axis])
        << "The axis is not supported for tile splitting";
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = std::max(dstTensor->shape[axis], idxInput->shape[axis]);
    }
    for (int i = 0; i < idxInput->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            scatterTileInfo.idxInfo.offset[cur] = 0;
            scatterTileInfo.idxInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.dstInfo.offset[cur] = 0;
            scatterTileInfo.dstInfo.shape[cur] = dstTensor->shape[cur];
            scatterTileInfo.srcInfo.offset[cur] = 0;
            scatterTileInfo.srcInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.selfInfo.offset[cur] = 0;
            scatterTileInfo.selfInfo.shape[cur] = selfInput->shape[cur];
        } else {
            scatterTileInfo.idxInfo.offset[cur] = i % idxInput->shape[cur];
            scatterTileInfo.idxInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.dstInfo.offset[cur] = i;
            scatterTileInfo.dstInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.srcInfo.offset[cur] = i;
            scatterTileInfo.srcInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.selfInfo.offset[cur] = i;
            scatterTileInfo.selfInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                           tmpTile);
        }
        InnerTiledScatter(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatter(Function& function, const TileShape& tileShape, const ScatterPara& scatterPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The shape size of srcInput and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The shape size of idxInput and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The shape size of dst and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.selfInput->shape.size() == scatterPara.selfInput->offset.size())
        << "The shape size of selfInput and offset should be equal";

    ScatterTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
        TileInfo(scatterPara.selfInput->shape.size(), scatterPara.selfInput->offset.size()),
    };
    InnerTiledScatter(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatter(Function& function, const ScatterPara& scatterPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_SCATTER,
                                           {scatterPara.selfInput, scatterPara.idxInput, scatterPara.srcInput},
                                           {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

static void CheckScatterParamsInvalid(const Tensor& self, const Tensor& indices, const Tensor& src, int axis,
                                      const ScatterMode reduce)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SCATTER");
    CheckTensorsDataTypeConsistency(self.GetStorage(), src.GetStorage(), "SCATTER");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, "SCATTER");
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), indices.GetStorage(), src.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), indices.GetStorage(), "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), src.GetStorage(), "SCATTER");
    CheckTensorsFormatConsistency(indices.GetStorage(), src.GetStorage(), "SCATTER");
    CheckAxisRange(self, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, reduce <= ScatterMode::UNKNOWN)
        << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= src.GetShape()[i])
            << "The shape size of src and indices should be equal";
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= self.GetShape()[i])
            << "The shape size of src and indices should be equal";
    }
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SCATTER");
    CheckTensorShapeSize(self.GetStorage(), "SCATTER");
    CheckTensorShapeSize(indices.GetStorage(), "SCATTER");
    CheckTensorShapeSize(src.GetStorage(), "SCATTER");
}

Tensor Scatter(const Tensor& self, const Tensor& indices, const Tensor& src, int axis, ScatterMode reduce)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");

    CheckScatterParamsInvalid(self, indices, src, axis < 0 ? self.GetShape().size() + axis : axis, reduce);
    DataType orgDtype = self.GetDataType();
    auto operandSelfCast = Tensor(DataType::DT_FP32, self.GetShape());
    auto operandSrcCast = Tensor(DataType::DT_FP32, src.GetShape());
    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        operandSelfCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                               self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
        operandSrcCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                              src.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandSelfCast = self;
        operandSrcCast = src;
    }
    axis = axis < 0 ? operandSelfCast.GetShape().size() + axis : axis;
    Tensor result(operandSelfCast.GetStorage()->Datatype(), operandSelfCast.GetShape());
    result.GetStorage()->UpdateDynValidShape(operandSelfCast.GetStorage()->GetDynValidShape());
    CALL(Scatter, *Program::GetInstance().GetCurrentFunction(),
         {result.GetStorage(), operandSelfCast.GetStorage(), indices.GetStorage(), operandSrcCast.GetStorage(), axis,
          static_cast<int>(reduce)});

    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    orgDtype, CastMode::CAST_RINT);
    }
    return result;
}

void TiledScatterUpdate(size_t cur, Function& function, const TileShape& tileShape, Input& srcInput, Input& indexInput,
                        Input& dstInput, int axis, const LogicalTensorPtr& dst, TileInfo& dstTileInfo,
                        std::string cacheMode, int blockSize)
{
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto resultTile = dst->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile = indexInput.tensor.GetStorage()->View(function, indexInput.tileInfo.shape,
                                                              indexInput.tileInfo.offset);
        auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {resultTile});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = dst->shape[cur];
    }
    for (int i = 0; i < dst->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor.GetShape()[cur];
            if (cur <= 1) {
                indexInput.tileInfo.offset[cur] = 0;
                indexInput.tileInfo.shape[cur] = indexInput.tensor.GetShape()[cur];
            }
            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor.GetShape()[cur];
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - srcInput.tileInfo.offset[cur],
                                                    tmpTile);
            if (cur == 0) { // only cut index first axis
                indexInput.tileInfo.offset[cur] = i % indexInput.tensor.GetShape()[cur];
                indexInput.tileInfo.shape[cur] = std::min(
                    indexInput.tensor.GetShape()[cur] - indexInput.tileInfo.offset[cur], tmpTile);
            } else {
                indexInput.tileInfo.offset[1] = 0;
                indexInput.tileInfo.shape[1] = indexInput.tensor.GetShape()[1];
            }
            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = std::min(dst->shape[cur] - dstTileInfo.offset[cur], tmpTile);
        }
        TiledScatterUpdate(cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo,
                           cacheMode, blockSize);
    }
}

void TiledIndexScatterUpdate(size_t cur, Function& function, const TileShape& tileShape, Input& srcInput,
                             Input& indexInput, Input& dstInput, int axis, const std::shared_ptr<LogicalTensor>& dst,
                             TileInfo& dstTileInfo, std::string cacheMode, int blockSize)
{
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile = indexInput.tensor.GetStorage()->View(function, indexInput.tileInfo.shape,
                                                              indexInput.tileInfo.offset);
        auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {dst});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照srcShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = srcInput.tensor.GetShape()[cur];
    }

    for (int i = 0; i < srcInput.tensor.GetShape()[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) { // asis == 1
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor.GetShape()[cur];

            int64_t indexTileLen = vecTile[0];
            indexInput.tileInfo.offset[cur] = 0;
            indexInput.tileInfo.shape[cur] = std::min(indexInput.tensor.GetShape()[cur] - indexInput.tileInfo.offset[0],
                                                      indexTileLen);

            // indextileinfo need trans : [16,0] -> [0,16]
            indexInput.tileInfo.offset[cur] = indexInput.tileInfo.offset[0];
            indexInput.tileInfo.offset[0] = 0;

            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor.GetShape()[cur];
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - srcInput.tileInfo.offset[cur],
                                                    tmpTile);

            indexInput.tileInfo.offset[0] = i % indexInput.tensor.GetShape()[1];
            indexInput.tileInfo.shape[0] = indexInput.tensor.GetShape()[0]; // index axis 0

            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = tmpTile;
        }
        TiledIndexScatterUpdate(cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo,
                                cacheMode, blockSize);
    }
}

void TiledScatterUpdateFor2Dims(Function& function, const TileShape& tileShape, const LogicalTensorPtr& result,
                                const LogicalTensorPtr& src, const LogicalTensorPtr& index, const LogicalTensorPtr& dst,
                                int axis, std::string cacheMode, int blockSize)
{
    auto& vecTile = tileShape.GetVecTile();
    int64_t tileBS = vecTile[NUM_VALUE_0];
    int64_t tileD = vecTile[NUM_VALUE_1];
    int64_t s = index->shape[1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, s != 0) << "1 dim of index is zero.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileBS != 0) << "0 dim of tileshape is zero.";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, (tileBS <= s && s % tileBS == 0) || (tileBS > s && tileBS % s == 0))
        << "tileshape 0 is invalid, tileshape(" << tileBS << ", " << tileD << ")";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, tileD == src->shape[NUM_VALUE_1])
        << "The tileD and src shape[0] should be equal";
    int64_t tileB = CeilDiv(tileBS, s);
    int64_t tileS = tileBS < s ? tileBS : s;
    int64_t bsOffset = 0;
    for (int64_t bIdx = 0; bIdx < index->shape[0]; bIdx += tileB) {
        for (int64_t sIdx = 0; sIdx < index->shape[1]; sIdx += tileS) {
            auto indexTile = index->View(
                function, {std::min(index->shape[0] - bIdx, tileB), std::min(index->shape[1] - sIdx, tileS)},
                {bIdx, sIdx});
            for (int64_t j = 0; j < src->shape[1]; j += tileD) {
                auto srcTile = src->View(
                    function, {std::min(src->shape[0] - bsOffset, tileBS), std::min(src->shape[1] - j, tileD)},
                    {bsOffset, j});
                auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dst}, {result});
                op.SetAttribute("axis", axis);
                op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
            }
            bsOffset += tileBS;
        }
    }
}

void TiledScatterUpdateFor4Dims(Function& function, const TileShape& tileShape, const LogicalTensorPtr& result,
                                const LogicalTensorPtr& src, const LogicalTensorPtr& index, const LogicalTensorPtr& dst,
                                int axis, std::string cacheMode, int blockSize)
{
    auto& vecTile = tileShape.GetVecTile();
    int64_t tileB = vecTile[NUM_VALUE_0];
    int64_t tileS = vecTile[NUM_VALUE_1];
    int64_t tileN = vecTile[NUM_VALUE_2];
    int64_t tileD = vecTile[NUM_VALUE_3];
    for (int64_t i = 0; i < src->shape[0]; i += tileB) {
        for (int64_t j = 0; j < src->shape[1]; j += tileS) {
            auto indexTile = index->View(
                function, {std::min(index->shape[0] - i, tileB), std::min(index->shape[1] - j, tileS)}, {i, j});
            for (int64_t n = 0; n < src->shape[2]; n += tileN) {
                for (int64_t d = 0; d < src->shape[3]; d += tileD) {
                    auto srcTile = src->View(function,
                                             {std::min(src->shape[0] - i, tileB), std::min(src->shape[1] - j, tileS),
                                              std::min(src->shape[2] - n, tileN), std::min(src->shape[3] - d, tileD)},
                                             {i, j, n, d});
                    auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dst}, {result});
                    op.SetAttribute("axis", axis);
                    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
                }
            }
        }
    }
}

void TiledScatterUpdate(Function& function, const TileShape& tileShape, const LogicalTensorPtr& result,
                        const LogicalTensorPtr& src, const LogicalTensorPtr& index, const LogicalTensorPtr& dst,
                        int axis, std::string cacheMode, int blockSize)
{
    if (cacheMode == "PA_BSND") {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == NUM_VALUE_2 || src->shape.size() == NUM_VALUE_4)
            << "shape must be 2 or 4";
        if (src->shape.size() == NUM_VALUE_2) {
            TiledScatterUpdateFor2Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        } else if (src->shape.size() == NUM_VALUE_4) {
            TiledScatterUpdateFor4Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        }
        return;
    }
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == result->offset.size())
        << "The shape of result and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == src->offset.size())
        << "The shape of src and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, index->shape.size() == index->offset.size())
        << "The shape of index and offset should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo indexTileInfo(index->shape.size(), index->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());

    auto srcInput = Input{src, srcTileInfo};
    auto indexInput = Input{index, indexTileInfo};
    auto dstInput = Input{dst, dstTileInfo};
    auto& vecTile = tileShape.GetVecTile();
    if (axis == 1 && src->shape.size() == NUM_VALUE_2 && vecTile[1] == src->shape[1]) { // 2维切index场景
        TiledIndexScatterUpdate(0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo,
                                cacheMode, blockSize);
    } else {
        TiledScatterUpdate(0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo,
                           cacheMode, blockSize);
    }
}

void TensorScatterUpdate(Function& function, const LogicalTensorPtr& result, const LogicalTensorPtr& dst,
                         const LogicalTensorPtr& index, const LogicalTensorPtr& src, int axis, std::string cacheMode,
                         int blockSize)
{
    std::vector<int> newOffset(src->shape.size(), 0);

    // src: ub
    // index: ub
    // dst: gm
    // result: gm
    auto& op = function.AddOperation(Opcode::OP_INDEX_OUTCAST, {src, index, dst}, {result});
    op.SetAttribute("axis", axis);
    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
}

static void CheckScatterUpdateInput(const Tensor& input)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          (input.GetShape().size() == NUM_VALUE_2 &&
           (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0)) ||
              (input.GetShape().size() == NUM_VALUE_4 &&
               (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0 &&
                input.GetShape(NUM_VALUE_2) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_3) != NUM_VALUE_0)))
        << "The shape of input is invaild";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          input.GetShape().size() == NUM_VALUE_2 || input.GetShape().size() == NUM_VALUE_4)
        << "The shape size of input is invaild";
    CheckTensorDimRange(input.GetStorage(), 2, 4, "SCATTERUPDATE");
}

static void CheckScatterUpdateIndex(const Tensor& index)
{
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT64, DT_INT32, DT_INT16};
    CheckTensorDataType(index.GetStorage(), indexSupportedTypes, "SCATTERUPDATE");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, index.GetShape().size() == NUM_VALUE_2 &&
                                                  index.GetShape(NUM_VALUE_0) != NUM_VALUE_0 &&
                                                  index.GetShape(NUM_VALUE_1) != NUM_VALUE_0)
        << "The shape of index is invaild";
}

static void CheckScatterUpdateInvalid(const Tensor& dst, const Tensor& index, const Tensor& src)
{
    std::vector<LogicalTensorPtr> tensors = {dst.GetStorage(), src.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTERUPDATE");
    CheckScatterUpdateIndex(index);
    CheckScatterUpdateInput(src);
    CheckScatterUpdateInput(dst);
}

Tensor ScatterUpdate(const Tensor& dst, const Tensor& index, const Tensor& src, int axis, std::string cacheMode,
                     int chunkSize)
{
    DECLARE_TRACER();
    CheckTensorFormat(dst.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScatterUpdate");
    CheckTensorFormat(index.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScatterUpdate");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScatterUpdate");

    CheckScatterUpdateInvalid(dst, index, src);
    CheckAxisRange(dst, axis);

    Tensor result(dst.GetStorage()->Datatype(), dst.GetStorage()->GetShape(), "", dst.Format());
    if (std::find(dst.GetStorage()->GetShape().begin(), dst.GetStorage()->GetShape().end(), -1) !=
        dst.GetStorage()->GetShape().end()) {
        Tensor resTmp(dst.GetStorage()->Datatype(), dst.GetStorage()->GetDynValidShape(), "", dst.Format());
        result = resTmp;
    }

    if (cacheMode == "PA_NZ") {
        axis = 1;
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, src.GetShape().size() == NUM_VALUE_2)
            << "Only support 2 dim"; // only support 2 dim

        Tensor newIndex = Reshape(index, {1, index.GetShape()[0] * index.GetShape()[1]});
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
             newIndex.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    } else {
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
             index.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    }
    return result;
}

void TiledIndexPut(Function& function, const TileShape& tileShape, Input& inputSelf, Input& inputValues,
                   std::vector<Input>& inputIndices, const LogicalTensorPtr result, bool accumulate, size_t cur)
{
    size_t selfDim = inputSelf.tileInfo.shape.size();
    size_t valuesDim = inputValues.tileInfo.shape.size();
    size_t indicesCount = inputIndices.size();
    if (cur == valuesDim) {
        auto inputSelfTile = inputSelf.tensor.GetStorage()->View(function, inputSelf.tileInfo.shape,
                                                                 inputSelf.tileInfo.offset);
        auto inputValuesTile = inputValues.tensor.GetStorage()->View(function, inputValues.tileInfo.shape,
                                                                     inputValues.tileInfo.offset);
        std::vector<LogicalTensorPtr> inputsTile;
        inputsTile.push_back(inputSelfTile);
        inputsTile.push_back(inputValuesTile);
        for (size_t j = 0; j < indicesCount; j++) {
            auto inputIndicesTile = inputIndices[j].tensor.GetStorage()->View(function, inputIndices[j].tileInfo.shape,
                                                                              inputIndices[j].tileInfo.offset);
            inputsTile.push_back(inputIndicesTile);
        }
        auto& newOp = function.AddOperation(Opcode::OP_INDEX_PUT, inputsTile, {result});
        newOp.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        newOp.SetAttribute(OpAttributeKey::accumulate, accumulate);
        newOp.SetAttribute(OpAttributeKey::indicesSize, static_cast<int>(indicesCount));
        return;
    }
    const auto& vecTile = tileShape.GetVecTile();
    int64_t tileSize = inputValues.tensor.GetShape()[cur];
    if (cur < vecTile.size()) {
        tileSize = vecTile[cur];
    }
    for (int64_t i = 0, size = inputValues.tensor.GetShape()[cur]; i < size; i += tileSize) {
        if (cur != 0) {
            size_t selfIndex = selfDim - valuesDim + cur;
            inputSelf.tileInfo.shape[selfIndex] = std::min(inputSelf.tensor.GetShape()[selfIndex] - i, tileSize);
            inputSelf.tileInfo.offset[selfIndex] = i;
        }
        inputValues.tileInfo.shape[cur] = std::min(inputValues.tensor.GetShape()[cur] - i, tileSize);
        inputValues.tileInfo.offset[cur] = i;
        if (cur == 0) {
            for (size_t j = 0; j < indicesCount; ++j) {
                inputIndices[j].tileInfo.shape[cur] = std::min(inputIndices[j].tensor.GetShape()[cur] - i, tileSize);
                inputIndices[j].tileInfo.offset[cur] = i;
            }
        }
        TiledIndexPut(function, tileShape, inputSelf, inputValues, inputIndices, result, accumulate, cur + 1);
    }
}

void TiledIndexPut(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                   const LogicalTensorPtr& values, const std::vector<LogicalTensorPtr>& indices,
                   const LogicalTensorPtr& result, bool accumulate)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->GetShape().size() == self->GetOffset().size());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, values->GetShape().size() == values->GetOffset().size());
    for (size_t i = 0; i < indices.size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices[i]->GetShape().size() == indices[i]->GetOffset().size());
    }
    TileInfo valuesTileInfo(values->shape.size(), values->offset.size());
    TileInfo selfTileInfo(self->shape.size(), self->offset.size());
    auto inputValues = Input{values, valuesTileInfo};
    auto inputSelf = Input{self, selfTileInfo};
    for (size_t i = 0, size = self->shape.size(); i < size; ++i) {
        inputSelf.tileInfo.shape[i] = self->shape[i];
        inputSelf.tileInfo.offset[i] = 0;
    }
    std::vector<Input> inputIndices;
    for (size_t i = 0, size = indices.size(); i < size; ++i) {
        TileInfo indicesTileInfoTemp(indices[i]->shape.size(), indices[i]->offset.size());
        auto inputIndicesTemp = Input{indices[i], indicesTileInfoTemp};
        inputIndices.push_back(inputIndicesTemp);
    }
    TiledIndexPut(function, tileShape, inputSelf, inputValues, inputIndices, result, accumulate, 0);
}

void TensorIndexPut(Function& function, const LogicalTensorPtr& self, const LogicalTensors& indices,
                    const LogicalTensorPtr& values, const LogicalTensorPtr& dst, bool accumulate)
{
    LogicalTensors iOperands = indices;
    iOperands.insert(iOperands.begin(), {self, values});
    auto& op = function.AddOperation(Opcode::OP_INDEX_PUT, iOperands, {dst});
    op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    op.SetAttribute(OpAttributeKey::accumulate, accumulate);
    op.SetAttribute(OpAttributeKey::indicesSize, static_cast<int>(indices.size()));
}

static void CheckIndexPutParamsInvalid(const Tensor& self, const std::vector<Tensor>& indices, const Tensor& values)
{
    std::unordered_set<DataType> supportedTypes = {DT_INT8,  DT_UINT8,  DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                                   DT_INT64, DT_UINT64, DT_BF16,  DT_FP16,   DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "INDEXPUT");
    CheckTensorsDataTypeConsistency(self.GetStorage(), values.GetStorage(), "INDEXPUT");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT8,  DT_UINT8,  DT_INT16, DT_UINT16,
                                                        DT_INT32, DT_UINT32, DT_INT64, DT_UINT64};
    int indicesShape = -1;
    for (size_t i = 0; i < indices.size(); i++) {
        CheckTensorDataType(indices[i].GetStorage(), indexSupportedTypes, "INDEXPUT");
        CheckTensorDimRange(indices[i].GetStorage(), 1, 1, "INDEXPUT");
        if (indicesShape == -1) {
            indicesShape = indices[i].GetShape()[0];
        } else {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices[i].GetShape()[0] == indicesShape)
                << "Tensors in indices should have the same shape";
        }
        CheckTensorShapeSize(indices[i].GetStorage(), "INDEXPUT");
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.size() >= 1 && indices.size() <= 4)
        << "indicesSize is out of range [1, 4]";
    CheckTensorDimRange(self.GetStorage(), 1, 4, "INDEXPUT");
    CheckTensorDimRange(values.GetStorage(), 1, 4, "INDEXPUT");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self.GetShape().size() + 1 == indices.size() + values.GetShape().size())
        << "unsupport the inputs shape combination: dimSelf + 1 != indicesSize + dimValues";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, values.GetShape()[0] == indicesShape)
        << "valuesFirstDim should equal to indicesShape";
    for (size_t i = 1; i < values.GetShape().size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              self.GetShape()[self.GetShape().size() - i] == values.GetShape()[values.GetShape().size() - i])
            << "valuesShape should match selfShape";
    }
    CheckTensorShapeSize(self.GetStorage(), "INDEXPUT");
    CheckTensorShapeSize(values.GetStorage(), "INDEXPUT");
    CheckTensorsFormatConsistency(self.GetStorage(), values.GetStorage(), "INDEXPUT");
    for (size_t i = 0; i < indices.size(); i++) {
        CheckTensorsFormatConsistency(self.GetStorage(), indices[i].GetStorage(), "INDEXPUT");
        CheckTensorsFormatConsistency(values.GetStorage(), indices[i].GetStorage(), "INDEXPUT");
    }
}

void IndexPut_(Tensor& self, const std::vector<Tensor>& indices, const Tensor& values, bool accumulate)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexPut_");
    CheckTensorFormat(values.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexPut_");
    for (const auto& tensor : indices) {
        CheckTensorFormat(tensor.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexPut_");
    }

    CheckIndexPutParamsInvalid(self, indices, values);

    std::vector<LogicalTensorPtr> indicesLogical;
    for (size_t i = 0; i < indices.size(); i++) {
        indicesLogical.push_back(indices[i].GetStorage());
    }
    Tensor dst(self.GetDataType(), self.GetShape());
    CALL(IndexPut, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), indicesLogical, values.GetStorage(),
         dst.GetStorage(), accumulate);
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(self.GetStorage(), dst.GetStorage());
    self = dst;
}

template <typename T, DataType dataType>
Element GetCurStartElement(Element start, Element step, int id)
{
    T startValue;
    T stepValue;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        startValue = start.GetSignedData();
        stepValue = step.GetSignedData();
    } else if (dataType == DT_FP32) {
        startValue = (float)start.GetFloatData();
        stepValue = (float)step.GetFloatData();
    }
    T curStartValue = startValue + id * stepValue;
    Element curStart(dataType, curStartValue);
    return curStart;
}

const double EPSILON = (double)1e-12;
template <typename T, DataType dataType>
int64_t GetRangeResSize(Element& start, Element& end, Element& step)
{
    int64_t resultSize;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        int64_t startValue = start.GetSignedData();
        int64_t endValue = end.GetSignedData();
        int64_t stepValue = step.GetSignedData();
        if (abs(stepValue) <= 0) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "stepValue must not be 0";
        }
        resultSize = (endValue - startValue) % stepValue ? (endValue - startValue) / stepValue + 1 :
                                                           (endValue - startValue) / stepValue;
    } else if (dataType == DT_FP32) {
        double startValue = start.GetFloatData();
        double endValue = end.GetFloatData();
        double stepValue = step.GetFloatData();
        if (abs(stepValue) <= EPSILON) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "stepValue must not be 0";
        }
        resultSize = static_cast<int64_t>(std::ceil((endValue - startValue) / stepValue));
    }
    return resultSize;
}

void TiledRange(Function& function, const TileShape& tileShape, const Element start, const Element step,
                const LogicalTensorPtr& result)
{
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < result->shape[0]; i += vecTile[0]) {
        resultTileInfo.offset[0] = i;
        resultTileInfo.shape[0] = std::min(result->shape[0] - resultTileInfo.offset[0], vecTile[0]);
        int64_t curSizeValue = resultTileInfo.shape[0];
        Element curSize(DT_INT64, curSizeValue);
        Element curStart = start;

        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_RANGE, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "START", curStart);
        op.SetAttribute(OP_ATTR_PREFIX + "SIZE", curSize);
        op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
        SymbolicScalar tileIdx(i);
        op.SetAttribute(OpAttributeKey::dynScalar, tileIdx);
    }
    return;
}

LogicalTensorPtr TensorRange(Function& function, LogicalTensorPtr& result, Element& start, Element& step)
{
    auto& op = function.AddOperation(Opcode::OP_RANGE, {}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "START", start);
    op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
    Element size(DT_INT64, result->shape[0]);
    op.SetAttribute(OP_ATTR_PREFIX + "SIZE", size);
    return result;
}

Tensor RealRange(Element& start, Element& end, Element& step)
{
    DECLARE_TRACER();
    std::vector<int64_t> resTensorShape;
    int64_t resultSize;
    if (start.GetDataType() == DT_INT32) {
        resultSize = GetRangeResSize<int32_t, DT_INT32>(start, end, step);
    } else if (start.GetDataType() == DT_INT64) {
        resultSize = GetRangeResSize<int64_t, DT_INT64>(start, end, step);
    } else if (start.GetDataType() == DT_FP32) {
        resultSize = GetRangeResSize<float, DT_FP32>(start, end, step);
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported DataType " << DataType2String(start.GetDataType());
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, resultSize > 0)
        << "The positivity or negativity of the step should be aligned with the end-start";
    resTensorShape.push_back(resultSize);
    auto resTensor = Tensor(start.GetDataType(), resTensorShape);
    RETURN_CALL(Range, *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(), start, step);
}

bool IsDataTypeUnsupport(DataType dType)
{
    return dType != DT_FP32 && dType != DT_INT64 && dType != DT_INT32 && dType != DT_FP16 && dType != DT_BF16 &&
           dType != DT_INT16;
}

DataType GetComputeDataType(const Element& start, const Element& end, const Element& step)
{
    DataType startType = start.GetDataType();
    DataType endType = end.GetDataType();
    DataType stepType = step.GetDataType();
    if (IsDataTypeUnsupport(startType)) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported Start DataType " << DataType2String(startType);
    }
    if (IsDataTypeUnsupport(endType)) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported End DataType " << DataType2String(endType);
    }
    if (IsDataTypeUnsupport(stepType)) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported Step DataType " << DataType2String(stepType);
    }
    bool startIsFloat = (startType == DT_FP32 || startType == DT_FP16 || startType == DT_BF16);
    bool endIsFloat = (endType == DT_FP32 || endType == DT_FP16 || endType == DT_BF16);
    bool stepIsFloat = (stepType == DT_FP32 || stepType == DT_FP16 || stepType == DT_BF16);
    if (startIsFloat || endIsFloat || stepIsFloat) {
        return DT_FP32;
    }
    int64_t startValue = start.GetSignedData();
    int64_t endValue = end.GetSignedData();
    int64_t stepValue = step.GetSignedData();
    bool startFlag = startValue <= INT_MAX && startValue >= INT_MIN;
    bool endFlag = endValue <= INT_MAX && endValue >= INT_MIN;
    bool stepFlag = stepValue <= INT_MAX && stepValue >= INT_MIN;
    if (startFlag && endFlag && stepFlag) {
        return DT_INT32;
    }
    return DT_INT64;
}

DataType GetOutputDataType(const Element& start, const Element& end, const Element& step)
{
    DataType startType = start.GetDataType();
    DataType endType = end.GetDataType();
    DataType stepType = step.GetDataType();
    if (startType == DT_INT16 || endType == DT_INT16 || stepType == DT_INT16) {
        return DT_INT16;
    }
    if (startType == DT_FP32 || endType == DT_FP32 || stepType == DT_FP32) {
        return DT_FP32;
    }
    if (startType == DT_FP16 || endType == DT_FP16 || stepType == DT_FP16) {
        return DT_FP16;
    }
    if (startType == DT_BF16 || endType == DT_BF16 || stepType == DT_BF16) {
        return DT_BF16;
    }
    return DT_INT32;
}

Element GetElementWithDataType(const Element& element, DataType dataType)
{
    DataType elementType = element.GetDataType();
    bool elementIsFloat = (elementType == DT_FP32) || (elementType == DT_FP16) || (elementType == DT_BF16);
    if (elementIsFloat && dataType == DT_FP32) {
        return Element(dataType, element.GetFloatData());
    } else if (elementIsFloat && dataType != DT_FP32) {
        return Element(dataType, (int64_t)element.GetFloatData());
    } else if (!elementIsFloat && dataType == DT_FP32) {
        return Element(dataType, (double)element.GetSignedData());
    }
    return Element(dataType, element.GetSignedData());
}

Tensor Range(const Element& start, const Element& end, const Element& step)
{
    DataType dataType = GetComputeDataType(start, end, step);
    if (dataType != DT_FP32 && dataType != DT_INT32) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported Output DataType " << DataType2String(dataType);
    }
    DataType outputDataType = DT_INT32;
    outputDataType = GetOutputDataType(start, end, step);

    Element realStart = GetElementWithDataType(start, dataType);
    Element realEnd = GetElementWithDataType(end, dataType);
    Element realStep = GetElementWithDataType(step, dataType);
    auto resTensor = RealRange(realStart, realEnd, realStep);
    if (outputDataType == DT_BF16) {
        return Cast(resTensor, DT_BF16);
    }
    if (outputDataType == DT_FP16) {
        return Cast(resTensor, DT_FP16);
    }
    if (outputDataType == DT_INT16) {
        return Cast(resTensor, DT_INT16);
    }
    return resTensor;
}

Tensor GatherMask(const Tensor& self, const uint8_t patternMode)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "GatherMask");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_UINT16, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "GATHERMASK");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "GATHERMASK");
    CheckTensorShapeSize(self.GetStorage(), "GATHERMASK");
    auto shape = self.GetShape();
    auto& vecTile = TileShape::Current().GetVecTile();
    if (patternMode >= 1 && patternMode <= 6) {
        const auto& producers = self.GetStorage()->GetProducers();
        for (auto* op : producers) {
            if (op->GetOpcode() == Opcode::OP_VIEW) {
                const auto& inputShape = op->GetIOperands()[0]->GetShape();
                if (inputShape.size() == shape.size() && inputShape.back() > 0) {
                    CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape.back() >= inputShape.back())
                        << "GatherMask requires the last axis of self.shape not to be split by view. "
                        << "self.shape last axis: " << inputShape.back() << ", viewshape last axis: " << shape.back();
                }
            }
        }
    }
    if (patternMode == 1 || patternMode == 2) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape[shape.size() - 1] % 2 == 0)
            << "The last axis of input shape should be divisible by 2 when patternMode is 1 or 2";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.tile[vecTile.tile.size() - 1] % 2 == 0)
            << "The last axis of tileshape should be divisible by 2 when patternMode is 1 or 2";
        shape[shape.size() - 1] = shape[shape.size() - 1] / 2;
    } else if (patternMode == 3 || patternMode == 4 || patternMode == 5 || patternMode == 6) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape[shape.size() - 1] % 4 == 0)
            << "The last axis of input shape should be divisible by 4 when patternMode is 3, 4, 5 or 6";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.tile[vecTile.tile.size() - 1] % 4 == 0)
            << "The last axis of tileshape should be divisible by 4 when patternMode is 3, 4, 5 or 6";
        shape[shape.size() - 1] = shape[shape.size() - 1] / 4;
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, patternMode == 7)
            << "Just support patternMode is 1, 2, 3, 4, 5, 6, 7";
    }
    auto result = Tensor(self.GetDataType(), shape);

    if (!self.GetStorage()->GetDynValidShape().empty()) {
        std::vector<SymbolicScalar> outValidShape;
        for (auto dim : self.GetStorage()->GetDynValidShape()) {
            outValidShape.push_back(dim);
        }
        if (patternMode == 1 || patternMode == 2) {
            outValidShape[outValidShape.size() - 1] = outValidShape[outValidShape.size() - 1] / 2;
        } else if (patternMode == 3 || patternMode == 4 || patternMode == 5 || patternMode == 6) {
            outValidShape[outValidShape.size() - 1] = outValidShape[outValidShape.size() - 1] / 4;
        }
        result.GetStorage()->UpdateDynValidShape(outValidShape);
    }

    CALL(GatherMask, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(), patternMode);
    return result;
}

void TiledGatherMaskBuildIn(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                            const LogicalTensorPtr& result, TileInfo& resultTileInfo, const uint8_t patternMode)
{
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_GATHER_MASK, {inputTile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "patternMode", patternMode);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        if ((cur == input.tensor.GetShape().size() - 1) && (patternMode == 1 || patternMode == 2)) {
            resultTileInfo.offset[cur] = i / 2;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur] / 2);
        } else if ((cur == input.tensor.GetShape().size() - 1) &&
                   (patternMode == 3 || patternMode == 4 || patternMode == 5 || patternMode == 6)) {
            resultTileInfo.offset[cur] = i / 4;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur] / 4);
        } else {
            resultTileInfo.offset[cur] = i;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        }
        TiledGatherMaskBuildIn(function, tileShape, cur + 1, input, result, resultTileInfo, patternMode);
    }
}

void TiledGatherMaskBuildIn(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
                            const LogicalTensorPtr resOperand, const uint8_t patternMode)
{
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledGatherMaskBuildIn(function, tileShape, 0, input, resOperand, resultTileInfo, patternMode);
}

void IndexAddUBOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledIndexAddUB(function, tileShape, {iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis, alpha});
}

void IndexAddOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledIndexAdd(function, tileShape, {iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis, alpha});
}

void GatherOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledGatherOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
}

void GatherElementOperationTileFunc(Function& function, const TileShape& tileShape,
                                    const std::vector<LogicalTensorPtr>& iOperand,
                                    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledGatherElementOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
}

void ScatterElementSOperationTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatterElementS(function, tileShape, {oOperand[0], iOperand[0], iOperand[1], scalar, axis, scatterMode});
}

void ScatterOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatter(function, tileShape, {oOperand[0], iOperand[0], iOperand[1], iOperand[2], axis, scatterMode});
}

void IndexPutOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    std::vector<LogicalTensorPtr> indices = iOperand;
    constexpr size_t num2 = 2;
    indices.erase(indices.begin(), indices.begin() + num2);
    bool accumulate = op.GetBoolAttribute(OpAttributeKey::accumulate);
    TiledIndexPut(function, tileShape, iOperand[0], iOperand[1], indices, oOperand[0], accumulate);
}

void IndexOutcastOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute("axis");
    int blockSize = op.GetIntAttribute(OpAttributeKey::panzBlockSize);
    std::string cacheMode = op.GetStringAttribute(OpAttributeKey::cacheMode);
    TiledScatterUpdate(function, tileShape, oOperand[0], iOperand[0], iOperand[1], iOperand[2], axis, cacheMode,
                       blockSize);
}

void RangeOperationTileFunc(Function& function, const TileShape& tileShape,
                            [[maybe_unused]] const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    Element start = op.GetElementAttribute(OP_ATTR_PREFIX + "START");
    Element step = op.GetElementAttribute(OP_ATTR_PREFIX + "STEP");
    TiledRange(function, tileShape, start, step, oOperand[0]);
}

void GatherMaskBuildInOperationTileFunc(Function& function, const TileShape& tileShape,
                                        const std::vector<LogicalTensorPtr>& iOperand,
                                        const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    uint8_t patternMode = op.GetIntAttribute(OP_ATTR_PREFIX + "patternMode");
    if (patternMode >= 1 && patternMode <= 6) {
        const auto& viewShape = iOperand[0]->GetShape();
        std::vector<int64_t> originalShape;
        if (iOperand[0]->GetAttr("ORIGINAL_SHAPE", originalShape) && !originalShape.empty() &&
            originalShape.size() == viewShape.size() && originalShape.back() > 0 && viewShape.back() > 0) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, viewShape.back() >= originalShape.back())
                << "GatherMask requires the last axis of self.shape not to be split by view. "
                << "self.shape last axis: " << originalShape.back() << ", viewshape last axis: " << viewShape.back();
        }
    }
    TiledGatherMaskBuildIn(function, tileShape, iOperand[0], oOperand[0], patternMode);
}

REGISTER_OPERATION_TILED_FUNC(OP_INDEX_ADD_UB, Opcode::OP_INDEX_ADD_UB, IndexAddUBOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, IndexAddOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_GATHER, Opcode::OP_GATHER, GatherOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_GATHER_ELEMENT, Opcode::OP_GATHER_ELEMENT, GatherElementOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ScatterElementSOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER, Opcode::OP_SCATTER, ScatterOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_INDEX_PUT, Opcode::OP_INDEX_PUT, IndexPutOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, IndexOutcastOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RANGE, Opcode::OP_RANGE, RangeOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_GATHER_MASK_BUILDIN, Opcode::OP_GATHER_MASK_BUILDIN,
                              GatherMaskBuildInOperationTileFunc);

} // namespace npu::tile_fwk
