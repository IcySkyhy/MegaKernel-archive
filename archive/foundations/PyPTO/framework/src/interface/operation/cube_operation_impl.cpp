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
 * \file cube_operation_impl.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"

#include "tilefwk/error_code.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "tilefwk/tile_shape.h"

namespace npu {
namespace tile_fwk {
namespace Matrix {
const float EPSILON = 1e-6f;
const uint64_t VECTOR_TILE_SHAPE = 128;
static const std::vector<DataType> FP4_TYPES = {DataType::DT_FP4_E2M1X2, DataType::DT_FP4_E1M2X2, DataType::DT_FP4_E2M1,
                                                DataType::DT_FP4_E1M2};

template <typename T>
auto CeilAlign(T num_1, T num_2) -> T
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

inline bool CheckValidShape(const LogicalTensorPtr& tensorPtr)
{
    if (tensorPtr == nullptr) {
        return false;
    }
    return tensorPtr->GetDynValidShape().size() == SHAPE_DIM2;
}

inline size_t GetAlignSize(DataType dataType)
{
    bool isB4 = std::find(FP4_TYPES.begin(), FP4_TYPES.end(), dataType) != FP4_TYPES.end();
    return isB4 ? ALIGN_SIZE_64 : ALIGN_SIZE_32;
}

template <typename T1, typename T2 = T1>
LogicalTensorPtr AddOpView(Function& function, const LogicalTensorPtr& srcTensorPtr,
                           const MatmulTensorInfo& dstTensorInfo, const std::map<std::string, T1> opAttr = {},
                           const std::map<std::string, T2> extraOpAttr = {})
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, srcTensorPtr != nullptr)
        << "Original tensor for OpView operation is nullptr.";
    auto dstShape = dstTensorInfo.shape;
    if (dstTensorInfo.transFlag) {
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, dstShape.size() == SHAPE_DIM2 || dstShape.size() == SHAPE_DIM3)
            << "destination shape dimension is invalid: Expected dimensions == " << SHAPE_DIM2 << " or " << SHAPE_DIM3
            << ", actual dimensions: " << dstShape.size();
        std::swap(dstShape[0], dstShape[1]);
    }
    LogicalTensorPtr dstTensorPtr = std::make_shared<LogicalTensor>(function, dstTensorInfo.dtype, dstShape,
                                                                    SymbolicScalar::FromConcrete(dstShape),
                                                                    dstTensorInfo.format, dstTensorInfo.name);
    dstTensorPtr->UpdateDynValidShape(
        GetViewValidShape(srcTensorPtr->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
    if (dstTensorInfo.transFlag) {
        auto& dstValidShape = dstTensorPtr->GetDynValidShape();
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
               dstValidShape.size() == SHAPE_DIM2 || dstValidShape.size() == SHAPE_DIM3)
            << "dstValidShape dimension is invalid: Expected dimensions == " << SHAPE_DIM2 << " or " << SHAPE_DIM3
            << ", actual dimensions: " << dstValidShape.size();
        std::swap(dstValidShape[0], dstValidShape[1]);
    }
    auto& viewOp = function.AddOperation(config::GetSliceOpcode(), {srcTensorPtr}, {dstTensorPtr});
    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        dstTensorInfo.offset, SymbolicScalar::FromConcrete(dstTensorInfo.offset), dstTensorPtr->GetDynValidShape());
    viewAttribute->SetToType(dstTensorInfo.memType);
    viewOp.SetOpAttribute(viewAttribute);
    for (const auto& attrPair : opAttr) {
        viewOp.SetAttribute(attrPair.first, attrPair.second);
    }
    for (const auto& attrPair : extraOpAttr) {
        viewOp.SetAttribute(attrPair.first, attrPair.second);
    }
    return dstTensorPtr;
}

LogicalTensorPtr AddOpView(Function& function, const LogicalTensorPtr& srcTensorPtr,
                           const MatmulTensorInfo& dstTensorInfo)
{
    return AddOpView<int64_t>(function, srcTensorPtr, dstTensorInfo);
}

void SetAMulBAttr(const MatmulGraphNodes& tensorGraphNodes, const MatmulAttrParam& attrParam,
                  const MatmulIterInfo& iterInfo, Operation& op)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.aTensorPtr != nullptr)
        << "aTensorPtr is nullptr, check input tensor A.";

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.bTensorPtr != nullptr)
        << "bTensorPtr is nullptr, check input tensor B.";

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.outTensorPtr != nullptr)
        << "outTensorPtr is nullptr, check output tensor C.";

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.aTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.bTensorPtr->Format()) << 1) |
                     // 2含义：cTensorPtr的索引，同时也是cTensor NZ信息的编码偏移位数
                     (static_cast<int64_t>(tensorGraphNodes.outTensorPtr->Format()) << 2);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(A_MUL_B_ACT_M, attrParam.mValue);
    op.SetAttribute(A_MUL_B_ACT_K, attrParam.kValue);
    op.SetAttribute(A_MUL_B_ACT_N, attrParam.nValue);
    op.SetAttribute(A_MUL_B_GM_ACC, attrParam.gmAccumulationFlag);
    op.SetAttribute(A_MUL_B_TRANS_MODE_ATTR, static_cast<int64_t>(attrParam.transMode));

    if (attrParam.isGemv) {
        op.SetAttribute(OpAttributeKey::isGemv, true);
    }

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
        op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(attrParam.reluType));
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, attrParam.scaleValue));
    } else if (op.GetOpcode() == Opcode::OP_A_MULACC_B && iterInfo.isLastK) {
        op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(attrParam.reluType));
        if (attrParam.scaleValue != 0) {
            op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, attrParam.scaleValue));
        }
    }
}

void SetTensorGraphAttr(Operation& op, const MatmulExtendParam& param, bool gmAccumulationFlag,
                        const MatmulAttrParam& attrParam)
{
    op.SetAttribute(A_MUL_B_GM_ACC, gmAccumulationFlag);
    op.SetAttribute(A_MUL_B_TRANS_A, attrParam.transA);
    op.SetAttribute(A_MUL_B_TRANS_B, attrParam.transB);
    op.SetAttribute(A_MUL_B_BIAS_ATTR, (param.biasTensor.GetStorage() != nullptr));
    op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(param.reluType));
    op.SetAttribute(A_MUL_B_TRANS_MODE_ATTR, static_cast<int64_t>(param.transMode));
    // means perchannel
    if (param.scaleTensor.GetStorage() != nullptr) {
        op.SetAttribute(A_MUL_B_VECTOR_QUANT_FLAG, true);
    }
    // means pertensor
    if (fabs(param.scaleValue - 0) > EPSILON) {
        uint32_t scaleValueTmp = 0;
        memcpy_s(&scaleValueTmp, sizeof(scaleValueTmp), &param.scaleValue, sizeof(param.scaleValue));
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, static_cast<uint64_t>(scaleValueTmp)));
    }
    // mx matmul
    if (attrParam.hasMXScale) {
        op.SetAttribute(A_MUL_B_MX_ATTR, true);
        op.SetAttribute(A_MUL_B_SCALE_A_COPY_IN_MODE, attrParam.transAScale ? static_cast<int64_t>(CopyInMode::DN2NZ) :
                                                                              static_cast<int64_t>(CopyInMode::ND2NZ));
        op.SetAttribute(A_MUL_B_SCALE_B_COPY_IN_MODE, attrParam.transBScale ? static_cast<int64_t>(CopyInMode::DN2NZ) :
                                                                              static_cast<int64_t>(CopyInMode::ND2NZ));
    }

    auto matrixSize = TileShape::Current().GetMatrixSize();
    if (matrixSize.size() < MATRIX_MAXSIZE) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        return;
    }
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize[M_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize[N_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize[K_INDEX]);
}

void SetMatmulAttrParam(const Operation& op, MatmulAttrParam& param)
{
    param.mValue = (op.HasAttr(A_MUL_B_ACT_M)) ? op.GetIntAttribute(A_MUL_B_ACT_M) : 0;
    param.kValue = (op.HasAttr(A_MUL_B_ACT_K)) ? op.GetIntAttribute(A_MUL_B_ACT_K) : 0;
    param.nValue = (op.HasAttr(A_MUL_B_ACT_N)) ? op.GetIntAttribute(A_MUL_B_ACT_N) : 0;
    param.reluType = (op.HasAttr(A_MUL_B_RELU_ATTR)) ? op.GetIntAttribute(A_MUL_B_RELU_ATTR) : 0;
    param.scaleValue = (op.HasAttr(A_MUL_B_SCALE_ATTR)) ? op.GetElementAttribute(A_MUL_B_SCALE_ATTR).GetUnsignedData() :
                                                          Element(DataType::DT_UINT64, 0).GetUnsignedData();
    param.hasBias = (op.HasAttr(A_MUL_B_BIAS_ATTR)) ? op.GetBoolAttribute(A_MUL_B_BIAS_ATTR) : false;
    param.hasScale = (op.HasAttr(A_MUL_B_VECTOR_QUANT_FLAG)) ? op.GetBoolAttribute(A_MUL_B_VECTOR_QUANT_FLAG) : false;
    param.hasMXScale = op.HasAttr(A_MUL_B_MX_ATTR);
    param.transA = (op.HasAttr(A_MUL_B_TRANS_A)) ? op.GetBoolAttribute(A_MUL_B_TRANS_A) : false;
    param.transB = (op.HasAttr(A_MUL_B_TRANS_B)) ? op.GetBoolAttribute(A_MUL_B_TRANS_B) : false;
    param.gmAccumulationFlag = (op.HasAttr(A_MUL_B_GM_ACC)) ? op.GetBoolAttribute(A_MUL_B_GM_ACC) : false;
    param.transMode = (op.HasAttr(A_MUL_B_TRANS_MODE_ATTR)) ? op.GetIntAttribute(A_MUL_B_TRANS_MODE_ATTR) : 0;
    param.isGemv = (op.HasAttr(OpAttributeKey::isGemv)) && op.GetBoolAttribute(OpAttributeKey::isGemv);
    if (param.hasMXScale) {
        param.transAScale = op.GetIntAttribute(A_MUL_B_SCALE_A_COPY_IN_MODE) == static_cast<int64_t>(CopyInMode::DN2NZ);
        param.transBScale = op.GetIntAttribute(A_MUL_B_SCALE_B_COPY_IN_MODE) == static_cast<int64_t>(CopyInMode::DN2NZ);
    }
}

void SetTensorGraphNodes(const std::vector<LogicalTensorPtr>& operandVec, const LogicalTensorPtr& cTensorPtr,
                         const MatmulAttrParam& param, MatmulGraphNodes& tensorGraphNodes)
{
    size_t mxScaleSize = static_cast<size_t>(param.hasMXScale) * SHAPE_DIM2;
    size_t operandVecSize = SHAPE_DIM2 +
                            static_cast<size_t>(param.hasScale + param.hasBias + param.gmAccumulationFlag) +
                            mxScaleSize;
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, operandVec.size() == operandVecSize)
        << "Operand vector size mismatch: Expected size: " << operandVecSize << ", actual size: " << operandVec.size()
        << ", SHAPE_DIM2: " << SHAPE_DIM2 << ", hasScale: " << param.hasScale << ", hasBias: " << param.hasBias
        << ", gmAccumulationFlag: " << param.gmAccumulationFlag << ", hasMXScale: " << param.hasMXScale;
    tensorGraphNodes.aTensorPtr = operandVec[0];
    tensorGraphNodes.bTensorPtr = operandVec[1];
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.aTensorPtr != nullptr)
        << "aTensorPtr is nullptr, check input tensor A.";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.bTensorPtr != nullptr)
        << "bTensorPtr is nullptr, check input tensor B.";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, cTensorPtr != nullptr) << "cTensorPtr is nullptr.";
    tensorGraphNodes.outTensorPtr = cTensorPtr;
    size_t extraDim = static_cast<size_t>(param.hasScale) | (static_cast<size_t>(param.hasBias) << 1) |
                      (static_cast<size_t>(param.gmAccumulationFlag) << 2) |
                      (static_cast<size_t>(param.hasMXScale) << 3); // 2、3含义：编码偏移
    switch (extraDim) {
        case 0: // 无bias，无scale, 无gmTensor
            break;
        case 1: // 有scale
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 2: // 2含义：有bias
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 3: // 3含义：有bias, 有scale
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM3];
            break;
        case 4: // 4含义：有gmTensor
            tensorGraphNodes.gmAccumulationTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 8: // 8含义: mxmatmul场景
            tensorGraphNodes.aScaleTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.bScaleTensorPtr = operandVec[SHAPE_DIM3];
            break;
        case 9: // 9含义: mxmatmul场景,有scale
            tensorGraphNodes.aScaleTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.bScaleTensorPtr = operandVec[SHAPE_DIM3];
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM4];
            break;
        case 10: // 10含义: mxmatmul场景,有bias
            tensorGraphNodes.aScaleTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.bScaleTensorPtr = operandVec[SHAPE_DIM3];
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM4];
            break;
        case 11: // 11含义: mxmatmul场景,有bias,有scale
            tensorGraphNodes.aScaleTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.bScaleTensorPtr = operandVec[SHAPE_DIM3];
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM4];
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM5];
            break;
        default:
            ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, false) << "Invalid tensor graph";
    }
}

Status CheckOperandShape(const Tensor& operand1, const Tensor& operand2, const MatmulAttrParam& attrParam)
{
    // 检查 shape 维度一致性
    const Shape shape1 = operand1.GetShape();
    const Shape shape2 = operand2.GetShape();
    size_t operand1Dim = shape1.size();
    size_t operand2Dim = shape2.size();
    size_t offsetSize1 = operand1.GetStorage()->offset.size();
    size_t offsetSize2 = operand2.GetStorage()->offset.size();

    // 检查输入维度的一致性
    const bool isDimSame = (operand1Dim == operand2Dim);
    CHECK(ExternalError::INVALID_SHAPE, isDimSame)
        << "Shape dimension mismatch: operand1=" << operand1Dim << ", operand2=" << operand2Dim;

    // 检查 shape 与 offset 的一致性
    const bool isOperand1OffsetMatch = (operand1Dim == offsetSize1);
    const bool isOperand2OffsetMatch = (operand2Dim == offsetSize2);
    CHECK(ExternalError::INVALID_SHAPE, isOperand1OffsetMatch)
        << "operand1 shape size(" << operand1Dim << ") != offset size(" << offsetSize1 << ")";
    CHECK(ExternalError::INVALID_SHAPE, isOperand2OffsetMatch)
        << "operand2 shape size(" << operand2Dim << ") != offset size(" << offsetSize2 << ")";

    // 检查最小维度
    const bool Op1DimValid = (operand1Dim >= SHAPE_DIM2);
    const bool Op2DimValid = (operand2Dim >= SHAPE_DIM2);
    CHECK(ExternalError::INVALID_SHAPE, Op1DimValid) << "operand1 dimension(" << operand1Dim << ") must be >= 2";
    CHECK(ExternalError::INVALID_SHAPE, Op2DimValid) << "operand2 dimension(" << operand2Dim << ") must be >= 2";

    for (size_t i = 0; i < operand1Dim; ++i) {
        CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, shape1[i] != -1)
            << "operand1 dim[" << i << "] = " << shape1[i]
            << ". Dynamic shape tensors are not allowed as operation operands. "
            << "Use view in pypto.loop to get static shape tensors before computation.";
    }

    for (size_t i = 0; i < operand2Dim; ++i) {
        CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, shape2[i] != -1)
            << "operand2 dim[" << i << "] = " << shape2[i]
            << ". Dynamic shape tensors are not allowed as operation operands. "
            << "Use view in pypto.loop to get static shape tensors before computation.";
    }

    // 检查每个维度的值
    for (size_t i = 0; i < operand1Dim; ++i) {
        CHECK(ExternalError::INVALID_SHAPE, shape1[i] > 0)
            << "operand1 dim[" << i << "] = " << shape1[i] << ", must be > 0";
    }

    for (size_t i = 0; i < operand2Dim; ++i) {
        CHECK(ExternalError::INVALID_SHAPE, shape2[i] > 0)
            << "operand2 dim[" << i << "] = " << shape2[i] << ", must be > 0";
    }

    if (operand1.GetDataType() == DataType::DT_FP4_E2M1) {
        const int64_t operand1DimK = attrParam.transA ? shape1[shape1.size() - SHAPE_DIM2] : shape1[shape1.size() - 1];
        const int64_t operand2DimK = attrParam.transB ? shape2[shape2.size() - 1] : shape2[shape2.size() - SHAPE_DIM2];
        CHECK(ExternalError::INVALID_SHAPE, operand1DimK == operand2DimK)
            << "when the input is FP4E2M1/E1M2, the K-axis of the A matrix needs to be divided by 2.";
    }

    MATMUL_LOGD("CheckOperandShape: PASS");
    return SUCCESS;
}

Status CheckL1L0Tile(const int64_t L0Tile, const int64_t L1Tile, const std::string& L0TileName,
                     const std::string& L1TileName)
{
    CHECK(ExternalError::INVALID_TILE, L0Tile != 0) << L0TileName << " cannot be zero, got " << L0Tile;

    CHECK(ExternalError::INVALID_TILE, L0Tile <= L1Tile && L1Tile % L0Tile == 0)
        << "Invalid L1/L0 relation: " << L0TileName << "=" << L0Tile << ", " << L1TileName << "=" << L1Tile
        << ", require " << L0TileName << " <= " << L1TileName << " && " << L1TileName << " % " << L0TileName << " == 0";

    return SUCCESS;
}

Status CheckCubeTiling(const Tensor& operand1, const Tensor& operand2, const MatmulAttrParam& attrParam)
{
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int32_t kBL1Idx = 2;
    const int64_t kL0 = cubeTile.k[0];
    const int64_t kL1a = cubeTile.k[1];
    const int64_t kL1b = cubeTile.k[kBL1Idx];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t mL1 = cubeTile.m[1];
    const int64_t nL0 = cubeTile.n[0];
    const int64_t nL1 = cubeTile.n[1];
    CHECK(ExternalError::INVALID_TILE, kL0 > 0 && kL1a > 0 && kL1b > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0)
        << "Invalid tile values: kL0=" << kL0 << ", kL1a=" << kL1a << ", kL1b=" << kL1b << ", mL0=" << mL0
        << ", mL1=" << mL1 << ", nL0=" << nL0 << ", nL1=" << nL1;

    CHECK(ExternalError::INVALID_TILE, kL0 % ALIGN_SIZE_16 == 0 && nL0 % ALIGN_SIZE_16 == 0)
        << "kL0(" << kL0 << ") and nL0(" << nL0 << ") must be aligned to 16 elements";
    // 检查 L1/L0 tile 关系
    if (CheckL1L0Tile(kL0, kL1a, "kL0", "kL1a") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(kL0, kL1b, "kL0", "kL1b") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(nL0, nL1, "nL0", "nL1") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(mL0, mL1, "mL0", "mL1") != SUCCESS) {
        return FAILED;
    }
    size_t alignSizeA = GetAlignSize(operand1.GetDataType());
    size_t alignSizeB = GetAlignSize(operand2.GetDataType());
    CHECK(ExternalError::INVALID_TILE, alignSizeA != 0 && alignSizeB != 0)
        << "The alignSize is zero, please check!! alignSizeA=" << alignSizeA << ", alignSizeB=" << alignSizeB;

    CHECK(ExternalError::INVALID_TILE, kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
        << "kL0 * sizeof(dtype) = " << kL0 * BytesOf(operand1.GetDataType()) << " bytes, must be 32-byte aligned";

    CHECK(ExternalError::INVALID_TILE, nL0 * BytesOf(operand2.GetDataType()) % ALIGN_SIZE_32 == 0)
        << "nL0 * sizeof(dtype) = " << nL0 * BytesOf(operand2.GetDataType()) << " bytes, must be 32-byte aligned";
    if (operand1.Format() == TileOpFormat::TILEOP_ND) {
        if (attrParam.transA) { // For ND A transpose, mL0 must be 32B aligned
            CHECK(ExternalError::INVALID_TILE, mL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
                << "mL0 memory not aligned when transA=true: " << mL0 * BytesOf(operand1.GetDataType())
                << " bytes, must be 32-byte aligned";
        }
    }
    return SUCCESS;
}

void CheckOperandShapeBound(const Tensor& operand)
{
    auto opFormat = operand.Format();
    size_t alignSize = GetAlignSize(operand.GetDataType());
    CHECK(ExternalError::INVALID_SHAPE, alignSize != 0) << "The alignSize is zero, please check!!";
    if (opFormat == TileOpFormat::TILEOP_ND) {
        CHECK(ExternalError::INVALID_SHAPE, operand.GetShape().back() <= SHAPE_INNER_AXIS_MAX_SIZE)
            << "Current inner axis: " << operand.GetShape().back()
            << ", when input is ND format, inner axis must be less than 65535";

        CHECK(ExternalError::INVALID_SHAPE,
              operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] <= std::numeric_limits<int32_t>::max())
            << "Current outer axis: " << operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2]
            << ", when input is ND format, outer axis must be less than 2^31 - 1";

        if (operand.GetDataType() == DataType::DT_FP4_E2M1 || operand.GetDataType() == DataType::DT_FP4_E1M2) {
            CHECK(ExternalError::INVALID_SHAPE, (operand.GetShape().back() & 1) == 0)
                << "Current inner axis: " << operand.GetShape().back()
                << ", when input is ND format and 4bit dtype, inner axis must be even number";
        }
    } else {
        CHECK(ExternalError::INVALID_SHAPE,
              operand.GetShape().back() * BytesOf(operand.GetDataType()) % ALIGN_SIZE_32 == 0)
            << "Current inner axis: " << operand.GetShape().back()
            << ", when input is NZ format, inner axis shape must be 32-byte aligned";

        CHECK(ExternalError::INVALID_SHAPE,
              operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] % ALIGN_SIZE_16 == 0)
            << "Current outer axis: " << operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2]
            << ", when input is NZ format, outer axis shape must be 16-element aligned";
    }
}

void CheckByteAlign(const Tensor& operand, const std::string& tileName, int64_t tileVal)
{
    size_t alignSize = GetAlignSize(operand.GetDataType());
    int64_t totalBytes = tileVal * BytesOf(operand.GetDataType());
    CHECK(ExternalError::INVALID_TILE, alignSize != 0) << "The alignSize is zero, please check!!";

    CHECK(ExternalError::INVALID_TILE, tileVal * BytesOf(operand.GetDataType()) % alignSize == 0)
        << "Current length of " << tileName << ": " << (size_t)totalBytes
        << " bytes, the length must be aligned to 32 bytes(4bit dtype must be aligned to 64)";
}

void CheckElementAlign(const std::string& tileName, int64_t tileVal)
{
    CHECK(ExternalError::INVALID_TILE, tileVal % ALIGN_SIZE_16 == 0)
        << "Current length of " << tileName << ": " << (size_t)tileVal
        << " elements, the length must be aligned to 16 elements";
}

void CheckNZFormatAligned(const Tensor& operand1, const Tensor& operand2, const MatmulAttrParam& attrParam)
{
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int64_t kL0 = cubeTile.k[0];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t nL0 = cubeTile.n[0];
    auto opFormatA = operand1.Format();
    auto opFormatB = operand2.Format();
    if (opFormatA == TileOpFormat::TILEOP_NZ) {
        if (attrParam.transA) {
            CheckByteAlign(operand1, "mL0", mL0);
            CheckElementAlign("kL0", kL0);
        } else {
            CheckByteAlign(operand1, "kL0", kL0);
            CheckElementAlign("mL0", mL0);
        }
    }
    if (opFormatB == TileOpFormat::TILEOP_NZ) {
        if (attrParam.transB) {
            CheckByteAlign(operand2, "kL0", kL0);
            CheckElementAlign("nL0", nL0);
        } else {
            CheckByteAlign(operand2, "nL0", nL0);
            CheckElementAlign("kL0", kL0);
        }
    }
}

void CheckCMatrixNZFormatAligned(const DataType& outType, const Tensor& operand, const MatmulAttrParam& attrParam)
{
    auto& cubeType = TileShape::Current().GetCubeTile();
    const int64_t nL0 = cubeType.n[0];
    if (attrParam.isCMatrixNZ) {
        int64_t nView = attrParam.transB ? operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] :
                                           operand.GetShape()[operand.GetShape().size() - 1];
        if (outType == DataType::DT_INT32) {
            CHECK(ExternalError::INVALID_SHAPE, nView % ALIGN_SIZE_16 == 0)
                << "Current nView: " << nView
                << " elements, nView must be aligned to 16 elements when CMatrix is NZ and outType is int32";
            CHECK(ExternalError::INVALID_TILE, nL0 % ALIGN_SIZE_16 == 0)
                << "Current nL0: " << nL0
                << " elements, nL0 must be aligned to 16 elements when CMatrix is NZ and outType is int32";
        } else {
            const bool nViewIsAlign = ((nView * BytesOf(outType)) % ALIGN_SIZE_32) == 0;
            const bool nL0IsAlign = ((nL0 * BytesOf(outType)) % ALIGN_SIZE_32) == 0;
            CHECK(ExternalError::INVALID_SHAPE, nViewIsAlign)
                << "Current nView: " << nView * BytesOf(outType)
                << " bytes, nView must be aligned to 32 bytes when CMatrix is NZ";
            CHECK(ExternalError::INVALID_TILE, nL0IsAlign)
                << "Current nL0: " << nL0 * BytesOf(outType)
                << " bytes, nL0 must be aligned to 32 bytes when CMatrix is NZ";
        }
    }
}

void CheckBiasShapeParam(const Tensor& operand2, bool transB, const MatmulExtendParam& param = {})
{
    if (param.biasTensor.GetStorage() == nullptr) {
        return;
    }
    CHECK(ExternalError::INVALID_FORMAT, param.biasTensor.Format() == TileOpFormat::TILEOP_ND)
        << "Only support TILEOP_ND.";
    const Shape biasShape = param.biasTensor.GetShape();
    CHECK(ExternalError::INVALID_SHAPE, biasShape.size() > 1 && biasShape[biasShape.size() - SHAPE_DIM2] == 1)
        << "Bias tensor shape of the penultimate dimension mismatch: "
        << "Expected shape of the penultimate dimension to be 1, got " << biasShape[biasShape.size() - SHAPE_DIM2];
    const Shape bShape = operand2.GetShape();
    CHECK(ExternalError::INVALID_SHAPE, bShape.size() != 4 || biasShape.size() == 2)
        << "4D batch Matmul only support 2D bias currently.";
    int n = transB ? bShape[bShape.size() - SHAPE_DIM2] : bShape[bShape.size() - 1];
    CHECK(ExternalError::INVALID_SHAPE, biasShape[biasShape.size() - 1] == n)
        << "Bias tensor shape of the last dimension mismatch. Expected shape of the last dimension to be n, which is "
        << n << ", got " << biasShape[biasShape.size() - 1];
}

void CheckBiasParam(const Tensor& operand2, bool transB, const MatmulExtendParam& param = {})
{
    DataType inDtype = operand2.GetDataType();
    if (param.biasTensor.GetStorage() == nullptr) {
        return;
    }
    if (inDtype == DataType::DT_BF16 || inDtype == DataType::DT_FP32) {
        CHECK(ExternalError::INVALID_TYPE, param.biasTensor.GetDataType() == DataType::DT_FP32)
            << "When input tensor is DT_BF16 or DT_FP32, bias must be DT_FP32.";
    } else if (inDtype == DataType::DT_FP16) {
        CHECK(ExternalError::INVALID_TYPE, param.biasTensor.GetDataType() == DataType::DT_FP32 ||
                                               param.biasTensor.GetDataType() == DataType::DT_FP16)
            << "When input tensor is DT_FP16, bias must be DT_FP32 or DT_FP16.";
    } else if (inDtype == DataType::DT_INT8) {
        CHECK(ExternalError::INVALID_TYPE, param.biasTensor.GetDataType() == DataType::DT_INT32)
            << "When input tensor is DT_INT8, bias must be DT_INT32.";
    }
    CheckBiasShapeParam(operand2, transB, param);
}

void CheckA5BiasParam(const Tensor& operand2, bool transB, const MatmulExtendParam& param = {})
{
    if (param.biasTensor.GetStorage() == nullptr) {
        return;
    }
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510)
        << "Current check only supports bias validation for the A5 platform.";

    DataType inDtype = operand2.GetDataType();
    static const std::unordered_map<DataType, std::unordered_set<DataType>> InputToBiasTypeMapA5 = {
        {DataType::DT_FP16, {DataType::DT_FP16, DataType::DT_FP32}},
        {DataType::DT_BF16, {DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_FP8E5M2, {DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_FP8E4M3, {DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_FP4_E2M1X2, {DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_FP4_E1M2X2, {DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_FP4_E2M1, {DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_FP4_E1M2, {DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_HF8, {DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32}},
        {DataType::DT_INT8, {DataType::DT_INT32}},
        {DataType::DT_FP32, {DataType::DT_FP32}},
    };
    auto it = InputToBiasTypeMapA5.find(inDtype);
    if (it == InputToBiasTypeMapA5.end()) {
        CHECK(ExternalError::INVALID_TYPE, false)
            << "Unsupported input dtype " << DataType2String(inDtype) << " for matmul.";
        return;
    }
    auto getSupportedBiasTypesStr = [&]() -> std::string {
        std::string result;
        for (DataType dt : it->second) {
            if (!result.empty())
                result += ", ";
            result += DataType2String(dt);
        }
        return result;
    };
    auto biasDtype = param.biasTensor.GetDataType();
    CHECK(ExternalError::INVALID_TYPE, it->second.count(biasDtype) > 0)
        << "Input operand dtype is " << DataType2String(inDtype)
        << ", which only supports bias dtype: " << getSupportedBiasTypesStr() << ". But got bias dtype "
        << DataType2String(biasDtype) << ".";

    CheckBiasShapeParam(operand2, transB, param);
}

void CheckFixpipeParam(const Tensor& operand2, DataType outDtype, bool transB, const MatmulExtendParam& param = {})
{
    DataType inDtype = operand2.GetDataType();
    bool isFixpipeSupport = (outDtype == DataType::DT_FP16 && inDtype == DataType::DT_INT8) ||
                            outDtype == DataType::DT_INT8;
    if (param.scaleTensor.GetStorage() != nullptr) {
        CHECK(ExternalError::INVALID_FORMAT, param.scaleTensor.Format() == TileOpFormat::TILEOP_ND)
            << "Only support TILEOP_ND.";

        CHECK(ExternalError::INVALID_TYPE, param.scaleTensor.GetDataType() == DataType::DT_INT64 ||
                                               param.scaleTensor.GetDataType() == DataType::DT_UINT64)
            << "scaleTensor dataType: " << DataType2String(param.scaleTensor.GetDataType())
            << ". scaleTensor only support int64 and uint64 dtype currently.";

        CHECK(ExternalError::INVALID_TYPE, isFixpipeSupport)
            << "Data type mismatch in fixpipe scenario. Expected inDtype to be DT_INT8 with outDtype to be DT_FP16, or "
               "outDtype to be DT_INT8.";

        const Shape scaleShape = param.scaleTensor.GetShape();
        CHECK(ExternalError::INVALID_SHAPE, scaleShape.size() > 1 && scaleShape[scaleShape.size() - SHAPE_DIM2] == 1)
            << "Scale tensor shape of the penultimate dimension mismatch. "
            << "Expected shape of the penultimate dimension to be 1, got "
            << scaleShape[scaleShape.size() - SHAPE_DIM2];
        const Shape bShape = operand2.GetShape();
        int n = transB ? bShape[bShape.size() - SHAPE_DIM2] : bShape[bShape.size() - 1];
        CHECK(ExternalError::INVALID_SHAPE, scaleShape[scaleShape.size() - 1] == n)
            << "Scale tensor shape of the last fimension mismatch. "
            << "Expected shape of the last dimension to be n, which is " << n << ", got "
            << scaleShape[scaleShape.size() - 1];
    }
    if (fabs(param.scaleValue - 0) > EPSILON) {
        CHECK(ExternalError::INVALID_TYPE, isFixpipeSupport)
            << "Data type mismatch in pertensor scenario. Expected inDtype to be DT_INT8 with outDtype to be DT_FP16, "
               "or outDtype to be DT_INT8.";
    }
    if (isFixpipeSupport) {
        CHECK(ExternalError::INVALID_OPERATION,
              fabs(param.scaleValue - 0) > EPSILON || param.scaleTensor.GetStorage() != nullptr)
            << "Quantization error in INT8→FP16 or ANY→INT8 path: scaleValue must not be 0.0f, OR scaleTensor must not "
               "be null.";
    }
}

void CheckTransModeParam(DataType inDtype, const MatmulExtendParam& param = {})
{
    if (param.transMode != TransMode::CAST_NONE) {
        CHECK(ExternalError::INVALID_TYPE, inDtype == DataType::DT_FP32)
            << "The param of transMode is only supported when input data type is DT_FP32.";
    }
}

void CheckGmAccumulationParam(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix,
                              const MatmulAttrParam& attrParam, const MatmulExtendParam& param = {})
{
    auto& cubeTile = TileShape::Current().GetCubeTile();
    if (!cubeTile.enableSplitK) {
        return;
    }
    CHECK(ExternalError::INVALID_FORMAT, !attrParam.isCMatrixNZ)
        << "Gm accumulation with output NZ format is not supported.";

    CHECK(ExternalError::INVALID_OPERATION, param.scaleTensor.GetStorage() == nullptr &&
                                                param.biasTensor.GetStorage() == nullptr &&
                                                fabs(param.scaleValue - 0) < EPSILON)
        << "Fixpipe and bias cannot be used simultaneously with GM ACC";

    CHECK(ExternalError::INVALID_OPERATION, param.reluType == ReLuType::NoReLu)
        << "ReLU is not supported in split-k scenarios";

    CHECK(ExternalError::INVALID_OPERATION, outType == DataType::DT_FP32 || outType == DataType::DT_INT32)
        << "Output data type only support FP32 and INT32 when using GM accumulated";

    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
        << "Both aMatrix and bMatrix cannot get storage";

    auto aMatrixValidShape = aMatrix.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = bMatrix.GetStorage()->GetDynValidShape();

    CHECK(ExternalError::INVALID_SHAPE,
          (aMatrixValidShape.size() >= SHAPE_DIM2 && aMatrixValidShape.size() <= SHAPE_DIM4) &&
              (bMatrixValidShape.size() >= SHAPE_DIM2 && bMatrixValidShape.size() <= SHAPE_DIM4) &&
              cubeTile.k.size() == MAX_K_DIM_SIZE)
        << "The validShapes of aMatrix and bMatrix must be 2~4 Dim. Additionally, the K TileShape must be 3 Dim";

    int64_t kSizeA = attrParam.transA ? aMatrix.GetShape()[aMatrix.GetShape().size() - SHAPE_DIM2] :
                                        aMatrix.GetShape()[aMatrix.GetShape().size() - 1];
    int64_t kSizeB = attrParam.transB ? bMatrix.GetShape()[bMatrix.GetShape().size() - 1] :
                                        bMatrix.GetShape()[bMatrix.GetShape().size() - SHAPE_DIM2];
    CHECK(ExternalError::INVALID_SHAPE, kSizeA == kSizeB)
        << "Matrix K dimension mismatch, kSizeA: " << kSizeA << ", kSizeB: " << kSizeB;
}

void CheckOperandDtype(DataType outType, const Tensor& operand1, const Tensor& operand2)
{
    CHECK(ExternalError::INVALID_TYPE, outType == DataType::DT_FP32 || outType == DataType::DT_FP16 ||
                                           outType == DataType::DT_BF16 || outType == DataType::DT_INT32 ||
                                           outType == DataType::DT_INT8)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT8 are supported.";
    const DataType operand1Dtype = operand1.GetDataType();
    const DataType operand2Dtype = operand2.GetDataType();
    const bool isOperand1Fp8 = (operand1Dtype == DataType::DT_FP8E5M2 || operand1Dtype == DataType::DT_FP8E4M3);
    const bool isOperand1Fp4 = std::find(FP4_TYPES.begin(), FP4_TYPES.end(), operand1Dtype) != FP4_TYPES.end();
    CHECK(ExternalError::INVALID_TYPE,
          !isOperand1Fp8 || (operand2Dtype == DataType::DT_FP8E5M2 || operand2Dtype == DataType::DT_FP8E4M3))
        << "When operand1 is of type DT_FP8E4M3 or DT_FP8E5M2, operand2 must be DT_FP8E4M3 or DT_FP8E5M2. operand1 "
           "dataType: "
        << DataType2String(operand1Dtype) << ", operand2 dataType: " << DataType2String(operand2Dtype);

    CHECK(ExternalError::INVALID_TYPE, (isOperand1Fp4 == false && isOperand1Fp8 == false) ||
                                           (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510))
        << "When operand1 data type is DT_FP8E5M2/E4M3 or FP4_E2M1X2/E1M2X2 or FP4_E2M1/E1M2, only DAV_3510 "
           "architecture is supported.";

    CHECK(ExternalError::INVALID_FORMAT,
          (operand1Dtype != DataType::DT_FP8E5M2 && operand1Dtype != DataType::DT_FP4_E1M2X2) ||
              operand1.Format() == TileOpFormat::TILEOP_ND)
        << "When operand1 data type is DT_FP8E5M2 or DT_FP4_E1M2X2, format must be ND.";

    CHECK(ExternalError::INVALID_FORMAT,
          (operand2Dtype != DataType::DT_FP8E5M2 && operand2Dtype != DataType::DT_FP4_E1M2X2) ||
              operand2.Format() == TileOpFormat::TILEOP_ND)
        << "When operand2 data type is DT_FP8E5M2 or DT_FP4_E1M2X2, format must be ND.";

    CHECK(ExternalError::INVALID_TYPE, isOperand1Fp8 || (operand1Dtype == operand2Dtype))
        << "input dataType must be consistent. operand1 dataType: " << DataType2String(operand1Dtype)
        << ", operand2 dataType: " << DataType2String(operand2Dtype);
}

void CheckNullptr(const Tensor& operand1, const Tensor& operand2)
{
    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, operand1.GetStorage() != nullptr) << "A Tensor cannot be null";
    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, operand2.GetStorage() != nullptr) << "B Tensor cannot be null";
}

Status CheckMatmulOperands(DataType outType, const Tensor& operand1, const Tensor& operand2,
                           const MatmulAttrParam& attrParam, const MatmulExtendParam& param = {})
{
    MATMUL_LOGD("Begin Matmul Operand Legality Check.\n");
    CheckNullptr(operand1, operand2);
    // dtype valid check
    CheckOperandDtype(outType, operand1, operand2);
    // shape valid check
    CheckOperandShape(operand1, operand2, attrParam);
    // GM Acc valid check
    CheckGmAccumulationParam(outType, operand1, operand2, attrParam, param);
    // tile valid check
    CheckCubeTiling(operand1, operand2, attrParam);
    // shape bound valid check
    CheckOperandShapeBound(operand1);
    CheckOperandShapeBound(operand2);
    // input NZ format valid check
    CheckNZFormatAligned(operand1, operand2, attrParam);
    // output NZ format valid check
    CheckCMatrixNZFormatAligned(outType, operand2, attrParam);
    // bias and scale valid check
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        CheckA5BiasParam(operand2, attrParam.transB, param);
    } else {
        CheckBiasParam(operand2, attrParam.transB, param);
    }
    CheckFixpipeParam(operand2, outType, attrParam.transB, param);
    // trans mode valid check
    CheckTransModeParam(operand1.GetDataType(), param);
    MATMUL_LOGD("Finish Matmul Operand Legality Check.\n");
    return SUCCESS;
}

void CheckMXMatmulShape(const Tensor& aTensor, const Tensor& aScaleTensor, const Tensor& bTensor,
                        const Tensor& bScaleTensor, const MatmulAttrParam& attrParam)
{
    CHECK(ExternalError::INVALID_SHAPE,
          aScaleTensor.GetShape().size() == SHAPE_DIM3 && bScaleTensor.GetShape().size() == SHAPE_DIM3)
        << "The dimension of scaleTensor for mxmatmul must be equal to 3! The dimension of ascaleTensor: "
        << aScaleTensor.GetShape().size() << ", The dimension of bscaleTensor: " << bScaleTensor.GetShape().size();

    CHECK(ExternalError::INVALID_SHAPE, aTensor.GetShape().size() >= 2)
        << "aTensor dimension must be >= 2 for matmul! Current dim: " << aTensor.GetShape().size();
    CHECK(ExternalError::INVALID_SHAPE, bTensor.GetShape().size() >= 2)
        << "bTensor dimension must be >= 2 for matmul! Current dim: " << bTensor.GetShape().size();

    const int64_t aDimOffset = aTensor.GetShape().size() - 2;
    const int64_t bDimOffset = bTensor.GetShape().size() - 2;

    int64_t mSize = attrParam.transA ? aTensor.GetShape()[aDimOffset + 1] : aTensor.GetShape()[aDimOffset];
    int64_t nSize = attrParam.transB ? bTensor.GetShape()[bDimOffset] : bTensor.GetShape()[bDimOffset + 1];
    int64_t kSize = attrParam.transA ? aTensor.GetShape()[aDimOffset] : aTensor.GetShape()[aDimOffset + 1];

    int64_t mScaleSize = attrParam.transAScale ? aScaleTensor.GetShape()[1] : aScaleTensor.GetShape()[0];
    int64_t kAScaleSize0 = attrParam.transAScale ? aScaleTensor.GetShape()[0] : aScaleTensor.GetShape()[1];
    int64_t kAScaleSize1 = aScaleTensor.GetShape()[SHAPE_DIM2];
    int64_t kBScaleSize0 = attrParam.transBScale ? bScaleTensor.GetShape()[1] : bScaleTensor.GetShape()[0];
    int64_t kBScaleSize1 = bScaleTensor.GetShape()[SHAPE_DIM2];
    int64_t nScaleSize = attrParam.transBScale ? bScaleTensor.GetShape()[0] : bScaleTensor.GetShape()[1];

    CHECK(ExternalError::INVALID_SHAPE, kAScaleSize0 == kBScaleSize0)
        << "Scale Matrix K dimension mismatch, kAScaleSize: " << kAScaleSize0 << ", kBScaleSize: " << kBScaleSize0;

    CHECK(ExternalError::INVALID_SHAPE, kAScaleSize1 == NUM2 && kBScaleSize1 == NUM2)
        << "Scale Matrix Inner axis must be equal to 2, AScale Inner axis: " << kAScaleSize1
        << ", BScale Inner axis: " << kBScaleSize1;

    CHECK(ExternalError::INVALID_SHAPE, mSize == mScaleSize)
        << "Scale Matrix M dimension mismatch, mScaleSize: " << mScaleSize << ", mSize: " << mSize;

    CHECK(ExternalError::INVALID_SHAPE, nSize == nScaleSize)
        << "Scale Matrix N dimension mismatch, nScaleSize: " << nScaleSize << ", nSize: " << nSize;

    CHECK(ExternalError::INVALID_SHAPE, kAScaleSize0 == CeilDiv(kSize, ALIGN_SIZE_64))
        << "Scale Matrix K dimension does not match Input Matrix K. Expected: Scale Matrix K size = "
        << CeilDiv(kSize, ALIGN_SIZE_64) << ", but got: " << kAScaleSize0;
}

Status CheckMXMatmulOperands(const Tensor& aTensor, const Tensor& aScaleTensor, const Tensor& bTensor,
                             const Tensor& bScaleTensor, const MatmulAttrParam& attrParam)
{
    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aTensor.GetStorage() != nullptr) << "aMatrix cannot be nullptr";

    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, bTensor.GetStorage() != nullptr) << "bMatrix cannot be nullptr";

    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aScaleTensor.GetStorage() != nullptr) << "aScale cannot be nullptr";
    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, bScaleTensor.GetStorage() != nullptr) << "bScale cannot be nullptr";
    CHECK(ExternalError::INVALID_TYPE,
          aScaleTensor.GetDataType() == DataType::DT_FP8E8M0 && bScaleTensor.GetDataType() == DataType::DT_FP8E8M0)
        << "input scale dataType must be DT_FP8E8M0. aScaleTensor dataType: "
        << DataType2String(aScaleTensor.GetDataType())
        << ", bScaleTensor dataType: " << DataType2String(bScaleTensor.GetDataType());
    DataType inDType = aTensor.GetDataType();
    static const std::unordered_set<DataType> supportedTypes = {DataType::DT_FP8E4M3,    DataType::DT_FP8E5M2,
                                                                DataType::DT_FP4_E2M1X2, DataType::DT_FP4_E1M2X2,
                                                                DataType::DT_FP4_E2M1,   DataType::DT_FP4_E1M2};
    CHECK(ExternalError::INVALID_TYPE, supportedTypes.find(inDType) != supportedTypes.end())
        << "Unsupported input data type. Only support DT_FP8E4M3, DT_FP8E5M2, DT_FP4_E2M1X2, DT_FP4_E1M2X2, "
           "DT_FP4_E2M1, DT_FP4_E1M2";
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int64_t kL0 = cubeTile.k[0];
    CHECK(ExternalError::INVALID_TILE, kL0 % ALIGN_SIZE_64 == 0)
        << "Current length of kL0: " << kL0 << ", the length of kL0 for mx matmul must be aligned to 64 elements";

    if (aTensor.GetShape().size() == SHAPE_DIM3 || aTensor.GetShape().size() == SHAPE_DIM4) {
        CHECK(ExternalError::INVALID_OPERATION, !cubeTile.enableSplitK)
            << "When the input dimension is 3 or 4 and the data type is DT_FP8E4M3, DT_FP8E5M2 or DT_FP4_E2M1, "
               "enableSplitK is not supported.";
    }
    CheckOperandShape(aScaleTensor, bScaleTensor, attrParam);
    CheckMXMatmulShape(aTensor, aScaleTensor, bTensor, bScaleTensor, attrParam);
    return SUCCESS;
}

void SetMatmulTileInfo(const TileShape& tileShape, const MatmulAttrParam& attrParam,
                       const MatmulGraphNodes& tensorGraphNodes, MatmulTileInfo& tileInfo)
{
    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR,
          tensorGraphNodes.aTensorPtr != nullptr && tensorGraphNodes.bTensorPtr != nullptr)
        << "Both inputs must be non-nullptr.";

    CHECK(ExternalError::INVALID_SHAPE, tensorGraphNodes.aTensorPtr->GetShape().size() == SHAPE_DIM2 &&
                                            tensorGraphNodes.bTensorPtr->GetShape().size() == SHAPE_DIM2)
        << "Invalid tensor shape dimension, expected both tensors to have exactly " << SHAPE_DIM2
        << " dimensions. aTensorPtr shape dim: " << tensorGraphNodes.aTensorPtr->GetShape().size()
        << ", bTensorPtr shape dim: " << tensorGraphNodes.bTensorPtr->GetShape().size();

    tileInfo.mView = attrParam.transA ? tensorGraphNodes.aTensorPtr->shape[1] : tensorGraphNodes.aTensorPtr->shape[0];
    tileInfo.nView = attrParam.transB ? tensorGraphNodes.bTensorPtr->shape[0] : tensorGraphNodes.bTensorPtr->shape[1];
    int64_t kViewA = attrParam.transA ? tensorGraphNodes.aTensorPtr->shape[0] : tensorGraphNodes.aTensorPtr->shape[1];
    int64_t kViewB = attrParam.transB ? tensorGraphNodes.bTensorPtr->shape[1] : tensorGraphNodes.bTensorPtr->shape[0];

    CHECK(ExternalError::INVALID_SHAPE, kViewA == kViewB)
        << "Matrix K dimension mismatch, kViewA: " << kViewA << ", kViewB: " << kViewB;
    tileInfo.kView = kViewA;

    auto& cubeTile = tileShape.GetCubeTile();
    tileInfo.tileML0 = cubeTile.m[0];
    tileInfo.tileML1 = cubeTile.m[1];
    tileInfo.tileNL0 = cubeTile.n[0];
    tileInfo.tileNL1 = cubeTile.n[1];
    tileInfo.tileKL0 = cubeTile.k[0];
    tileInfo.tileKAL1 = cubeTile.k[1];
    tileInfo.tileKBL1 = cubeTile.k[2]; // 2含义：kBL1 tile的偏移
    int64_t tileKL1Min = std::min(tileInfo.tileKAL1, tileInfo.tileKBL1);
    int64_t tileKL1Max = std::max(tileInfo.tileKAL1, tileInfo.tileKBL1);

    CHECK(ExternalError::INVALID_TILE,
          tileKL1Max >= kViewA || (tileKL1Max > 0 && tileKL1Min > 0 && tileKL1Max % tileKL1Min == 0))
        << "Invalid tileKL1 configuration: tileKL1Max: " << tileKL1Max << ", kViewA: " << kViewA
        << ", tileKL1Min: " << tileKL1Min
        << ". Must satisfy: tileKL1Max >= kViewA OR (all values > 0 and tileKL1Max is divisible by tileKL1Min).";

    CHECK(ExternalError::INVALID_TILE, tileInfo.tileKL0 > 0 && tileKL1Min % tileInfo.tileKL0 == 0)
        << "tileKL0: " << tileInfo.tileKL0 << ", tileKL1Min: " << tileKL1Min
        << ". Must have: tileKL0 > 0 AND tileKL1Min is divisible by tileKL0.";
}

LogicalTensorPtr LinkBias(Function& function, const MatmulGraphNodes& tensorGraphNodes, const TileInfo& tileInfoL1,
                          const TileInfo& tileInfoBT)
{
    if (tensorGraphNodes.biasTensorPtr == nullptr) {
        return nullptr;
    }

    MatmulTensorInfo biasL1TensorInfo{
        "biasL1Tensor",  tensorGraphNodes.biasTensorPtr->Datatype(), tileInfoL1.shape,  tileInfoL1.offset,
        NodeType::LOCAL, tensorGraphNodes.biasTensorPtr->Format(),   MemoryType::MEM_L1};
    LogicalTensorPtr biasL1TensorPtr = AddOpView<int64_t>(
        function, tensorGraphNodes.biasTensorPtr, biasL1TensorInfo,
        {{A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(CopyInMode::ND2ND)}});

    DataType biasBtType;
    if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) { // kirin does not need to change bias dtype
        biasBtType = tensorGraphNodes.biasTensorPtr->Datatype();
    } else {
        biasBtType = (tensorGraphNodes.aTensorPtr->Datatype() == DataType::DT_INT8) ? DataType::DT_INT32 :
                                                                                      DataType::DT_FP32;
    }
    MatmulTensorInfo biasBtTensorInfo{"biasBtTensor",    biasBtType,      tileInfoBT.shape,
                                      tileInfoBT.offset, NodeType::LOCAL, biasL1TensorPtr->Format(),
                                      MemoryType::MEM_BT};
    LogicalTensorPtr biasBtTensorPtr = AddOpView(function, biasL1TensorPtr, biasBtTensorInfo);
    return biasBtTensorPtr;
}

LogicalTensorPtr LinkScale(Function& function, const MatmulGraphNodes& tensorGraphNodes, const TileInfo& tileInfoL1,
                           const TileInfo& tileInfoFB)
{
    if (tensorGraphNodes.scaleTensorPtr == nullptr) {
        return nullptr;
    }

    MatmulTensorInfo scaleL1TensorInfo{
        "scaleL1Tensor", tensorGraphNodes.scaleTensorPtr->Datatype(), tileInfoL1.shape,  tileInfoL1.offset,
        NodeType::LOCAL, tensorGraphNodes.scaleTensorPtr->Format(),   MemoryType::MEM_L1};
    LogicalTensorPtr scaleL1TensorPtr = AddOpView<int64_t>(
        function, tensorGraphNodes.scaleTensorPtr, scaleL1TensorInfo,
        {{A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(CopyInMode::ND2ND)}});

    MatmulTensorInfo scaleFbTensorInfo{"scaleFbTensor",
                                       scaleL1TensorPtr->Datatype(),
                                       tileInfoFB.shape,
                                       tileInfoFB.offset,
                                       NodeType::LOCAL,
                                       scaleL1TensorPtr->Format(),
                                       MemoryType::MEM_FIX_QUANT_PRE};
    LogicalTensorPtr scaleFbTensorPtr = AddOpView(function, scaleL1TensorPtr, scaleFbTensorInfo);
    return scaleFbTensorPtr;
}

LogicalTensorPtr LinkTensorA(Function& function, const MatmulGraphNodes& tensorGraphNodes,
                             const MatmulAttrParam& attrParam, const MatmulTileInfo& tileInfo,
                             const MatmulIterInfo& iterInfo, LogicalTensorPtr& aL1TensorPtr)
{
    if (iterInfo.kOffset % tileInfo.tileKAL1 == 0) {
        std::vector<int64_t> aL1Shape = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kAL1Size, iterInfo.mL0Size} :
                                                             std::vector<int64_t>{iterInfo.mL0Size, iterInfo.kAL1Size};
        std::vector<int64_t> aL1Offset = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kOffset, iterInfo.mOffset} :
                                                              std::vector<int64_t>{iterInfo.mOffset, iterInfo.kOffset};
        MatmulTensorInfo aL1TensorInfo{
            "aL1Tensor",     tensorGraphNodes.aTensorPtr->Datatype(), aL1Shape,          aL1Offset,
            NodeType::LOCAL, tensorGraphNodes.aTensorPtr->Format(),   MemoryType::MEM_L1};
        // 根据是否使用了MXScale标志来决定Padding模式
        int64_t paddingMode = attrParam.hasMXScale ? static_cast<int64_t>(PaddingMode::MX_PADDING_MODE) : 0;
        // K维度在dynValidShape中的索引：transA时K在第0维，非transA时K在第1维
        int64_t kIndex = attrParam.transA ? 0 : 1;
        std::map<std::string, int64_t> l1OpAttr = {{COPY_IN_L1_PADDING_MODE, paddingMode},
                                                   {COPY_IN_L1_K_INDEX, kIndex}};
        if (attrParam.isGemv) {
            l1OpAttr[A_MUL_B_COPY_IN_MODE] = static_cast<int64_t>(CopyInMode::ND2ND);
            l1OpAttr[OpAttributeKey::isGemv] = static_cast<int64_t>(1);
        }
        aL1TensorPtr = AddOpView<int64_t, int64_t>(function, tensorGraphNodes.aTensorPtr, aL1TensorInfo, l1OpAttr);
    }
    std::vector<int64_t> aL0Shape = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kL0Size, iterInfo.mL0Size} :
                                                         std::vector<int64_t>{iterInfo.mL0Size, iterInfo.kL0Size};
    std::vector<int64_t> aL0Offset = (attrParam.transA) ?
                                         std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKAL1, 0} :
                                         std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKAL1};
    MatmulTensorInfo aL0TensorInfo{"aL0Tensor",
                                   tensorGraphNodes.aTensorPtr->Datatype(),
                                   aL0Shape,
                                   aL0Offset,
                                   NodeType::LOCAL,
                                   tensorGraphNodes.aTensorPtr->Format(),
                                   MemoryType::MEM_L0A,
                                   attrParam.transA};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(aL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(aL0Shape);
    LogicalTensorPtr aL0TensorPtr = AddOpView<bool, std::vector<SymbolicScalar>>(
        function, aL1TensorPtr, aL0TensorInfo,
        {{L1_TO_L0_TRANSPOSE, attrParam.transA}, {A_MUL_B_MX_ATTR, attrParam.hasMXScale}},
        {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    // GEMV A 设 isGemv：transA 不支持 GEMV，源头已保证 isGemv 仅非 transA
    if (attrParam.isGemv) {
        auto& viewOp = *aL0TensorPtr->GetProducers().begin();
        viewOp->SetAttribute(OpAttributeKey::isGemv, static_cast<int64_t>(1));
    }
    return aL0TensorPtr;
}

LogicalTensorPtr LinkTensorB(Function& function, const MatmulGraphNodes& tensorGraphNodes,
                             const MatmulAttrParam& attrParam, const MatmulTileInfo& tileInfo,
                             const MatmulIterInfo& iterInfo, LogicalTensorPtr& bL1TensorPtr)
{
    if (iterInfo.kOffset % tileInfo.tileKBL1 == 0) {
        std::vector<int64_t> bL1Shape = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nL0Size, iterInfo.kBL1Size} :
                                                             std::vector<int64_t>{iterInfo.kBL1Size, iterInfo.nL0Size};
        std::vector<int64_t> bL1Offset = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nOffset, iterInfo.kOffset} :
                                                              std::vector<int64_t>{iterInfo.kOffset, iterInfo.nOffset};
        MatmulTensorInfo bL1TensorInfo{
            "bL1Tensor",     tensorGraphNodes.bTensorPtr->Datatype(), bL1Shape,          bL1Offset,
            NodeType::LOCAL, tensorGraphNodes.bTensorPtr->Format(),   MemoryType::MEM_L1};
        // 根据是否使用了MXScale标志来决定Padding模式
        int64_t paddingMode = attrParam.hasMXScale ? static_cast<int64_t>(PaddingMode::MX_PADDING_MODE) : 0;
        // K维度在dynValidShape中的索引：transB时K在第1维，非transB时K在第0维
        int64_t kIndex = attrParam.transB ? 1 : 0;
        bL1TensorPtr = AddOpView<int64_t, int64_t>(
            function, tensorGraphNodes.bTensorPtr, bL1TensorInfo,
            {{COPY_IN_L1_PADDING_MODE, paddingMode}, {COPY_IN_L1_K_INDEX, kIndex}});
    }
    std::vector<int64_t> bL0Shape = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nL0Size, iterInfo.kL0Size} :
                                                         std::vector<int64_t>{iterInfo.kL0Size, iterInfo.nL0Size};
    std::vector<int64_t> bL0Offset = (attrParam.transB) ?
                                         std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKBL1} :
                                         std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKBL1, 0};
    MatmulTensorInfo bL0TensorInfo{"bL0Tensor",
                                   tensorGraphNodes.bTensorPtr->Datatype(),
                                   bL0Shape,
                                   bL0Offset,
                                   NodeType::LOCAL,
                                   tensorGraphNodes.bTensorPtr->Format(),
                                   MemoryType::MEM_L0B,
                                   attrParam.transB};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(bL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(bL0Shape);
    LogicalTensorPtr bL0TensorPtr = AddOpView<bool, std::vector<SymbolicScalar>>(
        function, bL1TensorPtr, bL0TensorInfo,
        {{L1_TO_L0_TRANSPOSE, attrParam.transB}, {A_MUL_B_MX_ATTR, attrParam.hasMXScale}},
        {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return bL0TensorPtr;
}

LogicalTensorPtr LinkTensorAScale(Function& function, const MatmulGraphNodes& tensorGraphNodes,
                                  const MatmulAttrParam& attrParam, const MatmulTileInfo& tileInfo,
                                  const MatmulIterInfo& iterInfo, LogicalTensorPtr& aScaleL1TensorPtr)
{
    int64_t ALIGN_64 = 64;
    int64_t tileKAScaleL1Size = CeilAlign(iterInfo.kAL1Size, ALIGN_64) / ALIGN_64;
    int64_t tileKAScaleL0Size = CeilAlign(iterInfo.kL0Size, ALIGN_64) / ALIGN_64;
    int64_t kAScaleOffset = iterInfo.kOffset / ALIGN_64;
    int64_t copyInMode = attrParam.transAScale ? static_cast<int64_t>(CopyInMode::DN2NZ) :
                                                 static_cast<int64_t>(CopyInMode::ND2NZ);
    if (iterInfo.kOffset % tileInfo.tileKAL1 == 0) {
        std::vector<int64_t> aScaleL1Shape = attrParam.transAScale ?
                                                 std::vector<int64_t>{tileKAScaleL1Size, iterInfo.mL0Size, NUM2} :
                                                 std::vector<int64_t>{iterInfo.mL0Size, tileKAScaleL1Size, NUM2};
        std::vector<int64_t> aScaleL1Offset = attrParam.transAScale ?
                                                  std::vector<int64_t>{kAScaleOffset, iterInfo.mOffset, 0} :
                                                  std::vector<int64_t>{iterInfo.mOffset, kAScaleOffset, 0};
        MatmulTensorInfo aScaleL1TensorInfo{
            "aScaleL1Tensor", tensorGraphNodes.aScaleTensorPtr->Datatype(), aScaleL1Shape,      aScaleL1Offset,
            NodeType::LOCAL,  tensorGraphNodes.aScaleTensorPtr->Format(),   MemoryType::MEM_L1, attrParam.transAScale};
        aScaleL1TensorPtr = AddOpView<int64_t>(function, tensorGraphNodes.aScaleTensorPtr, aScaleL1TensorInfo,
                                               {{A_MUL_B_COPY_IN_MODE, copyInMode}});
    }
    std::vector<int64_t> aScaleL0Shape = std::vector<int64_t>{iterInfo.mL0Size, tileKAScaleL0Size, NUM2};
    std::vector<int64_t> aScaleL0Offset = std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKAL1 / ALIGN_SIZE_64,
                                                               0};
    MatmulTensorInfo aScaleL0TensorInfo{
        "aScaleL0Tensor", tensorGraphNodes.aScaleTensorPtr->Datatype(), aScaleL0Shape,        aScaleL0Offset,
        NodeType::LOCAL,  tensorGraphNodes.aScaleTensorPtr->Format(),   MemoryType::MEM_L0AMX};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(aScaleL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(aScaleL0Shape);
    LogicalTensorPtr aScaleL0TensorPtr = AddOpView<std::vector<SymbolicScalar>>(
        function, aScaleL1TensorPtr, aScaleL0TensorInfo,
        {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return aScaleL0TensorPtr;
}

LogicalTensorPtr LinkTensorBScale(Function& function, const MatmulGraphNodes& tensorGraphNodes,
                                  const MatmulAttrParam& attrParam, const MatmulTileInfo& tileInfo,
                                  const MatmulIterInfo& iterInfo, LogicalTensorPtr& bScaleL1TensorPtr)
{
    int64_t ALIGN_64 = 64;
    int64_t tileKBScaleL1Size = CeilAlign(iterInfo.kBL1Size, ALIGN_64) / ALIGN_64;
    int64_t tileKBScaleL0Size = CeilAlign(iterInfo.kL0Size, ALIGN_64) / ALIGN_64;
    int64_t kBScaleOffset = iterInfo.kOffset / ALIGN_64;
    int64_t copyInMode = attrParam.transBScale ? static_cast<int64_t>(CopyInMode::DN2NZ) :
                                                 static_cast<int64_t>(CopyInMode::ND2NZ);
    if (iterInfo.kOffset % tileInfo.tileKBL1 == 0) {
        std::vector<int64_t> bScaleL1Shape = attrParam.transBScale ?
                                                 std::vector<int64_t>{iterInfo.nL0Size, tileKBScaleL1Size, NUM2} :
                                                 std::vector<int64_t>{tileKBScaleL1Size, iterInfo.nL0Size, NUM2};
        std::vector<int64_t> bScaleL1Offset = attrParam.transBScale ?
                                                  std::vector<int64_t>{iterInfo.nOffset, kBScaleOffset, 0} :
                                                  std::vector<int64_t>{kBScaleOffset, iterInfo.nOffset, 0};
        MatmulTensorInfo bScaleL1TensorInfo{
            "bScaleL1Tensor", tensorGraphNodes.bScaleTensorPtr->Datatype(), bScaleL1Shape,      bScaleL1Offset,
            NodeType::LOCAL,  tensorGraphNodes.bScaleTensorPtr->Format(),   MemoryType::MEM_L1, attrParam.transBScale};
        bScaleL1TensorPtr = AddOpView<int64_t>(function, tensorGraphNodes.bScaleTensorPtr, bScaleL1TensorInfo,
                                               {{A_MUL_B_COPY_IN_MODE, copyInMode}});
    }
    std::vector<int64_t> bScaleL0Shape = std::vector<int64_t>{tileKBScaleL0Size, iterInfo.nL0Size, NUM2};
    std::vector<int64_t> bScaleL0Offset = std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKBL1 / ALIGN_SIZE_64, 0,
                                                               0};
    MatmulTensorInfo bScaleL0TensorInfo{
        "bScaleL0Tensor", tensorGraphNodes.bScaleTensorPtr->Datatype(), bScaleL0Shape,        bScaleL0Offset,
        NodeType::LOCAL,  tensorGraphNodes.bScaleTensorPtr->Format(),   MemoryType::MEM_L0BMX};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(bScaleL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(bScaleL0Shape);
    LogicalTensorPtr bScaleL0TensorPtr = AddOpView<std::vector<SymbolicScalar>>(
        function, bScaleL1TensorPtr, bScaleL0TensorInfo,
        {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return bScaleL0TensorPtr;
}

void LinkAMulB(Function& function, const MatmulGraphNodes& tensorGraphNodes, const MatmulAttrParam& attrParam,
               const MatmulIterInfo& iterInfo, MatmulGraphNodes& tileGraphNodes)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tileGraphNodes.aTensorPtr != nullptr &&
                                                     tileGraphNodes.bTensorPtr != nullptr &&
                                                     tileGraphNodes.outTensorPtr != nullptr)
        << "Inputs must be non-nullptr.";
    std::vector<LogicalTensorPtr> aMulBInputs;
    std::vector<LogicalTensorPtr> aMulBOutputs;
    const std::string matmulOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (iterInfo.isFirstK) {
        aMulBInputs = {tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr};
    } else {
        aMulBInputs = {tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr, tileGraphNodes.cL0PartialSumPtr};
    }
    // MX matmul 场景
    if (attrParam.hasMXScale) {
        aMulBInputs.push_back(tileGraphNodes.aScaleTensorPtr);
        aMulBInputs.push_back(tileGraphNodes.bScaleTensorPtr);
    }
    if (attrParam.gmAccumulationFlag) {
        // GM 累加场景
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, tensorGraphNodes.gmAccumulationTensorPtr != nullptr &&
                                                       attrParam.hasBias == false && attrParam.hasScale == false)
            << "In GM accumulation mode, neither bias nor scale is allowed.";
        tileGraphNodes.gmAccumulationTensorPtr = tensorGraphNodes.gmAccumulationTensorPtr->View(
            function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nOffset});
        aMulBInputs.push_back(tileGraphNodes.gmAccumulationTensorPtr);
    } else {
        // 普通场景
        if (iterInfo.isFirstK) {
            if (tileGraphNodes.biasTensorPtr != nullptr) {
                aMulBInputs.push_back(tileGraphNodes.biasTensorPtr);
            }
        }
        if (iterInfo.isLastK && tileGraphNodes.scaleTensorPtr != nullptr) {
            aMulBInputs.push_back(tileGraphNodes.scaleTensorPtr);
        }
    }
    if (iterInfo.isLastK) {
        aMulBOutputs = {tileGraphNodes.outTensorPtr};
    } else {
        tileGraphNodes.cL0PartialSumPtr = std::make_shared<LogicalTensor>(
            function, tileGraphNodes.outTensorPtr->Datatype(), tileGraphNodes.outTensorPtr->GetShape());
        if (CheckValidShape(tileGraphNodes.aTensorPtr) && CheckValidShape(tileGraphNodes.bTensorPtr)) {
            tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(
                {tileGraphNodes.aTensorPtr->GetDynValidShape()[0], tileGraphNodes.bTensorPtr->GetDynValidShape()[1]});
        }
        aMulBOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto& aMulBOp = function.AddOperation(matmulOpStr, aMulBInputs, aMulBOutputs);
    SetAMulBAttr(tensorGraphNodes, attrParam, iterInfo, aMulBOp);
}

void UpdateIterInfo(const MatmulTileInfo& tileInfo, MatmulIterInfo& iterInfo)
{
    iterInfo.kAL1Size = std::min(tileInfo.tileKAL1, tileInfo.kView - iterInfo.kOffset);
    iterInfo.kBL1Size = std::min(tileInfo.tileKBL1, tileInfo.kView - iterInfo.kOffset);
    iterInfo.kL0Size = std::min(tileInfo.tileKL0, tileInfo.kView - iterInfo.kOffset);
    iterInfo.isFirstK = (iterInfo.kOffset == 0);
    iterInfo.isLastK = (iterInfo.kOffset + tileInfo.tileKL0 >= tileInfo.kView);

    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, tileInfo.tileKAL1 > 0 && tileInfo.tileKBL1 > 0)
        << "Both tileKAL1 and tileKBL1 must be positive: tileKAL1: " << tileInfo.tileKAL1
        << ", tileKBL1: " << tileInfo.tileKBL1;
}

void ConstructTileGraph(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& operandVec,
                        const LogicalTensorPtr& cTensorPtr, const Operation& op)
{
    MATMUL_LOGD("ConstructTileGraph: Start.");
    MatmulAttrParam attrParam;
    SetMatmulAttrParam(op, attrParam);
    // tensor graph中的数据节点
    MatmulGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, attrParam, tensorGraphNodes);
    MatmulTileInfo tileInfo;
    SetMatmulTileInfo(tileShape, attrParam, tensorGraphNodes, tileInfo);
    MatmulIterInfo iterInfo;
    // tile graph中的数据节点
    MatmulGraphNodes tileGraphNodes;

    for (iterInfo.nOffset = 0; iterInfo.nOffset < tileInfo.nView; iterInfo.nOffset += tileInfo.tileNL0) {
        iterInfo.nL0Size = std::min(tileInfo.nView - iterInfo.nOffset, tileInfo.tileNL0);
        for (iterInfo.mOffset = 0; iterInfo.mOffset < tileInfo.mView; iterInfo.mOffset += tileInfo.tileML0) {
            tileGraphNodes.biasTensorPtr = LinkBias(function, tensorGraphNodes,
                                                    TileInfo({{1, iterInfo.nL0Size}, {0, iterInfo.nOffset}}),
                                                    TileInfo({{1, iterInfo.nL0Size}, {0, 0}}));
            tileGraphNodes.scaleTensorPtr = LinkScale(function, tensorGraphNodes,
                                                      TileInfo({{1, iterInfo.nL0Size}, {0, iterInfo.nOffset}}),
                                                      TileInfo({{1, iterInfo.nL0Size}, {0, 0}}));
            iterInfo.mL0Size = std::min(tileInfo.mView - iterInfo.mOffset, tileInfo.tileML0);
            tileGraphNodes.outTensorPtr = cTensorPtr->View(function, {iterInfo.mL0Size, iterInfo.nL0Size},
                                                           {iterInfo.mOffset, iterInfo.nOffset});
            LogicalTensorPtr aL1TensorPtr = nullptr;
            LogicalTensorPtr bL1TensorPtr = nullptr;
            LogicalTensorPtr aScaleL1TensorPtr = nullptr;
            LogicalTensorPtr bScaleL1TensorPtr = nullptr;
            for (iterInfo.kOffset = 0; iterInfo.kOffset < tileInfo.kView; iterInfo.kOffset += tileInfo.tileKL0) {
                UpdateIterInfo(tileInfo, iterInfo);
                tileGraphNodes.aTensorPtr = LinkTensorA(function, tensorGraphNodes, attrParam, tileInfo, iterInfo,
                                                        aL1TensorPtr);
                tileGraphNodes.bTensorPtr = LinkTensorB(function, tensorGraphNodes, attrParam, tileInfo, iterInfo,
                                                        bL1TensorPtr);
                if (attrParam.hasMXScale) {
                    tileGraphNodes.aScaleTensorPtr = LinkTensorAScale(function, tensorGraphNodes, attrParam, tileInfo,
                                                                      iterInfo, aScaleL1TensorPtr);
                    tileGraphNodes.bScaleTensorPtr = LinkTensorBScale(function, tensorGraphNodes, attrParam, tileInfo,
                                                                      iterInfo, bScaleL1TensorPtr);
                }
                LinkAMulB(function, tensorGraphNodes, attrParam, iterInfo, tileGraphNodes);
            }
        }
    }
    MATMUL_LOGD("ConstructTileGraph: Finish.");
}

void AddAMulBNode(const MatmulGraphNodes& tensorGraphNodes, const MatmulAttrParam& attrParam,
                  const MatmulExtendParam& extendParam = {})
{
    if (CheckValidShape(tensorGraphNodes.aTensorPtr) && CheckValidShape(tensorGraphNodes.bTensorPtr)) {
        SymbolicScalar mSizeDyn = attrParam.transA ? tensorGraphNodes.aTensorPtr->GetDynValidShape()[1] :
                                                     tensorGraphNodes.aTensorPtr->GetDynValidShape()[0];
        SymbolicScalar kSizeDyn = attrParam.transA ? tensorGraphNodes.aTensorPtr->GetDynValidShape()[0] :
                                                     tensorGraphNodes.aTensorPtr->GetDynValidShape()[1];
        SymbolicScalar nSizeDyn = attrParam.transB ? tensorGraphNodes.bTensorPtr->GetDynValidShape()[0] :
                                                     tensorGraphNodes.bTensorPtr->GetDynValidShape()[1];
        ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.outTensorPtr != nullptr)
            << "cTensorPtr is nullptr.";
        tensorGraphNodes.outTensorPtr->UpdateDynValidShape({mSizeDyn, nSizeDyn});
    }

    std::vector<LogicalTensorPtr> operandVec = {tensorGraphNodes.aTensorPtr, tensorGraphNodes.bTensorPtr};
    bool gmAccumulationFlag = false;
    if (attrParam.hasMXScale) {
        operandVec.push_back(tensorGraphNodes.aScaleTensorPtr);
        operandVec.push_back(tensorGraphNodes.bScaleTensorPtr);
    }
    if (tensorGraphNodes.gmAccumulationTensorPtr != nullptr) {
        operandVec.push_back(tensorGraphNodes.gmAccumulationTensorPtr);
        gmAccumulationFlag = true;
    }
    if (extendParam.biasTensor.GetStorage() != nullptr) {
        operandVec.push_back(extendParam.biasTensor.GetStorage());
    }
    if (extendParam.scaleTensor.GetStorage() != nullptr) {
        auto scaleTensorDType = extendParam.scaleTensor.GetStorage()->Datatype();
        ASSERT(MatmulErrorCode::ERR_PARAM_UNSUPPORTED,
               scaleTensorDType == DataType::DT_UINT64 || scaleTensorDType == DataType::DT_INT64)
            << "Unsupported scaleTensor data type. Only support DT_UINT64 and DT_INT64";
        operandVec.push_back(extendParam.scaleTensor.GetStorage());
    }
    Function* functionPtr = Program::GetInstance().GetCurrentFunction();

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, functionPtr != nullptr) << "functionPtr is nullptr.";
    auto& op = functionPtr->AddOperation(Opcode::OP_A_MUL_B, operandVec, {tensorGraphNodes.outTensorPtr});
    SetTensorGraphAttr(op, extendParam, gmAccumulationFlag, attrParam);
    // GEMV：非 transA 且 M==1；transA 退化普通 matmul
    if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) && !attrParam.transA) {
        const auto& aShape = tensorGraphNodes.aTensorPtr->GetShape();
        if (aShape.size() >= SHAPE_DIM2 && aShape[aShape.size() - SHAPE_DIM2] == 1) {
            op.SetAttribute(OpAttributeKey::isGemv, true);
        }
    }
}

Tensor ConstructTensorGraph(DataType dataType, MatmulGraphNodes& tensorGraphNodes, const MatmulAttrParam& attrParam,
                            const MatmulExtendParam& param = {})
{
    MATMUL_LOGD("ConstructTensorGraph: Start.");
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.aTensorPtr != nullptr) << "aTensorPtr is nullptr.";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.bTensorPtr != nullptr) << "bTensorPtr is nullptr.";
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, tensorGraphNodes.aTensorPtr->GetShape().size() >= SHAPE_DIM2)
        << "The dimension of aTensor must be >= 2! The dimension of aTensor: "
        << tensorGraphNodes.aTensorPtr->GetShape().size();
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, tensorGraphNodes.bTensorPtr->GetShape().size() >= SHAPE_DIM2)
        << "The dimension of bTensor must be >= 2! The dimension of bTensor: "
        << tensorGraphNodes.bTensorPtr->GetShape().size();
    int64_t mSize = attrParam.transA ? tensorGraphNodes.aTensorPtr->GetShape()[1] :
                                       tensorGraphNodes.aTensorPtr->GetShape()[0];
    int64_t kSizeA = attrParam.transA ? tensorGraphNodes.aTensorPtr->GetShape()[0] :
                                        tensorGraphNodes.aTensorPtr->GetShape()[1];
    int64_t kSizeB = attrParam.transB ? tensorGraphNodes.bTensorPtr->GetShape()[1] :
                                        tensorGraphNodes.bTensorPtr->GetShape()[0];
    int64_t nSize = attrParam.transB ? tensorGraphNodes.bTensorPtr->GetShape()[0] :
                                       tensorGraphNodes.bTensorPtr->GetShape()[1];

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, kSizeA == kSizeB)
        << "Matrix K dimension mismatch, kSizeA: " << kSizeA << ", kSizeB: " << kSizeB;
    Tensor cMatrix(dataType, {mSize, nSize}, "TensorC");
    if (attrParam.isCMatrixNZ) {
        ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, BytesOf(dataType) > 0)
            << "BytesOf(dataType): " << BytesOf(dataType) << ". Must be positive.";
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        cMatrix = Tensor(dataType, {mSize, CeilAlign(nSize, c0Size)}, "TensorC", TileOpFormat::TILEOP_NZ);
    }
    tensorGraphNodes.outTensorPtr = cMatrix.GetStorage();
    AddAMulBNode(tensorGraphNodes, attrParam, param);
    return cMatrix;
}

// 根据UB大小设置VecTile
static void SetVecTileBasedOnUbSize(DataType outType, const CubeTile& cubeTile)
{
    uint64_t ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    // Add的两个输入矩阵总大小不能超过UB大小限制
    if (cubeTile.m[0] * cubeTile.n[0] * BytesOf(outType) * 2 <= ubSize || outType == DT_INT32) {
        TileShape::Current().SetVecTile({cubeTile.m[0], cubeTile.n[0]});
    } else {
        TileShape::Current().SetVecTile({VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    }
}

static Tensor GetGmDeterministicAccumulationTensor(std::vector<Tensor> gmPartialSums, int64_t kLoop)
{
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, gmPartialSums.size() == static_cast<uint64_t>(kLoop))
        << "GmPartialSums' size mismatch kLoop.";
    for (int64_t kIdx = 1; kIdx < kLoop; ++kIdx) {
        gmPartialSums[0] = npu::tile_fwk::Add(gmPartialSums[0], gmPartialSums[kIdx]);
    }
    return gmPartialSums[0];
}

static Tensor ConstructGmAccumulationTensorGraph(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix,
                                                 const MatmulAttrParam& attrParam,
                                                 const MatmulExtendParam& extendParam = {})
{
    auto& cubeTile = TileShape::Current().GetCubeTile();
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
        << "Both aMatrix and bMatrix cannot get storage";
    auto aMatrixValidShape = aMatrix.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = bMatrix.GetStorage()->GetDynValidShape();
    SymbolicScalar mValidShape = attrParam.transA ? aMatrixValidShape[1] : aMatrixValidShape[0];
    SymbolicScalar nValidShape = attrParam.transB ? bMatrixValidShape[0] : bMatrixValidShape[1];
    SymbolicScalar kValidShape = attrParam.transA ? aMatrixValidShape[0] : aMatrixValidShape[1];
    SymbolicScalar kL1TileShape = std::min(cubeTile.k[1], cubeTile.k[2]);
    auto oriVecTile = TileShape::Current().GetVecTile();
    int64_t mSize = attrParam.transA ? aMatrix.GetShape()[1] : aMatrix.GetShape()[0];
    int64_t kSize = attrParam.transA ? aMatrix.GetShape()[0] : aMatrix.GetShape()[1];
    int64_t nSize = attrParam.transB ? bMatrix.GetShape()[0] : bMatrix.GetShape()[1];

    SetVecTileBasedOnUbSize(outType, cubeTile);
    Tensor gmAccumulationTensor = (outType == DT_INT32) ? Full(Element(outType, static_cast<int64_t>(0)), outType,
                                                               {mSize, nSize}, {mValidShape, nValidShape}) :
                                                          Tensor();
    std::vector<Tensor> gmPartialSums;
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, kL1TileShape != 0) << "kL1TileShape can not be 0";
    const int64_t kLoop = (kSize + kL1TileShape - 1) / kL1TileShape;
    const int64_t kL1Size = std::min(kSize, kL1TileShape);
    for (int64_t kIdx = 0; kIdx < kLoop; ++kIdx) {
        int64_t kViewShape = std::min(kSize - kL1Size * kIdx, kL1Size);
        SymbolicScalar
            partialKValidShape = (kValidShape - kL1Size * kIdx).Max(SymbolicScalar(0)).Min(SymbolicScalar(kViewShape));
        Tensor tensorA;
        if (attrParam.transA) {
            tensorA = View(aMatrix, {kViewShape, mSize}, {partialKValidShape, mValidShape}, {kL1Size * kIdx, 0});
        } else {
            tensorA = View(aMatrix, {mSize, kViewShape}, {mValidShape, partialKValidShape}, {0, kL1Size * kIdx});
        }
        Tensor tensorB;
        if (attrParam.transB) {
            tensorB = View(bMatrix, {nSize, kViewShape}, {nValidShape, partialKValidShape}, {0, kL1Size * kIdx});
        } else {
            tensorB = View(bMatrix, {kViewShape, nSize}, {partialKValidShape, nValidShape}, {kL1Size * kIdx, 0});
        }
        MatmulGraphNodes tensorGraphNodes(tensorA.GetStorage(), tensorB.GetStorage(),
                                          gmAccumulationTensor.GetStorage());
        Tensor gmPartialSum = ConstructTensorGraph(outType, tensorGraphNodes, attrParam, extendParam);
        gmPartialSums.emplace_back(gmPartialSum);
    }
    if (outType == DT_INT32) {
        gmAccumulationTensor = npu::tile_fwk::Reduce(gmPartialSums, ReduceMode::ATOMIC_ADD);
        ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, gmAccumulationTensor.GetStorage() != nullptr)
            << "ReduceAcc's result can not be null.";
        gmAccumulationTensor.GetStorage()->UpdateDynValidShape({mValidShape, nValidShape});
    } else {
        gmAccumulationTensor = GetGmDeterministicAccumulationTensor(gmPartialSums, kLoop);
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return gmAccumulationTensor;
}

Tensor Matmul(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix, bool isATrans, bool isBTrans,
              bool isCMatrixNZ)
{
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    MATMUL_LOGD("Matmul[Basic]: Start.");
    Status checkStatus = CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    CHECK(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "Matmul operands check failed";
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), bMatrix.GetStorage());
    auto& cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("Matmul: Using GM accumulation mode.");
        return ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam);
    }
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam);
}

Tensor Matmul(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix, const MatmulExtendParam& param,
              bool isATrans, bool isBTrans, bool isCMatrixNZ)
{
    MATMUL_LOGD("Matmul[Extend]: Start.");
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    Status checkStatus = CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam, param);
    CHECK(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "Matmul operands check failed";
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), bMatrix.GetStorage());
    auto& cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("Matmul: Using GM accumulation mode.");
        return ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam, param);
    }
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam, param);
}

static Tensor ConstructMXGmAccumulationTensorGraph(DataType outType, const Tensor& aMatrix, const Tensor& aScale,
                                                   const Tensor& bMatrix, const Tensor& bScale,
                                                   const MatmulAttrParam& attrParam)
{
    auto& cubeTile = TileShape::Current().GetCubeTile();
    auto aMatrixValidShape = aMatrix.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = bMatrix.GetStorage()->GetDynValidShape();
    SymbolicScalar mValidShape = attrParam.transA ? aMatrixValidShape[1] : aMatrixValidShape[0];
    SymbolicScalar nValidShape = attrParam.transB ? bMatrixValidShape[0] : bMatrixValidShape[1];
    SymbolicScalar kValidShape = attrParam.transA ? aMatrixValidShape[0] : aMatrixValidShape[1];
    SymbolicScalar kL1TileShape = std::min(cubeTile.k[1], cubeTile.k[2]);
    int64_t mSize = attrParam.transA ? aMatrix.GetShape()[1] : aMatrix.GetShape()[0];
    int64_t kSize = attrParam.transA ? aMatrix.GetShape()[0] : aMatrix.GetShape()[1];
    int64_t nSize = attrParam.transB ? bMatrix.GetShape()[0] : bMatrix.GetShape()[1];
    auto oriVecTile = TileShape::Current().GetVecTile();
    SetVecTileBasedOnUbSize(outType, cubeTile);
    std::vector<Tensor> gmPartialSums;
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, kL1TileShape != 0) << "kL1TileShape can not be 0";
    const int64_t kLoop = (kSize + kL1TileShape - 1) / kL1TileShape;
    const int64_t kL1Size = std::min(kSize, kL1TileShape);
    const int64_t kScaleL1Size = CeilDiv(kL1Size, ALIGN_SIZE_64);
    for (int64_t kIdx = 0; kIdx < kLoop; ++kIdx) {
        int64_t kViewShape = std::min(kSize - kL1Size * kIdx, kL1Size);
        int64_t kScaleViewShape = CeilDiv(kViewShape, ALIGN_SIZE_64);
        SymbolicScalar
            partialKValidShape = (kValidShape - kL1Size * kIdx).Max(SymbolicScalar(0)).Min(SymbolicScalar(kViewShape));
        SymbolicScalar partialkScaleValidShape = (partialKValidShape + ALIGN_SIZE_64 - 1) / ALIGN_SIZE_64;
        Tensor tensorA;
        if (attrParam.transA) {
            tensorA = View(aMatrix, {kViewShape, mSize}, {partialKValidShape, mValidShape}, {kL1Size * kIdx, 0});
        } else {
            tensorA = View(aMatrix, {mSize, kViewShape}, {mValidShape, partialKValidShape}, {0, kL1Size * kIdx});
        }
        Tensor scaleA;
        if (attrParam.transAScale) {
            scaleA = View(aScale, {kScaleViewShape, mSize, SHAPE_DIM2},
                          {partialkScaleValidShape, mValidShape, SHAPE_DIM2}, {kScaleL1Size * kIdx, 0, 0});
        } else {
            scaleA = View(aScale, {mSize, kScaleViewShape, SHAPE_DIM2},
                          {mValidShape, partialkScaleValidShape, SHAPE_DIM2}, {0, kScaleL1Size * kIdx, 0});
        }
        Tensor tensorB;
        if (attrParam.transB) {
            tensorB = View(bMatrix, {nSize, kViewShape}, {nValidShape, partialKValidShape}, {0, kL1Size * kIdx});
        } else {
            tensorB = View(bMatrix, {kViewShape, nSize}, {partialKValidShape, nValidShape}, {kL1Size * kIdx, 0});
        }
        Tensor scaleB;
        if (attrParam.transBScale) {
            scaleB = View(bScale, {nSize, kScaleViewShape, SHAPE_DIM2},
                          {nValidShape, partialkScaleValidShape, SHAPE_DIM2}, {0, kScaleL1Size * kIdx, 0});
        } else {
            scaleB = View(bScale, {kScaleViewShape, nSize, SHAPE_DIM2},
                          {partialkScaleValidShape, nValidShape, SHAPE_DIM2}, {kScaleL1Size * kIdx, 0, 0});
        }
        MatmulGraphNodes tensorGraphNodes(tensorA.GetStorage(), scaleA.GetStorage(), tensorB.GetStorage(),
                                          scaleB.GetStorage());
        Tensor gmPartialSum = ConstructTensorGraph(outType, tensorGraphNodes, attrParam);
        gmPartialSums.emplace_back(gmPartialSum);
    }
    for (int64_t kIdx = 1; kIdx < kLoop; ++kIdx) {
        gmPartialSums[0] = npu::tile_fwk::Add(gmPartialSums[0], gmPartialSums[kIdx]);
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return gmPartialSums[0];
}

Tensor MatmulMX(DataType outType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                const Tensor& bScale, bool isATrans, bool isAScaleTrans, bool isBTrans, bool isBScaleTrans,
                bool isCMatrixNZ)
{
    MATMUL_LOGD("MatmulMX[Basic]: Start.");
    MatmulAttrParam attrParam(isATrans, isAScaleTrans, isBTrans, isBScaleTrans, isCMatrixNZ);
    CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    CheckMXMatmulOperands(aMatrix, aScale, bMatrix, bScale, attrParam);
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), aScale.GetStorage(), bMatrix.GetStorage(),
                                      bScale.GetStorage());
    auto& cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("Matmul[Basic]: Using GM accumulation mode.");
        return ConstructMXGmAccumulationTensorGraph(outType, aMatrix, aScale, bMatrix, bScale, attrParam);
    }
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam);
}

Tensor MatmulMX(DataType outType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                const Tensor& bScale, const MatmulExtendParam& param, bool isATrans, bool isAScaleTrans, bool isBTrans,
                bool isBScaleTrans, bool isCMatrixNZ)
{
    MATMUL_LOGD("MatmulMX[Extend]: Start.");
    MatmulAttrParam attrParam(isATrans, isAScaleTrans, isBTrans, isBScaleTrans, isCMatrixNZ);
    CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam, param);
    CheckMXMatmulOperands(aMatrix, aScale, bMatrix, bScale, attrParam);
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), aScale.GetStorage(), bMatrix.GetStorage(),
                                      bScale.GetStorage());
    auto& cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("Matmul[Extend]: Using GM accumulation mode.");
        return ConstructMXGmAccumulationTensorGraph(outType, aMatrix, aScale, bMatrix, bScale, attrParam);
    }
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam, param);
}

void CheckABatchMulB(const Tensor& operand1, const Tensor& operand2, const MatmulExtendParam& param = {})
{
    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, operand1.GetStorage() != nullptr) << "A Tensor cannot be null";
    CHECK(MatmulErrorCode::ERR_RUNTIME_NULLPTR, operand2.GetStorage() != nullptr) << "B Tensor cannot be null";

    CHECK(ExternalError::INVALID_SHAPE, operand1.GetShape().size() == operand2.GetShape().size())
        << "The dimensions of operand1 and operand2 must be equal.";

    CHECK(ExternalError::INVALID_SHAPE,
          operand1.GetShape().size() == SHAPE_DIM3 || operand1.GetShape().size() == SHAPE_DIM4)
        << "Batch matmul only support 3 dimensions or 4 dimensions.";

    auto aMatrixValidShape = operand1.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = operand2.GetStorage()->GetDynValidShape();
    CHECK(ExternalError::INVALID_SHAPE, aMatrixValidShape.size() == operand1.GetShape().size() &&
                                            bMatrixValidShape.size() == operand2.GetShape().size())
        << "The input valid shape dimensions of BatchMatmul must match their shape dimensions.";

    for (uint64_t bIdx = 0; bIdx < operand1.GetShape().size() - SHAPE_DIM2; bIdx++) {
        const int64_t batchSizeA = operand1.GetShape()[bIdx];
        const int64_t batchSizeB = operand2.GetShape()[bIdx];
        CHECK(ExternalError::INVALID_SHAPE, batchSizeA == batchSizeB || batchSizeB == 1 || batchSizeA == 1)
            << "batchSize invalid: A" << bIdx << "= B" << bIdx << "or 1 allowed. A" << bIdx << ": " << batchSizeA
            << ", B" << bIdx << ": " << batchSizeB;
    }
    if (param.biasTensor.GetStorage() != nullptr && param.biasTensor.GetShape().size() != 2) {
        CHECK(ExternalError::INVALID_SHAPE, param.biasTensor.GetShape().size() == operand1.GetShape().size())
            << "Batch of bias does not match input tensor's batch";
        for (uint64_t bIdx = 0; bIdx < operand1.GetShape().size() - SHAPE_DIM2; bIdx++) {
            const int64_t batchSize = std::max(operand1.GetShape()[bIdx], operand2.GetShape()[bIdx]);
            const int64_t biasBatchSize = param.biasTensor.GetShape()[bIdx];
            CHECK(ExternalError::INVALID_SHAPE, batchSize == biasBatchSize || biasBatchSize == 1)
                << "bias batch size invalid: out" << bIdx << " = bias" << bIdx << " or 1 allowed. out" << bIdx << ": "
                << batchSize << ", bias" << bIdx << ": " << biasBatchSize;
        }
    }
    if (param.scaleTensor.GetStorage() != nullptr && param.scaleTensor.GetShape().size() != 2) {
        CHECK(ExternalError::INVALID_SHAPE, param.scaleTensor.GetShape().size() == operand1.GetShape().size())
            << "Batch of scale does not match input tensor's batch";
        for (uint64_t bIdx = 0; bIdx < operand1.GetShape().size() - SHAPE_DIM2; bIdx++) {
            const int64_t batchSize = std::max(operand1.GetShape()[bIdx], operand2.GetShape()[bIdx]);
            const int64_t scaleBatchSize = param.scaleTensor.GetShape()[bIdx];
            CHECK(ExternalError::INVALID_SHAPE, batchSize == scaleBatchSize || scaleBatchSize == 1)
                << "scale batch size invalid: out" << bIdx << " = scale" << bIdx << " or 1 allowed. out" << bIdx << ": "
                << batchSize << ", scale" << bIdx << ": " << scaleBatchSize;
        }
    }
}

std::pair<Tensor, Tensor> GetBatchTensor3D(int64_t batchSize, int64_t bIdx, const Tensor& operand,
                                           std::vector<SymbolicScalar>& validShape3D)
{
    int64_t offsetBatch = batchSize == 1 ? 0 : bIdx;
    Tensor tensorSingleBatch = View(operand, {1, operand.GetShape()[1], operand.GetShape()[SHAPE_DIM2]},
                                    GetViewValidShape(validShape3D, {offsetBatch, 0, 0}, {},
                                                      {1, operand.GetShape()[1], operand.GetShape()[SHAPE_DIM2]}),
                                    {offsetBatch, 0, 0});
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorSingleBatch.GetStorage() != nullptr)
        << "tensorSingleBatch 3D of View result must not be null";
    Tensor tensor = Reshape(tensorSingleBatch, {operand.GetShape()[1], operand.GetShape()[SHAPE_DIM2]},
                            std::vector<SymbolicScalar>(
                                {tensorSingleBatch.GetStorage()->GetDynValidShape()[0] * validShape3D[1],
                                 tensorSingleBatch.GetStorage()->GetDynValidShape()[0] * validShape3D[SHAPE_DIM2]}));
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensor.GetStorage() != nullptr)
        << "tensorSingleBatch 3D of Reshape result must not be null";
    return {tensor, tensorSingleBatch};
}

void SingleBatch3D(int64_t bIdx, int64_t batchSizeA, int64_t batchSizeB, int64_t mView, int64_t nView,
                   DataType dataType, const Tensor& operand1, const Tensor& operand2, const MatmulAttrParam& attrParam,
                   const MatmulExtendParam& extendParam, Tensor& result, std::vector<SymbolicScalar>& aValidShape3D,
                   std::vector<SymbolicScalar>& bValidShape3D)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, result.GetStorage() != nullptr)
        << "bacth matmul 3D's result must not be null";
    auto [aTensor, aTensorSingleBatch] = GetBatchTensor3D(batchSizeA, bIdx, operand1, aValidShape3D);
    auto [bTensor, bTensorSingleBatch] = GetBatchTensor3D(batchSizeB, bIdx, operand2, bValidShape3D);

    const auto mValid = attrParam.transA ? aValidShape3D[SHAPE_DIM2] : aValidShape3D[1];
    const auto nValid = attrParam.transB ? bValidShape3D[1] : bValidShape3D[SHAPE_DIM2];
    const auto bValid = batchSizeA == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[0] :
                                          aTensorSingleBatch.GetStorage()->GetDynValidShape()[0];
    Tensor cTensor(dataType, {mView, nView}, "cTensorSingleBatch");
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, cTensor.GetStorage() != nullptr) << "cTensor must not be null";
    cTensor.GetStorage()->UpdateDynValidShape({bValid * mValid, nValid});

    MatmulExtendParam batchParam;
    MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), bTensor.GetStorage());
    tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
    batchParam.scaleValue = extendParam.scaleValue;
    batchParam.transMode = extendParam.transMode;
    batchParam.reluType = extendParam.reluType;
    batchParam.scaleTensor = extendParam.scaleTensor;
    const Tensor scaleOperand = extendParam.scaleTensor;
    if (scaleOperand.GetStorage() != nullptr && scaleOperand.GetShape().size() == SHAPE_DIM3) {
        batchParam.scaleTensor = GetBatchTensor3D(scaleOperand.GetShape()[0], bIdx, scaleOperand,
                                                  scaleOperand.GetStorage()->GetDynValidShape())
                                     .first;
    }
    batchParam.biasTensor = extendParam.biasTensor;
    const Tensor biasOperand = extendParam.biasTensor;
    if (biasOperand.GetStorage() != nullptr && biasOperand.GetShape().size() == SHAPE_DIM3) {
        batchParam.biasTensor = GetBatchTensor3D(biasOperand.GetShape()[0], bIdx, biasOperand,
                                                 biasOperand.GetStorage()->GetDynValidShape())
                                    .first;
    }
    AddAMulBNode(tensorGraphNodes, attrParam, batchParam);
    Tensor cTensor3D = Reshape(cTensor, {1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
                               std::vector<SymbolicScalar>({bValid, mValid, nValid}));
    Assemble(cTensor3D, {bIdx, 0, 0}, result);
    result.GetStorage()->UpdateDynValidShape({std::max(aValidShape3D[0], bValidShape3D[0]), mValid, nValid});
}

void CheckBatchMatmulMXBias(const Tensor& aMatrix, const Tensor& bMatrix, const MatmulExtendParam& param)
{
    const Tensor& biasOperand = param.biasTensor;
    if (biasOperand.GetStorage() == nullptr) {
        return;
    }

    if (aMatrix.GetShape().size() == SHAPE_DIM3) {
        CHECK(ExternalError::INVALID_SHAPE,
              biasOperand.GetShape().size() == SHAPE_DIM2 || biasOperand.GetShape().size() == SHAPE_DIM3)
            << "3D BatchMatmulMX only supports 2D or 3D bias tensor, but got " << biasOperand.GetShape().size()
            << "D bias";
    } else if (aMatrix.GetShape().size() == SHAPE_DIM4) {
        CHECK(ExternalError::INVALID_SHAPE, biasOperand.GetShape().size() == SHAPE_DIM2)
            << "4D BatchMatmulMX only supports 2D bias tensor, but got " << biasOperand.GetShape().size() << "D bias";
    }

    if (biasOperand.GetShape().size() == SHAPE_DIM3) {
        const int64_t batchInput = std::max(aMatrix.GetShape()[0], bMatrix.GetShape()[0]);
        CHECK(ExternalError::INVALID_SHAPE, biasOperand.GetShape()[0] == batchInput)
            << "Bias batch dim must equal to input batch dim. Bias batch: " << biasOperand.GetShape()[0]
            << ", Input batch: " << batchInput;
    }

    if (biasOperand.GetShape().size() == SHAPE_DIM2) {
        CHECK(ExternalError::INVALID_SHAPE, biasOperand.GetShape()[0] == 1)
            << "2D bias must be [1, n] shape, first dim must be 1, but got " << biasOperand.GetShape()[0];
    } else if (biasOperand.GetShape().size() == SHAPE_DIM3) {
        CHECK(ExternalError::INVALID_SHAPE, biasOperand.GetShape()[SHAPE_DIM1] == 1)
            << "3D bias must be [b, 1, n] shape, second dim must be 1, but got " << biasOperand.GetShape()[1];
    }
}

Tensor ConstructBatchMatmulTensorGraph3D(DataType dataType, const Tensor& operand1, const Tensor& operand2,
                                         const MatmulAttrParam& attrParam, const MatmulExtendParam& extendParam = {})
{
    const int64_t batchSizeA = operand1.GetShape()[0];
    const int64_t batchSizeB = operand2.GetShape()[0];
    const int64_t batchSize = std::max(batchSizeA, batchSizeB);

    const int64_t mView = attrParam.transA ? operand1.GetShape()[SHAPE_DIM2] : operand1.GetShape()[1];
    const int64_t nView = attrParam.transB ? operand2.GetShape()[1] : operand2.GetShape()[SHAPE_DIM2];
    Tensor result = Tensor(dataType, {batchSize, mView, nView});
    auto oriVecTile = TileShape::Current().GetVecTile();
    const Tensor biasOperand = extendParam.biasTensor;
    const Tensor scaleOperand = extendParam.scaleTensor;
    TileShape::Current().SetVecTile({1, VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    auto aValidShape3D = operand1.GetStorage()->GetDynValidShape();
    auto bValidShape3D = operand2.GetStorage()->GetDynValidShape();

    for (int64_t bIdx = 0; bIdx < batchSize; bIdx++) {
        SingleBatch3D(bIdx, batchSizeA, batchSizeB, mView, nView, dataType, operand1, operand2, attrParam, extendParam,
                      result, aValidShape3D, bValidShape3D);
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

std::pair<Tensor, Tensor> GetBatchTensor4D(int64_t batchSize1, int64_t batchSize2, int64_t bIdx1, int64_t bIdx2,
                                           const Tensor& operand, std::vector<SymbolicScalar>& validShape4D)
{
    int64_t offsetBatch1 = batchSize1 == 1 ? 0 : bIdx1;
    int64_t offsetBatch2 = batchSize2 == 1 ? 0 : bIdx2;
    Tensor tensorSingleBatch = View(
        operand, {1, 1, operand.GetShape()[SHAPE_DIM2], operand.GetShape()[SHAPE_DIM3]},
        GetViewValidShape(validShape4D, {offsetBatch1, offsetBatch2, 0, 0}, {},
                          {1, 1, operand.GetShape()[SHAPE_DIM2], operand.GetShape()[SHAPE_DIM3]}),
        {offsetBatch1, offsetBatch2, 0, 0});
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorSingleBatch.GetStorage() != nullptr)
        << "tensorSingleBatch 4D of View result must not be null";
    Tensor tensor = Reshape(
        tensorSingleBatch, {operand.GetShape()[SHAPE_DIM2], operand.GetShape()[SHAPE_DIM3]},
        std::vector<SymbolicScalar>(
            {tensorSingleBatch.GetStorage()->GetDynValidShape()[0] *
                 tensorSingleBatch.GetStorage()->GetDynValidShape()[1] * validShape4D[SHAPE_DIM2],
             tensorSingleBatch.GetStorage()->GetDynValidShape()[0] *
                 tensorSingleBatch.GetStorage()->GetDynValidShape()[1] * validShape4D[SHAPE_DIM3]}));
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensor.GetStorage() != nullptr)
        << "tensorSingleBatch 4D of Reshape result must not be null";
    return {tensor, tensorSingleBatch};
}

void SingleBatch4D(int64_t bIdx1, int64_t bIdx2, int64_t batchSizeA1, int64_t batchSizeA2, int64_t batchSizeB1,
                   int64_t batchSizeB2, int64_t mView, int64_t nView, DataType dataType, const Tensor& operand1,
                   const Tensor& operand2, const MatmulAttrParam& attrParam, const MatmulExtendParam& extendParam,
                   Tensor& result, std::vector<SymbolicScalar>& aValidShape4D,
                   std::vector<SymbolicScalar>& bValidShape4D)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, result.GetStorage() != nullptr)
        << "bacth matmul 4D's result must not be null";
    auto [aTensor, aTensorSingleBatch] = GetBatchTensor4D(batchSizeA1, batchSizeA2, bIdx1, bIdx2, operand1,
                                                          aValidShape4D);
    auto [bTensor, bTensorSingleBatch] = GetBatchTensor4D(batchSizeB1, batchSizeB2, bIdx1, bIdx2, operand2,
                                                          bValidShape4D);

    const auto mValid = attrParam.transA ? aValidShape4D[SHAPE_DIM3] : aValidShape4D[SHAPE_DIM2];
    const auto nValid = attrParam.transB ? bValidShape4D[SHAPE_DIM2] : bValidShape4D[SHAPE_DIM3];
    const auto b0Valid = batchSizeA1 == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[0] :
                                            aTensorSingleBatch.GetStorage()->GetDynValidShape()[0];
    const auto b1Valid = batchSizeA2 == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[1] :
                                            aTensorSingleBatch.GetStorage()->GetDynValidShape()[1];
    Tensor cTensor(dataType, {mView, nView}, "cTensorSingleBatch");
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, cTensor.GetStorage() != nullptr) << "cTensor must not be null";
    cTensor.GetStorage()->UpdateDynValidShape({b0Valid * b1Valid * mValid, nValid});

    MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), bTensor.GetStorage());
    tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
    MatmulExtendParam batchParam;
    batchParam.reluType = extendParam.reluType;
    batchParam.scaleValue = extendParam.scaleValue;
    batchParam.transMode = extendParam.transMode;
    batchParam.biasTensor = extendParam.biasTensor;
    batchParam.scaleTensor = extendParam.scaleTensor;
    const Tensor scaleOperand = extendParam.scaleTensor;
    if (scaleOperand.GetStorage() != nullptr && scaleOperand.GetShape().size() == SHAPE_DIM4) {
        batchParam.scaleTensor = GetBatchTensor4D(scaleOperand.GetShape()[0], scaleOperand.GetShape()[1], bIdx1, bIdx2,
                                                  scaleOperand, scaleOperand.GetStorage()->GetDynValidShape())
                                     .first;
    }
    AddAMulBNode(tensorGraphNodes, attrParam, batchParam);
    Tensor cTensor4D = Reshape(cTensor, {1, 1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
                               std::vector<SymbolicScalar>({b0Valid, b1Valid, mValid, nValid}));
    Assemble(cTensor4D, {bIdx1, bIdx2, 0, 0}, result);
    result.GetStorage()->UpdateDynValidShape(
        {std::max(aValidShape4D[0], bValidShape4D[0]), std::max(aValidShape4D[1], bValidShape4D[1]), mValid, nValid});
}

Tensor ConstructBatchMatmulTensorGraph4D(DataType dataType, const Tensor& operand1, const Tensor& operand2,
                                         const MatmulAttrParam& attrParam, const MatmulExtendParam& extendParam = {})
{
    const int64_t batchSizeA1 = operand1.GetShape()[0];
    const int64_t batchSizeA2 = operand1.GetShape()[1];
    const int64_t batchSizeB1 = operand2.GetShape()[0];
    const int64_t batchSizeB2 = operand2.GetShape()[1];
    const int64_t batchSize1 = std::max(batchSizeA1, batchSizeB1);
    const int64_t batchSize2 = std::max(batchSizeA2, batchSizeB2);
    const int64_t mView = attrParam.transA ? operand1.GetShape()[SHAPE_DIM3] : operand1.GetShape()[SHAPE_DIM2];
    const int64_t nView = attrParam.transB ? operand2.GetShape()[SHAPE_DIM2] : operand2.GetShape()[SHAPE_DIM3];
    Tensor result = Tensor(dataType, {batchSize1, batchSize2, mView, nView});
    auto oriVecTile = TileShape::Current().GetVecTile();
    const Tensor biasOperand = extendParam.biasTensor;
    const Tensor scaleOperand = extendParam.scaleTensor;
    TileShape::Current().SetVecTile({1, 1, VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    auto aValidShape4D = operand1.GetStorage()->GetDynValidShape();
    auto bValidShape4D = operand2.GetStorage()->GetDynValidShape();
    for (int64_t bIdx1 = 0; bIdx1 < batchSize1; bIdx1++) {
        for (int64_t bIdx2 = 0; bIdx2 < batchSize2; bIdx2++) {
            SingleBatch4D(bIdx1, bIdx2, batchSizeA1, batchSizeA2, batchSizeB1, batchSizeB2, mView, nView, dataType,
                          operand1, operand2, attrParam, extendParam, result, aValidShape4D, bValidShape4D);
        }
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

static Tensor ComputeGmAccumulationKernel(DataType outType, const Tensor& aTensor2D, const Tensor& bTensor2D,
                                          const MatmulAttrParam& attrParam, const MatmulExtendParam& extendParam,
                                          const SymbolicScalar& mValidShape, const SymbolicScalar& nValidShape,
                                          const SymbolicScalar& kValidShape, const SymbolicScalar& kL1TileShape,
                                          int64_t mSize2D, int64_t kSize2D, int64_t nSize2D,
                                          const SymbolicScalar& bValidShape, const SymbolicScalar& ndMValidShape,
                                          const SymbolicScalar& ndNValidShape)
{
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, kL1TileShape != 0) << "kL1TileShape can not be 0";
    Tensor gmAccumulationTensor = (outType == DT_INT32) ?
                                      Full(Element(outType, static_cast<int64_t>(0)), outType, {mSize2D, nSize2D},
                                           {bValidShape * ndMValidShape, ndNValidShape}) :
                                      Tensor();
    std::vector<Tensor> gmPartialSums;
    const int64_t kLoop = (kSize2D + kL1TileShape - 1) / kL1TileShape;
    const int64_t kL1Size = std::min(kSize2D, kL1TileShape);
    for (int64_t kIdx = 0; kIdx < kLoop; ++kIdx) {
        SymbolicScalar
            partialKValidShape = (kValidShape - kL1Size * kIdx).Max(SymbolicScalar(0)).Min(SymbolicScalar(kL1Size));
        Tensor tensorA = attrParam.transA ? View(aTensor2D, {kL1Size, mSize2D}, {partialKValidShape, mValidShape},
                                                 {kL1Size * kIdx, 0}) :
                                            View(aTensor2D, {mSize2D, kL1Size}, {mValidShape, partialKValidShape},
                                                 {0, kL1Size * kIdx});
        Tensor tensorB = attrParam.transB ? View(bTensor2D, {nSize2D, kL1Size}, {nValidShape, partialKValidShape},
                                                 {0, kL1Size * kIdx}) :
                                            View(bTensor2D, {kL1Size, nSize2D}, {partialKValidShape, nValidShape},
                                                 {kL1Size * kIdx, 0});
        MatmulGraphNodes tensorGraphNodes(tensorA.GetStorage(), tensorB.GetStorage(),
                                          gmAccumulationTensor.GetStorage());
        Tensor gmPartialSum = ConstructTensorGraph(outType, tensorGraphNodes, attrParam, extendParam);
        ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, gmPartialSum.GetStorage() != nullptr)
            << "tmp tensor of kSplit must not be null";
        gmPartialSum.GetStorage()->UpdateDynValidShape({bValidShape * ndMValidShape, ndNValidShape});
        gmPartialSums.emplace_back(gmPartialSum);
    }
    Tensor cTensor(outType, {mSize2D, nSize2D}, "batchResult2D");
    if (outType == DT_INT32) {
        cTensor = npu::tile_fwk::Reduce(gmPartialSums, ReduceMode::ATOMIC_ADD);
    } else {
        cTensor = GetGmDeterministicAccumulationTensor(gmPartialSums, kLoop);
    }
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, cTensor.GetStorage() != nullptr) << "result cTensor must not be null";
    cTensor.GetStorage()->UpdateDynValidShape({bValidShape * ndMValidShape, ndNValidShape});
    return cTensor;
}

static Tensor ConstructSingleBatchGmAccumulation(DataType outType, const Tensor& aTensor2D, const Tensor& bTensor2D,
                                                 const MatmulAttrParam& attrParam, const MatmulExtendParam& extendParam,
                                                 const CubeTile& cubeTile, const SymbolicScalar& bValidShape,
                                                 const SymbolicScalar& ndMValidShape,
                                                 const SymbolicScalar& ndNValidShape)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aTensor2D.GetStorage() != nullptr)
        << "aTensor2D storage must not be null";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, bTensor2D.GetStorage() != nullptr)
        << "bTensor2D storage must not be null";
    auto aValidShape2D = aTensor2D.GetStorage()->GetDynValidShape();
    auto bValidShape2D = bTensor2D.GetStorage()->GetDynValidShape();
    SymbolicScalar mValidShape = attrParam.transA ? aValidShape2D[1] : aValidShape2D[0];
    SymbolicScalar nValidShape = attrParam.transB ? bValidShape2D[0] : bValidShape2D[1];
    SymbolicScalar kValidShape = attrParam.transA ? aValidShape2D[0] : aValidShape2D[1];
    SymbolicScalar kL1TileShape = std::min(cubeTile.k[1], cubeTile.k[2]);
    int64_t mSize2D = attrParam.transA ? aTensor2D.GetShape()[1] : aTensor2D.GetShape()[0];
    int64_t kSize2D = attrParam.transA ? aTensor2D.GetShape()[0] : aTensor2D.GetShape()[1];
    int64_t nSize2D = attrParam.transB ? bTensor2D.GetShape()[0] : bTensor2D.GetShape()[1];
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, kL1TileShape != 0) << "kL1TileShape can not be 0";
    SetVecTileBasedOnUbSize(outType, cubeTile);
    return ComputeGmAccumulationKernel(outType, aTensor2D, bTensor2D, attrParam, extendParam, mValidShape, nValidShape,
                                       kValidShape, kL1TileShape, mSize2D, kSize2D, nSize2D, bValidShape, ndMValidShape,
                                       ndNValidShape);
}

static Tensor ConstructBatchGmAccumulationTensorGraph3D(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix,
                                                        const MatmulAttrParam& attrParam,
                                                        const MatmulExtendParam& extendParam = {})
{
    auto& cubeTile = TileShape::Current().GetCubeTile();
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
        << "Matrix A and Matrix B must not be null";
    const int64_t batchSizeA = aMatrix.GetShape()[0];
    const int64_t batchSizeB = bMatrix.GetShape()[0];
    const int64_t batchSize = std::max(batchSizeA, batchSizeB);
    const int64_t mView = attrParam.transA ? aMatrix.GetShape()[SHAPE_DIM2] : aMatrix.GetShape()[1];
    const int64_t nView = attrParam.transB ? bMatrix.GetShape()[1] : bMatrix.GetShape()[SHAPE_DIM2];
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH,
           (attrParam.transA ? aMatrix.GetShape()[1] : aMatrix.GetShape()[SHAPE_DIM2]) ==
               (attrParam.transB ? bMatrix.GetShape()[SHAPE_DIM2] : bMatrix.GetShape()[1]))
        << "Matrix K dimension mismatch";
    Tensor result = Tensor(outType, {batchSize, mView, nView});
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, result.GetStorage() != nullptr)
        << "bacth matmul 3D's result must not be null when using KSplit.";
    auto oriVecTile = TileShape::Current().GetVecTile();
    TileShape::Current().SetVecTile({1, VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    auto aValidShape3D = aMatrix.GetStorage()->GetDynValidShape();
    auto bValidShape3D = bMatrix.GetStorage()->GetDynValidShape();
    SymbolicScalar mValidShape = attrParam.transA ? aValidShape3D[SHAPE_DIM2] : aValidShape3D[1];
    SymbolicScalar nValidShape = attrParam.transB ? bValidShape3D[1] : bValidShape3D[SHAPE_DIM2];
    SymbolicScalar kL1TileShape = std::min(cubeTile.k[1], cubeTile.k[2]);
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, kL1TileShape != 0) << "kL1TileShape can not be 0";
    for (int64_t bIdx = 0; bIdx < batchSize; ++bIdx) {
        auto [aTensor2D, aTensorSingleBatch] = GetBatchTensor3D(batchSizeA, bIdx, aMatrix, aValidShape3D);
        auto [bTensor2D, bTensorSingleBatch] = GetBatchTensor3D(batchSizeB, bIdx, bMatrix, bValidShape3D);
        SymbolicScalar bValid = batchSizeA == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[0] :
                                                  aTensorSingleBatch.GetStorage()->GetDynValidShape()[0];
        Tensor cTensor = ConstructSingleBatchGmAccumulation(outType, aTensor2D, bTensor2D, attrParam, extendParam,
                                                            cubeTile, bValid, mValidShape, nValidShape);
        Tensor batchResult3D = Reshape(cTensor, {1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
                                       std::vector<SymbolicScalar>({bValid, mValidShape, nValidShape}));
        Assemble(batchResult3D, {bIdx, 0, 0}, result);
    }
    result.GetStorage()->UpdateDynValidShape({std::max(aValidShape3D[0], bValidShape3D[0]), mValidShape, nValidShape});
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

static Tensor ConstructBatchGmAccumulationTensorGraph4D(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix,
                                                        const MatmulAttrParam& attrParam,
                                                        const MatmulExtendParam& extendParam = {})
{
    auto& cubeTile = TileShape::Current().GetCubeTile();
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
        << "Both aMatrix and bMatrix cannot get storage";
    const int64_t batchSizeA1 = aMatrix.GetShape()[0], batchSizeA2 = aMatrix.GetShape()[1];
    const int64_t batchSizeB1 = bMatrix.GetShape()[0], batchSizeB2 = bMatrix.GetShape()[1];
    const int64_t batchSize1 = std::max(batchSizeA1, batchSizeB1);
    const int64_t batchSize2 = std::max(batchSizeA2, batchSizeB2);
    const int64_t mView = attrParam.transA ? aMatrix.GetShape()[SHAPE_DIM3] : aMatrix.GetShape()[SHAPE_DIM2];
    const int64_t nView = attrParam.transB ? bMatrix.GetShape()[SHAPE_DIM2] : bMatrix.GetShape()[SHAPE_DIM3];
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH,
           (attrParam.transA ? aMatrix.GetShape()[SHAPE_DIM2] : aMatrix.GetShape()[SHAPE_DIM3]) ==
               (attrParam.transB ? bMatrix.GetShape()[SHAPE_DIM3] : bMatrix.GetShape()[SHAPE_DIM2]))
        << "Matrix K dimension mismatch";
    Tensor result = Tensor(outType, {batchSize1, batchSize2, mView, nView});
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, result.GetStorage() != nullptr)
        << "bacth matmul 4D's result must not be null using KSplit.";
    auto oriVecTile = TileShape::Current().GetVecTile();
    TileShape::Current().SetVecTile({1, 1, VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    auto aValidShape4D = aMatrix.GetStorage()->GetDynValidShape();
    auto bValidShape4D = bMatrix.GetStorage()->GetDynValidShape();
    SymbolicScalar mValidShape = attrParam.transA ? aValidShape4D[SHAPE_DIM3] : aValidShape4D[SHAPE_DIM2];
    SymbolicScalar nValidShape = attrParam.transB ? bValidShape4D[SHAPE_DIM2] : bValidShape4D[SHAPE_DIM3];
    SymbolicScalar kL1TileShape = std::min(cubeTile.k[1], cubeTile.k[2]);
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, kL1TileShape != 0) << "kL1TileShape can not be 0";
    for (int64_t bIdx1 = 0; bIdx1 < batchSize1; ++bIdx1) {
        for (int64_t bIdx2 = 0; bIdx2 < batchSize2; ++bIdx2) {
            auto [aTensor2D, aTensorSingleBatch] = GetBatchTensor4D(batchSizeA1, batchSizeA2, bIdx1, bIdx2, aMatrix,
                                                                    aValidShape4D);
            auto [bTensor2D, bTensorSingleBatch] = GetBatchTensor4D(batchSizeB1, batchSizeB2, bIdx1, bIdx2, bMatrix,
                                                                    bValidShape4D);
            SymbolicScalar b0Valid = batchSizeA1 == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[0] :
                                                        aTensorSingleBatch.GetStorage()->GetDynValidShape()[0];
            SymbolicScalar b1Valid = batchSizeA2 == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[1] :
                                                        aTensorSingleBatch.GetStorage()->GetDynValidShape()[1];
            Tensor cTensor = ConstructSingleBatchGmAccumulation(outType, aTensor2D, bTensor2D, attrParam, extendParam,
                                                                cubeTile, b0Valid * b1Valid, mValidShape, nValidShape);
            Tensor batchResult4D = Reshape(cTensor, {1, 1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
                                           std::vector<SymbolicScalar>({b0Valid, b1Valid, mValidShape, nValidShape}));
            Assemble(batchResult4D, {bIdx1, bIdx2, 0, 0}, result);
        }
    }
    result.GetStorage()->UpdateDynValidShape({std::max(aValidShape4D[0], bValidShape4D[0]),
                                              std::max(aValidShape4D[1], bValidShape4D[1]), mValidShape, nValidShape});
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

Tensor BatchMatmul(DataType dataType, const Tensor& aMatrix, const Tensor& bMatrix, const bool isTransA,
                   const bool isTransB, const bool isCMatrixNZ)
{
    MatmulAttrParam attrParam(isTransA, isTransB, isCMatrixNZ);
    CheckMatmulOperands(dataType, aMatrix, bMatrix, attrParam);
    CheckABatchMulB(aMatrix, bMatrix);
    auto& cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("BatchMatmul: Using GM accumulation mode.");
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            return ConstructBatchGmAccumulationTensorGraph4D(dataType, aMatrix, bMatrix, attrParam);
        } else {
            return ConstructBatchGmAccumulationTensorGraph3D(dataType, aMatrix, bMatrix, attrParam);
        }
    }
    if (aMatrix.GetShape().size() == SHAPE_DIM4) {
        return ConstructBatchMatmulTensorGraph4D(dataType, aMatrix, bMatrix, attrParam);
    } else {
        return ConstructBatchMatmulTensorGraph3D(dataType, aMatrix, bMatrix, attrParam);
    }
}

Tensor BatchMatmul(DataType dataType, const Tensor& aMatrix, const Tensor& bMatrix, const MatmulExtendParam& param,
                   const bool isTransA, const bool isTransB, const bool isCMatrixNZ)
{
    MatmulAttrParam attrParam(isTransA, isTransB, isCMatrixNZ);
    CheckMatmulOperands(dataType, aMatrix, bMatrix, attrParam, param);
    CheckABatchMulB(aMatrix, bMatrix, param);
    auto& cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("BatchMatmul: Using GM accumulation mode.");
        if (aMatrix.GetShape().size() == SHAPE_DIM4) {
            return ConstructBatchGmAccumulationTensorGraph4D(dataType, aMatrix, bMatrix, attrParam, param);
        } else {
            return ConstructBatchGmAccumulationTensorGraph3D(dataType, aMatrix, bMatrix, attrParam, param);
        }
    }
    if (aMatrix.GetShape().size() == SHAPE_DIM4) {
        return ConstructBatchMatmulTensorGraph4D(dataType, aMatrix, bMatrix, attrParam, param);
    } else {
        return ConstructBatchMatmulTensorGraph3D(dataType, aMatrix, bMatrix, attrParam, param);
    }
}

Tensor ConstructBatchMatmulMXTensorGraph3D(DataType dataType, const Tensor& aMatrix, const Tensor& aScale,
                                           const Tensor& bMatrix, const Tensor& bScale,
                                           const MatmulAttrParam& attrParam, const MatmulExtendParam& param = {})
{
    const int64_t batchSizeA = aMatrix.GetShape()[0];
    const int64_t batchSizeB = bMatrix.GetShape()[0];
    const int64_t batchSize = std::max(batchSizeA, batchSizeB);

    const int64_t mView = attrParam.transA ? aMatrix.GetShape()[SHAPE_DIM2] : aMatrix.GetShape()[1];
    const int64_t nView = attrParam.transB ? bMatrix.GetShape()[1] : bMatrix.GetShape()[SHAPE_DIM2];
    Tensor result = Tensor(dataType, {batchSize, mView, nView});
    auto oriVecTile = TileShape::Current().GetVecTile();
    const Tensor biasOperand = param.biasTensor;
    const Tensor scaleOperand = param.scaleTensor;
    TileShape::Current().SetVecTile({1, VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    auto aScaleValidShape4D = aScale.GetStorage()->GetDynValidShape();
    auto bScaleValidShape4D = bScale.GetStorage()->GetDynValidShape();
    auto aValidShape3D = aMatrix.GetStorage()->GetDynValidShape();
    auto bValidShape3D = bMatrix.GetStorage()->GetDynValidShape();

    const auto mValid = attrParam.transA ? aValidShape3D[SHAPE_DIM2] : aValidShape3D[1];
    const auto nValid = attrParam.transB ? bValidShape3D[1] : bValidShape3D[SHAPE_DIM2];

    for (int64_t bIdx = 0; bIdx < batchSize; bIdx++) {
        auto [aTensor, aTensorSingleBatch] = GetBatchTensor3D(batchSizeA, bIdx, aMatrix, aValidShape3D);
        auto [bTensor, bTensorSingleBatch] = GetBatchTensor3D(batchSizeB, bIdx, bMatrix, bValidShape3D);

        const auto bValid = batchSizeA == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[0] :
                                              aTensorSingleBatch.GetStorage()->GetDynValidShape()[0];

        Tensor cTensor(dataType, {mView, nView}, "cTensorSingleBatch");
        cTensor.GetStorage()->UpdateDynValidShape({bValid * mValid, nValid});
        MatmulExtendParam batchParam = param;
        if (biasOperand.GetStorage() != nullptr && biasOperand.GetShape().size() == SHAPE_DIM3) {
            int64_t offsetBatchBias = biasOperand.GetShape()[0] == 1 ? 0 : bIdx;
            auto biasValidShape3D = biasOperand.GetStorage()->GetDynValidShape();

            Tensor biasTensorSingleBatch = View(
                biasOperand, {1, biasOperand.GetShape()[1], biasOperand.GetShape()[SHAPE_DIM2]},
                std::vector<SymbolicScalar>({1, biasValidShape3D[1], biasValidShape3D[SHAPE_DIM2]}),
                {offsetBatchBias, 0, 0});

            Tensor biasTensor = Reshape(
                biasTensorSingleBatch, {biasOperand.GetShape()[1], biasOperand.GetShape()[SHAPE_DIM2]},
                std::vector<SymbolicScalar>({biasValidShape3D[1], biasValidShape3D[SHAPE_DIM2]}));
            batchParam.biasTensor = biasTensor;
        } else if (biasOperand.GetStorage() != nullptr && biasOperand.GetShape().size() == SHAPE_DIM2) {
            batchParam.biasTensor = biasOperand;
        }
        if (scaleOperand.GetStorage() != nullptr) {
            batchParam.scaleTensor = scaleOperand;
        }
        batchParam.reluType = param.reluType;
        batchParam.scaleValue = param.scaleValue;
        MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), aScale.GetStorage(), bTensor.GetStorage(),
                                          bScale.GetStorage());
        tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
        AddAMulBNode(tensorGraphNodes, attrParam, batchParam);
        ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, cTensor.GetStorage() != nullptr)
            << "result tensor must not be null";
        auto cValidShape2D = cTensor.GetStorage()->GetDynValidShape();
        Tensor cTensor3D = Reshape(cTensor, {1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
                                   std::vector<SymbolicScalar>({bValid, mValid, nValid}));
        Assemble(cTensor3D, {bIdx, 0, 0}, result);
        result.GetStorage()->UpdateDynValidShape({std::max(aValidShape3D[0], bValidShape3D[0]), mValid, nValid});
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

Tensor ConstructBatchMatmulMXTensorGraph4D(DataType dataType, const Tensor& aMatrix, const Tensor& aScale,
                                           const Tensor& bMatrix, const Tensor& bScale,
                                           const MatmulAttrParam& attrParam, const MatmulExtendParam& param = {})
{
    const int64_t batchSizeA1 = aMatrix.GetShape()[0];
    const int64_t batchSizeA2 = aMatrix.GetShape()[1];
    const int64_t batchSizeB1 = bMatrix.GetShape()[0];
    const int64_t batchSizeB2 = bMatrix.GetShape()[1];
    const int64_t batchSize1 = std::max(batchSizeA1, batchSizeB1);
    const int64_t batchSize2 = std::max(batchSizeA2, batchSizeB2);

    const int64_t mView = attrParam.transA ? aMatrix.GetShape()[SHAPE_DIM3] : aMatrix.GetShape()[SHAPE_DIM2];
    const int64_t nView = attrParam.transB ? bMatrix.GetShape()[SHAPE_DIM2] : bMatrix.GetShape()[SHAPE_DIM3];
    Tensor result = Tensor(dataType, {batchSize1, batchSize2, mView, nView});

    const Tensor biasOperand = param.biasTensor;
    const Tensor scaleOperand = param.scaleTensor;
    auto oriVecTile = TileShape::Current().GetVecTile();
    TileShape::Current().SetVecTile({1, 1, VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    auto aValidShape4D = aMatrix.GetStorage()->GetDynValidShape();
    auto bValidShape4D = bMatrix.GetStorage()->GetDynValidShape();
    auto aScaleValidShape5D = aScale.GetStorage()->GetDynValidShape();
    auto bScaleValidShape5D = bScale.GetStorage()->GetDynValidShape();
    const auto mValid = attrParam.transA ? aValidShape4D[SHAPE_DIM3] : aValidShape4D[SHAPE_DIM2];
    const auto nValid = attrParam.transB ? bValidShape4D[SHAPE_DIM2] : bValidShape4D[SHAPE_DIM3];

    for (int64_t bIdx1 = 0; bIdx1 < batchSize1; bIdx1++) {
        for (int64_t bIdx2 = 0; bIdx2 < batchSize2; bIdx2++) {
            auto [aTensor, aTensorSingleBatch] = GetBatchTensor4D(batchSizeA1, batchSizeA2, bIdx1, bIdx2, aMatrix,
                                                                  aValidShape4D);
            auto [bTensor, bTensorSingleBatch] = GetBatchTensor4D(batchSizeB1, batchSizeB2, bIdx1, bIdx2, bMatrix,
                                                                  bValidShape4D);

            const auto b0Valid = batchSizeA1 == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[0] :
                                                    aTensorSingleBatch.GetStorage()->GetDynValidShape()[0];
            const auto b1Valid = batchSizeA2 == 1 ? bTensorSingleBatch.GetStorage()->GetDynValidShape()[1] :
                                                    aTensorSingleBatch.GetStorage()->GetDynValidShape()[1];

            MatmulExtendParam batchParam = param;
            if (biasOperand.GetStorage() != nullptr) {
                batchParam.biasTensor = biasOperand;
            }
            if (scaleOperand.GetStorage() != nullptr) {
                batchParam.scaleTensor = scaleOperand;
            }
            batchParam.reluType = param.reluType;
            batchParam.scaleValue = param.scaleValue;
            Tensor cTensor(dataType, {mView, nView}, "cTensorSingleBatch");
            cTensor.GetStorage()->UpdateDynValidShape({b0Valid * b1Valid * mValid, nValid});
            MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), aScale.GetStorage(), bTensor.GetStorage(),
                                              bScale.GetStorage());
            tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
            AddAMulBNode(tensorGraphNodes, attrParam, batchParam);
            ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, cTensor.GetStorage() != nullptr)
                << "result tensor must not be null";
            auto cValidShape2D = cTensor.GetStorage()->GetDynValidShape();
            Tensor cTensor4D = Reshape(cTensor, {1, 1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
                                       std::vector<SymbolicScalar>({b0Valid, b1Valid, mValid, nValid}));
            Assemble(cTensor4D, {bIdx1, bIdx2, 0, 0}, result);
            result.GetStorage()->UpdateDynValidShape({std::max(aValidShape4D[0], bValidShape4D[0]),
                                                      std::max(aValidShape4D[1], bValidShape4D[1]), mValid, nValid});
        }
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

Tensor BatchMatmulMX(DataType dataType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                     const Tensor& bScale, const bool isTransA, bool isAScaleTrans, const bool isTransB,
                     bool isBScaleTrans, const bool isCMatrixNZ)
{
    MATMUL_LOGD("BatchMatmulMX[Basic]: Start.");
    MatmulAttrParam attrParam(isTransA, isAScaleTrans, isTransB, isBScaleTrans, isCMatrixNZ);
    Status checkStatus = CheckMatmulOperands(dataType, aMatrix, bMatrix, attrParam);
    CHECK(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "MXMatmul operands check failed";
    Status checkMXStatus = CheckMXMatmulOperands(aMatrix, aScale, bMatrix, bScale, attrParam);
    CHECK(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkMXStatus == SUCCESS) << "MXMatmul operands check failed";
    CheckABatchMulB(aMatrix, bMatrix);
    if (aMatrix.GetShape().size() == SHAPE_DIM4) {
        return ConstructBatchMatmulMXTensorGraph4D(dataType, aMatrix, aScale, bMatrix, bScale, attrParam);
    } else {
        return ConstructBatchMatmulMXTensorGraph3D(dataType, aMatrix, aScale, bMatrix, bScale, attrParam);
    }
}

Tensor BatchMatmulMX(DataType dataType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                     const Tensor& bScale, const MatmulExtendParam& param, const bool isTransA, bool isAScaleTrans,
                     const bool isTransB, bool isBScaleTrans, const bool isCMatrixNZ)
{
    MATMUL_LOGD("BatchMatmulMX[Basic]: Start.");
    MatmulAttrParam attrParam(isTransA, isAScaleTrans, isTransB, isBScaleTrans, isCMatrixNZ);
    Status checkStatus = CheckMatmulOperands(dataType, aMatrix, bMatrix, attrParam, param);
    CHECK(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "MXMatmul operands check failed";
    Status checkMXStatus = CheckMXMatmulOperands(aMatrix, aScale, bMatrix, bScale, attrParam);
    CHECK(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkMXStatus == SUCCESS) << "MXMatmul operands check failed";
    CheckABatchMulB(aMatrix, bMatrix);
    CheckBatchMatmulMXBias(aMatrix, bMatrix, param);
    if (aMatrix.GetShape().size() == SHAPE_DIM4) {
        return ConstructBatchMatmulMXTensorGraph4D(dataType, aMatrix, aScale, bMatrix, bScale, attrParam, param);
    } else {
        return ConstructBatchMatmulMXTensorGraph3D(dataType, aMatrix, aScale, bMatrix, bScale, attrParam, param);
    }
}

// 定制接口：用于Transpose + BMM + Transpose融合场景
// 当前仅支持：(M, B, K) @ (B, K, N) -> (M, B, N)
Tensor TransposedBatchMatmul(DataType dataType, const Tensor& aMatrix, const Tensor& bMatrix)
{
    CHECK(ExternalError::INVALID_SHAPE,
          aMatrix.GetShape().size() == SHAPE_DIM3 && bMatrix.GetShape().size() == SHAPE_DIM3)
        << "TransposedBatchMatmul only support 3-dim inputs, aMatrix dim: " << aMatrix.GetShape().size()
        << ", bMatrix dim: " << bMatrix.GetShape().size();
    const int64_t mSize = aMatrix.GetShape()[0];
    const int64_t batchSizeA = aMatrix.GetShape()[1];
    const int64_t kaSize = aMatrix.GetShape()[SHAPE_DIM2];
    const int64_t batchSizeB = bMatrix.GetShape()[0];
    const int64_t kbSize = bMatrix.GetShape()[1];
    const int64_t nSize = bMatrix.GetShape()[SHAPE_DIM2];
    CHECK(ExternalError::INVALID_SHAPE, batchSizeA == batchSizeB)
        << "batchSize invalid, expect batchSizeA = batchSizeB, given batchSizeA: " << batchSizeA
        << ", batchSizeB: " << batchSizeB;

    CHECK(ExternalError::INVALID_SHAPE, kaSize == kbSize)
        << "kSize invalid, expect kaSize = kbSize, given kaSize: " << kaSize << ", kbSize: " << kbSize;

    // 128: custom tile shape size
    auto oriVecTile = TileShape::Current().GetVecTile();
    TileShape::Current().SetVecTile({1, VECTOR_TILE_SHAPE, VECTOR_TILE_SHAPE});
    Tensor aMatrixFused = Reshape(aMatrix, {mSize, batchSizeA * kaSize});
    Tensor cMatrix(dataType, {mSize, batchSizeA * nSize});
    for (int64_t bIdx = 0; bIdx < batchSizeA; ++bIdx) {
        Tensor aTensor = View(aMatrixFused, {mSize, kaSize}, std::vector<SymbolicScalar>({mSize, kaSize}),
                              {0, bIdx * kaSize});
        Tensor bTensorSingleBatch = View(bMatrix, {1, kbSize, nSize}, std::vector<SymbolicScalar>({1, kbSize, nSize}),
                                         {bIdx, 0, 0});
        Tensor bTensor = Reshape(bTensorSingleBatch, {kbSize, nSize});
        Tensor cTensor(dataType, {mSize, nSize}, "TensorC");
        MatmulAttrParam attrParam(false, false, false);
        MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), bTensor.GetStorage());
        tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
        AddAMulBNode(tensorGraphNodes, attrParam);
        Assemble(cTensor, {0, bIdx * nSize}, cMatrix);
    }
    Tensor result = Reshape(cMatrix, {mSize, batchSizeA, nSize});
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}
} // namespace Matrix
} // namespace tile_fwk
} // namespace npu
