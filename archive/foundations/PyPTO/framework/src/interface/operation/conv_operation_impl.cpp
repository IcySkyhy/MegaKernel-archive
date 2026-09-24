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
 * \file conv_operation_impl.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/conv/conv_utils.h"
#include "interface/operation/conv/conv_vec_tile_inference.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

const std::string LoadStoreConvOpAttributeKey::cutW = "CUT_W";
const std::string LoadStoreConvOpAttributeKey::realCutW = "REAL_CUT_W";
const std::string LoadStoreConvOpAttributeKey::copyInMode = "COPY_IN_MODE";
const std::string LoadStoreConvOpAttributeKey::copyOutMode = "COPY_OUT_MODE";
const std::string LoadStoreConvOpAttributeKey::isFmap = "IS_FMAP";
const std::string LoadStoreConvOpAttributeKey::isConv3D = "IS_CONV3D";
const std::string LoadStoreConvOpAttributeKey::reluType = "RELU_TYPE";

bool IsArch32Platform() { return Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201; }
std::vector<int64_t> rotateVector(const std::vector<int64_t>& input, size_t shift)
{
    std::vector<int64_t> result = input;
    std::rotate(result.begin(), result.begin() + shift, result.end());
    return result;
}

void CheckValueRange(int64_t value, const std::string& name, int64_t min, int64_t max, const std::string& formula = "")
{
    std::ostringstream oss;
    oss << "Invalid " << name << ":" << value << ", expected range [" << min << "," << max << "].";
    if (!formula.empty()) {
        oss << "Formula: " << formula;
    }
    CHECK(ExternalError::OUT_OF_RANGE, value >= min && value <= max) << oss.str();
}

int64_t ConvAlignB(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

void SetTensorOpAttr(Operation& op, const LogicalTensorPtr& inputTensor, const LogicalTensorPtr& weightTensor,
                     const LogicalTensorPtr& resTensor, const ConvAttrParam& convAttrParam)
{
    op.SetAttribute(CONV_BIAS_ATTR, convAttrParam.hasBias);
    op.SetAttribute(CONV_GROUPS_ATTR, convAttrParam.groups);
    op.SetAttribute(CONV_PADDINGS_ATTR, convAttrParam.paddings);
    op.SetAttribute(CONV_STRIDES_ATTR, convAttrParam.strides);
    op.SetAttribute(CONV_DILATIONS_ATTR, convAttrParam.dilations);
    op.SetAttribute(CONV_3D_FLAG, convAttrParam.isConv3D);
    op.SetAttribute(CONV_ORI_FMAP_SHAPE_ATTR, inputTensor->GetShape());
    op.SetAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR, weightTensor->GetShape());
    op.SetAttribute(CONV_ORI_RES_SHAPE_ATTR, resTensor->GetShape());
    op.SetAttribute("dynamicResValidShape", resTensor->GetDynValidShape());
    op.SetAttribute(CONV_RELU_ATTR, convAttrParam.reluType);
}

std::vector<LogicalTensorPtr> GetOperandVecIn(std::vector<LogicalTensorPtr> operandVecIn,
                                              const ConvAttrParam& convAttrParam)
{
    int64_t cin0 = ALIGN_SIZE_32 / BytesOf(operandVecIn[INPUT_FMAP_IDX]->Datatype());
    int64_t batch = operandVecIn[INPUT_FMAP_IDX]->GetShape()[NCHW_N_IDX];
    int64_t hi = convAttrParam.isConv3D ? operandVecIn[INPUT_FMAP_IDX]->GetShape()[NCDHW_H_IDX] :
                                          operandVecIn[INPUT_FMAP_IDX]->GetShape()[NCHW_H_IDX];
    int64_t wi = convAttrParam.isConv3D ? operandVecIn[INPUT_FMAP_IDX]->GetShape()[NCDHW_W_IDX] :
                                          operandVecIn[INPUT_FMAP_IDX]->GetShape()[NCHW_W_IDX];
    int64_t cout = operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_N_IDX];
    int64_t cinPerGroup = operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_C_IDX];
    int64_t kh = convAttrParam.isConv3D ? operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_H_IDX] :
                                          operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_H_IDX];
    int64_t kw = convAttrParam.isConv3D ? operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_W_IDX] :
                                          operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_W_IDX];
    int64_t cin1PerGroup = CeilDiv(cinPerGroup, cin0);
    int64_t cout1PerGroup = CeilDiv(cout / convAttrParam.groups, MKN_N_VALUE);
    std::vector<int64_t> inputNzShape = {batch, convAttrParam.groups * cin1PerGroup, hi, wi, cin0};
    std::vector<int64_t> weightFzShape = {convAttrParam.groups * cin1PerGroup * kh * kw, cout1PerGroup, MKN_N_VALUE,
                                          cin0};
    TileOpFormat inputNzFormat = TileOpFormat::TILEOP_NC1HWC0;
    TileOpFormat weightFzFormat = TileOpFormat::TILEOP_FRACTAL_Z;
    if (convAttrParam.isConv3D) {
        inputNzFormat = TileOpFormat::TILEOP_NDC1HWC0;
        weightFzFormat = TileOpFormat::TILEOP_FRACTAL_Z_3D;
        int64_t din = operandVecIn[INPUT_FMAP_IDX]->GetShape()[NCDHW_D_IDX];
        int64_t kd = operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_D_IDX];
        inputNzShape = {batch, din, convAttrParam.groups * cin1PerGroup, hi, wi, cin0};
        weightFzShape = {convAttrParam.groups * kd * cin1PerGroup * kh * kw, cout1PerGroup, MKN_N_VALUE, cin0};
    }
    Tensor inputNzTensor(operandVecIn[INPUT_FMAP_IDX]->Datatype(), inputNzShape, "TensorInputNz", inputNzFormat);
    Tensor weightFzTensor(operandVecIn[INPUT_FMAP_IDX]->Datatype(), weightFzShape, "TensorWeightFz", weightFzFormat);
    return {inputNzTensor.GetStorage(), weightFzTensor.GetStorage()};
}

Tensor GetFinalResTensorNZ2NZ(std::vector<LogicalTensorPtr> operandVecIn, const Tensor& resTensor,
                              const ConvAttrParam& convAttrParam)
{
    std::vector<int64_t> orgOutShape = {resTensor.GetShape()[NC1HWC0_N_IDX],
                                        operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_N_IDX],
                                        resTensor.GetShape()[NC1HWC0_H_IDX], resTensor.GetShape()[NC1HWC0_W_IDX]};
    if (convAttrParam.isConv3D) {
        orgOutShape = {resTensor.GetShape()[NDC1HWC0_N_IDX], operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_N_IDX],
                       resTensor.GetShape()[NDC1HWC0_D_IDX], resTensor.GetShape()[NDC1HWC0_H_IDX],
                       resTensor.GetShape()[NDC1HWC0_W_IDX]};
    }
    Tensor finalResTensor(resTensor.GetStorage()->Datatype(), orgOutShape, "TensorOut");
    std::vector<SymbolicScalar> resValidShape = resTensor.GetValidShape();
    SymbolicScalar validCout = operandVecIn[INPUT_WEIGHT_IDX]->GetDynValidShape()[NCHW_N_IDX];
    std::vector<SymbolicScalar> finalResValidShape = {resValidShape[NC1HWC0_N_IDX], validCout,
                                                      resValidShape[NC1HWC0_H_IDX], resValidShape[NC1HWC0_W_IDX]};
    if (convAttrParam.isConv3D) {
        finalResValidShape = {resValidShape[NDC1HWC0_N_IDX], validCout, resValidShape[NDC1HWC0_D_IDX],
                              resValidShape[NDC1HWC0_H_IDX], resValidShape[NDC1HWC0_W_IDX]};
    }
    finalResTensor.GetStorage()->UpdateDynValidShape(finalResValidShape);
    return finalResTensor;
}

Tensor ConstructTensorGraphNZ2NZ(Function* functionPtr, std::vector<LogicalTensorPtr> operandVecIn,
                                 const Tensor& resTensor, const ConvAttrParam& convAttrParam)
{
    std::vector<LogicalTensorPtr> operandVecOut = {resTensor.GetStorage()};
    std::vector<LogicalTensorPtr> operandVecInNZ = GetOperandVecIn(operandVecIn, convAttrParam);

    auto& inputTransOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, {operandVecIn[INPUT_FMAP_IDX]},
                                                   {operandVecInNZ[INPUT_FMAP_IDX]});
    inputTransOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(operandVecIn[INPUT_FMAP_IDX]->Format()));
    inputTransOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR,
                              static_cast<int64_t>(operandVecInNZ[INPUT_FMAP_IDX]->Format()));

    auto& weightTransOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, {operandVecIn[INPUT_WEIGHT_IDX]},
                                                    {operandVecInNZ[INPUT_WEIGHT_IDX]});
    weightTransOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR,
                               static_cast<int64_t>(operandVecIn[INPUT_WEIGHT_IDX]->Format()));
    weightTransOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR,
                               static_cast<int64_t>(operandVecInNZ[INPUT_WEIGHT_IDX]->Format()));

    if (convAttrParam.hasBias) {
        operandVecInNZ.push_back(operandVecIn[INPUT_BIAS_IDX]);
    }

    Opcode convOpCode = convAttrParam.isConv3D ? Opcode::OP_CONV3D : Opcode::OP_CONV2D;
    auto& op = functionPtr->AddOperation(convOpCode, operandVecInNZ, operandVecOut);

    std::vector<int64_t> orgOutShape = {resTensor.GetShape()[NC1HWC0_N_IDX],
                                        operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_N_IDX],
                                        resTensor.GetShape()[NC1HWC0_H_IDX], resTensor.GetShape()[NC1HWC0_W_IDX]};
    if (convAttrParam.isConv3D) {
        orgOutShape = {resTensor.GetShape()[NDC1HWC0_N_IDX], operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_N_IDX],
                       resTensor.GetShape()[NDC1HWC0_D_IDX], resTensor.GetShape()[NDC1HWC0_H_IDX],
                       resTensor.GetShape()[NDC1HWC0_W_IDX]};
    }
    Tensor finalResTensor = GetFinalResTensorNZ2NZ(operandVecIn, resTensor, convAttrParam);
    auto& orgResOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, operandVecOut, {finalResTensor.GetStorage()});
    orgResOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(resTensor.Format()));
    orgResOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR, static_cast<int64_t>(finalResTensor.Format()));
    SetTensorOpAttr(op, operandVecIn[INPUT_FMAP_IDX], operandVecIn[INPUT_WEIGHT_IDX], finalResTensor.GetStorage(),
                    convAttrParam);

    if (convAttrParam.isConv1D) {
        auto vecTiles = Conv::InferConvVecTileShapes(op, operandVecIn[INPUT_FMAP_IDX]->Datatype());
        orgOutShape = {resTensor.GetShape()[NC1HWC0_N_IDX], operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_N_IDX],
                       resTensor.GetShape()[NC1HWC0_W_IDX]};
        Tensor finalRes3DimTensor(resTensor.GetStorage()->Datatype(), orgOutShape, "TensorOut3Dim");
        std::vector<SymbolicScalar> resValidShape = resTensor.GetValidShape();
        finalRes3DimTensor.GetStorage()->UpdateDynValidShape(
            {resValidShape[NC1HWC0_N_IDX], operandVecIn[INPUT_WEIGHT_IDX]->GetDynValidShape()[NCHW_N_IDX],
             resValidShape[NC1HWC0_W_IDX]});
        VecTile savedVecTile = TileShape::Current().GetVecTile();
        TileShape::Current().SetVecTile(Conv::GetReshapeVecTile(vecTiles.outVecTile, false));
        auto& reshapeResOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {finalResTensor.GetStorage()},
                                                       {finalRes3DimTensor.GetStorage()});
        TileShape::Current().SetVecTile(savedVecTile);
        reshapeResOp.SetAttribute(OpAttributeKey::isConv, true);
        return finalRes3DimTensor;
    }
    return finalResTensor;
}

Tensor ConstructTensorGraph(const Tensor& inputTensor, const Tensor& weightTensor, const Tensor& biasTensor,
                            const Tensor& resTensor, ConvAttrParam& convAttrParam)
{
    // add Conv node
    Function* functionPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, functionPtr != nullptr) << "functionPtr is nullptr.";
    std::vector<LogicalTensorPtr> operandVecIn = {inputTensor.GetStorage(), weightTensor.GetStorage()};
    std::vector<LogicalTensorPtr> operandVecOut = {resTensor.GetStorage()};
    if (convAttrParam.isConv1D) {
        // conv1d case, unsqueeze input to NC1W
        auto convTile = TileShape::Current().GetConvTile();
        auto vecTiles = Conv::InferConvVecTileShapes(convTile, inputTensor.GetStorage()->Datatype(),
                                                     inputTensor.GetShape(), weightTensor.GetShape(),
                                                     convAttrParam.isConv3D, true, convAttrParam.groups);
        VecTile savedVecTile = TileShape::Current().GetVecTile();

        std::vector<int64_t> fmap4DimShape{inputTensor.GetShape()[NCHW_N_IDX], inputTensor.GetShape()[NCHW_C_IDX], 1,
                                           inputTensor.GetShape()[NCHW_H_IDX]};
        Tensor fmap4DimTensor(inputTensor.GetStorage()->Datatype(), fmap4DimShape, "", inputTensor.Format());
        TileShape::Current().SetVecTile(Conv::GetReshapeVecTile(vecTiles.fmapVecTile, true));
        auto& reshapeFmapOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {inputTensor.GetStorage()},
                                                        {fmap4DimTensor.GetStorage()});

        std::vector<int64_t> weight4DimShape{weightTensor.GetShape()[NCHW_N_IDX], weightTensor.GetShape()[NCHW_C_IDX],
                                             1, weightTensor.GetShape()[NCHW_H_IDX]};
        Tensor weigth4DimTensor(weightTensor.GetStorage()->Datatype(), weight4DimShape, "", weightTensor.Format());
        std::vector<SymbolicScalar> weightValidShape = weightTensor.GetValidShape();
        weigth4DimTensor.GetStorage()->UpdateDynValidShape(
            {weightValidShape[NCL_N_IDX], weightValidShape[NCL_C_IDX], 1, weightValidShape[NCL_L_IDX]});
        TileShape::Current().SetVecTile(Conv::GetReshapeVecTile(vecTiles.weightVecTile, true));
        auto& reshapeWeightOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {weightTensor.GetStorage()},
                                                          {weigth4DimTensor.GetStorage()});
        TileShape::Current().SetVecTile(savedVecTile);

        reshapeFmapOp.SetAttribute(OpAttributeKey::isConv, true);
        reshapeWeightOp.SetAttribute(OpAttributeKey::isConv, true);
        operandVecIn = {fmap4DimTensor.GetStorage(), weigth4DimTensor.GetStorage()};
        // conv1d case, squeeze output to NCL
        std::vector<int64_t> res4DimShape{inputTensor.GetShape()[NCHW_N_IDX], weightTensor.GetShape()[NCHW_N_IDX], 1,
                                          resTensor.GetShape()[NCHW_H_IDX]};
        Tensor res4DimTensor(resTensor.GetStorage()->Datatype(), res4DimShape, "", resTensor.Format());
        operandVecOut = {res4DimTensor.GetStorage()};
    }
    if (!biasTensor.IsEmpty()) {
        convAttrParam.hasBias = true;
        std::vector<int64_t> bias2DimShape{1, biasTensor.GetShape()[0]};
        Tensor bias2DimTensor(biasTensor.GetStorage()->Datatype(), bias2DimShape, "", biasTensor.Format());
        auto& reshapeBiasOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {biasTensor.GetStorage()},
                                                        {bias2DimTensor.GetStorage()});
        reshapeBiasOp.SetAttribute(OpAttributeKey::isConv, true);
        operandVecIn.push_back(bias2DimTensor.GetStorage());
    }
    if (IsArch32Platform()) {
        return ConstructTensorGraphNZ2NZ(functionPtr, operandVecIn, resTensor, convAttrParam);
    }
    Opcode convOpCode = convAttrParam.isConv3D ? Opcode::OP_CONV3D : Opcode::OP_CONV2D;
    auto& op = functionPtr->AddOperation(convOpCode, operandVecIn, operandVecOut);
    SetTensorOpAttr(op, operandVecIn[INPUT_FMAP_IDX], operandVecIn[INPUT_WEIGHT_IDX], operandVecOut[0], convAttrParam);

    if (convAttrParam.isConv1D) {
        auto convTile = TileShape::Current().GetConvTile();
        auto vecTiles = Conv::InferConvVecTileShapes(convTile, inputTensor.GetStorage()->Datatype(),
                                                     inputTensor.GetShape(), weightTensor.GetShape(),
                                                     convAttrParam.isConv3D, true, convAttrParam.groups);
        VecTile savedVecTile = TileShape::Current().GetVecTile();
        TileShape::Current().SetVecTile(Conv::GetReshapeVecTile(vecTiles.outVecTile, false));
        auto& reshapeResOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, operandVecOut, {resTensor.GetStorage()});
        TileShape::Current().SetVecTile(savedVecTile);
        reshapeResOp.SetAttribute(OpAttributeKey::isConv, true);
    }
    return resTensor;
}

void SetConvAttrParam(const Operation& op, ConvAttrParam& convAttrParam)
{
    convAttrParam.isConv3D = (op.HasAttr(CONV_3D_FLAG)) ? op.GetBoolAttribute(CONV_3D_FLAG) : false;
    convAttrParam.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorIntAttribute(CONV_PADDINGS_ATTR) :
                             convAttrParam.isConv3D           ? CONV3D_ATTR_DEFAULT_LIST :
                                                                CONV2D_PAD_ATTR_DEFAULT_LIST;
    convAttrParam.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorIntAttribute(CONV_STRIDES_ATTR) :
                            convAttrParam.isConv3D          ? CONV3D_ATTR_DEFAULT_LIST :
                                                              CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorIntAttribute(CONV_DILATIONS_ATTR) :
                              convAttrParam.isConv3D            ? CONV3D_ATTR_DEFAULT_LIST :
                                                                  CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.groups = (op.HasAttr(CONV_GROUPS_ATTR)) ? op.GetIntAttribute(CONV_GROUPS_ATTR) : 1;
    convAttrParam.hasBias = (op.HasAttr(CONV_BIAS_ATTR)) ? op.GetBoolAttribute(CONV_BIAS_ATTR) : false;
    convAttrParam.reluType = (op.HasAttr(CONV_RELU_ATTR)) ? op.GetIntAttribute(CONV_RELU_ATTR) : 0;
    convAttrParam.isInOutTensorNZ = false;
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_FMAP_SHAPE_ATTR))
        << "Conv ori fmapshape should be set when InOut Tensor NZ mode.";
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_WEIGHT_SHAPE_ATTR))
        << "Conv ori weightshape should be set when InOut Tensor NZ mode.";
    convAttrParam.oriFmapShape = op.GetVectorIntAttribute(CONV_ORI_FMAP_SHAPE_ATTR);
    convAttrParam.oriWeightShape = op.GetVectorIntAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR);
    convAttrParam.oriResShape = op.GetVectorIntAttribute(CONV_ORI_RES_SHAPE_ATTR);
    convAttrParam.dynValidResShape = op.GetVectorSymbolicScalarAttribute("dynamicResValidShape");
}

void SetTensorGraphNodes(const std::vector<LogicalTensorPtr>& operandVec, const LogicalTensorPtr& cTensorPtr,
                         const ConvAttrParam& convAttrParam, ConvGraphNodes& tensorGraphNodes)
{
    // set tensor GraphNodes
    size_t operandVecSize = SHAPE_DIM2 + static_cast<size_t>(convAttrParam.hasBias);
    ASSERT(ConvExpandFuncError::EXPANDFUNC_PARAMS_INVALID, operandVec.size() == operandVecSize)
        << "Operand vector size mismatch: "
        << "Expected size: " << operandVecSize << ", actual size: " << operandVec.size()
        << ", Conv Common Input: " << SHAPE_DIM2 << ", hasBias: " << convAttrParam.hasBias;

    tensorGraphNodes.fmapTensorPtr = operandVec[INPUT_FMAP_IDX];
    tensorGraphNodes.weightTensorPtr = operandVec[INPUT_WEIGHT_IDX];
    if (convAttrParam.hasBias) {
        tensorGraphNodes.biasTensorPtr = operandVec[INPUT_BIAS_IDX];
    }
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR,
           tensorGraphNodes.fmapTensorPtr != nullptr && tensorGraphNodes.weightTensorPtr != nullptr)
        << "Expected aTensorPtr and bTensorPtr to be non-nullptr.";

    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, cTensorPtr != nullptr) << "cTensorPtr is nullptr.";
    tensorGraphNodes.resTensorPtr = cTensorPtr;
}

void SetDynValidShapeInfo(const ConvGraphNodes& tensorGraphNodes, const ConvAttrParam& convAttrParam,
                          ConvTileInfo& convTileInfo)
{
    convTileInfo.dynValidBatch = tensorGraphNodes.fmapTensorPtr->GetDynValidShape()[0];
    convTileInfo.dynValidCout = convAttrParam.dynValidResShape[1];
    convTileInfo.dynValidHout = convAttrParam.isConv3D ? convAttrParam.dynValidResShape[3] :
                                                         convAttrParam.dynValidResShape[2];
    convTileInfo.dynValidWout = convAttrParam.isConv3D ? convAttrParam.dynValidResShape[4] :
                                                         convAttrParam.dynValidResShape[3];
    convTileInfo.dynValidDout = convAttrParam.isConv3D ? convAttrParam.dynValidResShape[NCDHW_D_IDX] :
                                                         SymbolicScalar(1);
}

void SetConvShapeInfo(const TileShape& tileShape, const ConvGraphNodes& tensorGraphNodes,
                      const ConvAttrParam& convAttrParam, ConvTileInfo& convTileInfo)
{
    // set org shape
    convTileInfo.orgBatch = convAttrParam.isConv3D ? convAttrParam.oriFmapShape[NCDHW_N_IDX] :
                                                     convAttrParam.oriFmapShape[NCHW_N_IDX];
    convTileInfo.orgHin = convAttrParam.isConv3D ? convAttrParam.oriFmapShape[NCDHW_H_IDX] :
                                                   convAttrParam.oriFmapShape[NCHW_H_IDX];
    convTileInfo.orgWin = convAttrParam.isConv3D ? convAttrParam.oriFmapShape[NCDHW_W_IDX] :
                                                   convAttrParam.oriFmapShape[NCHW_W_IDX];
    convTileInfo.orgCin = convAttrParam.isConv3D ? convAttrParam.oriFmapShape[NCDHW_C_IDX] :
                                                   convAttrParam.oriFmapShape[NCHW_C_IDX];
    convTileInfo.orgHout = convAttrParam.isConv3D ? convAttrParam.oriResShape[NCDHW_H_IDX] :
                                                    convAttrParam.oriResShape[NCHW_H_IDX];
    convTileInfo.orgWout = convAttrParam.isConv3D ? convAttrParam.oriResShape[NCDHW_W_IDX] :
                                                    convAttrParam.oriResShape[NCHW_W_IDX];
    convTileInfo.orgDin = convAttrParam.isConv3D ? convAttrParam.oriFmapShape[NCDHW_D_IDX] : 1;
    convTileInfo.orgDout = convAttrParam.isConv3D ? convAttrParam.oriResShape[NCDHW_D_IDX] : 1;
    convTileInfo.cin0 = ALIGN_SIZE_32 / BytesOf(tensorGraphNodes.fmapTensorPtr->Datatype());
    convTileInfo.orgCout = convAttrParam.isConv3D ? convAttrParam.oriWeightShape[NCDHW_N_IDX] :
                                                    convAttrParam.oriWeightShape[NCHW_N_IDX];
    convTileInfo.orgKh = convAttrParam.isConv3D ? convAttrParam.oriWeightShape[NCDHW_H_IDX] :
                                                  convAttrParam.oriWeightShape[NCHW_H_IDX];
    convTileInfo.orgKw = convAttrParam.isConv3D ? convAttrParam.oriWeightShape[NCDHW_W_IDX] :
                                                  convAttrParam.oriWeightShape[NCHW_W_IDX];
    convTileInfo.orgKd = convAttrParam.isConv3D ? convAttrParam.oriWeightShape[NCDHW_D_IDX] : 1;
    int64_t cinPerGroup = convTileInfo.orgCin / convAttrParam.groups;
    convTileInfo.orgHoutWout = convTileInfo.orgHout * convTileInfo.orgWout;
    convTileInfo.kPerGroup = ConvAlignB(cinPerGroup, convTileInfo.cin0) * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.coutPerGroup = convTileInfo.orgCout / convAttrParam.groups;
    // set tileshape info
    auto& convTile = tileShape.GetConvTile();
    convTileInfo.kAL1 = convTile.tileL1Info.tileCinFmap * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.kBL1 = convTile.tileL1Info.tileCinWeight * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.nBL1 = convTile.tileL1Info.tileN;
    convTileInfo.hAL1In = convTile.tileL1Info.tileHin;
    convTileInfo.wAL1In = convTile.tileL1Info.tileWin;
    convTileInfo.hAL1Out = convTile.tileL1Info.tileHout;
    convTileInfo.wAL1Out = convTile.tileL1Info.tileWout;
    convTileInfo.kL0 = convTile.tileL0Info.tileK;
    convTileInfo.hL0 = convTile.tileL0Info.tileH;
    convTileInfo.wL0 = convTile.tileL0Info.tileW;
    convTileInfo.nL0 = convTile.tileL0Info.tileN;
    // set dyn valid shape
    SetDynValidShapeInfo(tensorGraphNodes, convAttrParam, convTileInfo);
}

LogicalTensorPtr ConstructBiasTile(Function& function, const ConvGraphNodes& tensorGraphNodes, ConvIterInfo& iterInfo,
                                   ConvTileInfo& convTileInfo)
{
    std::vector<int64_t> dstBiasL1Shape = std::vector<int64_t>{1, iterInfo.nL0Size};
    std::vector<int64_t> dstBiasL1Offset = std::vector<int64_t>{
        0, iterInfo.groupOffset * convTileInfo.coutPerGroup + iterInfo.nL1Offset + iterInfo.nL0Offset};
    LogicalTensorPtr dstBiasl1TensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.biasTensorPtr->Datatype(), dstBiasL1Shape,
        SymbolicScalar::FromConcrete(dstBiasL1Shape), tensorGraphNodes.biasTensorPtr->Format(), "biasL1Tensor");
    dstBiasl1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBiasL1Shape));
    auto& sliceOpBiasL1 = function.AddOperation(config::GetSliceOpcode(), {tensorGraphNodes.biasTensorPtr},
                                                {dstBiasl1TensorPtr});
    auto viewAttributeBiasL1 = std::make_shared<ViewOpAttribute>(dstBiasL1Offset, MemoryType::MEM_L1,
                                                                 SymbolicScalar::FromConcrete(dstBiasL1Offset),
                                                                 dstBiasl1TensorPtr->GetDynValidShape());
    sliceOpBiasL1.SetOpAttribute(viewAttributeBiasL1);
    sliceOpBiasL1.SetAttribute(Matrix::A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(Matrix::CopyInMode::ND2ND));

    std::vector<int64_t> dstBiasBtShape = std::vector<int64_t>{1, iterInfo.nL0Size};
    std::vector<int64_t> dstBiasBtOffset = std::vector<int64_t>{0, iterInfo.nL0Offset};
    LogicalTensorPtr dstBiasBtTensorPtr = std::make_shared<LogicalTensor>(
        function, DataType::DT_FP32, dstBiasBtShape, SymbolicScalar::FromConcrete(dstBiasBtShape),
        tensorGraphNodes.biasTensorPtr->Format(), "biasBtTensor");
    dstBiasBtTensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBiasBtShape));
    auto& sliceOpBiasBt = function.AddOperation(config::GetSliceOpcode(), {dstBiasl1TensorPtr}, {dstBiasBtTensorPtr});
    auto viewAttributeBiasBt = std::make_shared<ViewOpAttribute>(dstBiasBtOffset, MemoryType::MEM_BT,
                                                                 SymbolicScalar::FromConcrete(dstBiasBtOffset),
                                                                 dstBiasBtTensorPtr->GetDynValidShape());
    sliceOpBiasBt.SetOpAttribute(viewAttributeBiasBt);

    return dstBiasBtTensorPtr;
}

void SetImg2ColAttr(Operation& load3dOpAl0, const ConvAttrParam& convAttrParam, ConvIterInfo& iterInfo,
                    const ConvTileInfo& convTileInfo)
{
    int64_t strideH = convAttrParam.strides[0];
    int64_t strideW = convAttrParam.strides[1];
    int64_t dilationH = convAttrParam.dilations[0];
    int64_t dilationW = convAttrParam.dilations[1];
    int64_t dilatedKernelH = (convTileInfo.orgKh - 1) * dilationH + 1;
    int64_t dilatedKernelW = (convTileInfo.orgKw - 1) * dilationW + 1;
    load3dOpAl0.SetAttribute(OpAttributeKey::strideH, strideH);
    load3dOpAl0.SetAttribute(OpAttributeKey::strideW, strideW);
    load3dOpAl0.SetAttribute(OpAttributeKey::dilationH, dilationH);
    load3dOpAl0.SetAttribute(OpAttributeKey::dilationW, dilationW);
    load3dOpAl0.SetAttribute(OpAttributeKey::filterH, convTileInfo.orgKh);
    load3dOpAl0.SetAttribute(OpAttributeKey::filterW, convTileInfo.orgKw);
    // cal H padding
    if (iterInfo.hL1InOffset >= 0) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingTop, 0);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingTop, 0 - iterInfo.hL1InOffset);
    }
    int64_t hinAL1Used = (iterInfo.houtL1Size - 1) * strideH + dilatedKernelH;
    int64_t hinBottomPadOffset = iterInfo.hL1InOffset + hinAL1Used;
    if (hinBottomPadOffset > convTileInfo.orgHin) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingBottom, hinBottomPadOffset - convTileInfo.orgHin);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingBottom, 0);
    }
    // cal W padding
    if (iterInfo.wL1InOffset >= 0) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingLeft, 0);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingLeft, 0 - iterInfo.wL1InOffset);
    }
    int64_t winAL1Used = (iterInfo.woutL1Size - 1) * strideW + dilatedKernelW;
    int64_t winRightPadOffset = iterInfo.wL1InOffset + winAL1Used;
    if (winRightPadOffset > convTileInfo.orgWin) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingRight, winRightPadOffset - convTileInfo.orgWin);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingRight, 0);
    }
    // cal postm postk
    int64_t mStartPt = iterInfo.hL0Offset * iterInfo.woutL1Size + iterInfo.wL0Offset;
    int64_t kStartPt = iterInfo.kL0Offset % convTileInfo.kAL1;
    load3dOpAl0.SetAttribute(OpAttributeKey::postM, mStartPt);
    load3dOpAl0.SetAttribute(OpAttributeKey::postK, kStartPt);
    // set pad value
    load3dOpAl0.SetAttribute(OpAttributeKey::padValue, 0);
    // set load3dv2 params
    load3dOpAl0.SetAttribute(OpAttributeKey::repeatStride, iterInfo.repeatStride);
    load3dOpAl0.SetAttribute(OpAttributeKey::repeatTime, iterInfo.repeatTime);
    load3dOpAl0.SetAttribute(OpAttributeKey::wStride, iterInfo.wStride);
    // set conv/conv3d flag
    load3dOpAl0.SetAttribute(OpAttributeKey::isConv, true);
    load3dOpAl0.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
}

void SetCopyInAL1Op(Operation& copyInOpAl1, const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo,
                    const ConvAttrParam& convAttrParam, const std::vector<int64_t>& dstAL1Shape,
                    const std::vector<int64_t>& srcGmValidShape, const int64_t& srcCinOffset, bool hasTransFormatL1)
{
    copyInOpAl1.SetAttribute(OpAttributeKey::isConv, true);
    copyInOpAl1.SetAttribute(OpAttributeKey::filterH, convTileInfo.orgKh);
    copyInOpAl1.SetAttribute(OpAttributeKey::filterW, convTileInfo.orgKw);
    copyInOpAl1.SetAttribute(LoadStoreConvOpAttributeKey::isFmap, true);
    copyInOpAl1.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
    int64_t src_n_offset = iterInfo.batchOffset;
    int64_t src_c_offset = iterInfo.groupOffset * (convTileInfo.orgCin / convAttrParam.groups) + srcCinOffset;
    int64_t src_d_offset = convAttrParam.isConv3D ?
                               (iterInfo.dinL1Offset +
                                (iterInfo.kL0Offset / convTileInfo.kPerGroup) * convAttrParam.dilations[NUM2]) :
                               0;
    int64_t src_h_offset = iterInfo.hL1InOffset > 0 ? iterInfo.hL1InOffset : 0;
    int64_t src_w_offset = iterInfo.wL1InOffset > 0 ? iterInfo.wL1InOffset : 0;

    std::vector<int64_t> srcGmOffset;
    std::vector<int64_t> srcGmShape;
    if (IsArch32Platform()) {
        if (!hasTransFormatL1) {
            copyInOpAl1.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode,
                                     static_cast<int64_t>(CopyInMode::COPY_MOD_NZ2NZ));
        }
        int64_t cin1PerGroup = CeilDiv(convTileInfo.orgCin / convAttrParam.groups, convTileInfo.cin0);
        int64_t cin1Offset = iterInfo.groupOffset * cin1PerGroup + srcCinOffset / convTileInfo.cin0;
        if (convAttrParam.isConv3D) {
            srcGmOffset = {src_n_offset, src_d_offset, cin1Offset, src_h_offset, src_w_offset, 0};
            srcGmShape = {1,
                          iterInfo.dkAL1Size,
                          CeilDiv(srcGmValidShape[1], convTileInfo.cin0),
                          iterInfo.hinL1Size,
                          iterInfo.winL1Size,
                          convTileInfo.cin0};
        } else {
            srcGmOffset = {src_n_offset, cin1Offset, src_h_offset, src_w_offset, 0};
            srcGmShape = {1, CeilDiv(srcGmValidShape[1], convTileInfo.cin0), iterInfo.hinL1Size, iterInfo.winL1Size,
                          convTileInfo.cin0};
        }
    } else {
        if (!hasTransFormatL1) {
            copyInOpAl1.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode,
                                     static_cast<int64_t>(CopyInMode::COPY_MOD_DN2NZ));
        }
        srcGmOffset = {src_n_offset, src_c_offset, src_h_offset, src_w_offset};
        if (convAttrParam.isConv3D) {
            srcGmOffset = {src_n_offset, src_c_offset, src_d_offset, src_h_offset, src_w_offset};
        }
        srcGmShape = srcGmValidShape;
    }

    const auto& viewToShape = hasTransFormatL1 ? srcGmValidShape : dstAL1Shape;
    auto viewAttr = std::make_shared<ViewOpAttribute>(srcGmOffset, MemoryType::MEM_L1,
                                                      SymbolicScalar::FromConcrete(srcGmOffset),
                                                      SymbolicScalar::FromConcrete(viewToShape));
    copyInOpAl1.SetOpAttribute(viewAttr);
    iterInfo.aL1UpadateFlag = false;
}

static void ConstructFmapL1Tile(Function& function, const ConvGraphNodes& tensorGraphNodes,
                                const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo,
                                LogicalTensorPtr& dstAL1TensorPtr, const ConvAttrParam& convAttrParam)
{
    iterInfo.kAL1Size = std::min((convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset), convTileInfo.kAL1);
    int64_t cin1AL1Size = (iterInfo.kAL1Size / convTileInfo.cin0) / (convTileInfo.orgKh * convTileInfo.orgKw);
    std::vector<int64_t> dstAL1Shape = std::vector<int64_t>{1, cin1AL1Size, iterInfo.hinL1Size, iterInfo.winL1Size,
                                                            convTileInfo.cin0};
    int64_t srcCinOffset = (iterInfo.kL0Offset % convTileInfo.kPerGroup) / (convTileInfo.orgKh * convTileInfo.orgKw);
    int64_t srcGmCin = std::min(convTileInfo.orgCin / convAttrParam.groups - srcCinOffset,
                                convTileInfo.kAL1 / (convTileInfo.orgKh * convTileInfo.orgKw));
    std::vector<int64_t> srcGmValidShape = std::vector<int64_t>{1, srcGmCin, iterInfo.hinL1Size, iterInfo.winL1Size};

    if (convAttrParam.isConv3D) {
        iterInfo.dkAL1Size = 1;
        if (iterInfo.kAL1Size > convTileInfo.kPerGroup) {
            srcCinOffset = 0;
            iterInfo.dkAL1Size = iterInfo.kAL1Size / convTileInfo.kPerGroup;
            cin1AL1Size = (iterInfo.kAL1Size / (iterInfo.dkAL1Size * convTileInfo.cin0)) /
                          (convTileInfo.orgKh * convTileInfo.orgKw);
        }
        dstAL1Shape = std::vector<int64_t>{
            1, iterInfo.dkAL1Size, cin1AL1Size, iterInfo.hinL1Size, iterInfo.winL1Size, convTileInfo.cin0};
        srcGmValidShape = std::vector<int64_t>{1, srcGmCin, iterInfo.dkAL1Size, iterInfo.hinL1Size, iterInfo.winL1Size};
    }

    dstAL1TensorPtr = std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL1Shape,
                                                      SymbolicScalar::FromConcrete(dstAL1Shape),
                                                      tensorGraphNodes.fmapTensorPtr->Format(), "aL1Tensor");
    dstAL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL1Shape));

    bool isArch32 = IsArch32Platform();
    if (isArch32) {
        auto& copyInOpAl1 = function.AddOperation(config::GetSliceOpcode(), {tensorGraphNodes.fmapTensorPtr},
                                                  {dstAL1TensorPtr});
        copyInOpAl1.SetAttribute(OpAttributeKey::isConv, true);
        SetCopyInAL1Op(copyInOpAl1, convTileInfo, iterInfo, convAttrParam, dstAL1Shape, srcGmValidShape, srcCinOffset,
                       false);
    } else {
        auto viewL1TensorPtr = std::make_shared<LogicalTensor>(
            function, tensorGraphNodes.fmapTensorPtr->Datatype(), srcGmValidShape,
            SymbolicScalar::FromConcrete(srcGmValidShape), tensorGraphNodes.fmapTensorPtr->Format(), "viewL1Tensor");
        viewL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(srcGmValidShape));

        auto& sliceOpAl1 = function.AddOperation(config::GetSliceOpcode(), {tensorGraphNodes.fmapTensorPtr},
                                                 {viewL1TensorPtr});
        sliceOpAl1.SetAttribute(OpAttributeKey::isConv, true);
        SetCopyInAL1Op(sliceOpAl1, convTileInfo, iterInfo, convAttrParam, dstAL1Shape, srcGmValidShape, srcCinOffset,
                       true);

        auto& transFormatL1Op = function.AddOperation(Opcode::OP_TRANS_FORMAT_L1, {viewL1TensorPtr}, {dstAL1TensorPtr});
        transFormatL1Op.SetAttribute(OpAttributeKey::isConv, true);
        transFormatL1Op.SetAttribute(LoadStoreConvOpAttributeKey::isFmap, true);
        transFormatL1Op.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
        transFormatL1Op.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode,
                                     static_cast<int64_t>(CopyInMode::COPY_MOD_DN2NZ));
        transFormatL1Op.SetAttribute(OpAttributeKey::filterH, convTileInfo.orgKh);
        transFormatL1Op.SetAttribute(OpAttributeKey::filterW, convTileInfo.orgKw);
    }
}

LogicalTensorPtr ConstructFmapTile(Function& function, const ConvGraphNodes& tensorGraphNodes,
                                   const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo,
                                   LogicalTensorPtr& dstAL1TensorPtr, const ConvAttrParam& convAttrParam)
{
    if (iterInfo.kL0Offset % convTileInfo.kAL1 == 0) {
        iterInfo.aL1UpadateFlag = true;
    }

    // L1层级 Fmap 展开
    if (iterInfo.aL1UpadateFlag) {
        ConstructFmapL1Tile(function, tensorGraphNodes, convTileInfo, iterInfo, dstAL1TensorPtr, convAttrParam);
    }

    // 二层展开
    // load3dv2()
    std::vector<int64_t> dstAL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE),
                                                            iterInfo.kL0Size};

    LogicalTensorPtr dstAL0TensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL0Shape,
        SymbolicScalar::FromConcrete({iterInfo.mL0Size, iterInfo.kL0Size}), tensorGraphNodes.fmapTensorPtr->Format(),
        "aL0Tensor");

    dstAL0TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL0Shape));

    auto& load3dOpAl0 = function.AddOperation(Opcode::OP_LOAD3D_CONV, {dstAL1TensorPtr}, {dstAL0TensorPtr});
    load3dOpAl0.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstAL0Shape));
    SetImg2ColAttr(load3dOpAl0, convAttrParam, iterInfo, convTileInfo);

    return dstAL0TensorPtr;
}

void SetCopyInBL1Op(Operation& copyInOpBl1, const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo,
                    const ConvAttrParam& convAttrParam, const std::vector<int64_t>& dstBL1Shape,
                    const std::vector<int64_t>& srcGmValidShape, const int64_t& srcCinOffset, bool hasTransFormatL1)
{
    copyInOpBl1.SetAttribute(OpAttributeKey::isConv, true);
    copyInOpBl1.SetAttribute(OpAttributeKey::filterH, convTileInfo.orgKh);
    copyInOpBl1.SetAttribute(OpAttributeKey::filterW, convTileInfo.orgKw);
    if (convAttrParam.isConv3D) {
        copyInOpBl1.SetAttribute("CONV_DK_L1_SIZE", iterInfo.dkBL1Size);
    }
    copyInOpBl1.SetAttribute(LoadStoreConvOpAttributeKey::isFmap, false);
    copyInOpBl1.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
    int64_t src_n_offset = iterInfo.groupOffset * convTileInfo.coutPerGroup + iterInfo.nL1Offset;
    int64_t src_c_offset = srcCinOffset;
    int64_t src_d_offset = 0;
    if (convAttrParam.isConv3D) {
        src_d_offset = (iterInfo.doL1Offset * convAttrParam.strides[NUM2] - convAttrParam.paddings[NUM4]) < 0 ?
                           (convTileInfo.orgKd - iterInfo.dkBL1SrcOffset +
                            (iterInfo.kL0Offset / convTileInfo.kPerGroup)) :
                           (iterInfo.kL0Offset / convTileInfo.kPerGroup);
    }

    std::vector<int64_t> srcGmOffset;
    std::vector<int64_t> srcGmShape;
    if (IsArch32Platform()) {
        if (!hasTransFormatL1) {
            copyInOpBl1.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode,
                                     static_cast<int64_t>(CopyInMode::COPY_MOD_NZ2NZ));
        }
        int64_t cout1Offset = iterInfo.nL1Offset / MKN_N_VALUE;
        int64_t cin1Offset = src_c_offset / convTileInfo.cin0;
        int64_t khxkw = convTileInfo.orgKh * convTileInfo.orgKw;
        int64_t cin1 = CeilDiv(convTileInfo.orgCin / convAttrParam.groups, convTileInfo.cin0);
        if (convAttrParam.isConv3D) {
            srcGmOffset = {((iterInfo.groupOffset * convTileInfo.orgKd + src_d_offset) * cin1 + cin1Offset) * khxkw,
                           cout1Offset, 0, 0};
            srcGmShape = {CeilDiv(srcGmValidShape[1], convTileInfo.cin0) * iterInfo.dkBL1Size * khxkw,
                          CeilDiv(iterInfo.nL1Size, MKN_N_VALUE), MKN_N_VALUE, convTileInfo.cin0};
        } else {
            srcGmOffset = {(iterInfo.groupOffset * cin1 + cin1Offset) * khxkw, cout1Offset, 0, 0};
            srcGmShape = {CeilDiv(srcGmValidShape[1], convTileInfo.cin0) * khxkw,
                          CeilDiv(iterInfo.nL1Size, MKN_N_VALUE), MKN_N_VALUE, convTileInfo.cin0};
        }
    } else {
        if (!hasTransFormatL1) {
            copyInOpBl1.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode,
                                     static_cast<int64_t>(CopyInMode::COPY_MOD_DN2NZ));
        }
        srcGmOffset = {src_n_offset, src_c_offset, 0, 0};
        if (convAttrParam.isConv3D) {
            srcGmOffset = {src_n_offset, src_c_offset, src_d_offset, 0, 0};
        }
        srcGmShape = srcGmValidShape;
    }
    const auto& viewToShape = hasTransFormatL1 ? srcGmValidShape : dstBL1Shape;
    auto viewAttr = std::make_shared<ViewOpAttribute>(srcGmOffset, MemoryType::MEM_L1,
                                                      SymbolicScalar::FromConcrete(srcGmOffset),
                                                      SymbolicScalar::FromConcrete(viewToShape));
    copyInOpBl1.SetOpAttribute(viewAttr);
    iterInfo.bL1UpadateFlag = false;
}

static void ConstructWeightL1Tile(Function& function, const ConvGraphNodes& tensorGraphNodes,
                                  const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo,
                                  LogicalTensorPtr& dstBL1TensorPtr, const ConvAttrParam& convAttrParam)
{
    iterInfo.kBL1Size = std::min(convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset, convTileInfo.kBL1);
    std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{
        iterInfo.kBL1Size / convTileInfo.cin0, CeilDiv(iterInfo.nL1Size, MKN_N_VALUE), MKN_N_VALUE, convTileInfo.cin0};
    int64_t srcCinOffset = (iterInfo.kL0Offset % convTileInfo.kPerGroup) / (convTileInfo.orgKh * convTileInfo.orgKw);
    int64_t srcGmCin = std::min(convTileInfo.orgCin / convAttrParam.groups - srcCinOffset,
                                convTileInfo.kBL1 / (convTileInfo.orgKh * convTileInfo.orgKw));
    std::vector<int64_t> srcGmValidShape = std::vector<int64_t>{iterInfo.nL1Size, srcGmCin, convTileInfo.orgKh,
                                                                convTileInfo.orgKw};
    if (convAttrParam.isConv3D) {
        iterInfo.dkBL1Size = 1;
        if (iterInfo.kBL1Size > convTileInfo.kPerGroup) {
            srcCinOffset = 0;
            iterInfo.dkBL1Size = iterInfo.kBL1Size / convTileInfo.kPerGroup;
        }
        dstBL1Shape = std::vector<int64_t>{iterInfo.kBL1Size / convTileInfo.cin0,
                                           CeilDiv(iterInfo.nL1Size, MKN_N_VALUE), MKN_N_VALUE, convTileInfo.cin0};
        srcGmValidShape = std::vector<int64_t>{iterInfo.nL1Size, srcGmCin, iterInfo.dkBL1Size, convTileInfo.orgKh,
                                               convTileInfo.orgKw};
    }
    dstBL1TensorPtr = std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(),
                                                      dstBL1Shape, SymbolicScalar::FromConcrete(dstBL1Shape),
                                                      tensorGraphNodes.weightTensorPtr->Format(), "bL1Tensor");
    dstBL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL1Shape));
    bool isArch32 = IsArch32Platform();
    if (isArch32) {
        auto& copyInOpBl1 = function.AddOperation(config::GetSliceOpcode(), {tensorGraphNodes.weightTensorPtr},
                                                  {dstBL1TensorPtr});
        copyInOpBl1.SetAttribute(OpAttributeKey::isConv, true);
        SetCopyInBL1Op(copyInOpBl1, convTileInfo, iterInfo, convAttrParam, dstBL1Shape, srcGmValidShape, srcCinOffset,
                       false);
    } else {
        auto viewL1TensorPtr = std::make_shared<LogicalTensor>(
            function, tensorGraphNodes.weightTensorPtr->Datatype(), srcGmValidShape,
            SymbolicScalar::FromConcrete(srcGmValidShape), tensorGraphNodes.weightTensorPtr->Format(), "viewL1Tensor");
        viewL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(srcGmValidShape));

        auto& sliceOpBl1 = function.AddOperation(config::GetSliceOpcode(), {tensorGraphNodes.weightTensorPtr},
                                                 {viewL1TensorPtr});
        sliceOpBl1.SetAttribute(OpAttributeKey::isConv, true);
        SetCopyInBL1Op(sliceOpBl1, convTileInfo, iterInfo, convAttrParam, dstBL1Shape, srcGmValidShape, srcCinOffset,
                       true);

        auto& transFormatL1Op = function.AddOperation(Opcode::OP_TRANS_FORMAT_L1, {viewL1TensorPtr}, {dstBL1TensorPtr});
        transFormatL1Op.SetAttribute(OpAttributeKey::isConv, true);
        transFormatL1Op.SetAttribute(LoadStoreConvOpAttributeKey::isFmap, false);
        transFormatL1Op.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
        transFormatL1Op.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode,
                                     static_cast<int64_t>(CopyInMode::COPY_MOD_DN2NZ));
        transFormatL1Op.SetAttribute(OpAttributeKey::filterH, convTileInfo.orgKh);
        transFormatL1Op.SetAttribute(OpAttributeKey::filterW, convTileInfo.orgKw);
        if (convAttrParam.isConv3D) {
            transFormatL1Op.SetAttribute("CONV_DK_L1_SIZE", iterInfo.dkBL1Size);
        }
    }
}

LogicalTensorPtr ConstructWeightTile(Function& function, const ConvGraphNodes& tensorGraphNodes,
                                     const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo,
                                     LogicalTensorPtr& dstBL1TensorPtr, const ConvAttrParam& convAttrParam)
{
    if (iterInfo.kL0Offset % convTileInfo.kBL1 == 0) {
        iterInfo.bL1UpadateFlag = true;
    }
    // L1层级 Weight 展开
    if (iterInfo.bL1UpadateFlag) {
        ConstructWeightL1Tile(function, tensorGraphNodes, convTileInfo, iterInfo, dstBL1TensorPtr, convAttrParam);
    }
    // load2d()
    std::vector<int64_t> dstBL0Shape = std::vector<int64_t>{iterInfo.kL0Size,
                                                            ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
    LogicalTensorPtr dstBL0TensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL0Shape,
        SymbolicScalar::FromConcrete({iterInfo.kL0Size, iterInfo.nL0Size}), tensorGraphNodes.weightTensorPtr->Format(),
        "bL0Tensor");
    dstBL0TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL0Shape));
    auto& load2dOpBl0 = function.AddOperation(Opcode::OP_LOAD2D_CONV, {dstBL1TensorPtr}, {dstBL0TensorPtr});
    load2dOpBl0.SetAttribute(OpAttributeKey::postK, iterInfo.kL0Offset % convTileInfo.kBL1);
    load2dOpBl0.SetAttribute(OpAttributeKey::postN, iterInfo.nL0Offset);
    load2dOpBl0.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstBL0Shape));
    load2dOpBl0.SetAttribute(OpAttributeKey::isConv, true);
    return dstBL0TensorPtr;
}

void SetAMulBAttr(const ConvGraphNodes& tensorGraphNodes, const ConvTileInfo& convTileInfo, Operation& op)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_OP_NULLPTR, tensorGraphNodes.fmapTensorPtr != nullptr &&
                                                                  tensorGraphNodes.weightTensorPtr != nullptr &&
                                                                  tensorGraphNodes.resTensorPtr != nullptr)
        << "Expected fmapTensorPtr, weightTensorPtr, and resTensorPtr to be non-nullptr.";

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.fmapTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.weightTensorPtr->Format()) << 1) |
                     (static_cast<int64_t>(tensorGraphNodes.resTensorPtr->Format()) << 2);
    op.SetAttribute(OpAttributeKey::isConv, true);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(A_MUL_B_ACT_M, convTileInfo.hL0 * convTileInfo.wL0);
    op.SetAttribute(A_MUL_B_ACT_K, convTileInfo.kL0);
    op.SetAttribute(A_MUL_B_ACT_N, convTileInfo.nL0);

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
    }
}

LogicalTensorPtr DoMmad(Function& function, const ConvAttrParam& convAttrParam, const ConvGraphNodes& tensorGraphNodes,
                        ConvGraphNodes& tileGraphNodes, const ConvTileInfo& convTileInfo, const ConvIterInfo& iterInfo)
{
    std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{
        convTileInfo.dynValidBatchL0 * convTileInfo.dynValidDoutL0 * convTileInfo.dynValidHoutL0 *
            convTileInfo.dynValidWoutL0,
        convTileInfo.dynValidCoutL0};
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tileGraphNodes.fmapTensorPtr != nullptr &&
                                                                tileGraphNodes.weightTensorPtr != nullptr &&
                                                                tileGraphNodes.resTensorPtr != nullptr)
        << "Inputs and res must be non-nullptr.";
    // MMAD node add
    std::vector<LogicalTensorPtr> mmadInputs;
    std::vector<LogicalTensorPtr> mmadOutputs;
    const std::string MmadOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (iterInfo.isFirstK) {
        mmadInputs = {tileGraphNodes.fmapTensorPtr, tileGraphNodes.weightTensorPtr};
        if (convAttrParam.hasBias) {
            ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tileGraphNodes.biasTensorPtr != nullptr)
                << "bias must be non-nullptr when hasBias Flag.";
            mmadInputs.push_back(tileGraphNodes.biasTensorPtr);
        }
    } else {
        mmadInputs = {tileGraphNodes.fmapTensorPtr, tileGraphNodes.weightTensorPtr, tileGraphNodes.cL0PartialSumPtr};
    }

    if (iterInfo.isLastK) {
        mmadOutputs = {tileGraphNodes.resTensorPtr};
    } else {
        std::vector<int64_t> cL0PartialSumShape = {ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE),
                                                   ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
        tileGraphNodes.cL0PartialSumPtr = std::make_shared<LogicalTensor>(
            function, DataType::DT_FP32, cL0PartialSumShape, dstCL0DynValidShape, TileOpFormat::TILEOP_NZ,
            "cL0PartialSumTensor");
        tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(dstCL0DynValidShape);
        mmadOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto& aMulBOp = function.AddOperation(MmadOpStr, mmadInputs, mmadOutputs);
    SetAMulBAttr(tensorGraphNodes, convTileInfo, aMulBOp);

    return mmadOutputs[0];
}

std::vector<int64_t> GetCopyOutDstOffset(const ConvAttrParam& convAttrParam, const ConvTileInfo& convTileInfo,
                                         const ConvIterInfo& iterInfo)
{
    int64_t dst_n_offset = iterInfo.batchOffset;
    int64_t dst_c_offset = iterInfo.groupOffset * convTileInfo.coutPerGroup + iterInfo.nL1Offset + iterInfo.nL0Offset;
    int64_t dst_d_offset = iterInfo.doL1Offset;
    int64_t dst_h_offset = iterInfo.hL1OutOffset + iterInfo.hL0Offset;
    int64_t dst_w_offset = iterInfo.wL1OutOffset + iterInfo.wL0Offset;

    std::vector<int64_t> dstResGmOffset;
    if (IsArch32Platform()) {
        int64_t cout1PerGroup = CeilDiv(convTileInfo.coutPerGroup, convTileInfo.cin0);
        int64_t cout1Offset = iterInfo.groupOffset * cout1PerGroup +
                              (iterInfo.nL1Offset + iterInfo.nL0Offset) / convTileInfo.cin0;
        if (convAttrParam.isConv3D) {
            dstResGmOffset = {dst_n_offset, dst_d_offset, cout1Offset, dst_h_offset, dst_w_offset, 0};
        } else {
            dstResGmOffset = {dst_n_offset, cout1Offset, dst_h_offset, dst_w_offset, 0};
        }
    } else {
        dstResGmOffset = {dst_n_offset, dst_c_offset, dst_h_offset, dst_w_offset};
        if (convAttrParam.isConv3D) {
            dstResGmOffset = {dst_n_offset, dst_c_offset, dst_d_offset, dst_h_offset, dst_w_offset};
        }
    }
    return dstResGmOffset;
}

void ConstrucCopyOutTile(Function& function, const ConvAttrParam& convAttrParam, const ConvGraphNodes& tensorGraphNodes,
                         const ConvTileInfo& convTileInfo, const ConvIterInfo& iterInfo,
                         const LogicalTensorPtr& resCl0TensorPtr)
{
    std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{
        convTileInfo.dynValidBatchL0 * convTileInfo.dynValidDoutL0 * convTileInfo.dynValidHoutL0 *
            convTileInfo.dynValidWoutL0,
        convTileInfo.dynValidCoutL0};
    bool isArch32 = IsArch32Platform();

    resCl0TensorPtr->UpdateDynValidShape(dstCL0DynValidShape);

    int64_t cutW = std::min(iterInfo.woutL1Size - iterInfo.wL0Offset, convTileInfo.wL0);
    int64_t copyOutMode = static_cast<int64_t>(isArch32 ? CopyOutMode::COPY_MOD_NZ2NZ : CopyOutMode::COPY_MOD_NZ2DN);
    std::vector<int64_t> dstResGmOffset = GetCopyOutDstOffset(convAttrParam, convTileInfo, iterInfo);
    int64_t curH = std::min(convTileInfo.hL0, iterInfo.houtL1Size - iterInfo.hL0Offset);
    int64_t curW = std::min(convTileInfo.wL0, iterInfo.woutL1Size - iterInfo.wL0Offset);
    std::vector<int64_t> l0CTileResShape;
    std::vector<SymbolicScalar> l0CTileResDynValidShape;
    if (isArch32) {
        dstCL0DynValidShape = {
            convTileInfo.dynValidBatchL0 * convTileInfo.dynValidDoutL0 * convTileInfo.dynValidHoutL0 *
                convTileInfo.dynValidWoutL0,
            (convTileInfo.dynValidCoutL0 + convTileInfo.cin0 - 1) / convTileInfo.cin0 * convTileInfo.cin0};
        l0CTileResShape = convAttrParam.isConv3D ?
                              std::vector<int64_t>{1,    1,    CeilDiv(iterInfo.nL0Size, convTileInfo.cin0),
                                                   curH, curW, convTileInfo.cin0} :
                              std::vector<int64_t>{1, CeilDiv(iterInfo.nL0Size, convTileInfo.cin0), curH, curW,
                                                   convTileInfo.cin0};
        l0CTileResDynValidShape = convAttrParam.isConv3D ?
                                      std::vector<SymbolicScalar>{
                                          convTileInfo.dynValidBatchL0,
                                          convTileInfo.dynValidDoutL0,
                                          (convTileInfo.dynValidCoutL0 + convTileInfo.cin0 - 1) / convTileInfo.cin0,
                                          convTileInfo.dynValidHoutL0,
                                          convTileInfo.dynValidWoutL0,
                                          SymbolicScalar(convTileInfo.cin0)} :
                                      std::vector<SymbolicScalar>{
                                          convTileInfo.dynValidBatchL0,
                                          (convTileInfo.dynValidCoutL0 + convTileInfo.cin0 - 1) / convTileInfo.cin0,
                                          convTileInfo.dynValidHoutL0, convTileInfo.dynValidWoutL0,
                                          SymbolicScalar(convTileInfo.cin0)};
    } else {
        l0CTileResShape = convAttrParam.isConv3D ? std::vector<int64_t>{1, iterInfo.nL0Size, 1, curH, curW} :
                                                   std::vector<int64_t>{1, iterInfo.nL0Size, curH, curW};
        l0CTileResDynValidShape = convAttrParam.isConv3D ?
                                      std::vector<SymbolicScalar>{
                                          convTileInfo.dynValidBatchL0, convTileInfo.dynValidCoutL0,
                                          convTileInfo.dynValidDoutL0, convTileInfo.dynValidHoutL0,
                                          convTileInfo.dynValidWoutL0} :
                                      std::vector<SymbolicScalar>{
                                          convTileInfo.dynValidBatchL0, convTileInfo.dynValidCoutL0,
                                          convTileInfo.dynValidHoutL0, convTileInfo.dynValidWoutL0};
    }
    auto resRawShape = tensorGraphNodes.resTensorPtr->tensor->GetRawShape();
    auto resDynRawShape = tensorGraphNodes.resTensorPtr->tensor->GetDynRawShape();

    auto transFormatL0CTensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.resTensorPtr->Datatype(), l0CTileResShape, l0CTileResDynValidShape,
        tensorGraphNodes.resTensorPtr->Format(), "transFormatL0CTensor");

    auto& transFormatL0COp = function.AddOperation(Opcode::OP_TRANS_FORMAT_L0C, {resCl0TensorPtr},
                                                   {transFormatL0CTensorPtr});
    transFormatL0COp.SetAttribute(OpAttributeKey::isConv, true);
    transFormatL0COp.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
    transFormatL0COp.SetAttribute(LoadStoreConvOpAttributeKey::cutW, cutW);
    transFormatL0COp.SetAttribute(LoadStoreConvOpAttributeKey::realCutW, convTileInfo.dynValidWoutL0);
    transFormatL0COp.SetAttribute(LoadStoreConvOpAttributeKey::copyOutMode, copyOutMode);
    transFormatL0COp.SetAttribute(OpAttributeKey::l0cValidMN, dstCL0DynValidShape);
    transFormatL0COp.SetAttribute(LoadStoreConvOpAttributeKey::reluType, convAttrParam.reluType);

    auto& contractOp = function.AddOperation(config::GetContractOpcode(), {transFormatL0CTensorPtr},
                                             {tensorGraphNodes.resTensorPtr});
    contractOp.SetAttribute(OpAttributeKey::isConv, true);
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(dstResGmOffset,
                                                              SymbolicScalar::FromConcrete(dstResGmOffset));
    contractOp.SetOpAttribute(assembleAttr);
}

void Cal3DDkL1Size(const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo, const ConvAttrParam& convAttrParam)
{
    // cal dk in L1, not support dk in L1 = 0 now, kerneld <= padd
    iterInfo.dkL1Size = 1;
    if (convAttrParam.isConv3D) {
        iterInfo.dkL1Size = convTileInfo.orgKd;
        iterInfo.dinL1Offset = iterInfo.doL1Offset * convAttrParam.strides[NUM2] - convAttrParam.paddings[NUM4];
        int64_t srcDkOffset = iterInfo.dinL1Offset;
        if (iterInfo.dinL1Offset < 0) {
            int64_t tmpKd = CeilDiv(-iterInfo.dinL1Offset, convAttrParam.dilations[NUM2]);
            iterInfo.dkL1Size -= tmpKd;
            iterInfo.dkBL1SrcOffset = iterInfo.dkL1Size;
            srcDkOffset = iterInfo.dinL1Offset + tmpKd * convAttrParam.dilations[NUM2];
        }
        int64_t kdL1EndOffset = iterInfo.dinL1Offset + (convTileInfo.orgKd - 1) * convAttrParam.dilations[NUM2] + 1;
        if (kdL1EndOffset > convTileInfo.orgDin) {
            int64_t tmpKd = CeilDiv(kdL1EndOffset - convTileInfo.orgDin, convAttrParam.dilations[NUM2]);
            iterInfo.dkL1Size -= tmpKd;
        }
        iterInfo.dinL1Offset = srcDkOffset;
    }
}

void UpdateL1IterInfo(const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo, const ConvAttrParam& convAttrParam)
{
    // update iterInfo L1
    // cal winL1Size
    iterInfo.houtL1Size = std::min(convTileInfo.orgHout - iterInfo.hL1OutOffset, convTileInfo.hAL1Out);
    iterInfo.hL1InOffset = iterInfo.hL1OutOffset * convAttrParam.strides[0] - convAttrParam.paddings[0];
    int64_t needHL1Size = (iterInfo.houtL1Size - 1) * convAttrParam.strides[0] +
                          (convTileInfo.orgKh - 1) * convAttrParam.dilations[0] + 1;
    if (iterInfo.hL1InOffset < 0) {
        // start pos locate in pad
        iterInfo.hinL1Size = needHL1Size + iterInfo.hL1InOffset;
        if (iterInfo.hL1InOffset + needHL1Size <= 0) {
            // all locate in pad
            iterInfo.hinL1Size = 0;
        }
        if (iterInfo.hinL1Size > convTileInfo.orgHin) {
            // w all load l1
            iterInfo.hinL1Size = convTileInfo.orgHin;
        }
    } else if (convTileInfo.orgHin - iterInfo.hL1InOffset <= 0) {
        // start pos locate in bottom pad
        iterInfo.hinL1Size = 0;
    } else {
        iterInfo.hinL1Size = std::min(convTileInfo.orgHin - iterInfo.hL1InOffset, needHL1Size);
    }
    // cal winL1Size
    iterInfo.woutL1Size = std::min(convTileInfo.orgWout - iterInfo.wL1OutOffset, convTileInfo.wAL1Out);
    iterInfo.wL1InOffset = iterInfo.wL1OutOffset * convAttrParam.strides[1] - convAttrParam.paddings[NUM2];
    int64_t needWL1Size = (iterInfo.woutL1Size - 1) * convAttrParam.strides[1] +
                          (convTileInfo.orgKw - 1) * convAttrParam.dilations[1] + 1;
    if (iterInfo.wL1InOffset < 0) {
        // start pos locate in pad
        iterInfo.winL1Size = needWL1Size + iterInfo.wL1InOffset;
        if (iterInfo.wL1InOffset + needWL1Size <= 0) {
            // all locate in pad
            iterInfo.winL1Size = 0;
        }
        if (iterInfo.winL1Size > convTileInfo.orgWin) {
            // w all load l1
            iterInfo.winL1Size = convTileInfo.orgWin;
        }
    } else if (convTileInfo.orgWin - iterInfo.wL1InOffset <= 0) {
        // start pos locate in right pad
        iterInfo.winL1Size = 0;
    } else {
        iterInfo.winL1Size = std::min(convTileInfo.orgWin - iterInfo.wL1InOffset, needWL1Size);
    }
    // cal nL1Size
    iterInfo.nL1Size = std::min(convTileInfo.coutPerGroup - iterInfo.nL1Offset, convTileInfo.nBL1);
    Cal3DDkL1Size(convTileInfo, iterInfo, convAttrParam);
}

void UpdateL0IterInfo(const ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo)
{
    // update iterInfo
    iterInfo.kL0Size = std::min(convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset, convTileInfo.kL0);
    iterInfo.isFirstK = iterInfo.kL0Offset == 0 ? true : false;
    iterInfo.isLastK = iterInfo.kL0Offset + convTileInfo.kL0 >= convTileInfo.kPerGroup * iterInfo.dkL1Size ? true :
                                                                                                             false;
}

void CalL0DynValidShape(ConvTileInfo& convTileInfo, ConvIterInfo& iterInfo, const ConvAttrParam& convAttrParam)
{
    // cal l0 dyn valid shape
    // when batch validshape = 0，set m to 0, make sure no tstore
    convTileInfo.dynValidBatchL0 = std::min(1, std::max(convTileInfo.dynValidBatch - iterInfo.batchOffset, 0));
    convTileInfo.dynValidDoutL0 = std::min(1, std::max(convTileInfo.dynValidDout - iterInfo.doL1Offset, 0));
    convTileInfo.dynValidHoutL0 = std::min(
        convTileInfo.hL0,
        std::max(std::min(std::max(convTileInfo.dynValidHout - iterInfo.hL1OutOffset, 0), convTileInfo.hAL1Out) -
                     iterInfo.hL0Offset,
                 0));
    convTileInfo.dynValidWoutL0 = std::min(
        convTileInfo.wL0,
        std::max(std::min(std::max(convTileInfo.dynValidWout - iterInfo.wL1OutOffset, 0), convTileInfo.wAL1Out) -
                     iterInfo.wL0Offset,
                 0));
    convTileInfo.dynValidCoutL0 = std::min(
        convTileInfo.nL0,
        std::max(std::min(std::max(convTileInfo.dynValidCout / convAttrParam.groups - iterInfo.nL1Offset, 0),
                          convTileInfo.nBL1) -
                     iterInfo.nL0Offset,
                 0));
}

void IterL0ExpandFunc(Function& function, ConvIterInfo& iterInfo, ConvTileInfo& convTileInfo,
                      const ConvAttrParam& convAttrParam, const ConvGraphNodes& tensorGraphNodes,
                      ConvGraphNodes& tileGraphNodes)
{
    LogicalTensorPtr fmapL1TensorPtr = nullptr;
    LogicalTensorPtr weightL1TensorPtr = nullptr;
    LogicalTensorPtr resCl0TensorPtr = nullptr;
    for (iterInfo.nL0Offset = 0; iterInfo.nL0Offset < iterInfo.nL1Size; iterInfo.nL0Offset += convTileInfo.nL0) {
        iterInfo.nL0Size = std::min(iterInfo.nL1Size - iterInfo.nL0Offset, convTileInfo.nL0);
        for (iterInfo.hL0Offset = 0; iterInfo.hL0Offset < iterInfo.houtL1Size; iterInfo.hL0Offset += convTileInfo.hL0) {
            for (iterInfo.wL0Offset = 0; iterInfo.wL0Offset < iterInfo.woutL1Size;
                 iterInfo.wL0Offset += convTileInfo.wL0) {
                int64_t curH = std::min(convTileInfo.hL0, iterInfo.houtL1Size - iterInfo.hL0Offset);
                int64_t curW = std::min(convTileInfo.wL0, iterInfo.woutL1Size - iterInfo.wL0Offset);
                iterInfo.mL0Size = curH * curW;
                if (curH > 1 && convTileInfo.wL0 != convTileInfo.wAL1Out) {
                    iterInfo.repeatStride = iterInfo.woutL1Size;
                    iterInfo.repeatTime = curH;
                    iterInfo.wStride = curW;
                } else {
                    iterInfo.repeatTime = 1;
                    iterInfo.wStride = ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE);
                }
                // bias 载入
                if (convAttrParam.hasBias) {
                    // get bias in bt tile for mmad
                    tileGraphNodes.biasTensorPtr = ConstructBiasTile(function, tensorGraphNodes, iterInfo,
                                                                     convTileInfo);
                }
                // set res tile
                std::vector<int64_t> dstCL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE),
                                                                        ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
                CalL0DynValidShape(convTileInfo, iterInfo, convAttrParam);
                std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{
                    convTileInfo.dynValidBatchL0 * convTileInfo.dynValidDoutL0 * convTileInfo.dynValidHoutL0 *
                        convTileInfo.dynValidWoutL0,
                    convTileInfo.dynValidCoutL0};
                tileGraphNodes.resTensorPtr = std::make_shared<LogicalTensor>(
                    function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstCL0Shape, dstCL0DynValidShape,
                    tensorGraphNodes.fmapTensorPtr->Format(), "cL0Tensor");
                for (iterInfo.kL0Offset = 0; iterInfo.kL0Offset < convTileInfo.kPerGroup * iterInfo.dkL1Size;
                     iterInfo.kL0Offset += convTileInfo.kL0) {
                    UpdateL0IterInfo(convTileInfo, iterInfo);
                    // fmap and weight link
                    tileGraphNodes.fmapTensorPtr = ConstructFmapTile(function, tensorGraphNodes, convTileInfo, iterInfo,
                                                                     fmapL1TensorPtr, convAttrParam);
                    tileGraphNodes.weightTensorPtr = ConstructWeightTile(function, tensorGraphNodes, convTileInfo,
                                                                         iterInfo, weightL1TensorPtr, convAttrParam);
                    // add mmad node
                    resCl0TensorPtr = DoMmad(function, convAttrParam, tensorGraphNodes, tileGraphNodes, convTileInfo,
                                             iterInfo);
                }
                ConstrucCopyOutTile(function, convAttrParam, tensorGraphNodes, convTileInfo, iterInfo, resCl0TensorPtr);
            }
        }
    }
}

void IterOneBatchFunc(Function& function, ConvIterInfo& iterInfo, ConvTileInfo& convTileInfo,
                      const ConvAttrParam& convAttrParam, const ConvGraphNodes& tensorGraphNodes,
                      ConvGraphNodes& tileGraphNodes)
{
    for (iterInfo.doL1Offset = 0; iterInfo.doL1Offset < convTileInfo.orgDout; iterInfo.doL1Offset += 1) {
        for (iterInfo.nL1Offset = 0; iterInfo.nL1Offset < convTileInfo.coutPerGroup;
             iterInfo.nL1Offset += convTileInfo.nBL1) {
            iterInfo.bL1UpadateFlag = true;
            for (iterInfo.hL1OutOffset = 0; iterInfo.hL1OutOffset < convTileInfo.orgHout;
                 iterInfo.hL1OutOffset += convTileInfo.hAL1Out) {
                for (iterInfo.wL1OutOffset = 0; iterInfo.wL1OutOffset < convTileInfo.orgWout;
                     iterInfo.wL1OutOffset += convTileInfo.wAL1Out) {
                    iterInfo.aL1UpadateFlag = true;
                    UpdateL1IterInfo(convTileInfo, iterInfo, convAttrParam);
                    // iterate L0 buffer expand
                    IterL0ExpandFunc(function, iterInfo, convTileInfo, convAttrParam, tensorGraphNodes, tileGraphNodes);
                }
            }
        }
    }
}

void ConstructTileGraph(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& operandVec,
                        const LogicalTensorPtr& cTensorPtr, const Operation& op)
{
    // op attr set
    ConvAttrParam convAttrParam;
    SetConvAttrParam(op, convAttrParam);
    // set tensor graph node info
    ConvGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, convAttrParam, tensorGraphNodes);
    // save tile info
    ConvTileInfo convTileInfo;
    SetConvShapeInfo(tileShape, tensorGraphNodes, convAttrParam, convTileInfo);
    // save iter info
    ConvIterInfo iterInfo;
    // set tile graph node info
    ConvGraphNodes tileGraphNodes;

    for (iterInfo.groupOffset = 0; iterInfo.groupOffset < convAttrParam.groups; iterInfo.groupOffset += 1) {
        for (iterInfo.batchOffset = 0; iterInfo.batchOffset < convTileInfo.orgBatch; iterInfo.batchOffset += 1) {
            IterOneBatchFunc(function, iterInfo, convTileInfo, convAttrParam, tensorGraphNodes, tileGraphNodes);
        }
    }
}

std::vector<int64_t> GetResTensorShape(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
                                       const ConvAttrParam& convAttrParam)
{
    int64_t batchOut = inputTensor.GetShape()[NCHW_N_IDX];
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, convAttrParam);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, convAttrParam);
    std::vector<int64_t> resTensorShape;
    if (IsArch32Platform()) {
        int64_t cOut0 = ALIGN_SIZE_32 / BytesOf(outType);
        int64_t cOut1 = convAttrParam.groups * CeilDiv(cOut / convAttrParam.groups, cOut0);
        resTensorShape = {batchOut, cOut1, hOut, wOut, cOut0};
        if (convAttrParam.isConv3D) {
            int64_t dOut = ConvComputeDo(inputTensor, weightTensor, convAttrParam);
            resTensorShape = {batchOut, dOut, cOut1, hOut, wOut, cOut0};
        }
    } else {
        resTensorShape = {batchOut, cOut, hOut, wOut};
        if (convAttrParam.isConv1D) {
            resTensorShape = {batchOut, cOut, wOut};
        }
        if (convAttrParam.isConv3D) {
            int64_t dOut = ConvComputeDo(inputTensor, weightTensor, convAttrParam);
            resTensorShape = {batchOut, cOut, dOut, hOut, wOut};
        }
    }
    return resTensorShape;
}

std::vector<SymbolicScalar> GetResTensorDynValidShape(DataType outType, const Tensor& inputTensor,
                                                      const Tensor& weightTensor, const ConvAttrParam& convAttrParam)
{
    SymbolicScalar batchOut = inputTensor.GetValidShape()[NCHW_N_IDX];
    SymbolicScalar cOut = weightTensor.GetValidShape()[NCHW_N_IDX];
    SymbolicScalar hOut = ConvComputeValidHo(inputTensor, weightTensor, convAttrParam);
    SymbolicScalar wOut = ConvComputeValidWo(inputTensor, weightTensor, convAttrParam);
    std::vector<SymbolicScalar> resTensorDynValidShape;
    if (IsArch32Platform()) {
        int64_t cOut0 = static_cast<int64_t>(ALIGN_SIZE_32 / BytesOf(outType));
        SymbolicScalar cOut1 = convAttrParam.groups * ((cOut / convAttrParam.groups + cOut0 - 1) / cOut0);
        resTensorDynValidShape = {batchOut, cOut1, hOut, wOut, cOut0};
        if (convAttrParam.isConv3D) {
            SymbolicScalar dOut = ConvComputeValidDo(inputTensor, weightTensor, convAttrParam);
            resTensorDynValidShape = {batchOut, dOut, cOut1, hOut, wOut, cOut0};
        }
    } else {
        resTensorDynValidShape = {batchOut, cOut, hOut, wOut};
        if (convAttrParam.isConv1D) {
            resTensorDynValidShape = {batchOut, cOut, wOut};
        }
        if (convAttrParam.isConv3D) {
            SymbolicScalar dOut = ConvComputeValidDo(inputTensor, weightTensor, convAttrParam);
            resTensorDynValidShape = {batchOut, cOut, dOut, hOut, wOut};
        }
    }
    return resTensorDynValidShape;
}

Tensor Conv(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
            const std::vector<int64_t>& strides, const std::vector<SymbolicScalar>& paddings,
            const std::vector<int64_t>& dilations, const ConvExtendParam& extendParam, const int64_t groups)
{
    std::vector<int64_t> finalPaddings = SymbolicScalar::Concrete(paddings, 0);
    std::vector<int64_t> finalDilations = dilations;
    std::vector<int64_t> finalStrides = strides;
    if (dilations.size() == CONV3D_INPUT_DIM - NUM2 && strides.size() == CONV3D_INPUT_DIM - NUM2 &&
        paddings.size() == NUM2 * (CONV3D_INPUT_DIM - NUM2)) {
        finalDilations = rotateVector(dilations, 1);
        finalStrides = rotateVector(strides, 1);
        finalPaddings = rotateVector(SymbolicScalar::Concrete(paddings, 0), NUM2);
    }
    const Tensor& biasTensor = extendParam.biasTensor;
    // init and set attr
    ConvAttrParam convAttrParam(finalPaddings, finalStrides, finalDilations, groups);
    convAttrParam.reluType = static_cast<int64_t>(extendParam.reluType);
    CheckConvOperands(outType, inputTensor, weightTensor, biasTensor, convAttrParam);
    std::vector<int64_t> resTensorShape = GetResTensorShape(outType, inputTensor, weightTensor, convAttrParam);
    std::vector<SymbolicScalar> resTensorDynValidShape = GetResTensorDynValidShape(outType, inputTensor, weightTensor,
                                                                                   convAttrParam);
    if (convAttrParam.isConv1D) {
        convAttrParam.paddings.insert(convAttrParam.paddings.begin(), NUM2, 0);
        convAttrParam.strides.insert(convAttrParam.strides.begin(), 1);
        convAttrParam.dilations.insert(convAttrParam.dilations.begin(), 1);
    }
    TileOpFormat outFormat = TileOpFormat::TILEOP_ND;
    if (IsArch32Platform()) {
        outFormat = convAttrParam.isConv3D ? TileOpFormat::TILEOP_NDC1HWC0 : TileOpFormat::TILEOP_NC1HWC0;
    }
    Tensor resTensor(outType, resTensorShape, "TensorC", outFormat);
    resTensor.GetStorage()->UpdateDynValidShape(resTensorDynValidShape);
    return ConstructTensorGraph(inputTensor, weightTensor, biasTensor, resTensor, convAttrParam);
}

} // namespace Conv
} // namespace tile_fwk
} // namespace npu
