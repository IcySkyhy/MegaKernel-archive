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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "codegen/utils/codegen_utils.h"
#include "tilefwk/error_code.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/opcode.h"
#include "securec.h"

namespace npu::tile_fwk {
namespace {
const std::unordered_set<Opcode> OP_SHAPE_FROM_ATTR{
    // copy in/out
    Opcode::OP_COPY_IN,
    Opcode::OP_COPY_OUT,
    // transpose move in/out
    Opcode::OP_TRANSPOSE_MOVEOUT,
    Opcode::OP_TRANSPOSE_MOVEIN,
    // index outcast
    Opcode::OP_INDEX_OUTCAST,
    // conv Load
    Opcode::OP_L1_COPY_IN_CONV,
    Opcode::OP_L0C_COPY_OUT_CONV,
    Opcode::OP_RESHAPE_COPY_IN,
    Opcode::OP_RESHAPE_COPY_OUT,
    Opcode::OP_L1_RESHAPE_COPY_IN,
    Opcode::OP_L0C_RESHAPE_COPY_OUT,
};
bool IsOpShapeFromAttr(Opcode opcode) { return OP_SHAPE_FROM_ATTR.find(opcode) != OP_SHAPE_FROM_ATTR.end(); }

const std::unordered_set<Opcode> SHMEM_COPY_OPS{Opcode::OP_SHMEM_GET, Opcode::OP_SHMEM_LOAD, Opcode::OP_SHMEM_PUT,
                                                Opcode::OP_SHMEM_STORE};
bool IsShmemCopyOp(Opcode opcode) { return SHMEM_COPY_OPS.find(opcode) != SHMEM_COPY_OPS.end(); }
} // namespace

template <typename T>
void CombineLastTwoAxis(std::vector<T>& shape, size_t shapeSize)
{
    if (shape.size() < NUM2) {
        return;
    }
    shape[shapeSize - 1] = shape[shapeSize - 1] * shape[shapeSize - NUM2];
    shape[shapeSize - NUM2] = 1;
}

bool CodeGenOp::NeedCombineAxis(const Operation& oper, bool isInput, size_t ioIdx) const
{
    std::vector<bool> needCombineIOIdx;
    if (!((isInput && oper.GetAttr(OpAttributeKey::inputCombineAxis, needCombineIOIdx)) ||
          (!isInput && oper.GetAttr(OpAttributeKey::outputCombineAxis, needCombineIOIdx)))) {
        return false;
    }
    CODEGEN_LOGI("needCombineIOIdx is %s", IntVecToStr(needCombineIOIdx).c_str());
    return ioIdx < needCombineIOIdx.size() && needCombineIOIdx[ioIdx];
}

void CodeGenOp::CombineAxisShape(const Operation& oper, int operandIdx)
{
    size_t dim = rawShape[operandIdx].size();
    if (dim <= 1) {
        CODEGEN_LOGW("raw shape dim is %zu, return", dim);
        return;
    }

    CombineLastTwoAxis(shape[operandIdx], dim);
    CombineLastTwoAxis(rawShape[operandIdx], dim);
    CombineLastTwoAxis(dynamicValidShape[operandIdx], dim);
    CombineLastTwoAxis(dynamicRawShape[operandIdx], dim);
    CODEGEN_LOGI("op code %s, operandIdx: %d, after CombineAxis shape is %s, raw shape is %s, "
                 "dynamicValidShape is %s, dynamicRawShape is %s",
                 oper.GetOpcodeStr().c_str(), operandIdx, IntVecToStr(shape[operandIdx]).c_str(),
                 IntVecToStr(rawShape[operandIdx]).c_str(), IntVecToStr(dynamicValidShape[operandIdx]).c_str(),
                 IntVecToStr(dynamicRawShape[operandIdx]).c_str());
}

void CodeGenOp::CombineAxisOffset(const Operation& oper, int operandIdx)
{
    auto originalRawShape = rawShape[operandIdx];
    size_t dim = originalRawShape.size();
    if (dim <= 1) {
        return;
    }

    CombineLastTwoAxisOffset(offset[operandIdx], originalRawShape, dim);
    CombineLastTwoAxisOffset(dynamicOffset[operandIdx], originalRawShape, dim);
    CODEGEN_LOGI("op code %s, operandIdx: %d, after CombineAxis offset is %s, dynamicOffset is %s",
                 oper.GetOpcodeStr().c_str(), operandIdx, IntVecToStr(offset[operandIdx]).c_str(),
                 IntVecToStr(dynamicOffset[operandIdx]).c_str());
}

void CodeGenOp::UpdateDynValidShapeFromAttr(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx)
{
    std::shared_ptr<CopyOpAttribute> attr = std::dynamic_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    if (attr == nullptr) {
        return;
    }

    if ((opCode == Opcode::OP_RESHAPE_COPY_IN || opCode == Opcode::OP_L1_RESHAPE_COPY_IN) &&
        logicalTensor.GetMemoryTypeOriginal() == MEM_DEVICE_DDR) {
        SetDynValidShapeFromAttr(attr->GetFromDynValidShape(), operandIdx);
        return;
    }

    if ((opCode == Opcode::OP_L0C_TO_L1 && logicalTensor.GetMemoryTypeOriginal() == MEM_L1) ||
        ((opCode == Opcode::OP_RESHAPE_COPY_OUT || opCode == Opcode::OP_L0C_RESHAPE_COPY_OUT) &&
         logicalTensor.GetMemoryTypeOriginal() == MEM_DEVICE_DDR)) {
        SetDynValidShapeFromAttr(attr->GetToDynValidShape(), operandIdx);
        return;
    }

    UpdateValidShapeForShmemCopyOps(oper, operandIdx, attr);
}

void CodeGenOp::UpdateShape(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx)
{
    CODEGEN_LOGI("UpdateShape: op code %s, operandIdx: %d, shape is %s, raw shape is %s, dynamicValidShape is %s, "
                 "dynamicRawShape is %s",
                 oper.GetOpcodeStr().c_str(), operandIdx, IntVecToStr(logicalTensor.shape).c_str(),
                 IntVecToStr(logicalTensor.tensor->rawshape).c_str(),
                 IntVecToStr(logicalTensor.GetDynValidShape()).c_str(),
                 IntVecToStr(logicalTensor.tensor->dynRawShape).c_str());

    // raw shape
    rawShape[operandIdx] = logicalTensor.tensor->rawshape;
    // tensor shape equals to valid shape in static scene by PASS, just use 'LogicalTensor::shape' member variable
    shape[operandIdx] = logicalTensor.shape;

    if (isDynamicFunction) {
        dynamicValidShape[operandIdx] = isMainBlock ? SymbolicScalar::FromConcrete(logicalTensor.shape) :
                                                      logicalTensor.GetDynValidShape();
    }

    ASSERT(OperErr::TENSOR_DIM_EXCEEDED, logicalTensor.shape.size() <= UPDATE_SHAPE_MAX_DIM)
        << "only support max dim: " << UPDATE_SHAPE_MAX_DIM << ", Tensor is " << logicalTensor.Dump();

    Opcode opcode = oper.GetOpcode();
    if (logicalTensor.GetMemoryTypeOriginal() == MEM_DEVICE_DDR && IsOpShapeFromAttr(opcode)) {
        std::shared_ptr<CopyOpAttribute> attr = std::dynamic_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
        ASSERT(OperErr::ATTRIBUTE_INVALID, attr != nullptr) << ": missing OpAttr in copy op: \n" << oper.Dump();
        // 1. for spilling GM scene 2. for conv
        shapeFromAttr[operandIdx] = attr->GetSpecifiedShape(1);
        dynamicRawShape[operandIdx] = OpImmediate::ToSpecified(attr->GetRawShape());
        CODEGEN_LOGI("(from op CopyOpAttribute) shapeFromAttr[%d] = %s, dynamicRawShape[%d] = %s", operandIdx,
                     IntVecToStr(shapeFromAttr[operandIdx]).c_str(), operandIdx,
                     IntVecToStr(dynamicRawShape[operandIdx]).c_str());
    }

    UpdateDynValidShapeFromAttr(oper, logicalTensor, operandIdx);
}

void CodeGenOp::UpdateValidShapeForShmemCopyOps(const Operation& oper, int operandIdx,
                                                std::shared_ptr<CopyOpAttribute> attr)
{
    Opcode opcode = oper.GetOpcode();
    if ((!IsShmemCopyOp(opcode)) || (operandIdx != 0)) {
        return;
    }
    const auto& validShape = (attr->GetFromDynValidShape().size() != 0) ? attr->GetFromDynValidShape() :
                                                                          attr->GetToDynValidShape();
    SetDynValidShapeFromAttr(validShape, operandIdx);
}

void CodeGenOp::UpdateOffsetValueFromAttr(const std::vector<OpImmediate>& offsets, int operandIdx)
{
    offsetFromAttr[operandIdx] = OpImmediate::ToSpecified(offsets);
    CODEGEN_LOGI("Set offsetFromAttr[%d] = %s", operandIdx, IntVecToStr(offsetFromAttr[operandIdx]).c_str());
}

void CodeGenOp::SetDynValidShapeFromAttr(const std::vector<OpImmediate>& toValidShape, int operandIdx)
{
    dynValidShapeFromOpAttr[operandIdx] = OpImmediate::ToSpecified(toValidShape);
    CODEGEN_LOGI("Set dynValidShapeFromOpAttr[%d] = %s", operandIdx,
                 IntVecToStr(dynValidShapeFromOpAttr[operandIdx]).c_str());
}

void CodeGenOp::UpdateOffsetForInput(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx)
{
    static const std::set<Opcode> cubeMDLOpCode = {
        Opcode::OP_L1_TO_L0A,   Opcode::OP_L1_TO_L0B,       Opcode::OP_L1_TO_L0_AT,
        Opcode::OP_L1_TO_L0_BT, Opcode::OP_L1_TO_BT,        Opcode::OP_L1_TO_FIX_QUANT_PRE,
        Opcode::OP_L0C_TO_L1,   Opcode::OP_L1_TO_L0A_SCALE, Opcode::OP_L1_TO_L0B_SCALE,
        Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1,      Opcode::OP_L0C_COPY_UB_DUAL_DST};
    static const std::set<Opcode> distOpcode = {Opcode::OP_SHMEM_PUT, Opcode::OP_SHMEM_STORE};
    bool cubeMDLCondition = cubeMDLOpCode.count(opCode);
    bool distCondition = distOpcode.count(opCode);
    bool useAttrShapeOffsetForInputGM = OpcodeManager::Inst().IsCopyIn(opCode) &&
                                        logicalTensor.GetMemoryTypeOriginal() == MEM_DEVICE_DDR;
    std::shared_ptr<CopyOpAttribute> attr = std::dynamic_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    if (attr != nullptr && (distCondition || cubeMDLCondition || useAttrShapeOffsetForInputGM)) {
        // only used for 1. L1 Copy; 2. spilling to gm scene(e.g., ooo spilling); 3. matmul Multi-Data Load scene.
        CODEGEN_LOGI("start update offset for GM input");
        UpdateOffsetValueFromAttr(attr->GetCopyInAttr().first, operandIdx);
        return;
    }

    offset[operandIdx] = logicalTensor.offset; // Local Tensor offset just use offset from LogicalTensor
    CODEGEN_LOGI("UpdateOffsetForInput logicalTensor offset is %s", IntVecToStr(offset[operandIdx]).c_str());
}

void CodeGenOp::UpdateOffsetForOutput(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx)
{
    static const std::set<Opcode> cubeMDLOutOpCode = {Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_COPY_UB,
                                                      Opcode::OP_UB_COPY_L1, Opcode::OP_L0C_COPY_UB_DUAL_DST};
    bool cubeMDLCondition = cubeMDLOutOpCode.count(opCode);
    bool useAttrShapeOffsetForOutputGM = OpcodeManager::Inst().IsCopyOut(opCode) &&
                                         logicalTensor.GetMemoryTypeOriginal() == MEM_DEVICE_DDR;
    std::shared_ptr<CopyOpAttribute> attr = std::dynamic_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    if (attr != nullptr && (cubeMDLCondition || useAttrShapeOffsetForOutputGM)) {
        // only used for 1. L1 Copy; 2. spilling to gm scene(e.g., ooo spilling); 3. matmul Multi-Data Load scene.
        CODEGEN_LOGI("start update offset for GM output");
        UpdateOffsetValueFromAttr(attr->GetCopyOutAttr().second, operandIdx);
        return;
    }

    offset[operandIdx] = logicalTensor.offset; // Local Tensor offset just use offset from LogicalTensor
    CODEGEN_LOGI("UpdateOffsetForOutput logicalTensor offset is %s", IntVecToStr(offset[operandIdx]).c_str());
}

void CodeGenOp::UpdateShapeAndOffset(const Operation& ops, const LogicalTensor& logicalTensor, bool isInput,
                                     int operandIdx, size_t ioIdx)
{
    UpdateShape(ops, logicalTensor, operandIdx);
    if (isInput) {
        UpdateOffsetForInput(ops, logicalTensor, operandIdx);
    } else {
        UpdateOffsetForOutput(ops, logicalTensor, operandIdx);
    }
    bool needCombineAxis = NeedCombineAxis(ops, isInput, ioIdx);
    if (needCombineAxis) {
        CombineAxisOffset(ops, operandIdx);
        CombineAxisShape(ops, operandIdx);
    }
}

void CodeGenOp::UpdateScalarValue(const Operation& ops)
{
    if (ops.HasAttr(OpAttributeKey::scalar)) {
        extOperandVal = ops.GetElementAttribute(OpAttributeKey::scalar);
    }
    if (ops.HasAttr(OpAttributeKey::dynScalar)) {
        extSymbolicScalar = ops.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    }
    if (ops.HasAttr(OpAttributeKey::vectorScalar)) {
        extScalarVec = ops.GetVectorElementAttribute(OpAttributeKey::vectorScalar);
    }
}

bool ShouldSkipIOperand(const std::shared_ptr<LogicalTensor>& tensor, const Operation& ops)
{
    Opcode opcode = ops.GetOpcode();
    if (opcode == Opcode::OP_A_MUL_B || opcode == Opcode::OP_A_MULACC_B) {
        bool isAcc = false;
        ops.GetAttr(OP_ATTR_PREFIX + "gm_acc", isAcc);
        return isAcc && tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
    }
    return false;
}

void CodeGenOp::Init(const Operation& ops)
{
    ASSERT(OperErr::OPERAND_COUNT_EXCEEDED, ops.iOperand.size() + ops.oOperand.size() <= MAX_OPERANDS)
        << "can not support ops.iOperand.size: " << ops.iOperand.size()
        << ", ops.oOperand.size: " << ops.oOperand.size() << ", Op is " << ops.Dump();

    isDynamicFunction = functionType == FunctionType::DYNAMIC_LOOP_PATH;
    isSupportDynamicAligned = isDynamicAligned || config::GetCodeGenOption<bool>(SUPPORT_DYNAMIC_ALIGNED);

    // update opcode and tileOpName
    UpdateTileOpInfo(ops);
    ASSERT(OperErr::OPERATION_INIT_FAILED, !tileOpName.empty()) << "empty tileOpName for ops: " << ops.Dump();

    isSupportTileTensor = ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true) &&
                          SUPPORT_TILETENSOR_OPS.find(opCode) != SUPPORT_TILETENSOR_OPS.end();
    CODEGEN_LOGI(
        "Init CodeGenOp from Operation, isDynamicFunction: %d, isSupportDynamicAligned: %d, isSupportTileTensor: %d",
        isDynamicFunction, isSupportDynamicAligned, isSupportTileTensor);

    opCodeStr = OpcodeManager::Inst().GetOpcodeStr(opCode);

    int operandIdx = 0;
    int oOperandCnt = 0;
    int iOperandCnt = 0;

    for (size_t i = 0; i < ops.oOperand.size(); ++i) {
        const auto& output = ops.oOperand[i];
        UpdateCodegenOpInfoByTensor(ops, false, output, operandIdx, i);
        ++oOperandCnt;
    }

    // if no output like WriteRemote OP, set operandIdx=1 for input
    if (operandIdx == 0) {
        operandIdx = 1;
    }

    for (size_t i = 0; i < ops.iOperand.size(); ++i) {
        const auto& input = ops.iOperand[i];
        if (ShouldSkipIOperand(input, ops)) {
            continue;
        }
        UpdateCodegenOpInfoByTensor(ops, true, input, operandIdx, i);
        ++iOperandCnt;
    }

    operandCnt = oOperandCnt + iOperandCnt;

    syncQueue = ops.syncQueue_;
    UpdateScalarValue(ops);
    UpdateOpAttribute(ops);
}

bool CodeGenOp::IsNeedUseNormalAddrAlloc(const Operation& ops) const
{
    std::string opcKey = OP_EMUOP_PREFIX + "opc";
    bool isTensorExtract = ops.HasAttr(opcKey) && (ops.GetIntAttribute(opcKey) == EMUOP_TENSOR_EXTRACT);
    bool res = !isSupportTileTensor || OpcodeManager::Inst().IsSharedMemory(opCode) || isTensorExtract;
    return res;
}

void CodeGenOp::UpdateCodegenOpInfoByTensor(const Operation& ops, bool isInput,
                                            const std::shared_ptr<LogicalTensor>& tensor, int& operandIdx, size_t ioIdx)
{
    operandWithMagic[operandIdx] = tensor->GetMagic();
    if (IsNeedUseNormalAddrAlloc(ops)) {
        sm->AddTensorUseNormalAlloc(tensor);
    }
    dynamicOffset[operandIdx] = tensor->GetDynOffset();
    operandDtype[operandIdx] = tensor->tensor->datatype;
    UpdateShapeAndOffset(ops, *tensor, isInput, operandIdx, ioIdx);
    auto it = OPERAND_TYPE_TO_MEMORY_TYPE.find(tensor->GetMemoryTypeOriginal());
    ASSERT(OperErr::OPERAND_TYPE_UNSUPPORTED, it != OPERAND_TYPE_TO_MEMORY_TYPE.end())
        << "can not support memory type: " << static_cast<size_t>(tensor->GetMemoryTypeOriginal()) << ", Tensor is "
        << tensor->Dump();
    operandType[operandIdx] = it->second;
    tensorAttrs[operandIdx] = tensor->GetAllAttr();
    ++operandIdx;
}

void CodeGenOp::UpdateOpAttribute(const Operation& ops)
{
    opAttrs = ops.GetAllAttr();
    isInputForceCombineAxis = ops.HasAttr(OpAttributeKey::inputCombineAxis);

    ConvertAttribute(ops);
}

std::string CodeGenOp::GenOpAttr(bool hasExistingParam) const
{
    if (opAttrs.empty()) {
        return {};
    }

    std::vector<std::string> attrList;
    for (const auto& kv : opAttrs) {
        if (kv.first.substr(0, OP_ATTR_PREFIX.size()) != OP_ATTR_PREFIX) {
            continue;
        }
        if (kv.second.type() == typeid(int64_t)) {
            attrList.push_back(std::to_string(AnyCast<int64_t>(kv.second)));
        } else if (kv.second.type() == typeid(bool)) {
            attrList.push_back(std::to_string(AnyCast<bool>(kv.second)));
        } else if (kv.second.type() == typeid(std::vector<int64_t>)) {
            auto vec = AnyCast<std::vector<int64_t>>(kv.second);
            for (auto v : vec) {
                attrList.push_back(std::to_string(v));
            }
        }
    }

    if (attrList.empty()) {
        return {};
    }

    std::string joined = JoinString(attrList, CONN_COMMA);
    return hasExistingParam ? CONN_COMMA + joined : joined;
}

void CodeGenOp::ConvertPoolAttribute(const Operation& operation)
{
    auto opc = operation.GetOpcode();
    if (opc != Opcode::OP_MAX_POOL) {
        return;
    }

    std::vector<std::string> intAttrStrList{
        ConvOpAttributeKey::paddingLeft,   ConvOpAttributeKey::paddingTop, ConvOpAttributeKey::paddingRight,
        ConvOpAttributeKey::paddingBottom, ConvOpAttributeKey::strideh,    ConvOpAttributeKey::stridew,
        PoolOpAttributeKey::poolh,         PoolOpAttributeKey::poolw,
    };
    for (size_t i = 0; i < intAttrStrList.size(); i++) {
        poolParams.push_back(operation.GetIntAttribute(intAttrStrList[i]));
    }
}

void CodeGenOp::ConvertAttribute(const Operation& operation)
{
    ASSERT(OperErr::OPERAND_COUNT_EXCEEDED, operation.iOperand.size() + operation.oOperand.size() <= MAX_OPERANDS)
        << "can not support operation.iOperand.size: " << operation.iOperand.size()
        << ", operation.oOperand.size: " << operation.oOperand.size() << ", Op is " << operation.Dump();
    if (opCode == Opcode::OP_CONV2D || opCode == Opcode::OP_CONV3D || opCode == Opcode::OP_CONV_ADD) {
        std::vector<std::string> intAttrStrList{
            ConvOpAttributeKey::cin,        ConvOpAttributeKey::cout,         ConvOpAttributeKey::paddingLeft,
            ConvOpAttributeKey::paddingTop, ConvOpAttributeKey::paddingRight, ConvOpAttributeKey::paddingBottom,
            ConvOpAttributeKey::strideh,    ConvOpAttributeKey::stridew,      ConvOpAttributeKey::hposX,
            ConvOpAttributeKey::hsteP,      ConvOpAttributeKey::wposX,        ConvOpAttributeKey::wstep,
            ConvOpAttributeKey::hoffsetY,   ConvOpAttributeKey::woffsetY,     ConvOpAttributeKey::reluType,
            ConvOpAttributeKey::reluAlpha,  ConvOpAttributeKey::clearFlag,    ConvOpAttributeKey::hasAccFlag,
            ConvOpAttributeKey::hasEltFlag, ConvOpAttributeKey::hasBiasFlag,  ConvOpAttributeKey::eltBrcbFlag,
            ConvOpAttributeKey::eltMode,
        };
        // (Cin, Cout, PaddingLeft, PaddingTop, PaddingRight, PaddingBottom, Stride1, Stride2, HPosX, HStep, WPosX,
        // WStep, HOffsetY, WOffsetY, reluType, relu_alpha, clearFlag, has_acc_flag, has_elt_flag, has_bias_flag,
        // elt_brcb_flag, elt_mode, hasQuantPreVector, hasQuantPostVector, hasAntiqVector)
        for (size_t i = 0; i < intAttrStrList.size(); i++) {
            convParams.push_back(operation.GetIntAttribute(intAttrStrList[i]));
        }
        std::vector<std::string> longAttrStrList{
            FixpOpAttributeKey::quantPreScalar,
            FixpOpAttributeKey::quantPostScalar,
            FixpOpAttributeKey::antiqScalar,
        };
        for (size_t i = 0; i < longAttrStrList.size(); i++) {
            convParams.push_back(operation.GetIntAttribute(longAttrStrList[i]));
        }
    }
    if (opCode == Opcode::OP_L1_COPY_IN_FRACTAL_Z) {
        convParams.push_back(operation.GetIntAttribute(ConvOpAttributeKey::fmapC0));
    }

    if (opCode == Opcode::OP_L1_TO_FIX || opCode == Opcode::OP_L1_TO_FIX_RELU_PRE ||
        opCode == Opcode::OP_L1_TO_FIX_RELU_POST || opCode == Opcode::OP_L1_TO_FIX_QUANT_POST ||
        opCode == Opcode::OP_L1_TO_FIX_ELT_ANTIQ || opCode == Opcode::OP_L1_TO_FIX_MTE2_ANTIQ) {
        convParams.push_back(operation.GetIntAttribute(FixpOpAttributeKey::fbAddrSpace));
    }

    ConvertPoolAttribute(operation);
}

void CodeGenOp::UpdateTileOpInfo(const Operation& ops)
{
    opCode = ops.GetOpcode();
    tileOpName = GetTileOpName(opCode);

    CODEGEN_LOGI("enter tileOpName is %s, opcode = %s", tileOpName.c_str(),
                 OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());

    if (opCode == Opcode::OP_COPY_IN && !ops.oOperand.empty()) {
        MemoryType memtype = ops.oOperand[0]->GetMemoryTypeOriginal();
        if (memtype == MemoryType::MEM_UB) {
            tileOpName = "TileOp::UBCopyIn";
            opCode = Opcode::OP_UB_COPY_IN;
        } else if (memtype == MemoryType::MEM_L1) {
            tileOpName = "TileOp::L1CopyIn";
            opCode = Opcode::OP_L1_COPY_IN;
        }
    } else if (opCode == Opcode::OP_COPY_OUT && !ops.iOperand.empty()) {
        MemoryType memtype = ops.iOperand[0]->GetMemoryTypeOriginal();
        if (memtype == MemoryType::MEM_UB) {
            tileOpName = "TileOp::UBCopyOut";
            opCode = Opcode::OP_UB_COPY_OUT;
        } else if (memtype == MemoryType::MEM_L1) {
            tileOpName = "TileOp::L1CopyOut";
            opCode = Opcode::OP_L1_COPY_OUT;
        } else if (memtype == MemoryType::MEM_L0C) {
            tileOpName = "TileOp::L0CCopyOut";
            opCode = Opcode::OP_L0C_COPY_OUT;
        }
    }

    if (!isDynamicFunction || DISTRIBUTED_OPS.count(opCode)) {
        return;
    }

    std::string dynPrefix = "Dyn";
    size_t nameSpaceLen = std::strlen("TileOp::");
    bool isNeedInsertDynPrefix = isDynamicFunction &&
                                 SUPPORT_DYNAMIC_UNALIGNED_OPS.find(opCode) != SUPPORT_DYNAMIC_UNALIGNED_OPS.end();
    CODEGEN_LOGI("isNeedInsertDynPrefix is %d, opcode = %s", isNeedInsertDynPrefix,
                 OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());
    if (isNeedInsertDynPrefix) {
        tileOpName.insert(nameSpaceLen, dynPrefix);
    }

    CODEGEN_LOGI("after UpdateTileOpInfo: tileOpName = %s, opCode = %s", tileOpName.c_str(),
                 OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());
}

} // namespace npu::tile_fwk
