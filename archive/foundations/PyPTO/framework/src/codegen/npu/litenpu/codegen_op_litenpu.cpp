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
 * \file codegen_op_litenpu.cpp
 * \brief
 */

#include "codegen_op_litenpu.h"
#include "codegen/npu/litenpu/codegen_litenpu.h"

namespace npu::tile_fwk {

const int RUNTIME_COA_BASE = 2;
const int RUNTIME_COA_IDX = 3;

// ensure funcType is static, and isUnderDynamicFunc is false
CodeGenOpLiteNPU::CodeGenOpLiteNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOpNPU(ctx), subFunc(ctx.subFunc)
{
    InitOpsGenMap();
    forBlkMgr_ = ctx.forBlockManager;
    CodeGenOp::Init(ctx.operation);
    UpdateGmParamIdx(ctx.operation);
    UpdateTileTensorInfo();
}

TileTensor CodeGenOpLiteNPU::QueryTileTensorByIdx(int paramIdx) const
{
    const int tensorMagic = operandWithMagic[paramIdx];
    const int opMagic = originalOp.GetOpMagic();
    const TileTensor* tileTensor = nullptr;
    tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);
    if (tileTensor != nullptr) {
        CODEGEN_LOGI("QueryTileTensorByIdx found: %s", tileTensor->ToString().c_str());
        return *tileTensor;
    }

    ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false) << "TileTensor: paramIdx " << paramIdx << ", tensor magic "
                                                << tensorMagic << ", op magic " << opMagic << " is not found !!!";
    static TileTensor emptyTileTensor;
    return emptyTileTensor;
}

std::string CodeGenOpLiteNPU::GenGmParamVar(unsigned gmParamIdx) const
{
    return std::string("RealizedGM") + std::to_string(paramLocation[gmParamIdx]) + ".Addr";
}

TileTensor CodeGenOpLiteNPU::BuildTileTensor(int paramIdx, const std::string& usingType,
                                             const TileTensorShape& tileTensorShape)
{
    int64_t gmOffset{0};
    bool isSpillToGm = GetTensorAttr(paramIdx, OpAttributeKey::workspaceBaseOffset, gmOffset);

    TileTensor tileTensor;
    tileTensor.isConstant = functionType == FunctionType::STATIC || isMainBlock;
    tileTensor.magic = operandWithMagic[paramIdx];
    tileTensor.isInLoop = tileTensorShape.isInLoop;

    tileTensor.dim = tileTensor.isConstant ? tileTensorShape.shape.size() : tileTensorShape.dynamicValidShape.size();

    tileTensor.dtype = operandDtype[paramIdx];
    tileTensor.bufType = operandType[paramIdx];

    if (tileTensor.bufType == OperandType::BUF_DDR) {
        tileTensor.bufVar = isSpillToGm ? GenGMAddrExprWithOffset(paramIdx, CODEGEN_LITENPU_WORKSPACE) :
                                          GenGmParamVar(paramIdx);
    } else {
        tileTensor.bufVar = sm->QueryVarNameByTensorMagic(tileTensor.magic, true);
    }

    tileTensor.usingType = usingType;

    tileTensor.tensorName = sm->GenTensorName(tileTensor.bufType);
    UpdateTileTensorShapeAndStride(paramIdx, tileTensor, isSpillToGm, tileTensorShape);

    tileTensor.localBufOffset = offset[paramIdx];

    return tileTensor;
}

void CodeGenOpLiteNPU::UpdateTileTensorShapeAndStride([[maybe_unused]] int paramIdx, TileTensor& tileTensor,
                                                      [[maybe_unused]] bool isSpillToGm,
                                                      const TileTensorShape& tileTensorShape)
{
    auto newShape = tileTensorShape.shape;
    auto newRawShape = tileTensorShape.rawShape;
    auto newDynValidShape = tileTensorShape.dynamicValidShape;
    CODEGEN_LOGI("newShape is %s, newRawShape is %s, newDynValidShape is %s", IntVecToStr(newShape).c_str(),
                 IntVecToStr(newRawShape).c_str(), IntVecToStr(newDynValidShape).c_str());

    tileTensor.rawShape = newRawShape;

    // ---- static or "main block" ----
    if (functionType == FunctionType::STATIC) {
        for (auto s : newShape) {
            tileTensor.shape.emplace_back(std::to_string(s));
        }
        tileTensor.stride = BuildStride(newRawShape);
        return;
    }
}

std::vector<std::string> CodeGenOpLiteNPU::GetGmOffsetForTileTensor(unsigned gmIdx) const
{
    int dim = static_cast<int>(rawShape[gmIdx].size());

    if (GetOffsetFromAttr(gmIdx)[ID0].IsValid()) {
        return GenSymbolicArgument(GetOffsetFromAttr(gmIdx));
    }

    return GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
}

void CodeGenOpLiteNPU::UpdateGmParamIdx(const Operation& oper)
{
    auto inParamLocSize = oper.inParamLocation_.size();
    auto outParamLocSize = oper.outParamLocation_.size();

    // Ops like UB_ALLOC have output operands, but does not have output
    // param locs, so here we should not assert 'outParamLocSize == outputTensors.size()' !
    ASSERT(OperErr::OPERAND_COUNT_NOT_MATCHED, inParamLocSize <= oper.iOperand.size())
        << "size of Op.inParamLocation_ is larger than input operands, Op is " << oper.Dump();
    ASSERT(OperErr::OPERAND_COUNT_NOT_MATCHED, outParamLocSize <= oper.oOperand.size())
        << "size of Op.outParamLocation_ is larger than output operands, Op is " << oper.Dump();

    CODEGEN_LOGI("inParamLocation = %s", IntVecToStr(oper.inParamLocation_).c_str());
    CODEGEN_LOGI("outParamLocation = %s", IntVecToStr(oper.outParamLocation_).c_str());

    std::copy(oper.outParamLocation_.begin(), oper.outParamLocation_.end(), paramLocation);
    std::copy(oper.inParamLocation_.begin(), oper.inParamLocation_.end(), paramLocation + oper.oOperand.size());
}

bool CodeGenOpLiteNPU::ContainsRuntimeCoaGetParmOffset(const std::vector<SymbolicScalar>& offsets) const
{
    for (const auto& oset : offsets) {
        auto raw = oset.Raw();
        if (raw != nullptr && raw->IsExpressionCall("RUNTIME_COA_GET_PARAM_OFFSET")) {
            return true;
        }
    }
    return false;
}

void CodeGenOpLiteNPU::ExtractCoaGetParamOffset(const SymbolicScalar& expr, int& base, int& idx) const
{
    auto raw = expr.Raw();
    auto exprRaw = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
    if (exprRaw == nullptr) {
        return;
    }
    const auto& operands = exprRaw->OperandList();
    if (operands.size() < RUNTIME_COA_IDX + 1) {
        return;
    }
    if (operands[RUNTIME_COA_BASE]->IsImmediate()) {
        base = static_cast<int>(
            std::static_pointer_cast<RawSymbolicImmediate>(operands[RUNTIME_COA_BASE])->Immediate());
    }
    if (operands[RUNTIME_COA_IDX]->IsImmediate()) {
        idx = static_cast<int>(std::static_pointer_cast<RawSymbolicImmediate>(operands[RUNTIME_COA_IDX])->Immediate());
    }
}
std::vector<SymbolicScalar> CodeGenOpLiteNPU::GetStaticOffsetFromLinearArgList(
    const std::vector<SymbolicScalar>& dynOffset) const
{
    std::vector<SymbolicScalar> result;
    std::vector<int> offsetFromDynOffset;

    for (size_t i = 0; i < dynOffset.size(); ++i) {
        int base = -1;
        int idx = -1;
        ExtractCoaGetParamOffset(dynOffset[i], base, idx);
        if (base < 0 || idx < 0) {
            continue;
        }
        int tmpOffset = base + idx;
        if (tmpOffset < 0) {
            continue;
        }
        offsetFromDynOffset.push_back(tmpOffset);
    }

    Function& subFunction = subFunc;
    if (subFunction.HasParent()) {
        Function* parentFunction = &subFunction.Parent();
        std::vector<std::shared_ptr<CallOpAttribute>> parentCallopAttrList = parentFunction->GetCallopAttrList();
        for (size_t i = 0; i < parentCallopAttrList.size(); ++i) {
            auto& callopAttr = parentCallopAttrList[i];
            std::vector<SymbolicScalar>& linearArgList = callopAttr->GetLinearArgList();
            for (size_t j = 0; j < offsetFromDynOffset.size(); ++j) {
                int linearIdx = offsetFromDynOffset[j] + 1;
                if (linearIdx < 0 || linearIdx >= static_cast<int>(linearArgList.size())) {
                    continue;
                }
                result.push_back(linearArgList[linearIdx]);
            }
        }
    }

    return result;
}

std::vector<SymbolicScalar> CodeGenOpLiteNPU::GetOffsetFromAttr(int idx) const
{
    auto dynOffset = CodeGenOpNPU::GetOffsetFromAttr(idx);
    if (!ContainsRuntimeCoaGetParmOffset(dynOffset)) {
        return dynOffset;
    }
    return GetStaticOffsetFromLinearArgList(dynOffset);
}

void CodeGenOpLiteNPU::UpdateGemvTileOpName()
{
    auto it = opAttrs.find(OpAttributeKey::isGemv);
    if (it == opAttrs.end()) {
        return;
    }
    bool isGemv = false;
    if (it->second.type() == typeid(bool)) {
        isGemv = AnyCast<bool>(it->second);
    } else if (it->second.type() == typeid(int64_t)) {
        isGemv = AnyCast<int64_t>(it->second) != 0;
    } else {
        return;
    }
    if (!isGemv) {
        return;
    }
    switch (opCode) {
        case Opcode::OP_A_MUL_B:
        case Opcode::OP_A_MULACC_B:
            tileOpName = "TGEMV";
            break;
        case Opcode::OP_L1_TO_L0A:
            tileOpName = "TExtractL1ToL0ND2ND";
            break;
        case Opcode::OP_UB_COPY_L1:
            tileOpName = "TCopyUB2L1ND2ND";
            break;
        default:
            break;
    }
}

} // namespace npu::tile_fwk
