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
 * \file codegen_for_block.cpp
 * \brief
 */

#include "codegen_for_block.h"

#include "interface/operation/operation.h"
#include "interface/tensor/symbolic_scalar.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {
const std::string DEFAULT_TENSOR_OFFSET = "tileOffsets";
constexpr const int BINARY_OP_INPUT_CNT = 2;

std::string ForNode::Print() const
{
    std::ostringstream os;
    os << "for (";
    PrintInit(os);
    PrintCond(os);
    PrintUpdate(os);
    os << ") {\n";
    return os.str();
}

void ForNode::PrintInit(std::ostringstream& os) const
{
    os << "uint16_t " << loopVar << " = " << SymbolicExpressionTable::BuildExpression(start) << SEMICOLON_BLANK;
}

void ForNode::PrintCond(std::ostringstream& os) const
{
    os << loopVar << " < " << SymbolicExpressionTable::BuildExpression(extent) << SEMICOLON_BLANK;
}

void ForNode::PrintUpdate(std::ostringstream& os) const
{
    if (step.ConcreteValid() && step.Concrete() == 1) {
        os << "++" << loopVar;
    } else {
        os << loopVar << " += " << SymbolicExpressionTable::BuildExpression(step);
    }
}

void ForBlockManager::UpdateAxesList(const std::vector<SymbolicScalar>& axesList)
{
    axesList_ = axesList;
    FillVecWithDummyInHead<SymbolicScalar>(axesList_, MAX_LOOP_DEPTH - axesList.size(), 1);
    CODEGEN_LOGI("axesList_ after fill is : %s, ", IntVecToStr(axesList_).c_str());
    std::vector<std::string> offsetInLoop(axesList_.size(), "");
    for (size_t i = 0; i < axesList_.size(); ++i) {
        std::string loopVar = "idx" + std::to_string(i);
        ForNode forNode{loopVar, 0, axesList_[i], 1};
        forNodes_.push_back(forNode);
        offsetInLoop[i] = axesList_[i].ConcreteValid() && axesList_[i].Concrete() == 1 ? "0" : loopVar;
    }

    allOffsetsInLoop_[offsetInLoop] = DEFAULT_TENSOR_OFFSET;
    defaultOffset_ = offsetInLoop;
    ++offsetCnt_;
}

std::string ForBlockManager::Print() const
{
    std::ostringstream os;
    PrintForHeader(os);
    PrintForBody(os);
    PrintForEnd(os);
    return os.str();
}

void ForBlockManager::PrintForHeader(std::ostringstream& os) const
{
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        PrintIndent(os, i);
        os << forNodes_[i].Print();
    }
}

void ForBlockManager::PrintForBody(std::ostringstream& os) const
{
    PrintOffsetDef(os);
    PrintSetAddrs(os);
    PrintTileOps(os);
}

void ForBlockManager::PrintForEnd(std::ostringstream& os) const
{
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        PrintIndent(os, MAX_LOOP_DEPTH - i - 1);
        os << "}\n";
    }
}

void ForBlockManager::PrintOffsetDef(std::ostringstream& os) const
{
    for (const auto& [offsetInLoop, offsetName] : allOffsetsInLoop_) {
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        os << "auto " << offsetName << " = "
           << "TileOffset" << WrapParamByParentheses(offsetInLoop) << STMT_END;
    }
}

void ForBlockManager::PrintSetAddrs(std::ostringstream& os) const
{
    for (const auto& [tensor, offset] : tensorOffset_) {
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        PrintSetAddrSingle(os, tensor, offset);
    }
}

void ForBlockManager::PrintSetAddrSingle(std::ostringstream& os, const std::string& tensor,
                                         const std::string& offset) const
{
    std::string fullDimTensor = sm_->QueryTileTensorFullDimByTensorInLoop(tensor);
    os << tensor << ".SetAddr(" << fullDimTensor << ".GetLinearAddr(" << offset << "));\n";
}

void ForBlockManager::PrintTileOps(std::ostringstream& os) const
{
    for (const auto& tileOp : opList_) {
        CODEGEN_LOGI("tileOp is : %s", tileOp.c_str());
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        os << tileOp;
    }
}

bool NeedUpdateOffsetInLoop(Opcode opCode, int tensorMagic, const Operation& oper)
{
    // expand op only update input offset
    if (opCode == Opcode::OP_EXPAND) {
        const auto& tensors = oper.GetIOperands();
        bool isInput = std::any_of(tensors.begin(), tensors.end(),
                                   [&](const auto& tensor) { return tensor->GetMagic() == tensorMagic; });
        return isInput;
    }

    return true;
}

/*
Different tensors use different offsets, generate code like:
for idx0 in axis0 {
  for idx1 in axis1 {
    for idx2 in axis2 {
        auto tileOffsets = TileOffset(idx0, idx1, idx2);
        auto src0tileOffsets = TileOffset(0, 0, idx2);
        auto src1tileOffsets = TileOffset(0, idx1, 0);
        ubTensor_2_low2DimInLoop.SetAddr(ubTensor_2.GetLinearAddr(src1tileOffsets));
        ubTensor_0_low2DimInLoop.SetAddr(ubTensor_0.GetLinearAddr(src0tileOffsets));
        ubTensor_4_low2DimInLoop.SetAddr(ubTensor_4.GetLinearAddr(tileOffsets));
        TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_4_low2DimInLoop, ubTensor_0_low2DimInLoop, ubTensor_2_low2DimInLoop);
    }
  }
}
 */
void ForBlockManager::UpdateTensorOffsetInLoop(Opcode opCode, int tensorMagic, const Operation& oper,
                                               const std::string& tensorNameInLoop)
{
    if (!NeedUpdateOffsetInLoop(opCode, tensorMagic, oper)) {
        tensorOffset_[tensorNameInLoop] = DEFAULT_TENSOR_OFFSET;
        return;
    }

    auto newOffset = BuildOffsetInLoop(tensorMagic, oper);

    std::string offsetName;
    auto iter = allOffsetsInLoop_.find(newOffset);
    if (iter != allOffsetsInLoop_.end()) {
        offsetName = iter->second;
    } else {
        offsetName = DEFAULT_TENSOR_OFFSET + "_" + std::to_string(offsetCnt_);
        allOffsetsInLoop_[newOffset] = offsetName;
        ++offsetCnt_;
    }
    tensorOffset_[tensorNameInLoop] = offsetName;
}

bool IsExpandOrBrcAxes(const std::vector<int64_t>& rawShape, const std::vector<int>& axes, int axis)
{
    return rawShape[axis] == 1 || std::find(axes.begin(), axes.end(), axis) != axes.end();
}

std::vector<int> GetNormalizedExpandAxes(unsigned originalDimSize, int tensorMagic, const Operation& oper)
{
    std::vector<int> normalizedExpandAxes;
    if (oper.GetOpcode() == Opcode::OP_EXPAND) {
        auto axes = oper.GetVectorIntAttribute(OpAttributeKey::expandDims);
        CODEGEN_LOGI("expandDims is : %s", IntVecToStr(axes).c_str());
        normalizedExpandAxes = NormalizeExpandAxes(axes, originalDimSize, SHAPE_DIM5);
        return normalizedExpandAxes;
    }

    auto brcOperand = oper.GetVectorIntAttribute(OpAttributeKey::brcOperand);
    CODEGEN_LOGI("brcOperand is : %s", IntVecToStr(brcOperand).c_str());
    if (brcOperand.empty()) {
        return normalizedExpandAxes;
    }

    // brcOperand [0, 1, 1, 2, 2] means:
    // axis 0: brcOperand value is 0, means BroadcastOperand::NONE, do not need broadcast,
    // axis 1, 2: brcOperand value is 1, means BroadcastOperand::LEFT_OPERAND, left operand need broadcast,
    // axis 3, 4: brcOperand value is 2, means BroadcastOperand::RIGHT_OPERAND, right operand need broadcast
    FillVecWithDummyInHead<int64_t>(brcOperand, MAX_DIM - brcOperand.size(), 0);

    const auto& inputs = oper.GetIOperands();
    ASSERT(OperErr::OPERAND_COUNT_NOT_MATCHED, inputs.size() >= BINARY_OP_INPUT_CNT)
        << "GetIOperands size is " << inputs.size() << ", op is " << oper.Dump();

    for (size_t i = 0; i < brcOperand.size(); ++i) {
        if ((tensorMagic == inputs[0]->GetMagic() && brcOperand[i] == ToUnderlying(BroadcastOperand::LEFT_OPERAND)) ||
            (tensorMagic == inputs[1]->GetMagic() && brcOperand[i] == ToUnderlying(BroadcastOperand::RIGHT_OPERAND))) {
            // if current axis need broadcast, record axis in normalizedExpandAxes
            normalizedExpandAxes.push_back(i);
        }
    }

    return normalizedExpandAxes;
}

OffsetInLoop ForBlockManager::BuildOffsetInLoop(int tensorMagic, const Operation& oper)
{
    int opMagic = oper.GetOpMagic();
    auto tileTensor = sm_->QueryTileTensorByMagic(tensorMagic, opMagic);
    ASSERT(GenCodeErr::TENSOR_NOT_FOUND, tileTensor != nullptr)
        << "QueryTileTensorByMagic tensor magic is " << tensorMagic << ", op magic is " << opMagic;

    auto rawShape = tileTensor->rawShape;
    FillVecWithDummyInHead<int64_t>(rawShape, SHAPE_DIM5 - rawShape.size(), 1);

    // normalize axes up to MAX_DIM(e.g. SHAPE_DIM5)
    // e.g. expand [2, 1, 1, 6] to [2, 8, 4, 6] axes is [1, 2], after normalizedExpandAxes is [2, 3].
    // Hoisted loop axes is [0, 1, 2], so axis ‘2’ is in normalizedExpandAxes and newOffset[2] should be set to zero.
    std::vector<int> normalizedExpandAxes = GetNormalizedExpandAxes(tileTensor->rawShape.size(), tensorMagic, oper);

    CODEGEN_LOGI("rawShape is : %s, normalizedExpandAxes is : %s", IntVecToStr(rawShape).c_str(),
                 IntVecToStr(normalizedExpandAxes).c_str());

    auto newOffset = defaultOffset_;
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        if (IsExpandOrBrcAxes(rawShape, normalizedExpandAxes, i)) {
            newOffset[i] = "0";
        }
    }

    CODEGEN_LOGI("newOffset is : %s", IntVecToStr(newOffset).c_str());
    return newOffset;
}

void ForBlockManager::AddTensorInLoopBody(const std::string& tensorFullDim, const TileTensor& tileTensor,
                                          const Operation& oper, Opcode opCode)
{
    int opMagic = oper.GetOpMagic();
    CODEGEN_LOGI("AddTensorInLoopBody : tensorFullDim: %s, loop tensor: %s,op magic: %d, op code: %s",
                 tensorFullDim.c_str(), tileTensor.tensorName.c_str(), opMagic,
                 OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());
    std::string tensorNameInLoop = sm_->AddTileTensor(opMagic, tileTensor);
    sm_->InsertTensorNameInLoopToFullDim(tensorNameInLoop, tensorFullDim);
    UpdateTensorOffsetInLoop(opCode, tileTensor.magic, oper, tensorNameInLoop);
}

} // namespace npu::tile_fwk
