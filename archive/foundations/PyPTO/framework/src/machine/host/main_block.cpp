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
 * \file main_block.cpp
 * \brief
 */

#include "main_block.h"
#include "codegen/codegen.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {

static bool IsEnableVF()
{
    if (config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) == 1) {
        return true;
    }
    bool enableVF = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
    enableVF = enableVF && config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
    return enableVF;
}

MainBlockCondBulider::MainBlockCondBulider() = default;

void MainBlockCondBulider::DisableMainBlock()
{
    mainBlockDisabled_ = true;
    mainBlockCondGroup_.clear();
    mainBlockStrSet_.clear();
    exprConstMap_.clear();
}

void MainBlockCondBulider::AddUniqueCondition(const SymbolicScalar& newCond)
{
    if (mainBlockDisabled_) {
        return;
    }

    if (newCond.Raw() != nullptr && newCond.Raw()->IsImmediate()) {
        auto imm = std::dynamic_pointer_cast<RawSymbolicImmediate>(newCond.Raw());
        if (imm != nullptr && imm->Immediate() == 0) {
            DisableMainBlock();
            return;
        }
    }

    SymbolicScalar cond = SymbolicScalar(false);
    std::string condStr = newCond.Dump();
    if ((mainBlockStrSet_.find(condStr) != mainBlockStrSet_.end()) ||
        (mainBlockStrSet_.find(cond.Dump()) != mainBlockStrSet_.end())) {
        return;
    }

    mainBlockStrSet_.insert(condStr);
    mainBlockCondGroup_.push_back(newCond);
}

bool MainBlockCondBulider::CheckShapeEquality(const Shape& shape, const std::vector<SymbolicScalar>& dynShape)
{
    SymbolicScalar cond = SymbolicScalar(false);
    if (shape.size() != dynShape.size()) {
        AddUniqueCondition(cond);
        return false;
    }

    for (uint32_t i = 0; i < shape.size(); i++) {
        if (shape[i] == -1) { // -1: copy_in, copy_out and callop dynamic axis shape
            continue;
        }
        if (mainBlockDisabled_) {
            return false;
        }
        std::string exprKey = dynShape[i].Dump();
        auto it = exprConstMap_.find(exprKey);
        if (it != exprConstMap_.end() && it->second != shape[i]) {
            MACHINE_LOGW("mainBlock condition contradiction: same expression %s requires "
                         "shape to be both %ld and %ld, disabling mainblock",
                         exprKey.c_str(), it->second, static_cast<int64_t>(shape[i]));
            DisableMainBlock();
            return false;
        }
        exprConstMap_[exprKey] = shape[i];
        cond = (shape[i] == dynShape[i]);
        AddUniqueCondition(cond);
        if (mainBlockDisabled_) {
            return false;
        }
    }
    return true;
}

void MainBlockCondBulider::CollectCallopMainBlockConds(Function* func)
{
    if (!IsEnableVF()) {
        AddUniqueCondition(SymbolicScalar(false));
        return;
    }

    auto checkOperand = [&](auto& op, auto& shape, auto& validshape, const char* tag) -> bool {
        auto cond = CheckShapeEquality(shape, validshape);
        if (!cond) {
            MACHINE_LOGW("get mainBlock flag false, op code %s, %s shape is %s, validShape is %s",
                         op.GetOpcodeStr().c_str(), tag, IntVecToStr(shape).c_str(), IntVecToStr(validshape).c_str());
        }
        return cond;
    };

    for (auto& op : func->Operations()) {
        for (auto& iop : op.GetIOperands()) {
            if (!checkOperand(op, iop->shape, iop->GetDynValidShape(), "iop")) {
                return;
            }
        }
        for (auto& oop : op.GetOOperands()) {
            if (!checkOperand(op, oop->shape, oop->GetDynValidShape(), "oop")) {
                return;
            }
        }
    }
}

bool MainBlockCondBulider::CheckLeafOperand(const Operation& op, const std::shared_ptr<LogicalTensor>& iop,
                                            int coaIndexBase, const std::vector<SymbolicScalar>& linearArgList,
                                            const char* tag)
{
    // coaIndexBase < 0 表示该 operand 未被 NormalizeCoa 处理（未调用 SetIOpAtt/SetOOpAtt），静态shape
    if (coaIndexBase < 0) {
        return true;
    }

    int dim = static_cast<int>(iop->shape.size());
    if (dim == 0) {
        MACHINE_LOGW("get mainBlock flag false, op code %s, %s shape dim is 0", op.GetOpcodeStr().c_str(), tag);
        DisableMainBlock();
        return false;
    }

    int shapeStart = coaIndexBase + COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_SHAPE;
    int validShapeStart = coaIndexBase + COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_VALIDSHAPE;
    if (shapeStart + dim > static_cast<int>(linearArgList.size()) ||
        validShapeStart + dim > static_cast<int>(linearArgList.size())) {
        MACHINE_LOGW("get mainBlock flag false, op code %s, %s linearArgList too small: size=%zu, need=%d",
                     op.GetOpcodeStr().c_str(), tag, linearArgList.size(), std::max(shapeStart, validShapeStart) + dim);
        DisableMainBlock();
        return false;
    }

    std::vector<SymbolicScalar> shapeScalars(linearArgList.begin() + shapeStart,
                                             linearArgList.begin() + shapeStart + dim);
    Shape shape = SymbolicScalar::Concrete(shapeScalars, -1);

    std::vector<SymbolicScalar> dynValidShape(linearArgList.begin() + validShapeStart,
                                              linearArgList.begin() + validShapeStart + dim);

    auto cond = CheckShapeEquality(shape, dynValidShape);
    if (!cond) {
        MACHINE_LOGW("get mainBlock flag false, op code %s, %s shape is %s, validShape is %s",
                     op.GetOpcodeStr().c_str(), tag, IntVecToStr(shape).c_str(), IntVecToStr(dynValidShape).c_str());
    }
    return cond;
}

void MainBlockCondBulider::CollectLeafMainBlockConds(Function* func, const std::vector<SymbolicScalar>& linearArgList)
{
    if (!IsEnableVF()) {
        AddUniqueCondition(SymbolicScalar(false));
        return;
    }

    for (auto& op : func->Operations()) {
        if (mainBlockDisabled_) {
            return;
        }

        for (size_t k = 0; k < op.GetIOperands().size(); k++) {
            auto& iop = op.GetIOperands()[k];
            int coaIndexBase = op.GetIOpAttrOffset(k);
            if (!CheckLeafOperand(op, iop, coaIndexBase, linearArgList, "iop")) {
                return;
            }
        }

        for (size_t k = 0; k < op.GetOOperands().size(); k++) {
            auto& oop = op.GetOOperands()[k];
            int coaIndexBase = op.GetOOpAttrOffset(k);
            if (!CheckLeafOperand(op, oop, coaIndexBase, linearArgList, "oop")) {
                return;
            }
        }
    }
}

SymbolicScalar MainBlockCondBulider::BuildMainBlockExpression()
{
    SymbolicScalar runtimeSelect("RUNTIME_Select");
    SymbolicScalar runtimeAnd("RUNTIME_And");
    SymbolicScalar cond = false;
    if (mainBlockDisabled_ || mainBlockCondGroup_.empty()) {
        return runtimeSelect(cond, 1, 0);
    }

    if (mainBlockCondGroup_.size() > MAX_RUNTIME_AND_NESTING_DEPTH) {
        MACHINE_LOGW("runtimeAnd nesting depth too large (%zu), disable mainblock", mainBlockCondGroup_.size());
        return runtimeSelect(false, 1, 0);
    }

    cond = true;
    for (const auto& iter : mainBlockCondGroup_) {
        std::string exprStr = iter.Dump();
        if (exprStr.find("RUNTIME_GetTensorDataInt32") != std::string::npos) {
            MACHINE_LOGW("AICPU does not support RUNTIME_GetTensorDataInt32");
            return runtimeSelect(false, 1, 0);
        }
        cond = runtimeAnd(cond, iter);
    }

    cond = runtimeSelect(cond, 1, 0);
    return cond;
}

void MainBlockCondBulider::Gencode(Function* function, int devRootIndex)
{
    if (IsEnableVF()) {
        bool isDynamicAligned = function->paramConfigs_.dynamicAlignedOps;
        npu::tile_fwk::CodeGenCtx codeGenCtxMainBlock("", config::GetEmitPath("kernel_aicore"), true, isDynamicAligned,
                                                      devRootIndex);
        npu::tile_fwk::CodeGen codeGenMainBlock(codeGenCtxMainBlock);
        codeGenMainBlock.GenCode(*function);
    }
}

const std::vector<SymbolicScalar>& MainBlockCondBulider::GetCondGroup() const { return mainBlockCondGroup_; }

const std::unordered_set<std::string>& MainBlockCondBulider::GetCondStrSet() const { return mainBlockStrSet_; }
} // namespace npu::tile_fwk
