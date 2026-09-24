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
 * \file cube_process.cpp
 * \brief
 */

#include "cube_process.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PreGraphProcess"

namespace npu::tile_fwk {
/*
resetDdr: A_MUL_B的清零后DDR输入，数据边表依赖
copyOutOp: 当前A_MUL_B链路最终搬出的L0C_Copy_Out
功能: 当前Matmul链路最终的CopyOut属性需要与清零时的CopyOut属性对齐
限制条件: 依赖前端使能切K场景下，Matmul的tile展开中显示对清零后的Gm按C矩阵的切分大小做切分
*/
void AlignCopyOutAttr(LogicalTensorPtr& resetDdr, Operation* copyOutOp)
{
    if (resetDdr->GetProducers().size() == 1) {
        auto ddrResetCopyOut = *resetDdr->GetProducers().begin();
        if (ddrResetCopyOut->GetOpcode() != Opcode::OP_COPY_OUT) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "DDR reset Op requires to be OP_COPY_OUT, but %s[%d]; Please check the Opcode. %s",
                              ddrResetCopyOut->GetOpcodeStr().c_str(), ddrResetCopyOut->GetOpMagic(),
                              GetFormatBacktrace(copyOutOp).c_str());
            return;
        }
        auto ddrResetCopyOutAttr = std::static_pointer_cast<CopyOpAttribute>(ddrResetCopyOut->GetOpAttribute());
        auto L0CCopyOutAttr = std::static_pointer_cast<CopyOpAttribute>(copyOutOp->GetOpAttribute());
        ddrResetCopyOutAttr->SetRawShape(L0CCopyOutAttr->GetRawShape());
        ddrResetCopyOutAttr->SetToOffset(L0CCopyOutAttr->GetToOffset());
    }
}

Status CubeProcess::AddL1CopyInAttr(const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue,
                                    int nValue) const
{
    auto copyInOp = *(input->GetProducers().begin());
    auto tensorL0 = copyInOp->GetIOperands().front();
    auto L1CopyInOp = *(tensorL0->GetProducers().begin());
    if (L1CopyInOp->GetOpcode() == Opcode::OP_VIEW || L1CopyInOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        /*
        1. View 对应大包搬运场景
        gm -> L1_COPY_IN -> L1 ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
                              \ ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
        2. Assemble 对应 Gather On L1 场景
        */
        tensorL0 = L1CopyInOp->GetIOperands().front();
        L1CopyInOp = *(tensorL0->GetProducers().begin());
    }
    /*L0C copy L1*/
    if (L1CopyInOp->GetOpcode() == Opcode::OP_L0C_TO_L1) {
        return SUCCESS;
    }
    L1CopyInOp->SetAttribute(COPY_IS_NZ, nzValue);
    APASS_LOG_DEBUG_F(Elements::Operation, "Update %s[%d] attr is_Nz: %d", L1CopyInOp->GetOpcodeStr().c_str(),
                      L1CopyInOp->GetOpMagic(), nzValue);
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0A || copyInOp->GetOpcode() == Opcode::OP_LOAD3D_CONV) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, mValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        APASS_LOG_DEBUG_F(Elements::Operation, "OP_L1_TO_L0A: Outer: %d, Inner: %d.", mValue, kValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0B || copyInOp->GetOpcode() == Opcode::OP_LOAD2D_CONV) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, nValue);
        APASS_LOG_DEBUG_F(Elements::Operation, "OP_L1_TO_L0B: Outer: %d, Inner: %d.", kValue, nValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_AT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, mValue);
        APASS_LOG_DEBUG_F(Elements::Operation, "OP_L1_TO_L0_AT: Outer: %d, Inner: %d.", kValue, mValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_BT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, nValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        APASS_LOG_DEBUG_F(Elements::Operation, "OP_L1_TO_L0_BT: Outer: %d, Inner: %d.", nValue, kValue);
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "Invalid Cube input %d, produced by %s[%d]. %s", input->GetMagic(),
                      copyInOp->GetOpcodeStr().c_str(), copyInOp->GetOpMagic(), GetFormatBacktrace(copyInOp).c_str());
    return FAILED;
}

Status CubeProcess::AddL0cCopyOutAttr(const std::shared_ptr<LogicalTensor> output, int nzValue, int mValue,
                                      int nValue) const
{
    for (auto& childOp : output->GetConsumers()) {
        if (childOp->GetOpcode() != Opcode::OP_COPY_OUT) {
            continue;
        }
        childOp->SetAttribute(COPY_IS_NZ, nzValue);
        childOp->SetAttribute(L0C_COPY_OUT_OUTER, mValue);
        childOp->SetAttribute(L0C_COPY_OUT_INNER, nValue);
        APASS_LOG_DEBUG_F(Elements::Operation, "Update %s[%d] attr is_Nz: %d, curH: %d, curW: %d.",
                          childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic(), nzValue, mValue, nValue);
    }
    return SUCCESS;
}

Status CubeProcess::UpdateCopyAttr(Operation& op) const
{
    int32_t nzAttr = op.GetIntAttribute(MATMUL_NZ_ATTR);
    auto mValue = (op.HasAttr(A_MUL_B_ACT_M)) ? op.GetIntAttribute(A_MUL_B_ACT_M) : 0;
    auto kValue = (op.HasAttr(A_MUL_B_ACT_K)) ? op.GetIntAttribute(A_MUL_B_ACT_K) : 0;
    auto nValue = (op.HasAttr(A_MUL_B_ACT_N)) ? op.GetIntAttribute(A_MUL_B_ACT_N) : 0;

    int aIsNz = nzAttr % 2;
    int bIsNz = (nzAttr >> 1) % 2;
    int cIsNz = (nzAttr >> 2) % 2;
    APASS_LOG_DEBUG_F(
        Elements::Operation,
        "Retrive %s[%d] attr done, aIsNz: %d, bIsNz: %d, cIsNz: %d, mValue: %ld, kValue: %ld, nValue: %ld.",
        op.GetOpcodeStr().c_str(), op.GetOpMagic(), aIsNz, bIsNz, cIsNz, static_cast<long>(mValue),
        static_cast<long>(kValue), static_cast<long>(nValue));
    for (auto& input : op.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            if (AddL1CopyInAttr(input, aIsNz, mValue, kValue, nValue) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Set Attr for matrix A L1_COPY_IN of %s[%d] failed. %s",
                                  op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            continue;
        }
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            if (AddL1CopyInAttr(input, bIsNz, mValue, kValue, nValue) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Set Attr for matrix B L1_COPY_IN of %s[%d] failed. %s",
                                  op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    for (auto& output : op.GetOOperands()) {
        if (AddL0cCopyOutAttr(output, cIsNz, mValue, nValue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Set Attr for L0C_COPY_OUT of %s[%d] failed. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status CubeProcess::CheckValidCube(const Operation& op)
{
    /* 校验有且只有一个输出 */
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has output num != 1; Please check ooperands. %s",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    /* 校验输出: 1. 非空，2. mem类型为L0C, 3.有消费者 */
    auto outputL0C = op.GetOOperands().front();
    if (outputL0C == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] output is nullptr; Please check outputL0C. %s",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (outputL0C->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] output is NOT L0C; Please check outputL0C MemoryType. %s",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (outputL0C->GetConsumers().size() < 1) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "%s[%d] output has EMPTY consumers; Please check outputL0C consumer size. %s",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status CubeProcess::UpdateL0cDtype(Operation& op)
{
    std::pair<DataType, DataType> inputDtypes = std::make_pair(DataType::DT_FP16, DataType::DT_FP16);
    for (auto& input : op.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            inputDtypes.first = input->Datatype();
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            inputDtypes.second = input->Datatype();
        }
    }
    if (supportDtypeMap.count(inputDtypes)) {
        DataType outDtype = supportDtypeMap.at(inputDtypes);
        for (auto& output : op.GetOOperands()) {
            output->tensor->datatype = outDtype;
        }
        return SUCCESS;
    } else {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "%s[%d] has unsupport input dtypes (L0A: %s, L0B: %s), update L0C dtype Failed. %s",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), DataType2String(inputDtypes.first, true),
                          DataType2String(inputDtypes.second, true), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
}

void CubeProcess::DFSSearch(Operation* op, std::vector<Operation*>& l0CCopyOuts,
                            std::unordered_set<Operation*>& visitedOp)
{
    if (op == nullptr || visitedOp.count(op)) {
        return;
    }
    visitedOp.insert(op);
    OpCalcType opCalType = OpcodeManager::Inst().GetOpCalcType(op->GetOpcode());
    if (opCalType == OpCalcType::MOVE_OUT &&
        op->GetIOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
        l0CCopyOuts.emplace_back(op);
        return;
    }
    for (auto consumerOp : op->ConsumerOps()) {
        DFSSearch(consumerOp, l0CCopyOuts, visitedOp);
    }
}

Status CubeProcess::GetL0CCopyOuts(Operation& op, std::vector<Operation*>& l0CCopyOuts)
{
    std::unordered_set<Operation*> visitedOp;
    DFSSearch(&op, l0CCopyOuts, visitedOp);
    if (l0CCopyOuts.size() == 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has no l0CCopyOuts, please check. %s", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    for (auto chainEndCopyOut : l0CCopyOuts) {
        if (chainEndCopyOut->GetOOperands().size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has more than ONE outputs. %s",
                              chainEndCopyOut->GetOpcodeStr().c_str(), chainEndCopyOut->GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto finalOutput = chainEndCopyOut->GetOOperands().front();
        if (finalOutput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR &&
            finalOutput->GetMemoryTypeOriginal() != MemoryType::MEM_L1 &&
            finalOutput->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has invalid output memType: %s. %s",
                              chainEndCopyOut->GetOpcodeStr().c_str(), chainEndCopyOut->GetOpMagic(),
                              MemoryTypeToString(finalOutput->GetMemoryTypeOriginal()).c_str(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status CubeProcess::ReconnectGraph(Operation& mulOp, std::vector<Operation*> copyOutOps)
{
    auto inputs = mulOp.GetIOperands();
    for (const auto& input : inputs) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_FIX_QUANT_PRE) {
            mulOp.EraseInput(input);
            input->RemoveConsumer(mulOp);
            for (auto copyOutOp : copyOutOps) {
                copyOutOp->iOperand.emplace_back(input);
                input->AddConsumer(copyOutOp);
            }
        }
    }
    TransferAttr(mulOp, copyOutOps);
    return SUCCESS;
}

Status CubeProcess::TransferAttr(Operation& mulOp, std::vector<Operation*> copyOutOps)
{
    for (auto copyOutOp : copyOutOps) {
        if (mulOp.HasAttr(A_MUL_B_SCALE_ATTR)) {
            copyOutOp->SetAttribute(A_MUL_B_SCALE_ATTR, mulOp.GetElementAttribute(A_MUL_B_SCALE_ATTR));
        }
        if (mulOp.HasAttr(A_MUL_B_RELU_ATTR)) {
            copyOutOp->SetAttribute(A_MUL_B_RELU_ATTR, mulOp.GetIntAttribute(A_MUL_B_RELU_ATTR));
        }
    }
    return SUCCESS;
}

Status CubeProcess::AlignGMTensor(Function& function, std::vector<Operation*>& l0CCopyOuts, Operation& mulOp)
{
    Operation* chainEndCopyOut{nullptr};
    for (auto& copyOut : l0CCopyOuts) {
        if (copyOut->GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            if (function.IsFromOutCast(copyOut->GetOOperands().front())) {
                chainEndCopyOut = copyOut;
                break;
            } else {
                chainEndCopyOut = copyOut;
            }
        }
    }
    for (auto& input : mulOp.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (function.IsFromInCast(input)) {
            APASS_LOG_WARN_F(
                Elements::Operation,
                "PreGraphProcess:CubeProcess::UpdateCubeOp::AlignGMTensor: OP_A_MUL_B iOperand tensor[%d] is incast.",
                input->GetMagic());
            continue;
        }
        if (chainEndCopyOut == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find chainEndCopyOut, AlignGMTensor failed.");
            return FAILED;
        }
        auto finalOutput = chainEndCopyOut->GetOOperands().front();
        input->tensor = finalOutput->tensor;
        for (auto& copyOut : l0CCopyOuts) {
            if (copyOut->GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                copyOut->GetOOperands().front()->tensor = finalOutput->tensor;
            }
        }
        /*
        强制要求当前Matmul链路输出的Gm仅存在一个清零的Op，暂时通过指定清零和ReduceAcc使用的vec tilesize与tileM x
        tileN相同 后续通过前端提供使能切K的API保证
        */
        AlignCopyOutAttr(input, chainEndCopyOut);
    }
    return SUCCESS;
}

Status CubeProcess::UpdateCubeOp(Function& function)
{
    if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        return SUCCESS;
    }
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_A_MUL_B && op.GetOpcode() != Opcode::OP_A_MULACC_B) {
            continue;
        }
        if (CheckValidCube(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] is invalid. %s", op.GetOpcodeStr().c_str(), op.GetOpMagic(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        // l0CCopyOuts包含所有从L0C搬出的op
        std::vector<Operation*> l0CCopyOuts{};
        if (GetL0CCopyOuts(op, l0CCopyOuts) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Get CopyOuts for %s[%d] failed. %s", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }

        // Align copy out GM with the reset GM
        if (AlignGMTensor(function, l0CCopyOuts, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "UpdateCubeOp failed at AlignGMTensor.");
            return FAILED;
        }

        if (UpdateL0cDtype(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Update L0C dtype for %s[%d] failed. %s", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (UpdateCopyAttr(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Set Attr for %s[%d] failed. %s", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }

        if (op.GetOpcode() == Opcode::OP_A_MUL_B || op.GetOpcode() == Opcode::OP_A_MULACC_B) {
            // FixPipe支持随路量化图重连 & 量化属性传递到L0C_COPY_OUT
            ReconnectGraph(op, l0CCopyOuts);
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
