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
 * \file dev_encode_function_dupped_data.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"

namespace npu::tile_fwk::dynamic {
constexpr const uint8_t MAIN_BLOCK_SIZE = 2;
std::string DevAscendFunctionDuppedData::Dump(int indent) const
{
    if (GetSource()->GetOperationSize() != GetOperationSize()) {
        DEV_ERROR(ProgEncodeErr::FUNC_OP_SIZE_MISMATCH,
                  "#ctrl.encode.func_op: GetOperationSize mismatch: source=%zu, self=%u",
                  GetSource()->GetOperationSize(), GetOperationSize());
    }
    DEV_ASSERT(ProgEncodeErr::FUNC_OP_SIZE_MISMATCH, GetSource()->GetOperationSize() == GetOperationSize());
    std::string INDENT(indent, ' ');
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');

    std::ostringstream oss;
    oss << INDENT << "DevFunctionDupped " << GetSource()->GetFuncKey() << " {\n";
    for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
        oss << INDENTINNER << "#incast:" << incastIndex << " = " << GetIncastAddress(incastIndex).Dump() << "\n";
    }
    for (size_t outcastIndex = 0; outcastIndex < GetOutcastSize(); outcastIndex++) {
        oss << INDENTINNER << "#outcast:" << outcastIndex << " = " << GetOutcastAddress(outcastIndex).Dump() << "\n";
    }
    for (size_t operationIndex = 0; operationIndex < GetOperationSize(); operationIndex++) {
        oss << INDENTINNER << "!" << operationIndex;
        oss << " #pred:" << GetSource()->GetOperationDepGraphPredCount(operationIndex);
        oss << " #succ:[";
        size_t succSize;
        auto succList = GetSource()->GetOperationDepGraphSuccAddr(operationIndex, succSize);
        for (size_t j = 0; j < succSize; j++) {
            oss << Delim(j != 0, ",") << "[" << j << "]=!" << succList[j];
        }
        oss << "]";
        oss << " #dynpred:" << GetOperationCurrPredCount(operationIndex);
        oss << " #dynsucc:" << GetOperationStitch(operationIndex).Dump();
        oss << "\n";
    }
    oss << INDENTINNER << "#expr:[";
    for (size_t exprIndex = 0; exprIndex < GetExpressionSize(); exprIndex++) {
        oss << Delim(exprIndex != 0, ",") << "[" << exprIndex << "]=" << GetExpression(exprIndex);
    }
    oss << "]";
    oss << INDENT << "}\n";
    return oss.str();
}

void DevAscendFunctionDupped::DumpTopo(std::ofstream& os, int seqNo, int funcIdx, const DevCceBinary* cceBinary,
                                       bool enableVFFusion, const DeviceTask* devTask) const
{
    auto func = GetSource();
    for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
        int cceIndex = func->GetOperationAttrCalleeIndex(opIdx);
        if (enableVFFusion) {
            cceIndex = (func->GetOperationAttrCalleeIndex(opIdx) + 1) / MAIN_BLOCK_SIZE;
        }
        auto& cceInfo = cceBinary[cceIndex];
        int32_t wrapId = -1;
        if (devTask != nullptr && devTask->mixTaskData.wrapIdNum > 0) {
            auto opWrapList = reinterpret_cast<int32_t*>(devTask->mixTaskData.opWrapList[funcIdx]);
            if (opWrapList != nullptr && opWrapList[opIdx] != -1) {
                wrapId = static_cast<int32_t>(MakeMixWrapID(funcIdx, static_cast<uint32_t>(opWrapList[opIdx])));
            }
        }
        auto& succList = func->GetOperationDepGraphSuccList(opIdx);
        os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," << func->funcKey << "," << func->rootHash << ","
           << func->GetOperationDebugOpmagic(opIdx) << "," << cceIndex << "," << cceInfo.funcHash << ","
           << cceInfo.coreType << "," << cceInfo.psgId << "," << wrapId << "," << succList.size();
        for (size_t j = 0; j < succList.size(); j++) {
            os << "," << MakeTaskID(funcIdx, func->At(succList, j));
        }
        auto& stitch = GetOperationStitch(opIdx);
        stitch.ForEach([&os](uint32_t id) { os << "," << id; });
        os << "\n";
    }
}

#if DEBUG_INFINITE_LIFETIME
void DevAscendFunctionDupped::DumpTensorAddrInfo(std::vector<std::string>& infos, uint32_t seqNo, uint32_t funcIdx)
{
    // seqNo,taskId,rawMagic,address,dtype,bytesOfDtype,(shapes,)
    auto* srcFunc = GetSource();

    auto dumpOperand = [&](const DevAscendOperationOperandInfo& operandInfo, size_t opIdx) {
        std::stringstream os;
        uint64_t rawIdx = srcFunc->GetTensor(operandInfo.tensorIndex)->rawIndex;
        auto* rawTensor = srcFunc->GetRawTensor(rawIdx);
        os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," << rawTensor->rawMagic << ","
           << GetRawTensorAddrEx(rawIdx) << "," << DataType2String(rawTensor->dataType, true) << ","
           << BytesOf(rawTensor->dataType);

        uint32_t dimSize = rawTensor->GetDim();
        os << ",(";
        bool isFirstDim = true;
        for (uint32_t i = 0; i < dimSize; i++) {
            if (isFirstDim) {
                isFirstDim = false;
            } else {
                os << ",";
            }
            os << rawTensor->shape.At(i, GetExpressionAddr());
        }
        os << ")";

        os << "\n";
        infos.emplace_back(std::move(os).str());
    };

    for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
        for (size_t iopIdx = 0; iopIdx < srcFunc->GetOperationIOperandSize(opIdx); iopIdx++) {
            auto& iopInfo = srcFunc->GetOperationIOperandInfo(opIdx, iopIdx);
            dumpOperand(iopInfo, opIdx);
        }
        for (size_t oopIdx = 0; oopIdx < srcFunc->GetOperationOOperandSize(opIdx); oopIdx++) {
            auto& oopInfo = srcFunc->GetOperationOOperandInfo(opIdx, oopIdx);
            dumpOperand(oopInfo, opIdx);
        }
    }
}
#endif // DEBUG_INFINITE_LIFETIME

static void FlushStream(std::vector<std::string>& lines, std::stringstream& oss)
{
    lines.push_back(std::move(oss).str());
    oss.clear();
    oss.str("");
}

void DevAscendFunctionDupped::DumpRawShape(const DevAscendRawTensor* rawTensor, uint32_t dimSize,
                                           std::vector<std::string>& lines, std::stringstream& oss) const
{
    oss << "        rawShape=[";
    bool isFirstDim = true;
    for (uint32_t i = 0; i < dimSize; i++) {
        if (isFirstDim) {
            isFirstDim = false;
        } else {
            oss << ", ";
        }
        oss << rawTensor->shape.At(i, GetExpressionAddr());
    }
    oss << "]";
    FlushStream(lines, oss);
}

void DevAscendFunctionDupped::DumpOperandShape(uint32_t dimSize, size_t opIdx, size_t operandIdx, bool isIn,
                                               std::vector<std::string>& lines, std::stringstream& oss) const
{
    uint64_t offset[DEV_SHAPE_DIM_MAX];
    uint64_t shape[DEV_SHAPE_DIM_MAX];
    GetFuncTensorOffsetAndShape(offset, shape, dimSize, opIdx, operandIdx, isIn);

    oss << "          offset=[";
    bool isFirstDim = true;
    for (uint32_t i = 0; i < dimSize; i++) {
        if (isFirstDim) {
            isFirstDim = false;
        } else {
            oss << ", ";
        }
        oss << offset[i];
    }
    oss << "]";
    FlushStream(lines, oss);

    oss << "           shape=[";
    isFirstDim = true;
    for (uint32_t i = 0; i < dimSize; i++) {
        if (isFirstDim) {
            isFirstDim = false;
        } else {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    FlushStream(lines, oss);
}

// Return result lines
std::vector<std::string> DevAscendFunctionDupped::DumpLeafs(uint32_t seqNo, uint32_t funcIdx) const
{
    std::vector<std::string> lines;
    std::stringstream oss;

    auto* srcFunc = GetSource();

    oss << "seqNo=" << seqNo << ", rootHash=" << srcFunc->rootHash;
    FlushStream(lines, oss);

    for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
        size_t iopNum = srcFunc->GetOperationIOperandSize(opIdx);
        size_t oopNum = srcFunc->GetOperationOOperandSize(opIdx);
        oss << "> taskId = " << MakeTaskID(funcIdx, opIdx) << ", opIdx=" << opIdx << ", #iop=" << iopNum
            << ", #oop=" << oopNum;
        FlushStream(lines, oss);

        for (size_t iopIdx = 0; iopIdx < iopNum; iopIdx++) {
            uint64_t rawIdx = srcFunc->GetOperationIOperand(opIdx, iopIdx)->rawIndex;
            auto* rawTensor = srcFunc->GetRawTensor(rawIdx);

            oss << "    iop [" << std::setw(IDENT_SIZE_THREE) << iopIdx << "]: rawMagic=" << rawTensor->rawMagic
                << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
            FlushStream(lines, oss);

            uint32_t dimSize = rawTensor->GetDim();
            DumpOperandShape(dimSize, opIdx, iopIdx, true, lines, oss);
            DumpRawShape(rawTensor, dimSize, lines, oss);
        }

        for (size_t oopIdx = 0; oopIdx < oopNum; oopIdx++) {
            uint64_t rawIdx = srcFunc->GetOperationOOperand(opIdx, oopIdx)->rawIndex;
            auto* rawTensor = srcFunc->GetRawTensor(rawIdx);

            oss << "    oop [" << std::setw(IDENT_SIZE_THREE) << oopIdx << "]: rawMagic=" << rawTensor->rawMagic
                << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
            FlushStream(lines, oss);

            uint32_t dimSize = rawTensor->GetDim();
            DumpOperandShape(dimSize, opIdx, oopIdx, false, lines, oss);
            DumpRawShape(rawTensor, dimSize, lines, oss);
        }
    }

    return lines;
}
} // namespace npu::tile_fwk::dynamic
