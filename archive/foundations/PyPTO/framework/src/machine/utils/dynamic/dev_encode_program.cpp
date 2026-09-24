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
 * \file dev_encode_program.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/device/dynamic/aot_binary.h"

namespace npu::tile_fwk::dynamic {
namespace {
const size_t WIDTH = 16;
const int ADDRESS_MIN_WIDTH = 6;
} // namespace

void DevAscendProgram::ResetFromLaunch()
{
    DevArgsPreservedParams preservedParams = BackupDevArgsParams(devArgs);
    memset_s(&devArgs, sizeof(devArgs), 0, sizeof(devArgs));
    RestoreDevArgsParams(devArgs, preservedParams);

    controlFlowBinaryAddr = nullptr;
    aotPoolLastId = AOT_POOL_ENTRY_INVALID;
    runtimeDataRingBufferInited = false;
    workspaceSize = 0;
    ctrlFlowCacheAnchor = nullptr;
    RelocProgram(reinterpret_cast<int64_t>(this), 0);
}

void DevAscendProgram::DumpCce(std::ostringstream& oss, int indent) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
    oss << INDENTINNER << "#cce:" << cceCodeList.size() << "\n";
    for (size_t i = 1; i < cceCodeList.size(); i++) {
        const DevCceBinary& cceCode = At(cceCodeList, i);
        oss << INDENTINNER << "#cce-" << i << " #CoreType:" << cceCode.coreType << " #FuncHash:" << cceCode.funcHash;
        oss << "\n";
    }
}

void DevAscendProgram::DumpControlFlow(const int indent, const bool dumpAddr, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
    oss << "====\n"; // Dump control flow code (begin)

    oss << INDENTINNER << "#HostControlCodeSize:" << hostControlFlowBinary.size();
    if (dumpAddr) {
        oss << " #HostControlCodeAddr:"
            << AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(hostControlFlowBinary, 0)));
    }
    oss << "\n";

    for (size_t i = 0; i < hostControlFlowBinary.size(); i += WIDTH) {
        oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
        for (size_t off = i; off < std::min(i + WIDTH, hostControlFlowBinary.size()); off++) {
            oss << " " << DumpByte(At(hostControlFlowBinary, off));
        }
        oss << "\n";
    }

    oss << "====\n"; // Dump control flow code: ^^^ Host / Dev vvv

    oss << INDENTINNER << "#DevControlCodeSize:" << devControlFlowBinary.size();
    if (dumpAddr) {
        oss << " #DevControlCodeAddr:"
            << AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(devControlFlowBinary, 0)));
    }
    oss << "\n";

    for (size_t i = 0; i < devControlFlowBinary.size(); i += WIDTH) {
        oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
        for (size_t off = i; off < std::min(i + WIDTH, devControlFlowBinary.size()); off++) {
            oss << " " << DumpByte(At(devControlFlowBinary, off));
        }
        oss << "\n";
    }

    oss << "====\n"; // Dump control flow code (ends)
}

void DevAscendProgram::DumpExpressionTable(const int indent, const bool dumpAddr, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
    oss << INDENTINNER << "#ExprCount:" << expressionTableSize << "\n";

    oss << INDENTINNER << "#ExprCodeSize:" << expressionTableBinary.size();
    if (dumpAddr) {
        if (expressionTableBinary.size() != 0) {
            oss << " #ExprCodeAddr:"
                << AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(expressionTableBinary, 0)));
        }
    }
    oss << "\n";

    for (size_t i = 0; i < expressionTableBinary.size(); i += WIDTH) {
        oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
        for (size_t off = i; off < std::min(i + WIDTH, expressionTableBinary.size()); off++) {
            oss << " " << DumpByte(At(expressionTableBinary, off));
        }
        oss << "\n";
    }

    oss << INDENTINNER << "#func:" << devEncodeList.size() << "\n";
    for (size_t i = 0; i < devEncodeList.size(); i++) {
        const DevAscendFunction* func = reinterpret_cast<const DevAscendFunction*>(&At(At(devEncodeList, i), 0));
        oss << func->Dump(IDENT_SIZE) << "\n";
    }
}

void DevAscendProgram::DumpBasicInfo(const int indent, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    oss << INDENTINNER << "#tensorMemBudget:" << memBudget.tensor.Total() << "\n";
    oss << INDENTINNER << "#metadataMemBudget:" << memBudget.metadata.Total() << "\n";
    oss << INDENTINNER << "#deviceSchMode:" << devArgs.machineConfig << "\n";
    oss << INDENTINNER << "#stitchMaxFunctionNum:" << stitchMaxFunctionNum << "\n";
    oss << INDENTINNER << "#runtimeOutcastPoolSize:" << memBudget.tensor.runtimeOutcastPoolSize << "\n";
    oss << INDENTINNER << "#memoryDrivenWorkspace:" << memBudget.tensor.memoryDrivenWorkspace << "\n";
    oss << INDENTINNER << "#stitchFunctionsize:" << stitchFunctionsize << "\n";
    oss << INDENTINNER << "#slot{" << slotSize << "}\n";
    oss << INDENTINNER << "#assembleSlot{" << assembleSlotSize << "}\n";
}

void DevAscendProgram::DumpSymbolTable(const int indent, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    oss << INDENTINNER << "#symbolCount:" << symbolTable.size() << "\n";
    for (size_t i = 0; i < symbolTable.size(); i++) {
        const DevAscendProgramSymbol& symbol = At(symbolTable, i);
        oss << INDENTINNER << "#symbol:" << symbol.index << " = " << &At(symbol.name, 0) << "\n";
    }
}

void DevAscendProgram::DumpInputOutputSlots(const int indent, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    oss << INDENTINNER << "#inputCount:" << startArgsInputTensorSlotIndexList.size() << "\n";
    for (size_t i = 0; i < startArgsInputTensorSlotIndexList.size(); i++) {
        oss << INDENTINNER << "#input:" << i << " -> #slot:" << At(startArgsInputTensorSlotIndexList, i) << "\n";
    }
    oss << INDENTINNER << "#outputCount:" << startArgsOutputTensorSlotIndexList.size() << "\n";
    for (size_t i = 0; i < startArgsOutputTensorSlotIndexList.size(); i++) {
        oss << INDENTINNER << "#output:" << i << " <- #slot:" << At(startArgsOutputTensorSlotIndexList, i) << "\n";
    }
}

void DevAscendProgram::DumpAssembleAndInplaceSlots(const int indent, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    oss << INDENTINNER << "#assembleSlotCount:" << assembleSlotIndexList.size() << "\n";
    for (size_t i = 0; i < assembleSlotIndexList.size(); i++) {
        oss << INDENTINNER << "#assembleSlot:" << i << " -> #slot:" << At(assembleSlotIndexList, i) << "\n";
    }
    oss << INDENTINNER << "#outputInplaceSlotCount:" << outputInplaceSlotList.size() << "\n";
    for (size_t i = 0; i < outputInplaceSlotList.size(); i++) {
        oss << INDENTINNER << "#outputInplaceSlot:" << i << " -> #slot:" << At(outputInplaceSlotList, i) << "\n";
    }
}

void DevAscendProgram::DumpUpdate(const int indent, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    for (size_t i = 0; i < updateList.size(); i++) {
        auto& update = At(updateList, i);
        oss << INDENTINNER << "#slot-update-" << i << ":" << !update.Empty();
        oss << " | slotindex:" << update.slotIndex;
        if (update.stitchCtrlBitMask != STITCH_CTRL_NONE) {
            oss << " | stitchCtrlBitMask:0x" << std::hex << static_cast<unsigned>(update.stitchCtrlBitMask) << std::dec;
        }
        if (update.isPartial) {
            oss << " | isPartial:true";
        }
        if (!update.Empty()) {
            oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(update.cellMatchTableDesc)
                << " | #cellMatchStaticTable:" << update.cellMatchRuntimePartialUpdateTable.size();
        }
        oss << "\n";
    }
}

void DevAscendProgram::DumpInputSymbols(const int indent, std::ostringstream& oss) const
{
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    for (size_t i = 0; i < startArgsInputSymbolIndexList.size(); i++) {
        oss << INDENTINNER << "#symbol:" << i << " -> #symbolTable:" << At(startArgsInputSymbolIndexList, i) << "\n";
    }
}

std::string DevAscendProgram::Dump(const int indent, const bool dumpAddr) const
{
    std::ostringstream oss;
    oss << "DevProgram {\n";

    DumpBasicInfo(indent, oss);
    DumpSymbolTable(indent, oss);
    DumpInputOutputSlots(indent, oss);
    DumpAssembleAndInplaceSlots(indent, oss);
    DumpUpdate(indent, oss);
    DumpInputSymbols(indent, oss);

    DumpExpressionTable(indent, dumpAddr, oss);
    DumpControlFlow(indent, dumpAddr, oss);
    DumpCce(oss, indent);

    oss << "}";
    return oss.str();
}

void DevAscendProgram::DumpFile(const std::string& filePath) const
{
    std::ofstream ofs(filePath);
    ofs << Dump();
    ofs.close();
}

#if ENABLE_COMPILE_VERBOSE_LOG
namespace {
using RelocRange = DevAscendProgram::DevRelocRange;

std::vector<RelocRange> CollectProgramRelocRanges(const DevAscendProgram& prog)
{
    return {
        prog.symbolTable, // 0
        prog.symbolTableNameList,
        prog.expressionTableOffsetList,
        prog.hostControlFlowBinary,
        prog.devControlFlowBinary,
        prog.devEncodeList, // 5
        prog.devEncodeDataList,
        prog.cceCodeList,
        prog.aicpuLeafCodeList,
        prog.aicpuLeafCodeDataList,
        prog.startArgsInputTensorSlotIndexList, // 10
        prog.startArgsOutputTensorSlotIndexList,
        prog.assembleSlotIndexList,
        prog.outputInplaceSlotList,
        prog.updateList,
        prog.cellMatchRuntimePartialUpdateTableList, // 15
        prog.prefetchInfoList,
        prog.disableL2List,
        prog.controlFlowCache.inputTensorDataList,
        prog.controlFlowCache.outputTensorDataList,
        prog.controlFlowCache.runtimeBackup.workspace.tensorAllocators[0].slottedOutcastsBlockList, // 20
        prog.controlFlowCache.runtimeBackup.slotContext.slotList,
        prog.controlFlowCache.runtimeBackup.workspace.runtimeOutcastTensorPool,
        prog.controlFlowCache.deviceTaskCacheList,
        prog.controlFlowCache.cacheData,
    };
}

void VerifySingleRangeValid(size_t idx, const RelocRange& range)
{
    if (range.begin > range.end) {
        DEV_ERROR(ProgEncodeErr::RANGE_VERIFY_FAILED,
                  "#ctrl.program.verify: Invalid range: range[%d].begin (0x%p) > range[%d].end (0x%p)", (int)idx,
                  (void*)range.begin, (int)idx, (void*)range.end);
    }
    DEV_ASSERT_MSG(ProgEncodeErr::RANGE_VERIFY_FAILED, range.begin <= range.end, "range:%d", (int)idx);
}

void VerifyAdjacentRanges(size_t prevIdx, const RelocRange& prev, const RelocRange& curr)
{
    if (prev.end > curr.begin) {
        DEV_ERROR(ProgEncodeErr::RANGE_VERIFY_FAILED,
                  "#ctrl.program.verify: Ranges overlap: range[%d].end (0x%p) > range[%d].begin (0x%p)", (int)prevIdx,
                  (void*)prev.end, (int)(prevIdx + 1), (void*)curr.begin);
    }
    DEV_ASSERT_MSG(ProgEncodeErr::RANGE_VERIFY_FAILED, prev.end <= curr.begin, "range:%d->%d", (int)prevIdx,
                   (int)(prevIdx + 1));
    VerifySingleRangeValid(prevIdx + 1, curr);
}

void VerifyProgramDataLayout(const std::vector<RelocRange>& rangeList, const uint8_t* data, uint64_t dataSize)
{
    if ((uintptr_t)data != rangeList[0].begin) {
        DEV_ERROR(ProgEncodeErr::RANGE_VERIFY_FAILED,
                  "#ctrl.program.verify: Assertion failed: data (0x%p) != rangeList[0].begin (0x%p)", data,
                  (void*)rangeList[0].begin);
    }
    DEV_ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, (uintptr_t)data == rangeList[0].begin);
    VerifySingleRangeValid(0, rangeList[0]);
    for (size_t k = 1; k < rangeList.size(); k++) {
        VerifyAdjacentRanges(k - 1, rangeList[k - 1], rangeList[k]);
    }
    uintptr_t lastEnd = rangeList.back().end;
    uintptr_t dataEnd = (uintptr_t)(&data[dataSize]);
    if (lastEnd != dataEnd) {
        DEV_ERROR(
            ProgEncodeErr::RANGE_VERIFY_FAILED,
            "#ctrl.program.verify: Last range end does not match data end: rangeList.back().end (0x%p) != dataEnd "
            "(0x%p)",
            (void*)lastEnd, (void*)dataEnd);
    }
    DEV_ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, lastEnd == dataEnd);
}
} // namespace
#endif

void DevAscendProgram::RuntimeVerify(uintptr_t workspaceBegin, uintptr_t workspaceEnd) const
{
    (void)workspaceBegin;
    (void)workspaceEnd;
    // Verbose-only: verify encoded DevRelocVector ranges are contiguous and non-overlapping.
#if ENABLE_COMPILE_VERBOSE_LOG
    VerifyProgramDataLayout(CollectProgramRelocRanges(*this), data, dataSize);
#endif
}
} // namespace npu::tile_fwk::dynamic
