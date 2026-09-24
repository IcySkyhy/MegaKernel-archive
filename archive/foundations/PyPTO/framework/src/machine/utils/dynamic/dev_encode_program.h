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
 * \file dev_encode_program.h
 * \brief
 */

#pragma once

#include "dev_encode_program_ctrlflow_cache.h"
#include "interface/tensor/symbol_handler.h"

namespace npu::tile_fwk {
class DyndevFunctionAttribute;
class Function;
} // namespace npu::tile_fwk

namespace npu::tile_fwk::dynamic {

inline constexpr uint64_t ALIGNMENT_32K = 32 * 1024;

struct EncodeDevAscendProgramInfo;

struct DevAscendProgramSymbol {
    DevRelocVector<char> name;
    uint64_t index;
};

struct RuntimeDataRingBufferHead;
struct DevAscendProgram {
    // shadow definition in `aicore_runtime_manager.h`, make sure the first 4 members are the same
    DeviceArgs devArgs;
    uint64_t workspaceSize;
    uint64_t l2CacheOffset;
    uint64_t configKey;
    // Set at encode (EncodeDevAscendProgram::Init). Must be non-zero: AOT pool uses 0 as empty-entry sentinel.
    uint64_t hashKey;
    uint32_t slotSize;
    uint32_t assembleSlotSize;
    uint32_t ctrlBlockDim{0};
    struct {
        struct {
            // root func inner tensors
            uint64_t rootInnerSpilledMem;

            // root func outcasts & non-dassemble-dst & DeviceTask inner tensors
            uint64_t devTaskInnerExclusiveOutcasts;

            // root func outcasts & non-dassemble-dst & DeviceTask boundary outcasts:
            //      MaxOutcastMem() * devTaskBoundaryOutcastNum
            uint64_t maxStaticOutcastMem{0};
            uint64_t maxDynamicAssembleOutcastMem{0};
            uint64_t devTaskBoundaryOutcastNum{0};
            uint64_t devTaskInnerTemporalOutcastNum{0};

            uint32_t parallelism{1};
            uint32_t runtimeOutcastPoolSize{0};
            uint32_t slottableOutcastSlotSize{0};
            uint32_t memoryDrivenWorkspace{0};

            uint64_t MaxOutcastMem() const { return std::max(maxStaticOutcastMem, maxDynamicAssembleOutcastMem); }

            uint64_t BoundaryAndInnerTemporalOutcastSlotNum() const
            {
                return devTaskBoundaryOutcastNum + devTaskInnerTemporalOutcastNum;
            }

            uint64_t Total() const
            {
                uint64_t total = rootInnerSpilledMem + devTaskInnerExclusiveOutcasts +
                                 MaxOutcastMem() * BoundaryAndInnerTemporalOutcastSlotNum();
                return AlignUp(total, ALIGNMENT_32K) * parallelism;
            }
        } tensor;
        struct AiCoreLeafSpill {
            uint64_t perCoreSpilledMem;
            uint64_t aicoreCount;
            uint64_t Total() const { return perCoreSpilledMem * aicoreCount; }
        } aicoreSpilled;
        struct {
            uint64_t general;
            uint64_t dynamicCellMatch{0};
            uint64_t stitchPool;
            uint64_t maxDynamicCellMatchTableMem{0};
            uint64_t dynamicCellMatchSlotNum{0};
            uint64_t stitchCacheSize{0};
            uint32_t generalSlabSize;
            uint32_t stitchSlabSize;

            uint64_t Total() const { return general + dynamicCellMatch + stitchPool; }
        } metadata;
        struct {
            uint64_t dumpTensor;
            uint64_t leafDump;

            uint64_t Total() const { return dumpTensor + leafDump; }
        } debug;

        uint64_t Total() const { return tensor.Total() + aicoreSpilled.Total() + debug.Total(); }
    } memBudget;
    DeviceRuntimeOffset deviceRuntimeOffset;
    const void* controlFlowBinaryAddr{nullptr};
    // Last AOT pool entry id that held this program's CF binary (hint for EnsureCached).
    uint8_t aotPoolLastId{0xFF};
    std::atomic<bool> runtimeDataRingBufferInited{false};
    uint32_t stitchFunctionsize{0};
    uint32_t stitchMaxFunctionNum{0};
    uint32_t ctrlFlowCacheSize{0};
    uint32_t disableCtrlFlowCache{0};
    uint32_t rootFuncMaxCallOpsize{0};
    uint32_t cellMatchTagSeq_{0};
    // Seed to UINT32_MAX so the first AdvanceStitchCacheEpoch wraps packed epoch to 0 and memset's virgin cache.
    // Also advanced when packed taskId (low 16 bits) wraps within a launch.
    uint32_t stitchCacheEpoch_{0xFFFFFFFF};
    DevRelocVector<DevAscendProgramSymbol> symbolTable;
    DevRelocVector<char> symbolTableNameList;
    uint64_t expressionTableSize;
    DevRelocVector<uint64_t> expressionTableOffsetList;
    DevRelocVector<uint8_t> preGuardPage;
    DevRelocVector<uint8_t> expressionTableBinary;
    DevRelocVector<uint8_t> hostControlFlowBinary; // compiled by system gcc (host arch)
    DevRelocVector<uint8_t> devControlFlowBinary;  // compiled by CANN gcc (ARM arch)
    DevRelocVector<uint8_t> postGuardPage;
    DevRelocVector<DevRelocVector<uint8_t>> devEncodeList;
    DevRelocVector<uint8_t> devEncodeDataList;
    DevRelocVector<DevCceBinary> cceCodeList;
    DevRelocVector<DevAicpuLeafBinary> aicpuLeafCodeList;
    DevRelocVector<int32_t> aicpuLeafCodeDataList;
    DevRelocVector<uint64_t> startArgsInputTensorSlotIndexList;
    DevRelocVector<uint64_t> startArgsOutputTensorSlotIndexList;
    DevRelocVector<uint64_t> startArgsInputSymbolIndexList;
    DevRelocVector<uint64_t> assembleSlotIndexList;
    DevRelocVector<uint64_t> outputInplaceSlotList;
    DevRelocVector<DevAscendProgramUpdate> updateList;
    DevRelocVector<uint64_t> cellMatchRuntimePartialUpdateTableList;
    DevRelocVector<PrefetchInfo> prefetchInfoList;
    DevRelocVector<uint8_t> disableL2List;
    DevControlFlowCache* ctrlFlowCacheAnchor{nullptr};
    DevControlFlowCache controlFlowCache;
#define programLastField controlFlowCache.cacheData
    uint64_t dataSize;
    uint8_t data[0];

    /*
     *      DevAscendProgramSymbol symbolTableData[]
     *      char symbolTableNameListData[]
     *      uint64_t expressionTableOffsetListData[]
     *      uint8_t preGuardPageData[PAGE_SIZE]
     *      uint8_t expressionTableBinaryData[]
     *      uint8_t hostControlFlowBinaryData[]
     *      uint8_t devControlFlowBinaryData[]
     *      DevRelocVector<uint8_t> devEncodeList[]
     *      uint8_t devEncodeDataList[]
     *      DevRelocVector<uint8_t> cceCodeList[]
     *      uint64_t startArgsInputTensorSlotIndexListData[]
     *      uint64_t startArgsOutputTensorSlotIndexListData[]
     *      uint64_t startArgsInputSymbolIndexListData[]
     *      uint64_t assembleSlotIndexList[]
     *      uint64_t outputInplaceSlotList[];
     *      DevAscendProgramUpdate updateList[]
     *      DevAscendProgramSlot slotList[]
     */

    RuntimeDataRingBufferHead* GetRuntimeDataList()
    {
        return reinterpret_cast<RuntimeDataRingBufferHead*>(devArgs.runtimeDataRingBufferAddr);
    }

    const RuntimeDataRingBufferHead* GetRuntimeDataList() const
    {
        return reinterpret_cast<const RuntimeDataRingBufferHead*>(devArgs.runtimeDataRingBufferAddr);
    }

    uint32_t GetCellMatchTagSeq() const { return cellMatchTagSeq_; }

    void SetCellMatchTagSeq(uint32_t value) { cellMatchTagSeq_ = value; }

    void IncrementCellMatchTagSeq() { cellMatchTagSeq_++; }

    template <typename T>
    const T& At(const DevRelocVector<T>& localvec, int index) const
    {
        return localvec[index];
    }
    template <typename T>
    T& At(DevRelocVector<T>& localvec, int index)
    {
        return localvec[index];
    }

    void DumpCce(std::ostringstream& oss, int indent) const;

    void DumpControlFlow(const int indent, const bool dumpAddr, std::ostringstream& oss) const;

    void DumpExpressionTable(const int indent, const bool dumpAddr, std::ostringstream& oss) const;

    void DumpBasicInfo(const int indent, std::ostringstream& oss) const;

    void DumpSymbolTable(const int indent, std::ostringstream& oss) const;

    void DumpInputOutputSlots(const int indent, std::ostringstream& oss) const;

    void DumpAssembleAndInplaceSlots(const int indent, std::ostringstream& oss) const;

    void DumpUpdate(const int indent, std::ostringstream& oss) const;

    void DumpInputSymbols(const int indent, std::ostringstream& oss) const;

    std::string Dump(const int indent = 0, const bool dumpAddr = false) const;

    void DumpFile(const std::string& filePath) const;

    std::vector<int> GetInputTensorSlotIndexList() const
    {
        std::vector<int> indexList;
        for (size_t i = 0; i < startArgsInputTensorSlotIndexList.size(); i++) {
            indexList.push_back(At(startArgsInputTensorSlotIndexList, i));
        }
        return indexList;
    }
    std::vector<int> GetOutputTensorSlotIndexList() const
    {
        std::vector<int> indexList;
        for (size_t i = 0; i < startArgsOutputTensorSlotIndexList.size(); i++) {
            indexList.push_back(At(startArgsOutputTensorSlotIndexList, i));
        }
        return indexList;
    }

    std::vector<int> GetAssembleTensorSlotIndexList() const
    {
        std::vector<int> indexList;
        for (size_t i = 0; i < assembleSlotIndexList.size(); i++) {
            indexList.push_back(At(assembleSlotIndexList, i));
        }
        return indexList;
    }

    std::vector<int> GetPartialUpdateTensorSlotIndexList() const
    {
        const int& front = At(assembleSlotIndexList, 0);
        const int& back = At(assembleSlotIndexList, assembleSlotIndexList.size() - 1);
        std::vector<int> slotIndexList(&front, &back + 1);
        return slotIndexList;
    }

    std::tuple<const void*, uint64_t> GetDevControlFlowBinary() const
    {
        return std::make_tuple(reinterpret_cast<const void*>(devControlFlowBinary.Data()),
                               (uint64_t)devControlFlowBinary.size());
    }

    std::tuple<const void*, uint64_t> GetHostControlFlowBinary() const
    {
        return std::make_tuple(reinterpret_cast<const void*>(hostControlFlowBinary.Data()),
                               (uint64_t)hostControlFlowBinary.size());
    }

    std::tuple<const void*, uint64_t, const uint64_t*, uint64_t> GetExpressionTableBinary() const
    {
        return std::make_tuple(reinterpret_cast<const void*>(expressionTableBinary.Data()),
                               static_cast<uint64_t>(expressionTableBinary.size()), expressionTableOffsetList.Data(),
                               static_cast<uint64_t>(expressionTableOffsetList.size()));
    }

    uint64_t GetSymbolTableSize() const { return symbolTable.size(); }

    uint64_t GetExpressionTableSize() const { return expressionTableSize; }

    uint64_t GetFunctionSize() const { return devEncodeList.size(); }

    DevAscendFunction* GetFunction(int index) const
    {
        return reinterpret_cast<DevAscendFunction*>(const_cast<uint8_t*>(devEncodeList[index].Data()));
    }

    DevAscendFunction* GetFunctionByRawName(const std::string& rawName) const
    {
        for (size_t i = 0; i < GetFunctionSize(); i++) {
            DevAscendFunction* func = GetFunction(static_cast<int>(i));
            if (func->GetRawName() == rawName) {
                return func;
            }
        }
        return nullptr;
    }

    const DevCceBinary* GetCceBinary(int index) const { return &cceCodeList[index]; }
    const DevAicpuLeafBinary* GetAicpuLeafBinary(int index) const { return &aicpuLeafCodeList[index]; }

    DevControlFlowCache* GetControlFlowCache() { return ctrlFlowCacheAnchor; }

    template <typename Ty>
    typename Ty::ElementType* RelocOffset(intptr_t shift, void*& offset, Ty& list)
    {
        typename Ty::ElementType* ptr = reinterpret_cast<typename Ty::ElementType*>(offset);
        offset = (void*)((uintptr_t)(offset) + list.ElementSize() * list.size());
        list.DeviceRelocData(shift);
        return ptr;
    }

    void RelocProgram(uint64_t srcProgram, uint64_t dstProgram, bool relocFunc = false)
    {
        intptr_t shift = static_cast<int64_t>(dstProgram) - static_cast<int64_t>(srcProgram);
        void* offset = data;

        auto symbolTablePtr = RelocOffset(shift, offset, symbolTable);
        for (size_t i = 0; i < symbolTable.size(); i++) {
            symbolTablePtr[i].name.DeviceRelocData(shift);
        }

        RelocOffset(shift, offset, symbolTableNameList);
        RelocOffset(shift, offset, expressionTableOffsetList);
        RelocOffset(shift, offset, preGuardPage);
        RelocOffset(shift, offset, expressionTableBinary);
        RelocOffset(shift, offset, hostControlFlowBinary);
        RelocOffset(shift, offset, devControlFlowBinary);

        auto devEncodeListPtr = RelocOffset(shift, offset, devEncodeList);
        for (size_t i = 0; i < devEncodeList.size(); i++) {
            devEncodeListPtr[i].DeviceRelocData(shift);
        }
        RelocOffset(shift, offset, devEncodeDataList);
        RelocOffset(shift, offset, cceCodeList);
        auto aicpuLeafCodeListPtr = RelocOffset(shift, offset, aicpuLeafCodeList);
        for (size_t i = 0; i < aicpuLeafCodeList.size(); i++) {
            aicpuLeafCodeListPtr[i].aicpuLeafCode.DeviceRelocData(shift);
        }
        RelocOffset(shift, offset, aicpuLeafCodeDataList);

        RelocOffset(shift, offset, startArgsInputTensorSlotIndexList);
        RelocOffset(shift, offset, startArgsOutputTensorSlotIndexList);
        RelocOffset(shift, offset, startArgsInputSymbolIndexList);
        RelocOffset(shift, offset, assembleSlotIndexList);
        RelocOffset(shift, offset, outputInplaceSlotList);
        auto updateListPtr = RelocOffset(shift, offset, updateList);
        for (size_t i = 0; i < updateList.size(); i++) {
            updateListPtr[i].cellMatchRuntimePartialUpdateTable.DeviceRelocDataMaybeNull(shift);
        }
        RelocOffset(shift, offset, cellMatchRuntimePartialUpdateTableList);

        RelocOffset(shift, offset, prefetchInfoList);
        RelocOffset(shift, offset, disableL2List);
        if (relocFunc) {
            for (int i = 0; i < static_cast<int>(GetFunctionSize()); i++) {
                DevAscendFunction* func = GetFunction(i);
                func->Reloc(reinterpret_cast<uint64_t>(func), true);
            }
        }

        RelocOffset(shift, offset, controlFlowCache.inputTensorDataList);
        RelocOffset(shift, offset, controlFlowCache.outputTensorDataList);
        for (uint32_t i = 0; i < SCH_DEVTASK_MAX_PARALLELISM; i++) {
            RelocOffset(shift, offset,
                        controlFlowCache.runtimeBackup.workspace.tensorAllocators[i].slottedOutcastsBlockList);
        }
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.slotContext.slotList);
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.workspace.runtimeOutcastTensorPool);
        RelocOffset(shift, offset, controlFlowCache.deviceTaskCacheList);
        RelocOffset(shift, offset, controlFlowCache.cacheData);
    }

    struct DevArgsPreservedParams {
        uint32_t nrAic;
        uint32_t nrAiv;
        uint32_t nrAicpu;
        uint32_t nrValidAic;
        uint32_t scheCpuNum;
        uint32_t die0MaxCpuid;
        uint32_t launchSchedAicpuNum;
        ArchInfo archInfo;
        uint64_t dynamicCellMatchAddr;
        uint64_t dynamicCellMatchCapacity;
        bool hasAicpuTask;
        bool launchSchedSameCluster;
    };

    DevArgsPreservedParams BackupDevArgsParams(const DeviceArgs& src)
    {
        DevArgsPreservedParams params;
        params.nrAic = src.nrAic;
        params.nrAiv = src.nrAiv;
        params.nrAicpu = src.nrAicpu;
        params.nrValidAic = src.nrValidAic;
        params.scheCpuNum = src.scheCpuNum;
        params.die0MaxCpuid = src.die0MaxCpuid;
        params.launchSchedAicpuNum = src.launchSchedAicpuNum;
        params.archInfo = src.archInfo;
        params.dynamicCellMatchAddr = src.dynamicCellMatchAddr;
        params.dynamicCellMatchCapacity = src.dynamicCellMatchCapacity;
        params.hasAicpuTask = src.hasAicpuTask;
        params.launchSchedSameCluster = src.launchSchedSameCluster;
        return params;
    }

    void RestoreDevArgsParams(DeviceArgs& dst, const DevArgsPreservedParams& params)
    {
        dst.nrAic = params.nrAic;
        dst.nrAiv = params.nrAiv;
        dst.nrAicpu = params.nrAicpu;
        dst.nrValidAic = params.nrValidAic;
        dst.scheCpuNum = params.scheCpuNum;
        dst.die0MaxCpuid = params.die0MaxCpuid;
        dst.launchSchedAicpuNum = params.launchSchedAicpuNum;
        dst.archInfo = params.archInfo;
        dst.dynamicCellMatchAddr = params.dynamicCellMatchAddr;
        dst.dynamicCellMatchCapacity = params.dynamicCellMatchCapacity;
        dst.hasAicpuTask = params.hasAicpuTask;
        dst.launchSchedSameCluster = params.launchSchedSameCluster;
    }

    void ResetFromLaunch();

    void ResetRerun()
    {
        uint64_t* RuntimePartialUpdateTable = cellMatchRuntimePartialUpdateTableList.Data();
        uint64_t RuntimePartialUpdateTableSize = cellMatchRuntimePartialUpdateTableList.DataSize();
        // Need set to AICORE_TASK_INIT
        memset_s(RuntimePartialUpdateTable, RuntimePartialUpdateTableSize, 0xFF, RuntimePartialUpdateTableSize);
    }

    struct DevRelocRange {
        template <typename T>
        DevRelocRange(const DevRelocVector<T>& v)
            : begin(reinterpret_cast<uintptr_t>(v.begin())), end(reinterpret_cast<uintptr_t>(v.end()))
        {}

        uintptr_t begin;
        uintptr_t end;
    };

    void RuntimeVerify(uintptr_t workspaceBegin, uintptr_t workspaceEnd) const;

    uint64_t GetSize() const
    {
        return reinterpret_cast<uintptr_t>(programLastField.End()) - reinterpret_cast<uintptr_t>(this);
    }

    const DeviceRuntimeOffset& GetDeviceRuntimeOffset() const { return deviceRuntimeOffset; }

    void SetParallelism(uint32_t parallelism) { memBudget.tensor.parallelism = parallelism; }
    uint32_t GetParallelism() { return memBudget.tensor.parallelism; }

    // Live boundary-outcast block count for ctrl-flow cache backup; falls back to outcastCacheDepthFallback.
    uint64_t GetCtrlFlowCacheSlottedOutcastBlockCount(uint64_t totalSlot, uint32_t outcastCacheDepthFallback = 0) const;

private:
    friend struct EncodeDevAscendProgramInfo;
    friend void EncodeDevAscendProgram(Function* func, uint64_t& offset, DevAscendProgram* base,
                                       const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap);
    friend void EncodeDevAscendProgramSizeOnly(uint64_t& offset, EncodeDevAscendProgramInfo& encodeInfo,
                                               const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap);
    friend void EncodeDevAscendProgramFull(Function* func, DevAscendProgram* base, uint64_t& offset,
                                           EncodeDevAscendProgramInfo& encodeInfo,
                                           const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap);

    void InitSymbolTable(uintdevptr_t& initOffset, SymbolicSymbolTable* symbolTableInput, bool fillContent);
    void InitExpressionTableBinary(uintdevptr_t& initOffset,
                                   const std::vector<std::vector<uint8_t>>& expressionTableBinaryListInput,
                                   bool fillContent);
    void InitControlFlowBinary(uintdevptr_t& initOffset, const std::vector<uint8_t>& hostControlFlowBinaryInput,
                               const std::vector<uint8_t>& devControlFlowBinaryInput, bool fillContent);
    void InitDevEncodeList(uintdevptr_t& initOffset, const std::vector<std::vector<uint8_t>>& devEncodeListInput,
                           bool fillContent);
    void InitCceCodeList(uintdevptr_t& initOffset, const std::vector<CceCodeInfo>& cceInfo, bool fillContent);
    void InitPrefetchInfoList(uintdevptr_t& initOffset, const std::vector<L2Info>& l2InfoList, bool fillContent);
    void InitDisableL2List(uintdevptr_t& initOffset, const std::vector<uint8_t>& disableL2, bool fillContent);
    void InitStartArgsABIParamList(uintdevptr_t& initOffset, const std::vector<int>& tStartArgsInputTensorSlotIndexList,
                                   const std::vector<int>& tStartArgsOutputTensorSlotIndexList,
                                   const std::vector<int>& tStartArgsInputSymbolIndexList,
                                   const std::vector<int>& tAsembleSlotIndexList,
                                   const std::vector<int>& tInplaceSlotIndexList, bool fillContent);
    void InitUpdateSlot(uintdevptr_t& initOffset, const std::vector<std::vector<uint8_t>>& devEncodeListInput,
                        const std::unordered_map<Function*, int>& rootFuncKeyDict,
                        const std::unordered_map<int, std::unordered_map<Function*, int>>& slotRootOutcastDict,
                        const std::vector<int>& tPartialUpdateSlotIndexList, bool fillContent,
                        const std::unordered_map<int, SlotMaskEntry>* stitchUpdateSlotMaskMap);
    void InitControlFlowCache(uintdevptr_t& initOffset, const std::shared_ptr<DyndevFunctionAttribute>& dyndevAttr,
                              bool fillContent, uint32_t outcastCacheDepthFallback = 0);
};

#include <cstddef>
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "machine/utils/device_log.h"

inline void FillRuntimeDynamicCellMatchPool(DevAscendProgram* devProg)
{
    if (devProg == nullptr) {
        return;
    }
    const uint64_t dynCmCap = devProg->devArgs.dynamicCellMatchCapacity;
    const uint64_t dynCmAddrU64 = devProg->devArgs.dynamicCellMatchAddr;
    if (dynCmAddrU64 == 0 || dynCmCap == 0) {
        return;
    }
    DEV_ASSERT_MSG(DevCommonErr::PARAM_INVALID, (dynCmCap % sizeof(uint64_t)) == 0,
                   "#ctrl.cellmatch.reset: dynamicCellMatch cap not uint64 aligned, cap=%lu", dynCmCap);
    const size_t numWords = static_cast<size_t>(dynCmCap / sizeof(uint64_t));
    auto* table = reinterpret_cast<uint64_t*>(dynCmAddrU64);
    for (size_t i = 0; i < numWords; ++i) {
        table[i] = AICORE_TASK_INIT;
    }
}

inline void AdvanceCellMatchTagSeq(DevAscendProgram* devProg)
{
    if (devProg->GetCellMatchTagSeq() + 1 >= CELL_MATCH_TAG_SEQ_MAX) {
        DEV_INFO("#ctrl.cellmatch.tag: cellMatchTagSeq overflow, reset dynamicCellMatch pool");
        FillRuntimeDynamicCellMatchPool(devProg);
        devProg->SetCellMatchTagSeq(0);
        return;
    }
    devProg->IncrementCellMatchTagSeq();
}

} // namespace npu::tile_fwk::dynamic
