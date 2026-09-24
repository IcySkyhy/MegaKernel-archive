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
 * \file dump_device_memory.cpp
 * \brief
 */
#include "machine/device/dump/dump_device_memory.h"

#ifndef __DEVICE__
#include <algorithm>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include "interface/configs/config_manager.h"
#include "interface/function/rebuildable_attribute.h"
#include "machine/utils/dep_verify_dump_path.h"
#include "machine/utils/dynamic/dev_callop_attribute.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#endif

namespace npu::tile_fwk::dynamic::mem_dump {

#ifndef __DEVICE__
namespace {

void OpenIfPathChanged(std::ofstream& ofs, std::string& lastPath, const std::string& path, const char* header)
{
    if (path == lastPath) {
        return;
    }
    if (ofs.is_open()) {
        ofs.close();
    }
    lastPath = path;
    ofs.open(path);
    if (ofs.is_open()) {
        ofs << header << '\n';
    }
}

inline bool DumpEnabled()
{
    static const bool cached = (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_RUNTIME_DEBUG_VERIFY);
    return cached;
}

struct TensorSummary {
    bool valid{false};
    uint32_t seqNo{0};
    uint32_t funcIdx{0};
    int rootIndex{0};
    int rawMagic{0};
    uint64_t base{0};
    uint64_t end{0};
    uint64_t size{0};
    std::string location;
};

struct LargestRootFunctionSummary {
    bool valid{false};
    std::string name;
    uint64_t unroll{0};
    uint64_t innerSpilledRaw{0};
    uint64_t leafSpilled{0};
    uint64_t totalOutcastRaw{0};
    uint64_t staticOutcast{0};
};

struct TensorWorkspaceBreakdown {
    uint64_t totalBytes{0};
    uint64_t bytesBeforeAlignmentAndParallelism{0};
    uint64_t rootInnerSpilledBytes{0};
    uint64_t devTaskInnerExclusiveOutcastBytes{0};
    uint64_t outcastPoolBytes{0};
    uint64_t maxOutcastBytes{0};
    uint64_t maxStaticOutcastBytes{0};
    uint64_t maxDynamicAssembleOutcastBytes{0};
    uint64_t devTaskBoundaryOutcastCount{0};
    uint64_t devTaskInnerTemporalOutcastCount{0};
    uint64_t boundaryAndInnerTemporalOutcastCount{0};
    uint32_t parallelism{0};
    uint32_t slottableOutcastSlotSize{0};
    uint32_t memoryDrivenWorkspace{0};
};

struct AiCoreSpilledBreakdown {
    uint64_t totalBytes{0};
    uint64_t perCoreBytes{0};
    uint64_t aicoreCount{0};
};

struct WorkspaceBreakdown {
    uint64_t totalBytes{0};
    LargestRootFunctionSummary largestRootFunction;
    TensorWorkspaceBreakdown tensor;
    AiCoreSpilledBreakdown aicoreSpilled;
};

struct MetadataBreakdown {
    uint64_t totalBytes{0};
    uint64_t generalBytes{0};
    uint64_t dynamicCellMatchBytes{0};
    uint64_t stitchPoolBytes{0};
    uint64_t maxDynamicCellMatchTableBytes{0};
    uint64_t dynamicCellMatchSlotCount{0};
    uint32_t generalSlabSize{0};
    uint32_t stitchSlabSize{0};
};

struct MemoryOverviewState {
    bool started{false};
    uint64_t workspaceSize{0};
    uint64_t tensorWorkspaceSize{0};
    uint64_t metadataSize{0};
    uint64_t globalTensorBytes{0};
    std::set<std::pair<uint64_t, uint64_t>> globalTensorRanges;
    TensorSummary largestGlobalTensor;
    TensorSummary largestWorkspaceTensor;
    WorkspaceBreakdown workspaceBreakdown;
    MetadataBreakdown metadataBreakdown;
};

std::mutex& OverviewMutex()
{
    static std::mutex mu;
    return mu;
}

MemoryOverviewState& OverviewState()
{
    static MemoryOverviewState state;
    return state;
}

void WriteSectionHeader(std::ofstream& ofs, const char* section) { ofs << '\n' << '[' << section << "]\n"; }

void WriteTensorSummary(std::ofstream& ofs, const char* section, const TensorSummary& tensor)
{
    WriteSectionHeader(ofs, section);
    ofs << "valid=" << (tensor.valid ? 1 : 0) << '\n';
    if (!tensor.valid) {
        return;
    }
    ofs << "seqNo=" << tensor.seqNo << '\n'
        << "funcIdx=" << tensor.funcIdx << '\n'
        << "rootIndex=" << tensor.rootIndex << '\n'
        << "rawMagic=" << tensor.rawMagic << '\n'
        << "base=" << tensor.base << '\n'
        << "end=" << tensor.end << '\n'
        << "size=" << tensor.size << '\n'
        << "location=" << tensor.location << '\n';
}

void WriteOverviewLocked()
{
    const MemoryOverviewState& state = OverviewState();
    if (!state.started) {
        return;
    }

    std::ofstream ofs(GetDepVerifyDumpDir() + "/mem_overview.txt", std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        return;
    }
    ofs << "schemaVersion=2\n";

    WriteSectionHeader(ofs, "overall");
    ofs << "globalTensorBytes=" << state.globalTensorBytes << '\n'
        << "workspaceBytes=" << state.workspaceSize << '\n'
        << "tensorWorkspaceBytes=" << state.tensorWorkspaceSize << '\n'
        << "metadataBytes=" << state.metadataSize << '\n'
        << "workspaceAndMetadataBytes=" << state.workspaceSize + state.metadataSize << '\n'
        << "trackedTotalBytes=" << state.globalTensorBytes + state.workspaceSize + state.metadataSize << '\n';

    const WorkspaceBreakdown& workspace = state.workspaceBreakdown;
    WriteSectionHeader(ofs, "workspace");
    ofs << "totalBytes=" << workspace.totalBytes << '\n';

    const LargestRootFunctionSummary& largestRoot = workspace.largestRootFunction;
    WriteSectionHeader(ofs, "workspace.largestRootFunction");
    ofs << "valid=" << (largestRoot.valid ? 1 : 0) << '\n';
    if (largestRoot.valid) {
        ofs << "name=" << largestRoot.name << '\n'
            << "unroll=" << largestRoot.unroll << '\n'
            << "innerSpilledRaw=" << largestRoot.innerSpilledRaw << '\n'
            << "leafSpilled=" << largestRoot.leafSpilled << '\n'
            << "totalOutcastRaw=" << largestRoot.totalOutcastRaw << '\n'
            << "staticOutcast=" << largestRoot.staticOutcast << '\n';
    }

    const TensorWorkspaceBreakdown& tensor = workspace.tensor;
    WriteSectionHeader(ofs, "workspace.tensor");
    ofs << "totalBytes=" << tensor.totalBytes << '\n'
        << "bytesBeforeAlignmentAndParallelism=" << tensor.bytesBeforeAlignmentAndParallelism << '\n'
        << "rootInnerSpilledBytes=" << tensor.rootInnerSpilledBytes << '\n'
        << "devTaskInnerExclusiveOutcastBytes=" << tensor.devTaskInnerExclusiveOutcastBytes << '\n'
        << "outcastPoolBytes=" << tensor.outcastPoolBytes << '\n'
        << "maxOutcastBytes=" << tensor.maxOutcastBytes << '\n'
        << "maxStaticOutcastBytes=" << tensor.maxStaticOutcastBytes << '\n'
        << "maxDynamicAssembleOutcastBytes=" << tensor.maxDynamicAssembleOutcastBytes << '\n'
        << "devTaskBoundaryOutcastCount=" << tensor.devTaskBoundaryOutcastCount << '\n'
        << "devTaskInnerTemporalOutcastCount=" << tensor.devTaskInnerTemporalOutcastCount << '\n'
        << "boundaryAndInnerTemporalOutcastCount=" << tensor.boundaryAndInnerTemporalOutcastCount << '\n'
        << "parallelism=" << tensor.parallelism << '\n'
        << "slottableOutcastSlotSize=" << tensor.slottableOutcastSlotSize << '\n'
        << "memoryDrivenWorkspace=" << tensor.memoryDrivenWorkspace << '\n';

    const AiCoreSpilledBreakdown& aicoreSpilled = workspace.aicoreSpilled;
    WriteSectionHeader(ofs, "workspace.aicoreSpilled");
    ofs << "totalBytes=" << aicoreSpilled.totalBytes << '\n'
        << "perCoreBytes=" << aicoreSpilled.perCoreBytes << '\n'
        << "aicoreCount=" << aicoreSpilled.aicoreCount << '\n';

    const MetadataBreakdown& metadata = state.metadataBreakdown;
    WriteSectionHeader(ofs, "metadata");
    ofs << "totalBytes=" << metadata.totalBytes << '\n'
        << "generalBytes=" << metadata.generalBytes << '\n'
        << "dynamicCellMatchBytes=" << metadata.dynamicCellMatchBytes << '\n'
        << "stitchPoolBytes=" << metadata.stitchPoolBytes << '\n'
        << "maxDynamicCellMatchTableBytes=" << metadata.maxDynamicCellMatchTableBytes << '\n'
        << "dynamicCellMatchSlotCount=" << metadata.dynamicCellMatchSlotCount << '\n'
        << "generalSlabSize=" << metadata.generalSlabSize << '\n'
        << "stitchSlabSize=" << metadata.stitchSlabSize << '\n';

    WriteTensorSummary(ofs, "rawTensorStatistics.largestGlobalTensor", state.largestGlobalTensor);
    WriteTensorSummary(ofs, "rawTensorStatistics.largestWorkspaceTensor", state.largestWorkspaceTensor);
}

void RecordTensorOverview(uint32_t seqNo, uint32_t funcIdx, int rootIndex, const DevAscendRawTensor* rawTensor,
                          uint64_t base, uint64_t end, const char* location)
{
    if (end <= base) {
        return;
    }

    std::lock_guard<std::mutex> lock(OverviewMutex());
    MemoryOverviewState& state = OverviewState();
    if (!state.started) {
        return;
    }

    TensorSummary tensor{true, seqNo, funcIdx, rootIndex, rawTensor->rawMagic, base, end, end - base, location};
    std::pair<uint64_t, uint64_t> range{base, end};
    bool isWorkspace = rawTensor->ioProperty != DevIOProperty::ROOT_INCAST &&
                       rawTensor->ioProperty != DevIOProperty::ROOT_OUTCAST;
    if (isWorkspace) {
        if (!state.largestWorkspaceTensor.valid || tensor.size > state.largestWorkspaceTensor.size) {
            state.largestWorkspaceTensor = tensor;
        }
        return;
    }

    if (base == 0) {
        return;
    }
    if (state.globalTensorRanges.insert(range).second) {
        state.globalTensorBytes += tensor.size;
    }
    if (!state.largestGlobalTensor.valid || tensor.size > state.largestGlobalTensor.size) {
        state.largestGlobalTensor = tensor;
    }
}

void FlushOverview()
{
    std::lock_guard<std::mutex> lock(OverviewMutex());
    WriteOverviewLocked();
}

const char* LocationName(DevIOProperty ioProperty)
{
    switch (ioProperty) {
        case DevIOProperty::ROOT_INCAST:
            return "INCAST";
        case DevIOProperty::ROOT_OUTCAST:
            return "OUTCAST";
        default:
            return "LOCAL";
    }
}

void WriteU64List(std::ofstream& ofs, const uint64_t* values, int dims)
{
    ofs << '[';
    for (int i = 0; i < dims; ++i) {
        if (i != 0) {
            ofs << ';';
        }
        ofs << values[i];
    }
    ofs << ']';
}

uint64_t TensorBaseAddr(const DevAscendFunctionDupped& dup, const DevAscendRawTensor* rawTensor)
{
    if (rawTensor->ioProperty == DevIOProperty::ROOT_INCAST) {
        AddressDescriptor desc = dup.GetIncastAddress(rawTensor->ioIndex);
        return desc.IsAddress() ? desc.GetAddressValue() : 0ULL;
    }
    if (rawTensor->ioProperty == DevIOProperty::ROOT_OUTCAST) {
        AddressDescriptor desc = dup.GetOutcastAddress(rawTensor->ioIndex);
        return desc.IsAddress() ? desc.GetAddressValue() : 0ULL;
    }
    return dup.RuntimeWorkspace() + rawTensor->addrOffset;
}

void AppendWorkspaceRow(uint32_t seqNo, uint32_t funcIdx, int rootIndex, uint64_t wsBegin, uint64_t wsEnd)
{
    static std::mutex mu;
    static std::ofstream ofs;
    static std::string lastPath;
    std::lock_guard<std::mutex> lock(mu);
    OpenIfPathChanged(ofs, lastPath, GetDepVerifyDumpDir() + "/mem_workspace_range.csv",
                      "seqNo,funcIdx,rootIndex,wsBegin,wsEnd");
    if (!ofs.is_open()) {
        return;
    }
    ofs << seqNo << ',' << funcIdx << ',' << rootIndex << ',' << wsBegin << ',' << wsEnd << '\n';
}

void AppendAccessRow(uint32_t seqNo, uint32_t funcIdx, uint32_t opIdx, uint32_t taskId, int rootIndex, char accessType,
                     uint64_t base, uint64_t end, const uint64_t* offset, const uint64_t* shape,
                     const uint64_t* rawShape, int dims, const char* location, bool allConcrete, int rawMagic)
{
    static std::mutex mu;
    static std::ofstream ofs;
    static std::string lastPath;
    std::lock_guard<std::mutex> lock(mu);
    OpenIfPathChanged(
        ofs, lastPath, GetDepVerifyDumpDir() + "/mem_rawtensor_access.csv",
        "seqNo,funcIdx,opIdx,taskId,rootIndex,accessType,base,end,offset,shape,rawShape,location,allConcrete,rawMagic");
    if (!ofs.is_open()) {
        return;
    }
    ofs << seqNo << ',' << funcIdx << ',' << opIdx << ',' << taskId << ',' << rootIndex << ',' << accessType << ','
        << base << ',' << end << ',';
    WriteU64List(ofs, offset, dims);
    ofs << ',';
    WriteU64List(ofs, shape, dims);
    ofs << ',';
    WriteU64List(ofs, rawShape, dims);
    ofs << ',' << location << ',' << (allConcrete ? 1 : 0) << ',' << rawMagic << '\n';
}

void DumpOperandAccess(uint32_t seqNo, uint32_t funcIdx, uint32_t opIdx, uint32_t taskId, int rootIndex,
                       const DevAscendFunctionDupped& dup, DevAscendFunction* func, const uint64_t* exprList,
                       int operandIndex, bool isIOperand, char accessType)
{
    const DevAscendOperationOperandInfo& info = func->GetOperationOperandInfo(opIdx, operandIndex, isIOperand);
    const DevAscendTensor* operand = isIOperand ? func->GetOperationIOperand(opIdx, operandIndex) :
                                                  func->GetOperationOOperand(opIdx, operandIndex);
    int rawIndex = static_cast<int>(operand->rawIndex);
    const DevAscendRawTensor* rawTensor = func->GetRawTensor(rawIndex);
    int dims = info.GetDim();

    uint64_t offset[DEV_SHAPE_DIM_MAX] = {0};
    uint64_t shape[DEV_SHAPE_DIM_MAX] = {0};
    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {0};
    bool offsetShapeConcrete = GetTensorOffsetAndShape<false>(func, offset, shape, exprList, dims, opIdx,
                                                              info.staticOffsetAttrBeginIndex,
                                                              info.staticValidShapeAttrBeginIndex);
    bool rawShapeConcrete = GetTensorRawShape<false>(func, rawShape, exprList, dims, opIdx,
                                                     info.staticRawShapeAttrBeginIndex);

    uint64_t base = TensorBaseAddr(dup, rawTensor);
    uint64_t size = rawTensor->GetMemoryRequirement(exprList);

    AppendAccessRow(seqNo, funcIdx, opIdx, taskId, rootIndex, accessType, base, base + size, offset, shape, rawShape,
                    dims, LocationName(rawTensor->ioProperty), offsetShapeConcrete && rawShapeConcrete,
                    rawTensor->rawMagic);
    RecordTensorOverview(seqNo, funcIdx, rootIndex, rawTensor, base, base + size, LocationName(rawTensor->ioProperty));
}

} // namespace

void DumpMemoryOverview(const DevAscendProgram& devProg, Function* func, uint64_t workspaceSize)
{
    if (!DumpEnabled()) {
        return;
    }

    const auto& tensor = devProg.memBudget.tensor;
    const auto& aicoreSpilled = devProg.memBudget.aicoreSpilled;
    const auto& metadata = devProg.memBudget.metadata;
    uint64_t outcastCount = tensor.BoundaryAndInnerTemporalOutcastSlotNum();
    uint64_t maxOutcast = tensor.MaxOutcastMem();
    uint64_t tensorBeforeAlignment = tensor.rootInnerSpilledMem + tensor.devTaskInnerExclusiveOutcasts +
                                     maxOutcast * outcastCount;

    LargestRootFunctionSummary largestRootFunction;
    if (func != nullptr) {
        auto* checker = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableWorkspaceDesc>(func);
        const auto& rootList = checker->Get().rootFuncDescList;
        auto largestRoot = std::max_element(rootList.begin(), rootList.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.rootInnerSpilledRawMem < rhs.rootInnerSpilledRawMem;
        });
        if (largestRoot != rootList.end()) {
            largestRootFunction.valid = true;
            largestRootFunction.name = largestRoot->devFuncName;
            largestRootFunction.unroll = largestRoot->unroll;
            largestRootFunction.innerSpilledRaw = largestRoot->rootInnerSpilledRawMem;
            largestRootFunction.leafSpilled = largestRoot->leafPerCoreSpilledMem;
            largestRootFunction.totalOutcastRaw = largestRoot->rootTotalExclusiveOutcastRawMem;
            largestRootFunction.staticOutcast = largestRoot->rootMaxExclusiveOutcastMem;
        }
    }

    std::lock_guard<std::mutex> lock(OverviewMutex());
    MemoryOverviewState& state = OverviewState();
    state = MemoryOverviewState{};
    state.started = true;
    state.workspaceSize = workspaceSize;
    state.tensorWorkspaceSize = tensor.Total();
    state.metadataSize = metadata.Total();

    WorkspaceBreakdown& workspace = state.workspaceBreakdown;
    workspace.totalBytes = workspaceSize;
    workspace.largestRootFunction = largestRootFunction;
    workspace.tensor.totalBytes = tensor.Total();
    workspace.tensor.bytesBeforeAlignmentAndParallelism = tensorBeforeAlignment;
    workspace.tensor.rootInnerSpilledBytes = tensor.rootInnerSpilledMem;
    workspace.tensor.devTaskInnerExclusiveOutcastBytes = tensor.devTaskInnerExclusiveOutcasts;
    workspace.tensor.outcastPoolBytes = maxOutcast * outcastCount;
    workspace.tensor.maxOutcastBytes = maxOutcast;
    workspace.tensor.maxStaticOutcastBytes = tensor.maxStaticOutcastMem;
    workspace.tensor.maxDynamicAssembleOutcastBytes = tensor.maxDynamicAssembleOutcastMem;
    workspace.tensor.devTaskBoundaryOutcastCount = tensor.devTaskBoundaryOutcastNum;
    workspace.tensor.devTaskInnerTemporalOutcastCount = tensor.devTaskInnerTemporalOutcastNum;
    workspace.tensor.boundaryAndInnerTemporalOutcastCount = outcastCount;
    workspace.tensor.parallelism = tensor.parallelism;
    workspace.tensor.slottableOutcastSlotSize = tensor.slottableOutcastSlotSize;
    workspace.tensor.memoryDrivenWorkspace = tensor.memoryDrivenWorkspace;
    workspace.aicoreSpilled.totalBytes = aicoreSpilled.Total();
    workspace.aicoreSpilled.perCoreBytes = aicoreSpilled.perCoreSpilledMem;
    workspace.aicoreSpilled.aicoreCount = aicoreSpilled.aicoreCount;

    MetadataBreakdown& metadataBreakdown = state.metadataBreakdown;
    metadataBreakdown.totalBytes = metadata.Total();
    metadataBreakdown.generalBytes = metadata.general;
    metadataBreakdown.dynamicCellMatchBytes = metadata.dynamicCellMatch;
    metadataBreakdown.stitchPoolBytes = metadata.stitchPool;
    metadataBreakdown.maxDynamicCellMatchTableBytes = metadata.maxDynamicCellMatchTableMem;
    metadataBreakdown.dynamicCellMatchSlotCount = metadata.dynamicCellMatchSlotNum;
    metadataBreakdown.generalSlabSize = metadata.generalSlabSize;
    metadataBreakdown.stitchSlabSize = metadata.stitchSlabSize;
    WriteOverviewLocked();
}

void DumpRootMemory(uint32_t seqNo, uint32_t funcIdx, const DevAscendFunctionDupped& dup)
{
    if (!DumpEnabled()) {
        return;
    }
    DevAscendFunction* func = const_cast<DevAscendFunction*>(dup.GetSource());
    const uint64_t* exprList = dup.GetExpressionAddr();
    int rootIndex = func->GetRootIndex();

    uint64_t wsBegin = dup.RuntimeWorkspace();
    uint64_t wsEnd = wsBegin + func->rootInnerTensorWsMemoryRequirement;
    AppendWorkspaceRow(seqNo, funcIdx, rootIndex, wsBegin, wsEnd);

    uint32_t operationSize = dup.GetOperationSize();
    for (uint32_t opIdx = 0; opIdx < operationSize; ++opIdx) {
        uint32_t taskId = (funcIdx << 16) | (opIdx & 0xffffU);
        size_t iOperandSize = func->GetOperationIOperandSize(opIdx);
        for (size_t i = 0; i < iOperandSize; ++i) {
            DumpOperandAccess(seqNo, funcIdx, opIdx, taskId, rootIndex, dup, func, exprList, static_cast<int>(i),
                              /*isIOperand=*/true, 'R');
        }
        size_t oOperandSize = func->GetOperationOOperandSize(opIdx);
        for (size_t i = 0; i < oOperandSize; ++i) {
            DumpOperandAccess(seqNo, funcIdx, opIdx, taskId, rootIndex, dup, func, exprList, static_cast<int>(i),
                              /*isIOperand=*/false, 'W');
        }
    }
    FlushOverview();
}

#else // __DEVICE__

void DumpMemoryOverview(const DevAscendProgram&, Function*, uint64_t) {}

void DumpRootMemory(uint32_t, uint32_t, const DevAscendFunctionDupped&) {}

#endif // __DEVICE__

} // namespace npu::tile_fwk::dynamic::mem_dump
