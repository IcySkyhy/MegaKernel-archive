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
 * \file dev_encode_function_dupped_data.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "machine/utils/dynamic/allocator/allocators.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::dynamic {
constexpr int ARG_ATTR_TYPE = 4;
const uint32_t RAW_TENSOR_OFFSET_SIZE = 63;
const uint32_t RAW_TENSOR_DESC_PRE_SIZE = 8;

struct DevAscendFunctionDuppedData {
    DevAscendFunction* source_;
    DevAscendFunctionDuppedOperation operationList_;
    DevAscendFunctionDuppedVector incastList_;
    DevAscendFunctionDuppedVector outcastList_;
    DevAscendFunctionDuppedVector expressionList_;
    uintdevptr_t runtimeWorkspace_;
    RuntimeReuseInfo runtimeWsReuseInfo_;
    uintdevptr_t runtimeOutcastWorkspace_;
    intdevptr_t loopDieId_{-1};
    uint8_t data_[0];
    /*
     *  Duplicated:
     *      predcount_t                                         predCountListData[];
     *  Allocated (& zero-ed):
     *      AddressDescriptor                                   incastAddressListData[];
     *      AddressDescriptor                                   outcastAddressListData[];
     *      uint64_t                                            expressionListData[];
     *      DevAscendFunctionDuppedStitchList                   stitchListData[];
     */
#define GET_DATA(type, data, base, index) ((reinterpret_cast<type*>(const_cast<uint8_t*>((data) + (base)))[index]))
    uint32_t GetOperationSize() const { return operationList_.size; }
    const predcount_t& GetOperationCurrPredCount(int index) const
    {
        return GET_DATA(predcount_t, data_, operationList_.predCountBase, index);
    }
    predcount_t& GetOperationCurrPredCount(int index)
    {
        return GET_DATA(predcount_t, data_, operationList_.predCountBase, index);
    }

    uint32_t GetStitchSize() const { return operationList_.stitchCount; }
    const DevAscendFunctionDuppedStitchList& GetStitch(int index) const
    {
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index);
    }
    DevAscendFunctionDuppedStitchList& GetStitch(int index)
    {
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index);
    }

    uint64_t GetExpressionSize() const { return expressionList_.size; }
    const uint64_t& GetExpression(int index) const { return GET_DATA(uint64_t, data_, expressionList_.base, index); }
    uint64_t& GetExpression(int index) { return GET_DATA(uint64_t, data_, expressionList_.base, index); }

    uint64_t* GetExpressionAddr() const { return &GET_DATA(uint64_t, data_, expressionList_.base, 0); }

    uint64_t GetIncastSize() const { return incastList_.size; }
    AddressDescriptor GetIncastAddress(int index) const
    {
        return GET_DATA(AddressDescriptor, data_, incastList_.base, index);
    }
    AddressDescriptor& GetIncastAddress(int index)
    {
        return GET_DATA(AddressDescriptor, data_, incastList_.base, index);
    }

    uint64_t GetOutcastSize() const { return outcastList_.size; }
    AddressDescriptor GetOutcastAddress(int index) const
    {
        return GET_DATA(AddressDescriptor, data_, outcastList_.base, index);
    }
    AddressDescriptor& GetOutcastAddress(int index)
    {
        return GET_DATA(AddressDescriptor, data_, outcastList_.base, index);
    }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return runtimeWsReuseInfo_; }
    RuntimeReuseInfo& GetRuntimeReuseInfo() { return runtimeWsReuseInfo_; }

    uintdevptr_t GetRuntimeWorkspace() const { return runtimeWorkspace_; }
    uintdevptr_t& GetRuntimeWorkspace() { return runtimeWorkspace_; }

    uintdevptr_t GetRuntimeOutcastWorkspace() const { return runtimeOutcastWorkspace_; }
    uintdevptr_t& GetRuntimeOutcastWorkspace() { return runtimeOutcastWorkspace_; }

    DevAscendFunction* GetSource() const { return source_; }
    DevAscendFunction*& GetSource() { return source_; }

    const DevAscendFunctionDuppedStitchList& GetOperationStitch(int operationIndex, bool maybeNull = true) const
    {
        int stitchIndex = GetSource()->GetOperationStitchIndex(operationIndex);

        DEV_IF_NONDEVICE
        {
            if (!maybeNull && stitchIndex == 0) {
                DEV_ERROR(
                    ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                    "#ctrl.encode.stitch.outcast: GetOperationStitch: operationIndex=%d has invalid stitch index 0",
                    operationIndex);
            }
            DEV_ASSERT(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, maybeNull || stitchIndex != 0);
        }

        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, stitchIndex);
    }
    DevAscendFunctionDuppedStitchList& GetOperationStitch(int operationIndex, bool maybeNull = true)
    {
        int stitchIndex = GetSource()->GetOperationStitchIndex(operationIndex);

        DEV_IF_NONDEVICE
        {
            if (!maybeNull && stitchIndex == 0) {
                DEV_ERROR(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                          "#ctrl.encode.stitch.outcast: GetOperationStitch: operationIndex=%d has invalid stitch "
                          "index 0",
                          operationIndex);
            }
            DEV_ASSERT(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, maybeNull || stitchIndex != 0);
        }

        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, stitchIndex);
    }

    inline uint64_t GetIncastDataSize(int incastIndex) const
    {
        auto rawTensor = GetSource()->GetIncastRawTensor(incastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    inline uint64_t GetOutcastDataSize(int outcastIndex) const
    {
        auto rawTensor = GetSource()->GetOutcastRawTensor(outcastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    inline uint64_t GetRawTensorDataSize(int rawIndex)
    {
        auto rawTensor = GetSource()->GetRawTensor(rawIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    schema::range SchemaGetIncastRange(int arg) const
    {
        auto base = GetIncastAddress(arg).GetAddress();
        auto size = GetIncastDataSize(arg);
        return schema::Range(base, base + size);
    }
    schema::range SchemaGetOutcastRange(int arg) const
    {
        auto base = GetOutcastAddress(arg).GetAddress();
        auto size = GetOutcastDataSize(arg);
        return schema::Range(base, base + size);
    }

    schema::RActWorkspace SchemaGetWorkspace() const
    {
        auto workspaceBegin = GetRuntimeWorkspace();
        auto workspaceEnd = GetRuntimeWorkspace() + GetSource()->rootInnerTensorWsMemoryRequirement;
        return schema::RActWorkspace(schema::Range(workspaceBegin, workspaceEnd));
    }

    schema::ExpressionTable SchemaGetExpressionList() const
    {
        size_t expressionSize = GetExpressionSize();
        std::vector<schema::Int64Type> expressionList;
        for (size_t i = 0; i < expressionSize; i++) {
            expressionList.push_back(schema::Int64Type(GetExpression(i)));
        }
        return schema::ExpressionTable(expressionList);
    }

    // Initialize runtime tail fields that DuplicateRoot no longer zeroes via memset.
    // incast/outcast addresses are assigned in TryAllocateFunctionMemory;
    // expression table is filled by AOT SetExprBatch before stitch.
    void InitDuppedRuntimeTail()
    {
        for (uint32_t i = 0; i < operationList_.stitchCount; ++i) {
            GetStitch(i).Head() = nullptr;
        }
    }

    std::string Dump(int indent = 0) const;
};

struct DevAscendFunctionDupped {
    DevAscendFunctionDupped() = default;
    explicit DevAscendFunctionDupped(WsAllocation tinyAlloc) : dupTiny_(tinyAlloc) {}

    static DevAscendFunctionDupped DuplicateRoot(DevAscendFunction* func, WsAllocation tinyAlloc)
    {
        DevAscendFunctionDuppedData* dupData = tinyAlloc.As<DevAscendFunctionDuppedData>();
        DevAscendFunctionDuppedData* sourceData = func->GetDuppedData();
        (void)memcpy_s(reinterpret_cast<uint8_t*>(dupData), func->GetDuppedDataCopySize(), sourceData,
                       func->GetDuppedDataCopySize());
        dupData->GetSource() = func;

        DevAscendFunctionDupped dup(tinyAlloc);
        return dup;
    }

    void ReleaseDuppedMemory(WsMetadataAllocator& allocator) { (void)allocator; }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return DupData()->GetRuntimeReuseInfo(); }
    RuntimeReuseInfo& GetRuntimeReuseInfo() { return DupData()->GetRuntimeReuseInfo(); }

    uintdevptr_t RuntimeWorkspace() const { return DupData()->GetRuntimeWorkspace(); }
    uintdevptr_t& RuntimeWorkspace() { return DupData()->GetRuntimeWorkspace(); }

    uintdevptr_t RuntimeOutcastBase() const { return DupData()->GetRuntimeOutcastWorkspace(); }
    uintdevptr_t& RuntimeOutcastBase() { return DupData()->GetRuntimeOutcastWorkspace(); }

    const DevAscendFunction* GetSource() const { return DupData()->GetSource(); }
    DevAscendFunction* GetSource() { return DupData()->GetSource(); }

    inline const uint64_t& GetExpression(int arg) const { return DupData()->GetExpression(arg); };
    inline uint64_t& GetExpression(int arg) { return DupData()->GetExpression(arg); };
    inline uint64_t GetExpressionSize() const { return DupData()->GetExpressionSize(); }
    inline uint64_t* GetExpressionAddr() const { return DupData()->GetExpressionAddr(); }

    inline auto GetOperationSize() const { return DupData()->GetOperationSize(); }
    inline const predcount_t& GetOperationCurrPredCount(int arg) const
    {
        return DupData()->GetOperationCurrPredCount(arg);
    };
    inline predcount_t& GetOperationCurrPredCount(int arg) { return DupData()->GetOperationCurrPredCount(arg); };
    inline const auto& GetOperationStitch(int arg, bool maybeNull = true) const
    {
        return DupData()->GetOperationStitch(arg, maybeNull);
    };
    inline auto& GetOperationStitch(int arg, bool maybeNull = true)
    {
        return DupData()->GetOperationStitch(arg, maybeNull);
    };

    inline AddressDescriptor GetIncastAddress(int arg) const { return DupData()->GetIncastAddress(arg); };
    inline AddressDescriptor& GetIncastAddress(int arg) { return DupData()->GetIncastAddress(arg); };

    inline AddressDescriptor GetOutcastAddress(int arg) const { return DupData()->GetOutcastAddress(arg); };
    inline AddressDescriptor& GetOutcastAddress(int arg) { return DupData()->GetOutcastAddress(arg); };

    schema::expr SchemaGetExpressionTable() const
    {
        std::vector<schema::Int64Type> exprTable;
        uint64_t* exprAddr = GetExpressionAddr();
        uint64_t exprSize = GetExpressionSize();
        for (uint64_t i = 0; i < exprSize; i++) {
            exprTable.push_back(exprAddr[i]);
        }
        return schema::expr(exprTable);
    }

    inline uintdevptr_t GetRawTensorAddr(int rawIndex) const
    {
        uintdevptr_t addr = 0ULL;
        const DevAscendRawTensor* rawTensor = GetSource()->GetRawTensor(rawIndex);
        if (rawTensor->ioProperty == DevIOProperty::ROOT_INCAST) {
            AddressDescriptor incast = GetIncastAddress(rawTensor->ioIndex);
            DEV_ASSERT_MSG(TensorMetaErr::INCAST_ADDRESS_NULL, !incast.IsNullAddress(),
                           "Null incast: root [%s], rawIndex [%d], ioIndex [%d]", GetSource()->GetRawName(), rawIndex,
                           rawTensor->ioIndex);
            addr = incast.addr;
        } else if (rawTensor->ioProperty == DevIOProperty::ROOT_OUTCAST) {
            AddressDescriptor outcast = GetOutcastAddress(rawTensor->ioIndex);
            DEV_ASSERT_MSG(TensorMetaErr::OUTCAST_ADDRESS_NULL, !outcast.IsNullAddress(),
                           "Null outcast: root [%s], rawIndex [%d], ioIndex [%d]", GetSource()->GetRawName(), rawIndex,
                           rawTensor->ioIndex);
            addr = outcast.addr;
        } else {
            uintdevptr_t runtimeWorkspace = RuntimeWorkspace();
            DEV_ASSERT_MSG(TensorMetaErr::RUNTIME_WORKSPACE_NULL, runtimeWorkspace != 0,
                           "Trying to access inner tensor addr with zero runtime workspace: root [%s], rawIndex [%d]",
                           GetSource()->GetRawName(), rawIndex);
            addr = runtimeWorkspace + rawTensor->addrOffset;
        }
        return addr;
    }

    // for stitch
    inline void GetInTensorOffset(int32v8& offset, int operationIndex, int operandIndex) const
    {
        auto func = GetSource();
        auto& operandInfo = func->GetOperationIOperandInfo(operationIndex, operandIndex);

        const SymInt* offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) :
                                                      offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) :
                                                      offsetSymList[1].Value();
    }

    inline void GetOutTensorOffset(int32v8& offset, int operationIndex, int operandIndex) const
    {
        auto func = GetSource();
        auto& operandInfo = func->GetOperationOOperandInfo(operationIndex, operandIndex);

        const SymInt* offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) :
                                                      offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) :
                                                      offsetSymList[1].Value();
    }

    inline void GetFuncTensorOffsetAndShape(uint64_t offset[DEV_SHAPE_DIM_MAX], uint64_t shape[DEV_SHAPE_DIM_MAX],
                                            int dims, int operationIndex, int operandIndex,
                                            bool isIOperand = true) const
    {
        auto func = GetSource();
        auto& operandInfo = func->GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);
        GetTensorOffsetAndShape<false>(func, offset, shape, &GetExpression(0), dims, operationIndex,
                                       operandInfo.staticOffsetAttrBeginIndex, operandInfo.staticShapeAttrBeginIndex);
    }

    std::string Dump(int indent = 0) const { return DupData()->Dump(indent); }

    inline int64_t GetValue(const SymInt* attrs, int idx) const
    {
        return attrs[idx].IsExpression() ? funcData->exprTbl[attrs[idx].Value()] : attrs[idx].Value();
    }

    inline uint64_t GetRawTensorAddrEx(int idx) const
    {
        auto desc = funcData->rawTensorDesc[idx];
        if (desc.location == RAW_TENSOR_LOCATION_LOCAL)
            return funcData->workspaceAddr + desc.offsetOrIndex;
        else
            return funcData->rawTensorAddr[desc.offsetOrIndex] & ((1UL << RAW_TENSOR_OFFSET_SIZE) - 1);
    }

    std::string DumpDyn(int funcIdx, int operIdx, const DevCceBinary* cceBinary) const
    {
        std::stringstream oss;
        auto func = GetSource();

        auto attrBase = reinterpret_cast<SymInt*>(&funcData->opAttrs[funcData->opAtrrOffsets[operIdx]]);
        auto funcIndex = attrBase[0].Value();
        oss << std::hex << " #funcKey " << func->funcKey << " #operIndex " << operIdx
            << " #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
            << " #coreType: " << cceBinary[funcIndex].coreType << " #taskID:" << MakeTaskID(funcIdx, operIdx) << "\n";

        auto dumpAttr = [this, &oss](auto attrs, auto& info) {
            int attrIndex = info.staticOffsetAttrBeginIndex;
            auto rawIndex = attrs[attrIndex - 1].Value();
            oss << rawIndex << "@" << GetRawTensorAddrEx(rawIndex) << ", ";
            int dim = info.GetDim();
            for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
                oss << GetValue(attrs, attrIndex + i) << ", ";
            }
        };

        int offset = 0;
        for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
            auto& opInfo = func->GetOperationIOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            dumpAttr(attrBase, opInfo);
        }
        for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
            auto& opInfo = func->GetOperationOOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            dumpAttr(attrBase, opInfo);
        }
        for (size_t idx = static_cast<size_t>(offset); idx < func->GetOperationAttrSize(operIdx); idx++) {
            oss << GetValue(attrBase, idx) << ", ";
        }
        return oss.str();
    }

    void DumpTopo(std::ofstream& os, int seqNo, int funcIdx, const DevCceBinary* cceBinary, bool enableVFFusion,
                  const DeviceTask* devTask = nullptr) const;

#if DEBUG_INFINITE_LIFETIME
    void DumpTensorAddrInfo(std::vector<std::string>& infos, uint32_t seqNo, uint32_t funcIdx);
#endif // DEBUG_INFINITE_LIFETIME

    void DumpRawShape(const DevAscendRawTensor* rawTensor, uint32_t dimSize, std::vector<std::string>& lines,
                      std::stringstream& oss) const;

    void DumpOperandShape(uint32_t dimSize, size_t opIdx, size_t operandIdx, bool isIn, std::vector<std::string>& lines,
                          std::stringstream& oss) const;

    std::vector<std::string> DumpLeafs(uint32_t seqNo, uint32_t funcIdx) const;

    void DumpAttr(const DevAscendFunction* func, const SymInt* attrs, const DevAscendOperationOperandInfo& info,
                  std::stringstream& oss) const
    {
        int attrOffset = info.staticOffsetAttrBeginIndex;
        auto rawIndex = attrs[attrOffset - 1].Value();
        oss << "@(rawidx:" << rawIndex << " attridx:" << (attrOffset - 1) << ")"
            << ", ";

        uint64_t tensorIndex = info.tensorIndex;
        if (tensorIndex >= func->GetRawTensorSize()) {
            DEV_ERROR(TensorMetaErr::RAW_TENSOR_INDEX_OUT_OF_RANGE,
                      "#ctrl.encode.raw_tensor: Invalid rawIndex=%lu, exceeds raw tensor size=%lu", tensorIndex,
                      func->GetRawTensorSize());
            return;
        }
        int dim = info.GetDim();
        auto rawTensor = func->GetRawTensor(tensorIndex);
        if (dim != rawTensor->GetDim()) {
            DEV_ERROR(TensorMetaErr::SHAPE_VALUE_MISMATCH,
                      "#ctrl.encode.shape: Dimension mismatch: info.dim=%d, rawTensor->dim=%d", dim,
                      rawTensor->GetDim());
        }

        for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
            oss << GetValue(attrs, attrOffset + i) << ", ";
        }
    };

    void DumpFuncData(const DevAscendFunction* func, int funcIdx, const DevCceBinary* cceBinary,
                      std::stringstream& oss) const
    {
        oss << "#funcData: [\n" << std::dec;
        for (size_t operIdx = 0; operIdx < func->GetOperationSize(); operIdx++) {
            auto attrBase = &func->GetOperationAttr(operIdx, 0);
            auto funcIndex = attrBase[0].Value();
            oss << "  [" << operIdx << "]  #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
                << " #funcIndex: " << funcIndex << " #taskID:" << MakeTaskID(funcIdx, operIdx)
                << " #opMagic: " << func->GetOperationDebugOpmagic(operIdx) << "\n";
            oss << "  #invokeAttrs : ";
            int offset = 0;
            for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
                auto& opInfo = func->GetOperationIOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
                oss << " in:";
                DumpAttr(func, attrBase, opInfo, oss);
            }
            for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
                auto& opInfo = func->GetOperationOOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
                oss << " out:";
                DumpAttr(func, attrBase, opInfo, oss);
            }
            oss << "\n other attr:";
            for (size_t idx = offset; idx < func->GetOperationAttrSize(operIdx); idx++) {
                oss << GetValue(attrBase, idx) << ", ";
            }
            oss << "\n";
        }
    }

    std::string DumpMainBlockFlag()
    {
        std::stringstream oss;
        oss << "isMainBlock: [" << funcData->exprTbl[0] << "]";
        return oss.str();
    }

    std::string DumpDyn(int funcIdx, const DevCceBinary* cceBinary) const
    {
        std::stringstream oss;
        auto func = GetSource();
        for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
            oss << std::hex << "[" << opIdx << "] #predCnt:" << GetOperationCurrPredCount(opIdx);
            auto& succList = func->GetOperationDepGraphSuccList(opIdx);
            oss << " #succList: [";
            for (size_t j = 0; j < succList.size(); j++) {
                if (j != 0)
                    oss << ", ";
                oss << func->At(succList, j);
            }
            oss << ']';
            auto& stitch = GetOperationStitch(opIdx);
            if (!stitch.IsNull())
                oss << std::hex << " #stitch:" << stitch.Dump();
            oss << "\n";
        }

        oss << " #funcKey: " << func->funcKey << " #workspace: " << funcData->workspaceAddr << "\n";

        DumpFuncData(func, funcIdx, cceBinary, oss);

        oss << std::hex << "  #rawTensorAddrs: ";
        for (uint64_t i = 0; i < func->GetRawTensorDescSize(); i++) {
            if (i % RAW_TENSOR_DESC_PRE_SIZE == 0)
                oss << "\n   ";
            if (GetRawTensorAddrEx(i) != GetRawTensorAddr(i)) {
                DEV_ERROR(TensorMetaErr::TENSOR_ENCODE_PTR_MISMATCH,
                          "#ctrl.encode.tensor_ptr: Tensor address mismatch at index %lu: addr=%lu, addrEx=%lu.", i,
                          GetRawTensorAddr(i), GetRawTensorAddrEx(i));
            }
            auto desc = funcData->rawTensorDesc[i];
            oss << GetRawTensorAddrEx(i) << "(location:" << desc.location << " offsetOrIdex: " << desc.offsetOrIndex
                << ")"
                << ", ";
        }
        oss << "\n]";
        return oss.str();
    }

    bool IsNull() const { return !dupTiny_; }
    void ResetNull() { dupTiny_.Invalidate(); }
    DynFuncData* GetFuncData() { return funcData; }
    void SetFuncData(DynFuncData* data) { funcData = data; }

    DevAscendFunctionDuppedData* DupDataForDynFuncData() { return DupData(); }

private:
    const DevAscendFunctionDuppedData* DupData() const { return dupTiny_.As<DevAscendFunctionDuppedData>(); }
    DevAscendFunctionDuppedData* DupData() { return dupTiny_.As<DevAscendFunctionDuppedData>(); }

private:
    DynFuncData* funcData{nullptr}; // used by aicore
    WsAllocation dupTiny_;
};
} // namespace npu::tile_fwk::dynamic
