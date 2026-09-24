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
 * \file dev_encode_function.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "machine/utils/dynamic/dev_encode_operation.h"
#include "machine/utils/dynamic/dev_encode_function_param.h"

namespace npu::tile_fwk {
struct L2Info;
class RawTensor;
class LogicalTensor;
class Operation;
class Function;
class IncastOutcastLink;
class IncastOutcastSlot;
class SymbolicSymbolTable;
class SymbolicExpressionTable;
} // namespace npu::tile_fwk

namespace npu::tile_fwk::dynamic {
constexpr int INVALID_INDEX = -1;

struct SlotMaskEntry {
    uint8_t mask{0};
    bool isPartial{false};
    uint32_t maxReadCount{0};
    bool hasAtomicWrite{false};
};

struct DevAscendFunctionDuppedData;
struct DevAscendFunction {
    uint64_t rootHash;
    uint64_t funcKey;
    // source root function after duplication
    DevAscendFunction* sourceFunc{nullptr};

    // Fill base address after stitch
    uintdevptr_t runtimeWorkspace;
    // Base address of invoke entries
    uintdevptr_t opAttrs;

    int funcidx;

    int stackWorkSpaceSize;

    uint32_t getInputDataCount;
    uint32_t getTensorDataCount;
    uint64_t hubOpCount_{0};

    DevLocalVector<AddressDescriptor> incastAddressList;
    DevLocalVector<AddressDescriptor> outcastAddressList;

    DevLocalVector<uint64_t> expressionList;
#define allocateLastField expressionList

    DevAscendFunctionPredInfo predInfo_;
    uint64_t duppedDataAllocSize_;
    uint64_t duppedDataCopySize_;
    DevLocalVector<uint8_t> duppedData_;
    // wrap for mix subgraph schedule
public:
    uint64_t wrapIdNum_{0};
    int* GetOpWrapListAddr() { return &At(opWrapList_, 0); }

    // total memory requirement of non-root-incast/outcast raw tensors
    uint64_t rootInnerTensorWsMemoryRequirement{0};
    uint64_t exclusiveOutcastWsMemoryRequirement{0};
    uint32_t unrollTimes{1};

    uint32_t GetMaxC() const { return maxC_; }
    uint32_t GetMaxV() const { return maxV_; }
    void SetMaxCV(uint32_t maxC, uint32_t maxV)
    {
        maxC_ = maxC;
        maxV_ = maxV;
    }

private:
    DevLocalVector<int> opWrapList_;
    DevLocalVector<DevAscendRawTensor> rawTensorList_;
    DevLocalVector<DevRawTensorDesc> rawTensorDescList_;
    DevLocalVector<DevAscendTensor> tensorList_;
    DevLocalVector<int> noPredOpList_;
    DevLocalVector<int> noSuccOpList_;
    DevLocalVector<DevAscendOperation> operationList_;
    DevLocalVector<DevAscendOperationOperandInfo> operationOperandInfoList_;
    DevLocalVector<SymInt> operationAttrList_;
    DevLocalVector<int> opAttrOffsetList_;
    DevLocalVector<int> opCalleeList_;
    DevLocalVector<int> operationSuccList_;
    DevLocalVector<npu::tile_fwk::DevAscendFunctionOperationSuccInfo> operationSuccInfoList_;
    DevLocalVector<int> operationCopyOutResolveSuccIndexList_;

    DevLocalVector<DevAscendFunctionIncast> incastList;
    DevLocalVector<DevAscendFunctionOutcast> outcastList;
    DevLocalVector<int> slotList;
    DevLocalVector<int> redaccAssembleSlotList_;

    DevLocalVector<DevAscendFunctionCallOperandUse> useList;
    DevLocalVector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList_;
    DevLocalVector<uint32_t> stitchPolicyFullCoverOpList_;

    DevLocalVector<uint32_t> cellMatchRuntimeFullUpdateTableList;

    struct OutCastSlotEntry {
        int slotIdx;
        int outcastIdx;
    };
    struct IncastSlotEntry {
        int slotIdx;
        int incastIdx;
    };
    DevLocalVector<OutCastSlotEntry> updateOutCastSlotList;
    DevLocalVector<IncastSlotEntry> updateIncastSlotList;
    DevLocalVector<IncastSlotEntry> stitchIncastSlotList;
    DevLocalVector<OutCastSlotEntry> stitchOutCastSlotList;

    DevLocalVector<uint64_t> deadEndHubBitmap_;
    DevLocalVector<uint64_t> tailTaskBitmap_;
    DevLocalVector<char> rawName_;
#define sharedLastField rawName_
    uint32_t maxC_{0};
    uint32_t maxV_{0};

public:
    uint8_t data[0];
    /*
     *  Duplicated:
     *      AddressDescriptor                                   incastAddressListData;
     *      AddressDescriptor                                   outcastAddressListData;
     *      DevAscendOperationDynamicField                      opDynamicFieldListData[];
     *  Allocated:
     *      uint64_t                                            expressionListData[];
     *
     *  Shared:
     *      DevRawTensorDesc                              rawTensorDescListData[];
     *      DevAscendRawTensor                                  rawTensorListData[];
     *      DevAscendTensor                                     tensorListData[];
     *      int                                                 noPredOpListData[];
     *      int                                                 noSuccOpListData[];
     *      DevAscendOperation                                  operationListData[];
     *      DevAscendOperationOperandInfo                       operationOperandListData[];
     *      SymInt                                              operationAttrListData[];
     *      int                                                 operationSuccListData[];
     *      int                                                 operationCopyOutResolveSuccIndexData[];
     *      DevAscendFunctionIncast                             incastListData[];
     *      DevAscendFunctionOutcast                            outcastListData[];
     *      int                                                 slotListData[];
     *      int                                                 outputOutcastSlotList[];
     *      int                                                 assembleOutcastSlotList[];
     *      int                                                 offsetIdxListData[];
     *      int                                                 shapeIdxListData[];
     *      int                                                 producerConsumerListData[];
     *      char                                                rawNameData[];
     *      uint8_t                                             duppedData[];
     */

    template <typename T>
    const T& At(const DevLocalVector<T>& localvec, int index) const
    {
        return *reinterpret_cast<T*>((reinterpret_cast<uint64_t>(this) + localvec.Offset(index)));
    }
    template <typename T>
    T& At(const DevLocalVector<T>& localvec, int index)
    {
        return *reinterpret_cast<T*>((reinterpret_cast<uint64_t>(this) + localvec.Offset(index)));
    }
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

private:
    std::string DumpTensor(int tensorIndex) const;
    std::string DumpOperationAttr(int operationIndex, uint64_t* runtimeExpressionList = nullptr,
                                  bool dumpIndex = false) const;
    std::string DumpOperation(int operationIndex, int& totalAttrStartIdx,
                              const std::vector<uintdevptr_t>& ooperandAddrList = {},
                              const std::vector<uintdevptr_t>& ioperandAddrList = {},
                              uint64_t* runtimeExpressionList = nullptr) const;
    std::string DumpRawTensor(int rawIndex, uintdevptr_t addr = 0) const;
    std::string DumpIncast(int incastIndex, const std::string& indent, uint64_t* runtimeExpressionList = nullptr,
                           const std::vector<uintdevptr_t>& slotAddrList = {}) const;
    std::string DumpOutcast(int outcastIndex, const std::string& indent, uint64_t* runtimeExpressionList = nullptr,
                            const std::vector<uintdevptr_t>& slotAddrList = {}) const;

public:
    std::string Dump(int indent = 0) const;

    bool HasValueDepend() const { return getInputDataCount + getTensorDataCount; }

    schema::coa SchemaGetCoa(int operationIndex, uint64_t* runtimeExpressionList = nullptr,
                             bool dumpIndex = false) const
    {
        std::vector<schema::TextType> coaDataList;
        for (size_t j = 0; j < GetOperationAttrSize(operationIndex); j++) {
            const SymInt& s = GetOperationAttr(operationIndex, j);
            std::string textData;
            if (s.IsExpression()) {
                if (runtimeExpressionList != nullptr) {
                    textData = std::to_string(runtimeExpressionList[s.Value()]);
                } else {
                    textData = "?" + std::to_string(s.Value());
                }
            } else {
                textData = std::to_string(s.Value());
            }
            coaDataList.push_back(schema::TextType(textData));
        }
        return schema::coa(schema::coaType(coaDataList, dumpIndex));
    }

    template <typename T>
    uint64_t GetEndOffset(const DevLocalVector<T>& localvec) const
    {
        return localvec.End();
    }

    void Reloc(intptr_t /* shift */, bool /* relocShared */) {}

    int GetFuncKey() const { return funcKey; }

    const DevAscendFunction* GetSource() const { return sourceFunc; }
    DevAscendFunction* GetSource() { return sourceFunc; }

    int GetRootIndex() const { return funcKey; }

    const int& GetFuncidx() const { return funcidx; }
    int& GetFuncidx() { return funcidx; }

    const DevAscendFunctionPredInfo& GetPredInfo() const { return predInfo_; }
    uint64_t GetDuppedDataAllocSize() const { return duppedDataAllocSize_; }
    uint64_t GetDuppedDataCopySize() const { return duppedDataCopySize_; }
    DevAscendFunctionDuppedData* GetDuppedData() const
    {
        return reinterpret_cast<DevAscendFunctionDuppedData*>(const_cast<uint8_t*>(&At(duppedData_, 0)));
    }

    int32_t* GetOpAttrOffsetAddr() { return &At(opAttrOffsetList_, 0); }
    const int32_t* GetOpAttrOffsetAddr() const { return &At(opAttrOffsetList_, 0); }
    inline int32_t GetOpAttrOffsetSize() { return opAttrOffsetList_.size(); }
    int* GetCalleeIndexAddr() { return &At(opCalleeList_, 0); }

    uint64_t* GetExpressionAddr() { return &At(expressionList, 0); }
    uint64_t GetAllocateSize() const { return GetEndOffset(allocateLastField); }

    uint64_t GetSize() const { return GetEndOffset(sharedLastField); }

    uintdevptr_t GetRuntimeWorkspace() const { return runtimeWorkspace; }
    uintdevptr_t& GetRuntimeWorkspace() { return runtimeWorkspace; }

    inline AddressDescriptor GetIncastAddress(int index) const { return At(incastAddressList, index); }
    inline AddressDescriptor& GetIncastAddress(int index) { return At(incastAddressList, index); }

    inline AddressDescriptor GetOutcastAddress(int index) const { return At(outcastAddressList, index); }
    inline AddressDescriptor& GetOutcastAddress(int index) { return At(outcastAddressList, index); }

    inline uint64_t GetExpression(int tableIndex) const { return At(expressionList, tableIndex); }
    inline uint64_t& GetExpression(int tableIndex) { return At(expressionList, tableIndex); }
    inline uint64_t GetExpressionSize() const { return expressionList.size(); }
    inline uint64_t GetRawTensorSize() const { return rawTensorList_.size(); }
    inline const DevAscendRawTensor* GetRawTensor(const DevAscendTensor* tensor) const
    {
        int rawTensorIndex = tensor->rawIndex;
        return &At(rawTensorList_, rawTensorIndex);
    }
    inline const DevAscendRawTensor* GetRawTensor(int rawIndex) const { return &At(rawTensorList_, rawIndex); }
    inline DevAscendRawTensor* GetRawTensor(int rawIndex) { return &At(rawTensorList_, rawIndex); }
    inline const DevRawTensorDesc* GetRawTensorDesc(int rawIndex) const { return &At(rawTensorDescList_, rawIndex); }
    inline DevRawTensorDesc* GetRawTensorDesc(int rawIndex) { return &At(rawTensorDescList_, rawIndex); }
    inline size_t GetRawTensorDescSize() const { return rawTensorDescList_.size(); }

    inline uint64_t GetTensorSize() const { return tensorList_.size(); }
    inline const DevAscendTensor* GetTensor(int index) const { return &At(tensorList_, index); }
    inline DevAscendTensor* GetTensor(int index) { return &At(tensorList_, index); }

    inline size_t GetNoPredOpSize() const { return noPredOpList_.size(); }
    inline int GetNoPredOpIdx(size_t idx) const { return At(noPredOpList_, idx); }

    inline size_t GetNoSuccOpSize() const { return noSuccOpList_.size(); }
    inline int GetNoSuccOpIdx(size_t idx) const { return At(noSuccOpList_, idx); }

    inline size_t GetOperationSize() const { return operationList_.size(); }
    inline bool IsDeadEndHub(uint32_t opIndex) const
    {
        if (deadEndHubBitmap_.size() == 0)
            return false;
        uint32_t wordIdx = opIndex / 64;
        return (At(deadEndHubBitmap_, wordIdx) & (1ULL << (opIndex % 64))) != 0;
    }
    inline bool IsTailTask(uint32_t opIndex) const
    {
        if (tailTaskBitmap_.size() == 0) {
            return false;
        }
        uint32_t wordIdx = opIndex / 64;
        return (At(tailTaskBitmap_, wordIdx) & (1ULL << (opIndex % 64))) != 0;
    }
    inline void ClearTailTask(uint32_t opIndex)
    {
        if (tailTaskBitmap_.size() == 0) {
            return;
        }
        uint32_t wordIdx = opIndex / 64;
        At(tailTaskBitmap_, wordIdx) &= ~(1ULL << (opIndex % 64));
    }
    inline bool ClearDeadEndHub(uint32_t opIndex)
    {
        if (deadEndHubBitmap_.size() == 0)
            return false;
        uint32_t wordIdx = opIndex / 64;
        uint64_t bit = 1ULL << (opIndex % 64);
        auto& word = At(deadEndHubBitmap_, wordIdx);
        bool wasSet = (word & bit) != 0;
        word &= ~bit;
        return wasSet;
    }
    inline void PropagateDeadHubClear(uint32_t clearedOpIdx)
    {
        constexpr size_t kMaxStack = 128;
        uint32_t stack[kMaxStack];
        size_t stackSize = 0;
        stack[stackSize++] = clearedOpIdx;

        size_t opCount = GetOperationSize();
        while (stackSize > 0) {
            uint32_t target = stack[--stackSize];
            for (size_t op = 0; op < opCount; op++) {
                size_t succSize;
                const int* succList = GetOperationDepGraphSuccAddr(static_cast<int>(op), succSize);
                for (size_t j = 0; j < succSize; j++) {
                    if (static_cast<uint32_t>(succList[j]) != target)
                        continue;
                    ClearTailTask(static_cast<uint32_t>(op));
                    if (ClearDeadEndHub(static_cast<uint32_t>(op)) && stackSize < kMaxStack) {
                        stack[stackSize++] = static_cast<uint32_t>(op);
                    }
                    break;
                }
            }
        }
    }
    inline size_t GetBitmapByteSize() const { return deadEndHubBitmap_.size() * sizeof(uint64_t); }
    inline void BackupBitmapTo(uint64_t* deadEndBuf, uint64_t* tailBuf, size_t byteSize) const
    {
        if (byteSize == 0) {
            return;
        }
        DevMemcpyS(deadEndBuf, byteSize, &At(deadEndHubBitmap_, 0), byteSize);
        DevMemcpyS(tailBuf, byteSize, &At(tailTaskBitmap_, 0), byteSize);
    }
    inline void RestoreBitmapFrom(const uint64_t* deadEndBuf, const uint64_t* tailBuf, size_t byteSize)
    {
        if (byteSize == 0) {
            return;
        }
        DevMemcpyS(&At(deadEndHubBitmap_, 0), byteSize, deadEndBuf, byteSize);
        DevMemcpyS(&At(tailTaskBitmap_, 0), byteSize, tailBuf, byteSize);
    }
    inline uint32_t GetOperationStitchIndex(int operationIndex) const
    {
        return At(operationList_, operationIndex).stitchIndex;
    }
    inline uint32_t GetOperationDebugOpmagic(int operationIndex) const
    {
        return At(operationList_, operationIndex).debugOpmagic;
    }
    inline size_t GetOperationIOperandSize(int operationIndex) const
    {
        return At(operationList_, operationIndex).ioperandList.size();
    }
    inline size_t GetOperationOOperandSize(int operationIndex) const
    {
        return At(operationList_, operationIndex).ooperandList.size();
    }

    inline const DevAscendOperationOperandInfo& GetOperationIOperandInfo(int operationIndex, int operandIndex) const
    {
        return At(At(operationList_, operationIndex).ioperandList, operandIndex);
    }
    inline const DevAscendTensor* GetOperationIOperand(int operationIndex, int operandIndex) const
    {
        int tensorIndex = GetOperationIOperandInfo(operationIndex, operandIndex).tensorIndex;
        return GetTensor(tensorIndex);
    }

    inline const DevAscendOperationOperandInfo& GetOperationOOperandInfo(int operationIndex, int operandIndex) const
    {
        return At(At(operationList_, operationIndex).ooperandList, operandIndex);
    }
    inline const DevAscendTensor* GetOperationOOperand(int operationIndex, int operandIndex) const
    {
        int tensorIndex = GetOperationOOperandInfo(operationIndex, operandIndex).tensorIndex;
        return GetTensor(tensorIndex);
    }

    inline const DevAscendOperationOperandInfo& GetOperationOperandInfo(int operationIndex, int operandIndex,
                                                                        bool isIOperand = true) const
    {
        if (isIOperand) {
            return GetOperationIOperandInfo(operationIndex, operandIndex);
        } else {
            return GetOperationOOperandInfo(operationIndex, operandIndex);
        }
    }

    inline size_t GetOperationAttrSize(int operationIndex) const
    {
        return At(operationList_, operationIndex).attrList.size();
    }
    inline const SymInt& GetOperationAttr(int operationIndex, int attrIndex) const
    {
        return At(At(operationList_, operationIndex).attrList, attrIndex);
    }
    inline int GetOperationAttrCalleeIndex(int operationIndex) const
    {
        return GetOperationAttr(operationIndex, 0).Value();
    }

    inline int GetOpAttrSize() { return operationAttrList_.size(); }

    inline void FillOpAttrs(DevCceBinary* cceInfo) { (void)cceInfo; }

    inline int GetOperationSucc(size_t idx) const { return At(operationSuccList_, idx); }
    inline int& GetOperationSucc(size_t idx) { return At(operationSuccList_, idx); }

    inline npu::tile_fwk::DevAscendFunctionOperationSuccInfo GetOperationSuccInfo(size_t operationIndex) const
    {
        return At(operationSuccInfoList_, operationIndex);
    }
    inline npu::tile_fwk::DevAscendFunctionOperationSuccInfo& GetOperationSuccInfo(size_t operationIndex)
    {
        return At(operationSuccInfoList_, operationIndex);
    }

    inline const uint32_t& GetOperationDepGraphPredCount(int operationIndex) const
    {
        return At(operationList_, operationIndex).depGraphPredCount;
    }
    inline uint32_t& GetOperationDepGraphPredCount(int operationIndex)
    {
        return At(operationList_, operationIndex).depGraphPredCount;
    }

    inline const DevLocalVector<int>& GetOperationDepGraphSuccList(int operationIndex) const
    {
        return At(operationList_, operationIndex).depGraphSuccList;
    }

    inline const DevLocalVector<int>& GetOperationDepGraphCopyOutResolveSuccIndexList(int operationIndex) const
    {
        return At(operationList_, operationIndex).depGraphCopyOutResolveSuccIndexList;
    }

    inline const int* GetOperationDepGraphSuccAddr(int operationIndex, size_t& size) const
    {
        auto& succList = At(operationList_, operationIndex).depGraphSuccList;
        size = succList.size();
        return &At(succList, 0);
    }

    inline const int* GetOperationDepGraphCopyOutResolveSuccIndexAddr(int operationIndex, size_t& size) const
    {
        auto& succIndexList = At(operationList_, operationIndex).depGraphCopyOutResolveSuccIndexList;
        size = succIndexList.size();
        return &At(succIndexList, 0);
    }

    inline size_t GetIncastSize() const { return incastList.size(); }
    inline const struct DevAscendFunctionIncast& GetIncast(int index) const { return At(incastList, index); }
    inline struct DevAscendFunctionIncast& GetIncast(int index) { return At(incastList, index); }
    inline const DevAscendRawTensor* GetIncastRawTensor(int index) const
    {
        int tensorIndex = GetIncast(index).tensorIndex;
        return GetRawTensor(GetTensor(tensorIndex));
    }

    inline size_t GetUpdateOutCastSlotListSize() const { return updateOutCastSlotList.size(); }
    inline const OutCastSlotEntry& GetUpdateOutCastSlot(int index) const { return At(updateOutCastSlotList, index); }
    inline size_t GetStitchIncastSlotListSize() const { return stitchIncastSlotList.size(); }
    inline const IncastSlotEntry& GetStitchIncastSlot(int index) const { return At(stitchIncastSlotList, index); }
    inline size_t GetUpdateIncastSlotListSize() const { return updateIncastSlotList.size(); }
    inline const IncastSlotEntry& GetUpdateIncastSlot(int index) const { return At(updateIncastSlotList, index); }
    inline size_t GetStitchOutCastSlotListSize() const { return stitchOutCastSlotList.size(); }
    inline const OutCastSlotEntry& GetStitchOutCastSlot(int index) const { return At(stitchOutCastSlotList, index); }

    inline size_t GetOutcastSize() const { return outcastList.size(); }
    inline const struct DevAscendFunctionOutcast& GetOutcast(int index) const { return At(outcastList, index); }
    inline struct DevAscendFunctionOutcast& GetOutcast(int index) { return At(outcastList, index); }

    inline size_t GetRedaccAssembleSlotListSize() const { return redaccAssembleSlotList_.size(); }
    inline const int& GetRedaccAssembleSlotList(int index) const { return At(redaccAssembleSlotList_, index); }
    inline int& GetRedaccAssembleSlotList(int index) { return At(redaccAssembleSlotList_, index); }

    int LookupIncastBySlotIndex(int slotIndex) const;
    std::vector<int> LookupIncastBySlotIndexList(const std::vector<int>& slotIndexList) const;

    int LookupOutcastBySlotIndex(int slotIndex) const;
    std::vector<int> LookupOutcastBySlotIndexList(const std::vector<int>& slotIndexList) const;

    std::vector<std::tuple<int, int, int>> LookupConnectionSlotIndexFrom(const DevAscendFunction* func) const;

    void AppendOutcastConnections(std::vector<std::tuple<int, int, int>>& connectionList, int fromSlot, int incastIndex,
                                  const DevAscendFunction* func) const;

    inline const DevAscendRawTensor* GetOutcastRawTensor(int index) const
    {
        int tensorIndex = GetOutcast(index).tensorIndex;
        return GetRawTensor(GetTensor(tensorIndex));
    }

    inline void GetTensorOffset(uint64_t offset[DEV_SHAPE_DIM_MAX], const DevAscendRawTensor* rawTensor,
                                const DevAscendOperationOperandInfo& operandInfo) const
    {
        const SymInt* offsetSymList = &At(operationAttrList_, operandInfo.staticOffsetAttrBeginIndex);
        for (int i = 0; i < rawTensor->GetDim(); i++) {
            offset[i] = offsetSymList[i].IsExpression() ? At(expressionList, offsetSymList[i].Value()) :
                                                          offsetSymList[i].Value();
        }
    }

    inline const SymInt* GetSymoffset(int offset) const { return &At(operationAttrList_, offset); }

    struct SymIntPair {
        const SymInt* offsetSymList;
        const SymInt* shapeSymList;
    };
    inline SymIntPair GetTensorOffsetShapeSymList(int operationIndex, int offsetAttrIndex, int shapeAttrIndex) const
    {
        const SymInt* offsetSymList = &GetOperationAttr(operationIndex, offsetAttrIndex);
        const SymInt* shapeSymList = &GetOperationAttr(operationIndex, shapeAttrIndex);
        return SymIntPair{offsetSymList, shapeSymList};
    }

    inline const char* GetRawName() const { return &At(rawName_, 0); }

    void RefilterStitchUpdateSlotLists(const std::unordered_map<int, SlotMaskEntry>& stitchUpdateSlotMaskMap);

private:
    friend struct EncodeDevAscendFunctionInfo;

    void InitIncastOutcastAttr(uintdevptr_t& initOffset, const std::vector<std::shared_ptr<LogicalTensor>>& iList,
                               const std::vector<std::shared_ptr<LogicalTensor>>& oList, bool fillContent);
    void InitOperationDynamicField(uintdevptr_t& initOffset, DevAscendFunctionPredInfo predInfo, uint32_t stitchCount,
                                   const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                                   const SymbolicExpressionTable* expressionTable,
                                   const OrderedSet<Operation*>& callList,
                                   const std::vector<std::shared_ptr<LogicalTensor>>& incastTensorList,
                                   const std::vector<std::shared_ptr<LogicalTensor>>& outcastTensorList,
                                   const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
                                   bool fillContent);
    void FillExclusiveOutcastSlotMark(const IncastOutcastLink* inoutLink,
                                      std::vector<bool>& isExclusiveOutcastSlotMarks);
    void InitRawTensorAndMemoryRequirement(
        uintdevptr_t& initOffset, const OrderedSet<std::shared_ptr<RawTensor>>& incastRawList,
        const OrderedSet<std::shared_ptr<RawTensor>>& outcastRawList,
        const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
        const std::unordered_map<int, std::shared_ptr<RawTensor>>& rawMagicToRawTensor,
        const std::vector<EncodeRawTensorAttr>& rawAttrs, const EncodeDevAscendFunctionParam& param,
        const SymbolicExpressionTable* expressionTable, bool fillContent);

    void UpdateRawTensorDesc(const std::shared_ptr<RawTensor>& rawTensor, size_t i, size_t incastRawListSize,
                             DevAscendRawTensor& encoded);

    void InitTensor(uintdevptr_t& initOffset, const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist,
                    const OrderedSet<std::shared_ptr<RawTensor>>& rawList, bool fillContent);

    void InitOperation(uintdevptr_t& initOffset, const SymbolicExpressionTable* expressionTable,
                       const OrderedSet<Operation*>& callList, const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist,
                       const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
                       const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
                       const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
                       const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                       const std::vector<int32_t>& stitchIndexList, const std::vector<int>& noPredOpList,
                       const std::vector<int>& noSuccOpList,
                       const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict,
                       bool fillContent);
    void InitOperationNoPredNoSuccIndices(uintdevptr_t& initOffset, const OrderedSet<Operation*>& callList,
                                          const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
                                          const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
                                          const std::vector<int>& noPredOpList, const std::vector<int>& noSuccOpList,
                                          bool fillContent);
    void InitOperationBufferLayouts(
        uintdevptr_t& initOffset, const OrderedSet<Operation*>& callList,
        const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
        const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict);
    void FillOperationEncodedContent(
        const SymbolicExpressionTable* expressionTable, const OrderedSet<Operation*>& callList,
        const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist, const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
        const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
        const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
        const std::unordered_map<uint64_t, int>& calleeHashIndexDict, const std::vector<int32_t>& stitchIndexList,
        const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict, bool fillContent);
    void PopulateOperationEncodedContent(
        const SymbolicExpressionTable* expressionTable, const OrderedSet<Operation*>& callList,
        const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist, const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
        const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
        const std::unordered_map<uint64_t, int>& calleeHashIndexDict, const std::vector<int32_t>& stitchIndexList,
        const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict,
        DevAscendFunctionDuppedData* dupData);
    void PopulateOneEncodedOpOperandsAndAttrs(size_t index, int& operanSize, int& staticAttributeSize,
                                              const SymbolicExpressionTable* expressionTable,
                                              const OrderedSet<Operation*>& callList,
                                              const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist,
                                              const OrderedSet<std::shared_ptr<RawTensor>>& rawList,
                                              const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                                              const std::vector<int32_t>& stitchIndexList);
    void PopulateOneEncodedOpGraphEdges(
        size_t index, int& sucSize, int& copyOutResolveSuccIdxSize, const OrderedSet<Operation*>& callList,
        const std::unordered_map<Operation*, OrderedSet<Operation*>>& callOpSuccDict,
        const std::unordered_map<Operation*, std::vector<int>>& copyOutResolveSuccIndexListDict,
        const std::vector<int32_t>& stitchIndexList, DevAscendFunctionDuppedData* dupData);
    void VerifyOperationEncodedContent(const OrderedSet<Operation*>& callList,
                                       const std::unordered_map<Operation*, uint64_t>& callOpPredDict,
                                       DevAscendFunctionDuppedData* dupData);
    void InitWrapInfo(uintdevptr_t& initOffset, const OrderedSet<Operation*>& callList, bool fillContent,
                      const std::unordered_map<uint64_t, int>& calleeHashIndexDict,
                      const std::vector<CceCodeInfo>& cceCodeInfoList);
    void InitIncastOutcast(
        uintdevptr_t& initOffset, const std::vector<std::shared_ptr<LogicalTensor>>& incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>>& outcastTensorList,
        const OrderedSet<std::shared_ptr<LogicalTensor>>& tlist,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& incastOpAttrDict,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& outcastOpAttrDict,
        const EncodeDevAscendFunctionParam& param, const std::string& initRawName, bool fillContent,
        const std::unordered_map<int, int>& opIdxToHubOpIdx = {});

    void FillIncastUseList(DevLocalVector<DevAscendFunctionCallOperandUse>& useList, uint64_t& useSize,
                           const std::vector<std::shared_ptr<LogicalTensor>>& tensorList,
                           const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& attrDict,
                           bool fillContent, const std::unordered_map<int, int>& opIdxToHubOpIdx);

    void FillOutcastUseList(DevLocalVector<DevAscendFunctionCallOperandUse>& useList, uint64_t& useSize,
                            const std::vector<std::shared_ptr<LogicalTensor>>& tensorList,
                            const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr>& attrDict,
                            bool fillContent, const std::unordered_map<int, int>& opIdxToHubOpIdx);

    void InitStitchUpdateSlotPrefilterLists(uintdevptr_t& initOffset, const IncastOutcastSlot* slot, int totalSlot,
                                            const std::vector<std::shared_ptr<LogicalTensor>>& incastTensorList,
                                            const std::vector<std::shared_ptr<LogicalTensor>>& outcastTensorList,
                                            bool fillContent);
};
} // namespace npu::tile_fwk::dynamic
