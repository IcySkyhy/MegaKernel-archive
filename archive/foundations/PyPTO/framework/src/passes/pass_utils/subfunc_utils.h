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
 * \file subfunc_utils.h
 * \brief
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
#define PRIOR_SCHEDULING // comment it to disable PriorScheduling pass

#ifdef PRIOR_SCHEDULING
using setType = std::conditional<true, std::unordered_set<int>, std::set<int>>::type;
#else
using setType = std::conditional<false, std::unordered_set<int>, std::set<int>>::type;
#endif

constexpr int INVALID_IN_OUT_INDEX = -1;
// 每个调用子图的实参列表
class SubfuncInvokeInfoTy {
public:
    struct BaseParamPackTy {
        int paramLoc;
        int ddrId;
        Offset offset;
        Shape shape;
        Shape rawShape;
        DataType dType;
        LogicalTensorPtr tensor;
        int opMagic;
        int operandIdx;

    protected:
        BaseParamPackTy() = default;
        BaseParamPackTy(const int newParamLoc, const int newDdrId, const std::vector<int64_t>& newOffset,
                        const std::vector<int64_t>& newShape, const std::vector<int64_t>& newRawShape,
                        const DataType newDtype, const LogicalTensorPtr& newTensor, const int newOpMagic,
                        int newOperandIdx)
            : paramLoc(newParamLoc),
              ddrId(newDdrId),
              offset(newOffset),
              shape(newShape),
              rawShape(newRawShape),
              dType(newDtype),
              tensor(newTensor),
              opMagic(newOpMagic),
              operandIdx(newOperandIdx)
        {}
    };

    struct TensorParamPackTy : BaseParamPackTy {
        bool isOutputToGM;

        TensorParamPackTy(const int newParamLoc, const int newDdrId, const std::vector<int64_t>& newOffset,
                          const std::vector<int64_t>& newShape, const std::vector<int64_t>& newRawShape,
                          const DataType newDdType, const bool newIsOutputToGM, const LogicalTensorPtr& newTensor,
                          const int newOpMagic, int newOperandIdx)
            : BaseParamPackTy(newParamLoc, newDdrId, newOffset, newShape, newRawShape, newDdType, newTensor, newOpMagic,
                              newOperandIdx),
              isOutputToGM(newIsOutputToGM)
        {}

        TensorParamPackTy() = default;
        void Print(std::ostream& osm = std::cout) const;
        void DumpTensor(std::vector<int64_t>& invokeParam) const;
        bool operator==(const TensorParamPackTy& other) const;
        bool operator!=(const TensorParamPackTy& other) const;
    };

    struct IncastParamPackTy : BaseParamPackTy {
        bool isOutputToGM{false};

        IncastParamPackTy() = default;

        IncastParamPackTy(const int newParamLoc, const int newDdrId, const std::vector<int64_t>& newOffset,
                          const std::vector<int64_t>& newShape, const std::vector<int64_t>& newRawShape,
                          const DataType newDdType, const LogicalTensorPtr& newTensor, const int newOpMagic,
                          int newOperandIdx)
            : BaseParamPackTy(newParamLoc, newDdrId, newOffset, newShape, newRawShape, newDdType, newTensor, newOpMagic,
                              newOperandIdx)
        {}

        void Print(std::ostream& osm = std::cout) const;
        void DumpIncastInfo(std::vector<int64_t>& invokeParam) const;
        bool operator==(const IncastParamPackTy& other) const;
        bool operator!=(const IncastParamPackTy& other) const;
    };

    struct OutcastParamPackTy : BaseParamPackTy {
        int refCount;

        OutcastParamPackTy(const int newParamLoc, const int newDdrId, const int newRefCount,
                           const std::vector<int64_t>& newShape, const std::vector<int64_t>& rawshape,
                           const std::vector<int64_t>& newOffset, const DataType newDdType,
                           const LogicalTensorPtr& newTensor, const int newOpMagic, int newOperandIdx)
            : BaseParamPackTy(newParamLoc, newDdrId, newOffset, newShape, rawshape, newDdType, newTensor, newOpMagic,
                              newOperandIdx),
              refCount(newRefCount)
        {}

        OutcastParamPackTy() = default;

        void Print(std::ostream& osm = std::cout) const;
        void DumpOutcastInfo(std::vector<int64_t>& invokeParam) const;
        bool operator==(const OutcastParamPackTy& other) const;
        bool operator!=(const OutcastParamPackTy& other) const;
    };

    // seq_no is in called subgraph
    struct InCastInfoTy {
        int operandIdx;
        int realIncastDDRId;
        Offset offset;
        Shape shape;
        Shape rawShape;
        DataType dType;
        LogicalTensorPtr tensor;
        int opMagic;

        InCastInfoTy(const int newOperandIdx, const int newRealIncastDDRId, const std::vector<int64_t>& newOffset,
                     const std::vector<int64_t>& newShape, const std::vector<int64_t>& newRawShape,
                     const DataType dtype, const LogicalTensorPtr& newTensor, const int newOpMagic)
            : operandIdx(newOperandIdx),
              realIncastDDRId(newRealIncastDDRId),
              offset(newOffset),
              shape(newShape),
              rawShape(newRawShape),
              dType(dtype),
              tensor(newTensor),
              opMagic(newOpMagic)
        {}
    };

    // Input output tensors of this subgraph invoke
    struct TensorInfoTy {
        int operandIdx;
        int realDDRId;
        Offset offset;
        Shape shape;
        Shape rawShape;
        DataType dType;
        bool isOutputToGM;
        LogicalTensorPtr tensor;
        int opMagic;

        TensorInfoTy(const int newOperandIndex, const int newRealDDRId, const std::vector<int64_t>& newOffset,
                     const std::vector<int64_t>& newShape, const std::vector<int64_t>& newRawShape,
                     const DataType newDtype, const bool newIsOutputToGM, const LogicalTensorPtr& newTensor,
                     const int newOpMagic)
            : operandIdx(newOperandIndex),
              realDDRId(newRealDDRId),
              offset(newOffset),
              shape(newShape),
              rawShape(newRawShape),
              dType(newDtype),
              isOutputToGM(newIsOutputToGM),
              tensor(newTensor),
              opMagic(newOpMagic)
        {}
    };

    using TensorArgsTy = std::vector<TensorInfoTy>;
    // Incast connections
    using ExeSubgraphEdgeTy = std::tuple<int, int, InCastInfoTy>;
    // record all the connections for input subgraphs
    using ESgConnectionsTy = std::vector<ExeSubgraphEdgeTy>;

    struct SuccessorIncastRecTy {
        int successorESgId;
        int connectedOperandIdx;
        ExeSubgraphEdgeTy* successorIncast;
        int opMagic;

        SuccessorIncastRecTy(const int esgId, const int opIdx, ExeSubgraphEdgeTy* exeSubgraphEdgeTy,
                             const int newOpMagic)
            : successorESgId(esgId), connectedOperandIdx(opIdx), successorIncast(exeSubgraphEdgeTy), opMagic(newOpMagic)
        {}
    };

    using SuccessorIncastInfoTy = std::vector<SuccessorIncastRecTy>;
    struct OutCastInfoTy {
        int srcESgId;
        int operandIdx;
        int refCount;
        int realOutCastDDRId;
        SuccessorIncastInfoTy successorIncastInfo;
        Offset offset;
        Shape shape;
        Shape rawShape;
        DataType dType;
        LogicalTensorPtr tensor;
        int opMagic;

        OutCastInfoTy(const int newSrcESgId, int newOperandIdx, const int newRefCount, const int newDdrId,
                      const SuccessorIncastInfoTy& info, const std::vector<int64_t>& newOffset,
                      const std::vector<int64_t>& newShape, const std::vector<int64_t>& newRawShape,
                      const DataType dtype, const LogicalTensorPtr& newTensor, const int newOpMagic)
            : srcESgId(newSrcESgId),
              operandIdx(newOperandIdx),
              refCount(newRefCount),
              realOutCastDDRId(newDdrId),
              successorIncastInfo(info),
              offset(newOffset),
              shape(newShape),
              rawShape(newRawShape),
              dType(dtype),
              tensor(newTensor),
              opMagic(newOpMagic)
        {}

        OutCastInfoTy() = default;
    };
    using OutCastConnectionsTy = std::vector<OutCastInfoTy>;

    inline void UpdateProgramSubgraphId(const int psgId) { programSubgraphId_ = psgId; }

    inline int GetProgramId() const { return programSubgraphId_; }

    void ConstructActualInvokeParam(int esgId);

    void PrintInvokeInfo(const std::string& extraInfo) const;

    void PrettyPrintInvokeInfo(const int subgraphId) const;

    void DumpInvokeInfo(int64_t invokeParamMemOffset, int64_t* invokeParamPtr) const;

    inline const std::vector<TensorParamPackTy>& GetTensorParamList() const { return tensorParamList_; }

    inline const std::vector<IncastParamPackTy>& GetIncastTensorParamList() const { return incastTensorParamList_; }

    inline const std::vector<OutcastParamPackTy>& GetOutcastTensorParamList() const { return outcastTensorParamList_; }

    std::tuple<int, int, int> LookupInvokeArgs(const int paramLoc) const;

    bool operator==(const SubfuncInvokeInfoTy& other) const;
    bool operator!=(const SubfuncInvokeInfoTy& other) const;
    friend class Allocator;
    friend class MixDependencyAnalyzer;

    inline void RecordTensorArg(const int operandIdx, const int realDDRId, const std::vector<int64_t>& offset,
                                const std::vector<int64_t>& shape, const std::vector<int64_t>& rawShape,
                                const DataType dtype, const bool isOutputToGM, const LogicalTensorPtr& tensor,
                                const int opMagic)
    {
        tensorArgs_.emplace_back(operandIdx, realDDRId, offset, shape, rawShape, dtype, isOutputToGM, tensor, opMagic);
    }

    // Record Incast connection, build relation shape with outcast records
    inline void RecordConnection(const int srcESgId, const int dstESgId, const int operandIndex,
                                 const int realIncastDDRId, const std::vector<int64_t>& offset,
                                 const std::vector<int64_t>& shape, const std::vector<int64_t>& rawShape,
                                 const DataType dtype, const LogicalTensorPtr& tensor, const int opMagic)
    {
        connections_.emplace_back(
            srcESgId, dstESgId,
            InCastInfoTy{operandIndex, realIncastDDRId, offset, shape, rawShape, dtype, tensor, opMagic});
    }

    inline void RecordOutcast(const int srcESgId, int srcOperandIdx, const int refCount, const int realOutcastDDRId,
                              const SuccessorIncastInfoTy& incasts, const std::vector<int64_t>& offset,
                              const std::vector<int64_t>& shape, const std::vector<int64_t>& rawShape,
                              const DataType dtype, const LogicalTensorPtr& tensor, const int opMagic)
    {
        outCasts_.emplace_back(srcESgId, srcOperandIdx, refCount, realOutcastDDRId, incasts, offset, shape, rawShape,
                               dtype, tensor, opMagic);
    }

    // do some sorting after recording all infomations
    void DoFinishRecord();

    const ESgConnectionsTy& GetIncasts() const { return connections_; }

    const OutCastConnectionsTy& GetOutcasts() const { return outCasts_; }

    const TensorArgsTy& GetTensorArgs() const { return tensorArgs_; }

    CoreType GetGraphType() const { return graphType_; }

    void SetGraphType(const CoreType graphType) { graphType_ = graphType; }

    Json ToJson() const;
    Json DumpJson() const;
    void LoadIncastFromJson(const Json& incastJson, Function* belongTo);
    void LoadOutcastFromJson(const Json& outcastJson, Function* belongTo);
    void LoadTensorFromJson(const Json& tensorJson, Function* belongTo);
    void LoadJson(const Json& invokeInfoJson, Function* belongTo);
    void Print(const std::string& extInfo) const;

private:
    int programSubgraphId_; // The called merged subgraph id
    std::vector<TensorParamPackTy> tensorParamList_;
    std::vector<IncastParamPackTy> incastTensorParamList_;
    std::vector<OutcastParamPackTy> outcastTensorParamList_;
    CoreType graphType_{CoreType::AIV};
    TensorArgsTy tensorArgs_;
    ESgConnectionsTy connections_; // InCast
    OutCastConnectionsTy outCasts_;
    bool isFinalized_{false};
};

class SubfuncParam {
public:
    struct InCastParamTy {
        int paramLoc;
        int operandIdx;
        int symDDRId;
        Shape shape;
        Offset offset;
        std::string symName;
        std::string symbol;
        DataType dataType;

        InCastParamTy(const int newOperandIdx, const int newSymDDRId, const std::vector<int64_t>& newShape,
                      const std::vector<int64_t>& newOffset, const std::string& newSymName, const int newParamLoc,
                      const std::string newSymbol = "", const DataType newDataType = DataType::DT_BOTTOM)
            : paramLoc(newParamLoc),
              operandIdx(newOperandIdx),
              symDDRId(newSymDDRId),
              shape(newShape),
              offset(newOffset),
              symName(newSymName),
              symbol(newSymbol),
              dataType(newDataType)
        {}

        void Print(std::ostream& osm = std::cout) const;
        bool CompareParam(const SubfuncInvokeInfoTy::IncastParamPackTy& esgParam) const;
    };

    struct OutCastParamTy {
        int paramLoc;
        int operandIdx;
        int symDDRId;
        int refCount;
        Offset offset;
        Shape shape;
        std::string symName;
        std::string symbol;
        DataType dataType;

        OutCastParamTy(const int newOperandIdx, const int newSymDDRId, const int newRefCount,
                       const std::vector<int64_t>& newShape, const std::vector<int64_t>& newOffset,
                       const std::string& newSymName, const int newParamLoc, const std::string newSymbol = "",
                       const DataType newDataType = DataType::DT_BOTTOM)
            : paramLoc(newParamLoc),
              operandIdx(newOperandIdx),
              symDDRId(newSymDDRId),
              refCount(newRefCount),
              offset(newOffset),
              shape(newShape),
              symName(newSymName),
              symbol(newSymbol),
              dataType(newDataType)
        {}

        void Print(std::ostream& osm = std::cout) const;
        bool CompareParam(const SubfuncInvokeInfoTy::OutcastParamPackTy& esgParam) const;
    };

    struct TensorParamTy {
        int paramLoc;
        int operandIdx;
        int symDDRId;
        Offset symOffset;
        Shape shape;
        std::string symName;
        std::string symbol;
        DataType dataType;

        TensorParamTy(const int newOperandIdx, const int newSymDDRId, const std::vector<int64_t>& newShape,
                      const std::vector<int64_t>& newOffset, const std::string& newSymName, const int newParamLoc,
                      const std::string newSymbol = "", const DataType newDataType = DataType::DT_BOTTOM)
            : paramLoc(newParamLoc),
              operandIdx(newOperandIdx),
              symDDRId(newSymDDRId),
              symOffset(newOffset),
              shape(newShape),
              symName(newSymName),
              symbol(newSymbol),
              dataType(newDataType)
        {}

        void Print(std::ostream& osm = std::cout) const;
        bool CompareParam(const SubfuncInvokeInfoTy::TensorParamPackTy& esgParam) const;
    };

    using OutCastParamListTy = std::vector<OutCastParamTy>;
    using InCastParamListTy = std::vector<InCastParamTy>;
    using TensorParamListTy = std::vector<TensorParamTy>;

public:
    void AppendIncastParam(const int operandIdx, const int symDDRId, const std::vector<int64_t>& shape,
                           const std::vector<int64_t>& offset, const std::string& symName, const int paramLoc,
                           const std::string& symbol, const DataType dataType)
    {
        inCastArgs_.emplace_back(
            InCastParamTy(operandIdx, symDDRId, shape, offset, symName, paramLoc, symbol, dataType));
    }

    void AppendOutcastParam(const int operandIdx, const int symDDRId, const int refCount,
                            const std::vector<int64_t>& shape, const std::vector<int64_t>& offset,
                            const std::string& symName, const int paramLoc, const std::string& symbol,
                            const DataType dataType)
    {
        outCastArgs_.emplace_back(
            OutCastParamTy(operandIdx, symDDRId, refCount, shape, offset, symName, paramLoc, symbol, dataType));
    }

    void AppendTensorParam(const int operandIdx, const int symDDRId, const std::vector<int64_t>& shape,
                           const std::vector<int64_t>& offset, const std::string& symName, const int paramLoc,
                           const std::string& symbol, const DataType dataType)
    {
        tensorsArgs_.emplace_back(
            TensorParamTy(operandIdx, symDDRId, shape, offset, symName, paramLoc, symbol, dataType));
    }

    void Finalize() { isFinalized_ = true; }

    void PrettyPrint(const int psgId, std::ostream& osm = std::cout) const;
    Json ToJson() const;
    void FromJson(const Json& params);

public:
    TensorParamListTy tensorsArgs_;
    InCastParamListTy inCastArgs_;
    OutCastParamListTy outCastArgs_;
    bool isFinalized_ = false;
};

class SubfuncTopologyInfoTy {
    struct Entry {
        int esgId;
        int readyState;
        setType outGraph;
        uint32_t extType{0};
        uint32_t extParamNum{0};
        std::vector<int64_t> extParams;
    };

public:
    void SetTableSize(const int n) { topology_.reserve(n); }

    const std::vector<Entry>& GetTopology() const { return topology_; }

    void SetMaxM(const int maxM) { maxM_ = maxM; }

    void AddEntry(const int esgId, const int readState, const setType& succ);

    void UpdateEntry(const uint32_t extType, const uint32_t extParamNum, const std::vector<int64_t>& extParams);

    std::vector<int> TopoSort();

    void Print(std::ostream& osm = std::cout) const;

    Status DumpEachEntryInfo(int esgId, CoreType coreType, int64_t entryOffset, int64_t* entryParamPtr,
                             size_t entryParamTotalBytes, int32_t* readyStatePtr, size_t readyStateTotalSize) const;

    bool IsEsgReady(const int esgId) const;

    std::vector<int> GetSuccs(int esgId) const;

    Json DumpJson() const;
    void LoadJson(const Json& topoJson);

public:
    int maxM_;
    std::vector<Entry> topology_;
    std::vector<int> readyIds_;
};
} // namespace npu::tile_fwk
