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
 * \file logical_tensor.h
 * \brief
 */

#pragma once
#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <unordered_set>
#include <functional>
#include "tilefwk/data_type.h"
#include "interface/inner/pre_def.h"
#include "raw_tensor.h"
#include "interface/operation/attr_holder.h"
#include "symbolic_scalar.h"
#include "tensor_offset.h"

#include <nlohmann/json.hpp>
#include "storage.h"
#include "ir/expr.h"

using Json = nlohmann::json;
using namespace pypto;

namespace npu::tile_fwk {
class TileRange {
public:
    size_t start;
    size_t end; // exclusive end point
    int lifeStart = -1;
    int lifeEnd = -1;
    int memId = -1; // default = tensor->raw_id

    explicit TileRange(size_t s = 0, size_t e = 0, int id = -1) : start(s), end(e), memId(id) {}

    // Helper functions
    size_t Size() const { return end - start; }
    bool IsEmpty() const { return end <= start; }
};

class LogicalTensor : public std::enable_shared_from_this<LogicalTensor>, public AttrHolder, public ir::Var {
public:
    std::shared_ptr<RawTensor> tensor;
    Offset offset;
    Shape shape;
    Shape oriShape;
    std::vector<SymbolicScalar> dynOffset_;
    std::vector<SymbolicScalar> dynValidShape_;

    Shape storageShape;
    std::shared_ptr<Storage> storage_ = nullptr;
    uint64_t storageOffset_ = 0;
    int magic;

    TileRange memoryrange;

    LogicalTensor(Function& function, DataType t, Shape tshape, TileOpFormat tformat = TileOpFormat::TILEOP_ND,
                  std::string tname = "");
    LogicalTensor(Function& function, DataType t, Shape tshape, std::vector<SymbolicScalar> tValidShape,
                  TileOpFormat tformat = TileOpFormat::TILEOP_ND, std::string tname = "");
    LogicalTensor(Function& function, std::shared_ptr<RawTensor> rawTensor, Offset toffset, Shape tshape);
    LogicalTensor(Function& function, std::shared_ptr<RawTensor> rawTensor, Offset toffset, Shape tshape,
                  std::vector<SymbolicScalar> tValidShape);

    LogicalTensor(LogicalTensor&&) = delete;
    LogicalTensor(const LogicalTensor&) = delete;
    LogicalTensor& operator=(LogicalTensor&&) = delete;
    LogicalTensor& operator=(const LogicalTensor&) = delete;

    LogicalTensorPtr Clone(Function& dstFunc, bool create = false) const;
    LogicalTensorPtr NextVersion(Function& func, std::vector<ir::VarPtr>& tokens) const;

    ir::VarPtr Clone() const override;
    ir::VarPtr Clone(bool shareRawTensor) const;

    Function& BelongFunction() { return *function_; }
    const Function& BelongFunction() const { return *function_; }
    void UpdateBelongFunction(Function& newFunc);

    std::string DumpType() const;

    /* By default, RawTensor is dumped. In whole function dumping, we only dump the magic */
    Json DumpJson(Function& func, bool dumpRawTensor = true) const;
    static std::shared_ptr<LogicalTensor> LoadJson(
        Function& function, const std::unordered_map<int, std::shared_ptr<RawTensor>>& rawTensorDict,
        const Json& tensorDump);

    std::string DumpSSA(bool showFrom = true, bool showMem = false, bool showType = true) const;

    std::string Dump(bool showFrom = true, bool showMem = false) const;

    std::shared_ptr<LogicalTensor> View(Function& function, const Shape& newShape, const Offset& newOffset) const;

    DataType Datatype() const;
    std::string Symbol() const;
    TileOpFormat Format() const { return tensor->format; }

    MemoryType GetMemoryTypeOriginal() const;
    MemoryType GetMemoryTypeToBe() const;
    void CopyMemoryType(const std::shared_ptr<LogicalTensor>& other);
    void SetMemoryTypeBoth(MemoryType t, bool force = false);
    void SetMemoryTypeOriginal(MemoryType t, bool force = false);
    void SetMemoryTypeToBe(MemoryType t);
    bool MemoryConflict() const;
    size_t MemorySize() const;
    bool IsDummy() const;
    void SetIsDummy(bool dummy = true);

    bool Overlap(const std::shared_ptr<LogicalTensor>& other) const;

    int GetMagic() const { return magic; }
    void SetMagic(int m) { magic = m; }
    int GetRawMagic() const { return tensor->GetRawMagic(); }
    std::shared_ptr<RawTensor> GetRawTensor() const { return tensor; }
    ir::VarPtr GetReadToken() const { return readToken_; }
    ir::VarPtr GetWriteToken() const { return writeToken_; }
    void SetReadToken(const ir::VarPtr& token) { readToken_ = token; }
    void SetWriteToken(const ir::VarPtr& token) { writeToken_ = token; }
    const Offset& GetOffset() const { return offset; }
    const Shape& GetShape() const { return shape; }
    void UpdateOffset(const Offset& newOffset)
    {
        FE_ASSERT(FeError::INVALID_VAL, newOffset.size() == shape.size())
            << "newOffset.size(): " << newOffset.size() << ", shape.size(): " << shape.size();
        offset = newOffset;
    }
    void UpdateOffset(const TensorOffset& tensorOffset)
    {
        this->offset = tensorOffset.GetOffset();
        this->dynOffset_ = tensorOffset.GetDynOffset();
    }
    const TensorOffset GetTensorOffset() const { return TensorOffset(offset, dynOffset_); }
    void UpdateDynValidShape(const std::vector<SymbolicScalar>& dynValidShape) { dynValidShape_ = dynValidShape; }
    struct CompareOp {
        bool operator()(const Operation* a, const Operation* b) const;
    };

    auto& GetProducers() { return producers_; }
    auto& GetConsumers() { return consumers_; }
    auto& GetDependOps() { return dependOps_; }
    const auto& GetProducers() const { return producers_; }
    const auto& GetConsumers() const { return consumers_; }
    const auto& GetDependOps() const { return dependOps_; }
    bool HasProducer(Operation* operation) const { return producers_.count(operation) > 0; }
    bool HasConsumer(Operation* operation) const { return consumers_.count(operation) > 0; }
    bool HasDependOp(Operation* operation) const { return dependOps_.count(operation) > 0; }
    bool HasProducer(Operation& operation) const { return HasProducer(&operation); }
    bool HasConsumer(Operation& operation) const { return HasConsumer(&operation); }
    bool HasDependOp(Operation& operation) const { return HasDependOp(&operation); }
    void AddProducer(Operation* operation) { producers_.emplace(operation); }
    void AddConsumer(Operation* operation) { consumers_.emplace(operation); }
    void AddDependOp(Operation* operation) { dependOps_.emplace(operation); }
    void RemoveProducer(Operation* operation) { producers_.erase(operation); }
    void RemoveConsumer(Operation* operation) { consumers_.erase(operation); }
    void RemoveDependOp(Operation* operation) { dependOps_.erase(operation); }
    void AddProducer(Operation& operation) { AddProducer(&operation); }
    void AddConsumer(Operation& operation) { AddConsumer(&operation); }
    void AddDependOp(Operation& operation) { AddDependOp(&operation); }
    void RemoveProducer(Operation& operation) { RemoveProducer(&operation); }
    void RemoveConsumer(Operation& operation) { RemoveConsumer(&operation); }
    void RemoveDependOp(Operation& operation) { RemoveDependOp(&operation); }
    void ClearAllProducers() { producers_.clear(); }

    void operator<<(LogicalTensor& right);

    int64_t GetDataSize() const;

    bool IsOffsetAllZero() const
    {
        return std::all_of(offset.begin(), offset.end(), [](int value) { return value == 0; });
    }

    const std::vector<SymbolicScalar>& GetDynOffset() const { return dynOffset_; }
    const std::vector<SymbolicScalar>& GetDynValidShape() const { return dynValidShape_; }
    std::vector<SymbolicScalar>& GetDynValidShape() { return dynValidShape_; }

    void SetCachePolicy(CachePolicy policy, bool value)
    {
        if (tensor != nullptr) {
            tensor->SetCachePolicy(policy, value);
        }
    }
    bool GetCachePolicy(CachePolicy policy) const
    {
        if (tensor != nullptr) {
            return tensor->GetCachePolicy(policy);
        }
        return false;
    }

    bool IsGetTensorDataOutcast();

private:
    MemoryType memoryTypeOriginal_{MemoryType::MEM_UNKNOWN};
    MemoryType memoryTypeToBe_{MemoryType::MEM_UNKNOWN};
    int readyTime_{INVALID_TIME};
    int remainingTime_{INVALID_TIME};
    Function* function_;
    ir::VarPtr readToken_;
    ir::VarPtr writeToken_;

    std::set<Operation*, CompareOp> producers_;
    std::set<Operation*, CompareOp> consumers_;
    std::set<Operation*, CompareOp> dependOps_;
};

enum EmuOpcode : int64_t {
    EMUOP_TENSOR_EXTRACT,
    EMUOP_TENSOR_INSERT,
    EMUOP_TENSOR_GETDATA_DEPEND,
    EMUOP_TENSOR_GETDATA_IMPORT,
    EMUOP_TENSOR_SETDATA,
};

bool CheckEmuOpcode(const Operation* op, EmuOpcode opcode);
void SetEmuOpcode(Operation* op, EmuOpcode opcode);

bool TypeEqual(const LogicalTensorPtr a, const LogicalTensorPtr b);

Tensor TensorExtract(const Tensor& src, const std::vector<SymbolicScalar>& offset);
void TensorInsert(const Tensor& src, const std::vector<SymbolicScalar>& offset, Tensor& dst);

SymbolicScalar GetViewValidShapeDim(const SymbolicScalar& validShapeDim, const SymbolicScalar& viewOffsetDim,
                                    const SymbolicScalar& viewShapeDim);
std::vector<SymbolicScalar> GetViewValidShape(const std::vector<SymbolicScalar>& validShape, const Offset& viewOffset,
                                              const std::vector<SymbolicScalar>& viewDynOffset, const Shape& viewShape);

std::map<int, std::vector<RawSymbolicScalarPtr>> GetTensorDataDict(const SymbolicScalar& dimOffset);
std::map<int, std::vector<RawSymbolicScalarPtr>> GetTensorDataDict(const std::vector<SymbolicScalar>& offset);
std::map<int, std::vector<RawSymbolicScalarPtr>> GetTensorDataDict(
    const std::vector<std::reference_wrapper<SymbolicScalar>>& offset);

struct GetTensorDataIODesc {
    int ioType{-1};
    int ioTypeIndex{-1};
    // encode both incast & outcast
    SymbolicScalar address;
    GetTensorDataIODesc() = default;
    GetTensorDataIODesc(int ioType_, int ioTypeIndex_, SymbolicScalar address_)
        : ioType(ioType_), ioTypeIndex(ioTypeIndex_), address(address_)
    {}
};

int GetTensorDataGetIndex(const Operation* op);
void GetTensorDataSetIndex(Operation* op, int index);

int GetTensorDataGetCoaIndex(const Operation* op);
void GetTensorDataSetCoaIndex(Operation* op, int index);

struct GetTensorDataIODescDict : std::unordered_map<int, GetTensorDataIODesc> {
    std::string Dump() const;
};
constexpr int GET_TENSOR_DATA_OPERAND_INDEX_CALLEE = 0;
constexpr int GET_TENSOR_DATA_OPERAND_INDEX_INDEX = 1;
constexpr int GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE = 2;
constexpr int GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX = 3;
constexpr int GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS = 4;
constexpr int GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST = 0;
constexpr int GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST = 1;
SymbolicScalar GetTensorDataFillIO(const GetTensorDataIODescDict& iodescDict, const SymbolicScalar& dimOffset);
SymbolicScalar UpdateGetTensorDataIOIndex(size_t currOutcastIdx, size_t newOutcastIdx, const SymbolicScalar& scalar);
std::set<std::pair<int, int>> GetTensorDataUsage(const std::vector<std::reference_wrapper<SymbolicScalar>>& scalars);

constexpr int RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_SIZE_INDEX = 1;
constexpr int RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX = 2;
constexpr int RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX = 3;

} // namespace npu::tile_fwk
