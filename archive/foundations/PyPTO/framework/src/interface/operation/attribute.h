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
 * \file attribute.h
 * \brief
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <optional>
#include "interface/inner/hash_buffer.h"
#include "tilefwk/error.h"
#include "tilefwk/data_type.h"
#include "interface/cache/hash.h"
#include "interface/cache/hash.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class SubfuncInvokeInfoTy;

constexpr int32_t VALUE1 = 1;
constexpr int32_t VALUE3 = 3;
class OpAttribute {
public:
    virtual std::string Dump() const = 0;
    virtual Json DumpDynJson() = 0;
    virtual void LoadJson(const Json& attrJson) = 0;
    virtual std::shared_ptr<OpAttribute> Clone() const = 0;
    virtual void CollectAttrs(std::map<std::string, std::any>&) const {}

    virtual ~OpAttribute() = default;
};

template <typename T>
std::shared_ptr<OpAttribute> DeserializeFrom(const Json& attrJson, Function* function = nullptr)
{
    return std::static_pointer_cast<OpAttribute>(T::DeserializeFrom(attrJson, function));
}

class OpImmediate {
public:
    enum class OpImmediateKind {
        T_SCALAR_INVALID,
        T_SCALAR_SPECIFIED,
        T_SCALAR_PARAMETER,
    };

    OpImmediate() : kind_(OpImmediateKind::T_SCALAR_INVALID), specifiedValue_(-1), parameterIndex_(-1) {}
    OpImmediate(OpImmediateKind kind, const SymbolicScalar& data);
    OpImmediate(OpImmediateKind kind, int data) : OpImmediate(kind, SymbolicScalar(data)) {}
    explicit OpImmediate(const SymbolicScalar& value) : OpImmediate(OpImmediateKind::T_SCALAR_SPECIFIED, value) {}
    OpImmediate(const OpImmediate&) = default;
    OpImmediate& operator=(const OpImmediate&) = default;

    static OpImmediate Specified(const SymbolicScalar& value)
    {
        return OpImmediate(OpImmediateKind::T_SCALAR_SPECIFIED, value);
    }
    static std::vector<OpImmediate> Specified(const std::initializer_list<int64_t>& init)
    {
        std::vector<int64_t> value(init);
        return Specified(value);
    }
    static std::vector<OpImmediate> Specified(const std::vector<int64_t>& value)
    {
        std::vector<OpImmediate> res;
        for (auto& v : value) {
            res.push_back(Specified(SymbolicScalar(v)));
        }
        return res;
    }
    static std::vector<OpImmediate> Specified(const std::vector<SymbolicScalar>& value)
    {
        std::vector<OpImmediate> res;
        for (auto& v : value) {
            res.push_back(OpImmediate(v));
        }
        return res;
    }

    static std::vector<OpImmediate> Specified(const TensorOffset& offset)
    {
        std::vector<OpImmediate> result;
        if (offset.GetDynOffset().size() != 0) {
            result.resize(offset.GetDynOffset().size());
            std::transform(offset.GetDynOffset().begin(), offset.GetDynOffset().end(), result.begin(),
                           [](const SymbolicScalar& scalar) { return OpImmediate(scalar); });
        } else {
            result.resize(offset.GetOffset().size());
            std::transform(offset.GetOffset().begin(), offset.GetOffset().end(), result.begin(),
                           [](const int value) { return OpImmediate(SymbolicScalar(value)); });
        }
        return result;
    }

    static OpImmediate Parameter(int index) { return OpImmediate(OpImmediateKind::T_SCALAR_PARAMETER, index); }

    static std::vector<SymbolicScalar> ToSpecified(const std::vector<OpImmediate>& value)
    {
        std::vector<SymbolicScalar> res;
        for (auto& v : value) {
            res.push_back(v.GetSpecifiedValue());
        }
        return res;
    }

    static void NormalizeValue(SymbolicScalar& arg, OpImmediate& opImm, const SymbolicScalar& normCall,
                               bool valueToIndex)
    {
        FE_ASSERT(opImm.IsSpecified());
        SymbolicScalar value = opImm.GetSpecifiedValue();
        if (valueToIndex) {
            opImm = OpImmediate::Specified(normCall);
        }
        arg = value;
    }

    static void NormalizeValue(std::vector<SymbolicScalar>& operandCoaList, int operandCoaIndex,
                               std::vector<OpImmediate>& opImmList, int coaIndex, bool valueToIndex)
    {
        int offset = 0;
        for (auto& op : opImmList) {
            FE_ASSERT(op.IsSpecified());
            SymbolicScalar value = op.GetSpecifiedValue();
            if (valueToIndex) {
                op = OpImmediate::Parameter(coaIndex + offset);
            }
            operandCoaList[operandCoaIndex + offset] = value;
            offset++;
        }
    }

    static bool HasSymbolic(const std::vector<OpImmediate>& offsetList)
    {
        for (auto& offset : offsetList) {
            if (offset.IsParameter()) {
                continue;
            }
            const SymbolicScalar& ss = offset.GetSpecifiedValue();
            if (!ss.IsImmediate()) {
                return true;
            }
        }
        return false;
    }

    bool IsSpecified() const { return kind_ == OpImmediateKind::T_SCALAR_SPECIFIED; }
    bool IsParameter() const { return kind_ == OpImmediateKind::T_SCALAR_PARAMETER; }

    const SymbolicScalar& GetSpecifiedValue() const
    {
        FE_ASSERT(IsSpecified());
        return specifiedValue_;
    }
    SymbolicScalar& GetSpecifiedValue()
    {
        FE_ASSERT(IsSpecified());
        return specifiedValue_;
    }

    int GetParameterIndex() const
    {
        FE_ASSERT(IsParameter());
        return parameterIndex_;
    }

    OpImmediateKind Kind() const { return kind_; }

    OpImmediate operator+(const OpImmediate& rhs) const
    {
        FE_ASSERT(IsSpecified() && rhs.IsSpecified());
        return Specified(GetSpecifiedValue() + rhs.GetSpecifiedValue());
    }

    OpImmediate& operator-=(int offset)
    {
        FE_ASSERT(IsSpecified());
        specifiedValue_ = specifiedValue_ - offset;
        return *this;
    }

    std::string Dump() const;
    Json DumpDynJson();
    static OpImmediate DeserializeFrom(const Json& attrJson, size_t& despos);

private:
    OpImmediateKind kind_{OpImmediateKind::T_SCALAR_INVALID};
    SymbolicScalar specifiedValue_{-1};
    int parameterIndex_{-1};
};

class ViewOpAttribute : public OpAttribute {
public:
    explicit ViewOpAttribute(const Offset& fromOffset, const std::vector<SymbolicScalar>& fromDynOffset = {})
        : ViewOpAttribute(fromOffset, MemoryType::MEM_UNKNOWN, fromDynOffset)
    {}
    explicit ViewOpAttribute(const Offset& fromOffset, const std::vector<SymbolicScalar>& fromDynOffset,
                             const std::vector<SymbolicScalar>& toDynValidShape)
        : ViewOpAttribute(fromOffset, MEM_UNKNOWN, fromDynOffset, toDynValidShape)
    {}
    ViewOpAttribute(const Offset& fromOffset, MemoryType to, const std::vector<SymbolicScalar>& fromDynOffset = {},
                    const std::vector<SymbolicScalar>& toDynValidShape = {})
        : to_(to), fromOffset_(fromOffset), fromDynOffset_(fromDynOffset), toDynValidShape_(toDynValidShape)
    {}

    std::string Dump() const override;
    Json DumpDynJson() override;
    void LoadJson([[maybe_unused]] const Json& attrJson) override {};
    virtual std::shared_ptr<OpAttribute> Clone() const override;

    void SetToType(MemoryType to) { to_ = to; }
    MemoryType GetTo() const { return to_; }
    auto& GetFrom() { return fromOffset_; }
    const auto& GetFromOffset() const { return fromOffset_; }
    auto& GetFromOffset() { return fromOffset_; }
    const auto& GetFromDynOffset() const { return fromDynOffset_; }
    auto& GetFromDynOffset() { return fromDynOffset_; }
    void SetFromOffset(const Offset& fromOffset, const std::vector<SymbolicScalar>& fromDynOffset = {})
    {
        fromOffset_ = fromOffset;
        if (fromDynOffset.empty()) {
            fromDynOffset_ = OpImmediate::ToSpecified(OpImmediate::Specified(fromOffset));
        } else {
            fromDynOffset_ = fromDynOffset;
        }
    }

    TensorOffset GetFromTensorOffset() const { return TensorOffset(fromOffset_, fromDynOffset_); }

    auto& GetToDynValidShape() { return toDynValidShape_; }

    void SetToDynValidShape(const std::vector<SymbolicScalar>& toDynValidShape) { toDynValidShape_ = toDynValidShape; }

    static std::shared_ptr<ViewOpAttribute> DeserializeFrom(const Json& attrJson, [[maybe_unused]] Function* function);

    void CollectAttrs(std::map<std::string, std::any>& attrs) const override
    {
        if (!fromDynOffset_.empty()) {
            attrs["fromOffset"] = fromDynOffset_;
        } else {
            attrs["fromOffset"] = fromOffset_;
        }
        if (!toDynValidShape_.empty()) {
            attrs["toValidShape"] = toDynValidShape_;
        }
    }

private:
    MemoryType to_;
    Offset fromOffset_;
    std::vector<SymbolicScalar> fromDynOffset_;
    std::vector<SymbolicScalar> toDynValidShape_;
};

class AssembleOpAttribute : public OpAttribute {
public:
    explicit AssembleOpAttribute(const Offset& toOffset, const std::vector<SymbolicScalar>& toDynOffset = {})
        : AssembleOpAttribute(MemoryType::MEM_UNKNOWN, toOffset, toDynOffset)
    {}
    AssembleOpAttribute(MemoryType from, const Offset& toOffset, const std::vector<SymbolicScalar>& toDynOffset = {},
                        const std::vector<SymbolicScalar>& fromDynValidShape = {})
        : from_(from), toOffset_(toOffset), toDynOffset_(toDynOffset), fromDynValidShape_(fromDynValidShape)
    {}

    std::string Dump() const override;
    Json DumpDynJson() override;
    void LoadJson([[maybe_unused]] const Json& attrJson) override {};
    virtual std::shared_ptr<OpAttribute> Clone() const override;

    void SetFromType(MemoryType from) { from_ = from; }
    MemoryType GetFrom() const { return from_; }
    const auto& GetToOffset() const { return toOffset_; }
    auto& GetToOffset() { return toOffset_; }
    const auto& GetToDynOffset() const { return toDynOffset_; }
    auto& GetToDynOffset() { return toDynOffset_; }
    void SetToOffset(const Offset& toOffset, const std::vector<SymbolicScalar>& toDynOffset = {})
    {
        toOffset_ = toOffset;
        toDynOffset_ = toDynOffset;
    }
    auto& GetFromDynValidShape() { return fromDynValidShape_; }
    void SetFromDynValidShape(std::vector<SymbolicScalar>& fromDynValidShape)
    {
        fromDynValidShape_ = std::move(fromDynValidShape);
    }

    TensorOffset GetToTensorOffset() const { return TensorOffset(toOffset_, toDynOffset_); }

    static std::shared_ptr<AssembleOpAttribute> DeserializeFrom(const Json& attrJson,
                                                                [[maybe_unused]] Function* function);

    void CollectAttrs(std::map<std::string, std::any>& attrs) const override
    {
        if (!toDynOffset_.empty()) {
            attrs["toOffset"] = toDynOffset_;
        } else {
            attrs["toOffset"] = toOffset_;
        }
        if (!fromDynValidShape_.empty()) {
            attrs["fromValidShape"] = fromDynValidShape_;
        }
    }

private:
    MemoryType from_;
    Offset toOffset_;
    std::vector<SymbolicScalar> toDynOffset_;
    std::vector<SymbolicScalar> fromDynValidShape_;
};

/*
 * argList[i]
 * [0]: rawTensorIndex
 * [1-dim]: offset
 * [dim+1, 2*dim]: shape
 * [2*dim+1, 3*dim]: rawshape
 * [3*dim+1, 4*dim]: validshape
 *
 * linearArgList:
 * [0]: cceIndex
 * [1 ... 1 + argList[0].size() - 1]: argList[0]
 * [1 + argList[0].size() ... 1 + argList[0].size() + argList[1].size() - 1]: argList[1]
 * ...
 */
constexpr int COA_INDEX_TYPE_OFFSET = 0;
constexpr int COA_INDEX_TYPE_SHAPE = 1;
constexpr int COA_INDEX_TYPE_RAWSHAPE = 2;
constexpr int COA_INDEX_TYPE_VALIDSHAPE = 3;
constexpr int COA_INDEX_TYPE_COUNT = 4;
constexpr int COA_INDEX_BASE = 1;
constexpr int COA_INDEX_DIM_BASE = 1;

class CallOpAttribute : public OpAttribute {
public:
    CallOpAttribute() = default;

    CallOpAttribute(const FunctionHash& calleeHash, const std::vector<std::vector<SymbolicScalar>>& argList,
                    const std::string& calleMagicName = "", const std::map<int, SymbolicScalar>& outIndexToExpr = {},
                    const std::vector<SymbolicScalar>& linearArgList = {});

    std::string Dump() const override;
    Json DumpDynJson() override;
    std::string DumpAttr(int idx = -1) const;
    Json DumpInvokeInfoJson();
    void LoadJson([[maybe_unused]] const Json& attrJson) override {};
    virtual std::shared_ptr<OpAttribute> Clone() const override;

    const std::string& GetCalleeMagicName() const { return calleMagicName_; }
    const std::string& GetCalleeBracketName() const { return calleeBracketName_; }
    int GetCalleeMagic() const { return calleeMagic_; }
    const FunctionHash& GetCalleeHash() const { return calleeHash_; }
    void SetCalleeMagicName(const std::string& magicName) { calleMagicName_ = magicName; }
    void SetCalleeHash(const FunctionHash& hash) { calleeHash_ = hash; }

    const std::vector<std::vector<SymbolicScalar>>& GetArgList() const { return argList_; }
    std::vector<std::vector<SymbolicScalar>>& GetArgList() { return argList_; }
    std::vector<SymbolicScalar>& GetLinearArgList();
    std::vector<int64_t> GetLinearImmediateArgList(int begin, int end, bool returnEmptyForSymbolic);
    std::optional<SymbolicScalar> GetOutcastSymbolicExpr(int index) const
    {
        auto it = outIndexToExpr_.find(index);
        if (it == outIndexToExpr_.end()) {
            return std::nullopt;
        }
        return std::make_optional(it->second);
    }
    std::map<int, SymbolicScalar>& GetOutCastIndexToExpr() { return outIndexToExpr_; }

    std::shared_ptr<SubfuncInvokeInfoTy> invokeInfo_;

    static std::shared_ptr<CallOpAttribute> DeserializeFrom(const Json& attrJson, [[maybe_unused]] Function* function);
    // coreTask被调度到哪个wrap
    int32_t wrapId{-1};

private:
    std::string calleMagicName_;
    std::string calleeBracketName_;
    int calleeMagic_ = 0;
    FunctionHash calleeHash_;
    std::vector<std::vector<SymbolicScalar>> argList_;
    std::vector<SymbolicScalar> linearArgList_;
    std::map<int, SymbolicScalar> outIndexToExpr_;
};

class ConvertOpAttribute : public OpAttribute {
public:
    ConvertOpAttribute(MemoryType from, MemoryType to) : from_(from), to_(to) {}

    std::string Dump() const override;
    void LoadJson([[maybe_unused]] const Json& attrJson) override {};
    virtual std::shared_ptr<OpAttribute> Clone() const override;
    Json DumpDynJson() override;
    std::pair<MemoryType, MemoryType> GetConvertPath() const;
    static std::shared_ptr<ConvertOpAttribute> DeserializeFrom(const Json& attrJson,
                                                               [[maybe_unused]] Function* function);

private:
    MemoryType from_;
    MemoryType to_;
};

struct GmOutOfRangeCheckInfo {
    enum class AccessType : int64_t { READ_GM = 0, WRITE_GM = 1 };
    OpImmediate oneDimOffset;
    OpImmediate oneDimExtent;
    OpImmediate totalSize;
    AccessType accessType{AccessType::READ_GM};
};

class CopyOpAttribute : public OpAttribute {
public:
    // Copy Out ops, from -> CopyOp -> DDR
    CopyOpAttribute(MemoryType from, std::vector<OpImmediate> toOffset, std::vector<OpImmediate> shape,
                    std::vector<OpImmediate> rawShape, std::vector<OpImmediate> fromDynValidShape = {})
        : CopyOpAttribute({OpImmediate(), OpImmediate()}, std::move(toOffset), {from, MemoryType::MEM_DEVICE_DDR},
                          std::move(shape), std::move(rawShape), {}, std::move(fromDynValidShape), true)
    {}
    // Copy In ops, DDR -> CopyOp -> to
    CopyOpAttribute(std::vector<OpImmediate> fromOffset, MemoryType to, std::vector<OpImmediate> shape,
                    std::vector<OpImmediate> rawShape, std::vector<OpImmediate> toDynValidShape = {})
        : CopyOpAttribute(std::move(fromOffset), {OpImmediate(), OpImmediate()}, {MemoryType::MEM_DEVICE_DDR, to},
                          std::move(shape), std::move(rawShape), std::move(toDynValidShape), {}, false)
    {}

    [[nodiscard]] std::string Dump() const override;
    void LoadJson([[maybe_unused]] const Json& attrJson) override {};
    virtual std::shared_ptr<OpAttribute> Clone() const override;

    void SetFromOffset(std::vector<OpImmediate> fromOffset) { fromOffset_ = std::move(fromOffset); }
    void SetToOffset(std::vector<OpImmediate> toOffset) { toOffset_ = std::move(toOffset); }
    const std::vector<OpImmediate>& GetFromOffset() const { return fromOffset_; }
    std::vector<OpImmediate>& GetFromOffset() { return fromOffset_; }
    const std::vector<OpImmediate>& GetToOffset() const { return toOffset_; }
    std::vector<OpImmediate>& GetToOffset() { return toOffset_; }

    [[nodiscard]] std::pair<MemoryType, std::vector<OpImmediate>> GetCopyOutAttr() const;
    [[nodiscard]] std::pair<std::vector<OpImmediate>, MemoryType> GetCopyInAttr() const;
    [[nodiscard]] bool IsCopyOut() const { return isCopyOut_; }
    [[nodiscard]] const std::vector<OpImmediate>& GetShape() const { return tensorShape_; }
    [[nodiscard]] const std::vector<OpImmediate>& GetRawShape() const { return rawShape_; }
    [[nodiscard]] std::vector<OpImmediate>& GetRawShape() { return rawShape_; }
    [[nodiscard]] const std::vector<OpImmediate>& GetToDynValidShape() const { return toDynValidShape_; }
    [[nodiscard]] std::vector<OpImmediate>& GetToDynValidShape() { return toDynValidShape_; }
    [[nodiscard]] const std::vector<OpImmediate>& GetFromDynValidShape() const { return fromDynValidShape_; }
    [[nodiscard]] std::vector<OpImmediate>& GetFromDynValidShape() { return fromDynValidShape_; }
    [[nodiscard]] std::vector<int64_t> GetSpecifiedShape(int64_t defaultValue) const;
    void SetGmOutOfRangeCheck(const GmOutOfRangeCheckInfo& info) { gmOutOfRangeCheck_ = info; }
    const GmOutOfRangeCheckInfo* GetGmOutOfRangeCheck() const
    {
        return gmOutOfRangeCheck_.has_value() ? &gmOutOfRangeCheck_.value() : nullptr;
    }
    void SetShape(std::vector<OpImmediate> shape) { tensorShape_ = std::move(shape); }
    void SetRawShape(std::vector<OpImmediate> rawShape) { rawShape_ = std::move(rawShape); }
    void SetToDynValidShape(std::vector<OpImmediate> toDynValidShape) { toDynValidShape_ = std::move(toDynValidShape); }
    void SetFromDynValidShape(std::vector<OpImmediate> fromDynValidShape)
    {
        fromDynValidShape_ = std::move(fromDynValidShape);
    }
    bool IsDynToOffset() const;
    bool IsDynOffset(const std::vector<OpImmediate>& offset) const;
    bool IsDynFromOffset() const { return IsDynOffset(fromOffset_); }

    Json DumpDynJson() override;

    static std::shared_ptr<CopyOpAttribute> DeserializeFrom(const Json& attrJson, [[maybe_unused]] Function* function);

private:
    CopyOpAttribute(std::vector<OpImmediate> fromOffset, std::vector<OpImmediate> toOffset,
                    std::pair<MemoryType, MemoryType> memTypes, std::vector<OpImmediate> tensorShape,
                    std::vector<OpImmediate> rawShape, std::vector<OpImmediate> toDynValidShape,
                    std::vector<OpImmediate> fromDynValidShape, bool isCopyOut)
        : from_(memTypes.first),
          to_(memTypes.second),
          fromOffset_(std::move(fromOffset)),
          toOffset_(std::move(toOffset)),
          tensorShape_(std::move(tensorShape)),
          rawShape_(std::move(rawShape)),
          toDynValidShape_(std::move(toDynValidShape)),
          fromDynValidShape_(std::move(fromDynValidShape)),
          isCopyOut_(isCopyOut),
          gmOutOfRangeCheck_(std::nullopt)
    {}

    MemoryType from_;
    MemoryType to_;
    std::vector<OpImmediate> fromOffset_;
    std::vector<OpImmediate> toOffset_;
    std::vector<OpImmediate> tensorShape_;
    std::vector<OpImmediate> rawShape_;
    std::vector<OpImmediate> toDynValidShape_;
    std::vector<OpImmediate> fromDynValidShape_;
    OpImmediate baseAddr_;
    bool isCopyOut_;
    std::optional<GmOutOfRangeCheckInfo> gmOutOfRangeCheck_;
};

} // namespace npu::tile_fwk
