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
 * \file attribute.cpp
 * \brief
 */

#include <sstream>
#include <iomanip>
#include "attribute.h"
#include "passes/pass_utils/subfunc_utils.h"

using namespace npu::tile_fwk;

namespace {
constexpr int32_t WIDTH = 3;
}

OpImmediate::OpImmediate(OpImmediateKind kind, const SymbolicScalar& data) : kind_(kind)
{
    switch (kind) {
        case OpImmediateKind::T_SCALAR_SPECIFIED:
            specifiedValue_ = data;
            break;
        case OpImmediateKind::T_SCALAR_PARAMETER:
            parameterIndex_ = data;
            break;
        default:
            specifiedValue_ = SymbolicScalar(-1);
            parameterIndex_ = SymbolicScalar(-1);
            break;
    }
}

std::string OpImmediate::Dump() const
{
    std::string result;
    switch (kind_) {
        case OpImmediateKind::T_SCALAR_SPECIFIED:
            result = specifiedValue_.Dump();
            break;
        case OpImmediateKind::T_SCALAR_PARAMETER:
            result = std::to_string(parameterIndex_) + "(index)";
            break;
        default:
            break;
    }
    return result;
}

Json OpImmediate::DumpDynJson()
{
    Json res = Json::array();
    res.push_back(static_cast<int>(kind_));
    switch (kind_) {
        case OpImmediateKind::T_SCALAR_SPECIFIED: {
            SymbolicScalar value = specifiedValue_;
            res.push_back(ToJson(value));
        } break;
        case OpImmediateKind::T_SCALAR_PARAMETER: {
            res.push_back(parameterIndex_);
        } break;
        default:
            break;
    }
    return res;
}

OpImmediate OpImmediate::DeserializeFrom(const Json& attrJson, size_t& despos)
{
    OpImmediate result;
    switch (static_cast<OpImmediateKind>(attrJson[despos++])) {
        case OpImmediateKind::T_SCALAR_SPECIFIED: {
            auto symbolicJson = attrJson[despos++];
            if (symbolicJson.is_number()) {
                ScalarImmediateType immediateNum = static_cast<ScalarImmediateType>(symbolicJson);
                result = OpImmediate::Specified(SymbolicScalar(immediateNum));
            } else {
                SymbolicScalar sym = LoadSymbolicScalar(symbolicJson);
                result = OpImmediate::Specified(sym);
            }
        } break;
        case OpImmediateKind::T_SCALAR_PARAMETER: {
            ScalarImmediateType immediateNum = static_cast<ScalarImmediateType>(attrJson[despos++]);
            result = OpImmediate::Parameter(SymbolicScalar(immediateNum));
        } break;
        default:
            break;
    }
    return result;
}

std::string ViewOpAttribute::Dump() const
{
    std::stringstream ss;
    ss << "from offset:[";
    for (size_t i = 0; i < fromOffset_.size(); i++) {
        if (i != 0) {
            ss << ",";
        }
        ss << std::setw(WIDTH) << fromOffset_[i];
    }
    ss << "]";
    if (!fromDynOffset_.empty()) {
        ss << " dynoffset:[";
        for (size_t i = 0; i < fromDynOffset_.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << fromDynOffset_[i].Dump();
        }
        ss << "]";
    }
    ss << " to " << MemoryTypeToString(to_);
    if (!toDynValidShape_.empty()) {
        ss << " dynvalidshape:[";
        for (size_t i = 0; i < toDynValidShape_.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << toDynValidShape_[i].Dump();
        }
        ss << "]";
    }

    return ss.str();
}

Json ViewOpAttribute::DumpDynJson()
{
    Json res = Json::array();
    res.push_back(static_cast<int>(to_));
    res.push_back(static_cast<int32_t>(fromOffset_.size()));
    for (auto offset : fromOffset_) {
        res.push_back(offset);
    }
    res.push_back(static_cast<int32_t>(fromDynOffset_.size()));
    for (auto offset : fromDynOffset_) {
        auto joffset = ToJson(offset);
        if (joffset.size() > 0) {
            res.push_back(joffset);
        }
    }
    res.push_back(static_cast<int32_t>(toDynValidShape_.size()));
    for (auto shape : toDynValidShape_) {
        auto jshape = ToJson(shape);
        if (jshape.size() > 0) {
            res.push_back(jshape);
        }
    }
    return res;
}

std::shared_ptr<ViewOpAttribute> ViewOpAttribute::DeserializeFrom(const Json& attrJson,
                                                                  [[maybe_unused]] Function* function)
{
    int despos = 0;
    auto memType = attrJson[despos++];
    int offsetSize = attrJson[despos++];
    std::vector<int64_t> fromOffset;
    for (int i = 0; i < offsetSize; i++) {
        fromOffset.push_back(attrJson[despos++]);
    }
    int dynOffsetSize = attrJson[despos++];
    std::vector<SymbolicScalar> fromDynOffset;
    for (int i = 0; i < dynOffsetSize; i++) {
        fromDynOffset.push_back(LoadSymbolicScalar(attrJson[despos++]));
    }
    int dynValidShapeSize = attrJson[despos++];
    std::vector<SymbolicScalar> toDynValidShape;
    for (int i = 0; i < dynValidShapeSize; i++) {
        toDynValidShape.push_back(LoadSymbolicScalar(attrJson[despos++]));
    }
    return std::make_shared<ViewOpAttribute>(fromOffset, static_cast<MemoryType>(memType), fromDynOffset,
                                             toDynValidShape);
}

std::shared_ptr<OpAttribute> ViewOpAttribute::Clone() const
{
    return std::make_shared<ViewOpAttribute>(fromOffset_, to_, fromDynOffset_, toDynValidShape_);
}

std::string AssembleOpAttribute::Dump() const
{
    FE_ASSERT(!toOffset_.empty());
    std::stringstream ss;
    ss << "from " << MemoryTypeToString(from_);
    if (!fromDynValidShape_.empty()) {
        ss << " dynvalidshape:[";
        for (size_t i = 0; i < fromDynValidShape_.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << fromDynValidShape_[i].Dump();
        }
        ss << "]";
    }
    ss << " to offset:[";
    for (size_t i = 0; i < toOffset_.size(); i++) {
        if (i != 0) {
            ss << ",";
        }
        ss << std::setw(WIDTH) << toOffset_[i];
    }
    ss << "]";
    if (!toDynOffset_.empty()) {
        ss << " to dynoffset:[";
        for (size_t i = 0; i < toDynOffset_.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << toDynOffset_[i].Dump();
        }
        ss << "]";
    }
    return ss.str();
}

Json AssembleOpAttribute::DumpDynJson()
{
    Json res = Json::array();
    res.push_back(static_cast<int>(from_));
    res.push_back(static_cast<int32_t>(toOffset_.size()));
    for (auto offset : toOffset_) {
        res.push_back(offset);
    }
    res.push_back(static_cast<int32_t>(toDynOffset_.size()));
    for (auto offset : toDynOffset_) {
        auto joffset = ToJson(offset);
        if (joffset.size() > 0) {
            res.push_back(joffset);
        }
    }
    res.push_back(static_cast<int32_t>(fromDynValidShape_.size()));
    for (auto shape : fromDynValidShape_) {
        auto jshape = ToJson(shape);
        if (jshape.size() > 0) {
            res.push_back(jshape);
        }
    }
    return res;
}

std::shared_ptr<AssembleOpAttribute> AssembleOpAttribute::DeserializeFrom(const Json& attrJson,
                                                                          [[maybe_unused]] Function* function)
{
    int despos = 0;
    auto memType = attrJson[despos++];
    int offsetSize = attrJson[despos++];
    std::vector<int64_t> toOffset;
    for (int i = 0; i < offsetSize; i++) {
        toOffset.push_back(attrJson[despos++]);
    }
    int dynOffsetSize = attrJson[despos++];
    std::vector<SymbolicScalar> toDynOffset;
    for (int i = 0; i < dynOffsetSize; i++) {
        toOffset.push_back(LoadSymbolicScalar(attrJson[despos++]));
    }
    int dynValidShapeSize = attrJson[despos++];
    std::vector<SymbolicScalar> fromDynValidShape;
    for (int i = 0; i < dynValidShapeSize; i++) {
        fromDynValidShape.push_back(LoadSymbolicScalar(attrJson[despos++]));
    }
    return std::make_shared<AssembleOpAttribute>(static_cast<MemoryType>(memType), toOffset, toDynOffset,
                                                 fromDynValidShape);
}

std::shared_ptr<OpAttribute> AssembleOpAttribute::Clone() const
{
    return std::make_shared<AssembleOpAttribute>(from_, toOffset_, toDynOffset_, fromDynValidShape_);
}

CallOpAttribute::CallOpAttribute(const FunctionHash& calleeHash,
                                 const std::vector<std::vector<SymbolicScalar>>& argList,
                                 const std::string& calleMagicName, const std::map<int, SymbolicScalar>& outIndexToExpr,
                                 const std::vector<SymbolicScalar>& linearArgList)
    : invokeInfo_(std::make_shared<SubfuncInvokeInfoTy>()),
      calleeHash_(calleeHash),
      argList_(argList),
      linearArgList_(linearArgList),
      outIndexToExpr_(outIndexToExpr)
{
    // Make dump happy
    calleeBracketName_ = calleeHash_.Data() + "[" + calleeHash_.Data() + "]";
    calleMagicName_ = calleMagicName;
}

std::string CallOpAttribute::Dump() const
{
    std::stringstream ss;
    ss << calleeBracketName_ << "_" << calleeHash_.Data();
    ss << " attr:[";
    for (size_t i = 0; i < argList_.size(); i++) {
        if (i != 0) {
            ss << ",";
        }
        ss << i << "[";
        for (size_t j = 0; j < argList_[i].size(); j++) {
            if (j != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << argList_[i][j].Dump();
        }
        ss << "]";
    }
    ss << "]";
    return ss.str();
}

Json CallOpAttribute::DumpDynJson()
{
    Json res = Json::array();
    res.push_back(static_cast<uint64_t>(calleeHash_.GetHash()));
    res.push_back(static_cast<int32_t>(argList_.size()));
    for (size_t i = 0; i < argList_.size(); i++) {
        res.push_back(static_cast<int32_t>(argList_[i].size()));
        for (size_t j = 0; j < argList_[i].size(); j++) {
            res.push_back(ToJson(argList_[i][j]));
        }
    }
    res.push_back(static_cast<int32_t>(linearArgList_.size()));
    for (size_t i = 0; i < linearArgList_.size(); i++) {
        res.push_back(ToJson(linearArgList_[i]));
    }
    return res;
}

std::string CallOpAttribute::DumpAttr(int idx) const
{
    std::stringstream ss;
    if (idx < 0) {
        ss << "attr:[";
        for (size_t i = 0; i < argList_.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << i << "[";
            for (size_t j = 0; j < argList_[i].size(); j++) {
                if (j != 0) {
                    ss << ",";
                }
                ss << std::setw(WIDTH) << argList_[i][j].Dump();
            }
            ss << "]";
        }
        ss << "]";
    } else {
        FE_ASSERT(static_cast<size_t>(idx) < argList_.size())
            << "idx: " << static_cast<size_t>(idx) << "argList.size(): " << argList_.size();
        ss << "attr[" << idx << "][";
        for (size_t j = 0; j < argList_[idx].size(); j++) {
            if (j != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << argList_[idx][j].Dump();
        }
        ss << "]]";
    }
    return ss.str();
}

std::vector<SymbolicScalar>& CallOpAttribute::GetLinearArgList()
{
    if (linearArgList_.empty()) {
        // The first attr is callee info.
        linearArgList_.push_back(SymbolicScalar((int64_t)0));
        for (auto& l : argList_) {
            for (auto& arg : l) {
                linearArgList_.push_back(arg);
            }
        }
    }

    return linearArgList_;
}

std::vector<int64_t> CallOpAttribute::GetLinearImmediateArgList(int begin, int end, bool returnEmptyForSymbolic)
{
    std::vector<int64_t> result;

    auto& linearArgList = GetLinearArgList();
    for (int i = begin; i < end && i < static_cast<int>(linearArgList.size()); i++) {
        if (linearArgList[i].IsImmediate()) {
            result.push_back(linearArgList[i].Concrete());
        } else {
            if (returnEmptyForSymbolic) {
                return {};
            } else {
                FE_ASSERT(false) << "Invalid Immediate in " << Dump() << " index " << i << " = "
                                 << linearArgList[i].Dump();
            }
        }
    }

    return result;
}

std::shared_ptr<CallOpAttribute> CallOpAttribute::DeserializeFrom(const Json& attrJson,
                                                                  [[maybe_unused]] Function* function)
{
    // CallOp特殊：attrJson为整体的Json而不是单独的attr Json
    auto& attrJsonReal = attrJson["attr"];
    int despos = 0;
    FunctionHash calleeHash = static_cast<uint64_t>(attrJsonReal[despos++]);
    int32_t tensorCount = attrJsonReal[despos++];
    std::vector<std::vector<SymbolicScalar>> argList(tensorCount, std::vector<SymbolicScalar>());
    for (int i = 0; i < tensorCount; i++) {
        int32_t argSize = attrJsonReal[despos++];
        for (int j = 0; j < argSize; j++) {
            argList[i].push_back(LoadSymbolicScalar(attrJsonReal[despos++]));
        }
    }
    int32_t linearArgSize = attrJsonReal[despos++];
    std::vector<SymbolicScalar> linearArgList;
    for (int i = 0; i < linearArgSize; i++) {
        linearArgList.push_back(LoadSymbolicScalar(attrJsonReal[despos++]));
    }
    std::map<int, SymbolicScalar> outIndexToExpr;
    auto ret = std::make_shared<CallOpAttribute>(calleeHash, argList, "", outIndexToExpr, linearArgList);
    if (attrJson.count("invoke_info") != 0) {
        auto& invokeInfoJson = attrJson["invoke_info"];
        ret->invokeInfo_->LoadJson(invokeInfoJson, function);
    }
    return ret;
}

std::shared_ptr<OpAttribute> CallOpAttribute::Clone() const
{
    return std::make_shared<CallOpAttribute>(calleeHash_, argList_, calleMagicName_, outIndexToExpr_, linearArgList_);
}

Json CallOpAttribute::DumpInvokeInfoJson() { return invokeInfo_->DumpJson(); }

std::string ConvertOpAttribute::Dump() const { return MemoryTypeToString(from_) + "::" + MemoryTypeToString(to_); }

Json ConvertOpAttribute::DumpDynJson()
{
    Json res = Json::array();
    res.push_back(static_cast<int>(from_));
    res.push_back(static_cast<int>(to_));
    return res;
}

std::pair<MemoryType, MemoryType> ConvertOpAttribute::GetConvertPath() const { return {from_, to_}; }

std::shared_ptr<ConvertOpAttribute> ConvertOpAttribute::DeserializeFrom(const Json& attrJson,
                                                                        [[maybe_unused]] Function* function)
{
    HashBuffer buffer = attrJson.get<HashBuffer>();
    return std::make_shared<ConvertOpAttribute>(static_cast<MemoryType>(buffer[0]), static_cast<MemoryType>(buffer[1]));
}

std::shared_ptr<OpAttribute> ConvertOpAttribute::Clone() const
{
    return std::make_shared<ConvertOpAttribute>(from_, to_);
}

std::pair<MemoryType, std::vector<OpImmediate>> CopyOpAttribute::GetCopyOutAttr() const { return {from_, toOffset_}; }

std::pair<std::vector<OpImmediate>, MemoryType> CopyOpAttribute::GetCopyInAttr() const { return {fromOffset_, to_}; }

std::string CopyOpAttribute::Dump() const
{
    std::stringstream ss;
    ss << "shape[";
    for (size_t i = 0; i < tensorShape_.size(); i++) {
        if (i != 0) {
            ss << ",";
        }
        ss << std::setw(WIDTH) << tensorShape_[i].Dump();
    }
    ss << "],";
    ss << "rawshape[";
    for (size_t i = 0; i < rawShape_.size(); i++) {
        if (i != 0) {
            ss << ",";
        }
        ss << std::setw(WIDTH) << rawShape_[i].Dump();
    }
    ss << "],";
    if (isCopyOut_) {
        ss << "from " << MemoryTypeToString(from_);
        if (fromDynValidShape_.size() != 0) {
            ss << " dynvalidshape:[";
            for (size_t i = 0; i < fromDynValidShape_.size(); i++) {
                if (i != 0) {
                    ss << ",";
                }
                ss << std::setw(WIDTH) << fromDynValidShape_[i].Dump();
            }
            ss << "]";
        }
        ss << " to offset:[";
        for (size_t i = 0; i < toOffset_.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << toOffset_[i].Dump();
        }
        ss << "]";
    } else {
        ss << " from offset:[";
        for (size_t i = 0; i < fromOffset_.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << std::setw(WIDTH) << fromOffset_[i].Dump();
        }
        ss << "] to " << MemoryTypeToString(to_);
        if (toDynValidShape_.size() != 0) {
            ss << " dynvalidshape:[";
            for (size_t i = 0; i < toDynValidShape_.size(); i++) {
                if (i != 0) {
                    ss << ",";
                }
                ss << std::setw(WIDTH) << toDynValidShape_[i].Dump();
            }
            ss << "]";
        }
    }

    return ss.str();
}

Json CopyOpAttribute::DumpDynJson()
{
    Json res = Json::array();
    res.push_back(static_cast<int>(isCopyOut_));
    if (isCopyOut_) {
        res.push_back(static_cast<int32_t>(from_));
        res.push_back(static_cast<int32_t>(toOffset_.size()));
        for (auto toOffset : toOffset_) {
            Json offsetJson = toOffset.DumpDynJson();
            for (auto& offset : offsetJson) {
                res.push_back(offset);
            }
        }
    } else {
        res.push_back(static_cast<int32_t>(to_));
        res.push_back(static_cast<int32_t>(fromOffset_.size()));
        for (auto fromOffset : fromOffset_) {
            Json offsetJson = fromOffset.DumpDynJson();
            for (auto& offset : offsetJson) {
                res.push_back(offset);
            }
        }
    }
    for (auto& tensorshape : tensorShape_) {
        Json shapeJson = tensorshape.DumpDynJson();
        for (auto& shape : shapeJson) {
            res.push_back(shape);
        }
    }
    for (auto& rawShape : rawShape_) {
        Json shapeJson = rawShape.DumpDynJson();
        for (auto& shape : shapeJson) {
            res.push_back(shape);
        }
    }
    if (isCopyOut_) {
        res.push_back(static_cast<int32_t>(fromDynValidShape_.size()));
        for (auto& validShape : fromDynValidShape_) {
            Json validShapeJson = validShape.DumpDynJson();
            for (auto& shape : validShapeJson) {
                res.push_back(shape);
            }
        }
    } else {
        res.push_back(static_cast<int32_t>(toDynValidShape_.size()));
        for (auto& validShape : toDynValidShape_) {
            Json validShapeJson = validShape.DumpDynJson();
            for (auto& shape : validShapeJson) {
                res.push_back(shape);
            }
        }
    }
    return res;
}

std::vector<int64_t> CopyOpAttribute::GetSpecifiedShape(int64_t defaultValue) const
{
    std::vector<int64_t> result(tensorShape_.size(), defaultValue);
    for (size_t i = 0; i < tensorShape_.size(); ++i) {
        if (tensorShape_[i].IsSpecified()) {
            result[i] = tensorShape_[i].GetSpecifiedValue().ConcreteValid() ?
                            static_cast<int64_t>(tensorShape_[i].GetSpecifiedValue()) :
                            -1;
        }
    }
    return result;
}

bool CopyOpAttribute::IsDynToOffset() const
{
    for (auto& x : toOffset_) {
        if (x.GetSpecifiedValue().IsExpression() || x.GetSpecifiedValue().IsSymbol()) {
            return true;
        }
    }
    return false;
}

bool CopyOpAttribute::IsDynOffset(const std::vector<OpImmediate>& offset) const
{
    for (auto& x : offset) {
        if (!x.IsSpecified()) {
            continue;
        }
        if (x.GetSpecifiedValue().IsExpression() || x.GetSpecifiedValue().IsSymbol()) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<CopyOpAttribute> CopyOpAttribute::DeserializeFrom(const Json& attrJson,
                                                                  [[maybe_unused]] Function* function)
{
    if (attrJson.size() <= VALUE3) {
        std::vector<OpImmediate> vec;
        auto res = std::make_shared<CopyOpAttribute>(MemoryType::MEM_UNKNOWN, vec, vec, vec);
        FE_LOGE(FeError::INVALID_VAL, "CopyOpAttr json shape is illegal");
        return res;
    }
    std::vector<OpImmediate> shape;
    std::vector<OpImmediate> offset;
    std::vector<OpImmediate> rawShape;
    std::vector<OpImmediate> dynValidShape;
    size_t despos = 0;
    std::shared_ptr<CopyOpAttribute> result;
    bool isCopyOut;
    if (attrJson[despos].is_number()) {
        isCopyOut = (attrJson[despos].get<int>() != 0);
    } else {
        isCopyOut = attrJson[despos].get<bool>();
    }
    despos++;
    if (isCopyOut) {
        auto from = attrJson[despos++];
        auto size = attrJson[despos++];
        for (size_t i = 0; i < size; i++) {
            offset.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        }
        for (size_t i = 0; i < size; i++) {
            shape.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        }
        for (size_t i = 0; i < size; i++) {
            rawShape.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        }
        auto validShapeSize = attrJson[despos++];
        for (size_t i = 0; i < validShapeSize; i++) {
            dynValidShape.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        };
        result = std::make_shared<CopyOpAttribute>(static_cast<MemoryType>(from), offset, shape, rawShape,
                                                   dynValidShape);
    } else {
        auto to = attrJson[despos++];
        auto size = attrJson[despos++];
        for (size_t i = 0; i < size; i++) {
            offset.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        }
        for (size_t i = 0; i < size; i++) {
            shape.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        }
        for (size_t i = 0; i < size; i++) {
            rawShape.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        }
        auto validShapeSize = attrJson[despos++];
        for (size_t i = 0; i < validShapeSize; i++) {
            dynValidShape.push_back(OpImmediate::DeserializeFrom(attrJson, despos));
        }
        result = std::make_shared<CopyOpAttribute>(offset, static_cast<MemoryType>(to), shape, rawShape, dynValidShape);
    }
    return result;
}

std::shared_ptr<OpAttribute> CopyOpAttribute::Clone() const { return std::make_shared<CopyOpAttribute>(*this); }
