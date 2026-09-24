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
 * \file infer_param_index.cpp
 * \brief
 */

#include <vector>
#include "infer_param_index.h"
#include "interface/operation/opcode.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/topo_program.h"

#define MODULE_NAME "InferParamIndex"

namespace npu {
namespace tile_fwk {
std::string InferParamIndex::DumpParamIndex(const std::map<std::string, DynParamInfo>& dynParamTable)
{
    std::ostringstream ss;
    for (auto paramInfo : dynParamTable) {
        ss << "param: " << paramInfo.first << " ( ";
        ss << "tensorIdx: " << paramInfo.second.tensorIndex << ", ";
        ss << "dimsize: " << paramInfo.second.dimSize << ", ";
        ss << "type: " << static_cast<int>(paramInfo.second.type) << ", ";
        ss << "addrCoaIdx: " << paramInfo.second.tensorBaseAddrCoaIndex << ", ";
        ss << "dimIdx: " << paramInfo.second.dimIndex << " )" << std::endl;
    }
    return ss.str();
}
// 从 "sym_<magic>_dim_<idx>" 格式的符号名中解析出 tensor magic 与维度下标。
// 该格式由 ResetOutputDynValidShape 创建，用于给 CopyIn/CopyOut 的输出 valid shape 命名符号。
static bool ParseSymDimSymbolName(const std::string& name, int& magic, int& dimIdx)
{
    static const std::string kPrefix = "sym_";
    static const std::string kMid = "_dim_";
    if (name.rfind(kPrefix, 0) != 0) {
        return false;
    }
    size_t midPos = name.find(kMid, kPrefix.size());
    if (midPos == std::string::npos || midPos <= kPrefix.size() || midPos + kMid.size() >= name.size()) {
        return false;
    }
    try {
        magic = std::stoi(name.substr(kPrefix.size(), midPos - kPrefix.size()));
        dimIdx = std::stoi(name.substr(midPos + kMid.size()));
    } catch (...) {
        return false;
    }
    return true;
}

// 递归收集 valid shape 表达式中引用的 sym_ 符号并注册到 DynParamTable。
// CopyIn/CopyOut 输出的 valid shape 可能被后续 InferShape（如 AssembleInferFunc 的 max 组合）
// 覆盖为复合表达式，此时其自身符号不再以顶层符号形式出现，若不同样注册，
// codegen 生成的表达式会引用未声明符号导致内核编译失败。
static void CollectNestedDynParamSymbols(const SymbolicScalar& scalar, const std::map<int, int>& magic2Coa,
                                         const std::map<int, std::vector<SymbolicScalar>>& addr2ValidShape,
                                         const std::map<int, std::vector<SymbolicScalar>>& addr2ValidShapeSpecified,
                                         const std::map<int, int>& coa2TensorIndex,
                                         std::set<std::string>& visitedSymbol, Function& subFunc)
{
    auto raw = scalar.Raw();
    if (raw == nullptr) {
        return;
    }
    if (raw->IsSymbol()) {
        std::string name = raw->GetSymbolName();
        if (visitedSymbol.count(name) > 0) {
            return;
        }
        int magic = 0;
        int dimIdx = 0;
        if (!ParseSymDimSymbolName(name, magic, dimIdx)) {
            return;
        }
        auto magicIter = magic2Coa.find(magic);
        if (magicIter == magic2Coa.end()) {
            return;
        }
        int coa = magicIter->second;
        auto shapeIter = addr2ValidShape.find(coa);
        auto tensorIndexIter = coa2TensorIndex.find(coa);
        if (shapeIter == addr2ValidShape.end() || tensorIndexIter == coa2TensorIndex.end()) {
            return;
        }
        int dimSize = static_cast<int>(shapeIter->second.size());
        if (dimIdx < 0 || dimIdx >= dimSize) {
            return;
        }
        SymbolicScalar dynDim;
        auto specifiedIter = addr2ValidShapeSpecified.find(coa);
        if (specifiedIter != addr2ValidShapeSpecified.end() &&
            dimIdx < static_cast<int>(specifiedIter->second.size())) {
            dynDim = specifiedIter->second[dimIdx];
        }
        auto paramInfo = DynParamInfo{
            dimSize, tensorIndexIter->second, coa, DynParamInfoType::VALID_SHAPE, dimIdx, dynDim, false, ""};
        subFunc.InsertDynParam(name, paramInfo);
        visitedSymbol.insert(name);
        return;
    }
    if (raw->IsExpression()) {
        for (const auto& operand : raw->GetExpressionOperandList()) {
            CollectNestedDynParamSymbols(SymbolicScalar(operand), magic2Coa, addr2ValidShape, addr2ValidShapeSpecified,
                                         coa2TensorIndex, visitedSymbol, subFunc);
        }
    }
}

static bool IsInGMSpill(Operation& op)
{
    if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode())) {
        for (auto& iOperand : op.GetIOperands()) {
            if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                return true;
            }
        }
    }
    return false;
}

static bool IsOutGMSpill(Operation& op)
{
    if (OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
        for (auto& oOperand : op.GetOOperands()) {
            if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                return true;
            }
        }
    }
    return false;
}

Status InferParamIndex::ResetOutputDynValidShape(Operation& op, Function& function)
{
    if (ResetGmCopyDynValidShape(op, function)) {
        return SUCCESS;
    }
    for (auto outOperand : op.GetOOperands()) {
        std::vector<SymbolicScalar> validShape;
        if (OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode()) || setSymDimOps.count(op.GetOpcode())) {
            for (size_t dimIdx = 0U; dimIdx < outOperand->GetShape().size(); ++dimIdx) {
                validShape.emplace_back("sym_" + std::to_string(outOperand->GetMagic()) + "_dim_" +
                                        std::to_string(dimIdx));
            }
        }
        // 通信的输出要从opattr中获取不能直接使用normalize
        bool* distCopyType = op.GetAttr<bool>(OpAttributeKey::isDistCopyOut);
        bool shouldUpdateDynValidShape = !useSelfOps.count(op.GetOpcode()) &&
                                         (!function.IsFromOutCast(outOperand) || distCopyType);
        if (shouldUpdateDynValidShape) {
            outOperand->UpdateDynValidShape(validShape);
        }
    }
    return SUCCESS;
}

bool InferParamIndex::ResetGmCopyDynValidShape(Operation& op, Function& function)
{
    bool isCopyIn = IsInGMSpill(op);
    bool isCopyOut = IsOutGMSpill(op);
    if (!isCopyIn && !isCopyOut) {
        return false;
    }
    auto operands = isCopyIn ? op.GetIOperands() : op.GetOOperands();
    auto& casts = isCopyIn ? function.inCasts_ : function.outCasts_;
    auto operand = operands.front();
    if (find(casts.begin(), casts.end(), operand) != casts.end()) {
        return false;
    }
    bool* distCopyType = op.GetAttr<bool>(OpAttributeKey::isDistCopyOut);
    int tensorBaseAddrCoaIndex = IsCopyIn(op.GetOpcode()) ? op.GetIOpAttrOffset(0) : op.GetOOpAttrOffset(0);
    tensorBaseAddrCoaIndex = (distCopyType && !*distCopyType) ? op.GetIOpAttrOffset(0) : tensorBaseAddrCoaIndex;
    if (tensorBaseAddrCoaIndex != -1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op[%d] in function %s still uses its DynValidShape", op.GetOpMagic(),
                          function.GetRawName().c_str());
        return true;
    }
    std::vector<SymbolicScalar> validShape;
    op.GetOOperands().front()->UpdateDynValidShape(validShape);
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "op[%d] in function %s has cleared its DynValidShape what will be inferred again",
                      op.GetOpMagic(), function.GetRawName().c_str());
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    if (isCopyIn) {
        copyAttr->SetToDynValidShape(OpImmediate::Specified(validShape));
    } else {
        copyAttr->SetFromDynValidShape(OpImmediate::Specified(validShape));
    }
    return true;
}

Status InferParamIndex::ResetViewDynValidShape(const Operation& op)
{
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return SUCCESS;
    }
    auto newDynValidShape = viewOpAttribute->GetToDynValidShape();
    std::vector<int> newValidShape;
    for (auto validSym : newDynValidShape) {
        if (validSym.ConcreteValid()) {
            newValidShape.push_back(validSym.Concrete());
        }
    }
    if (newValidShape.size() == newDynValidShape.size()) {
        op.GetOOperands()[0]->UpdateDynValidShape(newDynValidShape);
        return SUCCESS;
    }
    viewOpAttribute->SetToDynValidShape(op.GetOOperands()[0]->GetDynValidShape());
    return SUCCESS;
}

Status InferParamIndex::ResetAssembleDynValidShape(const Operation& op)
{
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute*>(op.GetOpAttribute().get());
    if (assembleOpAttribute != nullptr) {
        auto emptyValidShape = std::vector<SymbolicScalar>();
        assembleOpAttribute->SetFromDynValidShape(emptyValidShape);
    }
    return SUCCESS;
}

Status InferParamIndex::ResetDynValidShape(Function& function)
{
    for (auto& op : function.Operations(false)) {
        if (ResetOutputDynValidShape(op, function) != SUCCESS) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Fail to reset the output operand shape of operation %d in function %s. Please check whether the shape "
                "is valid in your input graph.%s",
                op.GetOpMagic(), function.GetRawName().c_str(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (ResetViewDynValidShape(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Fail to reset the output operand shape of VIEW operation %d in function %s. %s",
                                  op.GetOpMagic(), function.GetRawName().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
        // 清空assemble的属性中的dynvalidshape，以便后续重新推导符号化的dynvalidshape
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (ResetAssembleDynValidShape(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Fail to reset the output operand shape of ASSEMBLE operation %d in function %s. %s",
                                  op.GetOpMagic(), function.GetRawName().c_str(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferParamIndex::InferShape(Function& function)
{
    auto opList = function.Operations(true, SortOperationsMode::LIGHTWEIGHT).DuplicatedOpList();
    if (opList.empty()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "There is no operation in function %s. Please check the operation list of the input graph",
                          function.GetRawName().c_str());
        return FAILED;
    }

    TopoProgramUtils::TopoProgram(opList, true);
    return SUCCESS;
}

Status InferParamIndex::InsertAddr2ValidShapeSpecified(
    Operation& op, std::map<int, std::vector<SymbolicScalar>>& addr2ValidShape,
    std::map<int, std::vector<SymbolicScalar>>& addr2ValidShapeSpecified, std::map<int, int>& magic2Coa)
{
    bool* distCopyType = op.GetAttr<bool>(OpAttributeKey::isDistCopyOut);
    // 暂不处理输入个数小于输出个数的copyIn，原因是coaIndex不够分
    if (IsCopyIn(op.GetOpcode())) {
        auto ioNum = op.GetIOperands().size();
        auto ooNum = op.GetOOperands().size();
        if (ioNum < ooNum) {
            APASS_LOG_ERROR_F(Elements::Operation, "Copyin[%d] does not support fewer inputs than outputs.",
                              op.GetOpMagic());
            return FAILED;
        }
    }

    for (size_t i = 0; i < op.GetOOperands().size(); i++) {
        int tensorBaseAddrCoaIndex = IsCopyIn(op.GetOpcode()) ? op.GetIOpAttrOffset(0) : op.GetOOpAttrOffset(i);
        tensorBaseAddrCoaIndex = (distCopyType && !*distCopyType) ? op.GetIOpAttrOffset(0) : tensorBaseAddrCoaIndex;
        if (tensorBaseAddrCoaIndex == -1) {
            continue;
        }
        if (OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode()) || setSymDimOps.count(op.GetOpcode())) {
            magic2Coa[op.GetOOperands()[i]->GetMagic()] = tensorBaseAddrCoaIndex;
        }
        if (addr2ValidShape.find(tensorBaseAddrCoaIndex) == addr2ValidShape.end()) {
            addr2ValidShape[tensorBaseAddrCoaIndex] = op.GetOOperands()[i]->GetDynValidShape();
            if (IsCopyIn(op.GetOpcode())) {
                auto attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
                if (attr->GetToDynValidShape().size() != 0 && attr->GetToDynValidShape()[0].IsSpecified()) {
                    addr2ValidShapeSpecified[tensorBaseAddrCoaIndex] = OpImmediate::ToSpecified(
                        attr->GetToDynValidShape());
                }
            }
            if (distCopyType && *distCopyType) {
                auto attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
                if (attr->GetFromDynValidShape().size() != 0) {
                    addr2ValidShapeSpecified[tensorBaseAddrCoaIndex] = OpImmediate::ToSpecified(
                        attr->GetFromDynValidShape());
                }
            }
        }
    }
    return SUCCESS;
}

Status InferParamIndex::UpdateValidShape(Function& subFunc, std::map<int, std::vector<SymbolicScalar>>& addr2ValidShape,
                                         std::map<int, std::vector<SymbolicScalar>>& addr2ValidShapeSpecified,
                                         std::map<int, int>& magic2Coa)
{
    for (auto& op : subFunc.Operations(false)) {
        if (InsertAddr2ValidShapeSpecified(op, addr2ValidShape, addr2ValidShapeSpecified, magic2Coa) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InsertAddr2ValidShapeSpecified failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status InferParamIndex::SetSubValidShape(Function& subFunc, std::map<int, std::vector<SymbolicScalar>>& addr2ValidShape,
                                         std::map<int, std::vector<SymbolicScalar>>& addr2ValidShapeSpecified,
                                         const std::map<int, int>& magic2Coa)
{
    std::set<std::string> visitedSymbol;
    std::map<int, int> coa2TensorIndex;
    int tensorIndex{0};
    for (auto validShape : addr2ValidShape) {
        coa2TensorIndex[validShape.first] = tensorIndex;
        int dimIdx{0};
        for (auto dim : validShape.second) {
            if (!dim.IsSymbol()) {
                continue;
            }
            if (visitedSymbol.count(dim.Dump()) > 0) {
                continue;
            }
            auto tensorBaseAddrCoaIndex = validShape.first;
            SymbolicScalar dynDim;
            if (addr2ValidShapeSpecified.count(tensorBaseAddrCoaIndex)) {
                dynDim = addr2ValidShapeSpecified[tensorBaseAddrCoaIndex][dimIdx];
            }
            auto paramInfo = DynParamInfo{static_cast<int>(validShape.second.size()),
                                          tensorIndex,
                                          tensorBaseAddrCoaIndex,
                                          DynParamInfoType::VALID_SHAPE,
                                          dimIdx,
                                          dynDim,
                                          false,
                                          ""};
            subFunc.InsertDynParam(dim.Dump(), paramInfo);
            dimIdx++;
        }
        tensorIndex++;
    }
    for (const auto& validShape : addr2ValidShape) {
        for (const auto& dim : validShape.second) {
            if (dim.IsSymbol()) {
                continue;
            }
            CollectNestedDynParamSymbols(dim, magic2Coa, addr2ValidShape, addr2ValidShapeSpecified, coa2TensorIndex,
                                         visitedSymbol, subFunc);
        }
    }
    return SUCCESS;
}

Status InferParamIndex::UpdateParamIndex(Function& function)
{
    for (auto& subProgram : function.rootFunc_->programs_) {
        auto& subFunc = *subProgram.second;
        if (ResetDynValidShape(subFunc) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function,
                              "ResetDynValidShape failed; Please check the ResetDynValidShape method.");
            return FAILED;
        }
        if (InferShape(subFunc) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Function, "Print function before update: %s\n", subFunc.Dump().c_str());
        std::map<int, std::vector<SymbolicScalar>> addr2ValidShape;
        std::map<int, std::vector<SymbolicScalar>> addr2ValidShapeSpecified;
        std::map<int, int> magic2Coa;
        if (UpdateValidShape(subFunc, addr2ValidShape, addr2ValidShapeSpecified, magic2Coa) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function,
                              "Update valid shape for the function %s failed. Please check above for more information.",
                              function.GetRawName().c_str());
            return FAILED;
        }
        if (SetSubValidShape(subFunc, addr2ValidShape, addr2ValidShapeSpecified, magic2Coa) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function,
                              "Update valid shape for the function %s failed. Please check above for more information.",
                              function.GetRawName().c_str());
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Function, "Print function after update: %s\n",
                          DumpParamIndex(subFunc.GetDynParamTable()).c_str());
    }
    return SUCCESS;
}

Status InferParamIndex::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start InferParamIndex.");
    if (UpdateParamIndex(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "UpdateParamIndex failed; Please check the UpdateParamIndex method.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End InferParamIndex By Sequential Execution.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
