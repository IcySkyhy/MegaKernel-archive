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
 * \file dyn_attr_to_static.cpp
 * \brief
 */

#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "interface/utils/common.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/dump_function_utils.h"

namespace npu {
namespace tile_fwk {
enum class BranchMode : int {
    DEFAULT_BRANCH_MODE = 0,
    STATIC_CONST_BRANCH_MODE = 1,
    VARIABLE_BRANCH_MODE = 2,
    CONST_BRANCH_MODE = 3
};

struct CoaInfo {
    CoaType macroType = CoaType::INVALID;
    int dim = -1;
    int base = -1;
    int idx = -1;

    static bool ParseParamOffset(const std::string& coaExpr, std::smatch& match)
    {
        static std::regex pattern("RUNTIME_COA_GET_PARAM_OFFSET\\((\\d+), (\\d+), (\\d+)\\)");
        return std::regex_search(coaExpr, match, pattern);
    }

    static bool ParseParamValidShape(const std::string& coaExpr, std::smatch& match)
    {
        static std::regex pattern("RUNTIME_COA_GET_PARAM_VALID_SHAPE\\((\\d+), (\\d+), (\\d+)\\)");
        return std::regex_search(coaExpr, match, pattern);
    }

    static bool ParseParam(const std::string& coaExpr, std::smatch& match)
    {
        static std::regex pattern("RUNTIME_COA_GET_PARAM\\((\\d+)\\)");
        return std::regex_search(coaExpr, match, pattern);
    }

    static bool ParseParamRawShape(const std::string& coaExpr, std::smatch& match)
    {
        static std::regex pattern("RUNTIME_COA_GET_PARAM_RAW_SHAPE\\((\\d+), (\\d+), (\\d+)\\)");
        return std::regex_search(coaExpr, match, pattern);
    }

    Status SToIParamShapeAndOffset(const std::smatch& match)
    {
        if (SToIWrapper(match[INPUT_PARAM_POS_ONE].str(), dim) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert dim.");
            return FAILED;
        }
        if (SToIWrapper(match[INPUT_PARAM_POS_TWO].str(), base) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert base.");
            return FAILED;
        }
        if (SToIWrapper(match[INPUT_PARAM_POS_THREE].str(), idx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert idx.");
            return FAILED;
        }
        return SUCCESS;
    }

    Status ParseCoaString(const std::string& coaExpr)
    {
        std::smatch match;
        if (ParseParamOffset(coaExpr, match)) {
            macroType = CoaType::PARAM_OFFSET;
            if (SToIParamShapeAndOffset(match) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "ParseCoaString failed to convert indices, CoaType::PARAM_OFFSET, input coaExpr %s.",
                                  coaExpr.c_str());
                return FAILED;
            }
        } else if (ParseParamValidShape(coaExpr, match)) {
            macroType = CoaType::PARAM_VALID_SHAPE;
            if (SToIParamShapeAndOffset(match) != SUCCESS) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "ParseCoaString failed to convert indices, CoaType::PARAM_VALID_SHAPE, input coaExpr %s.",
                    coaExpr.c_str());
                return FAILED;
            }
        } else if (ParseParam(coaExpr, match)) {
            macroType = CoaType::PARAM;
            if (SToIWrapper(match[INPUT_PARAM_POS_ONE].str(), idx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "ParseCoaString failed to convert indices, CoaType::PARAM, input coaExpr %s.",
                                  coaExpr.c_str());
                return FAILED;
            }
        } else if (coaExpr.find(MAYBE_CONST_POSTFIX) != std::string::npos) {
            APASS_LOG_ERROR_F(Elements::Operation, "This coaExpr %s has been processed", coaExpr.c_str());
            return FAILED;
        } else if (ParseParamRawShape(coaExpr, match)) {
            macroType = CoaType::PARAM_RAW_SHAPE;
            if (SToIParamShapeAndOffset(match) != SUCCESS) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "ParseCoaString failed to convert indices, CoaType::PARAM_RAW_SHAPE, input coaExpr %s.",
                    coaExpr.c_str());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "ParseCoaString input coaExpr %s is not recognized.",
                              coaExpr.c_str());
            return FAILED;
        }
        return SUCCESS;
    }

    int CalculateCoaIndex()
    {
        if (macroType == CoaType::PARAM_OFFSET) {
            return ((base) + 1) + OFFSET_INDEX_ORDER * (dim) + idx;
        } else if (macroType == CoaType::PARAM_VALID_SHAPE) {
            return ((base) + 1) + VALID_SHAPE_INDEX_ORDER * (dim) + idx;
        } else if (macroType == CoaType::PARAM_RAW_SHAPE) {
            return ((base) + 1) + RAWSHAPE_INDEX_ORDER * (dim) + idx;
        } else if (macroType == CoaType::PARAM) {
            return idx;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "GetCoaFinalIdx Coa type is invalid.");
        return 0;
    }

    SymbolicScalar BuildMaybeConstCoa(int isConst, int attrValue)
    {
        if (isConst == 1 && attrValue > -1) {
            return attrValue;
        }
        if (macroType == CoaType::PARAM_OFFSET) {
            return MAYBE_CONST_COA_GetOffset(isConst, attrValue, dim, base, idx);
        } else if (macroType == CoaType::PARAM_VALID_SHAPE) {
            return MAYBE_CONST_COA_GetValidShape(isConst, attrValue, dim, base, idx);
        } else if (macroType == CoaType::PARAM) {
            return MAYBE_CONST_COA_GetParam(isConst, attrValue, idx);
        } else if (macroType == CoaType::PARAM_RAW_SHAPE) {
            return MAYBE_CONST_COA_GetRawShape(isConst, attrValue, dim, base, idx);
        }
        APASS_LOG_ERROR_F(Elements::Operation, "BuildMaybeConstCoa Coa type is invalid.");
        return 0;
    }
};

struct IsConstMetric {
    static constexpr int kAttrValueUninit_ = -2;
    static constexpr int kNotConst_ = 2;
    static constexpr int kConstButNotStatic_ = 3;

    int isConst = 1;                   // not const = 2; const = 1; const but not static = 3;
    int attrValue = kAttrValueUninit_; // static >= 0; else -1.

    void MarkNotConst() { isConst = kNotConst_; }
    int GetIsConst() { return isConst; }
    int GetAttrValue() { return attrValue; }
    bool TryInitAndCheckEqual(int newValue)
    {
        APASS_LOG_DEBUG_F(Elements::Operation, "Update dynScalar value=%d.", newValue);
        if (newValue < 0) {
            isConst = 0;
            attrValue = -1;
            return false;
        }
        if (attrValue == kAttrValueUninit_) {
            attrValue = newValue;
            return true;
        }
        if (newValue != attrValue) {
            isConst = kConstButNotStatic_;
            attrValue = -1;
            return true;
        }
        return true;
    }
};

Status SToIWrapper(const std::string str, int& result)
{
    try {
        result = std::stoi(str);
        return SUCCESS;
    } catch (const std::exception& e) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert %s to int, error is %s.", str.c_str(), e.what());
    }
    return FAILED;
}

void DynAttrToStatic::RefSpecifiedValue(std::vector<SymbolicScalar>& oriList,
                                        std::vector<std::reference_wrapper<SymbolicScalar>>& newList) const
{
    for (auto& value : oriList) {
        newList.push_back(std::reference_wrapper<SymbolicScalar>(value));
    }
}

void DynAttrToStatic::FilterSpecifiedValue(std::vector<OpImmediate>& oriList,
                                           std::vector<std::reference_wrapper<SymbolicScalar>>& newList) const
{
    for (auto& value : oriList) {
        if (value.IsSpecified()) {
            newList.push_back(std::reference_wrapper<SymbolicScalar>(value.GetSpecifiedValue()));
        }
    }
}

std::vector<std::reference_wrapper<SymbolicScalar>> DynAttrToStatic::GetOpDynamicAttributeList(Operation& op)
{
    std::vector<std::reference_wrapper<SymbolicScalar>> dynamicAttributeList;
    auto opcode = op.GetOpcode();
    if (opcode == Opcode::OP_VIEW) {
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewAttr != nullptr) {
            RefSpecifiedValue(viewAttr->GetFromDynOffset(), dynamicAttributeList);
            RefSpecifiedValue(viewAttr->GetToDynValidShape(), dynamicAttributeList);
        }
        return dynamicAttributeList;
    }

    if (opcode == Opcode::OP_ASSEMBLE) {
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
        if (assembleAttr != nullptr) {
            RefSpecifiedValue(assembleAttr->GetToDynOffset(), dynamicAttributeList);
            RefSpecifiedValue(assembleAttr->GetFromDynValidShape(), dynamicAttributeList);
        }
        return dynamicAttributeList;
    }

    const std::set<Opcode> specifiedOps = {
        Opcode::OP_VEC_DUP,        Opcode::OP_EXPAND,       Opcode::OP_RESHAPE,        Opcode::OP_NCHW2NC1HWC0,
        Opcode::OP_NCHW2Fractal_Z, Opcode::OP_NC1HWC02NCHW, Opcode::OP_NCDHW2NDC1HWC0, Opcode::OP_NCDHW2FRACTAL_Z_3D,
        Opcode::OP_NDC1HWC02NCDHW};
    if (specifiedOps.count(opcode)) {
        auto& attrDict = op.GetAllAttr();
        auto it = attrDict.find(OpAttributeKey::dynScalar);
        if (it != attrDict.end()) {
            auto& value = pypto::AnyCastRef<SymbolicScalar>(it->second);
            dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(value));
        }
        it = attrDict.find(OpAttributeKey::transDataOffset);
        if (it != attrDict.end()) {
            auto& values = pypto::AnyCastRef<std::vector<SymbolicScalar>>(it->second);
            for (auto& value : values) {
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(value));
            }
        }
        return dynamicAttributeList;
    }

    if (OpcodeManager::Inst().IsCopyInOrOut(opcode) || OpcodeManager::Inst().IsCopyFromL1(opcode)) {
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
        if (copyAttr != nullptr) {
            FilterSpecifiedValue(copyAttr->GetToOffset(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetFromOffset(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetToDynValidShape(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetFromDynValidShape(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetRawShape(), dynamicAttributeList);
        }
    }
    return dynamicAttributeList;
}

Status DynAttrToStatic::GetCallee(const Operation& callop, Function*& callFunc)
{
    auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop.GetOpAttribute());
    callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
    if (callFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Get callee function %s failed.",
                          callopAttr->GetCalleeMagicName().c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status DynAttrToStatic::BuildLeafToCaller(Function* func)
{
    if (func->IsFunctionTypeAndGraphType(
            {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH},
            GraphType::TENSOR_GRAPH)) {
        for (auto callop : func->GetCallopList()) {
            Function* nextFunc = nullptr;
            if (GetCallee(*callop, nextFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, %s[%d] GetCallee failed.%s",
                                  func->GetRawName().c_str(), callop->GetOpcodeStr().c_str(), callop->GetOpMagic(),
                                  GetFormatBacktrace(callop).c_str());
                return FAILED;
            }
            if (BuildLeafToCaller(nextFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, nextFunc at %s failed",
                                  func->GetRawName().c_str(), nextFunc->GetRawName().c_str());
                return FAILED;
            }
        }
        return SUCCESS;
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        Function* rootFunc = func->GetRootFunction();
        return BuildLeafToCaller(rootFunc);
    } else if (func->GetGraphType() == GraphType::EXECUTE_GRAPH) {
        for (auto callop : func->GetCallopList()) {
            Function* leafFunc = nullptr;
            if (GetCallee(*callop, leafFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, %s[%d] GetCallee failed.%s",
                                  func->GetRawName().c_str(), callop->GetOpcodeStr().c_str(), callop->GetOpMagic(),
                                  GetFormatBacktrace(callop).c_str());
                return FAILED;
            }
            leaf2Caller[leafFunc].push_back(callop);
        }
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s entered unexpected function type %d.",
                      func->GetRawName().c_str(), static_cast<int>(func->GetFunctionType()));
    return FAILED;
}

Status DynAttrToStatic::BuildNewCoa(std::reference_wrapper<SymbolicScalar>& dynScalar,
                                    std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim)
{
    constexpr int kNonImmediateSentinel = -2; // sentinel for non-immediate callop attr
    // 1. 拆解dynScalar到对应的COA表达式
    std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynScalar);
    if (dynParamExpr.find(COA_PREFIX) != 1) { // dynParamExpr格式是"(RUNTIME_GET_COA_XXX"
        APASS_LOG_INFO_F(Elements::Operation, "BuildNewCoa skips non-COA dynamic expression.");
        return SUCCESS;
    }
    CoaInfo coaExpr;
    if (coaExpr.ParseCoaString(dynParamExpr) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BuildNewCoa found unexpected COA expression at dynParamExpr.");
        return FAILED;
    }
    int coaIndex = coaExpr.CalculateCoaIndex();
    APASS_LOG_DEBUG_F(Elements::Operation, "Update dynScalar[%s] : ", dynParamExpr.c_str());

    // 2. 遍历不同caller下的取值，确认是否是常数
    std::set<int> scalarValue;
    for (auto& argList : callopArglistOneDim) {
        auto& callopAttr = argList[coaIndex];
        if (!callopAttr.IsImmediate()) {
            scalarValue.insert(kNonImmediateSentinel);
        } else {
            scalarValue.insert(callopAttr.Concrete());
        }
        if (scalarValue.size() > 1 && scalarValue.count(kNonImmediateSentinel) > 0) {
            break;
        }
    }
    int branchMode = static_cast<int>(BranchMode::DEFAULT_BRANCH_MODE);
    int attrValue = kNonImmediateSentinel;
    if (scalarValue.size() == 1) {
        if (scalarValue.count(kNonImmediateSentinel) > 0) {
            branchMode = static_cast<int>(BranchMode::VARIABLE_BRANCH_MODE);
        } else {
            branchMode = static_cast<int>(BranchMode::STATIC_CONST_BRANCH_MODE);
            attrValue = *scalarValue.begin();
        }
    } else if (scalarValue.size() > 1) {
        if (scalarValue.count(kNonImmediateSentinel) == 0) {
            branchMode = static_cast<int>(BranchMode::CONST_BRANCH_MODE);
        }
    }

    // 3. 刷新新的COA宏
    APASS_LOG_INFO_F(Elements::Operation, "BuildNewCoa update dynScalar[%s] with isConst=%d, value=%d.",
                     dynParamExpr.c_str(), branchMode, attrValue);
    dynScalar.get() = coaExpr.BuildMaybeConstCoa(branchMode, attrValue);
    return SUCCESS;
}

inline int GetCoaIndex(const DynParamInfo& paramInfo)
{
    if (paramInfo.type == DynParamInfoType::VALID_SHAPE) {
        return ((paramInfo.tensorBaseAddrCoaIndex + 1) + VALID_SHAPE_INDEX_ORDER * paramInfo.dimSize +
                paramInfo.dimIndex);
    }
    if (paramInfo.type == DynParamInfoType::OFFSET) {
        return ((paramInfo.tensorBaseAddrCoaIndex + 1) + OFFSET_INDEX_ORDER * paramInfo.dimSize + paramInfo.dimIndex);
    }
    return paramInfo.dimIndex;
}

void ReplaceCommonSymbol(Function* leafFunc, std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim)
{
    VectorParamConsistencyChecker checker;
    for (auto& argList : callopArglistOneDim) {
        checker.RegisterCall(argList);
    }
    auto allRes1 = checker.GetAllConsistentIndexGroups();
    APASS_LOG_DEBUG_F(Elements::Operation, "Get all condicate params: %s.", checker.PrintIndexGroups(allRes1).c_str());
    std::map<size_t, size_t> index2GroupId;
    std::map<size_t, std::vector<size_t>> groupId2Index;
    for (size_t i = 0; i < allRes1.size(); i++) {
        for (size_t j = 0; j < allRes1[i].size(); j++) {
            index2GroupId[allRes1[i][j]] = i;
        }
    }
    std::map<std::string, int> symbol2CoaIdx;
    for (const auto& dynParam : leafFunc->GetDynParamTable()) {
        if (dynParam.second.dim.IsValid()) {
            std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynParam.second.dim);
            if (dynParamExpr.find(COA_PREFIX) != 1) {
                continue;
            }
        }
        int coaIndex = GetCoaIndex(dynParam.second);
        symbol2CoaIdx.emplace(dynParam.first, coaIndex);
        APASS_LOG_DEBUG_F(Elements::Operation, "Need Replace symbols %s idx %d", dynParam.first.c_str(), coaIndex);
    }
    std::map<size_t, std::string> index2BaseSymbol;
    for (auto [symbolStr, coaIdx] : symbol2CoaIdx) {
        if (index2GroupId.find(coaIdx) != index2GroupId.end()) {
            if (index2BaseSymbol.find(index2GroupId[coaIdx]) == index2BaseSymbol.end()) {
                leafFunc->GetMutableDynParam(symbolStr).isBaseParam = true;
                index2BaseSymbol[index2GroupId[coaIdx]] = symbolStr;
                APASS_LOG_INFO_F(Elements::Operation, "Mark coaIndex[%d] groupId[%zu] symbolStr[%s] as baseParam",
                                 coaIdx, index2GroupId[coaIdx], symbolStr.c_str());
            } else {
                leafFunc->GetMutableDynParam(symbolStr).replacedSymbol = index2BaseSymbol[index2GroupId[coaIdx]];
                APASS_LOG_INFO_F(
                    Elements::Operation, "Replace coaIndex[%d] groupId[%zu] symbolStr[%s] with baseParam[%s]", coaIdx,
                    index2GroupId[coaIdx], symbolStr.c_str(), index2BaseSymbol[index2GroupId[coaIdx]].c_str());
            }
        }
    }
}

void ReBuildConcreteParam(Function* leafFunc, std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim)
{
    std::map<std::string, int> concreteParamCoaIdx;
    for (auto& dynParam : leafFunc->GetDynParamTable()) {
        if (dynParam.second.dim.IsValid()) {
            std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynParam.second.dim);
            if (dynParamExpr.find(COA_PREFIX) != 1) {
                continue;
            }
        }
        if (!(dynParam.second.isBaseParam) && !(dynParam.second.replacedSymbol.empty())) {
            continue;
        }
        auto coaIdx = GetCoaIndex(dynParam.second);
        APASS_LOG_DEBUG_F(Elements::Operation, "Get concrete symbols %s idx %d", dynParam.first.c_str(), coaIdx);
        std::set<int> scalarValue;
        auto isConstParam = [&callopArglistOneDim, &scalarValue](int argIdx) {
            for (auto& argList : callopArglistOneDim) {
                auto& callopAttr = argList[argIdx];
                if (!callopAttr.IsImmediate()) {
                    scalarValue.insert(-2);
                } else {
                    scalarValue.insert(callopAttr.Concrete());
                }
                if (scalarValue.size() > 1) {
                    return false;
                }
            }
            return true;
        };
        if (!isConstParam(coaIdx) || *scalarValue.begin() < 0) {
            continue;
        }
        leafFunc->GetMutableDynParam(dynParam.first).dim = *scalarValue.begin();
    }
}

Status DynAttrToStatic::TryRemoveDynAttr(Function* leafFunc, std::vector<Operation*> callList)
{
    // 1. 为leafFunc拿到它所有caller的一维的callopArglistOneDim
    std::vector<std::vector<SymbolicScalar>> callopArglistOneDim;
    for (size_t i = 0; i < callList.size(); i++) {
        auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
        callopArglistOneDim.push_back(callop->GetLinearArgList());
    }
    for (auto& callop : callList) {
        auto rootFunction = callop->BelongTo();
        for (auto& inCast : rootFunction->GetIncast()) {
            rootInOutCast_.insert(inCast->GetRawMagic());
        }
        for (auto& outCast : rootFunction->GetOutcast()) {
            rootInOutCast_.insert(outCast->GetRawMagic());
        }
    }
    // 2. 依次为leafFunc的所有op拿到所有动态attr，为每个动态attr刷新coa宏
    auto operationViewer = leafFunc->Operations(false);
    for (size_t j = 0; j < operationViewer.size(); j++) {
        auto& op = operationViewer[j];
        std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = GetOpDynamicAttributeList(op);
        for (auto dynScalar : dynScalarList) {
            if (BuildNewCoa(dynScalar, callopArglistOneDim) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "TryRemoveDynAttr failed to execute BuildNewCoa for op [%d][%s].", op.GetOpMagic(),
                                  op.GetOpcodeStr().c_str());
                return FAILED;
            }
        }
        BuildParamAddr(op);
    }

    // 3. 为dynParam的赋值刷新coa宏
    ReplaceCommonSymbol(leafFunc, callopArglistOneDim);
    ReBuildConcreteParam(leafFunc, callopArglistOneDim);
    return SUCCESS;
}

std::pair<int, int> ParseRuntimeGetParamAddr(const std::string& input)
{
    // 默认返回 {-1, -1} 表示不匹配
    std::pair<int, int> paramArgs = {-1, -1};
    // 正则：严格匹配 RUNTIME_GET_PARAM_ADDR(xxx, 数字, 数字)
    std::regex pattern(R"(RUNTIME_GET_PARAM_ADDR\s*\(\s*[^,]+,\s*(\d+)\s*,\s*(\d+)\s*\))");
    std::smatch matches;
    if (std::regex_search(input, matches, pattern)) {
        try {
            paramArgs.first = std::stoi(matches[1].str());
            paramArgs.second = std::stoi(matches[2].str());
        } catch (...) {
            // 解析失败也返回 {-1,-1}
            return {-1, -1};
        }
    }
    return paramArgs;
}

// Helper function to set paramAddr attribute for a tensor
void UpdateTensorParamAddr(std::shared_ptr<LogicalTensor>& tensor, const std::set<int>& inOutCast)
{
    IRBuilder builder;
    std::map<TensorAddrKey, SymbolicScalar> paramAddrMap;
    tensor->GetAttr<std::map<TensorAddrKey, SymbolicScalar>>(TensorAttributeKey::tensorAddr, paramAddrMap);
    for (auto& paramAddr : paramAddrMap) {
        auto paramArgs = ParseRuntimeGetParamAddr(paramAddr.second.Dump());
        int aiCpuFlag = inOutCast.count(tensor->GetRawMagic()) ? 3 : 2;
        if (paramAddr.second.IsExpression() && paramArgs.first != -1 && paramArgs.second != -1) {
            paramAddr.second = GET_PARAM_ADDR_MAYBE_CONST(
                builder.CreateConstInt(static_cast<int64_t>(aiCpuFlag)),
                builder.CreateConstInt(static_cast<int64_t>(0)),
                builder.CreateConstInt(static_cast<int64_t>(paramArgs.first)),
                builder.CreateConstInt(static_cast<int64_t>(paramArgs.second)));
        }
    }
    tensor->SetAttr<std::map<TensorAddrKey, SymbolicScalar>>(TensorAttributeKey::tensorAddr, paramAddrMap);
}

void DynAttrToStatic::BuildParamAddr(Operation& op)
{
    for (auto& iOperand : op.GetIOperands()) {
        if (visitedTensors_.count(iOperand)) {
            continue;
        }
        if (iOperand->HasAttr(TensorAttributeKey::tensorAddr)) {
            UpdateTensorParamAddr(iOperand, rootInOutCast_);
            visitedTensors_.insert(iOperand);
        }
    }
    for (auto& oOperand : op.GetOOperands()) {
        if (visitedTensors_.count(oOperand)) {
            continue;
        }
        if (oOperand->HasAttr(TensorAttributeKey::tensorAddr)) {
            UpdateTensorParamAddr(oOperand, rootInOutCast_);
            visitedTensors_.insert(oOperand);
        }
    }
}

Status DynAttrToStatic::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "==============> Start DynAttrToStatic.");
    for (auto& inCast : function.GetIncast()) {
        rootInOutCast_.insert(inCast->GetRawMagic());
    }
    for (auto& outCast : function.GetOutcast()) {
        rootInOutCast_.insert(outCast->GetRawMagic());
    }
    // 1. 遍历所有rootFunc，找到每个leaf的所有caller，生成leaf2Caller map
    if (BuildLeafToCaller(&function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to call BuildLeafToCaller.");
        return FAILED;
    }

    // 2. 遍历leaf2Caller，尝试为每个leaf消除动态attributes
    for (const auto& pair : leaf2Caller) {
        if (TryRemoveDynAttr(pair.first, pair.second) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to call TryRemoveDynAttr for leafFunc %s.",
                              pair.first->GetRawName().c_str());
            return FAILED;
        }
    }

    APASS_LOG_INFO_F(Elements::Operation, "==============> End DynAttrToStatic.");
    return SUCCESS;
}

Status DynAttrToStatic::GetTileFunction(Function* function, std::unordered_set<Function*>& tileFunctionSet)
{
    DumpFunctionUtils utils;
    return utils.GetTileFunction(function, tileFunctionSet);
}

Status DynAttrToStatic::DumpFunctionJson(Function& function, const std::string& logFolder, bool beforeFunction)
{
    DumpFunctionUtils utils;
    return utils.DumpTileFunctionsJson(function, logFolder, beforeFunction,
                                       [this](Function& f, const std::string& folder, bool before) {
                                           return Pass::DumpFunctionJson(f, folder, before);
                                       });
}

Status DynAttrToStatic::PrintFunction(Function& function, const std::string& logFolder, bool beforeFunction)
{
    DumpFunctionUtils utils;
    return utils.PrintTileFunctions(
        function, logFolder, beforeFunction,
        [this](Function& f, const std::string& folder, bool before) { return Pass::PrintFunction(f, folder, before); });
}
} // namespace tile_fwk
} // namespace npu
