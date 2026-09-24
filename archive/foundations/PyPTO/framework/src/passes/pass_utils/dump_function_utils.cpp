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
 * \file dump_function_utils.cpp
 * \brief 实现遍历 TileFunction 并调用基类方法的公共工具
 */

#include "passes/pass_utils/dump_function_utils.h"
#include "interface/operation/opcode.h"
#include "interface/program/program.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "DumpFunctionUtils"

namespace npu {
namespace tile_fwk {

Status DumpFunctionUtils::GetCallee(const Operation& callop, Function*& callFunc)
{
    auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop.GetOpAttribute());
    callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
    if (callFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Get callee function %s failed.",
                          callopAttr->GetCalleeMagicName().c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status DumpFunctionUtils::GetTileFunction(Function* function, std::unordered_set<Function*>& tileFunctionSet)
{
    for (auto callop : function->GetCallopList()) {
        Function* nextFunc = nullptr;
        if (GetCallee(*callop, nextFunc) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetTileFunction, currFunc: %s, %s[%d] GetCallee failed.",
                              function->GetRawName().c_str(), callop->GetOpcodeStr().c_str(), callop->GetOpMagic());
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Function, "GetTileFunction, %s --%s[%d]--> %s", function->GetRawName().c_str(),
                          callop->GetOpcodeStr().c_str(), callop->GetOpMagic(), nextFunc->GetRawName().c_str());
        if (nextFunc->GetGraphType() == GraphType::TILE_GRAPH) {
            tileFunctionSet.emplace(nextFunc);
        } else {
            if (GetTileFunction(nextFunc, tileFunctionSet) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "GetTileFunction, currFunc: %s, nextFunc: %s, recursive search failed",
                                  function->GetRawName().c_str(), nextFunc->GetRawName().c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status DumpFunctionUtils::DumpTileFunctionsJson(Function& function, const std::string& logFolder, bool beforeFunction,
                                                std::function<Status(Function&, const std::string&, bool)> dumpFunc)
{
    std::unordered_set<Function*> tileFunctionSet;
    if (GetTileFunction(&function, tileFunctionSet) != SUCCESS) {
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Obtained a total of %zu tileFunctions", tileFunctionSet.size());
    for (auto tileFunc : tileFunctionSet) {
        APASS_LOG_DEBUG_F(Elements::Function, "Dump tileFunction[%s] json", tileFunc->GetRawName().c_str());
        if (dumpFunc(*tileFunc, logFolder, beforeFunction) != SUCCESS) {
            return FAILED;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Dump function[%s] json finished.", function.GetRawName().c_str());
    return SUCCESS;
}

Status DumpFunctionUtils::PrintTileFunctions(Function& function, const std::string& logFolder, bool beforeFunction,
                                             std::function<Status(Function&, const std::string&, bool)> printFunc)
{
    std::unordered_set<Function*> tileFunctionSet;
    if (GetTileFunction(&function, tileFunctionSet) != SUCCESS) {
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Obtained a total of %zu tileFunctions", tileFunctionSet.size());
    for (auto tileFunc : tileFunctionSet) {
        APASS_LOG_DEBUG_F(Elements::Function, "Print tileFunction[%s]", tileFunc->GetRawName().c_str());
        if (printFunc(*tileFunc, logFolder, beforeFunction) != SUCCESS) {
            return FAILED;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Print function[%s] finished.", function.GetRawName().c_str());
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
