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
 * \file backend.h
 * \brief
 */

#pragma once
#include <sstream>
#include <unordered_set>
#include "interface/machine/host/machine_task.h"
#include "interface/cache/function_cache.h"
namespace npu::tile_fwk {
class ExprBatchGenerator;

MachineTask* GenCode(MachineTask* task, FunctionCache& cache);

struct Linker {
    SymbolicSymbolTable& symbolTable_;

    DyndevFunctionAttribute::FunctionGroup& funcGroup_;
    DyndevFunctionAttribute::ExpressionTableDictGroup& exprTableDictGroup_;

    Linker(SymbolicSymbolTable& symbolTable, DyndevFunctionAttribute::FunctionGroup& funcGroup,
           DyndevFunctionAttribute::ExpressionTableDictGroup& exprTableDictGroup)
        : symbolTable_(symbolTable), funcGroup_(funcGroup), exprTableDictGroup_(exprTableDictGroup)
    {}

    void AddSymbol(SymbolicScalar& ss) { symbolTable_.AddSymbol(ss); }

    const DyndevFunctionAttribute::ExpressionTableDictGroup& GetExpressionTableDictGroup() const
    {
        return exprTableDictGroup_;
    }
    DyndevFunctionAttribute::ExpressionTableDictGroup& GetExpressionTableDictGroup() { return exprTableDictGroup_; }

    static std::string GetTitle(Function* func)
    {
        std::string title = "name=" + func->GetRawName() + " hash=" + std::to_string(func->GetFunctionHash().GetHash());
        return title;
    }

    void AddPrimaryExpressionForLoopBes(Function* func, const SymbolicScalar& ss)
    {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.loopList.InsertAndGetIndex(func);
        std::string key = SymbolicExpressionTable::GetExprKeyLoopBes(funcKey);

        auto& exprTable = exprTableDictGroup_.loopBesDict[func];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void AddPrimaryExpressionForLoopPathCond(Function* func, const SymbolicScalar& ss)
    {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.loopPathList.InsertAndGetIndex(func);
        auto condKey = funcGroup_.loopPathCondList[func].InsertAndGetIndex(ss.Raw());
        std::string key = SymbolicExpressionTable::GetExprKeyLoopPathCond(funcKey, condKey);

        auto& exprTable = exprTableDictGroup_.loopPathCondDict[func][ss.Raw()];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void AddPrimaryExpressionForDevRootCoa(Function* func, const SymbolicScalar& ss)
    {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.devRootList.InsertAndGetIndex(func);
        std::string key = SymbolicExpressionTable::GetExprKeyDevRootCoa(funcKey);

        auto& exprTable = exprTableDictGroup_.devRootCoaDict[func];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void AddPrimaryExpressionForDevLeafOp(Function* func, Operation* op, const SymbolicScalar& ss)
    {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.devLeafList.InsertAndGetIndex(func);
        auto opKey = funcGroup_.devLeafOpList[func].InsertAndGetIndex(op);
        std::string key = SymbolicExpressionTable::GetExprKeyDevLeafOp(funcKey, opKey);

        auto& exprTable = exprTableDictGroup_.devLeafOpDict[func][op];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void SetMainBlockExpressionForDevRootCoa(Function* func, const SymbolicScalar& ss)
    {
        auto& exprTable = exprTableDictGroup_.devRootCoaDict[func];
        exprTable.mainBlockScalar_ = ss;
    }

    SymbolicExpressionTable* LookupDevRootCoa(Function* func)
    {
        if (exprTableDictGroup_.devRootCoaDict.count(func)) {
            return &exprTableDictGroup_.devRootCoaDict[func];
        } else {
            return nullptr;
        }
    }

    SymbolicExpressionTable* LookupLoopBes(Function* func)
    {
        if (exprTableDictGroup_.loopBesDict.count(func)) {
            return &exprTableDictGroup_.loopBesDict[func];
        }
        return nullptr;
    }

    SymbolicSymbolTable* GetSymbolTable() { return &symbolTable_; }

private:
    void AddSymbolFromExpression(const SymbolicScalar& ss) { symbolTable_.AddSymbolFromExpression(ss); }
};

// Force link library compiler as nothing depends on it.
void ForceLinkLibraryCompiler();

struct ValDependTensorMeta {
    std::unordered_map<std::string, bool> tensorNameToDependCore;
    std::unordered_map<RawSymbolicScalarPtr, bool> valDependMap;
    bool disableCtrlFlowCache{false};
};

void InsertWaitCoreStart(SymbolicExpressionTable* exprTable, std::ostringstream& controlFlowOss,
                         ValDependTensorMeta& valDependTensorMeta, int indent);

// Functions used by ir_backend.cpp / control_flow_codegen.cpp
void FindAllExpression(FunctionCache& cache, Linker& linker, Function* func);
void BuildControlFlow(FunctionCache& cache, Linker& linker, const std::string& sectionName, Function* func,
                      std::unordered_map<int, int>& slotIdxMapping, DyndevFunctionAttribute::FunctionGroup& group,
                      std::unordered_map<Function*, Function*>& rootTileDict, std::ostringstream& controlFlowOss,
                      std::ostringstream& expressionOss, std::ostringstream& exprHeaderOss, int indent,
                      const std::string& expName, std::vector<std::string>& exprSrcFiles,
                      ValDependTensorMeta& valDependTensorMeta);
std::vector<Function*> GetCalleeList(FunctionCache& cache, Function* func);
std::string BuildControlFlowCallee(Function* func, int indent);
bool NeedCrossDie(Function* func, bool isLoop = false);
void GetReadyOnHostTensorsSet(std::unordered_set<int>& readyOnHostTensorsSet);
void BuildControlFlowHeader(ExprBatchGenerator& generator, Linker& linker, Function* func,
                            const std::string& sectionName, const std::string& expName,
                            std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                            std::ostringstream& exprHeaderOss, ValDependTensorMeta& valDependTensorMeta);
void BuildControlFlowFooter(ExprBatchGenerator& generator, std::ostringstream& controlFlowOss,
                            std::ostringstream& exprHeaderOss, int indent);
} // namespace npu::tile_fwk
