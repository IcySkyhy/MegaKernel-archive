/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/host/control_flow_codegen.h"

#include <algorithm>
#include <iomanip>
#include <unordered_set>

#include "interface/configs/config_manager.h"
#include "interface/machine/device/tilefwk/core_func_data.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "machine/host/expr_generator.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
namespace {
enum ParallelMode {
    DEFAULT = 0,
    PARALLEL,
};
} // namespace

std::vector<Function*> GetCalleeList(FunctionCache& cache, Function* func)
{
    std::vector<Function*> calleeList;

    std::vector<std::shared_ptr<CallOpAttribute>> callopAttrList = func->GetCallopAttrList();
    for (auto& callopAttr : callopAttrList) {
        auto hash = callopAttr->GetCalleeHash();
        Function* cacheFunction = cache.GetCacheFunction(hash);
        if (cacheFunction != nullptr) {
            calleeList.push_back(cacheFunction);
        } else {
            MACHINE_LOGE(HostBackEndErr::FUNCTION_CACHE_HASH_MISS, "Cannot find cache %lu", hash.GetHash());
        }
    }
    return calleeList;
}

bool TryGetRuntimeSlot(int slot, const std::unordered_map<int, int>& slotIdxMapping, int& runtimeSlot)
{
    auto iter = slotIdxMapping.find(slot);
    if (iter == slotIdxMapping.end()) {
        return false;
    }
    runtimeSlot = iter->second;
    return true;
}

std::string BuildControlFlowCallee(Function* func, int ident)
{
    std::ostringstream oss;
    auto span = func->GetSpan();
    if (!span.IsUnknown()) {
        oss << std::string(ident, ' ') << "// " << span.ToString() << "\n";
    }
    oss << std::string(ident, ' ') << "// "
        << "#name: " << func->GetRawName() << " #hash: " << func->GetFunctionHash()
        << " #magic: " << func->GetFuncMagic() << "\n";
    return oss.str();
}

ParallelMode GetFunctionParallelMode(Function* func)
{
    if (func->GetDynloopAttribute() != nullptr && func->GetDynloopAttribute()->parallel) {
        return ParallelMode::PARALLEL;
    }
    return ParallelMode::DEFAULT;
}

void InsertWaitCoreStartForLoopBounds(const std::shared_ptr<DynloopFunctionAttribute>& attr,
                                      std::ostringstream& controlFlowOss, ValDependTensorMeta& valDependTensorMeta,
                                      int indent)
{
    const SymbolicScalar* loopBounds[] = {&attr->Begin(), &attr->End(), &attr->Step()};
    for (const SymbolicScalar* boundExpr : loopBounds) {
        if (!boundExpr->IsValid() || boundExpr->IsImmediate()) {
            continue;
        }
        if (SymbolicExpressionTable::CheckExprDependCore(boundExpr->Raw(), valDependTensorMeta.tensorNameToDependCore,
                                                         valDependTensorMeta.valDependMap)) {
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "WaitAicoreStart(startArgs);\n";
            return;
        }
    }
}

void GenerateExpression(SymbolicExpressionTable* exprTable, int devRootKey, const std::string& expName,
                        std::vector<std::string>& exprSrcFiles, std::ostringstream& controlFlowOss,
                        std::ostringstream& exprHeaderOss, int indent, const GetInputCse* getInputCse)
{
    const auto& primaryExprs = exprTable->GetPrimaryExpressionSet();
    size_t totalExprs = primaryExprs.size();
    std::string outputDir = config::GetEmitPath("kernel_aicpu");
    ExprBatchGenerator generator(outputDir, devRootKey, totalExprs);
    generator.GenerateBatchFile(exprTable, controlFlowOss, exprHeaderOss, expName, primaryExprs, exprSrcFiles, indent,
                                devRootKey, getInputCse);
}

bool NeedCrossDie(Function* func, bool isLoop)
{
    if ((Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) &&
        (!isLoop || (GetFunctionParallelMode(func) == ParallelMode::PARALLEL))) {
        return true;
    }
    return false;
}

void BuildControlFlowHeader(ExprBatchGenerator& generator, Linker& linker, Function* func,
                            const std::string& sectionName, const std::string& expName,
                            std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                            std::ostringstream& exprHeaderOss, ValDependTensorMeta& valDependTensorMeta)
{
    controlFlowOss << "#define __TILE_FWK_AICPU__ 1\n"
                   << "#include <stdint.h>\n"
                   << "#include \"" << expName << "\"\n"
                   << "#include \"tilefwk/aikernel_data.h\"\n"
                   << "#include \"tilefwk/aicpu_runtime.h\"\n"
                   << "#include \"tilefwk/aicpu_distributed.h\"\n"
                   << "#include \"control_flow_expr_table.h\"\n";
    generator.HeaderFileBegin(exprHeaderOss);
    expressionOss << "\n/* Symbol table list */\n" << linker.GetSymbolTable()->BuildSymbolList();
    const std::vector<std::string>& inputNameList = Program::GetInstance().GetTensorSlotManager()->GetInputNameList();
    const std::vector<std::string>& outputNameList = Program::GetInstance().GetTensorSlotManager()->GetOutputNameList();
    std::unordered_set<int> readyOnHostTensorsSet;
    GetReadyOnHostTensorsSet(readyOnHostTensorsSet);
    expressionOss << "\n/* Input tensor list */\n";
    for (size_t idx = 0; idx < inputNameList.size(); idx++) {
        const auto inputName = AddArgPrefix(inputNameList[idx]);
        expressionOss << "#define " << inputName << " " << idx << "\n";
        valDependTensorMeta.tensorNameToDependCore[inputName] = (readyOnHostTensorsSet.count(idx) == 0);
    }
    expressionOss << "\n/* Output tensor list */\n";
    for (size_t idx = 0; idx < outputNameList.size(); idx++) {
        expressionOss << "#define " << AddArgPrefix(outputNameList[idx]) << " " << idx + inputNameList.size() << "\n";
    }
    controlFlowOss << "#define LOOP(idx, b, e, s) for (int64_t idx = (b), idxEnd = (e), idxStep = (s); idx < "
                      "idxEnd; idx += idxStep)\n"
                   << "namespace npu::tile_fwk {\n"
                   << BuildControlFlowCallee(func, 0) << "__attribute__((section(\"" << sectionName << ".entry"
                   << "\")))\n"
                   << "uint64_t ControlFlowEntry(void *ctx, int64_t *symbolTable, RuntimeCallEntryType "
                      "runtimeCallList[], DevStartArgsBase *startArgs) {\n";
}

void BuildControlFlowFooter(ExprBatchGenerator& generator, std::ostringstream& controlFlowOss,
                            std::ostringstream& exprHeaderOss, int indent)
{
    controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' '
                   << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_FINISH); // Notify finish \n";
    controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "return 0;\n";
    controlFlowOss << "}\n";
    controlFlowOss << "} // namespace npu::tile_fwk\n";
    generator.HeaderFileEnd(exprHeaderOss);
}

void MarkValueDependDisableCache(ControlFlowEmitCtx& ctx, Function* keyFunc)
{
    auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
    if (currDynFuncAttr->valueDependDescDict.count(keyFunc) == 0) {
        return;
    }
    auto valueDependDesc = currDynFuncAttr->valueDependDescDict[keyFunc];
    if (valueDependDesc.getInputDataCount + valueDependDesc.getTensorDataCount != 0) {
        // Value-depend: disable host ctrl-flow cache only; do not emit CACHESTOP stitch.
        ctx.valDependTensorMeta.disableCtrlFlowCache = true;
    }
}

void EmitSlotMarkNeedAlloc(ControlFlowEmitCtx& ctx, int runtimeSlot, int indent)
{
    ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_SlotMarkNeedAlloc(" << runtimeSlot << ");\n";
}

void EmitPathAssembleNeedAlloc(ControlFlowEmitCtx& ctx, Function* func, int indent)
{
    auto scope = func->GetSlotScope();
    auto dynAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
    for (auto slot : scope->constructAssembleSlotList) {
        int runtimeSlot = -1;
        if (!TryGetRuntimeSlot(slot, ctx.slotIdxMapping, runtimeSlot)) {
            continue;
        }
        if (dynAttr->constructAssembleNeedAllocRuntimeSlots.count(runtimeSlot) == 0) {
            continue;
        }
        EmitSlotMarkNeedAlloc(ctx, runtimeSlot, indent);
    }
}

void EmitExecuteAssembleNeedAlloc(ControlFlowEmitCtx& ctx, Function* tile, int indent)
{
    auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
    if (currDynFuncAttr->inoutLink.ioslotDict.count(tile) == 0) {
        return;
    }
    const IncastOutcastSlot& ioslot = currDynFuncAttr->inoutLink.ioslotDict.at(tile);
    const std::unordered_set<int> assembleSlotIndexSet(currDynFuncAttr->inoutLink.assembleSlotIndexList.begin(),
                                                       currDynFuncAttr->inoutLink.assembleSlotIndexList.end());
    ForEachNeedAllocAssembleOutcastSlot(tile, ioslot, assembleSlotIndexSet, [&](int slot) {
        int runtimeSlot = -1;
        if (!TryGetRuntimeSlot(slot, ctx.slotIdxMapping, runtimeSlot)) {
            return;
        }
        if (currDynFuncAttr->constructAssembleNeedAllocRuntimeSlots.count(runtimeSlot) == 0) {
            return;
        }
        EmitSlotMarkNeedAlloc(ctx, runtimeSlot, indent);
    });
}

void EmitIoArgDefines(ControlFlowEmitCtx& ctx)
{
    const std::vector<std::string>& inputNameList = Program::GetInstance().GetTensorSlotManager()->GetInputNameList();
    const std::vector<std::string>& outputNameList = Program::GetInstance().GetTensorSlotManager()->GetOutputNameList();
    std::unordered_set<int> readyOnHostTensorsSet;
    GetReadyOnHostTensorsSet(readyOnHostTensorsSet);

    ctx.expressionOss << "\n/* Input tensor list */\n";
    for (size_t idx = 0; idx < inputNameList.size(); idx++) {
        const auto inputName = AddArgPrefix(inputNameList[idx]);
        ctx.expressionOss << "#define " << inputName << " " << idx << "\n";
        ctx.valDependTensorMeta.tensorNameToDependCore[inputName] = (readyOnHostTensorsSet.count(idx) == 0);
    }

    ctx.expressionOss << "\n/* Output tensor list */\n";
    for (size_t idx = 0; idx < outputNameList.size(); idx++) {
        ctx.expressionOss << "#define " << AddArgPrefix(outputNameList[idx]) << " " << idx + inputNameList.size()
                          << "\n";
    }
}

void BuildControlFlowDynamic(ControlFlowEmitCtx& ctx, Function* func, int indent)
{
    ctx.controlFlowOss << "#define __TILE_FWK_AICPU__ 1\n"
                       << "#include <stdint.h>\n"
                       << "#include \"" << ctx.expName << "\"\n"
                       << "#include \"tilefwk/aikernel_data.h\"\n"
                       << "#include \"tilefwk/aicpu_runtime.h\"\n"
                       << "#include \"tilefwk/aicpu_distributed.h\"\n"
                       << "#include \"control_flow_expr_table.h\"\n";
    ExprBatchGenerator generator(config::GetEmitPath("kernel_aicpu"), 0, 0);
    generator.HeaderFileBegin(ctx.exprHeaderOss);
    ctx.expressionOss << "\n/* Symbol table list */\n" << ctx.linker.GetSymbolTable()->BuildSymbolList();
    EmitIoArgDefines(ctx);
    ctx.controlFlowOss << "#define LOOP(idx, b, e, s) for (int64_t idx = (b), idxEnd = (e), idxStep = (s); idx < "
                          "idxEnd; idx += idxStep)\n"
                       << "namespace npu::tile_fwk {\n"
                       << BuildControlFlowCallee(func, 0) << "__attribute__((section(\"" << ctx.sectionName << ".entry"
                       << "\")))\n"
                       << "uint64_t ControlFlowEntry(void *ctx, int64_t *symbolTable, RuntimeCallEntryType "
                          "runtimeCallList[], DevStartArgsBase *startArgs) {\n";
    if (ctx.getInputCse != nullptr) {
        ExprBatchGenerator::EmitGetInputCseStackInits(ctx.controlFlowOss, *ctx.getInputCse, indent + 1);
    }
    if (NeedCrossDie(func)) {
        ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootGetDieId(" << 0 << ");\n";
    }
    for (auto& callee : GetCalleeList(ctx.cache, func)) {
        BuildControlFlow(ctx, callee, indent + 1);
    }
    ctx.controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' '
                       << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_FINISH); // Notify finish \n";
    ctx.controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "return 0;\n";
    ctx.controlFlowOss << "}\n";
    ctx.controlFlowOss << "} // namespace npu::tile_fwk\n";
    generator.HeaderFileEnd(ctx.exprHeaderOss);
}

bool SupportParallelLoop() { return config::GetRuntimeOption<uint16_t>(DEVICE_SCHED_PARALLELISM) > 1; }

const std::unordered_map<std::string, std::string>* GetInputCseMap(const ControlFlowEmitCtx& ctx)
{
    if (ctx.getInputCse != nullptr && !ctx.getInputCse->keyToName.empty()) {
        return &ctx.getInputCse->keyToName;
    }
    return nullptr;
}

void EmitDynloopCondTree(ControlFlowEmitCtx& ctx, const std::shared_ptr<DynloopFunctionPathNode>& node, int condIndent)
{
    if (!node->cond.IsValid()) {
        BuildControlFlow(ctx, node->root, condIndent);
        return;
    }

    const auto* getInputCseMap = GetInputCseMap(ctx);
    std::string cond = SymbolicExpressionTable::BuildExpression(node->cond.Raw(), getInputCseMap);
    if (node->branchNodeList[1] != nullptr) {
        if (node->branchNodeList[0] != nullptr) {
            ctx.controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "if (" << cond << ") {\n";
            EmitDynloopCondTree(ctx, node->branchNodeList[1], condIndent + 1);
            ctx.controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "} else {\n";
            EmitDynloopCondTree(ctx, node->branchNodeList[0], condIndent + 1);
            ctx.controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "}\n";
        } else {
            EmitDynloopCondTree(ctx, node->branchNodeList[1], condIndent);
        }
        return;
    }
    if (node->branchNodeList[0] != nullptr) {
        EmitDynloopCondTree(ctx, node->branchNodeList[0], condIndent);
        return;
    }
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, false) << "Both conds is nullptr!";
}

void EmitDynamicLoopOpen(ControlFlowEmitCtx& ctx, Function* func, const std::shared_ptr<DynloopFunctionAttribute>& attr,
                         int indent)
{
    const bool parallel = attr->parallel && SupportParallelLoop();
    ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// hash=" << func->GetFunctionHash() << "\n";
    if (attr->submitBeforeLoop) {
        ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' '
                           << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_LOOP_BARRIER); // force submit before LOOP \n";
    }
    MarkValueDependDisableCache(ctx, func);
    InsertWaitCoreStartForLoopBounds(attr, ctx.controlFlowOss, ctx.valDependTensorMeta, indent);

    const auto* getInputCseMap = GetInputCseMap(ctx);
    const std::string iterVar = "VAR_" + attr->iterSymbolName;
    const std::string iterBegin = SymbolicExpressionTable::BuildExpression(attr->Begin().Raw(), getInputCseMap);
    const std::string iterEnd = SymbolicExpressionTable::BuildExpression(attr->End().Raw(), getInputCseMap);
    const std::string iterStep = SymbolicExpressionTable::BuildExpression(attr->Step().Raw(), getInputCseMap);
    ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "LOOP(" << iterVar << ", " << iterBegin << ", "
                       << iterEnd << ", " << iterStep << ") {\n";
    ctx.controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "VALUE_" << attr->iterSymbolName << " = "
                       << iterVar << ";\n";
    if (parallel) {
        ctx.controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' '
                           << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN); // entry parallel for loop \n";
    }
    if (NeedCrossDie(func, true)) {
        ctx.controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "RUNTIME_CalcLoopDieId("
                           << attr->iterSymbolName << ", " << iterVar << ", " << iterEnd << ", " << iterStep << ","
                           << DIE_NUM << ");\n";
    }
}

void AssertDynloopCalleeMatchesPaths(FunctionCache& cache, Function* func,
                                     const std::shared_ptr<DynloopFunctionAttribute>& attr)
{
    std::vector<Function*> calleeList = GetCalleeList(cache, func);
    std::sort(calleeList.begin(), calleeList.end());
    std::vector<Function*> pathRootList;
    pathRootList.reserve(attr->pathList.size());
    for (size_t i = 0; i < attr->pathList.size(); i++) {
        pathRootList.push_back(attr->pathList[i].root);
    }
    std::sort(pathRootList.begin(), pathRootList.end());
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, calleeList == pathRootList)
        << "calleeList size:" << calleeList.size() << " pathRootList size:" << pathRootList.size();
}

void EmitDynamicLoopClose(ControlFlowEmitCtx& ctx, Function* func,
                          const std::shared_ptr<DynloopFunctionAttribute>& attr, int indent)
{
    if (NeedCrossDie(func, true)) {
        ctx.controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "RUNTIME_ClearLoopDieId("
                           << attr->iterSymbolName << ");\n";
    }
    if (attr->parallel && SupportParallelLoop()) {
        ctx.controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' '
                           << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_PARALLEL_FOR_END); // leave parallel for loop \n";
    }
    ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "}\n";
}

void BuildControlFlowDynamicLoop(ControlFlowEmitCtx& ctx, Function* func, int indent)
{
    auto attr = func->GetDynloopAttribute();
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, attr != nullptr) << "attr is nullptr!";
    EmitDynamicLoopOpen(ctx, func, attr, indent);
    AssertDynloopCalleeMatchesPaths(ctx.cache, func, attr);
    EmitDynloopCondTree(ctx, attr->BuildPathNode(), indent + 1);
    EmitDynamicLoopClose(ctx, func, attr, indent);
}

void BuildControlFlowDynamicLoopPath(ControlFlowEmitCtx& ctx, Function* func, int indent)
{
    ctx.controlFlowOss << BuildControlFlowCallee(func, indent * TABSIZE);
    EmitPathAssembleNeedAlloc(ctx, func, indent);
    for (auto& callee : GetCalleeList(ctx.cache, func)) {
        BuildControlFlow(ctx, callee, indent + 1);
    }
}

void BuildControlFlowTileGraph(ControlFlowEmitCtx& ctx, Function* func, int indent)
{
    ctx.controlFlowOss << BuildControlFlowCallee(func, indent * TABSIZE);
    Function* root = func->GetRootFunction();
    ctx.rootTileDict[root] = func;
    BuildControlFlow(ctx, root, indent);
}

void BuildControlFlowExecuteGraph(ControlFlowEmitCtx& ctx, Function* func, int indent)
{
    if (ctx.group.devRootList.count(func) <= 0) {
        return;
    }
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, ctx.rootTileDict.count(func)) << "Function not found in rootTileDict";
    Function* tile = ctx.rootTileDict[func];
    MarkValueDependDisableCache(ctx, tile);

    int devRootKey = ctx.group.devRootList.GetIndex(func);
    ctx.controlFlowOss << BuildControlFlowCallee(func, indent * TABSIZE);
    EmitExecuteAssembleNeedAlloc(ctx, tile, indent);
    ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "uint64_t *exprList" << devRootKey
                       << " = (uint64_t *)RUNTIME_RootAlloc(" << devRootKey << "ULL);\n";

    SymbolicExpressionTable* exprTable = ctx.linker.LookupDevRootCoa(func);
    if (exprTable != nullptr) {
        InsertWaitCoreStart(exprTable, ctx.controlFlowOss, ctx.valDependTensorMeta, indent);
        GenerateExpression(exprTable, devRootKey, ctx.expName, ctx.exprSrcFiles, ctx.controlFlowOss, ctx.exprHeaderOss,
                           indent, ctx.getInputCse);
    }
    if (NeedCrossDie(func)) {
        ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootSetDieId(" << devRootKey << "ULL);\n";
    }
    ctx.controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootStitch(" << devRootKey << "ULL);\n";
}

void BuildControlFlow(ControlFlowEmitCtx& ctx, Function* func, int indent)
{
    auto funcType = func->GetFunctionType();
    if (funcType == FunctionType::DYNAMIC) {
        BuildControlFlowDynamic(ctx, func, indent);
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
        BuildControlFlowDynamicLoop(ctx, func, indent);
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH, GraphType::TENSOR_GRAPH)) {
        BuildControlFlowDynamicLoopPath(ctx, func, indent);
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        BuildControlFlowTileGraph(ctx, func, indent);
    } else if (func->GetGraphType() == GraphType::EXECUTE_GRAPH) {
        BuildControlFlowExecuteGraph(ctx, func, indent);
    } else {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, false)
            << "Impossible function type: " << GetFunctionTypeNameDict().Find(funcType);
    }
}

void BuildControlFlow(FunctionCache& cache, Linker& linker, const std::string& sectionName, Function* func,
                      std::unordered_map<int, int>& slotIdxMapping, DyndevFunctionAttribute::FunctionGroup& group,
                      std::unordered_map<Function*, Function*>& rootTileDict, std::ostringstream& controlFlowOss,
                      std::ostringstream& expressionOss, std::ostringstream& exprHeaderOss, int indent,
                      const std::string& expName, std::vector<std::string>& exprSrcFiles,
                      ValDependTensorMeta& valDependTensorMeta)
{
    ControlFlowEmitCtx ctx{
        cache,         linker,  sectionName,  slotIdxMapping,      group,  rootTileDict, controlFlowOss, expressionOss,
        exprHeaderOss, expName, exprSrcFiles, valDependTensorMeta, nullptr};
    BuildControlFlow(ctx, func, indent);
}

} // namespace npu::tile_fwk
