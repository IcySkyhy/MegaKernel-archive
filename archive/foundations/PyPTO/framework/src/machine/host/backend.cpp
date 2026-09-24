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
 * \file backend.cpp
 * \brief
 */

#include "machine/host/backend.h"
#include <dlfcn.h>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "tilefwk/comm_group_recorder.h"
#include "interface/program/program.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_stage_scope.h"
#include "passes/pass_mgr/pass_manager.h"
#include "codegen/codegen.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/utils/parallel_execute.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/compile/aicore_compiler.h"
#include "machine/compile/compile_control_bin.h"
#include "machine/host/expr_generator.h"
#include "machine/host/ir_backend.h"
#include "machine/host/control_flow_codegen.h"
#include "machine/host/dump_host_topo.h"
#include "machine/host/main_block.h"
#include "machine/host/mix_info.h"
#include "ir/scalar_expr.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk::dynamic;
namespace npu::tile_fwk {
constexpr uint32_t STITCH_FUNCTION_MAX_SIZE = 65535;
const std::set<FunctionType> DYNAMIC_FUNC_TYPE_SET = {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP,
                                                      FunctionType::DYNAMIC_LOOP_PATH};
constexpr const char* STAGE_DYNDEV_BUILD_CONTROL_FLOW = "DynDev:BuildControlFlow";
constexpr const char* STAGE_DYNDEV_CONTROL_FLOW_COMPILE = "DynDev:ControlFlowCompile";
constexpr const char* STAGE_DYNDEV_AICORE_KERNEL_COMPILE = "DynDev:AICoreKernelCompile";
constexpr const char* STAGE_DYNDEV_ENCODE = "DynDev:Encode";

void ForceLinkLibraryCompiler() {}

extern "C" int32_t Execute(MachineTask* task, FunctionCache& cache)
{
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) >= CS_TENSOR_GRAPH &&
        config::GetHostOption<int64_t>(COMPILE_STAGE) <= CS_EXECUTE_GRAPH) {
        MACHINE_LOGI("Compile stage terminates after execution graph generation.");
        return 0;
    }
    if (task == nullptr || task->GetFunction() == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "Machine task or function of machine task is null.");
        return 0;
    }
    Function* function = task->GetFunction();
    if (function->IsFunctionType(DYNAMIC_FUNC_TYPE_SET) && function->GetGraphType() == GraphType::TILE_GRAPH) {
        MACHINE_LOGI("The codegen of the current function is executed last");
        // When expression fusion, don't need tile graph codegen.
        return 0;
    }
    (void)GenCode(task, cache);
    /* finish compile add function cache */
    cache.Insert(function->GetFunctionHash(), *function);
    return 0;
}

static void HandleExecuteGraph(FunctionCache& cache, Linker& linker, Function* func);
void FindAllExpression(FunctionCache& cache, Linker& linker, Function* func)
{
    if (func->IsDynloop()) {
        auto dynloopAttr = func->GetDynloopAttribute();
        auto ss = SymbolicScalar(dynloopAttr->iterSymbolName);
        linker.AddSymbol(ss);
    }
    if (func->IsFunctionTypeAndGraphType(
            {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH},
            GraphType::TENSOR_GRAPH)) {
        MACHINE_LOGI("Compile control: %s", func->Dump().c_str());
        for (auto& callee : GetCalleeList(cache, func)) {
            FindAllExpression(cache, linker, callee);
        }
        if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
            auto attr = func->GetDynloopAttribute();
            linker.AddPrimaryExpressionForLoopBes(func, attr->Begin());
            linker.AddPrimaryExpressionForLoopBes(func, attr->End());
            linker.AddPrimaryExpressionForLoopBes(func, attr->Step());

            for (const DynloopFunctionPath& path : attr->GetPathList()) {
                Function* loopPath = path.GetRoot();
                for (auto& cond : path.GetPathCondList()) {
                    linker.AddPrimaryExpressionForLoopPathCond(loopPath, cond.GetCond());
                }
            }
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        MACHINE_LOGI("Compile tile: %s", func->Dump().c_str());
        Function* root = func->GetRootFunction();
        FindAllExpression(cache, linker, root);
    } else if (func->GetGraphType() == GraphType::EXECUTE_GRAPH) {
        HandleExecuteGraph(cache, linker, func);
    } else if (func->GetGraphType() == GraphType::BLOCK_GRAPH) {
        for (auto& op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VEC_DUP) {
                if (op.HasAttr(OpAttributeKey::dynScalar)) {
                    auto dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                    linker.AddPrimaryExpressionForDevLeafOp(func, &op, dynScalar);
                }
            }
        }
    } else {
        ASSERT(DevCommonErr::PARAM_INVALID, false)
            << "Impossible function type: " << GetFunctionTypeNameDict().Find(func->GetFunctionType());
    }
}

static void HandleExecuteGraph(FunctionCache& cache, Linker& linker, Function* func)
{
    MACHINE_LOGI("Compile root: %s", func->Dump().c_str());
    MainBlockCondBulider builder;
    builder.CollectCallopMainBlockConds(func);
    for (auto& callopAttr : func->GetCallopAttrList()) {
        for (auto& arg : callopAttr->GetLinearArgList()) {
            linker.AddPrimaryExpressionForDevRootCoa(func, arg);
        }
        auto hash = callopAttr->GetCalleeHash();
        Function* leafFunc = cache.GetCacheFunction(hash);
        if (leafFunc == nullptr) {
            continue;
        }
        builder.CollectLeafMainBlockConds(leafFunc, callopAttr->GetLinearArgList());
        FindAllExpression(cache, linker, leafFunc);
    }
    for (auto& incast : func->inCasts_) {
        for (auto& arg : incast->GetRawTensor()->GetDynRawShape()) {
            linker.AddPrimaryExpressionForDevRootCoa(func, arg);
        }
    }
    for (auto& outcast : func->outCasts_) {
        for (auto& arg : outcast->GetRawTensor()->GetDynRawShape()) {
            linker.AddPrimaryExpressionForDevRootCoa(func, arg);
        }
    }
    SymbolicScalar cond = builder.BuildMainBlockExpression();
    linker.SetMainBlockExpressionForDevRootCoa(func, cond);
}

static void AlignUpTo(std::vector<uint8_t>& code, int align, uint8_t padding)
{
    while (code.size() % align != 0) {
        code.push_back(padding);
    }
}

static void ReplaceSlotIndex(DyndevFunctionAttribute* attr, std::set<int>& slotUsed,
                             std::unordered_map<int, int>& slotIdxMapping)
{
    IncastOutcastLink& inoutLink = attr->inoutLink;

    for (auto idx : slotUsed) {
        if (!slotIdxMapping.count(idx)) {
            slotIdxMapping.emplace(idx, slotIdxMapping.size());
        }
    }

    auto replaceSlotIdx = [&slotIdxMapping](std::vector<int>& slots) {
        for (int& slot : slots) {
            slot = slotIdxMapping.count(slot) ? slotIdxMapping[slot] : -1;
        }
        slots.erase(std::remove(slots.begin(), slots.end(), -1), slots.end());
    };

    inoutLink.totalSlot = slotIdxMapping.size();
    for (Function* devRoot : attr->funcGroup.devRootList) {
        Function* devTile = attr->rootTileDict[devRoot];

        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, inoutLink.ioslotDict.count(devTile))
            << "Function pointer " << devTile->GetMagicName() << " not found in ioslotDict";
        IncastOutcastSlot& ioslot = inoutLink.ioslotDict[devTile];

        for (auto& incastSlots : ioslot.incastSlot) {
            replaceSlotIdx(incastSlots);
        }

        for (auto& outcastSlots : ioslot.outcastSlot) {
            replaceSlotIdx(outcastSlots);
        }
    }

    replaceSlotIdx(inoutLink.inputSlotIndexList);
    replaceSlotIdx(inoutLink.outputSlotIndexList);
    replaceSlotIdx(inoutLink.assembleSlotIndexList);
    replaceSlotIdx(inoutLink.shmemTensorSlotIndexList);
    replaceSlotIdx(inoutLink.partialUpdateSlotIdexList);
    for (auto& slot : inoutLink.inplaceSlotIndexList) {
        if (slot != -1)
            slot = slotIdxMapping[slot];
    }

    auto replaceSlotIdxForFunc = [replaceSlotIdx](Function* func) {
        std::shared_ptr<TensorSlotScope> scope = func->GetSlotScope();
        if (scope) {
            replaceSlotIdx(scope->constructAssembleSlotList);
        }
    };
    for (auto loopPathFunc : attr->funcGroup.loopPathList) {
        replaceSlotIdxForFunc(loopPathFunc);
    }

    inoutLink.UpdateRuntimeSlotKindSetList();
}

static void MarkUsedSlotsFromInoutLink(const IncastOutcastLink& inoutLink, std::set<int>& slotUsed)
{
    for (int slotIdx : inoutLink.inputSlotIndexList) {
        slotUsed.insert(slotIdx);
    }
    for (int slotIdx : inoutLink.outputSlotIndexList) {
        slotUsed.insert(slotIdx);
    }
    for (int slotIdx : inoutLink.shmemTensorSlotIndexList) {
        slotUsed.insert(slotIdx);
    }
    for (int slotIdx : inoutLink.assembleSlotIndexList) {
        slotUsed.insert(slotIdx);
    }
    // partialUpdateSlotIdexList的数据有问题
}

static void SimplifySlots(DyndevFunctionAttribute* attr, std::unordered_map<int, int>& slotIdxMapping)
{
    IncastOutcastLink& inoutLink = attr->inoutLink;
    std::set<int> slotUsed;

    MarkUsedSlotsFromInoutLink(inoutLink, slotUsed);
    for (Function* devRoot : attr->funcGroup.devRootList) {
        Function* devTile = attr->rootTileDict[devRoot];

        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, inoutLink.ioslotDict.count(devTile))
            << "Function pointer " << devTile->GetMagicName() << " not found in ioslotDict";
        IncastOutcastSlot& ioslot = inoutLink.ioslotDict[devTile];

        for (auto& incastSlots : ioslot.incastSlot) {
            if (incastSlots.empty()) {
                MACHINE_LOGW("devTile: %s", devTile->GetMagicName().c_str());
                continue;
            }
            int32_t simplifiedIncastSlot = -1;
            for (auto& incastSlot : incastSlots) {
                if (slotUsed.count(incastSlot)) {
                    simplifiedIncastSlot = incastSlot;
                    break;
                }
            }
            if (simplifiedIncastSlot != -1) {
                incastSlots.front() = simplifiedIncastSlot;
            }
            incastSlots.resize(1); // meaningless to maintain multi incast slots
            slotUsed.insert(incastSlots.front());
        }
    }

    for (Function* devRoot : attr->funcGroup.devRootList) {
        Function* devTile = attr->rootTileDict[devRoot];

        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, inoutLink.ioslotDict.count(devTile))
            << "Function pointer " << devTile->GetMagicName() << " not found in ioslotDict";
        IncastOutcastSlot& ioslot = inoutLink.ioslotDict[devTile];
        for (auto& outcastSlots : ioslot.outcastSlot) {
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, !outcastSlots.empty()) << "devTile: " << devTile->GetMagicName();
            bool outcastSlotFound = false;
            for (auto& outcastSlot : outcastSlots) {
                outcastSlotFound = outcastSlotFound || slotUsed.count(outcastSlot);
            }
            if (!outcastSlotFound) {
                slotUsed.insert(outcastSlots.front());
            }
        }
    }

    ReplaceSlotIndex(attr, slotUsed, slotIdxMapping);
}

static void BuildSlotRootIncastOutcastDict(DyndevFunctionAttribute* attr)
{
    IncastOutcastLink& inoutLink = attr->inoutLink;
    for (size_t idx = 0; idx < attr->funcGroup.devRootList.size(); idx++) {
        Function* devRoot = attr->funcGroup.devRootList[idx];
        Function* devTile = attr->rootTileDict[devRoot];

        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, inoutLink.ioslotDict.count(devTile))
            << "Function pointer " << devTile->GetMagicName() << " not found in ioslotDict";
        IncastOutcastSlot& ioslot = inoutLink.ioslotDict[devTile];
        for (size_t incastIndex = 0; incastIndex < ioslot.incastSlot.size(); incastIndex++) {
            for (auto& slotIndex : ioslot.incastSlot[incastIndex]) {
                attr->slotRootIncastDict[slotIndex][devRoot] = incastIndex;
            }
        }
        for (size_t outcastIndex = 0; outcastIndex < ioslot.outcastSlot.size(); outcastIndex++) {
            for (auto& slotIndex : ioslot.outcastSlot[outcastIndex]) {
                attr->slotRootOutcastDict[slotIndex][devRoot] = outcastIndex;
            }
        }
    }
}

static void BuildRootFuncKeyDict(DyndevFunctionAttribute* attr)
{
    for (size_t idx = 0; idx < attr->funcGroup.devRootList.size(); idx++) {
        int funcKey = (int)idx;
        Function* devRoot = attr->funcGroup.devRootList[idx];
        attr->rootFuncKeyDict[devRoot] = funcKey;
    }
}

static int GetOrCreateRuntimeSlot(int slot, std::unordered_map<int, int>& slotIdxMapping)
{
    if (!slotIdxMapping.count(slot)) {
        slotIdxMapping.emplace(slot, slotIdxMapping.size());
    }
    return slotIdxMapping.at(slot);
}

static void CollectRootTileDict(FunctionCache& cache, Function* func,
                                std::unordered_map<Function*, Function*>& rootTileDict)
{
    if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        rootTileDict[func->GetRootFunction()] = func;
    }
    for (auto& callee : GetCalleeList(cache, func)) {
        CollectRootTileDict(cache, callee, rootTileDict);
    }
}

static void AddConstructAssembleNeedAllocRuntimeSlot(DyndevFunctionAttribute* attr, int slot,
                                                     std::unordered_map<int, int>& slotIdxMapping)
{
    int runtimeSlot = GetOrCreateRuntimeSlot(slot, slotIdxMapping);
    attr->constructAssembleNeedAllocRuntimeSlots.insert(runtimeSlot);
}

static void CollectConstructAssembleRuntimeSlotsFromFunction(FunctionCache& cache, Function* func,
                                                             DyndevFunctionAttribute* attr,
                                                             std::unordered_map<int, int>& slotIdxMapping)
{
    if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH, GraphType::TENSOR_GRAPH)) {
        auto scope = func->GetSlotScope();
        if (scope != nullptr) {
            for (auto slot : scope->constructAssembleSlotList) {
                AddConstructAssembleNeedAllocRuntimeSlot(attr, slot, slotIdxMapping);
            }
        }
    }
    for (auto& callee : GetCalleeList(cache, func)) {
        CollectConstructAssembleRuntimeSlotsFromFunction(cache, callee, attr, slotIdxMapping);
    }
}

static void BuildConstructAssembleNeedAllocRuntimeSlots(FunctionCache& cache, Function* func,
                                                        DyndevFunctionAttribute* attr,
                                                        std::unordered_map<int, int>& slotIdxMapping)
{
    attr->constructAssembleNeedAllocRuntimeSlots.clear();
    CollectConstructAssembleRuntimeSlotsFromFunction(cache, func, attr, slotIdxMapping);

    const std::unordered_set<int> assembleSlotIndexSet(attr->inoutLink.assembleSlotIndexList.begin(),
                                                       attr->inoutLink.assembleSlotIndexList.end());
    for (Function* devRoot : attr->funcGroup.devRootList) {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, attr->rootTileDict.count(devRoot))
            << "Function not found in rootTileDict";
        Function* tile = attr->rootTileDict[devRoot];
        if (!attr->inoutLink.ioslotDict.count(tile)) {
            continue;
        }
        const IncastOutcastSlot& ioslot = attr->inoutLink.ioslotDict.at(tile);
        ForEachNeedAllocAssembleOutcastSlot(tile, ioslot, assembleSlotIndexSet, [&](int slot) {
            AddConstructAssembleNeedAllocRuntimeSlot(attr, slot, slotIdxMapping);
        });
    }
}

void InsertWaitCoreStart(SymbolicExpressionTable* exprTable, std::ostringstream& controlFlowOss,
                         ValDependTensorMeta& valDependTensorMeta, int indent)
{
    if (exprTable == nullptr) {
        return;
    }
    bool needSync = false;
    const auto& primaryExprs = exprTable->GetPrimaryExpressionSet();
    for (const auto& expr : primaryExprs) {
        if (expr == nullptr) {
            continue;
        }
        if (exprTable->CheckExprDependCore(expr, valDependTensorMeta.tensorNameToDependCore,
                                           valDependTensorMeta.valDependMap)) {
            needSync = true;
            break;
        }
    }
    if (needSync) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "WaitAicoreStart(startArgs);\n";
    }
}

void GetReadyOnHostTensorsSet(std::unordered_set<int>& readyOnHostTensorsSet)
{
    const auto& readyOnHostTensors = config::GetRuntimeOption<std::vector<std::string>>(READY_ON_HOST_TENSORS);
    auto attr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
    auto inputSize = attr->startArgsInputLogicalTensorList.size();
    for (const auto& tensorStr : readyOnHostTensors) {
        size_t i = 0;
        for (; i < inputSize; i++) {
            if (tensorStr == attr->startArgsInputLogicalTensorList[i]->Symbol()) {
                MACHINE_LOGI("Tensor[%zu][%s] is ready on host.", i, tensorStr.c_str());
                readyOnHostTensorsSet.insert(i);
                break;
            }
        }
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, i < inputSize)
            << "Tensor " << tensorStr << " not found in input list, please check [ready_on_host_tensors] config.";
    }
}
static std::string Arm64TargetTool(const std::string& bin)
{
    const char* homePath = std::getenv("ASCEND_HOME_PATH");
    if (homePath == nullptr) {
        return "";
    }
    // use toolchain from CANN for better compatibility as the controlflow will run on aicpu
    return std::string(homePath) + "/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-" + bin;
}

static void FillL2PrefetchInfo(std::shared_ptr<DyndevFunctionAttribute> attr)
{
    uint64_t idx = 0;
    for (auto& param : attr->startArgsInputTensorList) {
        const auto& tensor = param.get();
        auto asc_tensor = tensor.GetStorage();
        if (asc_tensor == nullptr) {
            idx++;
            attr->disableL2List.emplace_back(0);
            continue;
        }
        if (tensor.GetStorage()->GetCachePolicy(CachePolicy::PREFETCH)) {
            attr->l2InfoList.emplace_back(L2Info(tensor.GetStorage()->MemorySize(), idx));
        }
        if (tensor.GetStorage()->GetCachePolicy(CachePolicy::NONE_CACHEABLE)) {
            attr->disableL2List.emplace_back(1);
        } else {
            attr->disableL2List.emplace_back(0);
        }
        idx++;
    }
    for (auto& param : attr->startArgsOutputTensorList) {
        const auto& tensor = param.get();
        auto asc_tensor = tensor.GetStorage();
        if (asc_tensor == nullptr) {
            idx++;
            attr->disableL2List.emplace_back(0);
            continue;
        }
        if (tensor.GetStorage()->GetCachePolicy(CachePolicy::NONE_CACHEABLE)) {
            attr->disableL2List.emplace_back(1);
        } else {
            attr->disableL2List.emplace_back(0);
        }
        idx++;
    }
    MACHINE_LOGI("Need prefetch tensor size is:%zu\n", attr->l2InfoList.size());
    return;
}

[[maybe_unused]] static void FindLiteNPUKernel(const std::map<uint64_t, Function*>& leafDict, std::string& kernelPath)
{
    for (auto& [hash, leaf] : leafDict) {
        (void)hash;
        auto leafAttr = leaf->GetLeafFuncAttribute();
        if (leafAttr && !leafAttr->binPath.empty()) {
            kernelPath = leafAttr->binPath;
            return;
        }
    }
}

static void SetDyndevProgBinary(Function* function, bool disableCtrlFlowCache,
                                const std::unordered_map<int, dynamic::SlotMaskEntry>* stitchUpdateSlotMaskMap)
{
    if (function == nullptr || function->GetDyndevAttribute() == nullptr) {
        return;
    }
    std::shared_ptr<DyndevFunctionAttribute> dynAttrPtr = function->GetDyndevAttribute();
    uint64_t size = 0;
    dynamic::EncodeDevAscendProgram(function, size, nullptr, stitchUpdateSlotMaskMap);
    dynAttrPtr->devProgBinary.resize(size);

    dynamic::DevAscendProgram* devProg = reinterpret_cast<dynamic::DevAscendProgram*>(&dynAttrPtr->devProgBinary[0]);
    dynamic::EncodeDevAscendProgram(function, size, devProg, stitchUpdateSlotMaskMap);
    devProg->disableCtrlFlowCache = disableCtrlFlowCache ? 1 : 0;

    if (config::GetPassDefaultConfig(npu::tile_fwk::KEY_PRINT_PROGRAM, false)) {
        devProg->DumpFile(config::LogTopFolder() + "/program.tifwkbintxt");
        std::string loopDirPath = config::LogTopFolder() + "/loop";
        CreateDir(loopDirPath, true);
        for (size_t index = 0; index < dynAttrPtr->funcGroup.loopList.size(); index++) {
            Function* func = dynAttrPtr->funcGroup.loopList[index];
            func->DumpFile(loopDirPath + "/" + func->GetMagicName() + ".tifwkgr");
        }
    }
    devProg->RelocProgram(reinterpret_cast<int64_t>(devProg), 0);
    if (config::GetPassDefaultConfig(npu::tile_fwk::KEY_PRINT_PROGRAM, false)) {
        SaveFile(config::LogTopFolder() + "/program.tifwkbin", dynAttrPtr->devProgBinary);
    }
    MACHINE_LOGI("Dev prog binary size is:%zu\n", dynAttrPtr->devProgBinary.size());
}

std::vector<SymbolicExpressionTable*> GetAllExpressionTable(
    DyndevFunctionAttribute::ExpressionTableDictGroup& exprTableGroup)
{
    std::vector<SymbolicExpressionTable*> exprTableList;
    for (auto& [func, exprTable] : exprTableGroup.loopBesDict) {
        (void)func;
        exprTableList.push_back(&exprTable);
    }
    for (auto& [func, ifDict] : exprTableGroup.loopPathCondDict) {
        (void)func;
        for (auto& [expr, exprTable] : ifDict) {
            (void)expr;
            exprTableList.push_back(&exprTable);
        }
    }
    for (auto& [func, exprTable] : exprTableGroup.devRootCoaDict) {
        (void)func;
        exprTableList.push_back(&exprTable);
    }
    for (auto& [func, opDict] : exprTableGroup.devLeafOpDict) {
        (void)func;
        for (auto& [op, exprTable] : opDict) {
            (void)op;
            exprTableList.push_back(&exprTable);
        }
    }
    return exprTableList;
}

static void ConstructCodeInfo(struct EncodeDevAscendFunctionParam& encodeDevAscendFunctionParam,
                              std::map<uint64_t, Function*>& leafDict, std::shared_ptr<DyndevFunctionAttribute> attr)
{
    attr->cceCodeInfo.resize(leafDict.size() + 2);
    /* cceIdx 0 for dummy callop */
    attr->cceCodeInfo[0].coreType = static_cast<uint32_t>(CoreType::HUB);
    attr->cceCodeInfo[0].psgId = 0;
    attr->cceCodeInfo[0].funcHash = 0;
    encodeDevAscendFunctionParam.calleeHashIndexDict[0] = 0;

    /* cceIdx 1 for HUB_MIX dummy callop */
    attr->cceCodeInfo[1].coreType = static_cast<uint32_t>(CoreType::HUB_MIX);
    attr->cceCodeInfo[1].psgId = 0;
    attr->cceCodeInfo[1].funcHash = HUB_MIX_DUMMY_HASH;
    encodeDevAscendFunctionParam.calleeHashIndexDict[HUB_MIX_DUMMY_HASH] = 1;

    int leafIndex = 2;
    for (auto& [hash, leaf] : leafDict) {
        auto leafFuncAttr = leaf->GetLeafFuncAttribute();
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, leafFuncAttr != nullptr) << "leafFuncAttr is null\n";
        encodeDevAscendFunctionParam.calleeHashIndexDict[hash] = leafIndex;
        attr->devLeafIndex2Hash[leafIndex] = hash;
        MACHINE_LOGI("Dyndev.codegen: [ %d ] hash= %lu binpath= %s", leafIndex, hash, leafFuncAttr->binPath.c_str());
        attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(leafFuncAttr->coreType);
        if (leaf->IsDummyFunction())
            attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(CoreType::HUB);
        attr->cceCodeInfo[leafIndex].psgId = leaf->GetProgramId();
        attr->cceCodeInfo[leafIndex].funcHash = hash;
        attr->cceCodeInfo[leafIndex].aicpuLeafCode = leafFuncAttr->aicpuLeafCode;
        attr->cceCodeInfo[leafIndex].wrapVecId = static_cast<int32_t>(leafFuncAttr->aivCore);
        attr->cceCodeInfo[leafIndex].mixResourceType = static_cast<uint32_t>(leafFuncAttr->mixResourceType);
        leafIndex++;
    }

    encodeDevAscendFunctionParam.cceCodeInfoList = attr->cceCodeInfo;
    return;
}

static void EncodeOutcastProperty(EncodeDevAscendFunctionParam& encodeDevAscendFunctionParam,
                                  const IncastOutcastLink* inoutLink, const IncastOutcastSlot* slot)
{
    encodeDevAscendFunctionParam.outcastDescList.clear();
    encodeDevAscendFunctionParam.assembleSlotList.clear();
    Function* devRoot = encodeDevAscendFunctionParam.devRoot;

    std::unordered_map<std::shared_ptr<RawTensor>, int> incastDict;
    for (size_t incastIndex = 0; incastIndex < devRoot->GetIncast().size(); incastIndex++) {
        incastDict[devRoot->GetIncast()[incastIndex]->GetRawTensor()] = incastIndex;
    }

    std::vector<RuntimeSlotKindSet> outcastSlotKindSetList(slot->outcastSlot.size());
    for (size_t outcastIndex = 0; outcastIndex < slot->outcastSlot.size(); outcastIndex++) {
        for (auto& slotIndex : slot->outcastSlot[outcastIndex]) {
            outcastSlotKindSetList[outcastIndex] = outcastSlotKindSetList[outcastIndex] |
                                                   inoutLink->runtimeSlotKindSetList[slotIndex];
        }
    }
    encodeDevAscendFunctionParam.outcastDescList.resize(slot->outcastSlot.size());
    for (size_t outcastIndex = 0; outcastIndex < slot->outcastSlot.size(); outcastIndex++) {
        RuntimeSlotDesc& desc = encodeDevAscendFunctionParam.outcastDescList[outcastIndex];
        if (outcastSlotKindSetList[outcastIndex].Contains(RuntimeSlotKind::INPUT)) {
            desc.kind = RuntimeSlotKind::INPUT;
        } else if (outcastSlotKindSetList[outcastIndex].Contains(RuntimeSlotKind::OUTPUT)) {
            desc.kind = RuntimeSlotKind::OUTPUT;
        } else if (outcastSlotKindSetList[outcastIndex].Contains(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
            desc.kind = RuntimeSlotKind::ASSEMBLE_OUTCAST;
        } else {
            int incastIndex = -1;
            auto outcastRawTensor = devRoot->GetOutcast()[outcastIndex]->GetRawTensor();
            if (devRoot->outIncastLinkMap.count(outcastRawTensor)) {
                auto incastRawTensor = devRoot->outIncastLinkMap[outcastRawTensor];
                incastIndex = incastDict[incastRawTensor];
            }
            if (incastIndex != -1) {
                desc.kind = RuntimeSlotKind::INPLACE_INCAST;
                desc.inplaceIncastIndex = incastIndex;
            } else {
                desc.kind = RuntimeSlotKind::EXCLUSIVE_OUTCAST;
            }
        }
    }

    for (size_t outcastIndex = 0; outcastIndex < slot->outcastSlot.size(); outcastIndex++) {
        if (encodeDevAscendFunctionParam.outcastDescList[outcastIndex].kind == RuntimeSlotKind::ASSEMBLE_OUTCAST) {
            for (auto& slotIndex : slot->outcastSlot[outcastIndex]) {
                encodeDevAscendFunctionParam.assembleSlotList.push_back(slotIndex);
            }
        }
    }
}

static bool IsNeedDumpAicpuKernel(const std::string& inputFile)
{
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_FORCE_OVERWRITE, true)) {
        // force dump, default is true
        return true;
    }
    // not force dumprootTileDict
    if (npu::tile_fwk::IsPathExist(inputFile)) {
        return false;
    }
    return true;
}
static void OverCallOpMaxNum(Function* devRoot, DevAscendFunction* funcBin)
{
    uint32_t CallOpSize = funcBin->GetOperationSize();
    uint32_t CallOpmaxSize = MAX_STITCH_LEAFFUNC_NUM;
    auto funcMagicName = devRoot->GetRawName() + "_" + std::to_string(devRoot->GetFuncMagic());
    MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED,
                 "the loop function operation: %s size is %u hitting the maxinum single-loop-operation limit:%u.\n",
                 funcMagicName.c_str(), CallOpSize, CallOpmaxSize);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, CallOpSize <= CallOpmaxSize)
        << " loopFunction: " << funcMagicName << " CallOpSize: " << CallOpSize << " CallOpmaxSize: " << CallOpmaxSize;
}

static void CompileControlFlow(const std::string& aicpuDirPath, const std::string& funcName,
                               const std::string& constrolFlow, std::string express)
{
    if (std::getenv("ENABLE_CTRLFLOW_COMPILE") == nullptr) {
        return;
    }
    std::string controlFlowCompilepath = aicpuDirPath + "/" + funcName + "/aicpu";
    MACHINE_LOGD("Dumpath is %s, functionName %s, path is %s", aicpuDirPath.c_str(), funcName.c_str(),
                 controlFlowCompilepath.c_str());
    if (!CreateDir(controlFlowCompilepath, true)) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "Creat AicpuCompile dir not success\n");
        return;
    }
    std::string controlFlowFileName = controlFlowCompilepath + "/controlFlow_dev" + funcName + ".h";
    std::string expressFileName = controlFlowCompilepath + "/expression_0.h";
    if (!SaveFile(controlFlowFileName, constrolFlow) || !SaveFile(expressFileName, express)) {
        MACHINE_LOGD("Save controlFlow and express files failed\n");
        return;
    }
#ifdef BUILD_WITH_CANN
    if (std::getenv("ASCEND_HOME_PATH") != nullptr) {
        ASSERT(HostBackEndErr::GEN_DYNAMIC_OP_FAILED, TileFwkAiCpuCompile(funcName, aicpuDirPath))
            << ": PyPto Control Flow compile failed";
    }
#endif
}

int GetRootFuncNum(std::shared_ptr<DyndevFunctionAttribute> attr)
{
    bool enableVF = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
                    config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
    int rootFuncNum = static_cast<int>(attr->funcGroup.devRootList.size());
    if (enableVF) {
        rootFuncNum *= 2; // codegen with main block and tail block
    }
    return rootFuncNum;
}

static void CollectExpressions(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, Function* function,
                               bool useNewIr)
{
    if (useNewIr) {
        MACHINE_LOGI("BuildControlFlow: using new SCF IR traversal");
        FindAllExpressionFromIR(ctx, cache, linker, function);
    } else {
        FindAllExpression(cache, linker, function);
    }
}

static void PrepareDyndevAttrForControlFlow(Function* function, const std::shared_ptr<DyndevFunctionAttribute>& attr)
{
    FillL2PrefetchInfo(attr);
    attr->commGroupNames = npu::tile_fwk::Distributed::CommGroupRecorder::GetInstance().Output();
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    attr->inoutLink = slotManager->BuildIncastOutcastLink(function->GetRawName());
    int idx = 0;
    for (auto name : slotManager->GetInputNameList()) {
        attr->inputSymbolDict[AddArgPrefix(name)] = idx++;
    }
    for (auto name : slotManager->GetOutputNameList()) {
        attr->inputSymbolDict[AddArgPrefix(name)] = idx++;
    }
}

static void NormalizeExpressionTables(Linker& linker, std::ostringstream& expressionOss)
{
    expressionOss << "#ifndef TILE_FWK_EXPRESSION_H"
                  << "\n"
                  << "#define TILE_FWK_EXPRESSION_H"
                  << "\n";
    auto& exprTableGroup = linker.GetExpressionTableDictGroup();
    std::vector<SymbolicExpressionTable*> exprTableList = GetAllExpressionTable(exprTableGroup);
    linker.GetSymbolTable()->NormalizeForSymbol();
    for (auto exprTable : exprTableList) {
        exprTable->NormalizeForSymbolTable(*linker.GetSymbolTable());
        expressionOss << exprTable->BuildExpressionList();
    }
}

static void EmitControlFlowSources(FunctionCache& cache, Linker& linker, Function* function,
                                   const std::shared_ptr<DyndevFunctionAttribute>& attr, const std::string& expName,
                                   std::vector<std::string>& exprSrcFiles, ValDependTensorMeta& valDependTensorMeta,
                                   std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                                   std::unordered_map<int, int>& slotIdxMapping)
{
    std::ostringstream exprHeaderOss;
    attr->rootTileDict.clear();
    CollectRootTileDict(cache, function, attr->rootTileDict);
    BuildConstructAssembleNeedAllocRuntimeSlots(cache, function, attr.get(), slotIdxMapping);

    auto& exprTableGroup = linker.GetExpressionTableDictGroup();
    std::vector<SymbolicExpressionTable*> exprTableList = GetAllExpressionTable(exprTableGroup);
    GetInputCse getInputCse = ExprBatchGenerator::CollectGetInputCse(exprTableList);
    const std::string sectionName = ".pypto";
    ControlFlowEmitCtx cfCtx{
        cache,          linker,        sectionName,   slotIdxMapping, attr->funcGroup, attr->rootTileDict,
        controlFlowOss, expressionOss, exprHeaderOss, expName,        exprSrcFiles,    valDependTensorMeta,
        &getInputCse};
    BuildControlFlow(cfCtx, function, 0);
}

static void FinalizeControlFlowSlotMapping(const std::shared_ptr<DyndevFunctionAttribute>& attr,
                                           std::unordered_map<int, int>& slotIdxMapping)
{
    SimplifySlots(attr.get(), slotIdxMapping);
    for (auto slot : slotIdxMapping) {
        MACHINE_LOGD("slotIdx: %d, runtime slotIdx: %d", slot.first, slot.second);
    }
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    topo_dump::DumpSlotMapping(*slotManager, slotIdxMapping, attr->inoutLink);
    BuildSlotRootIncastOutcastDict(attr.get());
    BuildRootFuncKeyDict(attr.get());
}

static void RunBuildControlFlowStage(IrBackendContext& irBackendCtx, FunctionCache& cache, Linker& linker,
                                     Function* function, const std::shared_ptr<DyndevFunctionAttribute>& attr,
                                     const std::string& expName, std::vector<std::string>& exprSrcFiles,
                                     ValDependTensorMeta& valDependTensorMeta, std::string& controlFlowSource,
                                     std::string& expressionSource)
{
    // Host-machine compile sub-step index for monitor progress (1-based when enabled, -1 when disabled).
    const int hmStep = MonitorManager::Instance().AllocHostMachineStepIndex();
    MonitorStageScope buildControlFlowScope(STAGE_HOST_MACHINE, hmStep, STAGE_DYNDEV_BUILD_CONTROL_FLOW, 0);
    bool useNewIr = function->body_ != nullptr;
    CollectExpressions(irBackendCtx, cache, linker, function, useNewIr);
    PrepareDyndevAttrForControlFlow(function, attr);

    std::ostringstream controlFlowOss;
    std::ostringstream expressionOss;
    NormalizeExpressionTables(linker, expressionOss);

    std::unordered_map<int, int> slotIdxMapping;
    if (useNewIr) {
        std::ostringstream exprHeaderOss;
        attr->rootTileDict.clear();
        CollectRootTileDict(cache, function, attr->rootTileDict);
        BuildConstructAssembleNeedAllocRuntimeSlots(cache, function, attr.get(), slotIdxMapping);
        auto& exprTableGroup = linker.GetExpressionTableDictGroup();
        std::vector<SymbolicExpressionTable*> exprTableList = GetAllExpressionTable(exprTableGroup);
        GetInputCse getInputCse = ExprBatchGenerator::CollectGetInputCse(exprTableList);
        irBackendCtx.getInputCse = &getInputCse;
        BuildControlFlowFromIR(irBackendCtx, cache, linker, ".pypto", function, slotIdxMapping, attr->funcGroup,
                               attr->rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, 0, expName,
                               exprSrcFiles, valDependTensorMeta);
    } else {
        EmitControlFlowSources(cache, linker, function, attr, expName, exprSrcFiles, valDependTensorMeta,
                               controlFlowOss, expressionOss, slotIdxMapping);
    }
    expressionOss << "#endif/*TILE_FWK_EXPRESSION_H*/"
                  << "\n";

    controlFlowSource = controlFlowOss.str();
    expressionSource = expressionOss.str();
    FinalizeControlFlowSlotMapping(attr, slotIdxMapping);
}

static void RunCompileControlFlowStage(Function* function, const std::shared_ptr<DyndevFunctionAttribute>& attr,
                                       const std::string& aicpuDirPath, const std::string& controlFlowSource,
                                       const std::string& expressionSource, std::vector<std::string>& exprSrcFiles)
{
#ifdef __x86_64__
    std::string cflags = "-mno-sse2 -mno-sse";
#else
    std::string cflags = "-mgeneral-regs-only";
#endif
    if (IsAicoreResolveEnabled()) {
        cflags += " -DENABLE_AICORE_RESOLVE=1";
    } else {
        cflags += " -DENABLE_AICORE_RESOLVE=0";
    }
    // keep -D__DEVICE__ for the arm64 dev control flow compile, which is otherwise only added when extraCflag is empty
    std::string devCflags = "-D__DEVICE__ -DENABLE_AICORE_RESOLVE=";
    devCflags += IsAicoreResolveEnabled() ? "1" : "0";

    const std::string funcHash = function->GetFunctionHash().Data();
    const std::string arm64TargetToolPath = Arm64TargetTool("g++");
    const bool hasArm64DevCompile = IsPathExist(arm64TargetToolPath);
    const std::string controlFlowHostFilePath = aicpuDirPath + "/controlFlow_host_" + funcHash + ".cpp";

    const int hmStep = MonitorManager::Instance().AllocHostMachineStepIndex();
    const int srcCount = static_cast<int>(exprSrcFiles.size() + 1) * (hasArm64DevCompile ? 2 : 1);
    const std::string hmLabel = std::string(STAGE_DYNDEV_CONTROL_FLOW_COMPILE) + funcHash;
    MonitorStageScope aicpuHostCompileScope(STAGE_HOST_MACHINE, hmStep, hmLabel, srcCount);

    attr->hostControlFlowBinary = CompileAndLoadSection(controlFlowSource, controlFlowHostFilePath, aicpuDirPath,
                                                        exprSrcFiles, "g++", "ld", "objcopy", ".pypto",
                                                        IsNeedDumpAicpuKernel(controlFlowHostFilePath), cflags);
    AlignUpTo(attr->hostControlFlowBinary, 0x8, 0);
    std::string funcName = function->GetMagicName() + function->GetFunctionHash().Data();
    CompileControlFlow(aicpuDirPath, funcName, controlFlowSource, expressionSource);
    if (hasArm64DevCompile) {
        static const std::string BISHENG_LD_CMD = "ld.lld";
        std::string controlFlowDevFilePath = aicpuDirPath + "/controlFlow_dev_" + funcHash + ".cpp";
        MACHINE_LOGI("Compile control flow src file[%s] with arm64 target tool[%s].", controlFlowDevFilePath.c_str(),
                     arm64TargetToolPath.c_str());
        attr->devControlFlowBinary = CompileAndLoadSection(
            controlFlowSource, controlFlowDevFilePath, aicpuDirPath, exprSrcFiles, arm64TargetToolPath, BISHENG_LD_CMD,
            Arm64TargetTool("objcopy"), ".pypto", IsNeedDumpAicpuKernel(controlFlowDevFilePath), devCflags);
    } else {
        // brk #0
        MACHINE_LOGW("Arm64 target tool is not found.");
        attr->devControlFlowBinary = std::vector<uint8_t>{0xd4, 0x20, 0x00, 0x00};
    }
    AlignUpTo(attr->devControlFlowBinary, 0x8, 0);
}

#ifdef BUILD_WITH_CANN
static bool RunCompileAicoreKernelStage(Function* function, std::map<uint64_t, Function*>& leafDict,
                                        EncodeDevAscendFunctionParam& encodeDevAscendFunctionParam,
                                        const std::string& ccePath, std::string& kernelPath)
{
    const int hmStep = MonitorManager::Instance().AllocHostMachineStepIndex();
    MonitorStageScope aicoreKernelCompileScope(STAGE_HOST_MACHINE, hmStep, STAGE_DYNDEV_AICORE_KERNEL_COMPILE,
                                               static_cast<int>(leafDict.size()));
    if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        FindLiteNPUKernel(leafDict, kernelPath);
        return true;
    }
    int ret = CompileAICoreKernel(leafDict, encodeDevAscendFunctionParam, ccePath, function->GetFunctionHash().Data(),
                                  function->GetOriginalRawName(), kernelPath);
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::COMPILE_AICORE_FAILED, "Compile dynamic aicore.o failed.");
        return false;
    }
    return true;
}
#endif

static void RunEncodeStage(Function* function, const std::shared_ptr<DyndevFunctionAttribute>& attr, Linker& linker,
                           EncodeDevAscendFunctionParam& encodeDevAscendFunctionParam, bool disableCtrlFlowCache)
{
    const int hmStep = MonitorManager::Instance().AllocHostMachineStepIndex();
    MonitorStageScope encodeScope(STAGE_HOST_MACHINE, hmStep, STAGE_DYNDEV_ENCODE,
                                  static_cast<int>(attr->funcGroup.devRootList.size()));
    attr->devEncodeList.resize(attr->funcGroup.devRootList.size());
    topo_dump::StaticTopoCsvWriter staticTopo;
    for (auto& devRoot : attr->funcGroup.devRootList) {
        int devRootKey = attr->funcGroup.devRootList.GetIndex(devRoot);
        MACHINE_LOGI("Dyndev.encode: %s", devRoot->GetRawName().c_str());
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, attr->rootTileDict.count(devRoot))
            << "devRoot not found in rootTileDict";
        Function* devTile = attr->rootTileDict[devRoot];
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, attr->inoutLink.ioslotDict.count(devTile))
            << "devTile not found in rootTileDict";
        IncastOutcastSlot* slot = &attr->inoutLink.ioslotDict[devTile];
        encodeDevAscendFunctionParam.symbolTable = linker.GetSymbolTable();
        if (linker.GetExpressionTableDictGroup().devRootCoaDict.count(devRoot) != 0) {
            encodeDevAscendFunctionParam
                .expressionTable = &linker.GetExpressionTableDictGroup().devRootCoaDict.find(devRoot)->second;
        }
        encodeDevAscendFunctionParam.devRoot = devRoot;
        encodeDevAscendFunctionParam.slot = slot;
        EncodeOutcastProperty(encodeDevAscendFunctionParam, &attr->inoutLink, slot);
        uint64_t size = 0;
        EncodeDevAscendFunction(function, encodeDevAscendFunctionParam, size, nullptr);
        attr->devEncodeList[devRootKey].resize(size);
        DevAscendFunction* funcBin = reinterpret_cast<DevAscendFunction*>(&attr->devEncodeList[devRootKey][0]);
        funcBin->rootHash = devRoot->GetFunctionHash().GetHash();
        funcBin->funcKey = devRootKey;
        funcBin->stackWorkSpaceSize = devTile->GetStackWorkespaceSize();
        funcBin->getInputDataCount = 0;
        funcBin->getTensorDataCount = 0;
        EncodeDevAscendFunction(function, encodeDevAscendFunctionParam, size, funcBin);
        auto maxCVCoreUsage = devRoot->GetMaxCVCoreUsage();
        funcBin->SetMaxCV(maxCVCoreUsage.first, maxCVCoreUsage.second);
        staticTopo.WriteFunction(devRootKey, *funcBin);
        funcBin->Reloc(-reinterpret_cast<int64_t>(funcBin), true);
        uint32_t CallOpmaxSize = MAX_STITCH_LEAFFUNC_NUM;
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, CallOpmaxSize <= STITCH_FUNCTION_MAX_SIZE)
            << " CallOpmaxSize : " << CallOpmaxSize << "exceeds the maximum allowed value of 65535.";
        if (funcBin->GetOperationSize() > CallOpmaxSize) {
            OverCallOpMaxNum(devRoot, funcBin);
        }
    }
    auto stitchUpdateSlotMaskMap = dynamic::BuildStitchUpdateSlotMaskMap(attr);
    for (auto& devRoot : attr->funcGroup.devRootList) {
        int devRootKey = attr->funcGroup.devRootList.GetIndex(devRoot);
        DevAscendFunction* funcBin = reinterpret_cast<DevAscendFunction*>(&attr->devEncodeList[devRootKey][0]);
        funcBin->RefilterStitchUpdateSlotLists(stitchUpdateSlotMaskMap);
    }
    SetDyndevProgBinary(function, disableCtrlFlowCache, &stitchUpdateSlotMaskMap);
}

static void RunCodeGenStage(const std::shared_ptr<DyndevFunctionAttribute>& attr,
                            std::map<uint64_t, Function*>& leafDict)
{
    std::mutex leafDictMutex;

    MonitorManager::Instance().SetRootFuncCount(GetRootFuncNum(attr));

    std::deque<std::function<void(void)>> tasks;
    for (auto& devRoot : attr->funcGroup.devRootList) {
        // Capture the index before dispatch so parallel tasks do not touch devRootList.
        int devRootIndex = attr->funcGroup.devRootList.GetIndex(devRoot);
        std::function<void(void)> task = [&devRoot, &attr, &leafDict, &leafDictMutex, devRootIndex]() {
            Function* devTile = attr->rootTileDict[devRoot];
            bool isDynamicAligned = devTile->paramConfigs_.dynamicAlignedOps;
            npu::tile_fwk::CodeGenCtx codeGenCtx("", config::GetEmitPath("kernel_aicore"), false, isDynamicAligned,
                                                 devRootIndex);
            npu::tile_fwk::CodeGen codeGen(codeGenCtx);
            COMPILER_LOGI("Function :[%s] starts executing codegen and binary compilation",
                          devTile->GetMagicName().c_str());
            codeGen.GenCode(*devTile);
            MainBlockCondBulider::Gencode(devTile, devRootIndex);

            std::lock_guard<std::mutex> lock(leafDictMutex);
            for (auto& [psgId, leaf] : devRoot->programs_) {
                (void)psgId;
                auto hash = leaf->GetFunctionHash().GetHash();
                if (!leafDict.count(hash)) {
                    leafDict[hash] = leaf;
                    MACHINE_LOGI("Dyndev.codegen: %s", leaf->GetRawName().c_str());
                } else {
                    MACHINE_LOGE(HostBackEndErr::DUPLICATE_LEAF_FUNC_HASH, " Duplicate func hash %lu name %s", hash,
                                 leaf->GetRawName().c_str());
                }
            }
        };
        tasks.push_back(task);
    }

    MonitorManager::Instance().PrintCurrentTotalElapsed("Stage CodeGen cce code generation completed");
    unsigned threadNum = GetCGThreadNum();
    ParallelExecuteAndWait(threadNum, tasks);
}

static void CompileDyndevFunction(Function* function, FunctionCache& cache, [[maybe_unused]] const std::string& ccePath)
{
    ASSERT(HostBackEndErr::RUN_PASS_FAILED,
           (PassManager::Instance().RunPass(Program::GetInstance(), *function, "ExecuteGraph") == SUCCESS));
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && config::IsRuntimeDebugAllEnabled()) {
        mix_info::DumpMixInfo(function);
    }
    std::shared_ptr<DyndevFunctionAttribute> attr = function->GetDyndevAttribute();
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, attr != nullptr) << "DyndevFunctionAttribute is nullptr\n";
    Linker linker(attr->symbolTable, attr->funcGroup, attr->exprTableDictGroup);
    bool hasAicoreKernelLink = false;
#ifdef BUILD_WITH_CANN
    hasAicoreKernelLink = config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_CODEGEN_INSTRUCTION &&
                          std::getenv("ASCEND_HOME_PATH") != nullptr;
#endif
    // Each *StepCount is the number of HostMachine sub-stages (always 1 here), not op/workload.
    // Must match Run*Stage order and AllocHostMachineStepIndex() call count (3 without AICore link, 4 with).
    constexpr int buildControlFlowStepCount = 1;   // DynDev:BuildControlFlow
    constexpr int controlFlowCompileStepCount = 1; // DynDev:ControlFlowCompile
    constexpr int aicoreKernelStepCount = 1;       // DynDev:AICoreKernelCompile (when hasAicoreKernelLink)
    constexpr int encodeStepCount = 1;             // DynDev:Encode (MonitorStageScope opSize = devRoot count)
    MonitorManager::Instance().BeginHostMachineCompileGroup(buildControlFlowStepCount + controlFlowCompileStepCount +
                                                            (hasAicoreKernelLink ? aicoreKernelStepCount : 0) +
                                                            encodeStepCount);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    std::string aicpuDirPath = config::GetEmitPath("kernel_aicpu");
    npu::tile_fwk::CreateDir(aicpuDirPath, true);
    const std::string expName = "expression_" + std::to_string(tilingKey) + ".h";
    std::vector<std::string> exprSrcFiles;
    ValDependTensorMeta valDependTensorMeta;
    std::string controlFlowSource;
    std::string expressionSource;
    IrBackendContext irBackendCtx;

    RunBuildControlFlowStage(irBackendCtx, cache, linker, function, attr, expName, exprSrcFiles, valDependTensorMeta,
                             controlFlowSource, expressionSource);

    std::string expressionFilePath = aicpuDirPath + "/" + expName;
    if (IsNeedDumpAicpuKernel(expressionFilePath)) {
        SaveFile(expressionFilePath, expressionSource);
    }

    RunCompileControlFlowStage(function, attr, aicpuDirPath, controlFlowSource, expressionSource, exprSrcFiles);

    std::map<uint64_t, Function*> leafDict;
    RunCodeGenStage(attr, leafDict);

    struct EncodeDevAscendFunctionParam encodeDevAscendFunctionParam = {};
    ConstructCodeInfo(encodeDevAscendFunctionParam, leafDict, attr);
    encodeDevAscendFunctionParam.inoutLink = &attr->inoutLink;

    std::string kernelPath;
#ifdef BUILD_WITH_CANN
    if (hasAicoreKernelLink) {
        if (!RunCompileAicoreKernelStage(function, leafDict, encodeDevAscendFunctionParam, ccePath, kernelPath)) {
            return;
        }
    }
#endif

    attr->kernelBinary = ReadFile(kernelPath);
    MACHINE_LOGD("KernelBinary size[%zu].", attr->kernelBinary.size());

    RunEncodeStage(function, attr, linker, encodeDevAscendFunctionParam, valDependTensorMeta.disableCtrlFlowCache);
}

MachineTask* GenCode(MachineTask* task, FunctionCache& cache)
{
    npu::tile_fwk::CodeGenCtx codeGenCtx("", config::GetEmitPath("kernel_aicore"));
    npu::tile_fwk::CreateDir(codeGenCtx.cceDir, true);
    npu::tile_fwk::CodeGen codeGen(codeGenCtx);
    auto function = task->GetFunction();
    /* each leafFunction inside is compiled to a standalone object file.
     * the filepath of the object file is updated to the binPath_ member.
     */
    if (function->GetGraphType() == GraphType::TILE_GRAPH) {
        MonitorManager::Instance().SetRootFuncCount(1);
        MonitorStageScope codeGenScope("CodeGen");
        MonitorManager::Instance().SwitchStageReset();
        MonitorManager::Instance().PrintCurrentTotalElapsed("Stage CodeGen start for TILE_GRAPH");
        COMPILER_LOGI("Start (TILE_GRAPH) CodeGen stage...");
        codeGen.GenCode(*function);
        MainBlockCondBulider::Gencode(function);
    } else {
        if (function->IsFunctionType(FunctionType::DYNAMIC)) {
            MonitorStageScope codeGenScope("CodeGen");
            MonitorManager::Instance().SwitchStageReset();
            MonitorManager::Instance().PrintCurrentTotalElapsed("Stage CodeGen start for DYNAMIC");
            COMPILER_LOGI("Start (DYNAMIC) CodeGen stage...");
            std::string cce_path = RealPath(codeGenCtx.cceDir) + "/";
            CompileDyndevFunction(function, cache, cce_path);
        } else {
            COMPILER_LOGI("The current function does not need to do codegen");
        }
    }

    return task;
}
} // namespace npu::tile_fwk
