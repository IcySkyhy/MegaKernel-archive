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
 * \file loop_unroll.cpp
 * \brief
 */

#include "passes/tensor_graph_pass/loop_unroll.h"
#include "interface/machine/host/host_machine.h"
#include "passes/pass_log/pass_log.h"
#include "interface/configs/config_manager_ng.h"

#define MODULE_NAME "LoopUnroll"

namespace npu {
namespace tile_fwk {
Status LoopUnroll::GetCallee(const Operation* callop, Function*& callFunc)
{
    auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
    callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
    if (callFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Get callee function %s failed.",
                          callopAttr->GetCalleeMagicName().c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status LoopUnroll::MapLocalTensorToGlobal(const LogicalTensors& localTensor, LogicalTensors& globalTensor,
                                          std::unordered_map<int, LogicalTensorPtr> tensorLocal2Global)
{
    for (auto tensor : localTensor) {
        auto tensorMagic = tensor->GetMagic();
        if (tensorLocal2Global.find(tensorMagic) != tensorLocal2Global.end()) {
            globalTensor.push_back(tensorLocal2Global[tensorMagic]);
        } else {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor with local magic %d is not found in tensorLocal2Global map.",
                              tensorMagic);
            return FAILED;
        }
    }
    return SUCCESS;
}

void LoopUnroll::DeriveTensorStaticAttributes(LogicalTensorPtr tensor, EvaluateSymbol& evaluator,
                                              std::vector<int64_t>& staticShape)
{
    // 推导动态shape
    if (!tensor->GetDynValidShape().empty()) {
        staticShape = evaluator.EvaluateValidShape(tensor->GetDynValidShape());
    }
}

std::vector<SymbolicScalar> LoopUnroll::ConvertToSymbolicScalar(std::vector<int64_t> staticShape)
{
    std::vector<SymbolicScalar> staticSymbolicValidShape;
    auto immShape = OpImmediate::Specified(staticShape);
    for (auto immDim : immShape) {
        staticSymbolicValidShape.push_back(immDim.GetSpecifiedValue());
    }
    return staticSymbolicValidShape;
}

Status LoopUnroll::AddNewOperation(Operation* localOp,
                                   const std::unordered_map<int, LogicalTensorPtr> tensorLocal2Global,
                                   std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap,
                                   std::unordered_map<Operation*, std::vector<int64_t>> opDynShapeMap)
{
    LogicalTensors globalIOperands;
    if (MapLocalTensorToGlobal(localOp->GetIOperands(), globalIOperands, tensorLocal2Global) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] input MapLocalTensorToGlobal failed.%s",
                          localOp->GetOpcodeStr().c_str(), localOp->GetOpMagic(), GetFormatBacktrace(*localOp).c_str());
        return FAILED;
    }
    LogicalTensors globalOOperands;
    if (MapLocalTensorToGlobal(localOp->GetOOperands(), globalOOperands, tensorLocal2Global) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] output MapLocalTensorToGlobal failed.%s",
                          localOp->GetOpcodeStr().c_str(), localOp->GetOpMagic(), GetFormatBacktrace(*localOp).c_str());
        return FAILED;
    }
    Operation& cloneOp = localOp->CloneOperation(*topFunction_, globalIOperands, globalOOperands);
    // 更新op属性
    if (UpdateCloneOpAttributes(localOp, &cloneOp, opDynOffsetMap, opDynShapeMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCloneOpAttributes failed.%s",
                          GetFormatBacktrace(*localOp).c_str());
        return FAILED;
    }

    // 更新输出tensor动态属性
    UpdateOutTensorDynAttributes(localOp, &cloneOp, opDynOffsetMap, opDynShapeMap);
    return SUCCESS;
}

Status LoopUnroll::UpdateCloneOpAttributes(Operation* localOp, Operation* cloneOp,
                                           std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap,
                                           std::unordered_map<Operation*, std::vector<int64_t>> opDynShapeMap)
{
    // 更新shape相关属性
    if (opDynShapeMap.find(localOp) != opDynShapeMap.end()) {
        if (opDynShapeMap[localOp].empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] cannot find dynShape.%s", localOp->GetOpcodeStr().c_str(),
                              localOp->GetOpMagic(), GetFormatBacktrace(*localOp).c_str());
            return FAILED;
        }
        auto staticSymbolicValidShape = ConvertToSymbolicScalar(opDynShapeMap[localOp]);
        // 检查是否有validShape属性
        std::vector<SymbolicScalar> validShape;
        if (cloneOp->GetAttr("validShape", validShape)) {
            cloneOp->SetAttr("validShape", staticSymbolicValidShape);
        } else if (cloneOp->GetOpcode() == Opcode::OP_VIEW) {
            auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(cloneOp->GetOpAttribute());
            if (viewAttr && !viewAttr->GetToDynValidShape().empty()) {
                viewAttr->SetToDynValidShape(staticSymbolicValidShape);
            }
        } else if (cloneOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(cloneOp->GetOpAttribute());
            if (assembleAttr && !assembleAttr->GetFromDynValidShape().empty()) {
                assembleAttr->SetFromDynValidShape(staticSymbolicValidShape);
            }
        }
    }
    // 更新offset
    if (opDynOffsetMap.find(localOp) != opDynOffsetMap.end()) {
        if (opDynOffsetMap[localOp].empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] cannot find dynOffset.%s", localOp->GetOpcodeStr().c_str(),
                              localOp->GetOpMagic(), GetFormatBacktrace(*localOp).c_str());
            return FAILED;
        }
        auto staticSymbolicOffset = ConvertToSymbolicScalar(opDynOffsetMap[localOp]);
        if (cloneOp->GetOpcode() == Opcode::OP_VIEW) {
            auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(cloneOp->GetOpAttribute());
            if (viewAttr && !viewAttr->GetFromDynOffset().empty()) {
                viewAttr->SetFromOffset(opDynOffsetMap[localOp], staticSymbolicOffset);
            }
        } else if (cloneOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(cloneOp->GetOpAttribute());
            if (assembleAttr && !assembleAttr->GetToDynOffset().empty()) {
                assembleAttr->SetToOffset(opDynOffsetMap[localOp], staticSymbolicOffset);
            }
        }
    }
    return SUCCESS;
}

void LoopUnroll::UpdateOutTensorDynAttributes(Operation* originalOp, Operation* clonedOp,
                                              std::unordered_map<Operation*, std::vector<int64_t>>& opDynOffsetMap,
                                              std::unordered_map<Operation*, std::vector<int64_t>>& opDynShapeMap)
{
    for (auto& clonedTensor : clonedOp->GetOOperands()) {
        // 检查并更新动态shape
        if (!clonedTensor->GetDynValidShape().empty()) {
            if (opDynShapeMap.find(originalOp) != opDynShapeMap.end()) {
                auto staticSymbolicValidShape = ConvertToSymbolicScalar(opDynShapeMap[originalOp]);
                clonedTensor->UpdateDynValidShape(staticSymbolicValidShape);
            } else {
                std::vector<int64_t> staticShape;
                DeriveTensorStaticAttributes(clonedTensor, *evaluateSymbol_, staticShape);
                auto staticSymbolicValidShape = ConvertToSymbolicScalar(staticShape);
                clonedTensor->UpdateDynValidShape(staticSymbolicValidShape);
            }
        }
        // 检查并更新动态offset
        if ((clonedOp->GetOpcode() == Opcode::OP_VIEW || clonedOp->GetOpcode() == Opcode::OP_ASSEMBLE) &&
            !clonedTensor->GetDynOffset().empty()) {
            if (opDynOffsetMap.find(originalOp) != opDynOffsetMap.end()) {
                // 将std::vector<int>转换为std::vector<SymbolicScalar>作为动态offset
                auto staticSymbolicOffset = ConvertToSymbolicScalar(opDynOffsetMap[originalOp]);
                // 创建TensorOffset对象并更新
                TensorOffset tensorOffset(opDynOffsetMap[originalOp], staticSymbolicOffset);
                clonedTensor->UpdateOffset(tensorOffset);
            }
        }
    }
}

void LoopUnroll::EvaluateDynamicOpParams(Operation* op, EvaluateSymbol& evaluator,
                                         std::unordered_map<Operation*, std::vector<int64_t>>& opDynOffsetMap,
                                         std::unordered_map<Operation*, std::vector<int64_t>>& opDynShapeMap)
{
    auto opCode = op->GetOpcode();
    std::vector<SymbolicScalar> dynValidShape;
    std::vector<SymbolicScalar> dynOffset;
    std::vector<int64_t> originalOffset;
    // 特殊处理VIEW和ASSEMBLE的特定属性
    if (opCode == Opcode::OP_VIEW) {
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
        if (viewAttr) {
            originalOffset = viewAttr->GetFromOffset();
            dynValidShape = viewAttr->GetToDynValidShape();
            dynOffset = viewAttr->GetFromDynOffset();
        }
    } else if (opCode == Opcode::OP_ASSEMBLE) {
        auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
        if (assembleAttr) {
            originalOffset = assembleAttr->GetToOffset();
            dynOffset = assembleAttr->GetToDynOffset();
            dynValidShape = assembleAttr->GetFromDynValidShape();
        }
    } else {
        op->GetAttr("validShape", dynValidShape);
    }

    // 计算静态值
    if (!dynValidShape.empty()) {
        opDynShapeMap.insert({op, evaluator.EvaluateValidShape(dynValidShape)});
    }

    if (!dynOffset.empty()) {
        opDynOffsetMap.insert({op, evaluator.EvaluateOffset(originalOffset, dynOffset)});
    } else if ((opCode == Opcode::OP_VIEW || opCode == Opcode::OP_ASSEMBLE) && !originalOffset.empty()) {
        // 如果没有动态 offset，直接使用原始静态 offset
        opDynOffsetMap.insert({op, originalOffset});
    }
}

Operation* LoopUnroll::ExecuteFunctionLoopLookupSat(
    const std::shared_ptr<DynloopFunctionAttribute>& controlFlowExecution)
{
    for (auto& path : controlFlowExecution->pathList) {
        bool sat = true;
        for (auto cond : path.pathCondList) {
            if (static_cast<bool>(EvaluateSymbolicScalar(cond.GetCond())) != cond.IsSat()) {
                sat = false;
                break;
            }
        }
        if (!sat) {
            continue;
        }
        return path.callop;
    }
    return nullptr;
}

Status LoopUnroll::ExpandDynamicLoop(Operation* callop)
{
    Function* currFunction = nullptr;
    if (GetCallee(callop, currFunction) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] GetCallee failed.", callop->GetOpcodeStr().c_str(),
                          callop->GetOpMagic());
        return FAILED;
    }
    auto loop = currFunction->GetDynloopAttribute();
    ScalarImmediateType begin = EvaluateSymbolicScalar(loop->Begin());
    ScalarImmediateType end = EvaluateSymbolicScalar(loop->End());
    ScalarImmediateType step = EvaluateSymbolicScalar(loop->Step());
    for (ScalarImmediateType idx = begin; idx < end; idx += step) {
        evaluateSymbol_->UpdateSymbolDict(loop->IterSymbolName(), idx);
        Operation* expandCallop = ExecuteFunctionLoopLookupSat(loop);
        if (expandCallop == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteFunctionLoopLookupSat failed.");
            return FAILED;
        }
        UpdateGlobalTensorWAW();
        if (ExpandDynamicFunction(expandCallop) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] ExpandDynamic failed.", expandCallop->GetOpcodeStr().c_str(),
                              expandCallop->GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status LoopUnroll::ExpandDynamicFunction(Operation* callop)
{
    Function* currFunction = nullptr;
    if (GetCallee(callop, currFunction) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] GetCallee failed.", callop->GetOpcodeStr().c_str(),
                          callop->GetOpMagic());
        return FAILED;
    }
    if (currFunction->GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
        // Loop嵌套loop_path需要特殊处理
        if (ExpandDynamicLoop(callop) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] ExpandDynamicLoop failed.", callop->GetOpcodeStr().c_str(),
                              callop->GetOpMagic());
            return FAILED;
        }
        return SUCCESS;
    }

    if (currFunction->GetCallopList().size() > 0) { // callop层级，需要继续展开
        for (auto& op : currFunction->GetCallopList()) {
            if (ExpandDynamicFunction(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] ExpandDynamicFunction failed.",
                                  op->GetOpcodeStr().c_str(), op->GetOpMagic());
                return FAILED;
            }
        }
    } else {                                                          // op层级
        std::unordered_map<int, LogicalTensorPtr> tensorLocal2Global; // local tensor magic到global tensor的映射
        std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap;
        std::unordered_map<Operation*, std::vector<int64_t>> opDynShapeMap;
        for (auto& op : currFunction->Operations()) {
            EvaluateDynamicOpParams(&op, *evaluateSymbol_, opDynOffsetMap, opDynShapeMap);
            if (CreateGlobalTensor(opDynOffsetMap, tensorLocal2Global, &op, currFunction) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] CreateGlobalTensor failed.%s", op.GetOpcodeStr().c_str(),
                                  op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            if (AddNewOperation(&op, tensorLocal2Global, opDynOffsetMap, opDynShapeMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] AddNewOperation failed.%s", op.GetOpcodeStr().c_str(),
                                  op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

// currentFunctionPtr_->paramConfigs_.useNodeHash = GetConfig().Get<bool>(USE_NODE_HASH);
bool LoopUnroll::IsConvertingToStatic(Function* function)
{
    auto it = std::find(staticFuncNames_.begin(), staticFuncNames_.end(), function->GetRawName());
    if (it != staticFuncNames_.end()) {
        return true;
    }
    return false;
}

Function* LoopUnroll::CreateLoopFunc(Function* func, Function* callerParentFunc)
{
    // create function
    std::string funcName = callerParentFunc->GetRawName() + "_" + "LOOP1";
    auto funcMagicName = funcName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
    auto caller = std::make_shared<Function>(Program::GetInstance(), funcMagicName, funcName, callerParentFunc);
    caller->SetFunctionType(FunctionType::DYNAMIC_LOOP);
    caller->SetGraphType(GraphType::TENSOR_GRAPH);
    Program::GetInstance().InsertFuncToFunctionMap(funcMagicName, caller);

    // set loop attr
    auto loopRange = LoopRange(1);
    auto attr = std::make_shared<DynloopFunctionAttribute>("loop1", loopRange, loopRange, false);
    caller->SetDynloopAttribute(attr);

    Program::GetInstance().CreateCallerCalleeLink(caller.get(), func);
    std::vector<Operation*> callOpList = caller->GetCallopList();
    attr->IterationEnd(0, func, callOpList[0]);
    return caller.get();
}

Status LoopUnroll::CreateLoopUnrollFunc(Function* function)
{
    std::string funcName = function->GetRawName() + "_Loop_Unroll";
    auto funcMagicName = funcName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
    auto newFunc = std::make_unique<Function>(Program::GetInstance(), funcMagicName, funcName, nullptr);
    newFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    newFunc->SetGraphType(GraphType::TENSOR_GRAPH);

    Program::GetInstance().SetCurrentFunction(newFunc.get());
    if (Program::GetInstance().GetFunctionMap().count(funcMagicName) != 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "Function[%s] has exist in functionMap.", funcMagicName.c_str());
        return FAILED;
    }
    Program::GetInstance().InsertFuncToFunctionMap(funcMagicName, std::move(newFunc));
    Program::GetInstance().GetCurrentFunction()->SetUnderDynamicFunction(true);

    auto& paramConfigs = Program::GetInstance().GetCurrentFunction()->paramConfigs_;
    std::shared_ptr<ConfigScope> currentScope = ConfigManagerNg::GetInstance().CurrentScope();
    paramConfigs.sgPgLowerBound = currentScope->GetPassConfig<int>(SG_PG_LOWER_BOUND);
    paramConfigs.sgPartitionAlgorithm = currentScope->GetPassConfig<std::string>(SG_PARTITION_ALGORITHM);
    paramConfigs.sgParallelNum = currentScope->GetPassConfig<int>(SG_PARALLEL_NUM);
    paramConfigs.sgMgCopyInUpperBound = currentScope->GetPassConfig<int>(MG_COPYIN_UPPER_BOUND);
    paramConfigs.machineConfig_ = currentScope->GetRuntimeConfig<uint8_t>(DEVICE_SCHED_MODE);
    paramConfigs.cubeL1ReuseSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(CUBE_L1_REUSE_SETTING);
    paramConfigs.cubeNBufferSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(CUBE_NBUFFER_SETTING);
    paramConfigs.vecNBufferSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(VEC_NBUFFER_SETTING);
    paramConfigs.mgVecParallelLb = currentScope->GetPassConfig<int>(MG_VEC_PARALLEL_LB);
    topFunction_ = Program::GetInstance().GetCurrentFunction();
    auto& cache = Program::GetInstance().GetFunctionCache();
    cache.Insert(topFunction_->ComputeHash(), *topFunction_);
    return SUCCESS;
}

Status LoopUnroll::TopFunctionUnroll(Function* function, std::vector<Operation*> callopList)
{
    if (CreateLoopUnrollFunc(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CreateLoopUnrollFunc failed.");
        return FAILED;
    }
    for (auto incast : function->GetIncast()) {
        if (function->GetInCastSlot(incast).size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Incast[%d] has multi slot[%zu], not support now.",
                              incast->GetMagic(), function->GetInCastSlot(incast).size());
            return FAILED;
        }
        int slotIdx = function->GetInCastSlot(incast)[0];
        auto newIncast = incast->Clone(*topFunction_, true);
        lastWriteMap_[slotIdx] = std::make_pair(newIncast, true);
    }
    for (auto callop : callopList) {
        if (ExpandDynamicFunction(callop) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] ExpandDynamic failed.", callop->GetOpcodeStr().c_str(),
                              callop->GetOpMagic());
            return FAILED;
        }
    }
    function->ClearOperationGroups();
    function->ResetOperations();
    return SUCCESS;
}

Status LoopUnroll::UpdateTopFuncInoutCast(Function* function)
{
    auto scope = Program::GetInstance().GetTensorSlotManager()->EndScope();
    std::vector<int> incastSlot = Program::GetInstance().GetTensorSlotManager()->LookupSlotIndexConst(
        function->GetDyndevAttribute()->startArgsInputTensorList);
    for (auto& incast : function->GetIncast()) {
        if (function->GetInCastSlot(incast).size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Incast[%d] has multi slot[%zu], not support now.",
                              incast->GetMagic(), function->GetInCastSlot(incast).size());
            return FAILED;
        }
        int slotIdx = function->GetInCastSlot(incast)[0];
        if (std::find(incastSlot.begin(), incastSlot.end(), slotIdx) == incastSlot.end()) {
            continue;
        }
        if (lastWriteMap_.find(slotIdx) != lastWriteMap_.end()) {
            scope->ioslot.incastSlot.push_back({slotIdx});
            topFunction_->inCasts_.push_back(lastWriteMap_[slotIdx].first);
        }
    }
    std::vector<int> outcastSlot = Program::GetInstance().GetTensorSlotManager()->LookupSlotIndexConst(
        function->GetDyndevAttribute()->startArgsOutputTensorList);
    int idx = 0;
    for (auto& outcast : function->GetOutcast()) {
        if (function->GetOutCastSlot(outcast).size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Outcast[%d] has multi slot[%zu], not support now.",
                              outcast->GetMagic(), function->GetOutCastSlot(outcast).size());
            return FAILED;
        }
        int slotIdx = function->GetOutCastSlot(outcast)[0];
        if (std::find(outcastSlot.begin(), outcastSlot.end(), slotIdx) == outcastSlot.end()) {
            continue;
        }
        if (lastWriteMap_.find(slotIdx) != lastWriteMap_.end()) {
            scope->ioslot.outcastSlot.push_back({slotIdx});
            scope->ioslot.partialUpdateOutcastList.push_back(idx++);
            topFunction_->outCasts_.push_back(lastWriteMap_[slotIdx].first);
        }
    }
    return SUCCESS;
}

Status LoopUnroll::TraverseCallOp(Function* function)
{
    std::vector<Operation*> callopList = function->GetCallopList();
    if (IsConvertingToStatic(function)) { // 当前function是做静态转换入口的topFunction
        APASS_LOG_INFO_F(Elements::Function, "Begin unroll function[%s].", function->GetRawName().c_str());
        if (TopFunctionUnroll(function, callopList) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Function[%s] TopFunctionUnroll failed.",
                              function->GetRawName().c_str());
            return FAILED;
        }

        Program::GetInstance().GetTensorSlotManager()->scopeList.clear();
        Program::GetInstance().GetTensorSlotManager()->BeginScope(topFunction_);
        if (UpdateTopFuncInoutCast(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Function[%s] UpdateTopFuncInoutCast failed.",
                              function->GetRawName().c_str());
            return FAILED;
        }

        auto loopFunction = CreateLoopFunc(topFunction_, function);
        topFunction_->SetParent(loopFunction);
        function->outCasts_.clear();
        function->inCasts_.clear();
        Program::GetInstance().CreateCallerCalleeLink(function, loopFunction);
        HostMachine::GetInstance().ClearStashFuncQueue();
        Program::GetInstance().RefillCompileQueue(topFunction_);
        Program::GetInstance().RefillCompileQueue(loopFunction);
        Program::GetInstance().RefillCompileQueue(function);
    } else {
        for (auto callop : callopList) {
            Function* childFunction = nullptr;
            if (GetCallee(callop, childFunction) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] GetCallee failed.", callop->GetOpcodeStr().c_str(),
                                  callop->GetOpMagic());
                return FAILED;
            }
            if (TraverseCallOp(childFunction) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Child function[%s] TraverseCallOp failed.",
                                  childFunction->GetRawName().c_str());
                return FAILED;
            }
            if (IsConvertingToStatic(childFunction)) {
                CallOpAttribute* callOpAttr = static_cast<CallOpAttribute*>(callop->GetOpAttribute().get());
                callOpAttr->SetCalleeHash(childFunction->GetFunctionHash());
            }
        }
    }
    return SUCCESS;
}

Status LoopUnroll::FindOutputGlobalTensor(int slotIdx, std::unordered_map<int, LogicalTensorPtr>& tensor2Global,
                                          std::set<LogicalTensorPtr> input2Global, LogicalTensorPtr tensor,
                                          std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap)
{
    if (lastWriteMap_.find(slotIdx) == lastWriteMap_.end() || // 初次写入slot
        IsWARDepend(slotIdx, input2Global) ||                 // 与上次写入slot的tensor存在WAR关系
        (!IsNoOverlapWAW(slotIdx, tensor, opDynOffsetMap) &&
         lastWriteMap_[slotIdx].second)) { // 与上次写入slot的tensor存在重叠的WAW关系且跨LOOP
        lastWriteMap_[slotIdx] = {tensor->Clone(*topFunction_, true), false};
        tensor2Global[tensor->GetMagic()] = lastWriteMap_[slotIdx].first;
    } else if (IsNoOverlapWAW(slotIdx, tensor, opDynOffsetMap)) { // 与上次写入slot的tensor存在无重叠的WAW关系
        tensor2Global[tensor->GetMagic()] = lastWriteMap_[slotIdx].first;
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Illegal case.");
        return FAILED;
    }
    return SUCCESS;
}

Status LoopUnroll::FindInputGlobalTensor(int slotIdx, std::unordered_map<int, LogicalTensorPtr>& tensor2Global,
                                         LogicalTensorPtr tensor)
{
    if (lastWriteMap_.find(slotIdx) != lastWriteMap_.end()) { // 与上次写入slot的tensor存在RAW关系
        if (tensor2Global.find(tensor->GetMagic()) != tensor2Global.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] has exist in tensor2Global.", tensor->GetMagic());
            return FAILED;
        }
        tensor2Global[tensor->GetMagic()] = lastWriteMap_[slotIdx].first;
        lastWriteMap_[slotIdx].second = false;
    } else {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d][slot %d] cannot find RAW global tensor.", tensor->GetMagic(),
                          slotIdx);
        return FAILED;
    }
    return SUCCESS;
}

Status LoopUnroll::CreateLocal2Global(std::unordered_map<int, LogicalTensorPtr>& tensor2Global, LogicalTensorPtr tensor)
{
    if (tensor2Global.find(tensor->GetMagic()) == tensor2Global.end()) {
        LogicalTensorPtr cloneTensor = tensor->Clone(*topFunction_, true);
        if (cloneTensor == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Clone tensor[%d] failed.", tensor->GetMagic());
            return FAILED;
        }
        tensor2Global[tensor->GetMagic()] = cloneTensor;
    }
    return SUCCESS;
}

Status LoopUnroll::CreateGlobalTensor(std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap,
                                      std::unordered_map<int, LogicalTensorPtr>& tensor2Global, const Operation* op,
                                      Function* curFunc)
{
    std::set<LogicalTensorPtr> input2Global;
    for (auto& inTensor : op->GetIOperands()) {
        std::vector<int> slots = curFunc->GetInCastSlot(inTensor);
        if (slots.size() == 1) {
            if (FindInputGlobalTensor(slots[0], tensor2Global, inTensor) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] FindInputGlobalTensor failed.", inTensor->GetMagic());
                return FAILED;
            }
        } else if (slots.size() == 0) {
            if (CreateLocal2Global(tensor2Global, inTensor) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Local tensor[%d] create to global tensor failed.",
                                  inTensor->GetMagic());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] has multi slot[%zu], not support now.",
                              inTensor->GetMagic(), slots.size());
            return FAILED;
        }
        input2Global.insert(tensor2Global[inTensor->GetMagic()]);
    }
    for (auto& outTensor : op->GetOOperands()) {
        std::vector<int> slots = curFunc->GetOutCastSlot(outTensor);
        if (slots.size() == 1) {
            if (FindOutputGlobalTensor(slots[0], tensor2Global, input2Global, outTensor, opDynOffsetMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] FindInputGlobalTensor failed.", outTensor->GetMagic());
                return FAILED;
            }
        } else if (slots.size() == 0) {
            if (CreateLocal2Global(tensor2Global, outTensor) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Local tensor[%d] create to global tensor failed.",
                                  outTensor->GetMagic());
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] has multi slot[%zu], not support now.",
                              outTensor->GetMagic(), slots.size());
            return FAILED;
        }
    }
    return SUCCESS;
}

void LoopUnroll::UpdateGlobalTensorWAW()
{
    for (auto& globalTensor : lastWriteMap_) {
        globalTensor.second.second = true;
    }
}

bool LoopUnroll::IsWARDepend(const int slotIdx, std::set<LogicalTensorPtr> input2Global)
{
    auto globalTensor = lastWriteMap_.at(slotIdx);
    if (globalTensor.first->GetConsumers().empty()) {
        return false;
    }
    bool isDepend = false;
    for (auto& consumer : globalTensor.first->GetConsumers()) {
        FindSlotDepend(consumer, input2Global, isDepend);
        if (isDepend) {
            return true;
        }
    }
    return false;
}

void LoopUnroll::FindSlotDepend(const Operation* op, std::set<LogicalTensorPtr> input2Global, bool& isDepend)
{
    for (auto& outTensor : op->GetOOperands()) {
        if (input2Global.find(outTensor) != input2Global.end()) {
            isDepend = true;
            return;
        }

        for (auto& consumer : outTensor->GetConsumers()) {
            FindSlotDepend(consumer, input2Global, isDepend);
        }
    }
}

bool LoopUnroll::IsOverlapping(std::pair<std::vector<int64_t>, std::vector<int64_t>> tensor1,
                               std::pair<std::vector<int64_t>, std::vector<int64_t>> tensor2)
{
    for (size_t i = 0; i < tensor1.first.size(); ++i) {
        int64_t aStart = tensor1.second[i];
        int64_t aEnd = aStart + tensor1.first[i];
        int64_t bStart = tensor2.second[i];
        int64_t bEnd = bStart + tensor2.first[i];

        // 如果任意一维不重叠，则整体不重叠
        if (aEnd <= bStart || aStart >= bEnd) {
            return false;
        }
    }
    return true;
}

bool LoopUnroll::IsTensorOverlap(std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& tensors)
{
    if (tensors.empty()) {
        return false;
    }

    // 检查是否有重叠
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = i + 1; j < tensors.size(); ++j) {
            if (IsOverlapping(tensors[i], tensors[j])) {
                return false;
            }
        }
    }
    return true;
}

bool LoopUnroll::IsNoOverlapWAW(int slotIdx, LogicalTensorPtr tensor,
                                std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap)
{
    auto globalTensor = lastWriteMap_.at(slotIdx);
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> assembleList;
    for (auto& producer : globalTensor.first->GetProducers()) {
        if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            return false;
        }
        auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (!assembleAttr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot get %s[%d] assemble attr.", producer->GetOpcodeStr().c_str(),
                              producer->GetOpMagic());
            return false;
        }
        assembleList.push_back({producer->GetInputOperand(0)->GetShape(), assembleAttr->GetToOffset()});
    }

    for (auto& producer : tensor->GetProducers()) {
        if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            return false;
        }
        if (opDynOffsetMap.find(producer) != opDynOffsetMap.end()) {
            assembleList.push_back({producer->GetInputOperand(0)->GetShape(), opDynOffsetMap[producer]});
        } else {
            auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
            if (!assembleAttr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Cannot get %s[%d] assemble attr.",
                                  producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
                return false;
            }
            assembleList.push_back({producer->GetInputOperand(0)->GetShape(), assembleAttr->GetToOffset()});
        }
    }

    return IsTensorOverlap(assembleList);
}

Status LoopUnroll::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "==============> Start LoopUnroll.");
    staticFuncNames_ = GetConfig<std::vector<std::string>>("CONVERT_TO_STATIC", {});
    if (staticFuncNames_.size() == 0) {
        APASS_LOG_INFO_F(Elements::Function, "Found no names to convert to static function.");
        return SUCCESS;
    }
    evaluateSymbol_ = std::make_shared<EvaluateSymbol>();
    if (TraverseCallOp(&function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Function[%s] TraverseCallOp failed.", function.GetRawName().c_str());
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "==============> End LoopUnroll.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
