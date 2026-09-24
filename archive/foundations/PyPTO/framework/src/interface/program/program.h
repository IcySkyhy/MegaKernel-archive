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
 * \file program.h
 * \brief
 */

#pragma once

#include "interface/function/function.h"
#include "interface/cache/function_cache.h"

#ifndef ENABLE_HIDDENLOOP
#define ENABLE_HIDDENLOOP 1
#endif

namespace npu::tile_fwk {
class ConfigScope;

class Program {
public: // public api for torch
    std::vector<Function*> functionSequence_;
    Program();
    ~Program();

    static Program& GetInstance();
    void Reset();
    void AbandonIncompleteRecording();
    bool HasIncompleteRecordingState() const;
    bool BeginFunction(const std::string& funcName, const FunctionType funcType = FunctionType::STATIC,
                       const GraphType graphType = GraphType::TENSOR_GRAPH,
                       const std::vector<std::reference_wrapper<const Tensor>>& explicitOpArgs = {},
                       bool isHiddenFunction = false);
    std::tuple<Function*, Operation*, bool> EndFunction(const std::string& funcName, bool generateCall = true);

    Operation& ConnectCallerGusket(Function& caller, FunctionCallArgs& args) const;

    Operation& AddOperation(const std::string& opName, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                            const std::vector<std::shared_ptr<LogicalTensor>>& oOperand);

    Operation& AddOperation(const Opcode opCode, const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                            const std::vector<std::shared_ptr<LogicalTensor>>& oOperand);

    bool QueryAndUpdateCurrentFunction();

    std::string Name() const { return name_; }
    void SetName(std::string nStr) { name_ = nStr; }

    void InsertAliveTensor(Tensor* t) { aliveTensors_.insert(t); };
    void EraseAliveTensor(Tensor* t) { aliveTensors_.erase(t); }
    const auto& GetAliveTensors() const { return aliveTensors_; }

    const std::map<std::string, std::shared_ptr<npu::tile_fwk::Function>>& GetFunctionMap() const
    {
        return functionmap_;
    }
    size_t FunctionMapSize() const { return functionmap_.size(); }
    void InsertFuncToFunctionMap(const std::string& magicName, std::shared_ptr<Function> func)
    {
        FE_ASSERT(FeError::IS_EXIST, functionmap_.count(magicName) == 0)
            << magicName << " already exists in functionmap.";
        functionmap_.emplace(magicName, func);
    }
    std::shared_ptr<Function> GetFunctionByMagic(int funcMagic);
    Function* GetFunctionByMagicName(const std::string& magicName) const;
    Function* GetFunctionByRawName(const std::string& rawName) const;
    Function* GetCurrentFunction() { return currentFunctionPtr_; }
    void SetCurrentFunction(Function* function);

    std::shared_ptr<TensorSlotManager> GetTensorSlotManager()
    {
        if (tensorSlotManager_ == nullptr) {
            tensorSlotManager_ = std::make_shared<TensorSlotManager>();
        }
        return tensorSlotManager_;
    };
    void ResetTensorSlotManager()
    {
        if (tensorSlotManager_) {
            tensorSlotManager_->ClearSlotTensors();
        }
        tensorSlotManager_ = std::make_shared<TensorSlotManager>();
    }
    Json DumpJson(Function* mainFunc = nullptr) const;
    void LoadJson(Json& programJson);
    void DumpJsonFile(const std::string& fileName, Function* mainFunc = nullptr);
    std::string Dump() const; // Serialize Program briefly
    std::string DumpStack(const std::string& funcName = "") const;
    void PopStackAndUpdateCurrent();

    std::vector<std::reference_wrapper<RecordLoopFunc>> loopStack_;
    std::vector<std::reference_wrapper<RecordLoopFunc>>& GetLoopStack() { return loopStack_; }

    // Return current containing dynamic function.
    Function* GetCurrentDynamicFunction() const { return currentDynamicFunctionPtr_; }
    void SetCurrentDynamicFunction(Function* dynFunc)
    {
        FE_ASSERT(FeError::IS_EXIST, currentDynamicFunctionPtr_ != dynFunc)
            << "Under: " << currentDynamicFunctionPtr_->GetRawName() << " " << dynFunc->GetRawName();
        currentDynamicFunctionPtr_ = dynFunc;
    }

    void VerifyTensorGraph();
    void VerifyPass(Function* func, int passIndex, const std::string& passIdentifier);
    void VerifyExecuteGraph();

    void SetLastFunction(Function* func) { lastFunc_ = func; }
    Function* GetLastFunction() const { return lastFunc_; }

    std::optional<CacheValue> TryHitCahce(const FunctionHash& functionHash) { return functionCache_.Get(functionHash); }
    FunctionCache& GetFunctionCache() { return functionCache_; }
    std::shared_ptr<npu::tile_fwk::Function> GetFunctionSharedPtr(Function* rawPtr);

    // 动静归一
    void CreateCallerCalleeLink(Function* caller, Function* callee);
    void RefillCompileQueue(Function* func);
    void UpdateCompileTask();
    void ClearEmptyHiddenFunction();
    void SetParamConfig(Function* func, const std::shared_ptr<ConfigScope>& configScope) const;

private:
    std::string name_;
    std::vector<std::string> functionMagicNameStack_;
    std::string currentFunctionMagicName_;
    Function* currentFunctionPtr_;
    Function* lastFunc_{nullptr};
    Function* currentDynamicFunctionPtr_{nullptr};
    std::shared_ptr<TensorSlotManager> tensorSlotManager_;
    FunctionCache functionCache_;
    std::unordered_set<Tensor*> aliveTensors_;
    std::map<std::string, std::shared_ptr<npu::tile_fwk::Function>> functionmap_;

    void CreateInitFunction();
    Operation* FinishCurrentFunction(const std::shared_ptr<TensorSlotScope>& scope, bool generateCall);

    // Helper functions to reduce cyclomatic complexity of EndFunction
    void DumpTensorGraphIfNeeded(Function* result);
    void HandleTaskSubmission(Function* result);
#if ENABLE_HIDDENLOOP
    void EndHiddenLoop(Function* func, bool generateCall);
    void BeginHiddenLoop(Function* func, const FunctionType& funcType, const std::string funcName);
#endif
};
} // namespace npu::tile_fwk
