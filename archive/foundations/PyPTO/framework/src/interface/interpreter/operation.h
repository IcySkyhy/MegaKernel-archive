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
 * \file operation.h
 * \brief
 */
/*for flow verify tool */

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "interface/operation/attribute.h"
#include "interface/configs/config_manager.h"
#include "utils/file_utils.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {

constexpr int DATATYPE_EIGHT = 8;

class OpSyncQueue;

struct FunctionFrame;
class OperationInterpreter;

/// Shared by all per-thread OperationInterpreter instances under one FunctionInterpreter (mix-split CV sync sim).
class InterpreterSyncSimulationState {
public:
    void Reset();
    void Set(const OpSyncQueue& sq, int opMagic);
    void Wait(const OpSyncQueue& sq, int opMagic);

    static constexpr int64_t INTERPRETER_SYNC_SIM_WAIT_TIMEOUT_MS = 60000;

private:
    std::mutex interpreterSyncSimMutex_;
    std::condition_variable interpreterSyncSimCv_;
    std::unordered_map<int, uint32_t> interpreterSyncSimPending_;
};

struct ExecuteOperationContext {
    FunctionFrame* frame;
    OperationInterpreter* opInter;
    Operation* op;
    const std::vector<LogicalTensorDataPtr>* ioperandDataViewList;
    std::vector<LogicalTensorDataPtr>* ooperandDataViewList;
    std::vector<LogicalTensorDataPtr>* ooperandInplaceDataViewList;

    std::string Dump() const;
};

using Funcs = std::function<void(ExecuteOperationContext*)>;

class OperationInterpreter {
public:
    /// If \p sharedSyncSim is null, owns a private sync state (standalone / unit tests).
    explicit OperationInterpreter(std::shared_ptr<InterpreterSyncSimulationState> sharedSyncSim = {})
        : evaluateSymbol(std::make_shared<EvaluateSymbol>()),
          syncSim_(sharedSyncSim ? sharedSyncSim : std::make_shared<InterpreterSyncSimulationState>())
    {}

    std::shared_ptr<EvaluateSymbol> evaluateSymbol;

    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar& ss)
    {
        return evaluateSymbol->EvaluateSymbolicScalar(ss);
    }
    std::vector<int64_t> EvaluateOffset(const std::vector<int64_t>& offset,
                                        const std::vector<SymbolicScalar>& dynOffset,
                                        const std::vector<SymbolicScalar>& linearArgList = {})
    {
        return evaluateSymbol->EvaluateOffset(offset, dynOffset, linearArgList);
    }
    std::vector<int64_t> EvaluateOpImmediate(FunctionFrame* frame, const std::vector<OpImmediate>& opImmList);

    std::vector<int64_t> EvaluateValidShape(const std::vector<SymbolicScalar>& dynValidShape,
                                            const std::vector<SymbolicScalar>& linearArgList = {})
    {
        return evaluateSymbol->EvaluateValidShape(dynValidShape, linearArgList);
    }

    void ExecuteOperation(ExecuteOperationContext* ctx);

    /// Host simulation for CV_SYNC_* only; pending map keyed by syncQueue.eventId_. OP_SYNC_* is not simulated here.
    /// Mix-split parallel shares one InterpreterSyncSimulationState across per-thread OperationInterpreter instances.
    void ResetInterpreterSyncSimulation();
    void InterpreterSyncSimSet(const OpSyncQueue& sq, int opMagic);
    void InterpreterSyncSimWait(const OpSyncQueue& sq, int opMagic);

    std::shared_ptr<InterpreterSyncSimulationState> GetSyncSimulationState() const { return syncSim_; }

    // 注册默认函数
    static void RegisterFunc(const Opcode opcode, Funcs func)
    {
        operationInterpreterFuncs_()[opcode] = std::move(func);
    }

private:
    // 调用场景对应的函数 CallOperationInterpreterFunc
    void CallOperationInterpreterFunc(ExecuteOperationContext* ctx)
    {
        const Opcode opcode = ctx->op->GetOpcode();
        auto it = operationInterpreterFuncs_().find(opcode);
        if (it != operationInterpreterFuncs_().end()) {
            it->second(ctx);
        } else {
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false)
                << "opcode [" << ctx->op->GetOpcodeStr() << "]'s torch interface implementation is not registered";
        }
    }

    std::vector<LogicalTensorDataPtr> GetValidDataView(const std::vector<LogicalTensorDataPtr>& dataViewList) const
    {
        std::vector<LogicalTensorDataPtr> result;
        for (auto& dataView : dataViewList) {
            auto& validShape = dataView->GetValidShape();
            ASSERT(ExecuteOperationScene::EMPTY_VALIDSHAPE, validShape.size() != 0);
            if (validShape == dataView->GetShape()) {
                result.emplace_back(dataView);
            } else {
                result.emplace_back(dataView->View(validShape, dataView->GetOffset()));
            }
        }
        return result;
    }

    static std::unordered_map<Opcode, Funcs>& operationInterpreterFuncs_()
    {
        static std::unordered_map<Opcode, Funcs> instance;
        return instance;
    }

    std::shared_ptr<InterpreterSyncSimulationState> syncSim_;
};

// LogTensorList 用於在執行 Operation 出錯時打印張量資訊
void LogTensorList(const char* role, Operation* op, const LogicalTensors& tensors);

#define REGISTER_CALC_OP(OpCoreStr, OpType, FuncName)                                         \
    class OpCoreStr##ClacOpRegister {                                                         \
    public:                                                                                   \
        OpCoreStr##ClacOpRegister() { OperationInterpreter::RegisterFunc(OpType, FuncName); } \
    };                                                                                        \
    static OpCoreStr##ClacOpRegister OpCoreStr##_calcop_register

#undef CASE_DATA_TYPE_DIS
#undef CASE_DATA_TYPE
} // namespace npu::tile_fwk
