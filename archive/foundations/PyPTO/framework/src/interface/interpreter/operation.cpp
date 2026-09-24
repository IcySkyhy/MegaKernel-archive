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
 * \file operation.cpp
 * \brief
 */

#include <chrono>
#include <sstream>
#include <thread>

#include "interface/interpreter/function.h"
#include "interface/interpreter/interpreter_log.h"
#include "interface/interpreter/operation.h"
#include "interface/operation/operation_common.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

static std::string DumpShapeVec(const std::vector<int64_t>& shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

static std::string DumpSymbolicVec(const std::vector<SymbolicScalar>& symbols)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << symbols[i].Dump();
    }
    ss << "]";
    return ss.str();
}

void LogTensorList(const char* role, Operation* op, const LogicalTensors& tensors)
{
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }
        auto shapeStr = DumpShapeVec(tensor->shape);
        auto offsetStr = DumpShapeVec(tensor->offset);
        auto dynValidShapeStr = DumpSymbolicVec(tensor->GetDynValidShape());
        auto dynOffsetStr = DumpSymbolicVec(tensor->GetDynOffset());
        INTERPRETER_LOGE_FULL(ExecuteOperationScene::RUNTIME_EXCEPTION,
                              "ExecuteOperation error: op %s (magic=%d) %s[%zu] tensorMagic=%d, "
                              "shape=%s, offset=%s, dynValidShape=%s, dynOffset=%s",
                              op->GetOpcodeStr().c_str(), op->GetOpMagic(), role, i, tensor->magic, shapeStr.c_str(),
                              offsetStr.c_str(), dynValidShapeStr.c_str(), dynOffsetStr.c_str());
    }
}

static int64_t GetAsParameterCoaIndex(const RawSymbolicScalarPtr& value)
{
    if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM_OFFSET")) {
        auto& operands = value->GetExpressionOperandList();
        auto base = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX]->GetImmediateValue();
        auto dimIdx = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX]->GetImmediateValue();
        return base + COA_INDEX_DIM_BASE + dimIdx;
    } else if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM_VALID_SHAPE")) {
        auto& operands = value->GetExpressionOperandList();
        auto dim = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_SIZE_INDEX]->GetImmediateValue();
        auto base = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX]->GetImmediateValue();
        auto dimIdx = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX]->GetImmediateValue();
        return base + COA_INDEX_DIM_BASE + dim * 0x3 + dimIdx;
    } else if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM_RAW_SHAPE")) {
        auto& operands = value->GetExpressionOperandList();
        auto dim = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_SIZE_INDEX]->GetImmediateValue();
        auto base = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX]->GetImmediateValue();
        auto dimIdx = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX]->GetImmediateValue();
        return base + COA_INDEX_DIM_BASE + dim * 0x2 + dimIdx;
    } else if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM")) {
        auto& operands = value->GetExpressionOperandList();
        return operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_SIZE_INDEX]->GetImmediateValue();
    }
    return -1;
}

std::vector<int64_t> OperationInterpreter::EvaluateOpImmediate(FunctionFrame* frame,
                                                               const std::vector<OpImmediate>& opImmList)
{
    std::vector<int64_t> result;
    for (auto& opImm : opImmList) {
        int64_t res = 0;
        if (opImm.IsSpecified()) {
            auto opImmValue = opImm.GetSpecifiedValue();
            auto coaIndex = GetAsParameterCoaIndex(opImmValue.Raw());
            if (coaIndex != -1) {
                auto attr = frame->callopAttr->GetLinearArgList()[coaIndex];
                res = EvaluateSymbolicScalar(attr);
            } else {
                res = EvaluateSymbolicScalar(opImm.GetSpecifiedValue());
            }
        } else {
            int index = opImm.GetParameterIndex();
            auto attr = frame->callopAttr->GetLinearArgList()[index];
            res = EvaluateSymbolicScalar(attr);
        }
        result.push_back(res);
    }
    return result;
}

void OperationInterpreter::ExecuteOperation(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    const std::string opcodeStr = ctx->op->GetOpcodeStr();
    INTERPRETER_LOGI("[ExecuteOperation] opcode=%s opMagic=%d", opcodeStr.c_str(), ctx->op->GetOpMagic());

    auto iOperands = OperationInterpreter::GetValidDataView(*ctx->ioperandDataViewList);
    auto oOperands = OperationInterpreter::GetValidDataView(*ctx->ooperandInplaceDataViewList);
    if (ctx->op->GetOpcode() == Opcode::OP_RESHAPE) {
        iOperands = *ctx->ioperandDataViewList;
        oOperands = *ctx->ooperandInplaceDataViewList;
    }
    ExecuteOperationContext ctxValid = {ctx->frame, this, ctx->op, &iOperands, {}, &oOperands};
    try {
        OperationInterpreter::CallOperationInterpreterFunc(&ctxValid);
    } catch (std::exception& e) {
        auto* op = ctx->op;
        if (op != nullptr) {
            // 打印当前 op 输入 / 输出的动态信息，便于排查执行错误
            LogTensorList("input", op, op->GetIOperands());
            LogTensorList("output", op, op->GetOOperands());
        }
        auto func = ctx->frame->func;
        func->DumpFile(config::LogTensorGraphFolder() + "/" + func->GetRawName() + ".tifwkgr");
        std::string errMsg = e.what();
        auto firstNl = errMsg.find('\n');
        if (firstNl != std::string::npos) {
            auto secondNl = errMsg.find('\n', firstNl + 1);
            if (secondNl != std::string::npos) {
                errMsg = errMsg.substr(0, secondNl);
            }
        }
        throw std::runtime_error(std::to_string(ctx->frame->rootFuncHash) + ", " +
                                 std::to_string(ctx->frame->funcHash) + ", " + std::to_string(op->GetOpMagic()) + ", " +
                                 op->GetOpcodeStr() + "OpError\n" + ctx->Dump() + errMsg);
    }
}

std::string ExecuteOperationContext::Dump() const
{
    std::stringstream ss;
    ss << "func: " << frame->func->GetRawName() << "\n";

    auto span = op->GetSpan();
    if (!span.IsUnknown()) {
        ss << "filename: " << span.Filename() << "\n";
        ss << "lineno: " << span.BeginLine() << "\n";
    }

    auto printType = [&](auto& viewList) {
        for (size_t i = 0; i < viewList.size(); i++) {
            if (i != 0)
                ss << ", ";
            ss << viewList[i]->DumpType();
        }
    };

    ss << op->Dump();
    printType(*ooperandInplaceDataViewList);
    ss << " = " << op->GetOpcodeStr() << " ";
    printType(*ioperandDataViewList);
    ss << "\n";
    return ss.str();
}

namespace {
int InterpreterSyncSimEventKey(const OpSyncQueue& q) { return q.eventId_; }

std::string FormatInterpreterCvSyncSimLogCtx(const OpSyncQueue& q)
{
    std::ostringstream oss;
    oss << "eventId=" << q.eventId_ << " (pipePair=" << q.Dump() << " setCore=" << static_cast<int>(q.coreType_)
        << " waitCore=" << static_cast<int>(q.trigCoreType_) << " setAiv=" << static_cast<int>(q.setAivCore_)
        << " waitAiv=" << static_cast<int>(q.waitAivCore_) << ")";
    return oss.str();
}

std::string CurrentInterpreterThreadTag()
{
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}
} // namespace

void InterpreterSyncSimulationState::Reset()
{
    std::lock_guard<std::mutex> lock(interpreterSyncSimMutex_);
    interpreterSyncSimPending_.clear();
    interpreterSyncSimCv_.notify_all();
    INTERPRETER_LOGI("[InterpreterCvSyncSim] RESET cleared_all_pending thread=%s",
                     CurrentInterpreterThreadTag().c_str());
}

void InterpreterSyncSimulationState::Set(const OpSyncQueue& sq, int opMagic)
{
    const int evKey = InterpreterSyncSimEventKey(sq);
    std::lock_guard<std::mutex> lock(interpreterSyncSimMutex_);
    uint32_t& cnt = interpreterSyncSimPending_[evKey];
    ++cnt;
    interpreterSyncSimCv_.notify_all();
    const std::string desc = FormatInterpreterCvSyncSimLogCtx(sq);
    INTERPRETER_LOGI(
        "[InterpreterCvSyncSim] SET opMagic=%d eventId=%d key=eventId pending_after=%u detail={%s} thread=%s", opMagic,
        evKey, static_cast<unsigned>(cnt), desc.c_str(), CurrentInterpreterThreadTag().c_str());
}

void InterpreterSyncSimulationState::Wait(const OpSyncQueue& sq, int opMagic)
{
    const int evKey = InterpreterSyncSimEventKey(sq);
    std::unique_lock<std::mutex> lock(interpreterSyncSimMutex_);
    const auto prefIt = interpreterSyncSimPending_.find(evKey);
    const bool preflightReady = prefIt != interpreterSyncSimPending_.end() && prefIt->second > 0;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(INTERPRETER_SYNC_SIM_WAIT_TIMEOUT_MS);
    const auto tWaitStart = std::chrono::steady_clock::now();
    const bool ready = interpreterSyncSimCv_.wait_until(lock, deadline, [this, evKey] {
        auto it = interpreterSyncSimPending_.find(evKey);
        return it != interpreterSyncSimPending_.end() && it->second > 0;
    });
    const auto waitUs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                              tWaitStart)
                            .count();

    ASSERT(ControlFlowScene::INTERPRETER_SYNC_SIM_WAIT_TIMEOUT, ready)
        << "Timeout waiting for interpreter CV sync set (mix parallel / ordering), timeoutMs="
        << INTERPRETER_SYNC_SIM_WAIT_TIMEOUT_MS << ", opMagic=" << opMagic << ", eventId=" << evKey
        << ", sync_queue=" << sq.Dump();
    auto it = interpreterSyncSimPending_.find(evKey);
    ASSERT(ExecuteOperationScene::RUNTIME_EXCEPTION, it != interpreterSyncSimPending_.end() && it->second > 0)
        << "Interpreter CV sync wait race after wake, opMagic=" << opMagic << ", eventId=" << evKey
        << ", sync_queue=" << sq.Dump();
    const uint32_t pendingBefore = it->second;
    it->second--;
    const uint32_t pendingAfter = it->second;
    if (it->second == 0) {
        interpreterSyncSimPending_.erase(it);
    }
    const std::string desc = FormatInterpreterCvSyncSimLogCtx(sq);
    INTERPRETER_LOGI(
        "[InterpreterCvSyncSim] WAIT consumed opMagic=%d eventId=%d key=eventId wait_us=%lld preflight_ready=%s "
        "pending_before=%u pending_after=%u detail={%s} thread=%s",
        opMagic, evKey, static_cast<long long>(waitUs), preflightReady ? "true" : "false",
        static_cast<unsigned>(pendingBefore), static_cast<unsigned>(pendingAfter), desc.c_str(),
        CurrentInterpreterThreadTag().c_str());
}

void OperationInterpreter::ResetInterpreterSyncSimulation() { syncSim_->Reset(); }

void OperationInterpreter::InterpreterSyncSimSet(const OpSyncQueue& sq, int opMagic) { syncSim_->Set(sq, opMagic); }

void OperationInterpreter::InterpreterSyncSimWait(const OpSyncQueue& sq, int opMagic) { syncSim_->Wait(sq, opMagic); }

} // namespace npu::tile_fwk
