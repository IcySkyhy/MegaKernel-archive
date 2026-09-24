/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tilefwk/pypto_fwk_log.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_util.h"

namespace npu::tile_fwk {
namespace {
constexpr int kMillisecondsPerSecond = 1000;
constexpr int kPassDetailMonitorIntervalMs = 1;
} // namespace

MonitorImpl::MonitorImpl(MonitorManager* manager) : manager_(manager) {}

MonitorImpl::~MonitorImpl() { Stop(); }

void MonitorImpl::Start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_ && thread_->joinable()) {
        return;
    }
    stop_.store(false);
    thread_ = std::make_unique<std::thread>(&MonitorImpl::MonitorLoop, this);
}

void MonitorImpl::Stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
        stageStartFlag_.store(false);
        cv_.notify_all();
    }
    if (thread_ && thread_->joinable()) {
        thread_->join();
    }
    thread_.reset();
}

bool IsEnabledImmediate(MonitorManager* manager) { return manager->IsEnabled(); }

double GetTimeoutSecImmediate(MonitorManager* manager)
{
    double stageTimeoutSec = manager->GetTimeoutSec();
    // <= 0: stage timeout disabled (do not treat stages as already timed out).
    if (stageTimeoutSec <= 0.0) {
        stageTimeoutSec = static_cast<double>(std::numeric_limits<int>::max());
    }
    return stageTimeoutSec;
}

int GetTotalTimeoutSecImmediate(MonitorManager* manager)
{
    int totalTimeoutSec = manager->GetTotalTimeoutSec();
    if (totalTimeoutSec <= 0) {
        if (totalTimeoutSec < 0) {
            totalTimeoutSec = 600;
        } else {
            totalTimeoutSec = 0;
            manager->SetStageTimeoutFlag("Total");
        }
    }
    return totalTimeoutSec;
}

int GetIntervalSecImmediate(MonitorManager* manager)
{
    int intervalSec = manager->GetIntervalSec();
    if (intervalSec <= 0) {
        intervalSec = 60;
    }
    return intervalSec;
}

void MonitorImpl::StartMonitoring()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stageStartFlag_.store(true);
    }
    cv_.notify_all(); // 唤醒等待的线程
}

void MonitorImpl::StopMonitoring()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stageStartFlag_.store(false);
    }
    cv_.notify_all(); // 唤醒等待的线程
}

void MonitorImpl::PrintTotalTimeOut(double totalElapsed, int totalTimeoutSec)
{
    if ((totalElapsed >= totalTimeoutSec) && (manager_->GetStageTimeoutFlag("Total") == false)) {
        int currentTotalOpsize = manager_->GetFuncSumOpSize();
        manager_->SetStageTimeoutFlag("Total");
        manager_->RecordTimedOutStage("Total", totalElapsed);
        if (manager_->IsPassDetailEnabled()) {
            std::string warnMsg = "[Compiler Monitor] | [== WARNING ==] Total elapsed [" + FormatElapsed(totalElapsed) +
                                  "] exceeded the total time threshold [" +
                                  FormatElapsed(static_cast<double>(totalTimeoutSec)) +
                                  "] | Total number of op: " + std::to_string(currentTotalOpsize) +
                                  ", you can terminate the process by pressing Ctrl+C !!!";
            COMPILER_LOGI("%s", warnMsg.c_str());
            (void)fprintf(stdout, "%s\n", warnMsg.c_str());
            (void)fflush(stdout);
        }
    }
}

namespace {
// Per-tick context passed to each stage handler. Aggregates everything the
// table-driven dispatch needs so each handler stays a pure function of its inputs.
// `lastPrintTime` is a reference so the processing-heartbeat handler can advance the
// caller's local timer; `outBuffer` collects messages so the caller can flush all
// stage output in a single centralized fputs+fflush after dispatch completes.
struct StageTickCtx {
    MonitorManager* manager;
    const ActiveStageInfo& stageInfo;
    double currStageElapsed;
    double totalElapsed;
    double stageTimeoutSec;
    int preCost;
    int printIntervalSec;
    int& lastPrintTime;
    int currentFuncOpsize;
    const std::string& currentFuncName;
    const std::string& currentPassDesc;
    std::vector<std::string>& outBuffer;
};

inline bool ShouldEmitInstanceWarning(const StageTickCtx& ctx)
{
    return ctx.currStageElapsed >= ctx.stageTimeoutSec && ctx.stageInfo.warningPrinted == false;
}

inline bool ShouldEmitGlobalWarning(const StageTickCtx& ctx)
{
    return ctx.currStageElapsed >= ctx.stageTimeoutSec &&
           ctx.manager->GetStageTimeoutFlag(ctx.stageInfo.stageName) == false;
}

inline bool ShouldEmitProcessing(const StageTickCtx& ctx)
{
    if (ctx.currStageElapsed < static_cast<double>(ctx.preCost)) {
        return false;
    }
    int currentTime = static_cast<int>(ctx.currStageElapsed);
    return currentTime >= ctx.lastPrintTime + ctx.printIntervalSec;
}

// Append `msg` to the per-tick output buffer for centralized printf+fflush at
// end of the tick, and log it at INFO severity at the source (so log level is
// preserved per call site even though printf is centralized).
inline void BufferMonitorLine(const StageTickCtx& ctx, std::string msg)
{
    COMPILER_LOGI("%s", msg.c_str());
    ctx.outBuffer.emplace_back(std::move(msg));
}

inline void BufferMonitorWarning(const StageTickCtx& ctx, std::string msg)
{
    COMPILER_LOGW("%s", msg.c_str());
    ctx.outBuffer.emplace_back(std::move(msg));
}

constexpr const char* kWarningHeader = "[Compiler Monitor] | [** WARNING **] ";
constexpr const char* kProcessingPrefix = "  |__ [Compiler Monitor] ";
constexpr const char* kTerminateHint = ", you can terminate the process by pressing Ctrl+C !!!";

inline std::string FormatProgressIndex(int index, int total, int pw)
{
    return PadRight(std::to_string(index) + "/" + std::to_string(total), pw);
}

inline std::string FormatOpCountTail(int opCount, const std::string& extra = "")
{
    return " | Number of op: " + FormatOpCount(opCount) + extra;
}

inline std::string FormatStageElapsedThreshold(double elapsed, double threshold)
{
    return "] elapsed [" + FormatElapsed(elapsed) + "] exceeded the current stage total time threshold [" +
           FormatElapsed(threshold) + "]";
}

inline std::string FormatPassElapsedThreshold(double elapsed, double threshold)
{
    return "] elapsed [" + MonitorManager::FormatPassDurationForLog(elapsed) +
           "] exceeded the pass stage time threshold [" + MonitorManager::FormatPassDurationForLog(threshold) + "]";
}

// Monitor log message builders (timeout WARNING and processing heartbeat):
// - Progress (*MakeProgress*): multi-step flows with idx/total (e.g. Function 2/5, HostMachine 1/4).
// - Labeled (*MakeLabeled*): current stage name only, no idx/total (e.g. single-fn Prepare).
// - Pass (MakePassTimeoutWarn): Pass timeout only; op-scaled threshold and dedicated wording.
//   Pass processing heartbeat still uses MakeProgressProcessing when fnTotal > 1.

inline std::string MakeProgressTimeoutWarn(const std::string& label, int idx, int total, int pw,
                                           const std::string& stageBracket, double elapsed, double threshold,
                                           const std::string& tail, bool detailMode)
{
    return kWarningHeader + label + FormatProgressIndex(idx, total, pw) + " | Stage [" + stageBracket +
           FormatStageElapsedThreshold(elapsed, threshold) + tail + (detailMode ? kTerminateHint : "");
}

inline std::string MakeLabeledTimeoutWarn(const std::string& paddedLabel, const std::string& stage, double elapsed,
                                          double threshold, const std::string& tail, bool detailMode)
{
    return kWarningHeader + paddedLabel + "[" + stage + FormatStageElapsedThreshold(elapsed, threshold) + tail +
           (detailMode ? kTerminateHint : "");
}

inline std::string MakePassTimeoutWarn(int idx, int total, int pw, double elapsed, double threshold,
                                       const std::string& funcName, int opSize, const std::string& passDesc,
                                       bool detailMode)
{
    return std::string(kWarningHeader) + "Function: " + FormatProgressIndex(idx, total, pw) + " | Stage [Pass" +
           FormatPassElapsedThreshold(elapsed, threshold) + " | Func:[" + funcName + "]" +
           FormatOpCountTail(opSize, passDesc) + " | Standard: 200000 ops / 90.0s linear scaled" +
           (detailMode ? kTerminateHint : "");
}

inline std::string MakeProgressProcessing(const std::string& label, int idx, int total, int pw,
                                          const std::string& stageDisplay, double stageElapsed, double totalElapsed,
                                          const std::string& suffix = "")
{
    return kProcessingPrefix + label + FormatProgressIndex(idx, total, pw) + " | Stage: " + PadStageName(stageDisplay) +
           "(processing) | Stage elapsed: " + PadElapsed(FormatElapsed(stageElapsed)) +
           " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) + suffix;
}

inline std::string MakeLabeledProcessing(const std::string& paddedLabel, const std::string& stageDisplay,
                                         double stageElapsed, double totalElapsed, const std::string& suffix = "")
{
    return kProcessingPrefix + paddedLabel + PadStageName(stageDisplay) +
           "(processing) | Stage elapsed: " + PadElapsed(FormatElapsed(stageElapsed)) +
           " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) + suffix;
}

void HandleFuncToBin(const StageTickCtx& ctx)
{
    const auto& stageInfo = ctx.stageInfo;
    const std::string& stage = stageInfo.stageName;
    int totalRootFuncCount = ctx.manager->GetRootFuncCount();
    int pw = ctx.manager->GetProgressWidth();
    bool detailMode = ctx.manager->IsPassDetailEnabled();

    if (ShouldEmitInstanceWarning(ctx)) {
        ctx.manager->SetActiveStageWarningPrinted(stage, stageInfo.rootFuncIndex);
        if (detailMode) {
            BufferMonitorWarning(
                ctx, MakeProgressTimeoutWarn(
                         "RootFunc(parallel): ", stageInfo.rootFuncIndex, totalRootFuncCount, pw, stage,
                         ctx.currStageElapsed, ctx.stageTimeoutSec,
                         " | RootFunc:[" + stageInfo.rootFuncName + "]" + FormatOpCountTail(stageInfo.rootFuncOpSize),
                         true));
        } else {
            ctx.manager->RecordTimedOutStage(stage, ctx.currStageElapsed);
        }
    }
    if (detailMode && ShouldEmitProcessing(ctx)) {
        ctx.lastPrintTime = static_cast<int>(ctx.currStageElapsed);
        BufferMonitorLine(
            ctx, MakeProgressProcessing("RootFunc(parallel): ", stageInfo.rootFuncIndex, totalRootFuncCount, pw,
                                        "CodeGen" + stage, ctx.currStageElapsed, ctx.totalElapsed,
                                        " | RootFunc:[" + stageInfo.rootFuncName + "]"));
    }
}

void HandleHostMachine(const StageTickCtx& ctx)
{
    const auto& stageInfo = ctx.stageInfo;
    const std::string& stage = stageInfo.stageName;
    int totalHostMachineSteps = ctx.manager->GetHostMachineTotalSteps();
    int pw = ctx.manager->GetProgressWidth();
    const std::string& hostMachineStage = stageInfo.rootFuncName.empty() ? STAGE_HOST_MACHINE : stageInfo.rootFuncName;
    bool detailMode = ctx.manager->IsPassDetailEnabled();

    if (ShouldEmitInstanceWarning(ctx)) {
        ctx.manager->SetActiveStageWarningPrinted(stage, stageInfo.rootFuncIndex);
        if (detailMode) {
            BufferMonitorWarning(
                ctx, MakeProgressTimeoutWarn(
                         PadLabel("HostMachine: "), stageInfo.rootFuncIndex, totalHostMachineSteps, pw,
                         hostMachineStage, ctx.currStageElapsed, ctx.stageTimeoutSec,
                         " | Func:[" + ctx.currentFuncName + "]" + FormatOpCountTail(stageInfo.rootFuncOpSize), true));
        } else {
            ctx.manager->RecordTimedOutStage(stage, ctx.currStageElapsed);
        }
    }
    if (detailMode && ShouldEmitProcessing(ctx)) {
        ctx.lastPrintTime = static_cast<int>(ctx.currStageElapsed);
        BufferMonitorLine(
            ctx, MakeProgressProcessing(PadLabel("HostMachine: "), stageInfo.rootFuncIndex, totalHostMachineSteps, pw,
                                        hostMachineStage, ctx.currStageElapsed, ctx.totalElapsed));
    }
}

inline void EmitPassStageTimeoutWarningIfNeeded(const StageTickCtx& ctx)
{
    const auto& stageInfo = ctx.stageInfo;
    const std::string& stage = stageInfo.stageName;
    if (stage != "Pass") {
        return;
    }
    // compile_timeout_stage <= 0 disables stage timeout for all stages including Pass.
    if (ctx.manager->GetTimeoutSec() <= 0.0) {
        return;
    }
    double passStageTimeoutSec = MonitorManager::CalcPassStageTimeoutSec(stageInfo.functionOpSize);
    if (passStageTimeoutSec < 0.0 || ctx.currStageElapsed < passStageTimeoutSec ||
        ctx.manager->GetStageTimeoutFlag(stage)) {
        return;
    }
    ctx.manager->SetStageTimeoutFlag(stage);
    bool detailMode = ctx.manager->IsPassDetailEnabled();
    if (detailMode) {
        int currentFunctionIndex = stageInfo.functionIndex;
        int totalFunctionCount = ctx.manager->GetTotalFunctionCount();
        int pw = ctx.manager->GetProgressWidth();
        BufferMonitorWarning(ctx, MakePassTimeoutWarn(currentFunctionIndex, totalFunctionCount, pw,
                                                      ctx.currStageElapsed, passStageTimeoutSec, stageInfo.functionName,
                                                      stageInfo.functionOpSize, ctx.currentPassDesc, true));
    } else {
        ctx.manager->RecordTimedOutStage(stage, ctx.currStageElapsed);
    }
}

void HandleGenericStage(const StageTickCtx& ctx)
{
    const auto& stageInfo = ctx.stageInfo;
    const std::string& stage = stageInfo.stageName;
    const int fnIdx = stageInfo.functionIndex;
    const int fnTotal = ctx.manager->GetTotalFunctionCount();
    const int pw = ctx.manager->GetProgressWidth();
    const bool multiFn = fnTotal > 1 && fnIdx > 0;
    bool detailMode = ctx.manager->IsPassDetailEnabled();

    EmitPassStageTimeoutWarningIfNeeded(ctx);

    if (stage != "Pass" && ShouldEmitGlobalWarning(ctx)) {
        ctx.manager->SetStageTimeoutFlag(stage);
        if (detailMode) {
            if (multiFn) {
                BufferMonitorWarning(
                    ctx,
                    MakeProgressTimeoutWarn(
                        "Function: ", fnIdx, fnTotal, pw, stage, ctx.currStageElapsed, ctx.stageTimeoutSec,
                        " | Func:[" + ctx.currentFuncName + "]" +
                            FormatOpCountTail(ctx.currentFuncOpsize, stage == "CodeGen" ? "" : ctx.currentPassDesc),
                        true));
            } else {
                BufferMonitorWarning(
                    ctx, MakeLabeledTimeoutWarn(PadLabel("Stage: "), stage, ctx.currStageElapsed, ctx.stageTimeoutSec,
                                                FormatOpCountTail(ctx.currentFuncOpsize, ctx.currentPassDesc), true));
            }
        } else {
            ctx.manager->RecordTimedOutStage(stage, ctx.currStageElapsed);
        }
    }
    if (!detailMode || !ShouldEmitProcessing(ctx)) {
        return;
    }
    ctx.lastPrintTime = static_cast<int>(ctx.currStageElapsed);
    if (!multiFn) {
        BufferMonitorLine(ctx, MakeLabeledProcessing(PadLabel("Stage: "), stage, ctx.currStageElapsed, ctx.totalElapsed,
                                                     " | Stashed function: " + std::to_string(fnTotal)));
        return;
    }
    if (stage == "Pass") {
        BufferMonitorLine(
            ctx, MakeProgressProcessing("Function: ", fnIdx, fnTotal, pw, stage, ctx.currStageElapsed, ctx.totalElapsed,
                                        " | Func:[" + stageInfo.functionName + "]" + ctx.currentPassDesc));
        return;
    }
    BufferMonitorLine(ctx, MakeLabeledProcessing(PadLabel("Stage: "), stage, ctx.currStageElapsed, ctx.totalElapsed));
}

using StageTickHandler = void (*)(const StageTickCtx&);

StageTickHandler ResolveStageHandler(const std::string& stage)
{
    static const std::unordered_map<std::string, StageTickHandler> kStageHandlerTable = {
        {STAGE_FUNC_TO_BIN, &HandleFuncToBin},
        {STAGE_HOST_MACHINE, &HandleHostMachine},
    };
    auto it = kStageHandlerTable.find(stage);
    return (it != kStageHandlerTable.end()) ? it->second : &HandleGenericStage;
}

} // namespace

void MonitorImpl::MonitorLoop()
{
    bool checkEnable = IsEnabledImmediate(manager_);
    int printIntervalSec = GetIntervalSecImmediate(manager_);
    double stageTimeoutSec = GetTimeoutSecImmediate(manager_);
    int totalTimeoutSec = GetTotalTimeoutSecImmediate(manager_);

    COMPILER_LOGI("[Compiler Monitor] interval_sec=%d, stage_timeout_sec=%.3f, total_timeout_sec=%d, check_enable=%d",
                  printIntervalSec, stageTimeoutSec, totalTimeoutSec, checkEnable);

    int preCost = manager_->GetProcessingThresholdSec();
    int checkIntervalSec = 1;
    int lastPrintTime = 0;

    while (!stop_.load()) {
        // 检查 start_flag，如果为 false 则等待
        if (!stageStartFlag_.load()) {
            std::unique_lock<std::mutex> lock(mutex_);
            // 等待直到 start_flag 变为 true 或 stop_ 变为 true
            cv_.wait(lock, [this] { return stageStartFlag_.load() || stop_.load(); });
            if (stageStartFlag_.load()) {
                lastPrintTime = 0;
            }
            if (stop_.load()) {
                break;
            }
            lock.unlock();
        }

        // 检查是否超过600秒总超时
        auto now = std::chrono::steady_clock::now();
        auto totalStart = manager_->GetTotalStartTime();
        auto totalElapsed = std::chrono::duration<double>(now - totalStart).count();
        int currentFuncOpsize = manager_->GetCurrentFuncOpSize();

        // 当总时间超过total_timeout_sec
        PrintTotalTimeOut(totalElapsed, totalTimeoutSec);

        auto waitDuration = std::chrono::milliseconds(checkIntervalSec * kMillisecondsPerSecond);
        if (checkEnable && manager_->GetCurrentStageName() == STAGE_PASS && manager_->IsPassDetailEnabled()) {
            waitDuration = std::chrono::milliseconds(kPassDetailMonitorIntervalMs);
        }
        std::unique_lock<std::mutex> lock(mutex_);

        // 修改等待条件：检查 stop_ 和 stageStartFlag_
        cv_.wait_for(lock, waitDuration, [this] { return stop_.load() || !stageStartFlag_.load(); });
        if (stop_.load()) {
            break;
        }
        lock.unlock();

        // 如果 start_flag 变为 false，则回到循环开头重新等待
        if (!stageStartFlag_.load()) {
            continue;
        }

        if (!checkEnable) {
            continue;
        }

        auto activeStages = manager_->GetActiveStages();
        if (activeStages.empty()) {
            continue;
        }

        now = std::chrono::steady_clock::now();
        totalStart = manager_->GetTotalStartTime();
        totalElapsed = std::chrono::duration<double>(now - totalStart).count();

        PrintTotalTimeOut(totalElapsed, totalTimeoutSec);

        std::vector<std::string> outBuffer;
        outBuffer.reserve(activeStages.size() * 2);

        for (const auto& stageInfo : activeStages) {
            totalElapsed = std::chrono::duration<double>(now - totalStart).count();
            PrintTotalTimeOut(totalElapsed, totalTimeoutSec);
            double currStageElapsed = std::chrono::duration<double>(now - stageInfo.startTime).count();
            std::string currentFuncName = manager_->GetCurrentFunctionName();
            std::string currentPassDesc = manager_->GetCurrentPassDescription();

            StageTickCtx ctx{
                manager_,         stageInfo,     currStageElapsed,  totalElapsed,    stageTimeoutSec, preCost,
                printIntervalSec, lastPrintTime, currentFuncOpsize, currentFuncName, currentPassDesc, outBuffer,
            };
            ResolveStageHandler(stageInfo.stageName)(ctx);
        }

        if (!outBuffer.empty()) {
            std::string combined;
            for (const auto& line : outBuffer) {
                combined.append(line).append("\n");
            }
            (void)fputs(combined.c_str(), stdout);
            (void)fflush(stdout);
        }
    }
}

} // namespace npu::tile_fwk
