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
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <thread>

#include "tilefwk/pypto_fwk_log.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_util.h"

namespace npu::tile_fwk {
namespace {
constexpr int kPassTimeoutBaseOpSize = 200000;
constexpr double kSinglePassTimeoutBaseSec = 20.0;
constexpr double kPassStageTimeoutBaseSec = 90.0;
constexpr double kMillisecondsPerSecond = 1000.0;
constexpr int kPassDurationPrecision = 3;

double CalcPassTimeoutSec(int opSize, double baseTimeoutSec)
{
    if (opSize < 0) {
        return -1.0;
    }
    int normalizedOpSize = std::max(opSize, 1);
    return baseTimeoutSec * static_cast<double>(normalizedOpSize) / static_cast<double>(kPassTimeoutBaseOpSize);
}

std::string FormatPassDuration(double seconds)
{
    if (seconds < 0.0) {
        return "-1";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(kPassDurationPrecision) << seconds * kMillisecondsPerSecond << "ms";
    return oss.str();
}
} // namespace

MonitorManager::~MonitorManager() { Shutdown(); }

MonitorManager& MonitorManager::Instance()
{
    static MonitorManager instance;
    return instance;
}

double MonitorManager::CalcPassStageTimeoutSec(int opSize)
{
    return CalcPassTimeoutSec(opSize, kPassStageTimeoutBaseSec);
}

std::string MonitorManager::FormatPassDurationForLog(double seconds) { return FormatPassDuration(seconds); }

void MonitorManager::Initialize(bool enable, int intervalSec, double timeoutSec, int totalTimeoutSec,
                                bool passDetailEnable)
{
    std::lock_guard<std::mutex> lock(mutex_);
    this->SetCompilerMonitorOptions(enable, intervalSec, timeoutSec, totalTimeoutSec, enable && passDetailEnable);
    if (initialized_) {
        return;
    }
    if (!enable) {
        return;
    }
    impl_ = new MonitorImpl(this);
    currentStage_ = "Prepare";
    totalStart_ = std::chrono::steady_clock::now();
    stageStart_ = totalStart_;
    nextFunctionIndex_ = 1;
    impl_->Start();
    initialized_ = true;
    stageDoing_ = true;
    stageElapsedTotals_.clear();
    passCompileTimings_.clear();
    passElapsedTotals_.clear();
    lastPassDetailFunctionIndex_ = -1;
    lastPassDetailFunctionName_.clear();
    lastPassDetailStrategy_.clear();
    stageTimeoutFlag_.clear();
    stageTimeoutFlag_["Prepare"] = false;
    stageTimeoutFlag_["Pass"] = false;
    stageTimeoutFlag_["CodeGen"] = false;
    stageTimeoutFlag_[STAGE_FUNC_TO_BIN] = false;
    stageTimeoutFlag_["Total"] = false;
    summaryTimedOutStages_.clear();
    pythonStageEnded_ = false;
    hasSuspendedCodeGen_ = false;
    suspendedCodeGen_ = ActiveStageInfo{};
    stageElapsedTotals_["Prepare"] = 0.0;
    // Mark that Prepare stage has started (use env var for cross-.so communication)
    (void)setenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED", "1", 1);
    impl_->StartMonitoring();
    if (!passDetailEnable) {
        std::string thresholdMsg = "Threshold: " + std::to_string(totalTimeoutSec_.load()) +
                                   "s | You can adjust the timeout threshold via "
                                   "pypto.set_host_options(compile_timeout=...)";
        COMPILER_EVENT("%s", thresholdMsg.c_str());
    }
}

void MonitorManager::Shutdown()
{
    MonitorImpl* implToStop = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return;
        }
        if (impl_) {
            implToStop = impl_;
            impl_ = nullptr;
        }
        initialized_ = false;
        stageDoing_ = false;
    }
    if (implToStop) {
        implToStop->Stop();
        delete implToStop;
    }
}

void MonitorManager::RecordTimedOutStage(const std::string& name, double elapsed)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (summaryTimedOutStages_.find(name) == summaryTimedOutStages_.end()) {
        summaryTimedOutStages_[name] = elapsed;
    }
}

void MonitorManager::MaybeStartTotalClock()
{
    if (currentStage_.empty()) {
        totalStart_ = std::chrono::steady_clock::now();
        stageStart_ = totalStart_;
    }
}

double MonitorManager::GetCurrentStageElapsed(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return 0.0;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    if (stageDoing_ && name == currentStage_) {
        elapsed = std::chrono::duration<double>(now - stageStart_).count();
    }
    return elapsed;
}

void MonitorManager::SetTotalFunctionCount(int n)
{
    if (!enable_) {
        return;
    }
    MonitorImpl* toStart = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Check if Prepare stage was started via Python (env var indicates this)
        bool prepareStarted = (std::getenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED") != nullptr);
        if (!initialized_ && enable_ && !prepareStarted) {
            // First time initialization (no Prepare stage from Python)
            if (!impl_) {
                impl_ = new MonitorImpl(this);
            }
            currentStage_ = "Prepare";
            totalStart_ = std::chrono::steady_clock::now();
            stageStart_ = totalStart_;
            stageElapsedTotals_.clear();
            passCompileTimings_.clear();
            passElapsedTotals_.clear();
            lastPassDetailFunctionIndex_ = -1;
            lastPassDetailFunctionName_.clear();
            lastPassDetailStrategy_.clear();
            stageTimeoutFlag_.clear();
            pythonStageEnded_ = false;
            summaryTimedOutStages_.clear();
            hasSuspendedCodeGen_ = false;
            suspendedCodeGen_ = ActiveStageInfo{};
            toStart = impl_;
            initialized_ = true;
        }
        totalFunctionCount_ = n;
        nextFunctionIndex_ = 1;
        currentFunctionIndex_ = 0;
        (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", "0", 1); // 进程内唯一，避免多 .so 多单例
    }
    if (toStart != nullptr) {
        toStart->Start();
    }
}

int MonitorManager::GetAndIncrementNextFunctionIndex()
{
    if (!enable_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int k = nextFunctionIndex_++;
    return k;
}

void MonitorManager::SetCurrentFunctionIndex(int k)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    currentFunctionIndex_ = k;
    std::string val = std::to_string(k);
    (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", val.c_str(), 1);
}

void MonitorManager::SetRootFuncCount(int n)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    rootFuncCount_ = n;
    nextRootFuncIndex_ = 1;
    currentRootFuncIndex_ = 0;
}

int MonitorManager::PrepareNextRootFunc()
{
    if (!enable_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int k = nextRootFuncIndex_++;
    return k;
}

std::string MonitorManager::GetCurrentRootFuncName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentRootFunc_;
}

int MonitorManager::GetRootFuncCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return rootFuncCount_;
}

int MonitorManager::GetCurrentRootFuncIndex() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentRootFuncIndex_;
}

void MonitorManager::BeginHostMachineCompileGroup(int totalSteps)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    hostMachineTotalSteps_ = std::max(1, totalSteps);
    hostMachineNextStep_ = 1;
}

int MonitorManager::AllocHostMachineStepIndex()
{
    if (!enable_) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const int k = hostMachineNextStep_;
    ++hostMachineNextStep_;
    return k;
}

int MonitorManager::GetHostMachineTotalSteps() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::max(1, hostMachineTotalSteps_);
}

void MonitorManager::TryEndPrepareStage()
{
    if (!impl_ || !enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->StopMonitoring();
    if (!initialized_ || pythonStageEnded_) {
        return;
    }
    pythonStageEnded_ = true;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - stageStart_).count();
    stageElapsedTotals_["Prepare"] += elapsed;

    if (currentStage_ == "Prepare" && enable_) {
        double totalElapsedPrepare = std::chrono::duration<double>(std::chrono::steady_clock::now() - totalStart_)
                                         .count();
        std::string msg = "[Compiler Monitor] Stage: " + currentStage_ +
                          "(completed) | Stashed function: " + std::to_string(totalFunctionCount_) +
                          " | Stage elapsed: " + FormatElapsed(elapsed) +
                          " | Total elapsed: " + FormatElapsed(totalElapsedPrepare);
        COMPILER_LOGI(
            "%s. current thread_id:%s pid:%ld ppid:%ld", msg.c_str(),
            []() {
                std::stringstream ss;
                ss << std::this_thread::get_id();
                return ss.str();
            }()
                .c_str(),
            static_cast<long>(getpid()), static_cast<long>(getppid()));
        if (passDetailEnable_) {
            (void)fprintf(stdout, "%s\n", msg.c_str());
            (void)fflush(stdout);
        }
    }
    if (timeoutSec_.load() > 0.0 && elapsed > timeoutSec_.load() &&
        summaryTimedOutStages_.find("Prepare") == summaryTimedOutStages_.end()) {
        stageTimeoutFlag_["Prepare"] = true;
        summaryTimedOutStages_["Prepare"] = elapsed;
    }
}

void MonitorManager::NotifyCompilationFinished()
{
    MonitorImpl* implToStop = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return;
        }
        PrintCompilationFinished();
        if (impl_) {
            implToStop = impl_;
            impl_ = nullptr;
        }
        initialized_ = false;
        stageDoing_ = false;
    }
    if (implToStop) {
        implToStop->Stop();
        delete implToStop;
    }
}

double MonitorManager::ComputeTotalElapsed() const
{
    auto now = std::chrono::steady_clock::now();
    double totalElapsed = std::chrono::duration<double>(now - totalStart_).count();
    double stageTotal = 0.0;
    for (const auto& kv : stageElapsedTotals_) {
        // HostMachine is nested under the CodeGen MonitorStageScope in DYNAMIC path;
        // skip to avoid double-count against CodeGen wall-clock.
        if (kv.first == STAGE_HOST_MACHINE || kv.first == STAGE_FUNC_TO_BIN) {
            continue;
        }
        stageTotal += kv.second;
    }
    if (stageTotal > totalElapsed) {
        totalElapsed = stageTotal;
    }
    return totalElapsed;
}

void MonitorManager::PrintWatchdogSummary(double totalElapsed)
{
    if (totalElapsed < static_cast<double>(totalTimeoutSec_.load())) {
        lastTotalElapsed_ = totalElapsed;
        return;
    }
    static const std::vector<std::string> stageOrder = {"Prepare", "Pass", "HostMachine", "CodeGen"};
    double avgThreshold = static_cast<double>(totalTimeoutSec_.load()) / static_cast<double>(stageOrder.size());
    std::string detectedStage;
    std::vector<double> stageElapsed;
    for (const auto& name : stageOrder) {
        auto it = stageElapsedTotals_.find(name);
        double elapsed = (it != stageElapsedTotals_.end()) ? it->second : 0.0;
        stageElapsed.push_back(elapsed);
        if (elapsed > avgThreshold) {
            if (!detectedStage.empty()) {
                detectedStage += ", ";
            }
            detectedStage += name;
        }
    }
    std::string summaryMsg = "Threshold: " + std::to_string(totalTimeoutSec_.load()) +
                             "s | DetectedStage: " + (detectedStage.empty() ? "None" : detectedStage);
    for (size_t i = 0; i < stageOrder.size(); i++) {
        summaryMsg += " | " + stageOrder[i] + ": " + FormatElapsed(stageElapsed[i]);
    }
    COMPILER_WARN_EVENT("%s", summaryMsg.c_str());
    lastTotalElapsed_ = totalElapsed;
}

void MonitorManager::PrintDetailSummary(double totalElapsed)
{
    std::string compilationMsg = "[Compiler Monitor] Compilation finished " + std::to_string(currentFunctionIndex_) +
                                 "/" + std::to_string(totalFunctionCount_ > 0 ? totalFunctionCount_ : 1) +
                                 " | Total functions: " +
                                 std::to_string(totalFunctionCount_ > 0 ? totalFunctionCount_ : 1);
    (void)fprintf(stdout, "%s\n", compilationMsg.c_str());
    (void)fflush(stdout);
    COMPILER_LOGI("%s", compilationMsg.c_str());

    int n = totalFunctionCount_ > 0 ? totalFunctionCount_ : 1;
    std::ostringstream stageMsg;
    for (const auto& [stage, sec] : stageElapsedTotals_) {
        if (stage == "Pass" || stage == "CodeGen" || stage == STAGE_HOST_MACHINE) {
            stageMsg << " " << ("[" + stage + "]:") << std::fixed << std::setprecision(1) << sec << "s"
                     << " ";
        } else {
            stageMsg << " " << ("[" + stage + "]:") << std::fixed << std::setprecision(1) << sec << "s  (sum over " << n
                     << " functions)\n";
        }
    }
    std::string stageTimingMsg = "[Compiler Monitor] Stage timing (aggregated by stage):" + stageMsg.str();
    COMPILER_LOGI("%s", stageTimingMsg.c_str());
    (void)fprintf(stdout, "%s\n", stageTimingMsg.c_str());
    (void)fflush(stdout);

    std::string finalMsg = "[Compiler Monitor] Monitoring stopped | Total elapsed: " + FormatElapsed(totalElapsed);
    COMPILER_LOGI("%s", finalMsg.c_str());
    (void)fprintf(stdout, "%s\n", finalMsg.c_str());
    (void)fflush(stdout);

    lastTotalElapsed_ = totalElapsed;
}

void MonitorManager::PrintCompilationFinished()
{
    if (!enable_) {
        return;
    }
    double totalElapsed = ComputeTotalElapsed();
    if (!passDetailEnable_) {
        PrintWatchdogSummary(totalElapsed);
        return;
    }
    PrintDetailSummary(totalElapsed);
}

std::string MonitorManager::BuildPassCompileTimingsForFunction(int functionIndex, const std::string& functionName) const
{
    if (passCompileTimings_.empty()) {
        return "";
    }

    std::vector<PassCompileTiming> timings;
    for (const auto& timing : passCompileTimings_) {
        if (timing.functionIndex == functionIndex && timing.functionName == functionName) {
            timings.push_back(timing);
        }
    }
    if (timings.empty()) {
        return "";
    }

    std::sort(timings.begin(), timings.end(), [](const PassCompileTiming& lhs, const PassCompileTiming& rhs) {
        if (lhs.strategy != rhs.strategy) {
            return lhs.strategy < rhs.strategy;
        }
        return lhs.passIndex < rhs.passIndex;
    });

    std::ostringstream detailMsg;
    detailMsg << BuildPassFunctionHeaderLocked(functionIndex, functionName);
    std::string lastStrategy;
    for (const auto& timing : timings) {
        if (timing.strategy != lastStrategy) {
            detailMsg << "  |__ [Compiler Monitor] Strategy:[" << timing.strategy << "]\n";
            lastStrategy = timing.strategy;
        }
        double passTimeoutSec = CalcPassTimeoutSec(timing.functionOpSize, kSinglePassTimeoutBaseSec);
        bool passTimeoutEnabled = passTimeoutSec >= 0.0;
        bool passTimedOut = passTimeoutEnabled && timing.elapsedSec > passTimeoutSec;
        detailMsg << "  |__   [" << std::setw(2) << std::setfill('0') << timing.passIndex << std::setfill(' ') << "] "
                  << timing.passIdentifier << " | Pass elapsed: " << FormatPassDuration(timing.elapsedSec);
        if (passTimeoutEnabled) {
            detailMsg << " | Threshold: " << FormatPassDuration(passTimeoutSec);
        }
        detailMsg << (passTimedOut ? " | [** WARNING **] single pass timeout" : "")
                  << (timing.success ? "" : " | Result:[FAILED]") << "\n";
    }

    return detailMsg.str();
}

std::string MonitorManager::BuildPassFunctionHeaderLocked(int functionIndex, const std::string& functionName,
                                                          const std::string& strategy) const
{
    int totalFunctions = totalFunctionCount_ > 0 ? totalFunctionCount_ : 1;
    std::ostringstream oss;
    oss << "  |__ [Compiler Monitor] Function: " << functionIndex << "/" << totalFunctions << " | Pass timing detail";
    if (!strategy.empty()) {
        oss << " | Strategy:[" << strategy << "]";
    }
    oss << " | Func:[" << functionName << "]\n";
    return oss.str();
}

std::string MonitorManager::BuildPassProgressLineLocked(const std::string& passIdentifier, size_t passIndex,
                                                        int functionOpSize, const std::string& status,
                                                        double elapsedSec, bool success) const
{
    double passTimeoutSec = CalcPassTimeoutSec(functionOpSize, kSinglePassTimeoutBaseSec);
    bool passTimeoutEnabled = passTimeoutSec >= 0.0;
    bool passTimedOut = passTimeoutEnabled && elapsedSec > passTimeoutSec;

    std::ostringstream oss;
    oss << "  |__ [Compiler Monitor] [" << std::setw(2) << std::setfill('0') << passIndex << std::setfill(' ') << "] "
        << passIdentifier;
    if (status == "running") {
        oss << " start";
    } else {
        oss << " | Pass elapsed: " << FormatPassDuration(elapsedSec);
    }
    if (status != "running" && passTimeoutEnabled) {
        oss << " | Threshold: " << FormatPassDuration(passTimeoutSec);
    }
    if (passTimedOut) {
        oss << " | [** WARNING **] single pass timeout";
    }
    if (!success) {
        oss << " | Result:[FAILED]";
    }
    oss << "\n";
    return oss.str();
}

void MonitorManager::SetCompilerMonitorOptions(bool enable, int intervalSec, double timeoutSec, int totalTimeoutSec,
                                               bool passDetailEnable)
{
    enable_ = enable;
    passDetailEnable_ = passDetailEnable;
    intervalSec_.store((intervalSec > 0) ? intervalSec : 60);
    std::string intervalStr = std::to_string(intervalSec_.load());
    (void)setenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC", intervalStr.c_str(), 1);
    timeoutSec_.store((timeoutSec > 0.0) ? timeoutSec : 0.0);
    std::string timeoutStr = std::to_string(timeoutSec_.load());
    (void)setenv("PYPTO_COMPILER_MONITOR_TIMEOUT_SEC", timeoutStr.c_str(), 1);
    totalTimeoutSec_.store((totalTimeoutSec >= 0) ? totalTimeoutSec : 600);
    std::string totalTimeoutStr = std::to_string(totalTimeoutSec_.load());
    (void)setenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC", totalTimeoutStr.c_str(), 1);
}

bool MonitorManager::IsEnabled() const { return enable_; }

bool MonitorManager::IsPassDetailEnabled() const { return passDetailEnable_; }

void MonitorManager::SetStageTimeoutFlag(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stageTimeoutFlag_.count(name) > 0) {
        stageTimeoutFlag_[name] = true;
    }
}

bool MonitorManager::GetStageTimeoutFlag(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stageTimeoutFlag_.count(name) > 0) {
        return stageTimeoutFlag_[name];
    }
    return false;
}

void MonitorManager::SetActiveStageWarningPrinted(const std::string& name, int rootFuncIndex)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& info : activeStages_) {
        if (info.stageName == name && info.rootFuncIndex == rootFuncIndex) {
            info.warningPrinted = true;
            break;
        }
    }
}

int MonitorManager::GetIntervalSec() const { return intervalSec_.load(); }

double MonitorManager::GetTimeoutSec() const { return timeoutSec_.load(); }

int MonitorManager::GetTotalTimeoutSec() const { return totalTimeoutSec_.load(); }

std::string MonitorManager::GetCurrentStageName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentStage_;
}

void MonitorManager::SwitchStageReset()
{
    this->SetCurrentFunctionName("");
    this->SetCurrentFuncOpSize(0);
    this->SetFuncSumOpSize(0, true);
}

std::string MonitorManager::GetCurrentFunctionName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentFunction_;
}

void MonitorManager::SetCurrentFunctionName(const std::string& name)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    currentFunction_ = name;
}

int MonitorManager::GetCurrentFuncOpSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentFuncOpsize_;
}

void MonitorManager::SetCurrentFuncOpSize(int opSize, bool updateActiveStage)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    currentFuncOpsize_ = opSize;
    if (!updateActiveStage) {
        return;
    }
    for (auto& info : activeStages_) {
        if (info.stageName == "Pass" && info.functionIndex == currentFunctionIndex_ &&
            info.functionName == currentFunction_) {
            info.functionOpSize = opSize;
        }
    }
}

void MonitorManager::PrintCurrentTotalElapsed(std::string strTemp)
{
    if (!enable_ || !passDetailEnable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    double totalElapsed = std::chrono::duration<double>(now - totalStart_).count();
    std::string stageFinishMsg = "[Compiler Monitor] " + strTemp + " | Total elapsed: " + FormatElapsed(totalElapsed);
    (void)fprintf(stdout, "%s\n", stageFinishMsg.c_str());
    (void)fflush(stdout);
}

int MonitorManager::GetFuncSumOpSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return funcSumOpsize_;
}

void MonitorManager::SetFuncSumOpSize(size_t opSize, bool reset)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    funcSumOpsize_ += static_cast<int>(opSize);
    if (reset) {
        funcSumOpsize_ = 0;
    }
}

std::chrono::steady_clock::time_point MonitorManager::GetStageStartTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stageStart_;
}

std::chrono::steady_clock::time_point MonitorManager::GetTotalStartTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return totalStart_;
}

int MonitorManager::GetTotalFunctionCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return totalFunctionCount_;
}

int MonitorManager::GetCurrentFunctionIndex() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentFunctionIndex_;
}

std::unordered_map<std::string, double> MonitorManager::GetStageElapsedTotals() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stageElapsedTotals_;
}

void MonitorManager::RecordPassCompileTime(const std::string& strategy, const std::string& passIdentifier,
                                           size_t passIndex, const std::string& functionName, int functionIndex,
                                           int functionOpSize, double elapsedSec, bool success)
{
    std::string progressLine;
    bool passTimedOut = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_ || !initialized_) {
            return;
        }
        PassCompileTiming timing{functionIndex,  functionName, functionOpSize, strategy,
                                 passIdentifier, passIndex,    elapsedSec,     success};
        passCompileTimings_.push_back(timing);
        passElapsedTotals_[strategy + "::" + passIdentifier] += elapsedSec;
        if (passDetailEnable_ && currentStage_ == STAGE_PASS) {
            progressLine = BuildPassProgressLineLocked(passIdentifier, passIndex, functionOpSize, "completed",
                                                       elapsedSec, success);
            double passTimeoutSec = CalcPassTimeoutSec(functionOpSize, kSinglePassTimeoutBaseSec);
            passTimedOut = passTimeoutSec >= 0.0 && elapsedSec > passTimeoutSec;
        }
    }
    if (!passDetailEnable_) {
        if (timeoutSec_.load() > 0.0) {
            double passTimeoutSec = CalcPassTimeoutSec(functionOpSize, kSinglePassTimeoutBaseSec);
            if (passTimeoutSec >= 0.0 && elapsedSec > passTimeoutSec) {
                summaryTimedOutStages_["Pass"] = elapsedSec;
            }
        }
        return;
    }
    if (!progressLine.empty()) {
        std::lock_guard<std::mutex> printLock(passDetailPrintMutex_);
        (void)fprintf(stdout, "%s", progressLine.c_str());
        (void)fflush(stdout);
        if (passTimedOut) {
            COMPILER_LOGW("%s", progressLine.c_str());
        } else {
            COMPILER_LOGI("%s", progressLine.c_str());
        }
    }
}

std::vector<PassCompileTiming> MonitorManager::GetPassCompileTimings() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return passCompileTimings_;
}

std::map<std::string, double> MonitorManager::GetPassElapsedTotals() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return passElapsedTotals_;
}

void MonitorManager::StartPassCompile(const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
                                      const std::string& functionName, int functionIndex, int functionOpSize)
{
    std::string progressLine;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_ || !initialized_) {
            return;
        }
        currentPass_.active = true;
        currentPass_.functionIndex = functionIndex;
        currentPass_.functionName = functionName;
        currentPass_.functionOpSize = functionOpSize;
        currentPass_.strategy = strategy;
        currentPass_.passIdentifier = passIdentifier;
        currentPass_.passIndex = passIndex;
        currentPass_.startTime = std::chrono::steady_clock::now();
        if (passDetailEnable_ && currentStage_ == STAGE_PASS) {
            if (lastPassDetailFunctionIndex_ != functionIndex || lastPassDetailFunctionName_ != functionName ||
                lastPassDetailStrategy_ != strategy) {
                progressLine += BuildPassFunctionHeaderLocked(functionIndex, functionName, strategy);
                lastPassDetailFunctionIndex_ = functionIndex;
                lastPassDetailFunctionName_ = functionName;
                lastPassDetailStrategy_ = strategy;
            }
            progressLine += BuildPassProgressLineLocked(passIdentifier, passIndex, functionOpSize, "running", 0.0,
                                                        true);
        }
    }
    if (!progressLine.empty()) {
        std::lock_guard<std::mutex> printLock(passDetailPrintMutex_);
        (void)fprintf(stdout, "%s", progressLine.c_str());
        (void)fflush(stdout);
        COMPILER_LOGI("%s", progressLine.c_str());
    }
}

void MonitorManager::EndPassCompile(const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
                                    const std::string& functionName, int functionIndex)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!currentPass_.active) {
        return;
    }
    if (currentPass_.strategy == strategy && currentPass_.passIdentifier == passIdentifier &&
        currentPass_.passIndex == passIndex && currentPass_.functionName == functionName &&
        currentPass_.functionIndex == functionIndex) {
        currentPass_ = CurrentPassInfo{};
    }
}

std::string MonitorManager::GetCurrentPassDescription() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!currentPass_.active) {
        return "";
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - currentPass_.startTime).count();
    std::ostringstream oss;
    oss << " | Current pass:[" << currentPass_.passIdentifier << "]"
        << " | Pass idx:[" << currentPass_.passIndex << "]"
        << " | Pass elapsed:[" << FormatPassDuration(elapsed) << "]"
        << " | Strategy:[" << currentPass_.strategy << "]";
    return oss.str();
}

void MonitorManager::StartStage(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                                int rootFuncOpSize)
{
    COMPILER_LOGI("Stage ==[%s]== begin.", name.c_str());
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }

    // Suspend CodeGen while nested FuncToBin / HostMachine runs (avoid dual tick warnings),
    // but keep its startTime so the outer CodeGen EndStage still covers the full wall-clock span.
    if (name == STAGE_FUNC_TO_BIN || name == STAGE_HOST_MACHINE) {
        auto it = std::find_if(activeStages_.begin(), activeStages_.end(),
                               [](const ActiveStageInfo& info) { return info.stageName == "CodeGen"; });
        if (it != activeStages_.end()) {
            suspendedCodeGen_ = *it;
            hasSuspendedCodeGen_ = true;
            activeStages_.erase(it);
        }
    }

    if (rootFuncIndex == -1) {
        impl_->StartMonitoring();
    }
    MaybeStartTotalClock();
    currentStage_ = name;
    stageStart_ = std::chrono::steady_clock::now();
    stageDoing_ = true;

    ActiveStageInfo info;
    info.stageName = name;
    info.startTime = stageStart_;
    info.functionIndex = currentFunctionIndex_;
    info.functionName = currentFunction_;
    info.functionOpSize = currentFuncOpsize_;
    info.rootFuncIndex = (rootFuncIndex < 0) ? currentRootFuncIndex_ : rootFuncIndex;
    info.rootFuncName = (rootFuncIndex < 0) ? currentRootFunc_ : rootFuncName;
    info.rootFuncOpSize = rootFuncOpSize;
    activeStages_.push_back(info);
}

void MonitorManager::EndStage(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                              int rootFuncOpSize)
{
    std::lock_guard<std::mutex> lock(mutex_);
    int actualRootFuncIndex = rootFuncIndex;
    std::string actualRootFuncName = rootFuncName;
    auto stageStartTime = stageStart_;
    int actualFunctionIndex = currentFunctionIndex_;
    std::string actualFunctionName = currentFunction_;
    int actualFunctionOpSize = currentFuncOpsize_;

    auto it = activeStages_.rend();
    if (rootFuncIndex < 0) {
        it = std::find_if(activeStages_.rbegin(), activeStages_.rend(),
                          [&name](const ActiveStageInfo& info) { return info.stageName == name; });
        if (it != activeStages_.rend()) {
            actualRootFuncIndex = it->rootFuncIndex;
            actualRootFuncName = it->rootFuncName;
            stageStartTime = it->startTime;
            actualFunctionIndex = it->functionIndex;
            actualFunctionName = it->functionName;
            actualFunctionOpSize = it->functionOpSize;
        } else if (name == "CodeGen" && hasSuspendedCodeGen_) {
            stageStartTime = suspendedCodeGen_.startTime;
            actualRootFuncIndex = suspendedCodeGen_.rootFuncIndex;
            actualRootFuncName = suspendedCodeGen_.rootFuncName;
            actualFunctionIndex = suspendedCodeGen_.functionIndex;
            actualFunctionName = suspendedCodeGen_.functionName;
            actualFunctionOpSize = suspendedCodeGen_.functionOpSize;
            hasSuspendedCodeGen_ = false;
            suspendedCodeGen_ = ActiveStageInfo{};
        } else {
            actualRootFuncIndex = currentRootFuncIndex_;
            actualRootFuncName = currentRootFunc_;
        }
    } else {
        it = std::find_if(activeStages_.rbegin(), activeStages_.rend(), [&](const ActiveStageInfo& info) {
            return info.stageName == name && info.rootFuncIndex == rootFuncIndex;
        });
        if (it != activeStages_.rend()) {
            stageStartTime = it->startTime;
            actualFunctionIndex = it->functionIndex;
            actualFunctionName = it->functionName;
            actualFunctionOpSize = it->functionOpSize;
        }
    }

    if (it != activeStages_.rend()) {
        activeStages_.erase(std::prev(it.base()));
    }
    EndStageInternal(name, actualRootFuncIndex, actualRootFuncName, stageStartTime, rootFuncIndex, rootFuncOpSize,
                     actualFunctionIndex, actualFunctionName, actualFunctionOpSize);
}

std::string MonitorManager::BuildStageFinishMsg(const std::string& name, int rootFuncIndex,
                                                const std::string& rootFuncName, double elapsed, double totalElapsed,
                                                int functionIndex, const std::string& functionName, int functionOpSize,
                                                int rootFuncOpSize) const
{
    if (name == STAGE_FUNC_TO_BIN) {
        int pw = GetProgressWidth();
        return "[Compiler Monitor] " + PadLabel("Function(parallel): ") +
               PadRight(std::to_string(rootFuncIndex) + "/" + std::to_string(rootFuncCount_), pw) +
               " | Stage: " + PadStageName("CodeGen[" + name + "]") +
               "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
               " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) + " | Func:[" + rootFuncName +
               "] Ops: " + FormatOpCount(rootFuncOpSize);
    }
    if (name == STAGE_HOST_MACHINE) {
        int pw = GetProgressWidth();
        const int denom = std::max(hostMachineTotalSteps_, std::max(1, rootFuncIndex));
        const std::string& hostMachineStage = rootFuncName.empty() ? STAGE_HOST_MACHINE : rootFuncName;
        return "[Compiler Monitor] " + PadLabel("HostMachine: ") +
               PadRight(std::to_string(rootFuncIndex) + "/" + std::to_string(denom), pw) +
               " | Stage: " + PadStageName(hostMachineStage) +
               "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
               " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) +
               " | Ops: " + FormatOpCount(rootFuncOpSize);
    }
    if (name == "CodeGen") {
        return "[Compiler Monitor] Stage: " + name +
               "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
               " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed));
    }
    int pw = GetProgressWidth();
    return "[Compiler Monitor] " + PadLabel("Function: ") +
           PadRight(std::to_string(functionIndex) + "/" + std::to_string(totalFunctionCount_), pw) +
           " | Stage: " + PadStageName(name) + "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
           " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) + " | Func:[" + functionName +
           "] Ops: " + FormatOpCount(functionOpSize);
}

void MonitorManager::CheckStageTimeoutOnEnd(const std::string& name, double elapsed, int functionOpSize,
                                            int functionIndex, const std::string& functionName,
                                            bool stageWarningPrinted)
{
    // compile_timeout_stage <= 0: stage timeout disabled for all stages including Pass.
    if (timeoutSec_.load() <= 0.0) {
        return;
    }
    if (name == "Pass") {
        double timeoutSec = CalcPassTimeoutSec(functionOpSize, kPassStageTimeoutBaseSec);
        if (timeoutSec >= 0.0 && elapsed > timeoutSec && !stageWarningPrinted) {
            stageTimeoutFlag_[name] = true;
            summaryTimedOutStages_[name] = elapsed;
            if (passDetailEnable_) {
                int pw = GetProgressWidth();
                std::string passTimeoutWarnMsg = "[Compiler Monitor] | [** WARNING **] Function: " +
                                                 PadRight(std::to_string(functionIndex) + "/" +
                                                              std::to_string(totalFunctionCount_),
                                                          pw) +
                                                 " | Stage [Pass] elapsed [" + FormatPassDuration(elapsed) +
                                                 "] exceeded the pass stage time threshold [" +
                                                 FormatPassDuration(timeoutSec) + "] | Func:[" + functionName +
                                                 "] | Number of op: " + FormatOpCount(functionOpSize) +
                                                 " | Standard: 200000 ops / 90.0s linear scaled" +
                                                 ", you can terminate the process by pressing Ctrl+C !!!";
                (void)fprintf(stdout, "%s\n", passTimeoutWarnMsg.c_str());
                (void)fflush(stdout);
                COMPILER_LOGW("%s", passTimeoutWarnMsg.c_str());
            }
        }
    } else if (elapsed > timeoutSec_.load() && summaryTimedOutStages_.find(name) == summaryTimedOutStages_.end()) {
        stageTimeoutFlag_[name] = true;
        summaryTimedOutStages_[name] = elapsed;
    }
}

void MonitorManager::EndStageInternal(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                                      const std::chrono::steady_clock::time_point& startTime, int rootFuncIndexOriginal,
                                      int rootFuncOpSize, int functionIndex, const std::string& functionName,
                                      int functionOpSize)
{
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }
    bool stageWarningPrinted = false;
    auto timeoutFlagIt = stageTimeoutFlag_.find(name);
    if (timeoutFlagIt != stageTimeoutFlag_.end()) {
        stageWarningPrinted = timeoutFlagIt->second;
    }
    if (rootFuncIndexOriginal == -1) {
        impl_->StopMonitoring();
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - startTime).count();
    // FuncToBin is a CodeGen sub-stage: covered by CodeGen wall-clock, not listed separately.
    if (name != STAGE_FUNC_TO_BIN) {
        stageElapsedTotals_[name] += elapsed;
    }
    stageDoing_ = false;
    COMPILER_LOGI("Stage ==[%s]== end, sub stage cost %lfs.", name.c_str(),
                  name == STAGE_FUNC_TO_BIN ? elapsed : stageElapsedTotals_[name]);

    double totalElapsed = std::chrono::duration<double>(now - totalStart_).count();
    std::string stageFinishMsg = BuildStageFinishMsg(name, rootFuncIndex, rootFuncName, elapsed, totalElapsed,
                                                     functionIndex, functionName, functionOpSize, rootFuncOpSize);

    if (passDetailEnable_) {
        (void)fprintf(stdout, "%s\n", stageFinishMsg.c_str());
        (void)fflush(stdout);
    }
    COMPILER_LOGI("%s", stageFinishMsg.c_str());
    CheckStageTimeoutOnEnd(name, elapsed, functionOpSize, functionIndex, functionName, stageWarningPrinted);
    stageTimeoutFlag_[name] = false;
}

std::vector<ActiveStageInfo> MonitorManager::GetActiveStages() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return activeStages_;
}

int MonitorManager::GetProcessingThresholdSec() const { return processingThresholdSec_; }

void MonitorManager::SetProcessingThresholdSec(int sec) { processingThresholdSec_ = sec; }

int MonitorManager::GetProgressWidth() const
{
    auto digits = [](int n) { return static_cast<int>(std::to_string(std::max(n, 1)).size()); };
    int maxDigits = std::max({digits(totalFunctionCount_), digits(rootFuncCount_), digits(hostMachineTotalSteps_)});
    return 2 * maxDigits + 1;
}

double MonitorManager::GetTotalElapsed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return lastTotalElapsed_;
}

} // namespace npu::tile_fwk
