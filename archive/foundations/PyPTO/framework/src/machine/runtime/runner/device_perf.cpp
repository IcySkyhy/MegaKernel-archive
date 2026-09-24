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
 * \file device_perf.cpp
 * \brief
 */

#include "machine/runtime/runner/device_perf.h"
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/runtime_api.h"
#include "interface/configs/config_manager.h"
#include "machine/runtime/runner/dump_device_perf.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/device/dynamic/device_common.h"

namespace npu::tile_fwk {
namespace {
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
}

DevicePerf& DevicePerf::GetInstance()
{
    static DevicePerf instance;
    return instance;
}

DevicePerf::DevicePerf() : args_() {}
DevicePerf::~DevicePerf()
{
    MACHINE_LOGD("Start to cleanup perfData");
    StopMachinePerfTraceDumpThread();
    ReleasePerfData();
}

void DevicePerf::InitAndStartDumpThread(const DeviceArgs& args)
{
    args_ = args;
    InitPerfData();
    StartMachinePerfTraceDumpThread();
}

uint32_t DevicePerf::GetPerfDataSize() const { return args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; }

void DevicePerf::InitPerfData()
{
    // init perf data
    for (uint32_t i = 0; i < GetPerfDataSize(); i++) {
        perfData_.push_back(DevMallocWithAlignSize(PERF_DATA_TOTAL_SIZE, TWO_MB_HUGE_PAGE_FLAGS));
    }
}

void DevicePerf::ResetPerData() const
{
    for (uint32_t i = 0; i < GetPerfDataSize(); i++) {
        int rc = RuntimeMemset(perfData_[i], PERF_DATA_TOTAL_SIZE, 0, PERF_DATA_TOTAL_SIZE);
        if (rc != 0) {
            MACHINE_LOGW("CoreId %u, rtMemSet failed, rc: %d", i, rc);
        }
    }
}

void DevicePerf::ReleasePerfData()
{
    for (size_t i = 0; i < perfData_.size(); i++) {
        if (perfData_[i] != nullptr) {
            RuntimeFree(perfData_[i]);
            perfData_[i] = nullptr;
        }
    }
    perfData_.clear();
}

void DevicePerf::ResetMetrics(const uint32_t coreId)
{
    if (perfData_.empty() || perfData_.size() <= static_cast<size_t>(coreId)) {
        return;
    }
    if (args_.aicpuPerfAddr != 0) {
        if (!isPerfDataInited_) {
            RuntimeMemset(perfData_[coreId], sizeof(Metrics), 0, sizeof(Metrics));
            isPerfDataInited_ = true;
        }
    } else {
        RuntimeMemset(perfData_[coreId], sizeof(Metrics), 0, sizeof(Metrics));
    }
}

void DevicePerf::SyncProfData(bool debugEnable)
{
    if (debugEnable) {
        // 多轮控核，nrValidAic和scheCpuNum需实时刷新，否则泳道图会出错
        args_.nrValidAic = GetCfgBlockdim();
        args_.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args_.nrValidAic, args_.nrAicpu, args_.archInfo);
        dynamic::DumpAicoreTaskExectInfo(args_, perfData_);
    }
}

void DevicePerf::SetDebugEnable(const ToSubMachineConfig& profLevelConfig)
{
    for (uint32_t i = 0; i < GetPerfDataSize(); i++) {
        ResetMetrics(i);
        RuntimeMemcpyDirect(
            (reinterpret_cast<uint8_t*>(args_.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) +
                i * SHARED_BUFFER_SIZE,
            sizeof(uint64_t), reinterpret_cast<uint8_t*>(&perfData_[i]), sizeof(uint64_t),
            RtMemcpyKind::HOST_TO_DEVICE);
    }
    if (IsAicoreResolveEnabled()) {
        DevDfxArgs devDfxarg;
        RuntimeMemcpyDirect(&devDfxarg, sizeof(DevDfxArgs), reinterpret_cast<uint8_t*>(args_.devDfxArgAddr),
                            sizeof(DevDfxArgs), RtMemcpyKind::DEVICE_TO_HOST);
        devDfxarg.profLevel = profLevelConfig.profConfig.Contains(ProfConfig::AICORE_TIME) ? 2 : 0;
        RuntimeMemcpyDirect(reinterpret_cast<uint8_t*>(args_.devDfxArgAddr), sizeof(DevDfxArgs), &devDfxarg,
                            sizeof(DevDfxArgs), RtMemcpyKind::HOST_TO_DEVICE);
    }
    MACHINE_LOGD("Set debug enable aicore 0 devPtr: %p", perfData_[0]);
}

bool DevicePerf::RunPrepare() const
{
    bool ret = true;
    if (config::IsRuntimeDebugAllEnabled() || ENABLE_PERF_TRACE) {
        for (uint32_t i = 0; i < GetPerfDataSize(); i++) {
            ret = RuntimeMemcpyDirect(
                      (reinterpret_cast<uint8_t*>(args_.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) +
                          i * SHARED_BUFFER_SIZE,
                      sizeof(uint64_t), reinterpret_cast<const uint8_t*>(&perfData_[i]), sizeof(uint64_t),
                      RtMemcpyKind::HOST_TO_DEVICE) == RT_SUCCESS;
        }
    }
    return ret;
}

void DevicePerf::StartMachinePerfTraceDumpThread()
{
    if (args_.aicpuPerfAddr == 0) {
        return;
    }
    if (dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(false);
    dumpThread_ = std::thread(&DevicePerf::MachinePerfTraceDumpThread, this);
    MACHINE_LOGI("Dump thread started");
}

void DevicePerf::StopMachinePerfTraceDumpThread()
{
    if (!dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(true);
    if (dumpThread_.joinable()) {
        dumpThread_.join();
    }
    MACHINE_LOGD("Dump thread stopped");

    if (args_.aicpuPerfAddr != 0) {
        void* ptr = npu::tile_fwk::dynamic::ValueToPtr(args_.aicpuPerfAddr);
        if (ptr != nullptr) {
            RuntimeFree(ptr);
            args_.aicpuPerfAddr = 0;
        }
    }
}

void DevicePerf::MachinePerfTraceDumpThread()
{
    MACHINE_LOGD("Dump thread start to machine perf trace data");
    int32_t deviceId = static_cast<int32_t>(args_.deviceId);
    if (RuntimeSetDevice(deviceId) != 0) {
        MACHINE_LOGW("Dump perf thread set Device[%d] not success", deviceId);
    }
    while (!dumpThreadStopFlag_.load()) {
        usleep(10000);
        dynamic::DumpDevTaskPerfData(args_, perfData_, false);
    }
    MACHINE_LOGD("Dump thread final dump");
    dynamic::DumpDevTaskPerfData(args_, perfData_, true);
}
} // namespace npu::tile_fwk
