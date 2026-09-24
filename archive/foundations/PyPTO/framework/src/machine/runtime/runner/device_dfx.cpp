/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_dfx.cpp
 * \brief Device DFX args initialization: aicpu perf addr, log level, device id, perf trace flag.
 */

#include "machine/runtime/runner/device_dfx.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "interface/utils/common.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/device/dynamic/device_common.h"
#include "machine/device/tilefwk/aicore_print_base.h"

extern "C" {
__attribute__((weak)) int dlog_getlevel(int32_t moduled, int32_t* enableEvent);
__attribute__((weak)) int drvDeviceGetPhyIdByIndex(uint32_t logicDevId, uint32_t* phyDevId);
}

namespace npu::tile_fwk {

void DeviceDfx::InitAicpuPerfAddr(DeviceArgs& args)
{
    if (GetEnvVar("DUMP_DEVICE_PERF") == "true") {
        auto aicpuDevPtr = DevMallocWithAlignSize(MAX_ROUND_NUM * sizeof(MetricPerf), TWO_MB_HUGE_PAGE_FLAGS);
        if (aicpuDevPtr == nullptr) {
            MACHINE_LOGW("Aicpu per addr malloc failed");
        } else {
            args.aicpuPerfAddr = npu::tile_fwk::dynamic::PtrToValue(aicpuDevPtr);
        }
    }
}

void DeviceDfx::InitDevDfxArgs(const bool isPerfTrace, DevDfxArgs& devDfxArg)
{
    int logLevel = -1;
    if (dlog_getlevel != nullptr) {
        int32_t enableLog = -1;
        logLevel = dlog_getlevel(PYPTO, &enableLog);
    }
    devDfxArg.logLevel = logLevel;
    uint32_t logicalDevId = GetLogDeviceId();
    uint32_t phyDevId = 0;
    if (drvDeviceGetPhyIdByIndex != nullptr) {
        drvDeviceGetPhyIdByIndex(logicalDevId, &phyDevId);
    } else {
        MACHINE_LOGW("Get device Local deviceId failed");
    }
    MACHINE_LOGI("Current device info: logical devId: %u, phyDevId: %u", logicalDevId, phyDevId);
    devDfxArg.deviceId = phyDevId;
    if (isPerfTrace) {
        devDfxArg.isOpenPerfTrace = 1;
    }
    MACHINE_LOGI("Dfx info: log level is: %d, openPerTrace: %d, deviceId: %u", logLevel, devDfxArg.isOpenPerfTrace,
                 devDfxArg.deviceId);
}

bool DeviceDfx::Init(DeviceArgs& args)
{
    InitAicpuPerfAddr(args);

    DevDfxArgs dfxArgs;
    InitDevDfxArgs(args.aicpuPerfAddr != 0, dfxArgs);
    args.devDfxArgAddr = reinterpret_cast<uint64_t>(CopyDataToDevice(&dfxArgs, sizeof(DevDfxArgs)));
    if (args.devDfxArgAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Fail to copy dfx info from host to device.");
        return false;
    }
    if (InitAicorePrint(args) != 0) {
        MACHINE_LOGW("Failed to init aicore print host manager");
        return false;
    }
    return true;
}

int DeviceDfx::InitAicorePrint(const DeviceArgs& args) { return aicorePrintMgr_.Init(args); }

int DeviceDfx::DumpAicoreLog() { return aicorePrintMgr_.DumpAicoreLog(); }

DeviceDfx& DeviceDfx::GetInstance()
{
    static DeviceDfx devDfx;
    return devDfx;
}
} // namespace npu::tile_fwk
