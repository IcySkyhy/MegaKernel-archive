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
 * \file device_perf.h
 * \brief
 */

#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include "interface/machine/device/tilefwk/aicpu_common.h"

namespace npu::tile_fwk {
class DevicePerf {
public:
    static DevicePerf& GetInstance();
    DevicePerf();
    ~DevicePerf();
    void InitAndStartDumpThread(const DeviceArgs& args);
    bool RunPrepare() const;
    void ResetPerData() const;
    void SyncProfData(bool debugEnable);
    void SetDebugEnable(const ToSubMachineConfig& profLevelConfig);

private:
    uint32_t GetPerfDataSize() const;
    void InitPerfData();
    void ReleasePerfData();
    void ResetMetrics(const uint32_t coreId);
    void StartMachinePerfTraceDumpThread();
    void StopMachinePerfTraceDumpThread();
    void MachinePerfTraceDumpThread();

    DeviceArgs args_;
    bool isPerfDataInited_{false};
    std::vector<void*> perfData_;
    std::thread dumpThread_;
    std::atomic<bool> dumpThreadStopFlag_{false};
};
} // namespace npu::tile_fwk
