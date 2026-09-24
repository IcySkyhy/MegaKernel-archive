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
 * \file device_dfx.h
 * \brief Device DFX args initialization: aicpu perf addr, log level, device id, perf trace flag.
 */

#pragma once

#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "machine/runtime/runner/aicore_print_host_manager.h"

namespace npu::tile_fwk {
class DeviceDfx {
public:
    static DeviceDfx& GetInstance();
    bool Init(DeviceArgs& args);
    int DumpAicoreLog();

private:
    void InitAicpuPerfAddr(DeviceArgs& args);
    void InitDevDfxArgs(bool isPerfTrace, DevDfxArgs& devDfxArg);
    int InitAicorePrint(const DeviceArgs& args);
    AicorePrintHostManager aicorePrintMgr_;
};
} // namespace npu::tile_fwk
