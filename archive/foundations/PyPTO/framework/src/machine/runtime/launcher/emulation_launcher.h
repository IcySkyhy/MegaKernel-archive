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
 * \file device_launcher.h
 * \brief
 */

#ifndef SRC_MACHINE_EMULATION_LAUNCHER_H
#define SRC_MACHINE_EMULATION_LAUNCHER_H

#include <vector>

#include "interface/configs/config_manager.h"
#include "device_launcher.h"
#include "machine/runtime/memory_utils/emulation_memory_utils.h"
#include "machine/runtime/launcher/device_launcher_types.h"

namespace npu::tile_fwk::dynamic {
class EmulationLauncher {
public:
    static int EmulationLaunchOnceWithHostTensorData(Function* function, const std::vector<DeviceTensorData>& inputList,
                                                     const std::vector<DeviceTensorData>& outputList,
                                                     DevControlFlowCache* ctrlCache, EmulationMemoryUtils& memUtils,
                                                     const DeviceLauncherConfig& config = DeviceLauncherConfig());
    static int EmulationLaunchDeviceTensorData(Function* function, const std::vector<DeviceTensorData>& inputList,
                                               const std::vector<DeviceTensorData>& outputList,
                                               const DeviceLauncherConfig& config = DeviceLauncherConfig(),
                                               DevControlFlowCache* ctrlCache = nullptr);
    static int EmulationRunOnce(Function* function, DevControlFlowCache* ctrlCache,
                                const DeviceLauncherConfig& config = DeviceLauncherConfig());
    static int EmulationBuildControlFlowCache(DeviceKernelArgs& kArgs);
    static DevControlFlowCache* CreateHostCtrlFlowCache(DevAscendProgram* devProg, Function* function,
                                                        EmulationMemoryUtils& memUtils);
    static int BuildControlFlowCacheWithEmulationTensorData(
        Function* function, const std::vector<DeviceTensorData>& inputList,
        const std::vector<DeviceTensorData>& outputList, CachedOperator* cachedOperator,
        DevControlFlowCache** outCtrlFlowCache, EmulationMemoryUtils& memUtils,
        const DeviceLauncherConfig& config = DeviceLauncherConfig());

#define CONTROL_FLOW_CACHE_BASE_ADDR 0x100000000
#define CONTROL_FLOW_CACHE_TENSOR_SIZE 0x100000000
    static int BuildControlFlowCache(Function* function, EmulationMemoryUtils& memUtils,
                                     const std::vector<DeviceTensorData>& inputList = {},
                                     const std::vector<DeviceTensorData>& outputList = {},
                                     DevControlFlowCache** outCtrlFlowCache = nullptr,
                                     const DeviceLauncherConfig& config = DeviceLauncherConfig());
    static int BuildControlFlowCache(Function* function, DevControlFlowCache** outCtrlFlowCache,
                                     EmulationMemoryUtils& memUtils,
                                     const DeviceLauncherConfig& config = DeviceLauncherConfig());
};
} // namespace npu::tile_fwk::dynamic
#endif // SRC_MACHINE_EMULATION_LAUNCHER_H
