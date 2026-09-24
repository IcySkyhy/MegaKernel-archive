/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file eslmodel_launcher.h
 * \brief
 */

#pragma once

#include <vector>
#include <sys/mman.h>
#include "adapter/api/acl_define.h"
#include "machine/runtime/launcher/device_launcher_types.h"

namespace npu::tile_fwk::dynamic {

class EslModelLauncher {
public:
    static int EslModelRunOnce(void* kernel, const DeviceLauncherConfig& config = DeviceLauncherConfig(),
                               bool isDeviceData = false, const std::vector<DeviceTensorData>& tensors = {});
    static int EslModelLaunchDeviceTensorData(Function* function, const std::vector<DeviceTensorData>& inputList,
                                              const std::vector<DeviceTensorData>& outputList, RtStream aicpuStream,
                                              RtStream aicoreStream, void* kernel, const DeviceLauncherConfig& config,
                                              bool isDeviceData = false);
    static int DynamicKernelLaunchEsl(DeviceKernelArgs* kArgs, AclRtStream aicoreStream, void* kernel,
                                      bool isDeviceData = false);
    static int EslModelLaunchAicore(AclRtStream aicoreStream, void* kernel, DeviceKernelArgs* kernelArgs);
    static void CopyInputOutputData();
    static int EslModelLiteRunOnce(Function* function, std::vector<DeviceTensorData>& tensors);

private:
    static void LiteAllocDeviceMemory(const std::vector<DeviceTensorData>& tensors, std::vector<uint8_t*>& deviceAddrs,
                                      uint8_t*& workspaceAddr, Function* function);
    static void LiteRegisterKernel(Function* function, void*& hdl, int& stubFunc);
};
} // namespace npu::tile_fwk::dynamic
