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
 * \file pypto_aicpu_interface.cpp
 * \brief
 */

#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include "pypto_aicpu_interface.h"
#include "machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/machine_ws_intf.h"

namespace {
npu::tile_fwk::BackendServerHandleManager g_handleManager;
}
using namespace npu::tile_fwk;
extern "C" {
__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerNull(void* args)
{
#ifdef __DEVICE__
    InitLogSwitch();
#endif
    if (args == nullptr) {
        DEV_ERROR(DevCommonErr::NULLPTR, "#sche.task.pre.dyn.server: Server init input args is null");
        return 1;
    }
    auto kargs = (DeviceKernelArgs*)args;
    if (kargs == nullptr) {
        DEV_ERROR(DevCommonErr::NULLPTR, "#sche.task.pre.dyn.server: Server init DeviceKernelArgs is null");
        return 1;
    }
    auto devArgs = reinterpret_cast<DeviceArgs*>(kargs->cfgdata);
    auto data = reinterpret_cast<char*>(devArgs->aicpuSoBin);
    if (!g_handleManager.SaveSoFile(data, devArgs->aicpuSoLen, devArgs->deviceId)) {
        DEV_ERROR(DevCommonErr::FILE_ERROR, "#sche.task.pre.dyn.server: create so failed");
        return 1;
    }
    g_handleManager.SetTileFwkKernelMap();
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServer(void* args)
{
    auto ret = g_handleManager.ExecuteFunc(args, dyExecFuncKey);
    if (ret != 0) {
        DEV_ERROR(ServerKernelErr::KERNEL_EXEC_FAILED,
                  "#sche.task.run.dyn.server: TileFwk kernelFunc [%s] exec not Success", dynServerKernelFun.c_str());
        return 1;
    }
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerInit(void* args)
{
    auto ret = g_handleManager.ExecuteFunc(args, dyInitFuncKey);
    if (ret != 0) {
        DEV_ERROR(ServerKernelErr::KERNEL_EXEC_FAILED,
                  "#sche.task.pre.dyn.server.init: TileFwk kernelFunc [%s] exec not Success",
                  dynServerKernelInitFun.c_str());
        return 1;
    }
    return 0;
}
}
