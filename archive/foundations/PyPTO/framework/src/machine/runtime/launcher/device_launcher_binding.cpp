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
 * \file device_launcher_binding.cpp
 * \brief
 */

#include "machine/runtime/launcher/device_launcher_binding.h"
#include "adapter/api/runtime_define.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/context/device_launcher_context.h"

namespace npu::tile_fwk::dynamic {
int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(ExportedOperator* op,
                                                         const std::vector<DeviceTensorData>& inputList,
                                                         const std::vector<DeviceTensorData>& outputList,
                                                         DeviceStream aicoreStream, bool streamSynchronize,
                                                         uint8_t* devCtrlCache, const DeviceLauncherConfig& config)
{
    RtStream aicoreStreamValue = reinterpret_cast<RtStream>(aicoreStream);
    return DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
        op->GetFunction(), inputList, outputList, aicoreStreamValue, streamSynchronize, op,
        reinterpret_cast<DevControlFlowCache*>(devCtrlCache), config);
}

int DeviceSynchronize(DeviceStream aicpuStream)
{
    RtStream aicpuStreamValue = reinterpret_cast<RtStream>(aicpuStream);
    return DeviceLauncher::DeviceSynchronize(aicpuStreamValue, GetContextAiCoreStream());
}

int DeviceRunOnce(Function* function, uint8_t* hostCtrlCache, const DeviceLauncherConfig& config)
{
    return DeviceLauncher::DeviceRunOnce(function, reinterpret_cast<DevControlFlowCache*>(hostCtrlCache), config);
}

int HasInplaceArgs(Function* function) { return DeviceLauncher::HasInplaceArgs(function); }

void DeviceLauncherInit() { DeviceLauncherContext::Get().Initialize(); }

void DeviceLauncherFini() { DeviceLauncherContext::Get().Finalize(); }

void CopyDevToHost(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor)
{
    DeviceMemoryUtils().CopyFromDev((uint8_t*)hostTensor.GetAddr(), (uint8_t*)devTensor.GetAddr(),
                                    devTensor.GetDataSize());
}

void CopyHostToDev(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor)
{
    DeviceMemoryUtils().CopyToDev((uint8_t*)devTensor.GetAddr(), (uint8_t*)hostTensor.GetAddr(),
                                  devTensor.GetDataSize());
}

uint8_t* CopyHostToDev(uint8_t* data, uint64_t size)
{
    return DeviceMemoryUtils(false).CopyToDev((uint8_t*)data, size, nullptr);
}

void ChangeCaptureModeRelax() { ExchangeCaptureModeRelax(); }

void ChangeCaptureModeGlobal() { ExchangeCaptureModeGlobal(); }

void GetCaptureInfo(RtStream aicoreStream, AclMdlRI& rtModel, bool& isCapture)
{
    (void)GetStreamCaptureInfo(aicoreStream, rtModel, isCapture);
}

void* RegisterKernelBinary(const std::vector<uint8_t>& kernelBinary) { return RegisterKernelBin(kernelBinary); }

void UnregisterKernelBinary(void* hdl)
{
    int ret = RuntimeDevBinaryUnRegister(hdl);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_REGISTER_FAILED, "unregister kernel failed, ret: %d", ret);
    }
}

static std::unordered_map<ExportedOperator*, std::shared_ptr<ExportedOperator>> exportedOperatorDict;

ExportedOperator* ExportedOperatorBegin()
{
    std::shared_ptr<ExportedOperator> op = std::make_shared<ExportedOperator>();
    exportedOperatorDict[op.get()] = op;
    return op.get();
}

void ExportedOperatorEnd(ExportedOperator* op) { op->ResetFunction(Program::GetInstance().GetLastFunction()); }
} // namespace npu::tile_fwk::dynamic
