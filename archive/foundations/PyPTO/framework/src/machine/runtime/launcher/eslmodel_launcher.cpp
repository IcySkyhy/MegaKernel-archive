/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/launcher/eslmodel_launcher.h"
#include <thread>
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"
#include "interface/utils/op_info_manager.h"
#include "interface/program/program.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/memory_utils/eslmodel_memory_utils.h"
#include "machine/runtime/runner/runtime_utils.h"

extern "C" int DynTileFwkBackendKernelServer(void* targ);
namespace npu::tile_fwk::dynamic {

int EslModelLauncher::EslModelLaunchAicore(AclRtStream aicoreStream, void* kernel, DeviceKernelArgs* kernelArgs)
{
    RtArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void*> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    RtTaskCfgInfo cfg = {};
    cfg.schemMode = static_cast<uint8_t>(RtSchemModeType::BATCH);
    auto* devProg = (dynamic::DevAscendProgram*)(kernelArgs->cfgdata);
    auto blockDim = devProg->devArgs.nrValidAic;
    return RuntimeKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &cfg);
}

void EslModelLauncher::CopyInputOutputData()
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    for (size_t k = 0; k < inputDataList.size(); k++) {
        auto& inputData = inputDataList[k];
        if (inputData) {
            MemcpyS(inputData->GetDevPtr(), inputData->size(), (uint8_t*)inputData->data(), inputData->size());
        }
    }
    for (size_t k = 0; k < outputDataList.size(); k++) {
        auto& outputData = outputDataList[k];
        if (outputData) {
            MemcpyS(outputData->GetDevPtr(), outputData->size(), (uint8_t*)outputData->data(), outputData->size());
        }
    }
}

int EslModelLauncher::DynamicKernelLaunchEsl(DeviceKernelArgs* kArgs, AclRtStream aicoreStream, void* kernel,
                                             bool isDeviceData)
{
    auto* devProg = (dynamic::DevAscendProgram*)(kArgs->cfgdata);
    devProg->devArgs.nrAic = 32;
    devProg->devArgs.nrAiv = 64;
    EslModelLaunchAicore(aicoreStream, kernel, kArgs);
    if (!isDeviceData) {
        CopyInputOutputData();
    }
    devProg->devArgs.enableEslModel = true;
    size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    auto deviceTaskCtrlPoolAddr = devProg->devArgs.runtimeDataRingBufferAddr + sizeof(RuntimeDataRingBufferHead) +
                                  DEV_ARGS_SIZE;
    (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
    int launchAiCpuNum = static_cast<int>(devProg->devArgs.nrAicpu + dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
    std::vector<std::thread> aicpuThreads(launchAiCpuNum);
    std::atomic<int> idx{0};
    auto threadFun = [&](uint32_t runMode) {
        int tidx = idx++;
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        CPU_SET(tidx, &cpuSet);
        std::string name = "aicput" + std::to_string(tidx);
        pthread_setname_np(pthread_self(), name.c_str());
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);
        DeviceKernelArgs localArgs = *kArgs;
        localArgs.parameter.runMode = runMode;
        (void)DynTileFwkBackendKernelServer(&localArgs);
    };

    aicpuThreads[0] = std::thread(threadFun, RUN_SPLITTED_STREAM_CTRL);
    for (int i = 1; i < launchAiCpuNum; ++i) {
        aicpuThreads[i] = std::thread(threadFun, RUN_SPLITTED_STREAM_SCHE);
    }

    for (int i = 0; i < launchAiCpuNum; ++i) {
        if (aicpuThreads[i].joinable()) {
            aicpuThreads[i].join();
        }
    }
    EslModelMemoryUtils::UnmapAllMappings();
    return 0;
}

int EslModelLauncher::EslModelLaunchDeviceTensorData(Function* function, const std::vector<DeviceTensorData>& inputList,
                                                     const std::vector<DeviceTensorData>& outputList,
                                                     RtStream aicpuStream, RtStream aicoreStream, void* kernel,
                                                     const DeviceLauncherConfig& config, bool isDeviceData)
{
    MACHINE_LOGI("Kernel Launch");
    bool isCapture = false;

    DeviceLauncher::SetCaptureStream(aicoreStream, aicpuStream, isCapture);

    if (isCapture) {
        ExchangeCaptureModeRelax();
    }

    auto rc = AclInit(nullptr);
    if (rc != 0 && rc != ACLRT_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }

    auto dynAttr = function->GetDyndevAttribute();
    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    EslModelMemoryUtils eslMemoryUtil{true, false};
    DeviceLauncher::DeviceInitDistributedContext(eslMemoryUtil, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(eslMemoryUtil, kArgs, dynAttr->devProgBinary, nullptr, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(eslMemoryUtil, kArgs, inputList, outputList, dynAttr->disableL2List);
    if (isCapture) {
        ExchangeCaptureModeGlobal();
    }

    rc = DynamicKernelLaunchEsl(&kArgs, aicoreStream, kernel, isDeviceData);
    if (rc < 0) {
        return rc;
    }
    rc = RuntimeStreamSynchronize(aicoreStream);
    return rc;
}

int EslModelLauncher::EslModelRunOnce(void* kernel, const DeviceLauncherConfig& config, bool isDeviceData,
                                      const std::vector<DeviceTensorData>& tensors)
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    auto aicpuStream = GetStreamContext().GetScheStream();
    auto aicoreStream = GetStreamContext().GetAiCoreStream();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    EslModelMemoryUtils devMemoryHugePage(true);
    EslModelMemoryUtils devMemoryNotHugePage(false);
    Function* function = Program::GetInstance().GetLastFunction();
    int rc = 0;
    if (!isDeviceData) {
        std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(
            devMemoryHugePage, inputDataList, outputDataList);
        rc = EslModelLaunchDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, aicpuStream,
                                            aicoreStream, kernel, config);
        if (HasInplaceArgs(function) || outputDataList.size() == 0) {
            DeviceLauncher::CopyFromDev(devMemoryNotHugePage, inputDataList);
        }
    } else {
        auto currentStream = GetStreamContext().GetCurrentStream();
        if (currentStream != nullptr) {
            RuntimeStreamSynchronize(currentStream);
        }
        for (size_t i = 0; i < tensors.size(); i++) {
            inputDeviceDataList.push_back(tensors[i]);
        }
        rc = EslModelLaunchDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, aicpuStream,
                                            aicoreStream, kernel, config, true);
    }
    return rc;
}

void EslModelLauncher::LiteAllocDeviceMemory(const std::vector<DeviceTensorData>& tensors,
                                             std::vector<uint8_t*>& deviceAddrs, uint8_t*& workspaceAddr,
                                             Function* function)
{
    for (size_t i = 0; i < tensors.size(); i++) {
        uint8_t* deviceAddr = nullptr;
        AclRtMalloc((void**)&deviceAddr, tensors[i].GetDataSize(), AclRtMemMallocPolicy::HUGE_FIRST);
        AclRtMemcpy(deviceAddr, tensors[i].GetDataSize(), (uint8_t*)tensors[i].GetAddr(), tensors[i].GetDataSize(),
                    AclRtMemcpyKind::HOST_TO_DEVICE);
        deviceAddrs.push_back(deviceAddr);
    }

    auto dynAttr = function->GetDyndevAttribute();
    for (auto& devRoot : dynAttr->funcGroup.devRootList) {
        int64_t workspaceSize = dynAttr->rootTileDict[devRoot]->GetStackWorkespaceSize();
        if (workspaceSize > 0) {
            AclRtMalloc((void**)&workspaceAddr, workspaceSize, AclRtMemMallocPolicy::HUGE_FIRST);
            deviceAddrs.push_back(workspaceAddr);
            break;
        }
    }
}

void EslModelLauncher::LiteRegisterKernel(Function* function, void*& hdl, int& stubFunc)
{
    auto dynAttr = function->GetDyndevAttribute();
    std::vector<uint8_t>& kernelBinary = dynAttr->kernelBinary;
    RtDevBinary binary = {
        .magic = RT_DEV_BINARY_MAGIC_ELF,
        .version = 0,
        .data = kernelBinary.data(),
        .length = kernelBinary.size(),
    };
    int ret = RuntimeDevBinaryRegister(&binary, &hdl);
    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, ret == RT_SUCCESS) << "register kernel failed: " << ret;

    stubFunc = 1;
    std::string kernelName = "";
    for (auto& devRoot : dynAttr->funcGroup.devRootList) {
        kernelName = dynAttr->rootTileDict[devRoot]->GetMagicName() + "_main";
    }
    RuntimeFunctionRegister(hdl, &stubFunc, kernelName.c_str(), kernelName.c_str(), 0);
}

int EslModelLauncher::EslModelLiteRunOnce(Function* function, std::vector<DeviceTensorData>& tensors)
{
    ProgramData::GetInstance().Reset();

    // Allocate device memory (I/O + workspace)
    std::vector<uint8_t*> deviceAddrs;
    uint8_t* workspaceAddr = nullptr;
    LiteAllocDeviceMemory(tensors, deviceAddrs, workspaceAddr, function);

    // Init ACL, device and stream
    (void)AclInit(nullptr);
    int32_t deviceId = 0;
    AclRtSetDevice(deviceId);
    AclRtStream stream = nullptr;
    AclRtCreateStream(&stream);

    // Register and launch kernel
    void* hdl = nullptr;
    int stubFunc = 1;
    LiteRegisterKernel(function, hdl, stubFunc);
    RtArgsEx rtArgs = {};
    rtArgs.args = deviceAddrs.data();
    rtArgs.argsSize = deviceAddrs.size() * sizeof(void*);
    int ret = RuntimeKernelLaunch(&stubFunc, 1, rtArgs.args, rtArgs.argsSize, nullptr, stream);
    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, ret == RT_SUCCESS) << "LiteKernelLaunch failed: " << ret;

    // Synchronize and copy back
    ret = AclRtSynchronizeStream(stream);
    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, ret == RT_SUCCESS) << "Stream sync failed: " << ret;
    for (size_t i = 0; i < tensors.size(); i++) {
        AclRtMemcpy((uint8_t*)tensors[i].GetAddr(), tensors[i].GetDataSize(), deviceAddrs[i], tensors[i].GetDataSize(),
                    AclRtMemcpyKind::DEVICE_TO_HOST);
        AclRtFree(deviceAddrs[i]);
    }
    if (workspaceAddr != nullptr) {
        AclRtFree(workspaceAddr);
    }

    AclRtDestroyStream(stream);
    AclRtResetDevice(deviceId);
    AclFinalize();
    return ret;
}
} // namespace npu::tile_fwk::dynamic
