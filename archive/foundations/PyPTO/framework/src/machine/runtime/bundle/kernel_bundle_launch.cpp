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
 * \file kernel_bundle_launch.cpp
 * \brief Self-contained offline bundle launch (see header). Kept out of device_launcher.cpp on purpose.
 */

#include "machine/runtime/bundle/kernel_bundle_launch.h"

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "securec.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/msprof_api.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/bundle/kernel_bundle_dev_cache.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/context/device_launcher_context.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"

namespace npu::tile_fwk::dynamic {

namespace {
// aicpu launch primitive (topology from devProg), kept here so device_launcher stays free of bundle code.
int BundleLaunchAicpu(RtAicpuArgsEx& rtArgs, DevAscendProgram* devProg)
{
    auto ctrlStream = GetStreamContext().GetCtrlStream();
    auto schedStream = GetStreamContext().GetScheStream();
    int ret = 0;
    auto args = (AiCpuArgs*)rtArgs.args;
    const int nrAicpu = static_cast<int>(devProg->devArgs.nrAicpu);
    const bool launchSchedSameCluster = static_cast<int>(devProg->devArgs.launchSchedSameCluster);
    if (launchSchedSameCluster) {
        MACHINE_LOGW("When available AICPUs are insufficient, execute export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false"
                     "to disable the constraint that forces scheduling threads onto the same cluster.");
    }
    args->kArgs.parameter.ctrlBlockNum = static_cast<int>(devProg->ctrlBlockDim);
    auto startTime = MspfSysCycleTime();
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    ret = RuntimeAicpuKernelLaunchExWithArgs(static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC),
                                             "AST_DYN_AICPU", 1, &rtArgs, nullptr, ctrlStream,
                                             RT_KERNEL_USE_SPECIAL_TIMEOUT);
    HostProf::GetInstance().ReportHostProfInfo(ctrlStream, startTime, 1, MSPF_GE_TASK_TYPE_AI_CPU, false);
    if (ret != RT_SUCCESS) {
        return ret;
    }
    if (devProg->devArgs.enableAicoreResolve) {
        return ret;
    }
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
    startTime = MspfSysCycleTime();
    const int scheCpuNum = static_cast<int>(devProg->devArgs.scheCpuNum);
    ret = RuntimeAicpuKernelLaunchExWithArgs(static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC),
                                             "AST_DYN_AICPU", nrAicpu, &rtArgs, nullptr, schedStream,
                                             RT_KERNEL_USE_SPECIAL_TIMEOUT);
    HostProf::GetInstance().ReportHostProfInfo(schedStream, startTime, scheCpuNum, MSPF_GE_TASK_TYPE_AI_CPU, false);
    return ret;
}

// aicore launch primitive: bundle always runs debugEnable=false, so the debug-sync branch is omitted.
int BundleLaunchAicore(AclRtStream aicoreStream, void* kernel, RtArgsEx& rtArgs, RtTaskCfgInfo& rtTaskCfg,
                       DevAscendProgram* devProg)
{
    auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    int blockDim = static_cast<int>(devProg->ctrlBlockDim);
    if (blockDim == 0) {
        blockDim = static_cast<int>(devProg->devArgs.nrValidAic);
    }
    auto startTime = MspfSysCycleTime();
    auto ret = RuntimeKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &rtTaskCfg);
    HostProf::GetInstance().ReportHostProfInfo(aicoreStream, startTime, blockDim, MSPF_GE_TASK_TYPE_MIX_AIC, true);
    return ret;
}
} // namespace

int LaunchBundleKernelOnce(const std::vector<uint8_t>& devProgBinary, void* binHandle, uint64_t cacheKey,
                           const std::vector<DeviceTensorData>& tensorList,
                           const std::vector<uint8_t>& hostCtrlFlowCache,
                           const std::vector<DevDynamicCellMatchStridePatch>& cellMatchStridePatches,
                           void* workspaceAddr, RtStream aicoreStream, bool streamSynchronize,
                           const DeviceLauncherConfig& config)
{
    (void)config;
    MACHINE_LOGI("Kernel Launch (bundle)");
    if (binHandle == nullptr) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "[kernel-bundle] null kernel bin handle.");
        return -1;
    }
    aicoreStream = aicoreStream == nullptr ? GetContextAiCoreStream() : aicoreStream;

    int rc = AclInit(nullptr);
    if (rc != 0 && rc != ACLRT_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }
    DeviceLauncher::CheckAscendDriverVersionOnboard();
    CheckDeviceId();

    auto* devProg = reinterpret_cast<DevAscendProgram*>(const_cast<uint8_t*>(devProgBinary.data()));

    // Control-flow cache: copied host->device once per bundle, reused across launches (freed at process exit).
    uint8_t* devCtrlCache = bundle::KernelBundleDevCache::Instance().GetOrCopy(cacheKey, hostCtrlFlowCache);

    // Build AiCpuArgs buffer: one unified operand list ([nTensors, 0], outputs folded in), then the trailing
    // cell-match stride patch count + patches (device applies via ApplyDynamicCellMatchDescPatchesFromLaunchArgs).
    const uint64_t tensorCount = tensorList.size();
    const size_t argSize = sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + tensorCount * sizeof(DevTensorData) +
                           sizeof(uint64_t) + cellMatchStridePatches.size() * sizeof(DevDynamicCellMatchStridePatch);
    std::vector<int64_t> aicpuArgBuf((argSize + sizeof(int64_t) - 1) / sizeof(int64_t), 0);
    auto* aicpuArgs = new (aicpuArgBuf.data()) AiCpuArgs();
    aicpuArgs->kArgs.inputs = nullptr;
    aicpuArgs->kArgs.outputs = nullptr;

    // Fill device-side kArgs: cfgdata (base-0 devProg), metadata, distributed ctx. Topology is re-derived from
    // the target platform in DeviceInitTilingData.
    DeviceMemoryUtils devMem;
    DeviceLauncher::FillDeviceKernelArgs(devMem, const_cast<std::vector<uint8_t>&>(devProgBinary), aicpuArgs->kArgs,
                                         std::vector<std::string>{});

    // Fill tensor descriptors: [nTensors, 0] then the flat tensor array (kernel addresses operands by position).
    int64_t* inputp = reinterpret_cast<int64_t*>(aicpuArgs + 1);
    inputp[0] = static_cast<int64_t>(tensorCount);
    inputp[1] = 0; // nOut: outputs are folded into tensorList
    auto* tensorData = reinterpret_cast<DevTensorData*>(inputp + 2);
    for (const auto& t : tensorList) {
        tensorData->address = reinterpret_cast<uint64_t>(t.GetAddr());
        tensorData->dataType = t.GetDataType();
        const auto& shape = t.GetShape();
        tensorData->shape.dimSize = static_cast<int>(shape.size());
        for (int j = 0; j < tensorData->shape.dimSize; ++j) {
            tensorData->shape.dim[j] = shape[j];
        }
        tensorData++;
    }
    // Trailing cell-match stride patch count (0 when empty) + patch array, after the flat tensor list.
    WriteDynamicCellMatchStridePatchesToLaunchArgs(inputp, cellMatchStridePatches);

    // Per-launch kArgs (ctrl cache device ptr / workspace / monotonic round / dynamic mem budgets).
    static std::atomic<int64_t> bundleSequence{0};
    aicpuArgs->kArgs.ctrlFlowCache = reinterpret_cast<int64_t*>(devCtrlCache);
    aicpuArgs->kArgs.workspace = reinterpret_cast<int64_t*>(workspaceAddr);
    aicpuArgs->kArgs.parameter.globalRound = ++bundleSequence;
    aicpuArgs->kArgs.maxDynamicAssembleOutcastMem = devProg->memBudget.tensor.maxDynamicAssembleOutcastMem;
    aicpuArgs->kArgs.maxDynamicCellMatchTableMem = devProg->memBudget.metadata.maxDynamicCellMatchTableMem;

    // Dynamic cell-match metadata pool: a standalone device buffer (size = memBudget.metadata.dynamicCellMatch,
    // computed in EvalWorkspaceForShapes), allocated once per bundle and reused. The device (device_ctrl InitDyn)
    // overwrites devArgs.dynamicCellMatch{Addr,Capacity} from these kArgs and inits the pool in place.
    const uint64_t dynamicCellMatchBytes = devProg->memBudget.metadata.dynamicCellMatch;
    uint8_t* cellMatchAddr = bundle::KernelBundleDevCache::Instance().GetOrAllocCellMatch(cacheKey,
                                                                                          dynamicCellMatchBytes);
    aicpuArgs->kArgs.runtimeDynamicCellMatchAddr = reinterpret_cast<uint64_t>(cellMatchAddr);
    aicpuArgs->kArgs.runtimeDynamicCellMatchCapacity = cellMatchAddr != nullptr ? dynamicCellMatchBytes : 0;

    // rt args.
    RtAicpuArgsEx rtAicpuArgs;
    (void)memset_s(&rtAicpuArgs, sizeof(RtAicpuArgsEx), 0, sizeof(RtAicpuArgsEx));
    rtAicpuArgs.kernelNameAddrOffset = offsetof(AiCpuArgs, kernelName);
    rtAicpuArgs.soNameAddrOffset = offsetof(AiCpuArgs, soName);
    rtAicpuArgs.hostInputInfoNum = 1;
    RtHostInputInfo hostInfo;
    hostInfo.addrOffset = offsetof(AiCpuArgs, kArgs.inputs);
    hostInfo.dataOffset = sizeof(AiCpuArgs);
    rtAicpuArgs.hostInputInfoPtr = &hostInfo;
    rtAicpuArgs.timeout = AICPU_EXECUTE_TIMEOUT;
    rtAicpuArgs.args = aicpuArgs;
    rtAicpuArgs.argsSize = static_cast<uint32_t>(aicpuArgBuf.size() * sizeof(int64_t));

    std::vector<void*> kernelArgs(0x7, nullptr);
    RtArgsEx rtAicoreArgs;
    (void)memset_s(&rtAicoreArgs, sizeof(RtArgsEx), 0, sizeof(RtArgsEx));
    rtAicoreArgs.args = kernelArgs.data();
    rtAicoreArgs.argsSize = static_cast<uint32_t>(kernelArgs.size() * sizeof(void*));

    RtTaskCfgInfo rtTaskCfg;
    (void)memset_s(&rtTaskCfg, sizeof(RtTaskCfgInfo), 0, sizeof(RtTaskCfgInfo));
    rtTaskCfg.schemMode = static_cast<uint8_t>(RtSchemModeType::BATCH);

    // Launch (mirrors KernelModule::Launch), reusing the public DeviceLauncher helpers.
    const bool isCaptureMode = DeviceLauncher::IsCaptureMode();
    rc = DeviceLauncher::LaunchSyncTask(aicoreStream, isCaptureMode, /*launchEarlyMode=*/0);
    if (rc != RT_SUCCESS) {
        return rc;
    }
    DeviceLauncher::SetDevPerfAddr(false, isCaptureMode, devProg->devArgs.toSubMachineConfig);

    rc = BundleLaunchAicpu(rtAicpuArgs, devProg);
    if (rc != RT_SUCCESS) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_BUILTIN_OP_NULL_FAILED, "[kernel-bundle] launch aicpu failed: %d", rc);
        return rc;
    }

    static const char kBundleKernelName[] = "PyPTO_bundle_kernel";
    kernelArgs[0] = const_cast<char*>(kBundleKernelName);
    kernelArgs[4] = reinterpret_cast<int64_t*>(aicpuArgs + 1); // inputp
    kernelArgs[5] = aicpuArgs->kArgs.cfgdata;                  // 5 is cfgdata
    kernelArgs[6] = reinterpret_cast<DevTensorData*>(reinterpret_cast<int64_t*>(aicpuArgs + 1) + 2);

    rc = BundleLaunchAicore(aicoreStream, static_cast<RtBinHandle>(binHandle), rtAicoreArgs, rtTaskCfg, devProg);
    if (rc != RT_SUCCESS) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "[kernel-bundle] launch aicore failed: %d", rc);
        return rc;
    }

    if (streamSynchronize) {
        rc = DeviceLauncher::DynamicLaunchSynchronize(GetContextScheStream(), GetContextCtrlStream(), aicoreStream);
    }
    MACHINE_LOGI("finish Kernel Launch (bundle).");
    return rc;
}

} // namespace npu::tile_fwk::dynamic
