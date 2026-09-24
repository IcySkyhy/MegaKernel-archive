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
 * \file device_launcher.h
 * \brief
 */

#ifndef SRC_MACHINE_DEVICE_LAUNCHER_H
#define SRC_MACHINE_DEVICE_LAUNCHER_H

#include <cinttypes>
#include <atomic>
#include <unordered_map>
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_tensor_creator.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "machine/device/dynamic/device_common.h"
#include "machine/runtime/runner/runtime_agent.h"
#include "machine/runtime/memory_utils/device_memory_utils.h"
#include "machine/runtime/distributed/distributed_context.h"
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/device_perf.h"
#include "machine/runtime/runner/load_aicpu_op.h"
#include "machine/runtime/launcher/device_launcher_types.h"
#include "machine/runtime/launcher/launcher_router.h"

namespace npu::tile_fwk::dynamic {

class KernelBinary;

// 基于 DAV3510 架构硬件约束：aiCpuNum 与 die0MaxCpuid 的固定映射关系
constexpr uint32_t DAV3510_AICPU_NUM_6 = 6;      // DAV3510 6 个 AICPU 配置
constexpr uint32_t DAV3510_AICPU_NUM_7 = 7;      // DAV3510 7 个 AICPU 配置
constexpr uint32_t DAV3510_DIE0_MAX_CPUID_4 = 4; // 对应 aiCpuNum=6 的 die0 最大 CPU ID
constexpr uint32_t DAV3510_DIE0_MAX_CPUID_5 = 5; // 对应 aiCpuNum=7 的 die0 最大 CPU ID

inline void ValidateRuntimeDevice(int32_t targetDevId)
{
    int32_t currentDevId = 0;
    FE_ASSERT(FeError::INVALID_VAL, RuntimeGetDevice(&currentDevId) == RT_SUCCESS)
        << "Failed to get current device ID.";
    FE_ASSERT(FeError::INVALID_VAL, targetDevId == currentDevId)
        << "Device mismatch: torch.npu.set_device(device_id) must use the same device ID as the input tensors. "
        << "Current device ID: " << currentDevId << ", input tensor device ID: " << targetDevId << ".";
}

class AclModeGuard {
public:
    AclModeGuard(AclMdlRICaptureMode tmode) : mode(tmode) { AclMdlRICaptureThreadExchangeMode(&mode); }
    ~AclModeGuard() { AclMdlRICaptureThreadExchangeMode(&mode); }

private:
    AclMdlRICaptureMode mode;
};

class DeviceLauncher {
public:
    static constexpr uint32_t kDefaultAicNum = 25;
    static constexpr uint32_t kDefaultAivNum = 50;
    static constexpr uint32_t kDefaultTensorinfoSize = 16384;
    static DevAscendProgram* GetDevProg(Function* func)
    {
        return reinterpret_cast<DevAscendProgram*>(func->GetDyndevAttribute()->devProgBinary.data());
    }

    static bool HasInplaceArgs(Function* function) { return GetDevProg(function)->outputInplaceSlotList.size() != 0; }

    /** Query Ascend driver package version once per process (skipped on ArchInfo::DAV_3510). Abort if official x.y.z
     *  format and below 25.5; ACL failure, empty string, or non-official format skips gate with MACHINE_LOGW. */
    static void CheckAscendDriverVersionOnboard();

    static void InitDevArgs(DeviceArgs& devArgs);

    static void DeviceLauncherConfigFillDeviceInfo(const DeviceLauncherConfig& config)
    {
        DeviceLauncherConfig& devConfig = const_cast<DeviceLauncherConfig&>(config);
        int maxBlockDim = GetCfgBlockdim();
        int maxAicpuNum = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum());
        maxAicpuNum = maxAicpuNum > 0 ? maxAicpuNum : 5;

        if (devConfig.blockdim == 0 || devConfig.blockdim > maxBlockDim) {
            devConfig.blockdim = maxBlockDim;
        }

        if (devConfig.aicpuNum == 0 || devConfig.aicpuNum > maxAicpuNum) {
            devConfig.aicpuNum = maxAicpuNum;
        }
    }

    static void DeviceInitLauncherConfigForUser(std::vector<uint8_t>& devProgData)
    {
        auto* devProg = reinterpret_cast<DevAscendProgram*>(const_cast<uint8_t*>(devProgData.data()));
        devProg->devArgs.launchSchedAicpuNum = config::GetRuntimeOption<uint32_t>(LAUNCH_SCHED_AICPU_NUM);
        devProg->devArgs.launchSchedSameCluster = GetEnvVar("PYPTO_LAUNCH_SCHED_SAME_CLUSTER") != "false";
    }

    template <typename DeviceMemoryTy>
    static void AssignMetaAddr(DeviceMemoryTy& devMem, DeviceKernelArgs& kArgs, DevAscendProgram* devProg,
                               CachedOperator* cachedOperator)
    {
        (void)kArgs;

        FillDeviceRuntimeOffset(devProg, DEFAULT_RUNTIME_DATA_RING_BUFFER_COUNT);
        size_t runtimeDataSize = devProg->GetDeviceRuntimeOffset().size;
        size_t runtimeDataCount = devProg->GetDeviceRuntimeOffset().count;
        size_t runtimeDataRingBufferSize = RuntimeDataRingBufferHead::GetRingBufferSize(runtimeDataSize,
                                                                                        runtimeDataCount);
        if (cachedOperator && *CachedOperator::GetMetaDataDevAddrHolder(cachedOperator) != nullptr) {
            devProg->devArgs.runtimeDataRingBufferAddr = reinterpret_cast<uint64_t>(
                *CachedOperator::GetMetaDataDevAddrHolder(cachedOperator));
        } else {
            devProg->devArgs.runtimeDataRingBufferAddr = (uint64_t)devMem.AllocZero(runtimeDataRingBufferSize, nullptr);
        }

        uint64_t generalSize = devProg->memBudget.metadata.general;
        uint64_t stitchPoolSize = devProg->memBudget.metadata.stitchPool;
        MACHINE_LOGD("generalSize=%lu, stitchPoolSize=%lu, generalOffset=%#lx, stitchPoolOffset=%#lx.", generalSize,
                     stitchPoolSize, devProg->deviceRuntimeOffset.generalOffset,
                     devProg->deviceRuntimeOffset.stitchPoolOffset);
        return;
    }

    static uint32_t GetAiCpuNum(uint32_t aiCpuNum, uint32_t scheCpuNum, ArchInfo archInfo,
                                [[maybe_unused]] bool isSameCluster)
    {
        if (scheCpuNum == 1) {
            return scheCpuNum;
        }

        if (archInfo == ArchInfo::DAV_3510) {
            uint32_t oneDieMinCpuNum = aiCpuNum >> 1;
            uint32_t oneDieMaxCpuNum = oneDieMinCpuNum + (aiCpuNum - (oneDieMinCpuNum << 1));
            uint32_t oneDieMinScheCpuNum = scheCpuNum >> 1;
            uint32_t launchCpuNum = oneDieMaxCpuNum + oneDieMinScheCpuNum;
            return launchCpuNum < aiCpuNum ? launchCpuNum : scheCpuNum;
        } else {
            if (!isSameCluster) {
                MACHINE_LOGD("Set aicpu to not enforce the same cluster.");
                return scheCpuNum;
            }
            // sche = 2, need launch 3 aicpu ensure cluster; sche = 3, need launch 5 aicpu
            uint32_t launchCpuNum = 2 * scheCpuNum - 1; // 2 : ensure cluster success
                                                        // when launchCpuNum >= aiCpuNum, can't use same cluster
            return launchCpuNum < aiCpuNum ? launchCpuNum : scheCpuNum;
        }
    }

    static uint32_t GetDav3510DieMaxCpuid(uint32_t aiCpuNum)
    {
        // DAV3510 架构约束：aiCpuNum 与 die0MaxCpuid 的固定映射关系
        if (aiCpuNum == DAV3510_AICPU_NUM_6) {
            return DAV3510_DIE0_MAX_CPUID_4;
        } else if (aiCpuNum == DAV3510_AICPU_NUM_7) {
            return DAV3510_DIE0_MAX_CPUID_5;
        }

        // 默认返回 0：不启用 die 分离机制
        return 0;
    }

    static void PrepareDevProgArgsCpuInfo(DevAscendProgram* devProg, DeviceLauncherConfig& config)
    {
        uint32_t aiCpuNum = static_cast<uint32_t>(Platform::Instance().GetSoc().GetAICPUNum());
        // Read user configuration for launch_sched_aicpu_num
        uint32_t launchSchedAicpuNum = devProg->devArgs.launchSchedAicpuNum;
        if (launchSchedAicpuNum > 0) {
            // user configuration provided: use user config if within valid range
            if (launchSchedAicpuNum > aiCpuNum - dynamic::MAX_CONTROL_FLOW_AICPU_NUM) {
                MACHINE_LOGW(
                    "User configured launch_sched_aicpu_num=%u exceeds hardware max=%u, using max value instead.",
                    launchSchedAicpuNum, aiCpuNum);
            } else {
                aiCpuNum = launchSchedAicpuNum + dynamic::MAX_CONTROL_FLOW_AICPU_NUM; // use user configuration
                MACHINE_LOGD("Using user configured launch_sched_aicpu_num=%u.", launchSchedAicpuNum);
            }
        }

        if (devProg->devArgs.archInfo == ArchInfo::DAV_3510) {
            devProg->devArgs.die0MaxCpuid = GetDav3510DieMaxCpuid(aiCpuNum);
        }

        devProg->devArgs.nrValidAic = config.blockdim;
        devProg->devArgs.scheCpuNum = CalcSchAicpuNumByBlockDim(config.blockdim, aiCpuNum, devProg->devArgs.archInfo);
        config.aicpuNum = GetAiCpuNum(aiCpuNum, devProg->devArgs.scheCpuNum, devProg->devArgs.archInfo,
                                      devProg->devArgs.launchSchedSameCluster);
        devProg->devArgs.nrAicpu = config.aicpuNum;
    }

    // Prepare device program scheduling and memory budget related args (keeps <= 50 lines)
    static void PrepareDevProgArgs(DevAscendProgram* devProg, DeviceLauncherConfig& config,
                                   [[maybe_unused]] bool isDevice)
    {
        devProg->devArgs.taskId = 0;
        devProg->devArgs.nrAic = kDefaultAicNum;
        devProg->devArgs.nrAiv = kDefaultAivNum;
        devProg->devArgs.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        bool enableVFFusion = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
                              config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
        devProg->devArgs.enableVFFusion = (config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) == 1 ||
                                           enableVFFusion);

        if (IsPtoDataDumpEnabled()) { // dump tensor
            devProg->devArgs.hostPid = GetProcessId();
        }
        devProg->devArgs.enableAicoreResolve = IsAicoreResolveEnabled();
        if (isDevice) {
            devProg->devArgs.validGetPgMask = RuntimeAgent::GetAgent().GetValidGetPgMask();
        }

        MACHINE_LOGD("Set aicore blockdim=%d, aicpu blockdim=%d.", config.blockdim, config.aicpuNum);

        devProg->devArgs.enableCtrl = 1; // need set 0 if use custom cpu launch ctrl cpu
        if (config.dynWorkspaceSize != 0) {
            MACHINE_LOGW("[Deprecated] User provided dynamic workspace: %" PRId64, config.dynWorkspaceSize);
            devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = std::max(
                static_cast<int64_t>(devProg->memBudget.tensor.maxDynamicAssembleOutcastMem),
                AlignUp(config.dynWorkspaceSize, TENSOR_ADDR_ALIGNMENT));
        }

        if (isDevice) {
            InitDevArgs(devProg->devArgs);
        }

        devProg->workspaceSize = devProg->memBudget.Total();
        MACHINE_LOGI(
            "[workspaceSize] Metadata=%lu, workspaceSize=%lu, tensor=%lu, aicoreSpillen=%lu, debug.DumpTensor=%lu, "
            "leafDumpWorkspace=%lu.",
            devProg->memBudget.metadata.Total(), devProg->workspaceSize, devProg->memBudget.tensor.Total(),
            devProg->memBudget.aicoreSpilled.Total(), devProg->memBudget.debug.dumpTensor,
            devProg->memBudget.debug.leafDump);
        MACHINE_LOGI(
            "[workspaceSize] Tensor:rootInnerSpilledMem=%lu, devTaskInnerOutCasts=%lu, slotted=%lux%lu(slots).",
            devProg->memBudget.tensor.rootInnerSpilledMem, devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts,
            devProg->memBudget.tensor.MaxOutcastMem(),
            devProg->memBudget.tensor.BoundaryAndInnerTemporalOutcastSlotNum());
        MACHINE_LOGI("[workspaceSize] OutcastSlots: boundary=%lu, innerTemporal=%lu.",
                     devProg->memBudget.tensor.devTaskBoundaryOutcastNum,
                     devProg->memBudget.tensor.devTaskInnerTemporalOutcastNum);
    }

    static void FillSwimLaneEnableInfo(ToSubMachineConfig& toSubMachineConfig)
    {
        if (config::GetPlatformConfig(KEY_ENABLE_PROF_FUNC, false)) {
            toSubMachineConfig.profConfig.Add(ProfConfig::AICPU_FUNC);
        }
        if (config::GetPlatformConfig(KEY_ENABLE_PROF_AICORE_TIME, false) || config::IsRuntimeDebugAllEnabled()) {
            toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_TIME);
        }
        if (config::GetPlatformConfig(KEY_ENABLE_PROF_AICORE_PMU, false)) {
            toSubMachineConfig.profConfig.Add(ProfConfig::AICORE_PMU);
        }
    }

    // Fill metadata and kArgs (templated because it uses DeviceMemoryTy) (keeps <= 50 lines)
    template <typename DeviceMemoryTy>
    static void FillKernelMeta(DeviceMemoryTy& devMem, DeviceKernelArgs& kArgs, DevAscendProgram* devProg,
                               const std::vector<uint8_t>& devProgData, bool isCtrlCacheRecording,
                               const DeviceLauncherConfig& config, CachedOperator* cachedOperator)
    {
        AssignMetaAddr(devMem, kArgs, devProg, cachedOperator);
        devProg->l2CacheOffset = devMem.GetL2Offset();
        if (config.workspaceAddr) {
            kArgs.workspace = (int64_t*)config.workspaceAddr;
        } else if (!config.workspaceAllocByTorch && kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t*)devMem.AllocDev(devProg->workspaceSize,
                                                        CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        if (isCtrlCacheRecording) {
            kArgs.cfgdata = (int64_t*)devProg;
        } else if (CachedOperator::GetCfgDataDevAddrHolder(cachedOperator) &&
                   *CachedOperator::GetCfgDataDevAddrHolder(cachedOperator)) {
            /* Already copied, do not copy again. */
            kArgs.cfgdata = (int64_t*)*CachedOperator::GetCfgDataDevAddrHolder(cachedOperator);
        } else {
            kArgs.cfgdata = (int64_t*)devMem.CopyToDev(devProgData,
                                                       CachedOperator::GetCfgDataDevAddrHolder(cachedOperator));
        }
        kArgs.machineConfig = devProg->devArgs.machineConfig;
    }

    template <typename DeviceMemoryTy>
    static void FillDeviceKernelArgs(DeviceMemoryTy& memUtils, std::vector<uint8_t>& devProgData,
                                     DeviceKernelArgs& kargs, const std::vector<std::string>& groupNames)
    {
        DeviceLauncherConfig config;
        CachedOperator cache;
        DeviceLauncherConfigFillDeviceInfo(config);
        DeviceInitLauncherConfigForUser(devProgData);
        // KernelBinary init only; workspace is allocated per launch via module.alloc() (torch on NPU).
        config.workspaceAllocByTorch = true;
        DeviceInitTilingData(memUtils, kargs, devProgData, nullptr, config, &cache);
        DeviceInitDistributedContext(memUtils, groupNames, kargs);
    }

    template <typename DeviceMemoryTy>
    static void DeviceInitDistributedContext(DeviceMemoryTy& devMem, const std::vector<std::string>& groupNames,
                                             DeviceKernelArgs& kArgs)
    {
        using groupsKey = std::vector<std::string>;
        static std::map<groupsKey, int64_t*> deviceCommContextsMap;
        if (devMem.IsDevice()) {
            auto it = deviceCommContextsMap.find(groupNames);
            if (it != deviceCommContextsMap.end()) {
                kArgs.commContexts = it->second;
                return;
            }
        }
        std::vector<uint64_t> commContexts = devMem.IsDevice() ? DistributedContext::GetCommContext(groupNames) :
                                                                 DistributedContext::GetCommContextToHost(groupNames);
        commContexts.insert(commContexts.begin(), commContexts.size());
        kArgs.commContexts = reinterpret_cast<int64_t*>(devMem.CopyToDev(commContexts, nullptr));
        if (devMem.IsDevice()) {
            deviceCommContextsMap[groupNames] = kArgs.commContexts;
        }
    }

    template <typename DeviceMemoryTy>
    static void DeviceInitTilingData(DeviceMemoryTy& devMem, DeviceKernelArgs& kArgs,
                                     const std::vector<uint8_t>& devProgData, DevControlFlowCache* ctrlFlowCache,
                                     const DeviceLauncherConfig& config, CachedOperator* cachedOperator)
    {
        auto& mutableConfig = const_cast<DeviceLauncherConfig&>(config);
        auto* devProg = reinterpret_cast<DevAscendProgram*>(const_cast<uint8_t*>(devProgData.data()));
        PrepareDevProgArgsCpuInfo(devProg, mutableConfig);
        PrepareDevProgArgs(devProg, mutableConfig, devMem.IsDevice());
        // Fill all metadata and kernel args
        bool isCtrlCacheRecording = false;
        if (!devMem.IsDevice()) {
            isCtrlCacheRecording = ctrlFlowCache != nullptr ? ctrlFlowCache->IsRecording() :
                                                              devProg->controlFlowCache.IsRecording();
        }
        FillKernelMeta(devMem, kArgs, devProg, devProgData, isCtrlCacheRecording, config, cachedOperator);
        kArgs.ctrlFlowCache = reinterpret_cast<int64_t*>(ctrlFlowCache);
    }

    static void InitAicpuTaskInfo()
    {
        if (!inited_) {
            AiCpuArgs initArgs;
            (void)memcpy_s(tensorInfo_.data(), sizeof(AiCpuArgs), &initArgs, sizeof(AiCpuArgs));
            inited_ = true;
        }
    }

    /*
     *  inputs          |  inputSize  |
     *  outputs         |  outputSize |
     *                  |     ...     |
     * DevTensorData*   |    input0   |
     *                  |    input1   |
     *                  |     ...     |
     *                  |    output0  |
     *                  |     ...     |
     */
    template <typename DeviceMemoryTy>
    static void DeviceInitKernelInOuts(DeviceMemoryTy& devMem, DeviceKernelArgs& kArgs,
                                       const std::vector<DeviceTensorData>& inputList,
                                       const std::vector<DeviceTensorData>& outputList,
                                       const std::vector<uint8_t>& disableL2List, uint64_t maxPatchCount = 0)
    {
        size_t l2InfoSize = disableL2List.size();
        auto buildInouts = [&](const std::vector<DeviceTensorData>& tensorDataList, DevTensorData* data,
                               size_t& tensorIdx) {
            for (size_t k = 0; k < tensorDataList.size(); ++k) {
                auto& tensorData = tensorDataList[k];
                uint64_t addr = reinterpret_cast<uint64_t>(tensorData.GetAddr());
                if (unlikely(addr != 0 && tensorIdx < l2InfoSize && disableL2List[tensorIdx] == 1)) {
                    MACHINE_LOGI("Tensor[%zu]: ori=%#lx, l2offset=%lu.", tensorIdx, addr, devMem.GetL2Offset());
                    addr += devMem.GetL2Offset();
                }
                DevAscendTensorDataCreator::Init(data, addr, tensorData.GetShape().data(),
                                                 tensorData.GetShape().size());
                data++;
                tensorIdx++;
            }
            return;
        };
        size_t inputSize = inputList.size() * sizeof(DevTensorData);
        size_t outputSize = outputList.size() * sizeof(DevTensorData);
        size_t tensorSize = inputSize + outputSize + 2 * sizeof(uint64_t);
        size_t patchTailSize = sizeof(uint64_t) +
                               static_cast<size_t>(maxPatchCount) * sizeof(DevDynamicCellMatchStridePatch);
        size_t allSize = tensorSize + patchTailSize + sizeof(AiCpuArgs);

        if (unlikely(allSize > tensorInfo_.size())) {
            tensorInfo_.resize(allSize);
        }
        InitAicpuTaskInfo();
        auto data = reinterpret_cast<uint64_t*>(tensorInfo_.data() + sizeof(AiCpuArgs));
        *data = inputList.size();
        data++;
        *data = outputList.size();
        data++;
        auto dataPtr = reinterpret_cast<DevTensorData*>(data);
        size_t tensorIdx = 0;
        buildInouts(inputList, dataPtr, tensorIdx);
        dataPtr += inputList.size();
        buildInouts(outputList, dataPtr, tensorIdx);
        dataPtr += outputList.size();
        auto* patchCountPtr = reinterpret_cast<uint64_t*>(dataPtr);
        *patchCountPtr = 0;
        if (devMem.IsDevice()) {
            kArgs.inputs = reinterpret_cast<int64_t*>(tensorInfo_.data());
            kArgs.outputs = (int64_t*)allSize;
        } else {
            kArgs.inputs = reinterpret_cast<int64_t*>(tensorInfo_.data() + sizeof(AiCpuArgs));
            kArgs.outputs = kArgs.inputs + 1;
        }
        MACHINE_LOGD("Inputs=%p, outputs=%p, workspace=%p, cfgdata=%p, tensorSize=%zu bytes, patchTailSize=%zu bytes.",
                     kArgs.inputs, kArgs.outputs, kArgs.workspace, kArgs.cfgdata, tensorSize, patchTailSize);
    }

    template <typename DeviceMemoryTy>
    static std::pair<std::vector<DeviceTensorData>, std::vector<DeviceTensorData>> BuildInputOutputFromHost(
        DeviceMemoryTy& devMem, const std::vector<RawTensorDataPtr>& inputDataList,
        const std::vector<RawTensorDataPtr>& outputDataList)
    {
        std::vector<DeviceTensorData> inputDeviceDataList;
        std::vector<DeviceTensorData> outputDeviceDataList;
        for (size_t k = 0; k < inputDataList.size(); k++) {
            auto& inputData = inputDataList[k];
            std::vector<int64_t> shape;
            if (inputData) {
                inputData->SetDevPtr(nullptr);
                shape.insert(shape.end(), inputData->GetShape().begin(), inputData->GetShape().end());
                auto inAddr = devMem.CopyToDev(*inputData);
                inputDeviceDataList.emplace_back(inputData->GetDataType(), inAddr, shape);
            } else {
                inputDeviceDataList.emplace_back(DT_UINT8, nullptr, shape);
            }
        }
        for (size_t k = 0; k < outputDataList.size(); k++) {
            auto& outputData = outputDataList[k];
            std::vector<int64_t> shape;
            if (outputData) {
                outputData->SetDevPtr(nullptr);
                shape.insert(shape.end(), outputData->GetShape().begin(), outputData->GetShape().end());
                auto outAddr = devMem.CopyToDev(*outputData);
                outputDeviceDataList.emplace_back(outputData->GetDataType(), outAddr, shape);
            } else {
                outputDeviceDataList.emplace_back(DT_UINT8, nullptr, shape);
            }
        }
        return std::make_pair(inputDeviceDataList, outputDeviceDataList);
    }

    template <typename DeviceMemoryTy>
    static void CopyFromDev(DeviceMemoryTy devMem, const std::vector<RawTensorDataPtr>& outputs)
    {
        for (auto& output : outputs) {
            if (output) {
                devMem.CopyFromDev(*output);
            }
        }
    }

    static int SetCaptureStream(RtStream aicoreStream, RtStream aicpuStream, bool& isCapture);
    static int RunWithProfile(RtStream aicoreStream, RtStream aicpuStream, bool isCapture);
    static int DynamicLaunchSynchronize(RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream);
    static int DeviceLaunchOnceWithDeviceTensorData(Function* function, const std::vector<DeviceTensorData>& inputList,
                                                    const std::vector<DeviceTensorData>& outputList,
                                                    RtStream aicoreStream, bool streamSynchronize,
                                                    CachedOperator* cachedOperator,
                                                    DevControlFlowCache* ctrlCache = nullptr,
                                                    const DeviceLauncherConfig& config = DeviceLauncherConfig());

    static int DeviceSynchronize(RtStream aicpuStream, RtStream aicoreStream);
    static uint8_t* CopyControlFlowCache(DevControlFlowCache* ctrlCache);
    static void FreeControlFlowCache(uint8_t* ctrlCache);
    static bool IsCaptureMode();
    static void SaveStream(AclRtStream aicoreStream);
    static void GetCaptureInfo(AclRtStream aicoreStream, AclMdlRI& rtModel);
    static void AddAicpuStream(const bool isCapture, AclMdlRI& rtModel);
    static int LaunchAicpuKernel(AicpuLaunchDesc& launchDesc, [[maybe_unused]] bool debugEnable,
                                 [[maybe_unused]] Function* function,
                                 const std::vector<DeviceTensorData>& tensors = {});
    static int LaunchSyncTask(AclRtStream aicoreStream, bool isCaptureMode, int launchEarlyMode);
    static int LaunchAicoreKernel(AclRtStream aicoreStream, void* kernel, RtArgsEx& rtArgs, RtTaskCfgInfo& rtTaskCfg,
                                  bool debugEnable, [[maybe_unused]] Function* function);
    static int LaunchKernel(AclRtStream aicoreStream, uint8_t* ctrlFlowCache, KernelBinary* kernel, int64_t* workspace,
                            const std::vector<DeviceTensorData>& tensors, bool isDebugMode, int launchEarlyMode);
    static int DeviceRunOnce(Function* function, DevControlFlowCache* hostCtrlCache = nullptr,
                             const DeviceLauncherConfig& config = DeviceLauncherConfig());
    static void EmulationLaunch(Function* function, const std::vector<DeviceTensorData>& tensors,
                                DevControlFlowCache* ctrlCache, LaunchMode launchMode);
    static uint8_t* PrepareLaunch(KernelBinary* kernel, std::vector<DeviceTensorData>& tensors, AclMdlRI rtModel,
                                  LaunchMode launchMode);

    static void SetDevRunCacheKernelEnable(Function* func, bool enabled);
    static bool IsDevRunCacheKernelEnable(Function* func);
    static void SetDevRunCacheKernel(Function* func, uint8_t* devProg);
    static CachedOperator* GetDevRunCacheOperator(Function* func);
    static void SetDevPerfAddr(const bool debugEnable, const bool isCaptureMode,
                               const ToSubMachineConfig& profLevelConfig);
    static void DumpIOTensorsWithCann(AclRtStream stream, std::vector<DeviceTensorData>& tensors,
                                      const std::string& funcName);
    static int RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream);

private:
    static void DataDumpInit();
    static void DataDumpUnInit();
    struct DeviceRunCacheInfo {
        /* By default: devProg cache is enabled */
        bool devProgEnabled{true};
        CachedOperator cacheOperator;
    };

    static bool inited_;
    static std::vector<uint8_t> tensorInfo_;
    static std::unordered_map<Function*, DeviceRunCacheInfo> cacheInfoDict_;
    static std::atomic<int64_t> sequence_;
};
} // namespace npu::tile_fwk::dynamic
#endif // SRC_MACHINE_DEVICE_LAUNCHER_H
