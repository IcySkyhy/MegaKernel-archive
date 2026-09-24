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
 * \file test_dev_func_runner.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <thread>
#include <cstdint>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/runner/kernel_binary.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "interface/program/program.h"
#include "cost_model/simulation/backend.h"

using namespace npu::tile_fwk::dynamic;

namespace npu::tile_fwk {

struct MemoryHelper {
    MemoryHelper(bool isTest) : isTest_(isTest) {}
    ~MemoryHelper() = default;

    bool IsDevice() { return !isTest_; }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        if (devPtr == nullptr) {
            return nullptr;
        }
        if (isTest_)
            memcpy_s(devPtr, size, data, size);
        else
            RuntimeMemcpy(devPtr, size, data, size, RtMemcpyKind::HOST_TO_DEVICE);
        return devPtr;
    }

    void CopyFromDev(uint8_t* data, uint8_t* devPtr, uint64_t size)
    {
        if (isTest_)
            memcpy_s(data, size, devPtr, size);
        else
            RuntimeMemcpy(data, size, devPtr, size, RtMemcpyKind::DEVICE_TO_HOST);
    }

    uint8_t* AllocDev(size_t size, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        uint8_t* devPtr = nullptr;
        if (isTest_) {
            size_t alignSize = 512;
            size_t totalSize = size + alignSize;

            if (totalSize < size || totalSize > 0x500000000) {
                return nullptr;
            }

            uint8_t* rawPtr = (uint8_t*)malloc(totalSize);
            if (rawPtr == nullptr) {
                MACHINE_LOGE(DevCommonErr::MALLOC_FAILED, "[AllocDev] malloc totalSize %zu failed", totalSize);
                return nullptr;
            }
            std::shared_ptr<uint8_t> ptr(rawPtr, free);
            memset_s(rawPtr, totalSize, 0, totalSize);

            devPtr = (uint8_t*)((((uint64_t)rawPtr) + alignSize - 1) / alignSize * alignSize);
            testAllocatePtrs_.push_back(ptr);
        } else {
            DevMemoryPool::Instance().AllocDevAddr(&devPtr, size);
        }
        return devPtr;
    }

    uint8_t* AllocZero(uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        uint8_t* devPtr = AllocDev(size, nullptr);
        if (devPtr == nullptr) {
            return nullptr;
        }
        if (isTest_)
            memset(devPtr, 0, size);
        else
            RuntimeMemset(devPtr, size, 0, size);
        return devPtr;
    }

    template <typename T>
    T* CopyToDev(std::vector<T> data, uint8_t** cachedHolder)
    {
        (void)cachedHolder;
        return (T*)CopyToDev((uint8_t*)data.data(), data.size() * sizeof(T), nullptr);
    }

    uint8_t* CopyToDev(RawTensorData& data)
    {
        if (data.GetDevPtr() == nullptr) {
            auto devPtr = CopyToDev((uint8_t*)data.data(), data.size(), nullptr);
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData& tensorData)
    {
        CopyFromDev(tensorData.data(), tensorData.GetDevPtr(), tensorData.size());
    }

    static uint64_t GetL2Offset() { return GetRuntimeL2Offset(); }

    bool isTest_{true};
    std::vector<std::shared_ptr<uint8_t>> testAllocatePtrs_;
};

extern "C" int DynTileFwkBackendKernelServer(void* targ);
extern "C" int PyptoKernelCtrlServer(void* targ);

class DevFuncRunner : public DeviceLauncher {
public:
    static void Run(Function* function, const std::vector<RawTensorDataPtr>& inputs,
                    const std::vector<RawTensorDataPtr>& outputs,
                    const DeviceLauncherConfig& config = DeviceLauncherConfig())
    {
        ProgramData::GetInstance().AppendInputs(inputs);
        ProgramData::GetInstance().AppendOutputs(outputs);
        auto runner = DevFuncRunner(function, config);
        runner.RunDynamic(inputs, outputs);
    }

    // Run with incast/outcast from ProgramData
    static void Run(Function* function, const DeviceLauncherConfig& config = DeviceLauncherConfig())
    {
        auto& inputs = ProgramData::GetInstance().GetInputDataList();
        auto& outputs = ProgramData::GetInstance().GetOutputDataList();
        auto runner = DevFuncRunner(function, config);
        runner.RunDynamic(inputs, outputs);
    }

private:
    DevFuncRunner(Function* function, const DeviceLauncherConfig& config) : function_(function), config_(config) {}
    void RunDynamic(const std::vector<RawTensorDataPtr>& inputs, const std::vector<RawTensorDataPtr>& outputs)
    {
        if (function_ == nullptr || function_->GetDyndevAttribute() == nullptr) {
            return;
        }
        KernelLaunchPrecheck(inputs, outputs);
        DevAscendProgram* functionDevProg = reinterpret_cast<DevAscendProgram*>(
            function_->GetDyndevAttribute()->devProgBinary.data());
        if (config_.controlFlowCache) {
            functionDevProg->controlFlowCache.isRecording = true;
        }
        RunModel(inputs, outputs);
        if (functionDevProg->controlFlowCache.isRecording) {
            functionDevProg->controlFlowCache.isRecording = false;
            uint64_t contextWorkspaceAddr = functionDevProg->controlFlowCache.contextWorkspaceAddr;
            functionDevProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
            functionDevProg->controlFlowCache.RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr,
                                                                        nullptr, functionDevProg->GetParallelism());
            functionDevProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(functionDevProg), 0);
            functionDevProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
            functionDevProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
                reinterpret_cast<uint64_t>(functionDevProg),
                reinterpret_cast<uint64_t>(&functionDevProg->controlFlowCache), 0, 0);
            functionDevProg->ResetFromLaunch();
            functionDevProg->controlFlowCache.isActivated = true;
        }
        if (config_.onBoard) {
            RunOnBoard(inputs, outputs);
        }
    }

    void RunModel(const std::vector<RawTensorDataPtr>& inputs, const std::vector<RawTensorDataPtr>& outputs)
    {
        if (!config_.runModel) {
            return;
        }
        DeviceKernelArgs kArgs;
        MemoryHelper memoryHelper(true);
        (void)InitKernelRuntime(memoryHelper, kArgs);
        RefillDynamicBudgetsAndSyncCfg(memoryHelper, kArgs, inputs, outputs, true);
        for (int i = 0; i < (config_.controlFlowCache ? 1 : config_.repeatNum); i++) {
            InitKernelInOuts(memoryHelper, kArgs, inputs, outputs, true, {});
            std::cout << "!!! Run CostModel " << i << "\n";
            RunCostModel(&kArgs);
            std::cout << "!!! Run TestModel " << i << "\n";
            RunTestMode(&kArgs);
        }

        CopyBackInOut(memoryHelper, inputs, outputs);
        RunDynCostModel();
    }

    void RefillDynamicBudgetsAndSyncCfg(MemoryHelper& memoryHelper, DeviceKernelArgs& kArgs,
                                        const std::vector<RawTensorDataPtr>& inputs,
                                        const std::vector<RawTensorDataPtr>& outputs, bool syncMemBudgetToCfg)
    {
        auto dynAttr = function_->GetDyndevAttribute();
        auto* functionDevProg = reinterpret_cast<DevAscendProgram*>(dynAttr->devProgBinary.data());
        std::vector<DeviceTensorData> evalInputList;
        std::vector<DeviceTensorData> evalOutputList;
        std::tie(evalInputList, evalOutputList) = BuildInputOutputFromHost(memoryHelper, inputs, outputs);
        Evaluator eval{dynAttr->inputSymbolDict, &evalInputList, &evalOutputList};
        auto* cfgProg = reinterpret_cast<DevAscendProgram*>(kArgs.cfgdata);
        cellMatchDescPatches_ = PrepareHostDynamicCellMatchForLaunch(*dynAttr.get(), eval, functionDevProg);
        if (syncMemBudgetToCfg) {
            cfgProg->memBudget = functionDevProg->memBudget;
        }
        kArgs.maxDynamicAssembleOutcastMem = functionDevProg->memBudget.tensor.maxDynamicAssembleOutcastMem;
        kArgs.maxDynamicCellMatchTableMem = functionDevProg->memBudget.metadata.maxDynamicCellMatchTableMem;
        PatchRuntimeDynamicCellMatchMeta(memoryHelper, functionDevProg, kArgs);
    }

    bool IsDumpTensorEnable() const { return GetDevProg(function_)->memBudget.debug.dumpTensor != 0; }

    static void DumpDevDataBinary(std::ostream& os, const uint8_t* hostData, uint64_t size, const uint8_t* devptr)
    {
        /*
         * Format:
         *   8 bytes: address on device
         *   8 bytes: data block size
         *   n bytes: data block
         */
        uint64_t header[] = {
            reinterpret_cast<uint64_t>(devptr),
            size,
        };
        os.write(reinterpret_cast<const char*>(header), sizeof(header));
        if (hostData != nullptr) {
            os.write(reinterpret_cast<const char*>(hostData), size);
        } else {
            static constexpr uint64_t THROUGHPUT = UINT64_C(1024) * 1024 * 1024;
            std::vector<uint8_t> buf;
            buf.reserve(std::min(THROUGHPUT, size));
            for (uint64_t offset = 0; offset < size; offset += THROUGHPUT) {
                uint64_t blockSize = std::min(THROUGHPUT, size - offset);
                RuntimeMemcpy(buf.data(), buf.capacity(), devptr + offset, blockSize, RtMemcpyKind::DEVICE_TO_HOST);
                os.write(reinterpret_cast<const char*>(buf.data()), blockSize);
            }
        }
    }

    void DumpTensorContents(int64_t* workspace, const std::vector<RawTensorDataPtr>& inputs,
                            const std::vector<RawTensorDataPtr>& outputs)
    {
        auto* devProg = GetDevProg(function_);
        uint8_t* dumpTensorWsPtr = reinterpret_cast<uint8_t*>(workspace) + devProg->memBudget.Total() -
                                   devProg->memBudget.debug.dumpTensor;
        uint64_t dumpTensorWsUsed = 0;
        RuntimeMemcpy(&dumpTensorWsUsed, sizeof(uint64_t), dumpTensorWsPtr, sizeof(uint64_t),
                      RtMemcpyKind::DEVICE_TO_HOST);
        MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED, "[DumpTensor] dumpTensorWsPtr=%p, memory used=%lu\n", dumpTensorWsPtr,
                     dumpTensorWsUsed);

        std::string path = config::LogTopFolder() + "/dump_tensor.txt";
        std::ofstream fout(path, std::ios::out | std::ios::binary);

        auto printIODevAddrs = [&](const std::vector<RawTensorDataPtr>& ptrs) {
            uint64_t ptrNum = ptrs.size();
            fout.write(reinterpret_cast<const char*>(&ptrNum), sizeof(ptrNum));
            int idx = 0;
            for (auto& ptr : ptrs) {
                uint64_t devPtr = ptr ? reinterpret_cast<uint64_t>(ptr->GetDevPtr()) : 0;
                MACHINE_LOGW("[DumpTensor] devPtr %d = %lu\n", idx++, devPtr);
                fout.write(reinterpret_cast<const char*>(&devPtr), sizeof(devPtr));
            }
        };

        // write input/output devAddr list
        MACHINE_LOGW("[DumpTensor] #inputs=%zu\n", inputs.size());
        printIODevAddrs(inputs);
        MACHINE_LOGW("[DumpTensor] #outputs=%zu\n", outputs.size());
        printIODevAddrs(outputs);

        DumpDevDataBinary(fout, nullptr, dumpTensorWsUsed, dumpTensorWsPtr);
        for (auto& input : inputs) {
            if (input) {
                DumpDevDataBinary(fout, input->data(), input->GetDataSize(), input->GetDevPtr());
            }
        }
        for (auto& output : outputs) {
            if (output) {
                DumpDevDataBinary(fout, output->data(), output->GetDataSize(), output->GetDevPtr());
            }
        }
        fout.close();
    }

    void RunOnBoard(const std::vector<RawTensorDataPtr>& inputs, const std::vector<RawTensorDataPtr>& outputs)
    {
        std::cout << "!!! Kernel Launch " << "\n";
        int rc = AclInit(nullptr);
        if (rc != 0 && rc != ACLRT_ERROR_REPEAT_INITIALIZE) {
            MACHINE_LOGE(RtErr::RT_INIT_FAILED, "Acl init failed!!!");
            return;
        }
        CheckDeviceId();

        auto kernel = std::make_unique<KernelBinary>(Program::GetInstance().GetFunctionSharedPtr(function_));
        HostProf::GetInstance().SetProfFunction(function_);

        MemoryHelper memoryHelper(false);
        std::vector<DeviceTensorData> inputList;
        std::vector<DeviceTensorData> outputList;
        std::tie(inputList, outputList) = BuildInputOutputFromHost(memoryHelper, inputs, outputs);

        std::vector<DeviceTensorData> tensors;
        tensors.reserve(inputList.size() + outputList.size());
        tensors.insert(tensors.end(), inputList.begin(), inputList.end());
        tensors.insert(tensors.end(), outputList.begin(), outputList.end());
        int64_t wsSize = kernel->GetWorkspaceSize(tensors);
        int64_t* wsAddr = nullptr;
        if (wsSize > 0) {
            wsAddr = reinterpret_cast<int64_t*>(memoryHelper.AllocDev(static_cast<size_t>(wsSize), nullptr));
            if (wsAddr == nullptr) {
                MACHINE_LOGE(RtErr::RT_MALLOC_FAILED, "Malloc dev failed!!!");
                return;
            }
        }

        auto aicoreStream = GetContextAiCoreStream();
        uint8_t* ctrlFlowCache = DeviceLauncher::PrepareLaunch(kernel.get(), tensors, nullptr, LaunchMode::DEVICE_RT);

        for (int i = 0; i < config_.repeatNum; i++) {
            rc = DeviceLauncher::LaunchKernel(aicoreStream, ctrlFlowCache, kernel.get(), wsAddr, tensors, false, 0);
            EXPECT_EQ(rc, 0);

            rc = DeviceLauncher::DynamicLaunchSynchronize(GetContextScheStream(), GetContextCtrlStream(), aicoreStream);
            EXPECT_EQ(rc, 0);
            bool debugEnable = config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL;
            DevicePerf::GetInstance().SyncProfData(debugEnable);
        }
        CopyBackInOut(memoryHelper, inputs, outputs);
        if (IsDumpTensorEnable()) {
            DumpTensorContents(wsAddr, inputs, outputs);
        }
    }

    std::shared_ptr<DyndevFunctionAttribute> InitKernelRuntime(MemoryHelper& memoryHelper, DeviceKernelArgs& kArgs)
    {
        auto dynAttr = function_->GetDyndevAttribute();
        DeviceLauncherConfigFillDeviceInfo(config_);
        DeviceInitDistributedContext(memoryHelper, dynAttr->commGroupNames, kArgs);
        DeviceInitTilingData(memoryHelper, kArgs, dynAttr->devProgBinary, nullptr, config_, nullptr);
        return dynAttr;
    }

    void CopyBackInOut(MemoryHelper& memoryHelper, const std::vector<RawTensorDataPtr>& inputs,
                       const std::vector<RawTensorDataPtr>& outputs)
    {
        CopyFromDev(memoryHelper, outputs);
        if (outputs.size() == 0 || HasInplaceArgs(function_)) {
            CopyFromDev(memoryHelper, inputs);
        }
    }

    void RunCostModel(DeviceKernelArgs* kArgs)
    {
        if (!config::GetPlatformConfig(KEY_ENABLE_DYN_COST_MODEL, true)) {
            return;
        }
        Function* function = Program::GetInstance().GetLastFunction();
        if (function == nullptr) {
            return;
        }
        config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::LEAF_FUNCTION);
        auto attr = function->GetDyndevAttribute();
        auto* modelData = new CostModel::ModelData();
        modelData->functionTime.assign(attr->devLeafIndex2Hash.size(), 0);
        CostModelAgent costModelAgent;
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        for (const auto& [index, hash] : attr->devLeafIndex2Hash) {
            auto time = costModelAgent.GetLeafFunctionTimeCost(hash);
            modelData->functionTime[index] = time;
            DEV_INFO("devLeafIndex2Hash, %d -> %lu: %lu\n", index, hash, time);
        }
        costModelAgent.TerminateCostModel();
        kArgs->costmodeldata = modelData;
    }

    void RunDynCostModel()
    {
        if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM) {
            return;
        }
        std::string path = config::LogTopFolder() + "/dyn_topo.txt";
        config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::NORMAL);
        CostModelAgent costModelAgent;
        costModelAgent.SubmitTopo(path);
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
        MACHINE_LOGD("Finish Run DynCostMode which topo path is: %s", path.c_str());
    }

    void RunTestMode(DeviceKernelArgs* kArgs)
    {
        std::thread aicpus[DEVICE_MAX_AICPU_NUM];
        std::atomic<int> idx{0};
        auto* devProg = (DevAscendProgram*)(kArgs->cfgdata);
        size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
        (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
        auto threadNum = static_cast<int>(devProg->devArgs.nrAicpu + dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
        auto threadFun = [&](uint32_t runMode) {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            (void)sprintf_s(name, sizeof(name), "aicput%d", tidx);
            std::cout << "start thread: " << name << std::endl;
            pthread_setname_np(pthread_self(), name);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            DeviceKernelArgs localArgs = *kArgs;
            localArgs.parameter.runMode = runMode;
            auto rc = DynTileFwkBackendKernelServer(&localArgs);
            EXPECT_EQ(rc, 0);
        };
        aicpus[0] = std::thread(threadFun, RUN_SPLITTED_STREAM_CTRL);
        for (int i = 1; i < threadNum; i++) {
            aicpus[i] = std::thread(threadFun, RUN_SPLITTED_STREAM_SCHE);
        }

        for (int i = 0; i < threadNum; i++) {
            if (aicpus[i].joinable()) {
                aicpus[i].join();
            }
        }
    }

    void InitKernelInOuts(MemoryHelper& memoryHelper, DeviceKernelArgs& kArgs,
                          const std::vector<RawTensorDataPtr>& inputTensors,
                          const std::vector<RawTensorDataPtr>& outputTensors, [[maybe_unused]] bool isTest,
                          const std::vector<uint8_t>& disableL2List)
    {
        std::vector<DeviceTensorData> inputList;
        std::vector<DeviceTensorData> outputList;
        std::tie(inputList, outputList) = BuildInputOutputFromHost(memoryHelper, inputTensors, outputTensors);
        const uint64_t maxPatchCount = function_->GetDyndevAttribute()->dynamicCellMatchLaunchMetaList.size();
        DeviceInitKernelInOuts(memoryHelper, kArgs, inputList, outputList, disableL2List, maxPatchCount);
        // Device path: kArgs.inputs points at AiCpuArgs; tensor info (and patch tail) starts after it.
        int64_t* launchInputs = kArgs.inputs;
        if (memoryHelper.IsDevice()) {
            launchInputs = reinterpret_cast<int64_t*>(reinterpret_cast<uint8_t*>(kArgs.inputs) + sizeof(AiCpuArgs));
        }
        WriteDynamicCellMatchStridePatchesToLaunchArgs(launchInputs, cellMatchDescPatches_);
        MACHINE_LOGI("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
                     kArgs.cfgdata);
    }

    void KernelLaunchPrecheck(const std::vector<RawTensorDataPtr>& inputs, const std::vector<RawTensorDataPtr>& outputs)
    {
        auto checkInouts = [&](std::vector<std::reference_wrapper<const Tensor>>& tensorList,
                               const std::vector<RawTensorDataPtr>& dataList) {
            EXPECT_EQ(tensorList.size(), dataList.size()) << "argument num not match !!!!";
            for (size_t i = 0; i < tensorList.size(); i++) {
                auto& t = tensorList[i].get();
                auto& d = dataList[i];
                if (d) {
                    EXPECT_EQ(t.GetDataType(), d->GetDataType());
                    auto rawShape = t.GetStorage()->GetRawTensor()->GetDynRawShape();
                    auto shape = d->GetShape();
                    for (size_t k = 0; k < rawShape.size(); k++) {
                        if (rawShape[k].IsImmediate()) {
                            EXPECT_EQ(rawShape[k].Concrete(), shape[k]);
                        }
                    }
                }
            }
        };

        checkInouts(function_->GetDyndevAttribute()->startArgsInputTensorList, inputs);
        checkInouts(function_->GetDyndevAttribute()->startArgsOutputTensorList, outputs);
    }

private:
    Function* function_;
    DeviceLauncherConfig config_;
    std::vector<DevDynamicCellMatchStridePatch> cellMatchDescPatches_;
};
} // namespace npu::tile_fwk
