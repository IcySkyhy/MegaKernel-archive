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
 * \file cost_model_launcher.h
 * \brief
 */

#pragma once

#include <thread>
#include <cstdint>
#include <unistd.h>
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "cost_model/simulation/pv/PvModel.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "cost_model/simulation/backend.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {
class HostAgentStub {
public:
    HostAgentStub(HostAgentStub& other) = delete;

    void operator=(const HostAgentStub& other) = delete;

    static HostAgentStub* GetAgent()
    {
        static HostAgentStub inst;
        return &inst;
    }

    uint8_t* AllocHostAddr(uint64_t size)
    {
        auto hostPtr = (uint8_t*)malloc(size);
        CHECK(npu::tile_fwk::InternalError::SIM_INNER_ERROR, hostPtr != nullptr) << "alloc host addr failed.";
        allocatedHostAddr.emplace_back(hostPtr);
        return hostPtr;
    }

    void Finalize()
    {
        if (hostInited) {
            DestroyMemory();
        }
    }

    ~HostAgentStub() { Finalize(); }

protected:
    HostAgentStub() { Init(); }

    void DestroyMemory()
    {
        for (uint8_t* addr : allocatedHostAddr) {
            free(addr);
        }
    }

private:
    void Init() { hostInited = true; }

private:
    bool hostInited{false};

    std::vector<uint8_t*> allocatedHostAddr;
};

struct MemoryHelper {
    MemoryHelper(bool isTest) : isTest_(isTest) {}

    bool IsDevice() { return !isTest_; }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        auto ptr = npu::tile_fwk::dynamic::HostAgentStub::GetAgent()->AllocHostAddr(size);
        memcpy_s(ptr, size, data, size);
        return ptr;
    }

    template <typename T>
    T* CopyToDev(std::vector<T> data)
    {
        return (T*)CopyToDev((uint8_t*)data.data(), data.size() * sizeof(T));
    }

    template <typename T>
    T* CopyToDev(std::vector<T> data, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
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

    void CopyFromDev(uint8_t* data, uint8_t* devPtr, uint64_t size) { memcpy_s(data, size, devPtr, size); }

    uint8_t* AllocDev(size_t size, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        uint8_t* devPtr = npu::tile_fwk::dynamic::HostAgentStub::GetAgent()->AllocHostAddr(size);
        return devPtr;
    }

    uint8_t* AllocZero(uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        (void)cachedDevAddrHolder;
        uint8_t* devPtr = AllocDev(size, nullptr);
        memset_s(devPtr, size, 0, size);
        return devPtr;
    }

    void CopyFromDev(RawTensorData& t) { CopyFromDev(t.data(), t.GetDevPtr(), t.size()); }

    uint64_t GetL2Offset() { return 0; }

    bool isTest_{true};
};

class AiCorePvModelImpl : public CostModel::AiCoreModel {
private:
    std::shared_ptr<CostModel::DynPvModel> pv_;
    std::unordered_map<int, uint64_t> funcdata_;
    std::mutex mtx_;

public:
    explicit AiCorePvModelImpl(std::shared_ptr<CostModel::DynPvModel> pv) : pv_(pv) {}

    void InitData(int coreIdx, int64_t funcdata)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        funcdata_[coreIdx] = funcdata;
    }

    void SendTask(int coreIdx, uint64_t taskId)
    {
        auto funcdata = funcdata_[coreIdx];
        DynFuncHeader* header = reinterpret_cast<DynFuncHeader*>(funcdata);
        DynFuncData* data = reinterpret_cast<DynFuncData*>(header + 1);
        pv_->Run(data, coreIdx, FuncID(taskId), TaskID(taskId));
    }
};

extern "C" int DynTileFwkBackendKernelServer(void* targ);
extern "C" int PyptoKernelCtrlServer(void* targ);

class CostModelLauncher : public DeviceLauncher {
public:
    static void CostModelRunOnce(Function* function, const std::vector<RawTensorDataPtr>& inputs,
                                 const std::vector<RawTensorDataPtr>& outputs,
                                 const DeviceLauncherConfig& config = DeviceLauncherConfig())
    {
        auto runner = CostModelLauncher(function, config);
        runner.RunDynamic(inputs, outputs);
    }

    // Run with incast/outcast from ProgramData
    static void CostModelRunOnce(Function* function, const DeviceLauncherConfig& config = DeviceLauncherConfig())
    {
        auto& inputs = ProgramData::GetInstance().GetInputDataList();
        auto& outputs = ProgramData::GetInstance().GetOutputDataList();
        auto runner = CostModelLauncher(function, config);
        runner.RunDynamic(inputs, outputs);
    }

private:
    CostModelLauncher(Function* function, const DeviceLauncherConfig& config) : function_(function), config_(config) {}

    void RunDynamic(const std::vector<RawTensorDataPtr>& inputs, const std::vector<RawTensorDataPtr>& outputs)
    {
        if (function_ == nullptr || function_->GetDyndevAttribute() == nullptr ||
            config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM) {
            SIMULATION_LOGI("function is invalid or run mode is not simulation");
            return;
        }
        RunModel(inputs, outputs);
    }

    static void RunStatic() {}

    void RunModel(const std::vector<RawTensorDataPtr>& inputs, const std::vector<RawTensorDataPtr>& outputs)
    {
        DeviceKernelArgs kArgs;
        auto dynAttr = function_->GetDyndevAttribute();
        DeviceLauncherConfigFillDeviceInfo(config_);
        MemoryHelper memoryHelper(true);
        DeviceInitDistributedContext(memoryHelper, dynAttr->commGroupNames, kArgs);
        DeviceInitTilingData(memoryHelper, kArgs, dynAttr->devProgBinary, nullptr, config_, nullptr);
        RefillDynamicCellMatchAndMemBudget(memoryHelper, kArgs, inputs, outputs);
        InitKernelInOuts(kArgs, inputs, outputs, true);
        auto* functionDevProg = reinterpret_cast<DevAscendProgram*>(dynAttr->devProgBinary.data());
        kArgs.maxDynamicAssembleOutcastMem = functionDevProg->memBudget.tensor.maxDynamicAssembleOutcastMem;
        kArgs.maxDynamicCellMatchTableMem = functionDevProg->memBudget.metadata.maxDynamicCellMatchTableMem;
        RunCostModel(&kArgs);
        SIMULATION_LOGI("Run TestModel");
        RunTestMode(&kArgs);
        SIMULATION_LOGI("Run DynCostModel");
        RunDynCostModel();
        SIMULATION_LOGI("Run PvModel");
#ifdef BUILD_WITH_CANN
        RunPvModel(kArgs, inputs, outputs);
#endif
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
        CostModelAgent costModelAgent;
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
        CostModel::ModelData* modelData = new CostModel::ModelData();
        auto attr = function->GetDyndevAttribute();
        modelData->functionTime.resize(attr->devLeafIndex2Hash.size(), 0);
        for (const auto& [index, hash] : attr->devLeafIndex2Hash) {
            auto time = costModelAgent.GetLeafFunctionTimeCost(hash);
            DEV_INFO("devLeafIndex2Hash, %d -> %lu: %lu\n", index, hash, time);
            modelData->functionTime[index] = time;
        }
        kArgs->costmodeldata = modelData;
    }

    void RunDynCostModel()
    {
        config::SetSimConfig(KEY_SIM_MODE, CostModel::SimMode::NORMAL);
        CostModelAgent costModelAgent;

        std::string path = config::LogTopFolder() + "/dyn_topo.txt";
        costModelAgent.SubmitTopo(path);
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
    }

    void RunPvModel(DeviceKernelArgs& kArgs, const std::vector<RawTensorDataPtr>& inputs,
                    const std::vector<RawTensorDataPtr>& outputs)
    {
        if (std::getenv("ASCEND_HOME_PATH") == nullptr) {
            SIMULATION_LOGE(CostModel::PrecisionSimErrorScene::CANN_LOAD_FAILED,
                            "ASCEND_HOME_PATH is not set, precision simulation unavailable.");
            return;
        }

        try {
            pv_ = CostModel::PvModelFactory::CreateDyn();
            pv_->InitPv();
        } catch (const std::runtime_error& e) {
            return;
        }
        model_ = std::make_shared<AiCorePvModelImpl>(pv_);
        pv_->Codegen(function_);
        BuildPvKernelArgs(kArgs, inputs, outputs);
        RunTestMode(&kArgs);
        pv_->CopyTensorFromDev();
    }

    void BuildPvKernelArgs(DeviceKernelArgs& kArgs, const std::vector<RawTensorDataPtr>& inputs,
                           const std::vector<RawTensorDataPtr>& outputs)
    {
        MemoryHelper devMem{true};
        auto buildInouts = [&](auto& tensorList, DevTensorData* tensorData) {
            for (auto& t : tensorList) {
                auto addrs = reinterpret_cast<uint64_t>(pv_->CopyTensorToDev((uint8_t*)t->data(), t->size()));
                DevAscendTensorDataCreator::Init(tensorData, addrs, t->GetShape().data(), t->GetShape().size());
                tensorData++;
            }
            return;
        };

        std::vector<uint8_t>& devProgData = function_->GetDyndevAttribute()->devProgBinary;
        auto* devProg = reinterpret_cast<DevAscendProgram*>(const_cast<uint8_t*>(devProgData.data()));

        devProg->devArgs.nrAicpu = 1;
        devProg->devArgs.nrValidAic = 1;
        devProg->devArgs.scheCpuNum = 1;
        AssignMetaAddr(devMem, kArgs, devProg, nullptr);
        PatchRuntimeDynamicCellMatchMeta(devMem, devProg, kArgs);
        const uint64_t maxPatchCount = function_->GetDyndevAttribute()->dynamicCellMatchLaunchMetaList.size();
        size_t patchTailSize = sizeof(uint64_t) + maxPatchCount * sizeof(DevDynamicCellMatchStridePatch);
        size_t tensorSize = (inputs.size() + outputs.size()) * sizeof(DevTensorData) + 2 * sizeof(uint64_t) +
                            patchTailSize;
        std::vector<uint8_t> tensorInfo(tensorSize);
        auto data = reinterpret_cast<uint64_t*>(tensorInfo.data());
        *data = inputs.size();
        data++;
        *data = outputs.size();
        data++;
        auto dataPtr = reinterpret_cast<DevTensorData*>(data);
        buildInouts(inputs, dataPtr);
        dataPtr += inputs.size();
        buildInouts(outputs, dataPtr);
        WriteDynamicCellMatchStridePatchesToLaunchArgs(reinterpret_cast<int64_t*>(tensorInfo.data()),
                                                       cellMatchDescPatches_);
        kArgs.inputs = (int64_t*)pv_->CopyToDev(tensorInfo.data(), tensorSize);
        kArgs.outputs = kArgs.inputs + 1;
        kArgs.cfgdata = (int64_t*)pv_->CopyToDev(devProgData.data(), devProgData.size());
        kArgs.aicoreModel = model_.get();
    }

    void RunTestMode(DeviceKernelArgs* kArgs)
    {
        std::atomic<int> idx{0};
        auto* devProg = (DevAscendProgram*)(kArgs->cfgdata);
        size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        auto deviceTaskCtrlPoolAddr = devProg->devArgs.runtimeDataRingBufferAddr + sizeof(RuntimeDataRingBufferHead) +
                                      DEV_ARGS_SIZE;
        (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
        int launchAiCpuNum = static_cast<int>(devProg->devArgs.nrAicpu + dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
        std::vector<std::thread> aicpus(launchAiCpuNum);
        auto threadFun = [&](uint32_t runMode) {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            std::string name = "aicput" + std::to_string(tidx);
            pthread_setname_np(pthread_self(), name.c_str());
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            DeviceKernelArgs localArgs = *kArgs;
            localArgs.parameter.runMode = runMode;
            (void)DynTileFwkBackendKernelServer(&localArgs);
        };

        aicpus[0] = std::thread(threadFun, RUN_SPLITTED_STREAM_CTRL);
        for (int i = 1; i < launchAiCpuNum; i++) {
            aicpus[i] = std::thread(threadFun, RUN_SPLITTED_STREAM_SCHE);
        }

        for (int i = 0; i < launchAiCpuNum; i++) {
            if (aicpus[i].joinable()) {
                aicpus[i].join();
            }
        }
    }

    void InitKernelInOuts(DeviceKernelArgs& kArgs, const std::vector<RawTensorDataPtr>& inputTensors,
                          const std::vector<RawTensorDataPtr>& outputTensors, bool isTest)
    {
        std::vector<DeviceTensorData> inputList;
        std::vector<DeviceTensorData> outputList;
        MemoryHelper memoryHelper(isTest);
        std::tie(inputList, outputList) = BuildInputOutputFromHost(memoryHelper, inputTensors, outputTensors);
        const uint64_t maxPatchCount = function_->GetDyndevAttribute()->dynamicCellMatchLaunchMetaList.size();
        DeviceInitKernelInOuts(memoryHelper, kArgs, inputList, outputList, {}, maxPatchCount);
        WriteDynamicCellMatchStridePatchesToLaunchArgs(kArgs.inputs, cellMatchDescPatches_);
        SIMULATION_LOGI("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
                        kArgs.cfgdata);
    }

    void RefillDynamicCellMatchAndMemBudget(MemoryHelper& memoryHelper, DeviceKernelArgs& kArgs,
                                            const std::vector<RawTensorDataPtr>& inputs,
                                            const std::vector<RawTensorDataPtr>& outputs)
    {
        auto dynAttr = function_->GetDyndevAttribute();
        auto* functionDevProg = reinterpret_cast<DevAscendProgram*>(dynAttr->devProgBinary.data());
        std::vector<DeviceTensorData> evalInputList;
        std::vector<DeviceTensorData> evalOutputList;
        std::tie(evalInputList, evalOutputList) = BuildInputOutputFromHost(memoryHelper, inputs, outputs);
        Evaluator eval{dynAttr->inputSymbolDict, &evalInputList, &evalOutputList};
        auto* cfgProg = reinterpret_cast<DevAscendProgram*>(kArgs.cfgdata);
        cellMatchDescPatches_ = PrepareHostDynamicCellMatchForLaunch(*dynAttr.get(), eval, functionDevProg);
        cfgProg->memBudget = functionDevProg->memBudget;
        PatchRuntimeDynamicCellMatchMeta(memoryHelper, functionDevProg, kArgs);
    }

private:
    Function* function_;
    DeviceLauncherConfig config_;
    std::vector<DevDynamicCellMatchStridePatch> cellMatchDescPatches_;
    std::shared_ptr<CostModel::DynPvModel> pv_;
    std::shared_ptr<CostModel::AiCoreModel> model_;
}; // CostModelLauncher
} // namespace npu::tile_fwk::dynamic
