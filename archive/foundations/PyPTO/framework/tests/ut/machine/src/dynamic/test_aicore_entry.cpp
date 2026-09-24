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
 * \file test_aicore_entry.h
 * \brief
 */

#include "interface/utils/string_utils.h"
#include "interface/utils/common.h"
#include "machine/device/tilefwk/aicpu_common.h"
#include "aicore_emulation.h"
#include "test_machine_common.h"

using namespace npu::tile_fwk::machine;

struct AicoreTest : UnitTestBase {};

TEST_F(AicoreTest, InitGoodbye)
{
    KernelSharedBuffer sharedBuffer[0x1];
    memset_s(sharedBuffer, sizeof(sharedBuffer), 0, sizeof(sharedBuffer));
    KernelArgs* args = reinterpret_cast<KernelArgs*>(&sharedBuffer);
    args->waveBufferCpuToCore[0] = AICORE_SAY_GOODBYE;

    DeviceArgs devArgs;
    memset_s(&devArgs, sizeof(devArgs), 0, sizeof(devArgs));
    devArgs.sharedBuffer = (uint64_t)(uintptr_t)&sharedBuffer;
    std::unique_ptr<DevDfxArgs> devDfxArgs = std::make_unique<DevDfxArgs>();
    devArgs.devDfxArgAddr = (uint64_t)(uintptr_t)devDfxArgs.get();

    KernelEntry(0, 0, 0, 0, 0, (uint64_t)(uintptr_t)&devArgs);
    // Use AICORE_SAY_GOODBYE to exit
    EXPECT_EQ(args->dfxBuffer[3], STAGE_GET_PARALLEL_DEVTASK_TIMEOUT);
}

class MemoryEmulation {
public:
    MemoryEmulation(int aicpuCount, int aicCount, int aivCount)
        : aicpuCount_(aicpuCount), aicCount_(aicCount), aivCount_(aivCount)
    {}

    void Setup()
    {
        int sharedBufferSize = (aicCount_ + aivCount_) * sizeof(KernelSharedBuffer);
        sharedBuffer_.resize(aicCount_ + aivCount_);
        memset_s(sharedBuffer_.data(), sharedBufferSize, 0, sharedBufferSize);
        printBuffer_.resize((aicCount_ + aivCount_) * PRINT_BUFFER_SIZE);
        memset_s(printBuffer_.data(), printBuffer_.size(), 0, printBuffer_.size());
    }

    KernelSharedBuffer* GetSharedBuffer() { return sharedBuffer_.data(); }
    uint8_t* GetPrintBuffer(int idx) { return printBuffer_.data() + idx * PRINT_BUFFER_SIZE; }
    int GetAicpuCount() const { return aicpuCount_; }
    int GetAicCount() const { return aicCount_; }
    int GetAivCount() const { return aivCount_; }

private:
    int aicpuCount_;
    int aicCount_;
    int aivCount_;
    std::vector<KernelSharedBuffer> sharedBuffer_;
    std::vector<uint8_t> printBuffer_;
};

struct MultipleCore : ThreadAicoreEmulation {
    MultipleCore(std::shared_ptr<MemoryEmulation> memory_) : memory(memory_)
    {
        memset_s(&devArgs, sizeof(devArgs), 0, sizeof(devArgs));
        devArgs.sharedBuffer = (uint64_t)(uintptr_t)memory->GetSharedBuffer();
        devDfxArgs = std::make_unique<DevDfxArgs>();
        devArgs.devDfxArgAddr = (uint64_t)(uintptr_t)devDfxArgs.get();
        KernelSharedBuffer* buffer = memory->GetSharedBuffer();
        for (int i = 0; i < memory->GetAicCount() + memory->GetAivCount(); i++) {
            buffer[i].args.dfxBuffer[SHAK_BUF_PRINT_BUFFER_INDEX] = (uintptr_t)memory->GetPrintBuffer(i);
        }
    }

    void WaitAndStartKernelEntry()
    {
        while (GetAicoreInfoByThread() == nullptr) {
            std::this_thread::sleep_for(std::chrono::seconds(0));
        }
        KernelEntry(0, 0, 0, 0, 0, (uint64_t)(uintptr_t)&devArgs);
    }

    static void StartKernelEntry(std::shared_ptr<MultipleCore> aicore) { aicore->WaitAndStartKernelEntry(); }

    virtual void AicoreCallSubFuncTask(uint64_t funcIdx, CoreFuncParam* param, int64_t gmStackAddr,
                                       __gm__ int64_t* hcclContext, TaskStat* taskStat) override
    {
        UNUSED(funcIdx);
        UNUSED(param);
        UNUSED(gmStackAddr);
        UNUSED(hcclContext);
        UNUSED(taskStat);
        std::lock_guard<std::mutex> guard(traceMutex);
        traceList.emplace_back(GetAicoreInfoByThread()->GetCoreIdx(), funcIdx, *param);
    }

    void MainLoop()
    {
        const int rootCount = 0x2;
        const int attrCount = 0x100;
        const int leafCount = 0x40;
        const uint64_t attrListBase = 0x100000000;
        for (size_t k = 0; k < attrCount; k++) {
            devFuncAttrList.push_back(attrListBase + k);
        }
        for (size_t k = 0; k < leafCount; k++) {
            devFuncAttrOffsetList.push_back(k * attrCount / leafCount);
        }
        for (size_t k = 0; k < rootCount; k++) {
            devFuncExprTbl.push_back(0);
        }

        dynFuncDataList.resize(sizeof(DynFuncHeader) + sizeof(DynFuncData) * rootCount);
        dynFuncBinList.resize(rootCount);
        DynFuncHeader* dataList = reinterpret_cast<DynFuncHeader*>(dynFuncDataList.data());
        dataList->funcNum = rootCount;
        dataList->seqNo = 0;
        dataList->cceBinary = dynFuncBinList.data();
        dataList->startArgs = &startArgs;
        for (size_t k = 0; k < dataList->funcNum; k++) {
            dataList->At(k).opAttrs = devFuncAttrList.data();
            dataList->At(k).opAtrrOffsets = devFuncAttrOffsetList.data();
            dataList->At(k).exprTbl = reinterpret_cast<uint64_t*>(devFuncExprTbl.data());
        }

        for (size_t k = 0; k < attrCount; k++) {
            devFuncAttrList[k] = 0;
        }

        KernelSharedBuffer* buffer = memory->GetSharedBuffer();
        for (int i = 0; i < memory->GetAicCount() + memory->GetAivCount(); i++) {
            buffer[i].args.parallelDevTask.front = 0;
            buffer[i].args.parallelDevTask.rear = 1;
            buffer[i].args.parallelDevTask.ptrElements[0] = (uintptr_t)dataList;

            buffer[i].args.dfxBuffer[SHAK_BUF_PRINT_BUFFER_INDEX] = (uintptr_t)memory->GetPrintBuffer(i);
        }

        for (int i = 0; i < memory->GetAicCount() + memory->GetAivCount(); i++) {
            AicpuSetData(i, MakeTaskID(i % 0x2, i) + 1);
        }

        for (int i = 0; i < memory->GetAicCount() + memory->GetAivCount(); i++) {
            while (AicpuGetCond(i) == AICORE_TASK_INIT || (AicpuGetCond(i) & AICORE_FIN_MASK) == 0) {
                std::this_thread::sleep_for(std::chrono::seconds(0));
            }
        }

        for (int i = 0; i < memory->GetAicCount() + memory->GetAivCount(); i++) {
            AicpuSetData(i, AICORE_TASK_STOP + 1);
        }

        for (int i = 0; i < memory->GetAicCount() + memory->GetAivCount(); i++) {
            buffer[i].args.waveBufferCpuToCore[0] = AICORE_SAY_GOODBYE;
        }
    }

    DynFuncHeader* GetDataList() { return reinterpret_cast<DynFuncHeader*>(dynFuncDataList.data()); }

public:
    std::shared_ptr<MemoryEmulation> memory;
    DeviceArgs devArgs;
    std::unique_ptr<DevDfxArgs> devDfxArgs;
    std::vector<uint8_t> dynFuncDataList;
    std::vector<DynFuncBin> dynFuncBinList;
    std::vector<uint64_t> devFuncAttrList;
    std::vector<int32_t> devFuncAttrOffsetList;
    std::vector<uint64_t> devFuncExprTbl;
    DevStartArgsBase startArgs;

    std::mutex traceMutex;
    struct Trace {
        int coreIdx;
        int funcIdx;
        CoreFuncParam param;
        Trace(int coreIdx_, int funcIdx_, CoreFuncParam param_) : coreIdx(coreIdx_), funcIdx(funcIdx_), param(param_) {}
    };
    std::vector<Trace> traceList;
};

TEST_F(AicoreTest, MultipleCore)
{
    const int aicpuCount = 0x4;
    const int aicCount = 0x0;
    const int aivCount = 0x4;
    std::shared_ptr<MemoryEmulation> memory = std::make_shared<MemoryEmulation>(aicpuCount, aicCount, aivCount);
    memory->Setup();

    auto aicore = std::make_shared<MultipleCore>(memory);
    AicoreEmulationManager::GetInstance().SetupAicoreEmulation(std::static_pointer_cast<AicoreEmulationBase>(aicore));

    std::vector<std::shared_ptr<std::thread>> threadList;
    for (int i = 0; i < aicCount + aivCount; i++) {
        std::shared_ptr<std::thread> thread = std::make_shared<std::thread>(MultipleCore::StartKernelEntry, aicore);
        aicore->AppendAicore(thread, i, i);
        threadList.push_back(thread);
    }
    aicore->MainLoop();
    for (int i = 0; i < aicCount + aivCount; i++) {
        threadList[i]->join();
    }

    KernelSharedBuffer* buffer = memory->GetSharedBuffer();
    for (int i = 0; i < memory->GetAicCount() + memory->GetAivCount(); i++) {
        // normal exit
        EXPECT_EQ(buffer[i].args.dfxBuffer[2], STAGE_CORE_EXIT);
    }

    AicoreEmulationManager::GetInstance().Reset();
}
