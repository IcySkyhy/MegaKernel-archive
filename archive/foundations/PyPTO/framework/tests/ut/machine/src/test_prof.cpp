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
 * \file test_prof.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <regex>
#include <fstream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "securec.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/aicore_manager.h"
#include "interface/utils/common.h"
#include "machine/device/tilefwk/aicpu_common.h"

#include <iostream>

using namespace npu::tile_fwk::dynamic;

static void* AllocAligned(size_t alignment, size_t size)
{
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        return nullptr;
    }
    // init as zero to avoid random register values
    (void)memset_s(ptr, size, 0, size);
    return ptr;
}

class TestPro : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}

protected:
    std::unique_ptr<AiCoreManager> CreateAiCoreManager()
    {
        npu::tile_fwk::dynamic::SchThreadStatus status;
        auto aicoreMng = std::make_unique<AiCoreManager>(status);
        aicoreMng->aicNum_ = 1;
        aicoreMng->aivNum_ = 0;
        aicoreMng->aicStart_ = 0;
        aicoreMng->aicEnd_ = 1;
        aicoreMng->aicpuIdx_ = 0;
        return aicoreMng;
    }

    std::unique_ptr<PyPtoMsprofCommandHandle> CreateProfCommandHandle(uint64_t profSwitch = 0, uint32_t type = 0)
    {
        auto data = std::make_unique<PyPtoMsprofCommandHandle>();
        data->profSwitch = profSwitch;
        data->type = type;
        return data;
    }

    struct PmuTestEnv {
        std::unique_ptr<AiCoreManager> aicoreMng;
        std::unique_ptr<AiCoreProf> prof;
        uint8_t* regBuf;
        void* addr;
        int64_t regAddrsArr[1024];
        int64_t pmuEventAddrsArr[10];

        PmuTestEnv(size_t regBufSize)
        {
            npu::tile_fwk::dynamic::SchThreadStatus status;
            aicoreMng = std::make_unique<AiCoreManager>(status);
            aicoreMng->aicNum_ = 1;
            aicoreMng->aivNum_ = 0;
            aicoreMng->aicStart_ = 0;
            aicoreMng->aicEnd_ = 1;
            aicoreMng->aicpuIdx_ = 0;

            const uint32_t pageSize = static_cast<uint32_t>(sysconf(_SC_PAGESIZE));
            regBuf = reinterpret_cast<uint8_t*>(AllocAligned(pageSize, regBufSize));
            addr = reinterpret_cast<void*>(regBuf + 0x100);

            memset_s(regAddrsArr, sizeof(regAddrsArr), 0, sizeof(regAddrsArr));
            regAddrsArr[0] = reinterpret_cast<int64_t>(addr);

            memset_s(pmuEventAddrsArr, sizeof(pmuEventAddrsArr), 0, sizeof(pmuEventAddrsArr));

            prof = std::make_unique<AiCoreProf>(*aicoreMng);
        }

        ~PmuTestEnv()
        {
            if (regBuf != nullptr) {
                free(regBuf);
            }
        }

        void SetupPmuEvents(int eventCount)
        {
            for (int i = 0; i < eventCount; ++i) {
                pmuEventAddrsArr[i] = i + 1;
            }
        }
    };

    struct BasicProfTestEnv {
        std::unique_ptr<AiCoreManager> aicoreMng;
        std::unique_ptr<AiCoreProf> prof;
        int64_t* oriRegAddrs;
        int64_t* regAddrs;

        BasicProfTestEnv()
        {
            npu::tile_fwk::dynamic::SchThreadStatus status;
            aicoreMng = std::make_unique<AiCoreManager>(status);
            aicoreMng->aicNum_ = 0;
            aicoreMng->aivNum_ = 1;
            aicoreMng->aivEnd_ = 1;
            aicoreMng->aicEnd_ = 0;
            aicoreMng->aicpuIdx_ = 0;
            prof = std::make_unique<AiCoreProf>(*aicoreMng);

            oriRegAddrs = reinterpret_cast<int64_t*>(malloc(sizeof(int64_t) * 1024 * 2));
            regAddrs = oriRegAddrs + 1024;
            regAddrs[0] = reinterpret_cast<int64_t>(regAddrs);
        }

        ~BasicProfTestEnv()
        {
            if (oriRegAddrs != nullptr) {
                free(oriRegAddrs);
            }
        }

        void RunProfTest(int iterations = 8)
        {
            ProfConfig profConfig;
            std::unique_ptr<DeviceArgs> devArgs = std::make_unique<DeviceArgs>();
            devArgs->corePmuRegAddr = reinterpret_cast<int64_t>(regAddrs);
            prof->ProfInit(devArgs.get());
            prof->ProfStart();

            int32_t aicoreId = 0;
            TaskStat* taskStat = new TaskStat();
            taskStat->taskId = 0;
            taskStat->execEnd = 1;
            taskStat->execStart = 0;
            taskStat->subGraphId = 0;
            for (int i = 0; i < iterations; i++) {
                prof->ProfGetLog(aicoreId, taskStat);
            }
            int64_t flag = 0;
            prof->ProfGetSwitch(flag);

            prof->ProfStop();
            delete taskStat;
        }
    };
};

TEST_F(TestPro, test_ini)
{
    BasicProfTestEnv env;
    env.RunProfTest();
}

TEST_F(TestPro, test_prof_start_pmu_dav2201)
{
    PmuTestEnv env(0x6000);
    env.SetupPmuEvents(8);

    ProfConfig profConfig;
    std::unique_ptr<DeviceArgs> devArgs = std::make_unique<DeviceArgs>();
    profConfig.Add(ProfConfig::AICORE_PMU);
    devArgs->toSubMachineConfig.profConfig = profConfig;
    env.prof->ProfInit(devArgs.get());
    env.prof->ProfInitPmu(env.regAddrsArr, env.pmuEventAddrsArr);
    env.prof->ProfStartPmu();
    env.prof->ProfGetPmu(0, 0, 0, 0);
    env.prof->ProfStop();
}

TEST_F(TestPro, test_prof_start_pmu_dav3510)
{
    PmuTestEnv env(0x9000);
    env.SetupPmuEvents(10);

    ProfConfig profConfig;
    std::unique_ptr<DeviceArgs> devArgs = std::make_unique<DeviceArgs>();
    profConfig.Add(ProfConfig::AICORE_PMU);
    devArgs->toSubMachineConfig.profConfig = profConfig;
    devArgs->archInfo = ArchInfo::DAV_3510;
    env.prof->ProfInit(devArgs.get());
    env.prof->ProfInitPmu(env.regAddrsArr, env.pmuEventAddrsArr);
    env.prof->ProfStartPmu();
    env.prof->ProfGetPmu(0, 0, 0, 0);
    env.prof->ProfStop();
}
