/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <unistd.h>
#include "securec.h"

#define private public
#define protected public
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicore_manager.h"

using namespace npu::tile_fwk::dynamic;

static int32_t MockReportFunc(uint32_t, const VOID_PTR, uint32_t) { return 0; }

namespace {
struct ProfLogTestEnv {
    SchThreadStatus status;
    std::unique_ptr<AiCoreManager> aicoreMng;
    std::unique_ptr<AiCoreProf> prof;

    ProfLogTestEnv()
    {
        status.Init();
        aicoreMng = std::make_unique<AiCoreManager>(status);
        aicoreMng->aicNum_ = 1;
        aicoreMng->aivNum_ = 0;
        aicoreMng->aicStart_ = 0;
        aicoreMng->aicEnd_ = 1;
        aicoreMng->aicpuIdx_ = 0;
        prof = std::make_unique<AiCoreProf>(*aicoreMng);

        prof->profLevel_ = PROF_LEVEL_FUNC_LOG;
        prof->coreNum_ = 1;
        prof->profReportAdditionalInfoFunc_ = MockReportFunc;

        prof->logMsgSize_ = sizeof(PyPtoMsprofAdditionalInfo);
        prof->logHeadSize_ = sizeof(MsprofAicpuPyPtoLogHead);
        prof->logDataSize_ = sizeof(MsprofAicpuPyPtoLogData);
        prof->logMsg_.resize(1);
        prof->logHead_.resize(1, nullptr);
        prof->logData_.resize(1, nullptr);
        prof->logHead_[0] = reinterpret_cast<MsprofAicpuPyPtoLogHead*>(&prof->logMsg_[0].data);
        prof->logData_[0] = reinterpret_cast<MsprofAicpuPyPtoLogData*>(reinterpret_cast<uintptr_t>(prof->logHead_[0]) +
                                                                       prof->logHeadSize_);
        prof->logHead_[0]->cnt = 0;
        prof->logMsg_[0].magicNumber = 0x5A5AU;
        prof->logMsg_[0].level = PYPTO_MSPROF_REPORT_AICPU_LEVEL;
        prof->logMsg_[0].type = PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE;
        prof->logMsg_[0].threadId = 0;
        prof->logMsg_[0].dataLen = prof->logHeadSize_;
        prof->logHead_[0]->magicNumber = 0x6BD3U;
        prof->logHead_[0]->coreId = 0;
        prof->logHead_[0]->coreType = 0;
        prof->logHead_[0]->dataType = PROF_DATATYPE_LOG;
        prof->logHead_[0]->taskId = 0;
        prof->logHead_[0]->streamId = 0;
    }
};

struct ProfPmuTestEnv {
    SchThreadStatus status;
    std::unique_ptr<AiCoreManager> aicoreMng;
    std::unique_ptr<AiCoreProf> prof;
    uint8_t* regBuf{nullptr};
    int64_t regAddrsArr[1024]{};
    int64_t pmuEventAddrsArr[10]{};
    uint32_t pmuCntMem[20]{};
    uint32_t pmuCntTotalMem[4]{};

    ProfPmuTestEnv(size_t regBufSize = 0x6000, ArchInfo arch = ArchInfo::DAV_2201)
    {
        status.Init();
        aicoreMng = std::make_unique<AiCoreManager>(status);
        aicoreMng->aicNum_ = 1;
        aicoreMng->aivNum_ = 0;
        aicoreMng->aicStart_ = 0;
        aicoreMng->aicEnd_ = 1;
        aicoreMng->aicpuIdx_ = 0;

        const uint32_t pageSize = static_cast<uint32_t>(sysconf(_SC_PAGESIZE));
        regBuf = reinterpret_cast<uint8_t*>(aligned_alloc(pageSize, regBufSize));
        memset(regBuf, 0, regBufSize);
        void* addr = reinterpret_cast<void*>(regBuf + 0x100);
        regAddrsArr[0] = reinterpret_cast<int64_t>(addr);

        for (int i = 0; i < 10; ++i) {
            pmuEventAddrsArr[i] = i + 1;
        }

        prof = std::make_unique<AiCoreProf>(*aicoreMng);
        prof->archInfo_ = arch;
        prof->coreNum_ = 1;
        prof->profReportAdditionalInfoFunc_ = MockReportFunc;
        prof->profLevel_ = PROF_LEVEL_FUNC_LOG_PMU;

        ProfConfig profConfig;
        profConfig.Add(ProfConfig::AICORE_PMU);
        DeviceArgs devArgs{};
        devArgs.toSubMachineConfig.profConfig = profConfig;
        devArgs.archInfo = arch;
        devArgs.corePmuRegAddr = reinterpret_cast<int64_t>(regAddrsArr);
        prof->ProfInit(&devArgs);
        prof->ProfInitPmu(regAddrsArr, pmuEventAddrsArr);
    }

    ~ProfPmuTestEnv()
    {
        if (regBuf)
            free(regBuf);
    }
};
} // namespace

TEST(AicoreProfExtraTest, ProfGetLog_SingleEntry)
{
    ProfLogTestEnv env;
    TaskStat stat{};
    stat.taskId = 1;
    stat.execStart = 100;
    stat.execEnd = 200;

    env.prof->ProfGetLog(0, &stat);
}

TEST(AicoreProfExtraTest, ProfGetLog_FillAndFlush)
{
    ProfLogTestEnv env;
    TaskStat stat{};
    stat.taskId = 1;
    stat.execStart = 0;
    stat.execEnd = 1;

    for (uint32_t i = 0; i < env.prof->logDataMaxNum_; i++) {
        stat.taskId = static_cast<int32_t>(i);
        env.prof->ProfGetLog(0, &stat);
    }
}

TEST(AicoreProfExtraTest, ProfGetLog_MultipleBelowMax)
{
    ProfLogTestEnv env;
    TaskStat stat{};
    stat.execStart = 0;
    stat.execEnd = 1;

    for (uint32_t i = 0; i < env.prof->logDataMaxNum_ - 1; i++) {
        stat.taskId = static_cast<int32_t>(i);
        env.prof->ProfGetLog(0, &stat);
    }
}

TEST(AicoreProfExtraTest, ProfStopLog_WithData)
{
    ProfLogTestEnv env;
    TaskStat stat{};
    stat.taskId = 1;
    stat.execStart = 0;
    stat.execEnd = 1;
    env.prof->ProfGetLog(0, &stat);

    env.prof->ProfStop();
}

TEST(AicoreProfExtraTest, ProfStopLog_Empty)
{
    ProfLogTestEnv env;
    env.prof->ProfStop();
}

TEST(AicoreProfExtraTest, ProfGetSwitch_Log)
{
    ProfLogTestEnv env;
    int64_t flag = 0;
    env.prof->ProfGetSwitch(flag);
    EXPECT_EQ(flag & 0x1, 1);
}

TEST(AicoreProfExtraTest, ProfGetSwitch_Pmu)
{
    ProfPmuTestEnv env;
    int64_t flag = 0;
    env.prof->ProfGetSwitch(flag);
    EXPECT_EQ(flag & 0x3, 3);
}

TEST(AicoreProfExtraTest, FillPmuData_Dav2201)
{
    ProfPmuTestEnv env;
    int32_t coreIdx = 0;
    uint32_t subGraphId = 0;
    uint32_t taskId = 1;

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i + 10;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i + 100;

    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    MsprofAicpuPyPtoPmuData data{};
    env.prof->FillPmuData(data, coreIdx, subGraphId, taskId, 42);
    EXPECT_EQ(data.seqNo, 42u);
    EXPECT_EQ(data.taskId, 1u);
    EXPECT_EQ(data.pmuCnt0, 10u);
}

TEST(AicoreProfExtraTest, FillPmuData_Dav3510)
{
    ProfPmuTestEnv env(0x9000, ArchInfo::DAV_3510);
    int32_t coreIdx = 0;
    uint32_t subGraphId = 0;
    uint32_t taskId = 2;

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i + 20;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i + 200;

    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    MsprofAicpuPyPtoPmuData data{};
    env.prof->FillPmuData(data, coreIdx, subGraphId, taskId, 99);
    EXPECT_EQ(data.pmuCnt8, 28u);
    EXPECT_EQ(data.pmuCnt9, 29u);
}

TEST(AicoreProfExtraTest, DebugPmuData)
{
    ProfPmuTestEnv env;
    int32_t coreIdx = 0;

    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];

    MsprofAicpuPyPtoPmuData data{};
    data.seqNo = 1;
    data.taskId = 2;
    data.totalCyc = 1000;
    env.prof->DebugPmuData(coreIdx, data);
}

TEST(AicoreProfExtraTest, ProfGetPmu_FirstEntry)
{
    ProfPmuTestEnv env;
    int32_t coreIdx = 0;

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i;

    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    env.prof->pmuHead_[coreIdx]->cnt = 0;
    env.prof->ProfGetPmu(coreIdx, 0, 1, 42);
}

TEST(AicoreProfExtraTest, ProfGetPmu_MiddleEntry)
{
    ProfPmuTestEnv env;
    int32_t coreIdx = 0;

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i;

    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    env.prof->pmuHead_[coreIdx]->cnt = 1;
    env.prof->ProfGetPmu(coreIdx, 0, 2, 43);
}

TEST(AicoreProfExtraTest, ProfGetPmu_FlushAtMax)
{
    ProfPmuTestEnv env;
    int32_t coreIdx = 0;

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i;

    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    env.prof->pmuHead_[coreIdx]->cnt = 1;
    env.prof->ProfGetPmu(coreIdx, 0, 3, 44);
}

TEST(AicoreProfExtraTest, ProfStopPmu_RestoreCtrl0)
{
    ProfPmuTestEnv env;
    uint32_t ctrl0Mem = 0xABCD;
    env.prof->addrs_.ctrl0Addr = &ctrl0Mem;
    env.prof->ctrl0Val_ = 0x1234;
    env.prof->pmuHead_[0]->cnt = 0;
    env.prof->ProfStopPmu();
    EXPECT_EQ(ctrl0Mem, 0x1234u);
}

TEST(AicoreProfExtraTest, ProfStopPmu_Dav3510_RestoreCtrl)
{
    ProfPmuTestEnv env(0x9000, ArchInfo::DAV_3510);
    uint32_t ctrl0Mem = 0;
    uint32_t ctrl1Mem = 0;
    env.prof->addrs_.ctrl0Addr = &ctrl0Mem;
    env.prof->addrs_.ctrl1Addr = &ctrl1Mem;
    env.prof->ctrl0Val_ = 0xAAAA;
    env.prof->ctrl1Val_ = 0xBBBB;
    env.prof->pmuHead_[0]->cnt = 0;
    env.prof->ProfStopPmu();
    EXPECT_EQ(ctrl0Mem, 0xAAAAu);
    EXPECT_EQ(ctrl1Mem, 0xBBBBu);
}

TEST(AicoreProfExtraTest, ProfGetCurCpuTimestamp)
{
    ProfLogTestEnv env;
    uint64_t ts = env.prof->ProfGetCurCpuTimestamp();
    (void)ts;
}

TEST(AicoreProfExtraTest, AsmCntvc)
{
    ProfLogTestEnv env;
    uint64_t cntvct = 999;
    env.prof->AsmCntvc(cntvct);
#if defined(__aarch64__)
    (void)cntvct;
#else
    EXPECT_EQ(cntvct, 0u);
#endif
}

TEST(AicoreProfExtraTest, CreateProfLevel_AllBranches)
{
    ProfConfig cfg1;
    cfg1.Add(ProfConfig::AICORE_PMU);
    EXPECT_EQ(CreateProfLevel(cfg1), PROF_LEVEL_FUNC_LOG_PMU);

    ProfConfig cfg2;
    cfg2.Add(ProfConfig::AICORE_TIME);
    EXPECT_EQ(CreateProfLevel(cfg2), PROF_LEVEL_FUNC_LOG);

    ProfConfig cfg3;
    cfg3.Add(ProfConfig::AICPU_FUNC);
    EXPECT_EQ(CreateProfLevel(cfg3), PROF_LEVEL_FUNC);

    ProfConfig cfg4;
    EXPECT_EQ(CreateProfLevel(cfg4), PROF_LEVEL_OFF);
}

TEST(AicoreProfExtraTest, ProfStart_Off)
{
    ProfLogTestEnv env;
    env.prof->profLevel_ = PROF_LEVEL_OFF;
    env.prof->ProfStart();
}

TEST(AicoreProfExtraTest, ProfStart_Log)
{
    ProfLogTestEnv env;
    env.prof->profLevel_ = PROF_LEVEL_FUNC_LOG;
    env.prof->ProfStart();
}

static bool g_enableAdprofCheck = false;
static bool g_enableAdprofReport = false;

extern "C" int32_t AdprofCheckFeatureIsOn(uint64_t) { return g_enableAdprofCheck ? 1 : 0; }

extern "C" int32_t AdprofReportAdditionalInfo(uint32_t, const VOID_PTR, uint32_t)
{
    return g_enableAdprofReport ? 0 : -1;
}

struct AdprofGuard {
    bool prevCheck;
    bool prevReport;
    AdprofGuard(bool check, bool report) : prevCheck(g_enableAdprofCheck), prevReport(g_enableAdprofReport)
    {
        g_enableAdprofCheck = check;
        g_enableAdprofReport = report;
    }
    ~AdprofGuard()
    {
        g_enableAdprofCheck = prevCheck;
        g_enableAdprofReport = prevReport;
    }
};

TEST(AicoreProfExtraTest, ProfCheckLevel_Disabled)
{
    AdprofGuard guard(false, false);
    EXPECT_FALSE(ProfCheckLevel(PROF_TASK_TIME_L2));
}

TEST(AicoreProfExtraTest, ProfCheckLevel_Enabled)
{
    AdprofGuard guard(true, false);
    EXPECT_TRUE(ProfCheckLevel(PROF_TASK_TIME_L2));
}

TEST(AicoreProfExtraTest, ProfStop_WithLogDataEnabled)
{
    ProfLogTestEnv env;
    AdprofGuard guard(true, true);

    TaskStat stat{};
    stat.taskId = 1;
    stat.execStart = 0;
    stat.execEnd = 1;
    env.prof->ProfGetLog(0, &stat);
    EXPECT_EQ(env.prof->logHead_[0]->cnt, 1u);

    env.prof->ProfStop();
    EXPECT_EQ(env.prof->logHead_[0]->cnt, 0u);
}

TEST(AicoreProfExtraTest, ProfStop_EmptyLogEnabled)
{
    ProfLogTestEnv env;
    AdprofGuard guard(true, true);
    env.prof->ProfStop();
}

TEST(AicoreProfExtraTest, ProfGetLog_Enabled)
{
    ProfLogTestEnv env;
    AdprofGuard guard(true, true);

    TaskStat stat{};
    stat.taskId = 42;
    stat.execStart = 100;
    stat.execEnd = 200;
    env.prof->ProfGetLog(0, &stat);
    EXPECT_EQ(env.prof->logHead_[0]->cnt, 1u);
    EXPECT_EQ(env.prof->taskCnt_, 1u);
}

TEST(AicoreProfExtraTest, ProfGetLog_FillAndFlushEnabled)
{
    ProfLogTestEnv env;
    AdprofGuard guard(true, true);

    TaskStat stat{};
    stat.execStart = 0;
    stat.execEnd = 1;
    for (uint32_t i = 0; i < env.prof->logDataMaxNum_; i++) {
        stat.taskId = static_cast<int32_t>(i);
        env.prof->ProfGetLog(0, &stat);
    }
    EXPECT_EQ(env.prof->logHead_[0]->cnt, 0u);
}

TEST(AicoreProfExtraTest, ProfGetLog_MultipleBelowMaxEnabled)
{
    ProfLogTestEnv env;
    AdprofGuard guard(true, true);

    TaskStat stat{};
    stat.execStart = 0;
    stat.execEnd = 1;
    for (uint32_t i = 0; i < env.prof->logDataMaxNum_ - 1; i++) {
        stat.taskId = static_cast<int32_t>(i);
        env.prof->ProfGetLog(0, &stat);
    }
    EXPECT_EQ(env.prof->logHead_[0]->cnt, env.prof->logDataMaxNum_ - 1);
}

TEST(AicoreProfExtraTest, ProfStopPmu_WithDataEnabled)
{
    ProfPmuTestEnv env;
    AdprofGuard guard(true, true);

    uint32_t ctrl0Mem = 0;
    env.prof->addrs_.ctrl0Addr = &ctrl0Mem;
    env.prof->ctrl0Val_ = 0x1234;

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i;
    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    env.prof->pmuHead_[0]->cnt = 1;
    env.prof->ProfStopPmu();
    EXPECT_EQ(ctrl0Mem, 0x1234u);
    EXPECT_EQ(env.prof->pmuHead_[0]->cnt, 0u);
}

TEST(AicoreProfExtraTest, ProfGetPmu_EnabledFirstEntry)
{
    ProfPmuTestEnv env;
    AdprofGuard guard(true, true);

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i;
    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    env.prof->pmuHead_[0]->cnt = 0;
    env.prof->ProfGetPmu(0, 0, 1, 42);
    EXPECT_EQ(env.prof->pmuHead_[0]->cnt, 1u);
}

TEST(AicoreProfExtraTest, ProfGetPmu_EnabledMiddleEntry)
{
    ProfPmuTestEnv env;
    AdprofGuard guard(true, true);

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i;
    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    env.prof->pmuHead_[0]->cnt = 1;
    env.prof->ProfGetPmu(0, 0, 2, 43);
    EXPECT_EQ(env.prof->pmuHead_[0]->cnt, 2u);
}

TEST(AicoreProfExtraTest, ProfGetPmu_EnabledFlushAtMax)
{
    ProfPmuTestEnv env;
    AdprofGuard guard(true, true);

    for (int i = 0; i < 20; i++)
        env.pmuCntMem[i] = i;
    for (int i = 0; i < 4; i++)
        env.pmuCntTotalMem[i] = i;
    env.prof->pmuCnt0Plain_[0] = &env.pmuCntMem[0];
    env.prof->pmuCnt1Plain_[0] = &env.pmuCntMem[1];
    env.prof->pmuCnt2Plain_[0] = &env.pmuCntMem[2];
    env.prof->pmuCnt3Plain_[0] = &env.pmuCntMem[3];
    env.prof->pmuCnt4Plain_[0] = &env.pmuCntMem[4];
    env.prof->pmuCnt5Plain_[0] = &env.pmuCntMem[5];
    env.prof->pmuCnt6Plain_[0] = &env.pmuCntMem[6];
    env.prof->pmuCnt7Plain_[0] = &env.pmuCntMem[7];
    env.prof->pmuCnt8Plain_[0] = &env.pmuCntMem[8];
    env.prof->pmuCnt9Plain_[0] = &env.pmuCntMem[9];
    env.prof->pmuCntTotal0Plain_[0] = &env.pmuCntTotalMem[0];
    env.prof->pmuCntTotal1Plain_[0] = &env.pmuCntTotalMem[1];

    env.prof->pmuHead_[0]->cnt = env.prof->pmuDataMaxNum_ - 1;
    env.prof->ProfGetPmu(0, 0, 3, 44);
    EXPECT_EQ(env.prof->pmuHead_[0]->cnt, 0u);
}

TEST(AicoreProfExtraTest, ProfInit_WithAdprofReport)
{
    AdprofGuard guard(false, true);
    SchThreadStatus status;
    status.Init();
    auto aicoreMng = std::make_unique<AiCoreManager>(status);
    aicoreMng->aicNum_ = 1;
    aicoreMng->aivNum_ = 0;
    aicoreMng->aicStart_ = 0;
    aicoreMng->aicEnd_ = 1;
    aicoreMng->aicpuIdx_ = 0;
    auto prof = std::make_unique<AiCoreProf>(*aicoreMng);

    ProfConfig profConfig;
    DeviceArgs devArgs{};
    devArgs.toSubMachineConfig.profConfig = profConfig;
    devArgs.archInfo = ArchInfo::DAV_2201;
    prof->ProfInit(&devArgs);
    EXPECT_EQ(prof->profReportAdditionalInfoFunc_, AdprofReportAdditionalInfo);
}

TEST(AicoreProfExtraTest, ProfInit_WithoutAdprofReport)
{
    SchThreadStatus status;
    status.Init();
    auto aicoreMng = std::make_unique<AiCoreManager>(status);
    aicoreMng->aicNum_ = 1;
    aicoreMng->aivNum_ = 0;
    aicoreMng->aicStart_ = 0;
    aicoreMng->aicEnd_ = 1;
    aicoreMng->aicpuIdx_ = 0;
    auto prof = std::make_unique<AiCoreProf>(*aicoreMng);

    ProfConfig profConfig;
    DeviceArgs devArgs{};
    devArgs.toSubMachineConfig.profConfig = profConfig;
    devArgs.archInfo = ArchInfo::DAV_2201;
    prof->ProfInit(&devArgs);
    EXPECT_NE(prof->profReportAdditionalInfoFunc_, nullptr);
}

TEST(AicoreProfExtraTest, ReadPmuCounters_UnknownArch)
{
    ProfLogTestEnv env;
    env.prof->archInfo_ = ArchInfo::DAV_UNKNOWN;
    env.prof->ReadPmuCounters(0);
}

TEST(AicoreProfExtraTest, SetPmuEvents_UnknownArch)
{
    ProfLogTestEnv env;
    env.prof->archInfo_ = ArchInfo::DAV_UNKNOWN;
    uint8_t dummyMem[256] = {0};
    env.prof->SetPmuEvents(dummyMem, 0);
}

TEST(AicoreProfExtraTest, ProgramPmuStartForCore_UnknownArch)
{
    ProfLogTestEnv env;
    env.prof->archInfo_ = ArchInfo::DAV_UNKNOWN;
    uint8_t dummyMem[256] = {0};
    AiCoreProf::PmuCtrlAddrs addrs{};
    uint32_t ctrl0 = 0, ctrl1 = 0, start0 = 0, start1 = 0, stop0 = 0, stop1 = 0;
    addrs.ctrl0Addr = &ctrl0;
    addrs.ctrl1Addr = &ctrl1;
    addrs.startCntCyc0Addr = &start0;
    addrs.startCntCyc1Addr = &start1;
    addrs.stopCntCyc0Addr = &stop0;
    addrs.stopCntCyc1Addr = &stop1;
    env.prof->ProgramPmuStartForCore(dummyMem, 0, addrs);
}
