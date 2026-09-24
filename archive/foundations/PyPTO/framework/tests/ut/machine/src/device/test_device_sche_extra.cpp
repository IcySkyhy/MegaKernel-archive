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
#include <csignal>
#include <cstring>
#include <memory>

#define private public
#define protected public
#include "machine/device/dynamic/device_sche.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(DeviceScheExtraTest, GetSigHandle_AllSignals)
{
    DynMachineManager mgr;
    struct sigaction act {};
    act.sa_handler = SIG_DFL;
    mgr.oriFPEAct_ = act;
    mgr.oriBUSAct_ = act;
    mgr.oriSEGVAct_ = act;
    mgr.oriPIPEAct_ = act;
    mgr.oriILLAct_ = act;
    mgr.oriBordAct_ = act;

    EXPECT_EQ(mgr.GetSigHandle(SIGFPE), SIG_DFL);
    EXPECT_EQ(mgr.GetSigHandle(SIGBUS), SIG_DFL);
    EXPECT_EQ(mgr.GetSigHandle(SIGSEGV), SIG_DFL);
    EXPECT_EQ(mgr.GetSigHandle(SIGPIPE), SIG_DFL);
    EXPECT_EQ(mgr.GetSigHandle(SIGILL), SIG_DFL);
    EXPECT_EQ(mgr.GetSigHandle(SIGABRT), SIG_DFL);
    EXPECT_EQ(mgr.GetSigHandle(SIGUSR1), nullptr);
}

TEST(DeviceScheExtraTest, GetSigHandle_NullHandlers)
{
    DynMachineManager mgr;
    struct sigaction act {};
    act.sa_handler = SIG_IGN;
    mgr.oriFPEAct_ = act;
    mgr.oriBUSAct_ = act;
    mgr.oriSEGVAct_ = act;
    mgr.oriPIPEAct_ = act;
    mgr.oriILLAct_ = act;
    mgr.oriBordAct_ = act;

    EXPECT_EQ(mgr.GetSigHandle(SIGFPE), SIG_IGN);
    EXPECT_EQ(mgr.GetSigHandle(SIGBUS), SIG_IGN);
}

TEST(DeviceScheExtraTest, SigAct_AlreadyReset)
{
    DynMachineManager mgr;
    mgr.reset_.store(true);
    mgr.SigAct(SIGFPE, nullptr, nullptr);
    EXPECT_TRUE(mgr.reset_.load());
}

TEST(DeviceScheExtraTest, Entry_InvalidRunMode)
{
    DynMachineManager mgr;
    DeviceKernelArgs kargs{};
    kargs.parameter.runMode = static_cast<DeviceKernelRunMode>(999);
    DynMachineManager::KernelCtrlEntry entry{};
    int ret = mgr.Entry(&kargs, entry);
    EXPECT_EQ(ret, DEVICE_MACHINE_INVALID_RUN_MODE);
}

TEST(DeviceScheExtraTest, RunSchDeInit)
{
    DynMachineManager mgr;
    mgr.cpumask_.store(0xFF);
    mgr.threadIdxBitmap_.store(0xFF);
    mgr.schExitNum_.store(5);
    mgr.arbitrationLevel_.store(ARBIT_A2A3_SAME_CLUSTER);
    mgr.ctrlWaitLevel_.store(CTRL_WAIT_OK);
    mgr.initSch_.store(true);
    mgr.globalThreadIdx_.store(10);
    mgr.simCpuId_.store(3);

    mgr.RunSchDeInit();

    EXPECT_EQ(mgr.cpumask_.load(), 0u);
    EXPECT_EQ(mgr.threadIdxBitmap_.load(), 0u);
    EXPECT_EQ(mgr.schExitNum_.load(), 0u);
    EXPECT_EQ(mgr.arbitrationLevel_.load(), ARBIT_UNSET);
    EXPECT_EQ(mgr.ctrlWaitLevel_.load(), CTRL_WAIT_UNSET);
    EXPECT_FALSE(mgr.initSch_.load());
    EXPECT_EQ(mgr.globalThreadIdx_.load(), 0);
    EXPECT_EQ(mgr.simCpuId_.load(), 0);
}

TEST(DeviceScheExtraTest, RunSchInit)
{
    DynMachineManager mgr;
    DeviceArgs args{};
    args.scheCpuNum = 2;

    mgr.RunSchInit(&args);
    EXPECT_TRUE(mgr.initSch_.load());

    mgr.RunSchInit(&args);
}

TEST(DeviceScheExtraTest, UpdateScheNumForCtrl)
{
    auto startArgs = std::make_unique<DevStartArgs>();
    DynMachineManager::UpdateScheNumForCtrl(startArgs.get(), 3);
    EXPECT_EQ(startArgs->devCtrlState.arbitratedScehNum.load(), 3u);
}

TEST(DeviceScheExtraTest, DeviceSchedMachine_Init)
{
    DeviceSchedMachine schMachine;
    schMachine.init(2);
    EXPECT_EQ(schMachine.schAicpuNum_, 2u);
}

TEST(DeviceScheExtraTest, DeviceSchedMachine_SetDevSchedSyncMode)
{
    DeviceSchedMachine schMachine;
    schMachine.SetDevSchedSyncMode(1);
    EXPECT_EQ(schMachine.devScheSyncMode_, 1u);
    schMachine.SetDevSchedSyncMode(0);
    EXPECT_EQ(schMachine.devScheSyncMode_, 0u);
}

TEST(DeviceScheExtraTest, DeviceSchedMachine_SetStachSchduleContext)
{
    DeviceSchedMachine schMachine;
    schMachine.init(1);
    SchduleContext ctx;
    schMachine.SetStachSchduleContext(0, &ctx);
}

TEST(DeviceScheExtraTest, DeviceSchedMachine_CheckAndResetReg)
{
    DeviceSchedMachine schMachine;
    schMachine.init(1);
    bool result = schMachine.CheckAndResetReg();
    (void)result;
}

TEST(DeviceScheExtraTest, DeviceSchedMachine_RunThread_InvalidArgs)
{
    DeviceSchedMachine schMachine;
    schMachine.init(1);
    auto startArgs = std::make_unique<DevStartArgs>();
    DeviceArgs args{};
    args.nrAic = 0;
    args.nrValidAic = 0;
    args.nrAicpu = 1;
    args.scheCpuNum = 1;
    int ret = schMachine.RunThread(0, startArgs.get(), &args, 0, 1);
    EXPECT_NE(ret, 0);
}

TEST(DeviceScheExtraTest, DeviceSchedMachine_RunThread_SchedIdxOutOfRange)
{
    DeviceSchedMachine schMachine;
    schMachine.init(1);
    auto startArgs = std::make_unique<DevStartArgs>();
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrValidAic = 1;
    args.nrAicpu = 1;
    args.scheCpuNum = 1;
    int ret = schMachine.RunThread(0, startArgs.get(), &args, 5, 1);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(DeviceScheExtraTest, SignalReg)
{
    DynMachineManager mgr;
    mgr.SignalReg([](int, siginfo_t*, void*) {});
}

TEST(DeviceScheExtraTest, ReCalcDevArgsAicoreNum_Changed)
{
    DynMachineManager mgr;
    DeviceKernelArgs kargs{};
    kargs.parameter.ctrlBlockNum = 4;
    DevAscendProgram devProg{};
    devProg.devArgs.nrValidAic = 8;
    mgr.ReCalcDevArgsAicoreNum(&kargs, &devProg);
    EXPECT_EQ(devProg.devArgs.nrValidAic, 4u);
}

TEST(DeviceScheExtraTest, ReCalcDevArgsAicoreNum_Unchanged)
{
    DynMachineManager mgr;
    DeviceKernelArgs kargs{};
    kargs.parameter.ctrlBlockNum = 0;
    DevAscendProgram devProg{};
    devProg.devArgs.nrValidAic = 8;
    mgr.ReCalcDevArgsAicoreNum(&kargs, &devProg);
    EXPECT_EQ(devProg.devArgs.nrValidAic, 8u);
}

TEST(DeviceScheExtraTest, AllocThreadIdx_UnknownArch)
{
    DeviceArgs devArgs{};
    devArgs.archInfo = ArchInfo::DAV_UNKNOWN;
    std::atomic<int> threadIdx{0};
    std::atomic<uint64_t> cpumask{0};
    int arbitratedScheNum = 0;
    std::atomic<int> arbitrationLevel{ARBIT_UNSET};
    std::atomic<int> simCpuId{0};
    std::atomic<uint64_t> arbitrationCpumask{0};
    std::atomic<uint64_t> threadIdxBitmap{0};
    int curThreadIdx = -1;

    int ret = DynMachineManager::AllocThreadIdx(&devArgs, curThreadIdx, threadIdx, cpumask, arbitratedScheNum,
                                                arbitrationLevel, simCpuId, arbitrationCpumask, threadIdxBitmap);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(curThreadIdx, 1);
}
