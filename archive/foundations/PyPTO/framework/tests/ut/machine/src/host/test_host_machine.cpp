/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_host_machine.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "tilefwk/pypto_fwk_log.h"

#define private public
#include "interface/machine/host/host_machine.h"
#undef private

using namespace npu::tile_fwk;

extern "C" {
struct Backend {
    void* runPass;
    void* getResumePath;
    void* execute;
    void* simuExecute;
    void* platform;
    void* matchCache;

    static Backend& GetBackend();
};
}

class TestHostMachineLog : public testing::Test {
public:
    void SetUp() override
    {
        auto& hm = HostMachine::GetInstance();
        if (!hm.initialized_.load()) {
            hm.Init(HostMachineMode::API);
        }
        hm.curTask = nullptr;
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        auto& hm = HostMachine::GetInstance();
        hm.DestroyThread();
        if (hm.curTask != nullptr) {
            delete hm.curTask;
            hm.curTask = nullptr;
        }
        hm.curTaskId_ = 0;
        hm.finishQueue_.Clear();
        hm.compileQueue_.Clear();
        hm.agentQueue_.Clear();
        hm.stashedFuncQueue_.Clear();
        Program::GetInstance().Reset();
        Program::GetInstance().SetLastFunction(nullptr);
    }
};

TEST_F(TestHostMachineLog, SubTask_CurTaskAlreadyRunning)
{
    auto& hm = HostMachine::GetInstance();
    hm.mode_ = HostMachineMode::API;

    hm.SubTask(nullptr);
    EXPECT_NE(hm.curTask, nullptr);
    MachineTask* firstTask = hm.curTask;

    hm.SubTask(nullptr);
    EXPECT_NE(hm.curTask, nullptr);
    EXPECT_NE(hm.curTask, firstTask);

    delete firstTask;
}

TEST_F(TestHostMachineLog, Compile_NullTaskWhenCurTaskNull)
{
    auto& hm = HostMachine::GetInstance();
    hm.mode_ = HostMachineMode::API;
    hm.curTask = nullptr;

    MachineTask* result = hm.Compile(nullptr);
    EXPECT_EQ(result, nullptr);
}

TEST_F(TestHostMachineLog, MachineTask_SetFunctionAndSetError)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "TENSOR_test", nullptr);
    MachineTask task(0, nullptr);
    task.SetFunction(func.get());
    EXPECT_EQ(task.GetFunction(), func.get());
    task.SetError("test error msg");
    EXPECT_EQ(task.Error(), "test error msg");
}

TEST_F(TestHostMachineLog, WaitTaskFinish_ThrowsOnError)
{
    auto& hm = HostMachine::GetInstance();
    hm.mode_ = HostMachineMode::API;
    auto task = std::make_unique<MachineTask>(0, nullptr);
    task->SetError("compile failed");
    hm.finishQueue_.Push(std::move(task));
    hm.curTaskId_ = 1;
    EXPECT_THROW(hm.WaitTaskFinish(), std::runtime_error);
}

TEST_F(TestHostMachineLog, StashTask_NullFunction_ReturnsEarly)
{
    auto& hm = HostMachine::GetInstance();
    hm.StashTask(nullptr);
    EXPECT_EQ(hm.stashedFuncQueue_.Size(), 0U);
}

TEST_F(TestHostMachineLog, GetCacheKeyFromFunction_NullAndNonDynamic)
{
    EXPECT_TRUE(HostMachine::GetCacheKeyFromFunction(nullptr).empty());

    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "TENSOR_test", nullptr);
    Program::GetInstance().SetLastFunction(nullptr);
    auto key = HostMachine::GetCacheKeyFromFunction(func.get());
    EXPECT_EQ(key, func->GetFunctionHash().Data());
}

TEST_F(TestHostMachineLog, GetCacheKeyFromFunction_DynamicLastFunc)
{
    auto dynFunc = std::make_shared<Function>(Program::GetInstance(), "dyn", "TENSOR_dyn", nullptr);
    dynFunc->SetFunctionType(FunctionType::DYNAMIC);
    dynFunc->functionHash_ = FunctionHash(42);
    Program::GetInstance().SetLastFunction(dynFunc.get());

    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "TENSOR_test", nullptr);
    auto key = HostMachine::GetCacheKeyFromFunction(func.get());
    EXPECT_EQ(key, std::to_string(42U));
}

TEST_F(TestHostMachineLog, Compile_NoResumeFile_CallsCompileFunction)
{
    auto& hm = HostMachine::GetInstance();
    hm.mode_ = HostMachineMode::API;

    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "TENSOR_test", nullptr);
    MachineTask task(0, func.get());
    MachineTask* result = hm.Compile(&task);
    EXPECT_EQ(result, &task);
}
