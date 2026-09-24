/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file test_load_aicpu_op.cpp
 * \brief UT for machine/runtime/runner/load_aicpu_op.cpp
 *
 * In UT (without CANN), runtime stubs return RT_SUCCESS, and MACHINE_LOGE throws
 * via MACHINE_ASSERT(false). Tests cover both success and error paths.
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/load_aicpu_op.h"
#undef private
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;

namespace {
struct AicpuLaunchFixture {
    DeviceKernelArgs kArgs{};
    AicpuLaunchDesc launchDesc{};
    AicpuLaunchFixture()
    {
        launchDesc.args = reinterpret_cast<AiCpuArgs*>(&kArgs);
        launchDesc.argsSize = sizeof(DeviceKernelArgs);
        launchDesc.blockDim = 1U;
    }
};

void PopulateBuiltInFuncMap(LoadAicpuOp& op)
{
    op.builtInFuncMap_["PyptoInit"] = nullptr;
    op.builtInFuncMap_["PyptoRun"] = nullptr;
}
} // namespace

// ===== AicpuKernelLaunch (static) =====

TEST(LoadAicpuOpTest, AicpuKernelLaunch_ValidArgs_ReturnsZero)
{
    AicpuLaunchFixture f;
    EXPECT_EQ(LoadAicpuOp::AicpuKernelLaunch(nullptr, f.launchDesc), 0);
}

TEST(LoadAicpuOpTest, AicpuKernelLaunch_ZeroBlockDim_ReturnsZero)
{
    AicpuLaunchFixture f;
    f.launchDesc.blockDim = 0U;
    EXPECT_EQ(LoadAicpuOp::AicpuKernelLaunch(nullptr, f.launchDesc), 0);
}

// ===== LaunchWithHostArgs (static) =====

TEST(LoadAicpuOpTest, LaunchWithHostArgs_NoTimeout_ReturnsZero)
{
    AicpuLaunchFixture f;
    EXPECT_EQ(LoadAicpuOp::LaunchWithHostArgs(nullptr, f.launchDesc), 0);
}

TEST(LoadAicpuOpTest, LaunchWithHostArgs_WithTimeout_ReturnsZero)
{
    AicpuLaunchFixture f;
    f.launchDesc.timeout = 100;
    EXPECT_EQ(LoadAicpuOp::LaunchWithHostArgs(nullptr, f.launchDesc), 0);
}

// ===== LaunchBuiltInOpWithHostArgs =====

TEST(LoadAicpuOpTest, LaunchBuiltInOpWithHostArgs_ValidFuncName_ReturnsZero)
{
    AicpuLaunchFixture f;
    LoadAicpuOp op;
    PopulateBuiltInFuncMap(op);
    EXPECT_EQ(op.LaunchBuiltInOpWithHostArgs(f.launchDesc, "PyptoRun"), 0);
}

TEST(LoadAicpuOpTest, LaunchBuiltInOpWithHostArgs_WithTimeout_ReturnsZero)
{
    AicpuLaunchFixture f;
    f.launchDesc.timeout = 50;
    f.launchDesc.blockDim = 2U;
    LoadAicpuOp op;
    PopulateBuiltInFuncMap(op);
    EXPECT_EQ(op.LaunchBuiltInOpWithHostArgs(f.launchDesc, "PyptoRun"), 0);
}

TEST(LoadAicpuOpTest, LaunchBuiltInOpWithHostArgs_InvalidFuncName_Throws)
{
    AicpuLaunchFixture f;
    LoadAicpuOp op;
    EXPECT_ANY_THROW(op.LaunchBuiltInOpWithHostArgs(f.launchDesc, "UnknownFunc"));
}

// ===== GetBuiltInOpBinHandle =====

TEST(LoadAicpuOpTest, GetBuiltInOpBinHandle_ValidPath_PopulatesFuncMap)
{
    LoadAicpuOp op;
    EXPECT_EQ(op.GetBuiltInOpBinHandle(nullptr), 0);
    EXPECT_EQ(op.builtInFuncMap_.size(), 2U);
    EXPECT_TRUE(op.builtInFuncMap_.count("PyptoInit") > 0);
    EXPECT_TRUE(op.builtInFuncMap_.count("PyptoRun") > 0);
}

// ===== CustomAiCpuSoLoad =====

TEST(LoadAicpuOpTest, CustomAiCpuSoLoad_WithoutNewCann_NoCrash)
{
    LoadAicpuOp op;
    EXPECT_NO_THROW(op.CustomAiCpuSoLoad());
}

// ===== GetInstance =====

TEST(LoadAicpuOpTest, GetInstance_ReturnsSameSingleton)
{
    auto& inst1 = LoadAicpuOp::GetInstance();
    auto& inst2 = LoadAicpuOp::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}
