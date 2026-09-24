/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/device_dfx.h"
#undef private

using namespace npu::tile_fwk;

TEST(DeviceDfxTest, Singleton_AndInitDevDfxArgs)
{
    auto& inst1 = DeviceDfx::GetInstance();
    auto& inst2 = DeviceDfx::GetInstance();
    EXPECT_EQ(&inst1, &inst2);

    DevDfxArgs dfxArgsOff{};
    inst1.InitDevDfxArgs(false, dfxArgsOff);
    EXPECT_EQ(dfxArgsOff.isOpenPerfTrace, 0);

    DevDfxArgs dfxArgsOn{};
    inst1.InitDevDfxArgs(true, dfxArgsOn);
    EXPECT_EQ(dfxArgsOn.isOpenPerfTrace, 1);
}

TEST(DeviceDfxTest, InitAicpuPerfAddr_NoEnvVar)
{
    auto& dfx = DeviceDfx::GetInstance();
    DeviceArgs args{};
    args.aicpuPerfAddr = 0;
    dfx.InitAicpuPerfAddr(args);
    EXPECT_EQ(args.aicpuPerfAddr, 0u);
}
