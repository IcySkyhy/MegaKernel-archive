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
 * \file test_simulation.cpp
 * \brief UT for machine/simulation module (host_core_context + aicore_hardware)
 */

#include <gtest/gtest.h>
#include "machine/simulation/host_core_context.h"
#include "machine/simulation/aicore_hardware.h"

using namespace npu::tile_fwk::dynamic;

class SimulationTest : public testing::Test {
protected:
    void SetUp() override { (void)AicoreHardware::Global(); }
};

TEST_F(SimulationTest, HostCoreContext_DefaultAndSetCurrent)
{
    HostCoreContext ctx{};
    EXPECT_EQ(ctx.blockId, -1);
    EXPECT_EQ(ctx.phyId, -1);

    HostCoreCtx::SetCurrent({1, 2});
    const auto& current = HostCoreCtx::Current();
    EXPECT_EQ(current.blockId, 1);
    EXPECT_EQ(current.phyId, 2);

    HostCoreCtx::SetCurrent({10, 20});
    const auto& current2 = HostCoreCtx::Current();
    EXPECT_EQ(current2.blockId, 10);
    EXPECT_EQ(current2.phyId, 20);
}

TEST_F(SimulationTest, AicoreHardware_ResetAndReadWrite)
{
    auto& hw = AicoreHardware::Global();
    hw.Reset(4);
    EXPECT_EQ(hw.CoreNum(), 4u);

    hw.WriteMainBase(0, 100);
    EXPECT_EQ(hw.ReadMainBase(0), 100u);
    hw.WriteMainBase(3, 200);
    EXPECT_EQ(hw.ReadMainBase(3), 200u);
    EXPECT_EQ(hw.ReadMainBase(0), 100u);

    hw.WriteCond(1, 300);
    EXPECT_EQ(hw.ReadCond(1), 300u);
    EXPECT_EQ(hw.ReadCond(0), 0u);
}

TEST_F(SimulationTest, AicoreHardware_Reg32AddrAndInvalidBlockId)
{
    auto& hw = AicoreHardware::Global();
    hw.Reset(4);

    auto addr0 = hw.GetReg32Addr(0);
    auto addr1 = hw.GetReg32Addr(1);
    EXPECT_NE(addr0, 0u);
    EXPECT_NE(addr1, 0u);
    EXPECT_NE(addr0, addr1);

    hw.Reset(2);
    EXPECT_EQ(hw.ReadMainBase(99), 0u);
    EXPECT_EQ(hw.ReadCond(99), 0u);
    EXPECT_EQ(hw.GetReg32Addr(99), 0u);
    hw.WriteMainBase(99, 100);
    hw.WriteCond(99, 200);
    EXPECT_EQ(hw.ReadMainBase(0), 0u);
}
