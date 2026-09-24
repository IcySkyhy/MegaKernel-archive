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
 * \file test_simulation_extra.cpp
 * \brief UT for aicore_model_kernel_meta_hook.h and host_aicore_entry_adapter.h
 */

#include <gtest/gtest.h>
#include <thread>
#include "machine/simulation/aicore_model_kernel_meta_hook.h"
#include "machine/simulation/host_aicore_entry_adapter.h"
#include "machine/simulation/host_core_context.h"
#include "machine/simulation/aicore_hardware.h"
#include "machine/runtime/launcher/aicore_model_launcher.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(SimulationExtraTest, InitAicoreModelDevArgs_AllBranches)
{
    InitAicoreModelDefaultDevArgs(nullptr);

    std::vector<uint8_t> buf1(sizeof(DevAscendProgram), 0);
    auto* prog1 = reinterpret_cast<DevAscendProgram*>(buf1.data());
    prog1->devArgs.nrAic = 1;
    InitAicoreModelDefaultDevArgs(prog1);
    EXPECT_EQ(prog1->devArgs.nrAic, 1u);

    std::vector<uint8_t> buf2(sizeof(DevAscendProgram), 0);
    auto* prog2 = reinterpret_cast<DevAscendProgram*>(buf2.data());
    InitAicoreModelDefaultDevArgs(prog2);
    EXPECT_EQ(prog2->devArgs.nrAic, 24u);
    EXPECT_EQ(prog2->devArgs.nrAiv, 48u);
    EXPECT_EQ(prog2->devArgs.nrValidAic, 24u);
    EXPECT_EQ(prog2->devArgs.nrAicpu, 4u);
    EXPECT_EQ(prog2->devArgs.scheCpuNum, 3u);
}

TEST(SimulationExtraTest, InitAicoreModelCoreAddrsAndMetrics)
{
    std::vector<uint8_t> buf(sizeof(DevAscendProgram), 0);
    auto* prog = reinterpret_cast<DevAscendProgram*>(buf.data());
    InitAicoreModelDefaultDevArgs(prog);
    uint32_t totalCoreNum = prog->devArgs.nrAic + prog->devArgs.nrAiv + 1;

    AicoreModelMemoryUtils devMem;
    EXPECT_TRUE(InitAicoreModelCoreAddrs(devMem, prog, totalCoreNum));
    EXPECT_NE(prog->devArgs.sharedBuffer, 0ULL);
    EXPECT_NE(prog->devArgs.coreRegAddr, 0ULL);
    EXPECT_NE(prog->devArgs.corePmuAddr, 0ULL);
    EXPECT_NE(prog->devArgs.devDfxArgAddr, 0ULL);

    EXPECT_TRUE(InitAicoreModelMetrics(devMem, prog, totalCoreNum));

    std::vector<uint8_t> nullBuf(sizeof(DevAscendProgram), 0);
    auto* nullProg = reinterpret_cast<DevAscendProgram*>(nullBuf.data());
    EXPECT_ANY_THROW(InitAicoreModelMetrics(devMem, nullProg, 1));
}

TEST(SimulationExtraTest, AicoreModelInitKernelMetaDeviceArgs_FullFlow)
{
    std::vector<uint8_t> progBuf(sizeof(DevAscendProgram), 0);
    AicoreModelMemoryUtils devMem;
    DeviceKernelArgs kArgs{};

    AicoreModelInitKernelMetaDeviceArgs(devMem, kArgs, progBuf);
    auto* prog = reinterpret_cast<DevAscendProgram*>(progBuf.data());
    EXPECT_EQ(prog->devArgs.nrAic, 24u);
    EXPECT_NE(kArgs.cfgdata, nullptr);
}

TEST(SimulationExtraTest, HostAicoreEntryAdapter_AllFunctions)
{
    EXPECT_GE(get_sys_cnt(), 0ULL);

    HostCoreCtx::SetCurrent({5, 7});
    EXPECT_EQ(get_coreid(), 7);
    EXPECT_EQ(get_block_idx(), 5);
    EXPECT_EQ(get_subblockdim(), 0);
    EXPECT_EQ(get_subblockid(), 0);
    EXPECT_EQ(get_block_num(), 0);

    auto& hw = AicoreHardware::Global();
    hw.Reset(4);
    HostCoreCtx::SetCurrent({1, 2});

    hw.WriteMainBase(2, 12345);
    EXPECT_EQ(GetDataMainBase(), 12345ULL);

    set_cond(999);
    EXPECT_EQ(hw.ReadCond(2), 999ULL);

    CallSubFuncTask(0, nullptr, 0, nullptr);
}
