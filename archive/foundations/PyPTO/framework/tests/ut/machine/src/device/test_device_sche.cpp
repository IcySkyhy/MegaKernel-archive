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
 * \file test_device_sche.cpp
 * \brief UT for machine/device/dynamic/device_sche.cpp and dump_device_topo.cpp
 */

#include <gtest/gtest.h>
#include "machine/device/dynamic/device_sche.h"
#include "machine/device/dynamic/context/dump_device_topo.h"
#include "interface/configs/config_manager_ng.h"
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using namespace npu::tile_fwk::dynamic::topo_dump;

extern "C" int StaticTileFwkBackendKernelServer(void* targ);
extern "C" int DynTileFwkBackendKernelServerInit(void* targ);
extern "C" int DynTileFwkBackendKernelServer(void* targ);

TEST(DeviceScheTest, KernelServerEntry_AllRunModesAndDumpTopo)
{
    EXPECT_EQ(StaticTileFwkBackendKernelServer(nullptr), 0);
    EXPECT_EQ(DynTileFwkBackendKernelServerInit(nullptr), 0);

    DeviceKernelArgs invalidKargs{};
    invalidKargs.parameter.runMode = static_cast<DeviceKernelRunMode>(999);
    EXPECT_NE(DynTileFwkBackendKernelServer(&invalidKargs), 0);

    DeviceKernelArgs ctrlKargs{};
    ctrlKargs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    EXPECT_NE(DynTileFwkBackendKernelServer(&ctrlKargs), 0);

    config::SetDebugOption(CFG_RUNTIME_DBEUG_MODE, static_cast<int64_t>(0));
    DevAscendFunctionDupped dup{};
    DumpStitchEdge(dup, dup, 0, 0, 0, DeviceStitchContext::StitchKind::StitchPartial, 0);
}
