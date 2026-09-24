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
 * \file test_device_launcher_binding.cpp
 * \brief UT for machine/runtime/launcher/device_launcher_binding.cpp
 */

#include <gtest/gtest.h>
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "interface/program/program.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(DeviceLauncherBindingTest, AllBindingOps)
{
    DeviceLauncherInit();
    DeviceLauncherFini();

    ChangeCaptureModeRelax();
    ChangeCaptureModeGlobal();

    (void)DeviceSynchronize(0);

    AclMdlRI rtModel = nullptr;
    bool isCapture = false;
    GetCaptureInfo(nullptr, rtModel, isCapture);

    ExportedOperator* op = ExportedOperatorBegin();
    EXPECT_NE(op, nullptr);

    UnregisterKernelBinary(nullptr);
}
