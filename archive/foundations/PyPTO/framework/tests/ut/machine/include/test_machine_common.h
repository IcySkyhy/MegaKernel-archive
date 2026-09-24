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
 * \file test_machine_common.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/program/program.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/device_log.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/emulation_launcher.h"
#include "machine/device/dynamic/context/device_execute_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class UnitTestBase : public ::testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        ProgramData::GetInstance().Reset();

        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    }

    void TearDown() override {}
};

extern "C" int PyptoKernelCtrlServerRegisterTaskInspector(DeviceTaskInspectorEntry inspectorEntry, void* inspector);
