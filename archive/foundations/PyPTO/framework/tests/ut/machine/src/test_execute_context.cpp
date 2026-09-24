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
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/utils/dynamic/dev_start_args.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DeviceExecuteContextTest : public ::testing::Test {
protected:
    DeviceExecuteContextTest()
    {
        DevStartArgs args;
        context = new DeviceExecuteContext(&args);
    }

    ~DeviceExecuteContextTest() override { delete context; }

    struct DeviceExecuteContext* context;
};

TEST_F(DeviceExecuteContextTest, HandleParallelKey)
{
    void* result = context->CallRootFunctionStitch(RUNTIME_FUNCKEY_LOOP_BARRIER);
    EXPECT_EQ(result, nullptr);
    EXPECT_NE(context->parallelCtx.info.forId, 0);

    result = context->CallRootFunctionStitch(RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN);
    EXPECT_EQ(result, nullptr);
    EXPECT_NE(context->parallelCtx.info.iterId, 0);
    EXPECT_TRUE(context->parallelCtx.isInParallelForScope);

    result = context->CallRootFunctionStitch(RUNTIME_FUNCKEY_PARALLEL_FOR_END);
    EXPECT_EQ(result, nullptr);
    EXPECT_FALSE(context->parallelCtx.isInParallelForScope);
}
