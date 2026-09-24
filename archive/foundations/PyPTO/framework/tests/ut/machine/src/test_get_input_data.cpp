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

#include "interface/machine/device/tilefwk/aicpu_runtime.h"

using namespace npu::tile_fwk;

DevStartArgsBase* AiCoreRuntimeGetStartArgs(CoreFuncParam* param) { return param->funcHeader->startArgs; }

TEST(TestGetInputData, int32)
{
    DevTensorData devTensorList;
    DevStartArgsBase startArgs;
    DynFuncData funcData;
    DynFuncHeader funcHeader;
    CoreFuncParam funcParam, *param = &funcParam;

    param->funcData = &funcData;
    param->funcHeader = &funcHeader;
    funcHeader.startArgs = &startArgs;
    startArgs.devTensorList = &devTensorList;

    std::vector<int> data(4 * 4 * 4 * 4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = i;
    }

    devTensorList.address = (uint64_t)data.data();
    for (size_t i = 0; i < 4; ++i) {
        devTensorList.shape.dim[i] = 4;
    }

    int ret;
    devTensorList.shape.dimSize = 1;
    ret = RUNTIME_GetInputData(0, RUNTIME_int32_t, 1);
    EXPECT_EQ(ret, 1);

    devTensorList.shape.dimSize = 2;
    ret = RUNTIME_GetInputData(0, RUNTIME_int32_t, 1, 1);
    EXPECT_EQ(ret, 5);

    devTensorList.shape.dimSize = 3;
    ret = RUNTIME_GetInputData(0, RUNTIME_int32_t, 1, 1, 1);
    EXPECT_EQ(ret, 21);

    devTensorList.shape.dimSize = 4;
    ret = RUNTIME_GetInputData(0, RUNTIME_int32_t, 1, 1, 1, 1);
    EXPECT_EQ(ret, 85);
}

TEST(TestGetInputData, int64)
{
    DevTensorData devTensorList;
    DevStartArgsBase startArgs;
    DynFuncData funcData;
    DynFuncHeader funcHeader;
    CoreFuncParam funcParam, *param = &funcParam;

    param->funcData = &funcData;
    param->funcHeader = &funcHeader;
    funcHeader.startArgs = &startArgs;
    startArgs.devTensorList = &devTensorList;

    std::vector<int64_t> data(4 * 4 * 4 * 4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = i;
    }

    devTensorList.address = (uint64_t)data.data();
    for (size_t i = 0; i < 4; ++i) {
        devTensorList.shape.dim[i] = 4;
    }

    int ret;
    devTensorList.shape.dimSize = 1;
    ret = RUNTIME_GetInputData(0, RUNTIME_int64_t, 1);
    EXPECT_EQ(ret, 1);

    devTensorList.shape.dimSize = 2;
    ret = RUNTIME_GetInputData(0, RUNTIME_int64_t, 1, 1);
    EXPECT_EQ(ret, 5);

    devTensorList.shape.dimSize = 3;
    ret = RUNTIME_GetInputData(0, RUNTIME_int64_t, 1, 1, 1);
    EXPECT_EQ(ret, 21);

    devTensorList.shape.dimSize = 4;
    ret = RUNTIME_GetInputData(0, RUNTIME_int64_t, 1, 1, 1, 1);
    EXPECT_EQ(ret, 85);
}
