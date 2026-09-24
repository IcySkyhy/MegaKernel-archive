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
 * \file distributed_test_framework.h
 * \brief
 */

#pragma once

#include "distributed_op_test_suite.h"
#include "adapter/api/hcomm_define.h"

namespace npu::tile_fwk {
namespace Distributed {

struct HcomTestParam {
    HcommHandle hcclComm;
    int32_t rootRank;
    HcommRootInfo rootInfo;
};

void TestFrameworkInit(OpTestParam& testParam, HcomTestParam& hcomTestParam, int& physicalDeviceId);
void TestFrameworkInit2Groups(OpTestParam& testParam, HcomTestParam& hcomTestParam, int& physicalDeviceId);
void TestFrameworkDestroy(int32_t timeout);
std::string getTimeStamp();
} // namespace Distributed
} // namespace npu::tile_fwk
