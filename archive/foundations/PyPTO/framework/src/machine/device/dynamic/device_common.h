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
 * \file device_machine.h
 * \brief
 */

#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "aicore_constants.h"
#include "device_utils.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
const uint32_t MAX_DAV_2210_SCHEDULE_AICPU_NUM = 3;
inline uint32_t CalcSchAicpuNumByBlockDim(uint32_t blockDim, uint32_t aiCpuNum, ArchInfo archInfo)
{
    uint32_t maxScheCore = aiCpuNum - dynamic::MAX_CONTROL_FLOW_AICPU_NUM;
    if (archInfo == ArchInfo::DAV_2201) {
        maxScheCore = maxScheCore >= MAX_DAV_2210_SCHEDULE_AICPU_NUM ? MAX_DAV_2210_SCHEDULE_AICPU_NUM : maxScheCore;
    }

    if (blockDim > (maxScheCore - 1) * dynamic::MAX_MNG_AICORE_AVG_NUM) {
        return maxScheCore;
    }

    if (blockDim % dynamic::MAX_MNG_AICORE_AVG_NUM == 0) {
        return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM;
    }

    return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM + 1;
}

const int DEVICE_MAX_AICPU_NUM = 7;
const uint16_t AICPU_EXECUTE_TIMEOUT = 1080; // 18min

} // namespace npu::tile_fwk::dynamic
