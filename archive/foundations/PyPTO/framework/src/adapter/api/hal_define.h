/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hal_define.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace npu::tile_fwk {
enum HalError {
    HAL_ERROR_NONE = 0, /**< success */
    HAL_ERROR_RESERVED  /**< reserved */
};

enum class ProcessType {
    CP1 = 0,  /* aicpu_scheduler */
    CP2,      /* custom_process */
    DEV_ONLY, /* TDT */
    QS,       /* queue_scheduler */
    HCCP,     /* hccp server */
    USER,     /* user proc, can bind many on host or device. not surport quert from host pid */
    CPTYPE_MAX
};

enum class ResMapType { AICORE = 0, HSCB_AICORE, L2BUFF, C2C, MAP_TYPE_MAX };

#define RES_MAP_INFO_RSV_LEN 1
struct ResMapInfo {
    ProcessType target_proc_type;
    ResMapType res_type;
    unsigned int res_id;                    /* corresponding resource id if res_type is NOTIFY or CNT_NOTIFY */
    unsigned int flag;                      /* default is 0 */
    unsigned int rsv[RES_MAP_INFO_RSV_LEN]; /* default is 0 */
};
} // namespace npu::tile_fwk
