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
 * \file aicore_prof_dav3510_pmu.h
 * \brief
 */

#ifndef AICORE_PROF_DAV3510_PMU_H
#define AICORE_PROF_DAV3510_PMU_H

#include <cstdint>

namespace npu::tile_fwk::dynamic {

constexpr uint32_t MAX_PMU_CNT_3510 = 10;

namespace DAV_3510 {
const uint32_t PMU_CTRL_0 = 0x4200;
const uint32_t PMU_CTRL_1 = 0X2400;
const uint32_t PMU_CNT0 = 0x4210;
const uint32_t PMU_CNT1 = 0x4218;
const uint32_t PMU_CNT2 = 0x4220;
const uint32_t PMU_CNT3 = 0x4228;
const uint32_t PMU_CNT4 = 0x4230;
const uint32_t PMU_CNT5 = 0x4238;
const uint32_t PMU_CNT6 = 0x4240;
const uint32_t PMU_CNT7 = 0x4248;
const uint32_t PMU_CNT8 = 0x4250;
const uint32_t PMU_CNT9 = 0x4254;
const uint32_t PMU_CNT_TOTAL0 = 0x4260;
const uint32_t PMU_CNT_TOTAL1 = 0x4264;
const uint32_t PMU_CNT0_IDX = 0x2500;
const uint32_t PMU_CNT1_IDX = 0x2504;
const uint32_t PMU_CNT2_IDX = 0x2508;
const uint32_t PMU_CNT3_IDX = 0x250C;
const uint32_t PMU_CNT4_IDX = 0x2510;
const uint32_t PMU_CNT5_IDX = 0x2514;
const uint32_t PMU_CNT6_IDX = 0x2518;
const uint32_t PMU_CNT7_IDX = 0x251C;
const uint32_t PMU_CNT8_IDX = 0x2520;
const uint32_t PMU_CNT9_IDX = 0x2524;
const uint32_t PMU_START_CNT_CYC_0 = 0x42A0;
const uint32_t PMU_START_CNT_CYC_1 = 0x42A4;
const uint32_t PMU_STOP_CNT_CYC_0 = 0x42A8;
const uint32_t PMU_STOP_CNT_CYC_1 = 0x42AC;
}; // namespace DAV_3510

} // namespace npu::tile_fwk::dynamic

#endif // AICORE_PROF_DAV3510_PMU_H
