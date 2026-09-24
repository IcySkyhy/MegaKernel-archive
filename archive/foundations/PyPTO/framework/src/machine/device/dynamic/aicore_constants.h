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
 * \file aicore_constants.h
 * \brief
 */

#pragma once

namespace npu::tile_fwk::dynamic {
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

constexpr uint32_t MAX_AICORE_NUM = 108;
constexpr uint32_t NAX_AIV_TOTAL_NUM = 72;
const uint32_t CORE_NUM_PER_AI_CORE = 3;
constexpr uint32_t MAX_MANAGER_AIV_NUM = 72;
constexpr uint32_t MAX_RESOLVE_TASK_NUM = MAX_MANAGER_AIV_NUM * 2;
const uint8_t INVALID_COREIDX_POSITION = 0xff;

const uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
const uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
const uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;

constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_FIVE = 5;
constexpr uint32_t NUM_THIRTY_TWO = 32;

constexpr int32_t AICORE_COREID_MASK = 0x0FFF;
constexpr int32_t AICORE_BLOCKID_MASK = 0x0FFF;
} // namespace npu::tile_fwk::dynamic
