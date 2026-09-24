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
 * \file cycles.h
 * \brief
 */

#ifndef OP_CYCLES_H
#define OP_CYCLES_H

#include <functional>
#include <unordered_map>
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {
const std::unordered_map<std::string, std::unordered_map<DataType, int>> INTRIN_LATENCY_IN_OP{
    // Vector
    {"UB_ADD",
     {
         {DataType::DT_FP32, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_SUB",
     {
         {DataType::DT_FP16, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_MUL",
     {
         {DataType::DT_FP16, 7},
         {DataType::DT_FP32, 7},
         {DataType::DT_INT32, 5},
         {DataType::DT_INT16, 5},
     }},
    {"UB_DIV",
     {
         {DataType::DT_FP16, 13},
         {DataType::DT_FP32, 13},
     }},
    {"UB_EXP",
     {
         {DataType::DT_FP16, 12},
         {DataType::DT_FP32, 12},
     }},
    {"UB_SQRT",
     {
         {DataType::DT_FP16, 13},
         {DataType::DT_FP32, 13},
     }},
    {"UB_RSQRT",
     {
         {DataType::DT_FP16, 5},
         {DataType::DT_FP32, 5},
     }},
    {"UB_ADDS",
     {
         {DataType::DT_FP16, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_MULS",
     {
         {DataType::DT_FP16, 7},
         {DataType::DT_FP32, 7},
         {DataType::DT_INT32, 5},
         {DataType::DT_INT16, 5},
     }},
    {"UB_PAIRSUM",
     {
         {DataType::DT_FP16, 6},
         {DataType::DT_FP32, 6},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_PAIRMAX",
     {
         {DataType::DT_FP16, 4},
         {DataType::DT_FP32, 4},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_ROWEXPSUM",
     {
         {DataType::DT_FP16, 23},
         {DataType::DT_FP32, 20},
     }},
    {"UB_ROWEXPMAX",
     {
         //  vcadd + vector_dup(no data)
         {DataType::DT_FP16, 23},
         {DataType::DT_FP32, 20},
     }},
    {"UB_MAXIMUM",
     {
         {DataType::DT_FP16, 4},
         {DataType::DT_FP32, 4},
         {DataType::DT_INT32, 4},
         {DataType::DT_INT16, 4},
     }},
    {"UB_COMPACT",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_EXPAND",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_MOV",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_RECIPROCAL",
     {
         {DataType::DT_FP16, 5},
         {DataType::DT_FP32, 5},
     }},
    {"UB_ROWSUM",
     {
         {DataType::DT_FP16, 23},
         {DataType::DT_FP32, 20},
     }},
    {"UB_ROWMAX",
     {
         {DataType::DT_FP16, 9},
         {DataType::DT_FP32, 8},
     }}, // vcmax + vector_dup(no data)
         // DMA
    {"UB_COPY_IN",
     {
         {DataType::DT_FP16, 50},
         {DataType::DT_FP32, 50},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"UB_COPY_OUT",
     {
         {DataType::DT_FP16, 50},
         {DataType::DT_FP32, 50},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    // cube
    // mte
    {"L1_COPY_IN",
     {
         {DataType::DT_FP16, 200},
         {DataType::DT_FP32, 200},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L0C_COPY_OUT",
     {
         {DataType::DT_FP16, 200},
         {DataType::DT_FP32, 200},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_TO_L0A",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_TO_L0B",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    {"L1_TO_L0Bt",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 0},
         {DataType::DT_INT16, 0},
     }},
    // mad
    {"CUBE_A_MUL_B",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }},
    {"CUBE_A_MUL_Bt",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }},
    {"CUBE_A_MULACC_B",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }},
    {"CUBE_A_MULACC_Bt",
     {
         {DataType::DT_FP16, 0},
         {DataType::DT_FP32, 0},
         {DataType::DT_INT32, 7},
         {DataType::DT_INT16, 7},
     }},
};

const std::unordered_map<std::string, std::unordered_map<DataType, int>> INTRIN_PARALLELISM_IN_OP{
    // Vector
    {"UB_ADD",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_SUB",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_MUL",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_DIV",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
     }},
    {"UB_EXP",
     {
         {DataType::DT_FP16, 32},
         {DataType::DT_FP32, 32},
     }},
    {"UB_SQRT",
     {
         {DataType::DT_FP16, 32},
         {DataType::DT_FP32, 32},
     }},
    {"UB_RSQRT",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
     }},
    {"UB_ADDS",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_MULS",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_PAIRSUM",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_PAIRMAX",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_ROWEXPSUM",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
     }},
    {"UB_ROWEXPMAX",
     {
         //  vcadd + vector_dup(no data)
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
     }},
    {"UB_MAXIMUM",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_EXPAND",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_MOV",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"UB_RECIPROCAL",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
     }},
    {"UB_ROWSUM",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
     }},
    {"UB_ROWMAX",
     {
         {DataType::DT_FP16, 128},
         {DataType::DT_FP32, 64},
     }}, // vcmax + vector_dup(no data)
         // DMA
    {"UB_COPY_IN",
     {
         {DataType::DT_FP16, 64},
         {DataType::DT_FP32, 32},
         {DataType::DT_INT32, 32},
         {DataType::DT_INT16, 64},
     }},
    {"UB_COPY_OUT",
     {
         {DataType::DT_FP16, 64},
         {DataType::DT_FP32, 32},
         {DataType::DT_INT32, 32},
         {DataType::DT_INT16, 64},
     }},
    // mte
    {"L1_COPY_IN",
     {
         {DataType::DT_FP16, 64}, // 128B/cycle
         {DataType::DT_FP32, 32},
         {DataType::DT_INT32, 32},
         {DataType::DT_INT16, 64},
     }},
    {"L0C_COPY_OUT",
     {
         {DataType::DT_FP16, 64}, // 128B/cycle
         {DataType::DT_FP32, 32},
         {DataType::DT_INT32, 32},
         {DataType::DT_INT16, 64},
     }},
    {"L1_TO_L0A",
     {
         {DataType::DT_FP16, 128}, // 256B/cycle
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"L1_TO_L0B",
     {
         {DataType::DT_FP16, 128}, // 256B/cycle
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    {"L1_TO_L0Bt",
     {
         {DataType::DT_FP16, 128}, // 256B/cycle
         {DataType::DT_FP32, 64},
         {DataType::DT_INT32, 64},
         {DataType::DT_INT16, 128},
     }},
    // mad
    {"CUBE_A_MUL_B",
     {
         {DataType::DT_FP16, 256}, // 512B/cycle
         {DataType::DT_FP32, 128},
         {DataType::DT_INT32, 128},
         {DataType::DT_INT16, 256},
     }},
    {"CUBE_A_MUL_Bt",
     {
         {DataType::DT_FP16, 256}, // 512B/cycle
         {DataType::DT_FP32, 128},
         {DataType::DT_INT32, 128},
         {DataType::DT_INT16, 256},
     }},
    {"CUBE_A_MULACC_B",
     {
         {DataType::DT_FP16, 256}, // 512B/cycle
         {DataType::DT_FP32, 128},
         {DataType::DT_INT32, 128},
         {DataType::DT_INT16, 256},
     }},
    {"CUBE_A_MULACC_Bt",
     {
         {DataType::DT_FP16, 256}, // 512B/cycle
         {DataType::DT_FP32, 128},
         {DataType::DT_INT32, 128},
         {DataType::DT_INT16, 256},
     }},
};

const std::unordered_map<std::string, int> SYNC_OP_CYCLES{{"SYNC_SRC", 1}, {"SYNC_DST", 1}, {"BAR.V", 18}};

int64_t CalcUBCompactCycles(const std::vector<std::vector<int64_t>>& shape, DataType dtype);
const std::unordered_map<std::string,
                         std::function<int64_t(const std::vector<std::vector<int64_t>>& shape, DataType dtype)>>
    COMINE_INTRIN_CYCLES_IN_OP = {{"UB_COMPACT", [](const std::vector<std::vector<int64_t>>& shape, DataType dtype) {
                                       return CalcUBCompactCycles(shape, dtype);
                                   }}};

int64_t GetCycles(const std::string& op, const std::vector<std::vector<int64_t>>& shape, DataType dtype);
} // namespace npu::tile_fwk
#endif // OP_CYCLES_H
