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
 * \file test_common.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include "tilefwk/tilefwk_op.h"

using namespace npu::tile_fwk;

const std::string TEST_TILE_OP_PATH = "framework/tests/st/interface/tile_op/src/";

enum CpyMode { FULL, DIAG };

struct GMTensorInfoTest {
    float* Addr{nullptr};
    int64_t offset0{-1}; // TBD: should divide by TILESIZE or not? E.g . 128 or 128/128
    int64_t offset1{-1}; // TBD: should divide by TILESIZE or not? E.g . 128 or 128/128
};

struct InvokeEntryTest {
    int64_t SubGraphProgramId{-1};
    GMTensorInfoTest GMTensor[2];
};

using IfaTestParam = std::unordered_map<std::string, int>;

/* low latency params config */
inline IfaTestParam lowLatencyParams = {
    {"b", 4},
    {"nq", 32},
    {"s2", 256},
    {"timethreshold", 55},
};

inline IfaTileShapeConfig lowLatencyTileParams{
    256,                          // block size
    32,                           // nTile
    {256, 128},                   // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
    {32, 32, 256, 256, 128, 128}, // c1 tile for S1D@S2D
    {32, 256},                    // v1 tile for S1S2
    {32, 32, 256, 256, 128, 128}, // c2 tile for S1S2@S2D
    {32, 256},                    // v2 tile for S1D
};

/* hight throughput params param config */
inline IfaTestParam hightThroughputParams = {
    {"b", 32},
    {"nq", 128},
    {"s2", 4096},
    {"timethreshold", 280},
};

inline IfaTileShapeConfig hightThroughputTileParams{
    512,                            // block size
    128,                            // nTile
    {256, 128},                     // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
    {128, 128, 128, 256, 128, 128}, // c1 tile for S1D@S2D
    {32, 256},                      // v1 tile for S1S2
    {128, 128, 128, 256, 128, 128}, // c2 tile for S1S2@S2D
    {32, 256},                      // v2 tile for S1D
};

struct GraphInvokeInfoTest {
    int64_t GraphInvokeCount{0};
    InvokeEntryTest GraphInvokeList[40];
};

template <typename T>
DataType GetAstDtype()
{
    DataType astDtype = DataType::DT_BOTTOM;
    if constexpr (std::is_same<T, npu::tile_fwk::float16>::value) {
        astDtype = DataType::DT_FP16;
    }
    if constexpr (std::is_same<T, float>::value) {
        astDtype = DataType::DT_FP32;
    }
    if constexpr (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        astDtype = DataType::DT_BF16;
    }
    if constexpr (std::is_same<T, int8_t>::value) {
        astDtype = DT_INT8;
    }
    if constexpr (std::is_same<T, int32_t>::value) {
        astDtype = DT_INT32;
    }
    EXPECT_NE(astDtype, DT_BOTTOM);
    return astDtype;
}

inline int GetDeviceIdByEnvVar()
{
    const char* devIdPtr = std::getenv("TILE_FWK_DEVICE_ID");
    if (devIdPtr == nullptr) {
        return 0;
    }
    std::string devIdStr(devIdPtr);
    return std::stoi(devIdStr.c_str());
}
