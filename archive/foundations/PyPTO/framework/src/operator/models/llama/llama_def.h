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
 * \file llama_def.h
 * \brief
 */

#pragma once
#ifndef LLAMA_DEF_H
#define LLAMA_DEF_H

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

#define LLAMA_FUNCTION(n, ...) FUNCTION(#n, ##__VA_ARGS__)
#define LLAMA_PROGRAM(n, ...) PROGRAM(#n, ##__VA_ARGS__)

namespace npu::tile_fwk {

struct AttentionDims {
    int b;
    int n;
    int s;
    int d;
    int singleM;
    int singleN;
};

struct AttentionVecTileConfig {
    int defaultVecTileX;
    int defaultVecTileY;
    int softmaxTileX;
    int softmaxTileY;
    int updateTileX;
    int updateTileY;
    int castTileX;
    int castTileY;
};

struct AttentionCubeTileConfig {
    int c1L1M;
    int c1L1K;
    int c1L1N;
    int c2L1M;
    int c2L1K;
    int c2L1N;
    int c1L0 = 128;
    int c2L0 = 128;
};

struct KeyConfig {
    int max;
    int min;
    int dbType;
    int nBuffer;
    int isPartitionCv;
    int cTileX;
    int cTileY;
};

constexpr AttentionVecTileConfig DFS_VEC_CFG = {128, 128, 16, 128, 16, 128, 32, 128};
constexpr AttentionVecTileConfig SMALL_DFS_VEC_CFG = {64, 128, 16, 128, 16, 128, 32, 128};
constexpr AttentionCubeTileConfig DFS_CUBE_CFG = {128, 128, 128, 128, 128, 128};

constexpr AttentionVecTileConfig OOO_VEC_CFG = {64, 128, 16, 512, 32, 128, 32, 128};
constexpr AttentionCubeTileConfig OOO_CUBE_CFG = {128, 128, 512, 64, 256, 64, 128, 64};

constexpr KeyConfig DFT_BASIC_CFG = {8192, 1024, 1, 1, 0, 128, 128};
constexpr int DFT_SINGLE_M = 128;
constexpr int DFT_SINGLE_N = 128;

Tensor LlamaLayer(Tensor hiddenStates, const Tensor& attnWight, const Tensor& denseWeight, const Tensor& ffnWeight,
                  const AttentionDims& atDims, const AttentionVecTileConfig& vecCfg,
                  const AttentionCubeTileConfig& cubeCfg);

Tensor FlashAttention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& m, const Tensor& l,
                      const AttentionDims& atDims, const AttentionVecTileConfig& vecCfg,
                      const AttentionCubeTileConfig& cubeCfg);

Tensor FlashAttentionNew(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& m, const Tensor& l,
                         const AttentionDims& atDims);

static inline void SetC1CubeConfig(const AttentionCubeTileConfig& cubeCfg)
{
    TileShape::Current().SetCubeTile({cubeCfg.c1L0, cubeCfg.c1L1M}, {cubeCfg.c1L0, cubeCfg.c1L1K},
                                     {cubeCfg.c1L0, cubeCfg.c1L1N});
}

static inline void SetC2CubeConfig(const AttentionCubeTileConfig& cubeCfg)
{
    TileShape::Current().SetCubeTile({cubeCfg.c2L0, cubeCfg.c2L1M}, {cubeCfg.c2L0, cubeCfg.c2L1K},
                                     {cubeCfg.c2L0, cubeCfg.c2L1N});
}

} // namespace npu::tile_fwk

#endif
