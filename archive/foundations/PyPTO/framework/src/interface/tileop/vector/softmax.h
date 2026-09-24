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
 * \file softmax.h
 * \brief Online softmax tile operators.
 */

#ifndef TILEOP_TILE_OPERATOR_SOFTMAX__H
#define TILEOP_TILE_OPERATOR_SOFTMAX__H

#include "pto_tile.h"

#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
#define AICORE [aicore]
#else
#define AICORE
#endif
#define PTO_INLINE inline __attribute__((always_inline))
#define PTO_INST AICORE PTO_INLINE __attribute__((visibility("default")))
#define PTO_INTERNAL AICORE PTO_INLINE

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
TILEOP void TOnlineSoftmaxSyncV()
{
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
}

template <typename TileDataOut, typename TileDataIn, typename Scalar>
__tf__ PTO_INTERNAL void TCOLMAX_HIGH_PERF(TileDataOut& dstTile, TileDataIn& srcTile, Scalar scale)
{
    __ubuf__ typename TileDataIn::DType* dst = (__ubuf__ typename TileDataIn::DType*)__cce_get_tile_ptr(dstTile.data());
    __ubuf__ typename TileDataIn::DType* src = (__ubuf__ typename TileDataIn::DType*)__cce_get_tile_ptr(srcTile.data());

    __ubuf__ float* src0_ub = (__ubuf__ float*)src;
    __ubuf__ float* p0 = src0_ub + 4 * 64;
    __ubuf__ float* p1 = src0_ub + 5 * 64;
    __ubuf__ float* p2 = src0_ub + 6 * 64;
    __ubuf__ float* p3 = src0_ub + 7 * 64;
    __VEC_SCOPE__
    {
        vector_f32 max_0a, max_1a, max_2a, max_3a;

        vbr(max_0a, 0);
        vbr(max_1a, 0);
        vbr(max_2a, 0);
        vbr(max_3a, 0);

        vector_bool preg_108 = pset_b16(PAT_ALL);

        vlds(max_0a, src0_ub, 0 * 64, NORM);
        vlds(max_1a, src0_ub, 1 * 64, NORM);
        vlds(max_2a, src0_ub, 2 * 64, NORM);
        vlds(max_3a, src0_ub, 3 * 64, NORM);

        pto::RegTensor<float> v_row;
        for (uint16_t row = 4; row < uint16_t(TileDataIn::Rows); row += 4) {
            vlds(v_row, p0, 4 * 64, NORM, POST_UPDATE);
            vmax(max_0a, max_0a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p1, 4 * 64, NORM, POST_UPDATE);
            vmax(max_1a, max_1a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p2, 4 * 64, NORM, POST_UPDATE);
            vmax(max_2a, max_2a, v_row, preg_108, MODE_ZEROING);
            vlds(v_row, p3, 4 * 64, NORM, POST_UPDATE);
            vmax(max_3a, max_3a, v_row, preg_108, MODE_ZEROING);
        }

        vmax(max_0a, max_0a, max_1a, preg_108, MODE_ZEROING);
        vmax(max_2a, max_2a, max_3a, preg_108, MODE_ZEROING);
        vmax(max_0a, max_0a, max_2a, preg_108, MODE_ZEROING);
        vmuls(max_0a, max_0a, scale, preg_108, MODE_ZEROING);
        vsts(max_0a, (__ubuf__ float*)dst, 0, NORM_B16, preg_108);
    }
}

#define OP_TILE_OP_ONLINESOFTMAX TOnlineSoftmax
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename Scalar>
TILEOP void TOnlineSoftmax(T0 expScoresBf16, T1 columnMax, T2 columnSum, T3 scores, T4 scaledScores, T5 reduceWorkspace,
                           Scalar scale)
{
    auto expScoresTile = PtoTile<T0>(expScoresBf16);
    auto columnMaxTile = PtoTile<T1>(columnMax);
    auto columnSumTile = PtoTile<T2>(columnSum);
    auto scoresTile = PtoTile<T3>(scores);
    auto scaledScoresTile = PtoTile<T4>(scaledScores);
    auto reduceWorkspaceTile = PtoTile<T5>(reduceWorkspace);
    expScoresTile.Assign(expScoresBf16);
    columnMaxTile.Assign(columnMax);
    columnSumTile.Assign(columnSum);
    scoresTile.Assign(scores);
    scaledScoresTile.Assign(scaledScores);
    reduceWorkspaceTile.Assign(reduceWorkspace);

    auto& expScoresData = expScoresTile.Data();
    auto& columnMaxData = columnMaxTile.Data();
    auto& columnSumData = columnSumTile.Data();
    auto& scoresData = scoresTile.Data();
    auto& scaledScoresData = scaledScoresTile.Data();
    auto& reduceWorkspaceData = reduceWorkspaceTile.Data();

    TCOLMAX_HIGH_PERF(columnMaxData, scoresData, scale);

    TOnlineSoftmaxSyncV();
    [[pto::last_use(0, 1)]] pto::TMULS(scaledScoresData, scoresData, scale);
    TOnlineSoftmaxSyncV();

    pto::TCOLEXPANDSUB(scaledScoresData, scaledScoresData, columnMaxData);
    TOnlineSoftmaxSyncV();
    pto::TEXP(scaledScoresData, scaledScoresData);
    TOnlineSoftmaxSyncV();
    pto::TCVT<false>(expScoresData, scaledScoresData, pto::RoundMode::CAST_ROUND, pto::SaturationMode::OFF);
    TOnlineSoftmaxSyncV();
    [[pto::last_use(0, 1, 1)]] pto::TCOLSUM(columnSumData, scaledScoresData, reduceWorkspaceData, false);
}

#define OP_TILE_OP_ONLINESOFTMAXUPDATE TOnlineSoftmaxUpdate
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
TILEOP void TOnlineSoftmaxUpdate(T0 updatedMax, T1 updatedSum, T2 updatedOutput, T3 previousMax, T4 previousSum,
                                 T5 previousOutput, T6 currentMax, T7 currentSum, T8 currentOutput, T9 updateWorkspace)
{
    auto updatedMaxTile = PtoTile<T0>(updatedMax);
    auto updatedSumTile = PtoTile<T1>(updatedSum);
    auto updatedOutputTile = PtoTile<T2>(updatedOutput);
    auto previousMaxTile = PtoTile<T3>(previousMax);
    auto previousSumTile = PtoTile<T4>(previousSum);
    auto previousOutputTile = PtoTile<T5>(previousOutput);
    auto currentMaxTile = PtoTile<T6>(currentMax);
    auto currentSumTile = PtoTile<T7>(currentSum);
    auto currentOutputTile = PtoTile<T8>(currentOutput);

    const auto statisticLayout = updatedMax.GetLayout();
    const auto outputLayout = updatedOutput.GetLayout();
    auto statisticHeight = statisticLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto statisticWidth = statisticLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto outputHeight = outputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto outputWidth = outputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto statisticTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto outputTileH = TileOp::GetTensorTileShapeDim<T2, DIM_4TH, MAX_DIMS>();
    constexpr auto workspaceTileW = TileOp::GetTensorTileShapeDim<T9, DIM_5TH, MAX_DIMS>();
    constexpr size_t statisticStrideBytes = workspaceTileW * sizeof(typename T0::Type);
    using StatisticScratchTile = pto::Tile<pto::TileType::Vec, typename T0::Type, statisticTileH, workspaceTileW,
                                           pto::BLayout::RowMajor, -1, -1>;
    using OutputScratchTile = pto::Tile<pto::TileType::Vec, typename T2::Type, outputTileH, workspaceTileW,
                                        pto::BLayout::RowMajor, -1, -1>;
    StatisticScratchTile previousScaleTile(statisticHeight, statisticWidth);
    StatisticScratchTile currentScaleTile(statisticHeight, statisticWidth);
    StatisticScratchTile scaledCurrentSumTile(statisticHeight, statisticWidth);
    OutputScratchTile scaledCurrentOutputTile(outputHeight, outputWidth);

    updatedMaxTile.Assign(updatedMax);
    updatedSumTile.Assign(updatedSum);
    updatedOutputTile.Assign(updatedOutput);
    previousMaxTile.Assign(previousMax);
    previousSumTile.Assign(previousSum);
    previousOutputTile.Assign(previousOutput);
    currentMaxTile.Assign(currentMax);
    currentSumTile.Assign(currentSum);
    currentOutputTile.Assign(currentOutput);
    auto& updatedMaxData = updatedMaxTile.Data();
    auto& updatedSumData = updatedSumTile.Data();
    auto& updatedOutputData = updatedOutputTile.Data();
    auto& previousMaxData = previousMaxTile.Data();
    auto& previousSumData = previousSumTile.Data();
    auto& previousOutputData = previousOutputTile.Data();
    auto& currentMaxData = currentMaxTile.Data();
    auto& currentSumData = currentSumTile.Data();
    auto& currentOutputData = currentOutputTile.Data();

    pto::TASSIGN(previousScaleTile, (uint64_t)(updateWorkspace.GetAddr()));
    pto::TASSIGN(currentScaleTile, (uint64_t)(updateWorkspace.GetAddr() + statisticStrideBytes));
    pto::TASSIGN(scaledCurrentSumTile, (uint64_t)(updateWorkspace.GetAddr() + 2U * statisticStrideBytes));
    pto::TASSIGN(scaledCurrentOutputTile, (uint64_t)(updateWorkspace.GetAddr() + 3U * statisticStrideBytes));

    pto::TMAX(updatedMaxData, previousMaxData, currentMaxData);
    TOnlineSoftmaxSyncV();

    [[pto::last_use(0, 1, 0)]] pto::TSUB(previousScaleTile, previousMaxData, updatedMaxData);
    TOnlineSoftmaxSyncV();

    pto::TEXP(previousScaleTile, previousScaleTile);
    TOnlineSoftmaxSyncV();

    [[pto::last_use(0, 1, 0)]] pto::TSUB(currentScaleTile, currentMaxData, updatedMaxData);
    TOnlineSoftmaxSyncV();

    pto::TEXP(currentScaleTile, currentScaleTile);
    TOnlineSoftmaxSyncV();

    [[pto::last_use(0, 0, 1)]] pto::TMUL(updatedSumData, previousScaleTile, previousSumData);
    TOnlineSoftmaxSyncV();

    [[pto::last_use(0, 0, 1)]] pto::TMUL(scaledCurrentSumTile, currentScaleTile, currentSumData);
    TOnlineSoftmaxSyncV();
    [[pto::last_use(0, 0, 1)]] pto::TADD(updatedSumData, updatedSumData, scaledCurrentSumTile);
    TOnlineSoftmaxSyncV();

    [[pto::last_use(0, 1, 1)]] pto::TCOLEXPANDMUL(updatedOutputData, previousOutputData, previousScaleTile);
    TOnlineSoftmaxSyncV();

    [[pto::last_use(0, 1, 1)]] pto::TCOLEXPANDMUL(scaledCurrentOutputTile, currentOutputData, currentScaleTile);
    TOnlineSoftmaxSyncV();

    [[pto::last_use(0, 0, 1)]] pto::TADD(updatedOutputData, updatedOutputData, scaledCurrentOutputTile);
    TOnlineSoftmaxSyncV();
}

#endif // defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)

#endif // TILEOP_TILE_OPERATOR_SOFTMAX__H
