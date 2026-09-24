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
 * \file trans_data.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TRANS_DATA__H
#define TILEOP_TILE_OPERATOR_TRANS_DATA__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

TILEOP void Sync23_VS()
{
#ifdef __DAV_V220
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
#endif
}

TILEOP void Sync2_VS()
{
#ifdef __DAV_V220
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
}

TILEOP void SyncVS_3()
{
#ifdef __DAV_V220
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
}

#define OP_TILE_OP_TRANSDATA_NCHW2NC1HWC0 TTransDataNCHW2NC1HWC0
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2NC1HWC0(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;

    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    if (inputN == 0 || inputC == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    Sync2_VS();
    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileC, tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, tileH * tileW);
        for (LoopVar i = 0; i < inputN; i++) {
            pto::TASSIGN(tmpInputTile,
                         (uint64_t)(input.GetAddr() + (i * tileC + inputC) * tileH * tileW * inputTypeSize));
            pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        }
        pipe_barrier(PIPE_V);
    }

    constexpr int elementSize = tileN * tileC * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                        pto::ConvTileShape<tileN, tileC, tileH, tileW>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                         pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(dst.GetAddr()));
    auto tmpAreaAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_NCHW2Fractal_Z TTransDataNCHW2Fractal_Z
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2Fractal_Z(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;
    constexpr int elementSize = tileN * tileC * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;
    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (inputN == 0 || inputC == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                        pto::ConvTileShape<tileN, tileC, tileH, tileW>>;
    using tmpDst1TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                          pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmp1TileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                   tileH * tileW, C0>;
    inputTileData convInput;
    tmpDst1TileData convTmpDstNC1HWC0;
    tmp1TileData tmpTile;

    auto tmpDstNC1HWC0Addr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaTileAddr = tmpDstNC1HWC0Addr + elementSize;
    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDstNC1HWC0, (uint64_t)tmpDstNC1HWC0Addr);
    pto::TASSIGN(tmpTile, (uint64_t)tmpAreaTileAddr);

    auto inputPadN = (inputN + N0 - 1) / N0 * N0;
    auto padNSize = inputPadN - inputN;
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    Sync2_VS();
    if (padNSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileN, tileC * tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padNSize, tileC * tileH * tileW);
        pto::TASSIGN(tmpInputTile, (uint64_t)(input.GetAddr() + inputN * tileC * tileH * tileW * inputTypeSize));
        pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        pipe_barrier(PIPE_V);
    }

    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileC, tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, tileH * tileW);
        for (LoopVar i = 0; i < inputN; i++) {
            pto::TASSIGN(tmpInputTile,
                         (uint64_t)(input.GetAddr() + (i * tileC + inputC) * tileH * tileW * inputTypeSize));
            pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        }
        pipe_barrier(PIPE_V);
    }

    pto::TTRANS(convTmpDstNC1HWC0, convInput, tmpTile);
    pipe_barrier(PIPE_V);
    constexpr int64_t tileN1 = tileN / N0;
    using tmpDst2TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::FRACTAL_Z,
                                          pto::ConvTileShape<tileC1 * tileH * tileW, tileN1, N0, C0>>;
    tmpDst2TileData dstFractalZ;
    pto::TASSIGN(dstFractalZ, (uint64_t)dst.GetAddr());
    pto::TTRANS(dstFractalZ, convTmpDstNC1HWC0, tmpTile);
}

#define OP_TILE_OP_TRANSDATA_NC1HWC02NCHW TTransDataNC1HWC02NCHW
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNC1HWC02NCHW(DST dst, TMP tmpTensor, INPUT input)
{
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto C0 = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;
    auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC1 = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (inputN == 0 || inputC1 == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    constexpr int elementSize = tileN * tileC1 * tileH * tileW * C0;
    constexpr int bufferSize = elementSize * inputTypeSize;
    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<tileN, tileC1 * C0, tileH, tileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)dst.GetAddr());
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpTensor.GetAddr());
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_NCDHW2NDC1HWC0 TTransDataNCDHW2NDC1HWC0
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCDHW2NDC1HWC0(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileD = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;

    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputD = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    if (inputN == 0 || inputD == 0 || inputC == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    Sync23_VS();
    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileC, tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, tileH * tileW);
        for (LoopVar i = 0; i < inputN; i++) {
            for (LoopVar j = 0; j < inputD; j++) {
                pto::TASSIGN(tmpInputTile,
                             (uint64_t)(input.GetAddr() + i * tileD * tileC * tileH * tileW * inputTypeSize +
                                        (j * tileC + inputC) * tileH * tileW * inputTypeSize));
                pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
            }
        }
        pipe_barrier(PIPE_V);
    }

    constexpr int elementSize = tileN * tileD * tileC * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::GNCHW,
                                        pto::ConvTileShape<tileN, tileD, tileC, tileH, tileW>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::GNC1HWC0,
                                         pto::ConvTileShape<tileN, tileD, tileC1, tileH, tileW, C0>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>; // TODO
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(dst.GetAddr()));
    auto tmpAreaAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_NDC1HWC02NCDHW TTransDataNDC1HWC02NCDHW
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNDC1HWC02NCDHW(DST dst, TMP tmpTensor, INPUT input)
{
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto tileD = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto C0 = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;
    auto inputLayout = input.GetLayout();
    auto inputD = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC1 = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (inputD == 0 || inputC1 == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    constexpr int elementSize = tileD * tileC1 * tileH * tileW * C0;
    constexpr int bufferSize = elementSize * inputTypeSize;
    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<tileD, tileC1, tileH, tileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<tileD, tileC1 * C0, tileH, tileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)dst.GetAddr());
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpTensor.GetAddr());
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_NCDHW2FRACTAL_Z_3D TTransDataNCDHW2FRACTAL_Z_3D
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCDHW2FRACTAL_Z_3D(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileD = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;
    constexpr auto tileN1 = tileN / N0;
    constexpr int elementSize = tileN * tileC * tileD * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;
    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputD = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (inputN == 0 || inputC == 0 || inputD == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCDHW,
                                        pto::ConvTileShape<tileN, tileC, tileD, tileH, tileW>>;
    using tmpDst1TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize,
                                          pto::Layout::FRACTAL_Z_3D,
                                          pto::ConvTileShape<tileD * tileC1 * tileH * tileW, tileN1, N0, C0>>;
    using tmp1TileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                   tileH * tileW, C0>;
    inputTileData convInput;
    tmpDst1TileData ConvDst;
    tmp1TileData tmpTile;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(ConvDst, (uint64_t)dst.GetAddr());
    pto::TASSIGN(tmpTile, (uint64_t)tmpTensor.GetAddr());

    auto inputPadN = (inputN + N0 - 1) / N0 * N0;
    auto padNSize = inputPadN - inputN;
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    Sync2_VS();
    if (padNSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileN, tileC * tileD * tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padNSize, tileC * tileD * tileH * tileW);
        pto::TASSIGN(tmpInputTile,
                     (uint64_t)(input.GetAddr() + inputN * tileC * tileD * tileH * tileW * inputTypeSize));
        pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        pipe_barrier(PIPE_V);
    }

    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileC, tileD * tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, tileD * tileH * tileW);
        for (LoopVar i = 0; i < inputN; i++) {
            pto::TASSIGN(tmpInputTile,
                         (uint64_t)(input.GetAddr() + (i * tileC + inputC) * tileD * tileH * tileW * inputTypeSize));
            pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        }
        pipe_barrier(PIPE_V);
    }

    pto::TTRANS(ConvDst, convInput, tmpTile);
}
#endif
