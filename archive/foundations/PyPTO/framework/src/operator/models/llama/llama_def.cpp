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
 * \file llama_def.cpp
 * \brief
 */

#include "operator/models/llama/llama_def.h"

#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"

#ifndef LLAMA_FUNCTION
#define LLAMA_FUNCTION(n, ...)
#endif

#ifndef LLAMA_PROGRAM
#define LLAMA_PROGRAM(n, ...)
#endif

constexpr int T_SHAPE = 128;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr float F_1 = 1.0;
constexpr float F_NEGA_1 = -1.0;
namespace npu::tile_fwk {

void SetDefaultL0CubeConfig()
{
    TileShape::Current().SetCubeTile({T_SHAPE, T_SHAPE}, {T_SHAPE, T_SHAPE}, {T_SHAPE, T_SHAPE});
}

Tensor FlashAttention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& m, const Tensor& l,
                      const AttentionDims& atDims, const AttentionVecTileConfig& vecCfg,
                      const AttentionCubeTileConfig& cubeCfg)
{
    (void)m;
    (void)l;
    // q, k, v, result shape: [b*s, n*d]
    int dim0 = q.GetShape()[0];
    int dim1 = q.GetShape()[1];
    int b = atDims.b;
    int n = atDims.n;
    int s = dim0 / b;
    int d = dim1 / n;
    int singleM = atDims.singleM; // 128
    assert(singleM == 128);
    int singleN = atDims.singleN; // 1024
    int s1Loop = s / singleM;
    int s2Loop = s / singleN;

    std::cout << "FlashAttention, B, N, S, D --------" << b << "," << n << "," << s << "," << d << std::endl;
    std::cout << "s1Loop, s2Loop -------" << s1Loop << "," << s2Loop << "," << std::endl;

    auto bns = atDims.b * atDims.n * atDims.s;
    std::vector<int64_t> shapeReduce = {bns, 1};
    std::vector<float> max(bns, 0);
    std::vector<float> sum(bns, 0);

    std::map<std::vector<int64_t>, Tensor> lastOi;
    std::map<std::vector<int64_t>, Tensor> lastMi;
    std::map<std::vector<int64_t>, Tensor> lastLi;
    Tensor result;

    // LLAMA_FUNCTION(FlashAttention_L4) {
    for (int bIdx = 0; bIdx < b; bIdx++) {
        // LLAMA_FUNCTION(FlashAttention_L3) {
        for (int nIdx = 0; nIdx < n; nIdx++) {
            // LLAMA_FUNCTION(FlashAttention_L2) {
            for (int s2Idx = 0; s2Idx < s2Loop; s2Idx++) {
                // LLAMA_FUNCTION(FlashAttention_L1) {
                auto kj = View(k, {singleN, d}, {bIdx * s + s2Idx * singleN, nIdx * d});
                auto vj = View(v, {singleN, d}, {bIdx * s + s2Idx * singleN, nIdx * d});

                for (int s1Idx = 0; s1Idx < s1Loop; s1Idx++) {
                    // LLAMA_FUNCTION(FlashAttention_L0) {
                    auto qi = View(q, {singleM, d}, {bIdx * s + s1Idx * singleM, nIdx * d});
                    std::vector<int64_t> oiOffset = {bIdx * s + s1Idx * singleM, nIdx * d};
                    std::vector<int64_t> liOffset = {(bIdx * n + nIdx) * s + s1Idx * singleM, 0};
                    std::vector<int64_t> miOffset = {(bIdx * n + nIdx) * s + s1Idx * singleM, 0};
                    SetC1CubeConfig(cubeCfg);
                    auto sij = Matrix::Matmul(DataType::DT_FP32, qi, kj, false,
                                              true); // [128, 128], [128, 1024] => [128, 1024]

                    TileShape::Current().SetVecTile(vecCfg.softmaxTileX, vecCfg.softmaxTileY);

                    auto tildaMij = Amax(sij, -1, true);
                    auto tsub = Sub(sij, tildaMij);
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, DataType::DT_FP16);
                    auto tildaLij = Sum(tildaPij, -1, true);

                    SetC2CubeConfig(cubeCfg);

                    if (!s2Idx) {
                        auto oiTmp = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);
                        if (s2Loop == 1) {
                            auto liExpand = Reciprocal(tildaLij);
                            lastOi[oiOffset] = Mul(oiTmp, liExpand);
                        } else {
                            lastOi[oiOffset] = oiTmp;
                        }
                        lastLi[liOffset] = tildaLij;
                        lastMi[miOffset] = tildaMij;
                        continue;
                    }

                    ASSERT(lastOi.count(oiOffset) > 0);
                    ASSERT(lastLi.count(liOffset) > 0);
                    ASSERT(lastMi.count(miOffset) > 0);
                    auto oi = lastOi[oiOffset];
                    auto li = lastLi[liOffset];
                    auto mi = lastMi[miOffset];

                    auto miNew = Maximum(mi, tildaMij); // [128, 1], [128, 1] => [128, 1]
                    auto t1 = Sub(mi, miNew);           // [128, 1], [128, 1] => [128, 1]
                    auto t2 = Exp(t1);                  // [128, 1]
                    auto t3 = Sub(tildaMij, miNew);     // [128, 1], [128, 1] => [128, 1]
                    auto t4 = Exp(t3);                  // [128, 1]
                    auto t5 = Mul(t4, tildaLij);        // [128, 1], [128, 1] => [128, 1]
                    auto t6 = Mul(t2, li);              // [128, 1], [128, 1] => [128, 1]
                    auto liNew = Add(t6, t5);           // [128, 1], [128, 1] => [128, 1]

                    auto q3 = Mul(oi, t2);
                    auto q1 = Matrix::Matmul(DataType::DT_FP32, tildaPijF16, vj, false, false);
                    auto q2 = Mul(q1, t4);
                    auto oiTmp = Add(q3, q2); // [128, 128]
                    if (s2Idx == s2Loop - 1) {
                        lastOi[oiOffset] = Mul(oiTmp, Reciprocal(liNew));
                    } else {
                        lastOi[oiOffset] = oiTmp;
                    }
                    lastLi[liOffset] = liNew;
                    lastMi[miOffset] = miNew;
                }
            }
        }

        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation;
        for (auto& [offset, tensor] : lastOi) {
            aggregation.emplace_back(tensor, offset);
        }
        result = Assemble(aggregation);
        assert(result.GetShape()[0] == b * s);
        assert(result.GetShape()[1] == n * d);
    }
    return result;
}

Tensor MultiAttention(const Tensor& hiddenStates, const Tensor& weight, const Tensor& m, const Tensor& l,
                      const AttentionDims& atDims, const AttentionVecTileConfig& vecCfg,
                      const AttentionCubeTileConfig& cubeCfg)
{
    Tensor result;
    LLAMA_FUNCTION(MultiAttention)
    {
        auto x = Cast(hiddenStates, DataType::DT_FP16);

        auto qkv = Matrix::Matmul(DataType::DT_FP16, x, weight, false, false);
        auto q = View(qkv, hiddenStates.GetShape(), {0, 0});
        auto k = View(qkv, hiddenStates.GetShape(), {0, hiddenStates.GetShape()[1]});
        auto v = View(qkv, hiddenStates.GetShape(), {0, hiddenStates.GetShape()[1] * 2});

        result = FlashAttention(q, k, v, m, l, atDims, vecCfg, cubeCfg);
    }
    return result;
}

Tensor LlamaLayer(Tensor hiddenStates, const Tensor& attnWight, const Tensor& denseWeight, const Tensor& ffnWeight,
                  const AttentionDims& atDims, const AttentionVecTileConfig& vecCfg,
                  const AttentionCubeTileConfig& cubeCfg)
{
    TileShape::Current().SetVecTile(vecCfg.defaultVecTileX, vecCfg.defaultVecTileY);
    SetDefaultL0CubeConfig();
    auto shape = hiddenStates.GetShape();
    auto residual = hiddenStates;
    hiddenStates = RmsNorm(hiddenStates);

    auto bns = atDims.b * atDims.n * atDims.s;
    std::vector<int64_t> shapeReduce = {bns, 1};
    std::vector<float> max(bns, 0);
    std::vector<float> sum(bns, 0);

    Tensor m(DataType::DT_FP32, shapeReduce);
    Tensor l(DataType::DT_FP32, shapeReduce);
    auto attentionOut = MultiAttention(hiddenStates, attnWight, m, l, atDims, vecCfg, cubeCfg);

    auto attentionOutFp16 = Cast(attentionOut, DataType::DT_FP16);
    // Dense
    SetDefaultL0CubeConfig();
    auto denseOut = Matrix::Matmul(DataType::DT_FP32, attentionOutFp16, denseWeight, false, false);
    TileShape::Current().SetVecTile(vecCfg.defaultVecTileX, vecCfg.defaultVecTileY);
    hiddenStates = Add(residual, denseOut);

    // Fully Connected
    residual = hiddenStates;
    hiddenStates = RmsNorm(hiddenStates);

    Tensor mlpRes(DataType::DT_FP32, shape);

    auto a = Cast(hiddenStates, DataType::DT_FP16);
    auto gate = Matrix::Matmul(DataType::DT_FP32, a, ffnWeight, false,
                               false); // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]

    // swish: x / (1 + e^(-x))
    auto swish = Mul(gate, Element(DataType::DT_FP32, F_NEGA_1));
    swish = Exp(swish);
    swish = Add(swish, Element(DataType::DT_FP32, F_1));
    swish = Div(gate, swish);

    // up_proj
    // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]
    auto up = Matrix::Matmul(DataType::DT_FP32, a, ffnWeight, false, false);
    swish = Mul(swish, up);
    auto swishFp16 = Cast(swish, DataType::DT_FP16);

    // down_proj
    // [b*s, n*d*3] [n*d, n*d*3]^T => [b*s, n*d]
    mlpRes = Matrix::Matmul(DataType::DT_FP32, swishFp16, ffnWeight, false, true);
    return Add(residual, mlpRes);
}
} // namespace npu::tile_fwk
