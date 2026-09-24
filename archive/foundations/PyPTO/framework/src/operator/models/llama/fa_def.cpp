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
 * \file fa_def.cpp
 * \brief
 */

#include "operator/models/llama/llama_def.h"

namespace npu::tile_fwk {

static constexpr int NUM_64 = 64;
static constexpr int NUM_128 = 128;

Tensor FlashAttentionNew(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& m, const Tensor& l,
                         const AttentionDims& atDims)
{
    (void)m;
    (void)l;
    int dim0 = q.GetShape()[0];
    int dim1 = q.GetShape()[1];
    int b = atDims.b;
    int n = atDims.n;
    int s = dim0 / b;
    int d = dim1 / n;
    int singleM = atDims.singleM; // 128
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

    for (int bIdx = 0; bIdx < b; bIdx++) {
        for (int nIdx = 0; nIdx < n; nIdx++) {
            for (int s2Idx = 0; s2Idx < s2Loop; s2Idx++) {
                auto kj = View(k, {singleN, d}, {bIdx * s + s2Idx * singleN, nIdx * d});
                auto vj = View(v, {singleN, d}, {bIdx * s + s2Idx * singleN, nIdx * d});

                for (int s1Idx = 0; s1Idx < s1Loop; s1Idx++) {
                    auto qi = View(q, {singleM, d}, {bIdx * s + s1Idx * singleM, nIdx * d});
                    std::vector<int64_t> oiOffset = {bIdx * s + s1Idx * singleM, nIdx * d};
                    std::vector<int64_t> liOffset = {(bIdx * n + nIdx) * s + s1Idx * singleM, 0};
                    std::vector<int64_t> miOffset = {(bIdx * n + nIdx) * s + s1Idx * singleM, 0};
                    // [128, 128], [128, 1024] => [128, 1024]
                    auto sij = Matrix::Matmul(DataType::DT_FP32, qi, kj, false, true);

                    auto tildaMij = Amax(sij, -1, true);
                    auto tsub = Sub(sij, tildaMij);
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, DataType::DT_FP16);
                    auto tildaLij = Sum(tildaPij, -1, true);

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
    }

    std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation;
    for (auto& [offset, tensor] : lastOi) {
        aggregation.emplace_back(tensor, offset);
    }

    result = Assemble(aggregation);
    assert(result.GetShape()[0] == b * s);
    assert(result.GetShape()[1] == n * d);

    return result;
}

} // namespace npu::tile_fwk
