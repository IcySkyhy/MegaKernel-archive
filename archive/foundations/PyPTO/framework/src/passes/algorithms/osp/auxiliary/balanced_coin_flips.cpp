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
 * \file balanced_coin_flips.cpp
 * \brief
 */

#include "passes/algorithms/osp/auxiliary/balanced_coin_flips.h"

using namespace npu::tile_fwk::osp;

bool BiasedRandom::GetFlip()
{
    constexpr int genuineRandomSize = 3;
    constexpr int numberTwo = 2;
    int dieSize = numberTwo * genuineRandomSize + abs(trueBias_);
    std::uniform_int_distribution<int> distrib(0, dieSize - 1);
    int flip = distrib(gen_);
    if (trueBias_ >= 0) {
        if (flip >= genuineRandomSize) {
            trueBias_--;
            return true;
        } else {
            trueBias_++;
            return false;
        }
    } else {
        if (flip >= genuineRandomSize) {
            trueBias_++;
            return false;
        } else {
            trueBias_--;
            return true;
        }
    }
}

bool ThueMorseSequence::GetFlip()
{
    for (long unsigned int i = sequence_.size(); i <= next_; i++) {
        constexpr long unsigned int numberTwo = 2U;
        if (i % numberTwo == 0) {
            sequence_.emplace_back(sequence_[i / numberTwo]);
        } else {
            sequence_.emplace_back(!sequence_[i / numberTwo]);
        }
    }
    return sequence_[next_++];
}
