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
 * \file balanced_coin_flips.h
 * \brief
 */

#ifndef OSP_BALANCED_COIN_FLIPS_H
#define OSP_BALANCED_COIN_FLIPS_H

#include <random>
#include <vector>

namespace npu::tile_fwk {
namespace osp {
class BiasedRandom {
public:
    bool GetFlip();

    BiasedRandom(std::size_t seed = 1729U) : gen_(seed), trueBias_(0) {};

private:
    /// @brief Random number generator
    std::mt19937 gen_;
    /// @brief Biases the coin towards true
    int trueBias_;
};

/// @brief Generates the Thue Morse Sequence
/// @param shift Starting point in the sequence
class ThueMorseSequence {
public:
    bool GetFlip();

    ThueMorseSequence(long unsigned int shift = 0U) : next_(shift) { sequence_.emplace_back(false); }

private:
    long unsigned int next_;
    std::vector<bool> sequence_;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_BALANCED_COIN_FLIPS_H
