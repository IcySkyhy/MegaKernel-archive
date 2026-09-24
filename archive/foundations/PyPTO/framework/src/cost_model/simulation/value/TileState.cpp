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
 * \file TileState.cpp
 * \brief
 */

#include "cost_model/simulation/value/TileState.h"

namespace CostModel {

TileState::TileStateKeyTy TileState::TileKey(int rawMagic, OperandType bufType, std::vector<int>& shape,
                                             std::vector<int>& offset)
{
    TileState::TileStateKeyTy k(rawMagic, bufType, shape, offset);
    return k;
}

void TileState::Store(TileStateKeyTy& key, uint64_t value)
{
    if (store_.find(key) == store_.end()) {
        store_[key] = std::make_shared<TileMetaData>();
    }

    store_[key]->Put(value);

    if (storeMap_.find(key.rawMagic) == storeMap_.end()) {
        storeMap_[key.rawMagic] = std::unordered_set<TileStateKeyTy, TileStateKeyHash>();
    }
    storeMap_[key.rawMagic].insert(key);
}

void TileState::Ref(TileStateKeyTy& dst, TileStateKeyTy& src)
{
    if (store_.find(src) == store_.end()) {
        constexpr uint64_t missingValue = 88888888;
        store_[src] = std::make_shared<TileMetaData>();
        store_[src]->Put(missingValue);
    }
    store_[dst] = store_[src];
}

static bool TileContains(const TileState::TileStateKeyTy& p, const TileState::TileStateKeyTy& t)
{
    if (p.bufType == t.bufType) {
        return true;
    }
    return false;
}

uint64_t TileState::Load(TileStateKeyTy& key)
{
    if (store_.find(key) != store_.end()) {
        return store_[key]->Value();
    }

    if (storeMap_.find(key.rawMagic) != storeMap_.end()) {
        for (auto& s : storeMap_[key.rawMagic]) {
            auto value = store_[s]->Value();
            if (TileContains(s, key)) {
                Store(key, value);
                return value;
            }
        }
    }

    return 0;
}

size_t TileState::Order(TileStateKeyTy& key)
{
    if (store_.find(key) == store_.end()) {
        return 0;
    }
    return store_[key]->Order();
}
} // namespace CostModel
