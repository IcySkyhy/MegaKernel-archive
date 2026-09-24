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
 * \file TileState.h
 * \brief
 */

#pragma once

#include <unordered_map>

#include "interface/inner/hash_buffer.h"
#include "interface/utils/common.h"
#include "cost_model/simulation/common/ISA.h"

namespace CostModel {
class TileMetaData {
public:
    explicit TileMetaData() = default;

    uint64_t Value() { return value.empty() ? 0 : value.back(); };

    size_t Order() { return value.size(); }

    void Put(uint64_t v) { value.push_back(v); }

private:
    std::vector<uint64_t> value;
};

class TileState {
public:
    class TileStateKeyTy {
    public:
        TileStateKeyTy() = default;

        ~TileStateKeyTy() = default;

        TileStateKeyTy(int& m, OperandType& b, std::vector<int>& s, std::vector<int>& o)
            : rawMagic(m), bufType(b), shape(s), offset(o)
        {}

        bool operator==(const TileStateKeyTy& other) const
        {
            return rawMagic == other.rawMagic && bufType == other.bufType && shape == other.shape &&
                   offset == other.offset;
        }

        std::string Dump()
        {
            std::stringstream oss;
            oss << rawMagic << ", " << OperandTypeToStr(bufType) << ", ";
            oss << "offset = (";
            for (size_t i = 0; i < offset.size(); ++i) {
                oss << offset[i];
                if (i != offset.size() - 1) {
                    oss << ",";
                }
            }
            oss << "), ";
            oss << "shape = (";
            for (size_t i = 0; i < shape.size(); ++i) {
                oss << shape[i];
                if (i != shape.size() - 1) {
                    oss << ",";
                }
            }
            oss << ")";
            return oss.str();
        }

    public:
        int rawMagic;
        OperandType bufType;
        std::vector<int> shape;
        std::vector<int> offset;
    };

    class TileStateKeyHash {
    public:
        std::size_t operator()(const TileStateKeyTy& key) const noexcept
        {
            npu::tile_fwk::HashBuffer buffer(key.rawMagic, key.bufType, key.shape, key.offset);
            return buffer.Digest();
        }
    };

private:
    std::unordered_map<int, std::unordered_set<TileStateKeyTy, TileStateKeyHash>> storeMap_;
    std::unordered_map<TileStateKeyTy, std::shared_ptr<TileMetaData>, TileStateKeyHash> store_;

public:
    static TileStateKeyTy TileKey(int rawMagic, OperandType bufType, std::vector<int>& shape, std::vector<int>& offset);

    void Store(TileStateKeyTy& key, uint64_t value);

    uint64_t Load(TileStateKeyTy& key);

    size_t Order(TileStateKeyTy& key);

    void Ref(TileStateKeyTy& dst, TileStateKeyTy& src);
};
} // namespace CostModel
