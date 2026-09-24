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
 * \file tensor_offset.h
 * \brief
 */

#pragma once

#include <vector>
#include <set>
#include <string>
#include <memory>
#include <unordered_set>
#include <functional>

#include "tilefwk/error.h"
#include "symbolic_scalar.h"

namespace npu::tile_fwk {

/* replace std::vector<int64_t> using TensorOffset for further unify concrete offset and symbolic offset */
class TensorOffset {
public:
    TensorOffset(const std::vector<int64_t>& offset, const std::vector<SymbolicScalar>& dynOffset)
        : offset_(offset), dynOffset_(dynOffset)
    {}

    const std::vector<int64_t>& GetOffset() const { return offset_; }
    const std::vector<SymbolicScalar>& GetDynOffset() const { return dynOffset_; }
    template <typename Tret, typename Tlhs, typename Trhs>
    static std::vector<Tret> AddRaw(const std::vector<Tlhs>& lhs, const std::vector<Trhs>& rhs)
    {
        FE_ASSERT(FeError::INVALID_VAL, lhs.size() == rhs.size())
            << "lhs.size():" << lhs.size() << ", rhs.size():" << rhs.size();
        std::vector<Tret> ret(lhs.size());
        for (size_t k = 0; k < lhs.size(); k++) {
            ret[k] = lhs[k] + rhs[k];
        }
        return ret;
    }

    static std::vector<int64_t> Add(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs)
    {
        return AddRaw<int64_t, int64_t, int64_t>(lhs, rhs);
    }
    static std::vector<SymbolicScalar> Add(const std::vector<SymbolicScalar>& lhs, const std::vector<int64_t>& rhs)
    {
        return AddRaw<SymbolicScalar, SymbolicScalar, int64_t>(lhs, rhs);
    }
    static std::vector<SymbolicScalar> Add(const std::vector<int64_t>& lhs, const std::vector<SymbolicScalar>& rhs)
    {
        return AddRaw<SymbolicScalar, int64_t, SymbolicScalar>(lhs, rhs);
    }
    static std::vector<SymbolicScalar> Add(const std::vector<SymbolicScalar>& lhs,
                                           const std::vector<SymbolicScalar>& rhs)
    {
        return AddRaw<SymbolicScalar, SymbolicScalar, SymbolicScalar>(lhs, rhs);
    }

    static std::vector<int64_t> Zero(const std::vector<int64_t>& off) { return std::vector<int64_t>(off.size(), 0); }
    static bool IsZero(const std::vector<int64_t>& off) { return Zero(off) == off; }

    static std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> Add(const std::vector<int64_t>& lhs,
                                                                            const std::vector<SymbolicScalar>& lhsDyn,
                                                                            const std::vector<int64_t>& rhs,
                                                                            const std::vector<SymbolicScalar>& rhsDyn)
    {
        FE_ASSERT(FeError::INVALID_VAL, lhs.size() == rhs.size())
            << "lhs.size():" << lhs.size() << ", rhs.size():" << rhs.size();
        std::vector<int64_t> ret = Add(lhs, rhs);

        std::vector<SymbolicScalar> retDyn;

        if (lhsDyn.size() != 0 && rhsDyn.size() != 0) {
            retDyn = Add(lhsDyn, rhsDyn);
        } else if (lhsDyn.size() != 0) {
            retDyn = Add(lhsDyn, rhs);
        } else if (rhsDyn.size() != 0) {
            retDyn = Add(lhs, rhsDyn);
        }
        return std::make_pair(ret, retDyn);
    }

    static std::vector<int64_t> Sub(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs)
    {
        FE_ASSERT(FeError::INVALID_VAL, lhs.size() == rhs.size())
            << "lhs.size():" << lhs.size() << ", rhs.size():" << rhs.size();
        std::vector<int64_t> result(lhs.size());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), [](int a, int b) { return a - b; });
        return result;
    }

    static std::vector<SymbolicScalar> Sub(const std::vector<SymbolicScalar>& lhs, const std::vector<int64_t>& rhs)
    {
        if (lhs.size() == 0) {
            return {};
        }
        FE_ASSERT(FeError::INVALID_VAL, lhs.size() == rhs.size())
            << "lhs.size():" << lhs.size() << ", rhs.size():" << rhs.size();
        std::vector<SymbolicScalar> result(lhs.size());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                       [](const SymbolicScalar& a, int b) { return a - b; });
        return result;
    }

public:
    const std::vector<int64_t>& offset_;
    const std::vector<SymbolicScalar>& dynOffset_;
};
} // namespace npu::tile_fwk
