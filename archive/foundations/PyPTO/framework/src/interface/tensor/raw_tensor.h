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
 * \file raw_tensor.h
 * \brief
 */

#pragma once

#include <cstdlib>
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "tilefwk/tilefwk.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/data_type.h"
#include "tilefwk/pypto_fwk_log.h"

using Json = nlohmann::json;

namespace npu::tile_fwk {
class RawTensor {
public:
    int rawmagic;
    int memoryId{-1};
    int actualRawmagic = -1;
    Shape rawshape;
    std::vector<SymbolicScalar> dynRawShape;
    DataType datatype;
    TileOpFormat format;
    std::string symbol;
    uint64_t addrOffset = UINT64_MAX;
    RawTensor(DataType t, std::vector<int64_t> tshape, TileOpFormat format = TileOpFormat::TILEOP_ND,
              std::string tname = "", int trawmagic = -1);

    RawTensor(RawTensor&&) = delete;
    RawTensor(const RawTensor& other) = delete;
    RawTensor& operator=(RawTensor&&) = delete;
    RawTensor& operator=(const RawTensor&) = delete;

    Json DumpJson() const;
    static std::shared_ptr<RawTensor> LoadJson(const Json& rawTensorDump);

    std::string DumpType() const;
    std::string DumpSSA(bool showType = true, bool showSymbol = true) const;

    std::string Dump() const;

    bool IsDummy() const;
    void SetIsDummy(bool dummy = true);

    void AddRefCount(int value);
    void SetRefCount(int value) { refCount_ = value; }
    auto GetRefCount() const { return refCount_; }

    int GetRawMagic() const
    {
        if (actualRawmagic != -1) {
            return actualRawmagic;
        } else {
            return rawmagic;
        }
    }
    const std::string& GetSymbol() const { return symbol; }
    void SetSymbol(std::string s) { symbol = std::move(s); }
    DataType GetDataType() const { return datatype; }
    const Shape& GetRawShape() const { return rawshape; }
    int64_t GetRawShapeSize() const;
    int64_t GetRawDataSize() const;
    const std::vector<SymbolicScalar>& GetDynRawShape() const { return dynRawShape; }
    SymbolicScalar GetDynRawShape(int axis) const { return dynRawShape[axis]; }
    void UpdateDynRawShape(const std::vector<SymbolicScalar>& dynShape) { dynRawShape = dynShape; }
    void UpdateRawShape(const std::vector<int64_t>& trawShape)
    {
        rawshape = trawShape;
        dynRawShape = SymbolicScalar::FromConcrete(trawShape);
    }

    void SetCachePolicy(CachePolicy policy, bool value)
    {
        cachePolicy_[static_cast<int>(policy)] = value;
        if (value && (cachePolicy_[static_cast<int>(CachePolicy::PREFETCH)] ==
                      cachePolicy_[static_cast<int>(CachePolicy::NONE_CACHEABLE)])) {
            FE_LOGW("Prefetch and none cacheable can not apply at same time, use the first config policy.");
            cachePolicy_[static_cast<int>(policy)] = false;
        }
    }

    bool GetCachePolicy(CachePolicy policy) const { return cachePolicy_[static_cast<int>(policy)]; }

private:
    bool isDummy_{false};
    int refCount_{0}; // 被 npu::tile_fwk::Tensor引用的次数，用于outcast自动推导
    bool cachePolicy_[static_cast<int>(CachePolicy::MAX_NUM)] = {false};
};
} // namespace npu::tile_fwk
