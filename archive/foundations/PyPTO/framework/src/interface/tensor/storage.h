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
 * \file storage.h
 * \brief
 */

#pragma once
#include <string>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "tilefwk/data_type.h"

namespace npu::tile_fwk {
class Storage {
public:
    MemoryType type_;
    int64_t id_;

    uint64_t start_ = 0;
    uint64_t length_ = 0; // bytes
    Storage(MemoryType type, int64_t id, uint64_t length) : type_(type), id_(id), length_(length) {}

    nlohmann::json DumpJson() const
    {
        nlohmann::json ret;
        ret["type"] = type_;
        ret["id"] = id_;
        ret["start"] = start_;
        ret["length"] = length_;
        return ret;
    }

    static std::shared_ptr<Storage> LoadJson(const nlohmann::json& json)
    {
        std::shared_ptr<Storage> ret = std::make_shared<Storage>(MemoryType::MEM_UNKNOWN, -1, 0);
        ret->type_ = static_cast<MemoryType>(json["type"].get<int>());
        ret->id_ = json["id"].get<int64_t>();
        ret->start_ = json["start"].get<uint64_t>();
        ret->length_ = json["length"].get<uint64_t>();
        return ret;
    }
};
} // namespace npu::tile_fwk
