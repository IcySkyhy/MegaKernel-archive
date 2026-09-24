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
 * \file hash_buffer.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <array>

namespace npu::tile_fwk {

class HashBuffer : public std::basic_string<char32_t> {
public:
    template <typename... Tys>
    explicit HashBuffer(const Tys&... args)
    {
        Update(args...);
    }

    template <typename T>
    T Get(int index) const;

    void Append(uint64_t hash)
    {
        char32_t l = hash & 0xFFFFFFFF;
        char32_t h = hash >> 32;
        this->push_back(l);
        this->push_back(h);
    }

    void Append(int32_t n) { Append(static_cast<char32_t>(n)); }

    void Append(int64_t n) { Append(static_cast<uint64_t>(n)); }

    void Append(char32_t n) { this->push_back(n); }

    template <typename T>
    void Append(const std::vector<T>& v)
    {
        for (const auto& i : v) {
            this->Append(i);
        }
    }

    void Append(const std::string& s) { this->insert(this->end(), s.begin(), s.end()); }

    template <typename T, T N>
    void Append(const std::array<int, N>& v)
    {
        this->insert(this->end(), v.begin(), v.end());
    }

    template <typename T, T N>
    void Append(const std::array<int64_t, N>& v)
    {
        for (const auto& i : v) {
            this->Append(i);
        }
    }

    void Update() {}

    template <typename Ty>
    void Update(const Ty& arg)
    {
        Append(arg);
    }

    template <typename Ty, typename... Tys>
    void Update(const Ty& arg, const Tys&... args)
    {
        Append(arg);
        Update(args...);
    }

    std::size_t Digest() const
    {
        auto hash = std::hash<std::basic_string<char32_t>>()(*this);
        return hash;
    }
};

} // namespace npu::tile_fwk
