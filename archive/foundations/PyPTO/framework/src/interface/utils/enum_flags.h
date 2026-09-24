/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file enum_flags.h
 * \brief A robust bitset keyed by a scoped enum.
 */
#pragma once

#include <cstdint>
#include <type_traits>

namespace npu::tile_fwk {

// EnumFlags is a set of enumerators of E stored as a bitmask.
//
// Each enumerator e occupies bit static_cast<storage_type>(e). Enumerators whose
// underlying value is negative or not less than the bit width of Storage are
// mapped to a zero mask instead of triggering undefined behaviour, so the type
// stays well-defined for any enumerator value. Storage must be an unsigned
// integral type (defaults to uint32_t).
template <typename E, typename Storage = std::uint32_t>
class EnumFlags {
    static_assert(std::is_enum<E>::value, "EnumFlags: E must be an enum type");
    static_assert(std::is_unsigned<Storage>::value && std::is_integral<Storage>::value,
                  "EnumFlags: Storage must be an unsigned integral type");

public:
    using enum_type = E;
    using storage_type = Storage;

    constexpr EnumFlags() noexcept = default;
    constexpr explicit EnumFlags(storage_type bits) noexcept : bits_(bits) {}
    // Single-flag construction, e.g. EnumFlags<E>{E::A}.
    constexpr EnumFlags(E e) noexcept : bits_(Bit(e)) {}

    constexpr EnumFlags& Add(E e) noexcept
    {
        bits_ |= Bit(e);
        return *this;
    }
    constexpr EnumFlags& Remove(E e) noexcept
    {
        bits_ &= ~Bit(e);
        return *this;
    }
    constexpr bool Contains(E e) const noexcept { return (bits_ & Bit(e)) != storage_type{0}; }

    constexpr bool Empty() const noexcept { return bits_ == storage_type{0}; }
    constexpr bool Any() const noexcept { return bits_ != storage_type{0}; }
    constexpr bool Overlaps(EnumFlags other) const noexcept { return (bits_ & other.bits_) != storage_type{0}; }
    constexpr bool ContainsAll(EnumFlags other) const noexcept { return (bits_ & other.bits_) == other.bits_; }

    constexpr storage_type Value() const noexcept { return bits_; }

    constexpr EnumFlags operator|(EnumFlags rhs) const noexcept { return EnumFlags(bits_ | rhs.bits_, internal{}); }
    constexpr EnumFlags operator&(EnumFlags rhs) const noexcept { return EnumFlags(bits_ & rhs.bits_, internal{}); }
    constexpr EnumFlags operator^(EnumFlags rhs) const noexcept { return EnumFlags(bits_ ^ rhs.bits_, internal{}); }
    constexpr EnumFlags operator~() const noexcept { return EnumFlags(~bits_, internal{}); }
    constexpr EnumFlags& operator|=(EnumFlags rhs) noexcept
    {
        bits_ |= rhs.bits_;
        return *this;
    }
    constexpr EnumFlags& operator&=(EnumFlags rhs) noexcept
    {
        bits_ &= rhs.bits_;
        return *this;
    }
    constexpr EnumFlags& operator^=(EnumFlags rhs) noexcept
    {
        bits_ ^= rhs.bits_;
        return *this;
    }

    friend constexpr bool operator==(EnumFlags lhs, EnumFlags rhs) noexcept { return lhs.bits_ == rhs.bits_; }
    friend constexpr bool operator!=(EnumFlags lhs, EnumFlags rhs) noexcept { return lhs.bits_ != rhs.bits_; }

private:
    struct internal {};
    constexpr EnumFlags(storage_type bits, internal) noexcept : bits_(bits) {}

    using Underlying = typename std::underlying_type<E>::type;
    using UnsignedUnderlying = typename std::make_unsigned<Underlying>::type;

    static constexpr Storage Bit(E e) noexcept
    {
        // Cast through the unsigned underlying type so signed enums are handled,
        // then widen to uint64_t to test against the Storage bit width safely.
        const std::uint64_t raw = static_cast<std::uint64_t>(
            static_cast<UnsignedUnderlying>(static_cast<Underlying>(e)));
        constexpr std::uint64_t width = sizeof(Storage) * 8ULL;
        return (raw >= width) ? storage_type{0} : static_cast<storage_type>(storage_type{1} << raw);
    }

    storage_type bits_{0};
};

} // namespace npu::tile_fwk
