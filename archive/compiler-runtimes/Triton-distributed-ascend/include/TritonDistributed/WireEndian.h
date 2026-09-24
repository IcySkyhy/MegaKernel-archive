////////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////
// Shared little-/big-endian helpers for OpGraph.h and TileGraph.h wire I/O.
// Including both graph headers in one TU is safe because these helpers live
// here once behind this include guard.

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace triton_dist {
namespace detail {

inline constexpr uint8_t kEndianLE = 0;
inline constexpr uint8_t kEndianBE = 1;

// Compile-time host endianness detection. Prefers __BYTE_ORDER__ (GCC/Clang);
// falls back to little-endian (the overwhelmingly common case) otherwise.
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
inline constexpr bool kHostBigEndian = true;
#else
inline constexpr bool kHostBigEndian = false;
#endif

inline uint16_t bswap16(uint16_t v) {
#if defined(_MSC_VER)
    return _byteswap_ushort(v);
#else
    return __builtin_bswap16(v);
#endif
}
inline uint32_t bswap32(uint32_t v) {
#if defined(_MSC_VER)
    return _byteswap_ulong(v);
#else
    return __builtin_bswap32(v);
#endif
}
inline uint64_t bswap64(uint64_t v) {
#if defined(_MSC_VER)
    return _byteswap_uint64(v);
#else
    return __builtin_bswap64(v);
#endif
}

// Generic byte-swap dispatched by size. Handles signed/unsigned integers of all
// widths (via two's-complement-preserving casts) and IEEE floating-point types
// (via memcpy). Single-byte types are returned unchanged.
template <typename T>
inline T byteswap(T v) {
    if constexpr (std::is_same_v<T, double>) {
        uint64_t u; std::memcpy(&u, &v, sizeof(u));
        u = bswap64(u);
        T r; std::memcpy(&r, &u, sizeof(r));
        return r;
    } else if constexpr (std::is_same_v<T, float>) {
        uint32_t u; std::memcpy(&u, &v, sizeof(u));
        u = bswap32(u);
        T r; std::memcpy(&r, &u, sizeof(r));
        return r;
    } else if constexpr (sizeof(T) == 1) {
        return v;
    } else if constexpr (sizeof(T) == 2) {
        return static_cast<T>(bswap16(static_cast<uint16_t>(v)));
    } else if constexpr (sizeof(T) == 4) {
        return static_cast<T>(bswap32(static_cast<uint32_t>(v)));
    } else if constexpr (sizeof(T) == 8) {
        return static_cast<T>(bswap64(static_cast<uint64_t>(v)));
    } else {
        static_assert(sizeof(T) == 0, "unsupported byteswap width");
    }
}

// Read a multi-byte integer/float from an unaligned source, optionally
// byte-swapping when the wire endianness differs from the host endianness.
template <typename T>
inline T load_field(const void* p, bool swap) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    if (swap) v = byteswap(v);
    return v;
}

// Store a multi-byte integer/float to an unaligned destination, optionally
// byte-swapping when the wire endianness differs from the host endianness.
template <typename T>
inline void store_field(void* p, T v, bool swap) {
    if (swap) v = byteswap(v);
    std::memcpy(p, &v, sizeof(T));
}

inline void append_bytes(std::vector<uint8_t>& out, const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    out.insert(out.end(), b, b + n);
}

}  // namespace detail
}  // namespace triton_dist
