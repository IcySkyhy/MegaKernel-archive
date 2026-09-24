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
 * \file integral_range.h
 * \brief
 */

#ifndef OSP_INTEGRAL_RANGE_H
#define OSP_INTEGRAL_RANGE_H

#include <iterator>
#include <type_traits>

namespace npu::tile_fwk {
namespace osp {
/**
 * @brief A lightweight range class for iterating over a sequence of integral values.
 *
 * This class provides a view over a range of integers [start, finish), allowing iteration
 * without allocating memory for a container. It is useful for iterating over vertex indices
 * in a graph or any other sequence of numbers.
 *
 * @tparam T The integral type of the values (e.g., int, unsigned, size_t).
 */
template <typename T>
class IntegralRange {
    static_assert(std::is_integral<T>::value, "IntegralRange requires an integral type");

public:
    /**
     * @brief Iterator for the IntegralRange.
     *
     * This iterator satisfies the RandomAccessIterator concept.
     */
    class IntegralIterator { // public for std::reverse_iterator
    public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = void; // Not a real pointer
        using reference = T;  // Not a real reference

        /**
         * @brief Proxy object to support operator-> for integral types.
         */
        struct ArrowProxy {
            T value_;
            constexpr const T* operator->() const noexcept { return &value_; }
        };

        /**
         * @brief Default constructor. Initializes iterator to 0.
         */
        constexpr IntegralIterator() noexcept : current_(0) {}

        /**
         * @brief Constructs an iterator pointing to the given value.
         * @param start The starting value.
         */
        explicit constexpr IntegralIterator(value_type start) noexcept : current_(start) {}

        constexpr IntegralIterator(const IntegralIterator&) noexcept = default;
        constexpr IntegralIterator& operator=(const IntegralIterator&) noexcept = default;
        ~IntegralIterator() = default;

        /**
         * @brief Dereference operator.
         * @return The current integral value.
         */
        [[nodiscard]] constexpr value_type operator*() const noexcept { return current_; }

        /**
         * @brief Arrow operator.
         * @return A proxy object that allows access to the address of the value.
         */
        [[nodiscard]] constexpr ArrowProxy operator->() const noexcept { return ArrowProxy{current_}; }

        constexpr IntegralIterator& operator++() noexcept
        {
            ++current_;
            return *this;
        }

        constexpr IntegralIterator operator++(int) noexcept
        {
            IntegralIterator temp = *this;
            ++(*this);
            return temp;
        }

        constexpr IntegralIterator& operator--() noexcept
        {
            --current_;
            return *this;
        }

        constexpr IntegralIterator operator--(int) noexcept
        {
            IntegralIterator temp = *this;
            --(*this);
            return temp;
        }

        [[nodiscard]] constexpr bool operator==(const IntegralIterator& other) const noexcept
        {
            return current_ == other.current_;
        }

        [[nodiscard]] constexpr bool operator!=(const IntegralIterator& other) const noexcept
        {
            return !(*this == other);
        }

        constexpr IntegralIterator& operator+=(difference_type n) noexcept
        {
            current_ = static_cast<value_type>(current_ + n);
            return *this;
        }

        [[nodiscard]] constexpr IntegralIterator operator+(difference_type n) const noexcept
        {
            IntegralIterator temp = *this;
            return temp += n;
        }

        [[nodiscard]] friend constexpr IntegralIterator operator+(difference_type n,
                                                                  const IntegralIterator& it) noexcept
        {
            return it + n;
        }

        constexpr IntegralIterator& operator-=(difference_type n) noexcept
        {
            current_ = static_cast<value_type>(current_ - n);
            return *this;
        }

        [[nodiscard]] constexpr IntegralIterator operator-(difference_type n) const noexcept
        {
            IntegralIterator temp = *this;
            return temp -= n;
        }

        [[nodiscard]] constexpr difference_type operator-(const IntegralIterator& other) const noexcept
        {
            return static_cast<difference_type>(current_) - static_cast<difference_type>(other.current_);
        }

        [[nodiscard]] constexpr value_type operator[](difference_type n) const noexcept { return *(*this + n); }

        [[nodiscard]] constexpr bool operator<(const IntegralIterator& other) const noexcept
        {
            return current_ < other.current_;
        }

        [[nodiscard]] constexpr bool operator>(const IntegralIterator& other) const noexcept
        {
            return current_ > other.current_;
        }

        [[nodiscard]] constexpr bool operator<=(const IntegralIterator& other) const noexcept
        {
            return current_ <= other.current_;
        }

        [[nodiscard]] constexpr bool operator>=(const IntegralIterator& other) const noexcept
        {
            return current_ >= other.current_;
        }

    private:
        value_type current_;
    };

    using ReverseIntegralIterator = std::reverse_iterator<IntegralIterator>;

    /**
     * @brief Constructs a range [0, end).
     * @param end_ The exclusive upper bound.
     */
    constexpr IntegralRange(T end) noexcept : start_(static_cast<T>(0)), finish_(end) {}

    /**
     * @brief Constructs a range [start, end).
     * @param start_ The inclusive lower bound.
     * @param end_ The exclusive upper bound.
     */
    constexpr IntegralRange(T start, T end) noexcept : start_(start), finish_(end) {}

    [[nodiscard]] constexpr IntegralIterator begin() const noexcept { return IntegralIterator(start_); }

    [[nodiscard]] constexpr IntegralIterator cbegin() const noexcept { return IntegralIterator(start_); }

    [[nodiscard]] constexpr IntegralIterator end() const noexcept { return IntegralIterator(finish_); }

    [[nodiscard]] constexpr IntegralIterator cend() const noexcept { return IntegralIterator(finish_); }

    [[nodiscard]] constexpr ReverseIntegralIterator rbegin() const noexcept { return ReverseIntegralIterator(end()); }

    [[nodiscard]] constexpr ReverseIntegralIterator crbegin() const noexcept { return ReverseIntegralIterator(cend()); }

    [[nodiscard]] constexpr ReverseIntegralIterator rend() const noexcept { return ReverseIntegralIterator(begin()); }

    [[nodiscard]] constexpr ReverseIntegralIterator crend() const noexcept { return ReverseIntegralIterator(cbegin()); }

    /**
     * @brief Returns the number of elements in the range.
     * @return The size of the range.
     */
    [[nodiscard]] constexpr auto size() const noexcept { return finish_ - start_; }

    /**
     * @brief Checks if the range is empty.
     * @return True if the range is empty, false otherwise.
     */
    [[nodiscard]] constexpr bool empty() const noexcept { return start_ == finish_; }

private:
    T start_;
    T finish_;
};
} // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_INTEGRAL_RANGE_HPP
