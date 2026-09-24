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
 * \file type_traits.h
 * \brief
 */

#ifndef TILEOP_UTILS_TYPE_TRAITS_H
#define TILEOP_UTILS_TYPE_TRAITS_H
namespace Std {
// enable_if
template <bool, typename Tp = void>
struct enable_if {};

template <typename Tp>
struct enable_if<true, Tp> {
    using type = Tp;
};

template <bool Bp, typename Tp = void>
using enable_if_t = typename enable_if<Bp, Tp>::type;

// integral_constant
template <typename Tp, Tp v>
struct integral_constant {
    static constexpr const Tp value = v;
    using value_type = Tp;
    using type = integral_constant;

    [ host, aicore ] inline constexpr operator value_type() const noexcept { return value; }

    [ host, aicore ] inline constexpr value_type operator()() const noexcept { return value; }
};

template <typename Tp, Tp v>
constexpr const Tp integral_constant<Tp, v>::value;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool b>
using bool_constant = integral_constant<bool, b>;

template <size_t v>
using Int = integral_constant<size_t, v>;

// is_same
template <typename Tp, typename Up>
struct is_same : public false_type {};

template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

template <typename Tp, typename Up>
constexpr bool is_same_v = false;

template <typename Tp>
constexpr bool is_same_v<Tp, Tp> = true;

template <typename Tp, typename Up>
using IsSame = bool_constant<is_same_v<Tp, Up>>;

template <typename Tp, typename Up>
using IsNotSame = bool_constant<!is_same_v<Tp, Up>>;

// remove_const
template <typename Tp>
struct remove_const {
    using type = Tp;
};

template <typename Tp>
struct remove_const<const Tp> {
    using type = Tp;
};

template <typename Tp>
using remove_const_t = typename remove_const<Tp>::type;

// remove_volatile
template <typename Tp>
struct remove_volatile {
    using type = Tp;
};

template <typename Tp>
struct remove_volatile<volatile Tp> {
    using type = Tp;
};

template <typename Tp>
using remove_volatile_t = typename remove_volatile<Tp>::type;

// remove_cv
template <typename Tp>
struct remove_cv {
    using type = remove_volatile_t<remove_const_t<Tp>>;
};

template <typename Tp>
using remove_cv_t = remove_volatile_t<remove_const_t<Tp>>;

// remove_reference
template <typename Tp>
struct remove_reference {
    using type = Tp;
};

template <typename Tp>
struct remove_reference<Tp&> {
    using type = Tp;
};

template <typename Tp>
struct remove_reference<Tp&&> {
    using type = Tp;
};

template <typename Tp>
using remove_reference_t = typename remove_reference<Tp>::type;

// is_array
template <typename Tp>
struct is_array : public false_type {};

template <typename Tp>
struct is_array<Tp[]> : public true_type {};

template <typename Tp, size_t Np>
struct is_array<Tp[Np]> : public true_type {};

template <typename Tp>
constexpr bool is_array_v = is_array<Tp>::value;

// conditional
namespace conditional_impl {
template <bool>
struct IfImpl;

template <>
struct IfImpl<true> {
    template <typename IfRes, typename ElseRes>
    using Select = IfRes;
};

template <>
struct IfImpl<false> {
    template <typename IfRes, typename ElseRes>
    using Select = ElseRes;
};

template <bool Cond, typename IfRes, typename ElseRes>
using If = typename IfImpl<Cond>::template Select<IfRes, ElseRes>;
} // namespace conditional_impl

template <bool Bp, typename If, typename Then>
struct conditional {
    using type = If;
};

template <typename If, typename Then>
struct conditional<false, If, Then> {
    using type = Then;
};

template <bool Bp, typename If, typename Then>
using conditional_t = typename conditional<Bp, If, Then>::type;

// remove_extent
template <typename Tp>
struct remove_extent {
    using type = Tp;
};

template <typename Tp>
struct remove_extent<Tp[]> {
    using type = Tp;
};

template <typename Tp, size_t Np>
struct remove_extent<Tp[Np]> {
    using type = Tp;
};

template <typename Tp>
using remove_extent_t = typename remove_extent<Tp>::type;

// is_reference_v
template <typename Tp>
struct is_lvalue_reference : public false_type {};

template <typename Tp>
struct is_lvalue_reference<Tp&> : public true_type {};

template <typename Tp>
constexpr bool is_lvalue_reference_v = is_lvalue_reference<Tp>::value;

template <typename Tp>
struct is_rvalue_reference : public false_type {};

template <typename Tp>
struct is_rvalue_reference<Tp&&> : public true_type {};

template <typename Tp>
constexpr bool is_rvalue_reference_v = is_rvalue_reference<Tp>::value;

template <typename Tp>
struct is_reference : public false_type {};

template <typename Tp>
struct is_reference<Tp&> : public true_type {};

template <typename Tp>
struct is_reference<Tp&&> : public true_type {};

template <typename Tp>
constexpr bool is_reference_v = is_reference<Tp>::value;

// is_const_v
template <typename Tp>
struct is_const : public false_type {};

template <typename Tp>
struct is_const<Tp const> : public true_type {};

template <typename Tp>
constexpr bool is_const_v = is_const<Tp>::value;

// is_function
template <typename T>
struct is_function : public bool_constant<!(is_reference_v<T> || is_const_v<const T>)> {};

template <typename T>
constexpr bool is_function_v = is_function<T>::value;

// is_referenceable
struct IsReferenceableImpl {
    template <typename Tp>
    [host, aicore] inline static Tp& Test(int32_t);

    template <typename Tp>
    [host, aicore] inline static false_type Test(uint32_t);
};

template <typename Tp>
struct is_referenceable
    : integral_constant<bool, IsNotSame<decltype(IsReferenceableImpl::Test<Tp>(0)), false_type>::value> {};

template <typename Tp>
struct is_void : public is_same<remove_cv_t<Tp>, void> {};

template <typename Tp>
constexpr bool is_void_v = is_void<Tp>::value;

// add_pointer_t
template <typename Tp, bool = is_referenceable<Tp>::value || is_void<Tp>::value>
struct AddPointerImpl {
    using type = remove_reference_t<Tp>*;
};

template <typename Tp>
struct AddPointerImpl<Tp, false> {
    using type = Tp;
};

template <typename Tp>
using add_pointer_t = typename AddPointerImpl<Tp>::type;

template <typename Tp>
struct add_pointer {
    using type = add_pointer_t<Tp>;
};

// decay
template <typename Up, bool>
struct DecayImpl {
    using type = remove_cv_t<Up>;
};

template <typename Up>
struct DecayImpl<Up, true> {
public:
    using type = conditional_t<is_array<Up>::value, remove_extent_t<Up>*,
                               conditional_t<is_function<Up>::value, add_pointer_t<Up>, remove_cv_t<Up>>>;
};

template <typename Tp>
struct decay {
private:
    using Up = remove_reference_t<Tp>;

public:
    using type = typename DecayImpl<Up, is_referenceable<Up>::value>::type;
};

template <typename Tp>
using decay_t = typename decay<Tp>::type;

// is_integral_constant
template <typename T>
struct IsIntegralConstant : Std::false_type {};

template <size_t Value>
struct IsIntegralConstant<Std::Int<Value>> : Std::true_type {};

template <typename T>
constexpr bool IsIntegralConstantV = IsIntegralConstant<T>::value;

} // namespace Std
#endif // TILEOP_UTILS_TYPE_TRAITS_H
