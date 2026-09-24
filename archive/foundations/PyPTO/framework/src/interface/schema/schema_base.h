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
 * \file schema_base.h
 * \brief
 */

#pragma once
#ifndef SCHEMA_TRACE_BASE_H
#define SCHEMA_TRACE_BASE_H

#include <cstdint>
#include <cstddef>
#include <cstring>

#include <fstream>
#include <memory>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "securec.h"

namespace npu::tile_fwk::schema::type {

#define SCHEMA_ATTR_KEYWORD_PREFIX(top) (top ? "#" : "")
#define SCHEMA_ATTR_VALUE_LBOUND "{"
#define SCHEMA_ATTR_VALUE_RBOUND "}"
#define SCHEMA_ATTR_VALUE_SEPARATOR ","
#define SCHEMA_ARRAY_LBOUND "["
#define SCHEMA_ARRAY_RBOUND "]"
#define SCHEMA_ADDRESS_PREFIX "0x"

struct TypeBase {};

template <typename Ty0, typename... Tys>
struct MaxTypeSize {
    constexpr static int size = std::max(static_cast<size_t>(sizeof(Ty0)),
                                         static_cast<size_t>(MaxTypeSize<Tys...>::size));
};
template <typename Ty0>
struct MaxTypeSize<Ty0> {
    constexpr static int size = sizeof(Ty0);
};

template <typename Ty0, typename... Tys>
struct TypeCount {
    constexpr static int size = 1 + TypeCount<Tys...>::size;
};
template <typename Ty0>
struct TypeCount<Ty0> {
    constexpr static int size = 1;
};

template <typename BaseType, char prefix = 0>
struct IntegralTypeBase {
    IntegralTypeBase() = default;
    IntegralTypeBase(int data) : data_(data) {}

    std::string Dump(bool top __attribute__((unused)) = false) const
    {
        std::string data = std::to_string(data_);
        if (prefix != 0) {
            data = std::string(1, prefix) + data;
        }
        return data;
    }

private:
    BaseType data_{0};
};

struct Int32Type : IntegralTypeBase<int32_t, 0> {
    Int32Type() = default;
    template <typename... TyArgs>
    Int32Type(const TyArgs&... args) : IntegralTypeBase<int32_t, 0>(args...)
    {}
};

struct Int64Type : IntegralTypeBase<int64_t, 0> {
    Int64Type() = default;
    template <typename... TyArgs>
    Int64Type(const TyArgs&... args) : IntegralTypeBase<int64_t, 0>(args...)
    {}
};

struct UInt32Type : IntegralTypeBase<uint32_t, 0> {
    UInt32Type() = default;
    template <typename... TyArgs>
    UInt32Type(const TyArgs&... args) : IntegralTypeBase<uint32_t, 0>(args...)
    {}
};

struct UInt64Type : IntegralTypeBase<uint64_t, 0> {
    UInt64Type() = default;
    template <typename... TyArgs>
    UInt64Type(const TyArgs&... args) : IntegralTypeBase<uint64_t, 0>(args...)
    {}
};

template <char prefix = 0>
struct Int64IdType : IntegralTypeBase<int64_t, prefix> {
    Int64IdType() = default;
    template <typename... TyArgs>
    Int64IdType(const TyArgs&... args) : IntegralTypeBase<int64_t, prefix>(args...)
    {}
};

struct AddressType : TypeBase {
    AddressType() = default;
    AddressType(uintptr_t addr) : addr_(addr) {}

    static constexpr int addressWidth = 16;

    std::string Dump(bool top __attribute__((unused)) = false) const
    {
        std::ostringstream oss;
        oss << SCHEMA_ADDRESS_PREFIX << std::hex << addr_;
        return oss.str();
    }

private:
    uintptr_t addr_{0};
};

struct StringType : TypeBase {
    StringType() = default;
    StringType(const std::string& str) : str_(str) {}

    std::string Dump(bool top __attribute__((unused)) = false) const { return "\"" + str_ + "\""; }

private:
    std::string str_;
};

struct TextType : TypeBase {
    TextType() = default;
    TextType(const std::string& text) : text_(text) {}

    std::string Dump(bool top __attribute__((unused)) = false) const { return text_; }

private:
    std::string text_;
};

struct CoordType : TypeBase {
    CoordType() = default;
    template <typename... TyArgs>
    CoordType(const TyArgs&... args) : size_(TypeCount<TyArgs...>::size), dim_{args...}
    {}

    std::string Dump(bool top __attribute__((unused)) = false) const
    {
        std::ostringstream oss;
        oss << SCHEMA_ARRAY_LBOUND;
        for (int index = 0; index < size_; index++) {
            oss << (index == 0 ? "" : SCHEMA_ATTR_VALUE_SEPARATOR);
            oss << Int64Type(dim_[index]).Dump();
        }
        oss << SCHEMA_ARRAY_RBOUND;
        return oss.str();
    }

private:
    int size_;
    int64_t dim_[0x8];
};

template <typename ElementType>
struct ArrayType : TypeBase {
    ArrayType() = default;

    template <typename ContainerType>
    ArrayType(const ContainerType& container, bool dumpIndex = false)
        : elementList_(container.begin(), container.end()), dumpIndex_(dumpIndex)
    {}

    static std::string DumpIndex(int index) { return "/*" + std::to_string(index) + "*/"; }

    std::string Dump(bool top __attribute__((unused)) = false) const
    {
        std::ostringstream oss;
        oss << SCHEMA_ARRAY_LBOUND;
        int index = 0;
        for (auto& element : elementList_) {
            oss << (index == 0 ? "" : SCHEMA_ATTR_VALUE_SEPARATOR);
            if (dumpIndex_) {
                oss << DumpIndex(index);
            }
            oss << element.Dump(top);
            index++;
        }
        oss << SCHEMA_ARRAY_RBOUND;
        return oss.str();
    }

private:
    std::vector<ElementType> elementList_;
    bool dumpIndex_;
};

struct AttributeId : TypeBase {
    AttributeId() = default;
    AttributeId(const std::string& name) : name_(name) {}

    const std::string& Name() const { return name_; }
    std::string Dump(bool top = true) const { return SCHEMA_ATTR_KEYWORD_PREFIX(top) + name_; }

private:
    std::string name_;
};

template <typename Ty0>
struct AttributeCall_1 : AttributeId {
    using Base = AttributeId;
    AttributeCall_1() = default;
    AttributeCall_1(const std::string& name) : Base(name) {}
    AttributeCall_1(const std::string& name, const Ty0& arg0) : Base(name), arg0_(arg0) {}
    std::string Dump(bool top = true) const
    {
        return SCHEMA_ATTR_KEYWORD_PREFIX(top) + Base::Name() + SCHEMA_ATTR_VALUE_LBOUND + ArgDump() +
               SCHEMA_ATTR_VALUE_RBOUND;
    }

protected:
    std::string ArgDump() const { return arg0_.Dump(false); }

private:
    Ty0 arg0_;
};

template <typename Ty0, typename Ty1>
struct AttributeCall_2 : AttributeCall_1<Ty0> {
    using Base = AttributeCall_1<Ty0>;
    AttributeCall_2() = default;
    AttributeCall_2(const std::string& name) : Base(name) {};
    AttributeCall_2(const std::string& name, const Ty0& arg0, const Ty1& arg1) : Base(name, arg0), arg1_(arg1) {}
    std::string Dump(bool top = true) const
    {
        return SCHEMA_ATTR_KEYWORD_PREFIX(top) + Base::Name() + SCHEMA_ATTR_VALUE_LBOUND + ArgDump() +
               SCHEMA_ATTR_VALUE_RBOUND;
    }

protected:
    std::string ArgDump() const { return Base::ArgDump() + SCHEMA_ATTR_VALUE_SEPARATOR + arg1_.Dump(false); }

private:
    Ty1 arg1_;
};

template <typename Ty0, typename Ty1, typename Ty2>
struct AttributeCall_3 : AttributeCall_2<Ty0, Ty1> {
    using Base = AttributeCall_2<Ty0, Ty1>;
    AttributeCall_3() = default;
    AttributeCall_3(const std::string& name) : Base(name) {};
    AttributeCall_3(const std::string& name, const Ty0& arg0, const Ty1& arg1, const Ty2& arg2)
        : Base(name, arg0, arg1), arg2_(arg2)
    {}
    std::string Dump(bool top = true) const
    {
        return SCHEMA_ATTR_KEYWORD_PREFIX(top) + Base::Name() + SCHEMA_ATTR_VALUE_LBOUND + ArgDump() +
               SCHEMA_ATTR_VALUE_RBOUND;
    }

protected:
    std::string ArgDump() const { return Base::ArgDump() + SCHEMA_ATTR_VALUE_SEPARATOR + arg2_.Dump(false); }

private:
    Ty2 arg2_;
};

template <typename Ty0, typename Ty1, typename Ty2, typename Ty3>
struct AttributeCall_4 : AttributeCall_3<Ty0, Ty1, Ty2> {
    using Base = AttributeCall_3<Ty0, Ty1, Ty2>;
    AttributeCall_4() = default;
    AttributeCall_4(const std::string& name) : Base(name) {};
    AttributeCall_4(const std::string& name, const Ty0& arg0, const Ty1& arg1, const Ty2& arg2, const Ty3& arg3)
        : Base(name, arg0, arg1, arg2), arg3_(arg3)
    {}
    std::string Dump(bool top = true) const
    {
        return SCHEMA_ATTR_KEYWORD_PREFIX(top) + Base::Name() + SCHEMA_ATTR_VALUE_LBOUND + ArgDump() +
               SCHEMA_ATTR_VALUE_RBOUND;
    }

protected:
    std::string ArgDump() const { return Base::ArgDump() + SCHEMA_ATTR_VALUE_SEPARATOR + arg3_.Dump(false); }

private:
    Ty3 arg3_;
};

template <typename Ty0, typename Ty1, typename Ty2, typename Ty3, typename Ty4>
struct AttributeCall_5 : AttributeCall_4<Ty0, Ty1, Ty2, Ty3> {
    using Base = AttributeCall_4<Ty0, Ty1, Ty2, Ty3>;
    AttributeCall_5() = default;
    AttributeCall_5(const std::string& name) : Base(name) {};
    AttributeCall_5(const std::string& name, const Ty0& arg0, const Ty1& arg1, const Ty2& arg2, const Ty3& arg3,
                    const Ty4& arg4)
        : Base(name, arg0, arg1, arg2, arg3), arg4_(arg4)
    {}
    std::string Dump(bool top = true) const
    {
        return SCHEMA_ATTR_KEYWORD_PREFIX(top) + Base::Name() + SCHEMA_ATTR_VALUE_LBOUND + ArgDump() +
               SCHEMA_ATTR_VALUE_RBOUND;
    }

protected:
    std::string ArgDump() const { return Base::ArgDump() + SCHEMA_ATTR_VALUE_SEPARATOR + arg4_.Dump(false); }

private:
    Ty4 arg4_;
};

template <typename Ty0, typename Ty1, typename Ty2, typename Ty3, typename Ty4, typename Ty5>
struct AttributeCall_6 : AttributeCall_5<Ty0, Ty1, Ty2, Ty3, Ty4> {
    using Base = AttributeCall_5<Ty0, Ty1, Ty2, Ty3, Ty4>;
    AttributeCall_6() = default;
    AttributeCall_6(const std::string& name) : Base(name) {};
    AttributeCall_6(const std::string& name, const Ty0& arg0, const Ty1& arg1, const Ty2& arg2, const Ty3& arg3,
                    const Ty4& arg4, const Ty5& arg5)
        : Base(name, arg0, arg1, arg2, arg3, arg4), arg5_(arg5)
    {}
    std::string Dump(bool top = true) const
    {
        return SCHEMA_ATTR_KEYWORD_PREFIX(top) + Base::Name() + SCHEMA_ATTR_VALUE_LBOUND + ArgDump() +
               SCHEMA_ATTR_VALUE_RBOUND;
    }

protected:
    std::string ArgDump() const { return Base::ArgDump() + SCHEMA_ATTR_VALUE_SEPARATOR + arg5_.Dump(false); }

private:
    Ty5 arg5_;
};

template <typename Ty0, typename... Tys>
static inline std::string UnionTypeDump(const Ty0* arg0, Tys... args)
{
    if (arg0) {
        return arg0->Dump(false);
    } else {
        return UnionTypeDump(args...);
    }
}

template <typename Ty0>
static inline std::string UnionTypeDump(const Ty0* arg0)
{
    if (arg0) {
        return arg0->Dump(false);
    } else {
        return "";
    }
}

template <typename Ty0, typename... Tys>
struct UnionTypeSelect {
    using Base = UnionTypeSelect<Tys...>;
    UnionTypeSelect() = default;

    void Construct(const Ty0& arg0, unsigned char* thisUnionData)
    {
        val0_ = reinterpret_cast<Ty0*>(thisUnionData);
        new (val0_) Ty0(arg0);
    }
    template <typename Ty>
    void Construct(const Ty& arg, unsigned char* thisUnionData)
    {
        val0_ = nullptr;
        base_.Construct(arg, thisUnionData);
    }

    void Assign(const UnionTypeSelect<Ty0, Tys...>& arg, unsigned char* thisUnionData)
    {
        if (arg.val0_) {
            AssignArg(*arg.val0_, thisUnionData);
        } else {
            base_.Assign(arg.base_, thisUnionData);
        }
    }
    void Clear()
    {
        if (val0_) {
            val0_->~Ty0();
            val0_ = nullptr;
        } else {
            base_.Clear();
        }
    }
    std::string Dump(bool top = false) const
    {
        if (val0_) {
            return val0_->Dump(top);
        } else {
            return base_.Dump(top);
        }
    }

private:
    void AssignArg(const Ty0& arg0, unsigned char* thisUnionData)
    {
        val0_ = reinterpret_cast<Ty0*>(thisUnionData);
        new (val0_) Ty0(arg0);
    }
    Ty0* val0_{nullptr};
    Base base_;
};

template <typename Ty0>
struct UnionTypeSelect<Ty0> {
    UnionTypeSelect() = default;

    void Construct(const Ty0& arg0, unsigned char* thisUnionData)
    {
        val0_ = reinterpret_cast<Ty0*>(thisUnionData);
        new (val0_) Ty0(arg0);
    }

    void Assign(const UnionTypeSelect<Ty0>& arg, unsigned char* thisUnionData)
    {
        if (arg.val0_ != nullptr) {
            AssignArg(*arg.val0_, thisUnionData);
        }
    }
    void Clear()
    {
        if (val0_) {
            val0_->~Ty0();
            val0_ = nullptr;
        }
    }
    std::string Dump(bool top = false) const
    {
        if (val0_) {
            return val0_->Dump(top);
        } else {
            return "";
        }
    }

private:
    void AssignArg(const Ty0& arg0, unsigned char* thisUnionData)
    {
        val0_ = reinterpret_cast<Ty0*>(thisUnionData);
        new (val0_) Ty0(arg0);
    }
    Ty0* val0_{nullptr};
};

template <typename Ty0, typename... Tys>
struct UnionType {
    UnionType() = default;
    template <typename Ty>
    UnionType(const Ty& arg)
    {
        memset_s(data_, sizeof(data_), 0, sizeof(data_));
        select_.Construct(arg, data_);
    }
    ~UnionType() { select_.Clear(); }
    UnionType(const UnionType<Ty0, Tys...>& arg)
    {
        memset_s(data_, sizeof(data_), 0, sizeof(data_));
        select_.Assign(arg.select_, data_);
    }
    UnionType<Ty0, Tys...>& operator=(const UnionType<Ty0, Tys...>& arg)
    {
        select_.Clear();
        memset_s(data_, sizeof(data_), 0, sizeof(data_));
        select_.Assign(arg.select_, data_);
        return *this;
    }
    std::string Dump(bool top = true) const { return select_.Dump(top); }

private:
    UnionTypeSelect<Ty0, Tys...> select_;
    unsigned char data_[MaxTypeSize<Ty0, Tys...>::size];

    // Currently, move is not allowed.
    UnionType(UnionType<Ty0, Tys...>&& arg) = delete;
    UnionType<Ty0, Tys...>& operator=(UnionType<Ty0, Tys...>&& arg) = delete;
};

#define SCHEMA_DEF_TYPE_INHERIT(name, baseType)               \
    typedef npu::tile_fwk::schema::type::baseType name##Base; \
    struct name : name##Base {                                \
        name() = default;                                     \
        template <typename... TyArgs>                         \
        name(const TyArgs&... args) : name##Base(args...)     \
        {}                                                    \
    };

#define SCHEMA_DEF_TYPE_INHERIT_ID(name, baseType, prefix)            \
    typedef npu::tile_fwk::schema::type::baseType<prefix> name##Base; \
    struct name : name##Base {                                        \
        name() = default;                                             \
        template <typename... TyArgs>                                 \
        name(const TyArgs&... args) : name##Base(args...)             \
        {}                                                            \
    };

#define SCHEMA_DEF_TYPE_INT32(name) SCHEMA_DEF_TYPE_INHERIT(name, Int32Type)
#define SCHEMA_DEF_TYPE_UINT32(name) SCHEMA_DEF_TYPE_INHERIT(name, UInt32Type)

#define SCHEMA_DEF_TYPE_INT64_1(name) SCHEMA_DEF_TYPE_INHERIT(name, Int64Type)
#define SCHEMA_DEF_TYPE_INT64_2(name, prefix) SCHEMA_DEF_TYPE_INHERIT_ID(name, Int64IdType, prefix)
#define SCHEMA_DEF_TYPE_INT64(...) \
    SCHEMA_DEF_ATTR_CONCAT(SCHEMA_DEF_TYPE_INT64_, SCHEMA_DEF_ATTR_NR(__VA_ARGS__))(__VA_ARGS__)

#define SCHEMA_DEF_TYPE_UINT64(name) SCHEMA_DEF_TYPE_INHERIT(name, UInt64Type)
#define SCHEMA_DEF_TYPE_ADDRESS(name) SCHEMA_DEF_TYPE_INHERIT(name, AddressType)
#define SCHEMA_DEF_TYPE_STRING(name) SCHEMA_DEF_TYPE_INHERIT(name, StringType)
#define SCHEMA_DEF_TYPE_COORD(name) SCHEMA_DEF_TYPE_INHERIT(name, CoordType)
#define SCHEMA_DEF_TYPE_TEXT(name) SCHEMA_DEF_TYPE_INHERIT(name, TextType)

#define SCHEMA_DEF_TYPE_ARRAY(name, element)                            \
    typedef npu::tile_fwk::schema::type::ArrayType<element> name##Base; \
    struct name : name##Base {                                          \
        name() = default;                                               \
        template <typename... TyArgs>                                   \
        name(const TyArgs&... args) : name##Base(args...)               \
        {}                                                              \
    }

#define SCHEMA_DEF_TYPE_UNION(name, ...)                                    \
    typedef npu::tile_fwk::schema::type::UnionType<__VA_ARGS__> name##Base; \
    struct name : name##Base {                                              \
        name() = default;                                                   \
        template <typename... TyArgs>                                       \
        name(const TyArgs&... args) : name##Base(args...)                   \
        {}                                                                  \
    }

#define SCHEMA_DEF_ATTR_NR_(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, v, ...) v
#define SCHEMA_DEF_ATTR_NR(...) SCHEMA_DEF_ATTR_NR_(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define SCHEMA_DEF_ATTR_CONCAT_(name, nr) name##nr
#define SCHEMA_DEF_ATTR_CONCAT(name, nr) SCHEMA_DEF_ATTR_CONCAT_(name, nr)

#define SCHEMA_DEF_ATTR_KEYWORD_(name)                           \
    typedef npu::tile_fwk::schema::type::AttributeId name##Base; \
    struct name : name##Base {                                   \
        name() : name##Base(#name) {}                            \
        static const std::string Name() { return #name; }        \
    };
#define SCHEMA_DEF_ATTR_CALL_(name, ...)                                                                      \
    typedef npu::tile_fwk::schema::type::SCHEMA_DEF_ATTR_CONCAT(AttributeCall_,                               \
                                                                SCHEMA_DEF_ATTR_NR(__VA_ARGS__))<__VA_ARGS__> \
        name##Base;                                                                                           \
    struct name : name##Base {                                                                                \
        name() : name##Base(#name) {}                                                                         \
        template <typename... TyArgs>                                                                         \
        name(const TyArgs&... args) : name##Base(#name, args...)                                              \
        {}                                                                                                    \
        static const std::string Name() { return #name; }                                                     \
    };

#define SCHEMA_DEF_ATTR_NAME(name, text)                         \
    typedef npu::tile_fwk::schema::type::AttributeId name##Base; \
    struct name : name##Base {                                   \
        name() : name##Base(#text) {}                            \
        static const std::string Name() { return #text; }        \
    };

#define SCHEMA_DEF_ATTR_1(name) SCHEMA_DEF_ATTR_KEYWORD_(name)
#define SCHEMA_DEF_ATTR_2(name, arg0) SCHEMA_DEF_ATTR_CALL_(name, arg0)
#define SCHEMA_DEF_ATTR_3(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_4(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_5(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_6(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_7(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_8(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_9(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_10(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_11(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_12(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_13(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_14(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR_15(name, arg0, ...) SCHEMA_DEF_ATTR_CALL_(name, arg0, __VA_ARGS__)
#define SCHEMA_DEF_ATTR(...) SCHEMA_DEF_ATTR_CONCAT(SCHEMA_DEF_ATTR_, SCHEMA_DEF_ATTR_NR(__VA_ARGS__))(__VA_ARGS__)

} // namespace npu::tile_fwk::schema::type

#endif // SCHEMA_TRACE_BASE_H
