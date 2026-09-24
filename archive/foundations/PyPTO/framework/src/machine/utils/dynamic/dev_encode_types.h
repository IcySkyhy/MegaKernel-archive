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
 * \file dev_encode_types.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <vector>

#include "tilefwk/error.h"
#include "tilefwk/data_type.h"
#include "tilefwk/aicpu_common.h"
#include "tilefwk/aikernel_data.h"
#include "tilefwk/core_func_data.h"
#include "interface/utils/common.h"
#include "interface/schema/schema.h"
#include "machine/utils/device_log.h"
#include "machine/utils/device_switch.h"
#include "machine/utils/dynamic/item_pool.h"

namespace npu::tile_fwk::dynamic {
using int32v8 = int32_t __attribute__((vector_size(32)));
using int32v4 = int32_t __attribute__((vector_size(16)));
using uint32v8 = uint32_t __attribute__((vector_size(32)));
using uint32v4 = uint32_t __attribute__((vector_size(16)));
using uint16v4 = uint16_t __attribute__((vector_size(8)));
using uint16v8 = uint16_t __attribute__((vector_size(16)));

constexpr uint32_t IDENT_SIZE = 2;
constexpr uint32_t IDENT2_SIZE = 4;
constexpr uint32_t IDENT_SIZE_THREE = 3;

/* please modify macros in aicore.cpp at the same time !!! */
inline uint32_t MakeMixWrapID(uint32_t funcId, uint32_t opWrapId) { return (funcId << TASKID_TASK_BITS) | opWrapId; }

// 计算原始Pass op的wrapId
inline uint32_t GetOpWrapID(uint32_t mixWrapId) { return mixWrapId & TASKID_TASK_MASK; }

inline uint32_t MakeBatchTaskID(uint32_t batchNum) { return MakeTaskID(FUNC_ID_BATCH, batchNum); }

inline uint32_t FuncNum(uint32_t id) { return id & TASKID_TASK_MASK; }

inline bool IsInitTaskID(uint32_t id) { return id == AICORE_TASK_INIT; }

inline bool IsTaskFinish(uint32_t id, uint32_t finValue) { return (id | AICORE_FIN_MASK) == finValue; }

inline int64_t AlignUp(int64_t val, int64_t align) { return (((val) + (align)-1) & ~((align)-1)); }

using uintdevptr_t = uint64_t;
using intdevptr_t = int64_t;

template <typename T>
inline void HostAssign(T*& ptr, uintdevptr_t offset)
{
    ptr = reinterpret_cast<T*>(offset);
}
template <typename T>
inline void DeviceReloc(T*& ptr, intdevptr_t shift)
{
    ptr = reinterpret_cast<T*>(reinterpret_cast<intdevptr_t>(ptr) + shift);
}
template <typename T>
inline void DeviceRelocMaybeNull(T*& ptr, intdevptr_t shift)
{
    if (ptr != nullptr) {
        ptr = reinterpret_cast<T*>(reinterpret_cast<intdevptr_t>(ptr) + shift);
    }
}

struct RelocRange {
    RelocRange(uintdevptr_t src, uintdevptr_t dst) : src_(src), dst_(dst) {}

    template <typename T>
    inline void RelocNullable(T*& ptr) const
    {
        if (ptr != nullptr) {
            Reloc(ptr);
        }
    }
    template <typename T>
    inline void Reloc(T*& ptr) const
    {
        DeviceReloc(ptr, dst_ - src_);
    }
    inline void Reloc(uint64_t& addr) const { addr += dst_ - src_; }
    inline void RelocNullable(uint64_t& addr) const
    {
        if (addr != 0) {
            Reloc(addr);
        }
    }

    uintdevptr_t GetDst() const { return dst_; }

private:
    uintdevptr_t src_;
    uintdevptr_t dst_;
};

template <typename T>
struct DevRelocPtr {
    DevRelocPtr() = default;
    explicit DevRelocPtr(void* addr) : ptr_(addr) {}

    T* operator->() { return ptr_; }
    void operator=(void* addr) { ptr_ = reinterpret_cast<T*>(addr); }

    T& operator[](int32_t idx) { return ptr_[idx]; }

    void HostAssignPtr(uintdevptr_t offset) { HostAssign(ptr_, offset); }
    void DeviceRelocPtr(intdevptr_t shift) { DeviceReloc(ptr_, shift); }

private:
    /* Not assign-able, not moveable, not copyable */
    DevRelocPtr(const DevRelocPtr& other) = delete;
    DevRelocPtr& operator=(const DevRelocPtr&) = delete;
    DevRelocPtr& operator=(DevRelocPtr&&) = delete;

    T* ptr_{nullptr};
};

template <typename T>
struct DevRelocVector {
    DevRelocVector() = default;
    DevRelocVector(int size, T* data) : size_(size), data_(data) {}

    T& operator[](size_t idx)
    {
        if (idx >= size_) {
            DEV_ERROR(DevDataErr::DEV_RELOC_VECTOR_INDEX_OOB, "#data.valid: Index out of bounds: idx=%zu, size=%zu",
                      idx, size_);
        }
        DEV_ASSERT(DevDataErr::DEV_RELOC_VECTOR_INDEX_OOB, idx < size_);
        return data_[idx];
    }
    const T& operator[](size_t idx) const
    {
        if (idx >= size_) {
            DEV_ERROR(DevDataErr::DEV_RELOC_VECTOR_INDEX_OOB, "#data.valid: Index out of bounds: idx=%zu, size=%zu",
                      idx, size_);
        }
        DEV_ASSERT(DevDataErr::DEV_RELOC_VECTOR_INDEX_OOB, idx < size_);
        return data_[idx];
    }

    const T* begin() const { return data_; }
    T* begin() { return data_; }
    const T* end() const { return data_ + size_; }
    T* end() { return data_ + size_; }

    size_t size() const { return size_; }
    const T* Data() const { return data_; }
    T* Data() { return data_; }

    size_t DataSize() const { return size_ * sizeof(T); }

    void HostAssignDataSize(uintdevptr_t offset, size_t size)
    {
        HostAssign(data_, offset);
        size_ = size;
    }
    void HostAssignRangeOffsetSize(const DevRelocVector<T>& base, uintdevptr_t offset, size_t size)
    {
        HostAssignDataSize(reinterpret_cast<uintdevptr_t>((base.Data() + offset)), size);
    }
    void HostInitDataSizeOffset(uintdevptr_t& offset, size_t size)
    {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, offset % alignof(T) == 0)
            << "Offset is not properly aligned for type T"; // Ensure offset is aligned
        HostAssign(data_, offset);
        size_ = size;
        offset = reinterpret_cast<uintdevptr_t>(data_ + size);
    }
    void DeviceRelocData(intdevptr_t shift) { DeviceReloc(data_, AlignUp(shift, alignof(T))); }
    void DeviceRelocDataMaybeNull(intdevptr_t shift) { DeviceRelocMaybeNull(data_, AlignUp(shift, alignof(T))); }
    uintdevptr_t End() const { return reinterpret_cast<uintdevptr_t>(data_ + size_); }

    static uint64_t ElementSize() { return sizeof(T); }
    using ElementType = T;

private:
    size_t size_{0};
    T* data_{nullptr};
};

template <typename T>
struct DevLocalVector {
    size_t size() const { return size_; }

    uintdevptr_t Offset(int idx) const { return offset_ + sizeof(T) * idx; }

    uintdevptr_t End() const { return offset_ + sizeof(T) * size_; }

    void AssignOffsetSize(uintdevptr_t offset, size_t size)
    {
        offset_ = offset;
        size_ = size;
    }
    void AssignRangeOffsetSize(const DevLocalVector<T>& base, uintdevptr_t offset, size_t size)
    {
        AssignOffsetSize(base.offset_ + sizeof(T) * offset, size);
    }

    void HostInitDataSizeOffset(uintdevptr_t& offset, size_t size)
    {
        offset = AlignUp(offset, alignof(T));
        offset_ = offset;
        size_ = size;
        offset = offset_ + size_ * sizeof(T);
    }

    size_t ByteSize() const { return size_ * sizeof(T); }

private:
    uintdevptr_t offset_{0};
    size_t size_{0};
};

// flag need used bit 63, see also macro values in tileop/runtime.h
struct SymInt {
    uint64_t value : 63;
    uint64_t flag : 1; // 0 is const, 1 is expression

    SymInt() : SymInt(0) {}
    explicit SymInt(uint64_t val) : value(val), flag(0) {}
    SymInt(bool isExpr, uint64_t val) : value(val), flag(isExpr) {}
    SymInt& operator=(uint64_t val)
    {
        value = val;
        flag = 0;
        return *this;
    }

    uint64_t Value() const { return value; }
    bool IsExpression() const { return flag == 1; }
};

struct DevCceBinary {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
    int32_t wrapVecId{-1};
    uint8_t mixResourceType{0};
};
static_assert(sizeof(DynFuncBin) == sizeof(DevCceBinary));

struct DevAicpuLeafBinary {
    DevRelocVector<int32_t> aicpuLeafCode;
};

struct PrefetchInfo {
    uint64_t tensorSize;
    uint64_t tensorIdx;
};

enum class DevIOProperty : uint32_t {
    NONE,
    ROOT_INCAST,
    ROOT_OUTCAST,
};

static inline const BiMap<DevIOProperty>& GetDevIOPropertyDict()
{
    static BiMap<DevIOProperty> propertyDict = {
        {DevIOProperty::NONE, "NONE"},
        {DevIOProperty::ROOT_INCAST, "ROOT_INCAST"},
        {DevIOProperty::ROOT_OUTCAST, "ROOT_OUTCAST"},
    };
    return propertyDict;
}

static inline std::string DevIOProperty2String(DevIOProperty property) { return GetDevIOPropertyDict().Find(property); }

static inline std::string Delim(bool cond, const std::string& delim) { return cond ? delim : ""; }

static inline std::string DumpByte(uint8_t byte)
{
    char buf[0x10];
    (void)sprintf_s(buf, sizeof(buf), "0x%02x", byte);
    return buf;
}

static inline std::string DumpShape(const DevShape& shape)
{
    std::ostringstream oss;
    oss << "<";
    for (int k = 0; k < shape.dimSize; k++) {
        oss << Delim(k != 0, ",") << shape.dim[k];
    }
    oss << ">";
    return oss.str();
}

struct AddressDescriptor {
    union {
        struct {
            uint64_t rtOutcastIter : 63;
            uint64_t isRtOutcast : 1;
        };
        struct {
            uint64_t addr : 63;
            uint64_t : 1;
        };
        struct {
            uint64_t cacheValue : 60;
            uint64_t cacheKind : 4;
        };
    };

    static AddressDescriptor MakeFromAddress(uint64_t addr)
    {
        AddressDescriptor desc;
        desc.addr = addr;
        desc.isRtOutcast = 0;
        return desc;
    }

    static AddressDescriptor MakeFromRtOutcast(ItemPoolIter iter)
    {
        DEV_ASSERT_MSG(DevDataErr::DEV_RELOC_VECTOR_INDEX_OOB, (iter & (1ULL << 63)) == 0,
                       "RtOutcast iterator %" PRId64 " exceeds maximum allowed value", iter);
        AddressDescriptor desc;
        desc.rtOutcastIter = iter;
        desc.isRtOutcast = 1;
        return desc;
    }

    static AddressDescriptor MakeCache(uint64_t kind, uint64_t value)
    {
        AddressDescriptor desc;
        desc.cacheValue = value;
        desc.cacheKind = kind;
        return desc;
    }

    bool IsAddress() const { return !isRtOutcast; }
    uint64_t GetAddress() const
    {
        DEV_ASSERT_MSG(DevDataErr::DEV_RELOC_VECTOR_INDEX_OOB, IsAddress(),
                       "Attempt to get address from a non-address AddressDescriptor.");
        return addr;
    }
    uint64_t GetAddressValue() const { return addr; }
    bool IsNullAddress() const { return IsAddress() && addr == 0; }

    bool IsRtOutcast() const { return isRtOutcast; }
    ItemPoolIter GetRtOutcastIter() const
    {
        DEV_ASSERT_MSG(DevDataErr::DEV_RELOC_VECTOR_INDEX_OOB, IsRtOutcast(),
                       "Attempt to get runtime outcast iterator from a non-iterator AddressDescriptor.");
        return rtOutcastIter;
    }

public:
    static std::string DumpAddress(uintdevptr_t addr, int width = 0)
    {
        char bufData[0x20];
        (void)sprintf_s(bufData, sizeof(bufData), "%lx", addr);
        std::string buf = bufData;
        if (buf.size() < static_cast<size_t>(width)) {
            buf = std::string(width - buf.size(), '0') + buf;
        }
        return "&0x" + buf;
    }
    std::string Dump() const
    {
        std::stringstream ss;
        if (IsAddress()) {
            ss << DumpAddress(addr);
        } else {
            ss << "&&" << rtOutcastIter;
        }
        return ss.str();
    }
};
} // namespace npu::tile_fwk::dynamic
