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
 * \file common.h
 * \brief
 */

#pragma once

#include <sys/time.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cstdint>
#include <set>

#include "tilefwk/error.h"
#include "tilefwk/symbolic_scalar.h"
#include "core/bimap.h"
#include "tilefwk/core_func_data.h"

namespace npu::tile_fwk {

#ifdef __clang__
#define NO_UBSAN __attribute__((no_sanitize("unsigned-integer-overflow")))
#else
#define NO_UBSAN
#endif

enum Status : uint32_t {
    SUCCESS = 0,
    FAILED = 1,
    WARNING = 2,
};

#define CACHELINE_SIZE_FOR_B64 64

constexpr uint32_t DIST_COMM_GROUP_NUM = 2;

constexpr const int NUM2 = 2;
constexpr const int NUM3 = 3;
constexpr const int NUM4 = 4;
constexpr const int NUM1 = 1;
constexpr const int NUM150 = 150;
constexpr const int NUM16 = 16;

constexpr const int REPEAT_BLOCK_NUM = 64;
constexpr const int BLOCK_NUM = 8;
constexpr const int BLOCK_SIZE = 32;
constexpr const int LEN1024 = 1024;
constexpr const int LEN512 = 512;
constexpr const int LEN255 = 255;
constexpr const int LEN25 = 25;
constexpr const int LEN10 = 10;
constexpr const int MAX_REPEAT = 255;
constexpr unsigned REPEAT_BYTE = 256;
constexpr const unsigned MAX_TMP_BUF_SHAPE = 32 << 10;

constexpr const int RANK1 = 1;
constexpr const int RANK2 = 2;
constexpr const int RANK3 = 3;
constexpr const int RANK4 = 4;
constexpr const int SHAPE_DIM0 = 0;
constexpr const int SHAPE_DIM1 = 1;
constexpr const int SHAPE_DIM2 = 2;
constexpr const int SHAPE_DIM3 = 3;
constexpr const int SHAPE_DIM4 = 4;
constexpr const int SHAPE_DIM5 = 5;
constexpr const int SHAPE_DIM6 = 6;
constexpr const int ALIGN_SIZE_512 = 512;
constexpr const int ALIGN_SIZE_64 = 64;
constexpr const int ALIGN_SIZE_32 = 32;
constexpr const int ALIGN_SIZE_16 = 16;
constexpr const int VNCHWCONV_REPEAT = 16;
constexpr const int MAX_CAT_NUM_ONCE = 64;
constexpr const int TILE_VEC_FOUR_DIMS = 4;
constexpr const int MAX_DILATION_STRIDE = 63;
constexpr const int MAX_PAD_KERNEL = 255;
constexpr const int SHAPE_INNER_AXIS_MAX_SIZE = 65535;
constexpr const int MAX_SIZE = 1000000;

constexpr const int SHAPE_BUFFER_MAX_SIZE = 32;
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

inline constexpr uint64_t KIBI = 1024;
inline constexpr uint64_t MEBI = UINT64_C(1024) * 1024;
inline constexpr uint64_t GIBI = UINT64_C(1024) * 1024 * 1024;

constexpr const int INVALID_LOOP_GROUPID = -1;

template <typename ValueType = int64_t>
inline ValueType AlignUp(ValueType value, int64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T>
inline void HashCombine(std::size_t& seed, const T& val) NO_UBSAN
{
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 0x6) + (seed >> 0x2);
}

template <typename T>
inline void HashVal(size_t& seed, const T& val)
{
    HashCombine(seed, val);
}

template <typename T, typename... Args>
inline void HashVal(size_t& seed, const T& val, const Args&... args)
{
    HashCombine(seed, val);
    HashVal(seed, args...);
}

constexpr int SYMBOL_DDRID_BASE = 0x40000000;
constexpr int SYMBOL_STACK_BASE = 0x3f000000;

const int RUNTIME_QUEUE_SIZE = 2;

enum OperandType {
    BUF_UNKNOWN = -1,
    BUF_UB = 0,
    BUF_L1 = 1,
    BUF_L0A = 2,
    BUF_L0B = 3,
    BUF_L0C = 4,
    BUF_FIX = 5,
    BUF_BT = 6,
    BUF_DDR = 7,
    BUF_REG = 8,
    SCALAR = 9,
    BUF_L0AMX = 10,
    BUF_L0BMX = 11,
    TOTAL_BUF_TYPE,
};

std::string OperandTypeToStr(OperandType t);

enum QueueType {
    QUEUE_INVALID = -1,
    QUEUE_MTE_IN = 0,
    QUEUE_MTE_OUT = 1,
    QUEUE_MTE_BETWEEN = 2,
    QUEUE_VECTOR = 3,
    QUEUE_CUBE = 4,
    QUEUE_MAX = 5
};

enum IQType {
    IQ_VECTOR_BMU = 0,
    IQ_CUBE_BMU_L1 = 1,
    IQ_CUBE_BMU_L0A = 2,
    IQ_CUBE_BMU_L0B = 3,
    IQ_CUBE_BMU_L0C = 4,
    IQ_CUBE_BMU_FIX = 5,
    IQ_MTE1 = 6,
    IQ_MTE2 = 7,
    IQ_VECTOR_ALU = 8,
    IQ_CUBE_ALU = 9,
    IQ_MTE3 = 10,
    IQ_MTE_FIXP = 11,
    TOTAL_IQ_TYPE = 12
};

// hardware pipeline
enum PipeType {
    PIPE_S = 0, // Scalar Pipe
    PIPE_V,     // Vector Pipe, including{VectorOP write UB,  L0C->UB write}
    PIPE_M,     // Matrix Pipe, including{}
    PIPE_MTE1,  // L1->L0{A,B}
    PIPE_MTE2,  // OUT ->{L1, L0{A,B}, UB}
    PIPE_MTE3,  // UB ->{OUT,L1}
    PIPE_ALL,
    PIPE_MTE4 = 7, // MOV_UB_TO_OUT
    PIPE_MTE5 = 8, // MOV_OUT_TO_UB
    PIPE_V2 = 9,   // Lower priority vector pipe,
    PIPE_FIX = 10, // {L0C} ->{L1,UB,L1UB}
};

template <typename T>
inline std::string IntVecToStr(const std::vector<T>& shape)
{
    std::stringstream ss;
    ss << "[";

    if (!shape.empty()) {
        ss << shape[0];
        for (size_t i = 1; i < shape.size(); ++i) {
            ss << ", " << shape[i];
        }
    }

    ss << "]";
    return ss.str();
}

template <typename T>
void CombineLastTwoAxisOffset(std::vector<T>& offset, const std::vector<int64_t>& rawShape, size_t shapeSize)
{
    if (offset.empty()) {
        return;
    }
    ASSERT(TensorErr::TENSOR_SHAPE_MISMATCH, shapeSize >= NUM2)
        << "shapeSize " << shapeSize << " must be greater than " << NUM2;
    ASSERT(TensorErr::TENSOR_SHAPE_MISMATCH, offset.size() == shapeSize)
        << "offset size " << offset.size() << "must be equal to shapeSize " << shapeSize << ", offset "
        << IntVecToStr(offset);
    ASSERT(TensorErr::TENSOR_SHAPE_MISMATCH, rawShape.size() == shapeSize)
        << "rawShape size " << rawShape.size() << "must be equal to shapeSize " << shapeSize << ", rawShape "
        << IntVecToStr(rawShape);
    offset[shapeSize - 1] = offset[shapeSize - NUM2] * rawShape[shapeSize - 1] + offset[shapeSize - 1];
    offset[shapeSize - NUM2] = 0;
}

std::string SymbolicVecToStr(const std::vector<SymbolicScalar>& a);

constexpr int ParamLocOffset = 28;
constexpr int ParamLocTensor = 0;
constexpr int ParamLocIncast = 1;
constexpr int ParamLocOutcast = 2;

std::string ParamLocToStr(uint32_t loc);

inline const BiMap<CoreType>& GetCoreTypeDict()
{
    static BiMap<CoreType> dict{{
        {CoreType::AIV, "AIV"},
        {CoreType::AIC, "AIC"},
        {CoreType::MIX, "MIX"},
        {CoreType::AICPU, "AICPU"},
        {CoreType::HUB, "HUB"},
        {CoreType::GMATOMIC, "GMATOMIC"},
        {CoreType::HUB_MIX, "HUB_MIX"},
    }};
    return dict;
};

inline const BiMap<PipeType>& GetPipeTypeDict()
{
    static BiMap<PipeType> dict{{
        {PipeType::PIPE_S, "PIPE_S"},
        {PipeType::PIPE_V, "PIPE_V"},
        {PipeType::PIPE_M, "PIPE_M"},
        {PipeType::PIPE_MTE1, "PIPE_MTE1"},
        {PipeType::PIPE_MTE2, "PIPE_MTE2"},
        {PipeType::PIPE_MTE3, "PIPE_MTE3"},
        {PipeType::PIPE_ALL, "PIPE_ALL"},
        {PipeType::PIPE_MTE4, "PIPE_MTE4"},
        {PipeType::PIPE_MTE5, "PIPE_MTE5"},
        {PipeType::PIPE_V2, "PIPE_V2"},
        {PipeType::PIPE_FIX, "PIPE_FIX"},
    }};
    return dict;
};

template <typename T>
struct OrderedSet : std::unordered_map<T, int> {
    bool Insert(const T& data)
    {
        if (this->count(data) == 0) {
            this->insert(std::make_pair(data, this->size()));
            order.push_back(data);
            return true;
        }

        return false;
    }

    int InsertAndGetIndex(const T& data)
    {
        Insert(data);
        return GetIndex(data);
    }

    using OrderElementType = T;
    typename std::vector<OrderElementType>::iterator begin() { return order.begin(); }
    typename std::vector<OrderElementType>::iterator end() { return order.end(); }

    typename std::vector<OrderElementType>::const_iterator begin() const { return order.begin(); }
    typename std::vector<OrderElementType>::const_iterator end() const { return order.end(); }

    const T& operator[](int index) const { return order[index]; }
    T& operator[](int index) { return order[index]; }

    bool HasData(const T& data) const
    {
        return this->find(data) != static_cast<const std::unordered_map<T, int>&>(*this).end();
    }
    int GetIndex(const T& data) const { return this->find(data)->second; }

    // 保证order的确定性
    void Remove(const std::vector<T>& items)
    {
        std::unordered_set<T> toRemove(items.begin(), items.end());
        std::vector<T> newOrder;
        bool removed = false;
        for (auto& x : order) {
            if (toRemove.count(x) != 0) {
                this->erase(x);
                removed = true;
            } else {
                this->at(x) = newOrder.size();
                newOrder.push_back(x);
            }
        }
        if (removed) {
            order = std::move(newOrder);
        }
    }

    void Clear()
    {
        order.clear();
        this->clear();
    }

    bool operator==(const OrderedSet& rhs)
    {
        if (order.size() != rhs.size())
            return false;
        for (auto& x : rhs.order) {
            if (this->count(x) == 0)
                return false;
        }
        return true;
    }

    const std::vector<OrderElementType>& GetOrder() const { return order; }

    std::vector<OrderElementType> order;
};

template <typename Key, typename T, class Hash = std::hash<Key>>
struct OrderedMap {
    using OrderElementType = typename std::pair<Key, T>;
    typename std::vector<OrderElementType>::iterator begin() { return orderData.begin(); }
    typename std::vector<OrderElementType>::iterator end() { return orderData.end(); }

    typename std::vector<OrderElementType>::const_iterator begin() const { return orderData.begin(); }
    typename std::vector<OrderElementType>::const_iterator end() const { return orderData.end(); }

    T& operator[](const Key& key)
    {
        if (!orderDict.count(key)) {
            int size = orderDict.size();
            orderDict[key] = size;
            orderData.emplace_back(key, T());
        }
        return orderData[orderDict[key]].second;
    }

    std::unordered_map<Key, int, Hash> orderDict;
    std::vector<OrderElementType> orderData;
};

inline const BiMap<QueueType>& GetQueueNameDict()
{
    static BiMap<QueueType> dict{{
        {QUEUE_MTE_IN, "MTE_IN"},
        {QUEUE_MTE_OUT, "MTE_OUT"},
        {QUEUE_MTE_BETWEEN, "MTE_BETW"},
        {QUEUE_VECTOR, "VECTOR"},
        {QUEUE_CUBE, "CUBE"},
    }};
    return dict;
}

inline std::string QueueName(QueueType qt, const std::string& defaultValue = "ILLEGAL")
{
    return GetQueueNameDict().Find(qt, defaultValue);
}

inline QueueType QueueGetType(const std::string& name, QueueType defaultValue = QUEUE_INVALID)
{
    return GetQueueNameDict().Find(name, defaultValue);
}

inline std::string IQName(IQType qt)
{
    switch (qt) {
        case IQ_VECTOR_BMU:
            return "VECTOR_BMU";
        case IQ_CUBE_BMU_L1:
            return "CUBE_BMU_L1";
        case IQ_CUBE_BMU_L0A:
            return "CUBE_BMU_L0A";
        case IQ_CUBE_BMU_L0B:
            return "CUBE_BMU_L0B";
        case IQ_CUBE_BMU_L0C:
            return "CUBE_BMU_L0C";
        case IQ_CUBE_BMU_FIX:
            return "CUBE_BMU_FIX";
        case IQ_MTE1:
            return "MTE1";
        case IQ_MTE2:
            return "MTE2";
        case IQ_VECTOR_ALU:
            return "VECTOR_ALU";
        case IQ_CUBE_ALU:
            return "CUBE";
        case IQ_MTE3:
            return "MTE3";
        case IQ_MTE_FIXP:
            return "MTE_FIXP";
        default:
            return "ILLEGAL";
    }
}

inline const BiMap<OperandType>& GetBufferNameDict()
{
    static BiMap<OperandType> dict{{
        {BUF_UB, "UB"},
        {BUF_L1, "L1"},
        {BUF_L0A, "L0A"},
        {BUF_L0B, "L0B"},
        {BUF_L0C, "L0C"},
        {BUF_FIX, "FIX"},
        {BUF_DDR, "DDR"},
        {BUF_REG, "REG"},
        {SCALAR, "SCALAR"},
        {BUF_BT, "BT"},
    }};
    return dict;
}

inline std::string BufferName(OperandType buffer, const std::string& defaultValue = "UNKNOWN")
{
    return GetBufferNameDict().Find(buffer, defaultValue);
}

inline OperandType BufferGetType(const std::string& name, OperandType defaultValue = BUF_UNKNOWN)
{
    return GetBufferNameDict().Find(name, defaultValue);
}

// 辅助函数：去除字符串两端空格
std::string Trim(const std::string& str);

// 通用环境变量获取函数
std::string GetEnvVar(const std::string& varName, bool trim = true, bool toLower = false);

// 辅助函数：把字符串变成数组
std::vector<std::string> SplitString(const std::string& str);

// 判断环境变量 PYPTO_DATADUMP_ENABLE 是否为 true
inline bool IsPtoDataDumpEnabled()
{
    static const bool result = []() {
        std::string value = GetEnvVar("PYPTO_DATADUMP_ENABLE", true, true);
        return (value == "true");
    }();

    return result;
}

// 判断环境变量 ENABLE_AICORE_RESOLVE 是否为 true
inline bool IsAicoreResolveEnabled()
{
    static const bool result = []() {
        std::string value = GetEnvVar("ENABLE_AICORE_RESOLVE", true, true);
        return (value == "true");
    }();

    return result;
}

// 向上取整除法
inline int CeilDiv(int a, int b)
{
    ASSERT(b != 0) << "Param(b) must not be zero.";
    return a / b + (a % b != 0 ? 1 : 0);
}

// 向下取整除法
inline int FloorDiv(int a, int b)
{
    ASSERT(b != 0) << "Param(b) must not be zero.";
    return a / b;
}

// 求最小值
inline int Min(int a, int b) { return (a < b) ? a : b; }

// 求最大值
inline int Max(int a, int b) { return (a > b) ? a : b; }

inline std::set<int> PowersOf2(int n)
{
    std::set<int> result;
    ASSERT(n > 0) << "n: " << n;
    int power = 0;
    while (true) {
        int current = 1 << power; // 计算 2^power
        if (current > n) {
            break;
        }
        result.insert(current);
        power++;
    }
    return result;
}

class Defer {
public:
    Defer(std::function<void()> callback) : callback_(std::move(callback)) {}
    ~Defer() { callback_(); }

private:
    std::function<void()> callback_;
};

struct TimeStamp {
    TimeStamp() { Reset(); }
    uint64_t Duration() { return CurrentTime() - startTime; }
    void Reset() { startTime = CurrentTime(); }

    static uint64_t CurrentTime();

private:
    uint64_t startTime;
};

template <typename T>
inline bool HasNegativeNum(const std::vector<T>& vec)
{
    return std::any_of(vec.begin(), vec.end(), [](T num) { return num < 0; });
}

namespace Matrix {
const std::string OP_ATTR_PREFIX = "op_attr_";
const std::string L1_TO_L0_OFFSET = OP_ATTR_PREFIX + "l1_to_l0_offset";
const std::string L1_TO_L0_TILE = OP_ATTR_PREFIX + "l1_to_l0_tile";
const std::string A_MUL_B_COPY_IN_MODE = OP_ATTR_PREFIX + "copy_in_mode";

enum class CopyInMode : int64_t { ND2ND = 0, ND2NZ = 1, NZ2NZ = 2, DN2NZ = 3 };

enum class CopyOutMode : int64_t { NZ2ND = 0, NZ2NZ = 1, ND2ND = 2, NZ2DN = 3 };

enum class CopyMode : int64_t { UNKNOWN = 0, EXTRACT = 1, INSERT = 2, MOVE = 3 };

enum class PaddingMode : int64_t { NORMAL_PADDING_MODE = 0, MX_PADDING_MODE = 1 };
} // namespace Matrix

std::string SafeExecCommandWithOutput(const std::vector<std::string>& args);

int SafeExecCommand(const std::vector<std::string>& args);

struct TensorAddrKey {
    int funcMagic;
    int opMagic;
    bool operator==(const TensorAddrKey& other) const
    {
        return funcMagic == other.funcMagic && opMagic == other.opMagic;
    }
    bool operator<(const TensorAddrKey& other) const
    {
        return funcMagic == other.funcMagic ? opMagic < other.opMagic : funcMagic < other.funcMagic;
    }
};
} // namespace npu::tile_fwk
