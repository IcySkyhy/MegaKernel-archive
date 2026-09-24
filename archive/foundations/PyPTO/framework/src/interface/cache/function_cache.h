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
 * \file function_cache.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <optional>
#include <variant>
#include <mutex>
#include <unordered_map>
#include "interface/function/function.h"
#include "interface/cache/hash.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
// 保存CoreFunction在CoreFunctionTopo列表里的偏移
using CoreFunctionTopoOffsets = uint64_t[0];
#pragma pack(1)
struct CoreFunctionTopoCache {
    uint64_t dataSize;
    CoreFunctionTopoOffsets coreFunctionTopoOffsets; // 保存CoreFunctionTopo在CoreFunctionTopo_列表里的偏移
    CoreFunctionTopo coreFunctionTopo[0];            // CoreFunctionTopo列表
};

// 保存CoreFunctionBin的偏移
using CoreFunctionBinOffsets = uint64_t[0];
struct CoreFunctionBinCache {
    uint64_t dataSize;
    CoreFunctionBinOffsets coreFunctionBinOffsets;
    CoreFunctionBin coreFunctionBin[0];
};

// 初始的可以立即执行的CoreFunction的id列表
struct ReadyCoreFunctionCache {
    uint64_t dataSize;
    ReadyCoreFunction readyCoreFunction[0];
};

using HashKey = FunctionHash;
struct CacheHeader {
    uint64_t coreFunctionNum; // CoreFunction的个数，通过此值可以分配xxOffset的内存
    uint64_t virtualFunctionNum{0};
    uint64_t readyCoreFunctionNum; // 对应ReadyCoreFunctionCache -> ReadyCoreFunction个数
    uint64_t programFuncionNum;    // 同构后function个数，对应CoreFunctionBinCache -> CoreFunctionBin个数
};

struct CacheValue {
    CacheHeader header;
    HashKey tilingFuncKey;
    std::shared_ptr<CoreFunctionTopoCache> topoCache = nullptr;
    std::shared_ptr<CoreFunctionBinCache> binCache = nullptr;
    std::shared_ptr<ReadyCoreFunctionCache> readyListCache = nullptr;

    Function* GetFunction() { return cacheFunction; }

    void SetCacheFunction(Function* func) { cacheFunction = func; }

public:
    template <typename T>
    static std::shared_ptr<T> CreateCache(size_t size)
    {
        T* data = reinterpret_cast<T*>(new uint8_t[size]);
        auto ptr = std::shared_ptr<T>(data, [](T* p) { delete[] reinterpret_cast<uint8_t*>(p); });
        return ptr;
    }

private:
    Function* cacheFunction = nullptr;
};
#pragma pack()

class FunctionCache {
public:
    FunctionCache() = default;

    std::optional<CacheValue> Get(HashKey key);

    void Insert(const HashKey& key, Function& func);

    size_t Size();

    std::string GetHitRate();

    void Reset();

    virtual ~FunctionCache();

    Function* GetCacheFunction(const HashKey& key);

    void BuildHashDict(Function* func, std::unordered_map<FunctionHash, Function*>& hashDict)
    {
        std::vector<std::shared_ptr<CallOpAttribute>> callopAttrList = func->GetCallopAttrList();
        for (auto& callopAttr : callopAttrList) {
            auto hash = callopAttr->GetCalleeHash();
            Function* calleeFunction = GetCacheFunction(hash);
            if (calleeFunction == nullptr) {
                FE_LOGE(InternalError::FE_INNER_ERROR, "calleeFunction is nullptr, hash: %lu", hash.GetHash());
                continue;
            }
            hashDict[hash] = calleeFunction;
            BuildHashDict(calleeFunction, hashDict);
        }
        if (func->GetRootFunction()) {
            BuildHashDict(func->GetRootFunction(), hashDict);
        }
    }

private:
    void Insert(const HashKey& key, CacheValue value);
    void UpdateTopoCache(const Function& func, CacheValue& value);
    void UpdateBinCache(const Function& func, CacheValue& value);
    void UpdateReadyFunction(const Function& func, CacheValue& value);

private:
    std::unordered_map<HashKey, CacheValue> cache_;
    std::mutex lock_;
    int64_t getCnt_{0};
    int64_t hitCnt_{0};
};
} // namespace npu::tile_fwk
