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
 * \file topo_processor.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include "tilefwk/core_func_data.h"
#include "tilefwk/pypto_fwk_log.h"
#include "function_cache.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

struct IdList {
    std::unique_ptr<uint64_t[]> data;
    uint32_t len;
    uint64_t keyId;

    IdList(const uint64_t* src, uint32_t n, uint64_t id) : data(std::make_unique<uint64_t[]>(n)), len(n), keyId(id)
    {
        memcpy_s(data.get(), n * sizeof(uint64_t), src, n * sizeof(uint64_t));
    }
};

struct IdListKey {
    uint64_t hash;
    uint32_t len;

    bool operator==(const IdListKey& o) const { return hash == o.hash && len == o.len; }

    friend uint32_t HashValue(const IdListKey& k)
    {
        uint32_t shift = 16;
        return (k.hash & 0xFFFFFFFF) ^ (k.len << shift);
    }
};

struct IdListKeyHash {
    uint32_t operator()(const IdListKey& k) const { return HashValue(k); }
};

class TopoProcessor {
public:
    TopoProcessor(std::shared_ptr<CoreFunctionTopoCache> topoData, uint64_t topoNum)
        : srcTopoData_(topoData), srcTopoNum_(topoNum)
    {}

    ~TopoProcessor()
    {
        for (auto& items : newTopoIdToNewTopo_) {
            delete[] reinterpret_cast<uint8_t*>(items.second);
        }
    }

    /* 合并批量依赖处理 */
    std::tuple<std::shared_ptr<CoreFunctionTopoCache>, uint64_t> MergeBatchDepend(uint64_t batchDependNum,
                                                                                  uint32_t mergeNum)
    {
        ParseOldTopo(batchDependNum);
        GenVirtualSubgraphTopo(mergeNum);
        return GenFinalTopo();
    }

private:
    void ParseOldTopo(uint64_t batchDependNum)
    {
        for (uint64_t i = 0; i < srcTopoNum_; i++) {
            uint8_t* base = static_cast<uint8_t*>(static_cast<void*>(srcTopoData_.get()));
            CoreFunctionTopo* topoData = reinterpret_cast<CoreFunctionTopo*>(
                base + ((uint64_t*)base)[i + 1]); // [i+1] access srcTopoData_->coreFunctionTopoOffsets
            if (topoData->depNum < batchDependNum) {
                MACHINE_LOGD("[TopoProcessor]ignore proc topo %lu, dep num %lu", i, topoData->depNum);
                continue;
            }
            uint64_t tmpTopoId = ProcTopoBatchDepend(topoData);
            MACHINE_LOGD("[TopoProcessor]proc topo %lu, dep num %lu, new tmp topoid:%lu", i, topoData->depNum,
                         tmpTopoId);
        }
    }

    void GenVirtualSubgraphTopo(uint32_t mergeNum)
    {
        uint64_t virtualTopoId = srcTopoNum_;

        for (auto it = newTopoIdToOldTopo_.begin(); it != newTopoIdToOldTopo_.end(); ++it) {
            auto checkPureBatchDepend = [&it, this]() -> bool {
                std::vector<CoreFunctionTopo*> oldTopoVec = it->second;
                CoreFunctionTopo* oldTopo = oldTopoVec.front();
                for (uint64_t i = 0; i < oldTopo->depNum; i++) {
                    uint8_t* base = static_cast<uint8_t*>(static_cast<void*>(srcTopoData_.get()));
                    CoreFunctionTopo* topoData = reinterpret_cast<CoreFunctionTopo*>(
                        base + ((uint64_t*)base)[oldTopo->depIds[i] + 1]);
                    if (static_cast<uint64_t>(topoData->readyCount * (-1)) != oldTopoVec.size()) {
                        return false;
                    }
                }
                return true;
            };
            bool isPure = checkPureBatchDepend();
            if (isPure) {
                MACHINE_LOGD("[TopoProcessor]gen valid pure batch depend new topo %lu, size %lu, virtualTpopid:%lu",
                             it->first, it->second.size(), virtualTopoId);
                ConnectVirtualTopo(it->second, virtualTopoId, true);
            } else if (it->second.size() >= mergeNum) {
                MACHINE_LOGD("[TopoProcessor]gen valid mix batch depend new topo %lu, size %lu, virtualTpopid:%lu",
                             it->first, it->second.size(), virtualTopoId);
                ConnectVirtualTopo(it->second, virtualTopoId, false);
            } else {
                MACHINE_LOGD("[TopoProcessor]erase unvalid new topo %lu, size %lu", it->first, it->second.size());
            }
        }
    }

    void ConnectVirtualTopo(std::vector<CoreFunctionTopo*>& oldTopoVec, uint64_t& virtualTopoId, bool isPure) NO_UBSAN
    {
        CoreFunctionTopo* oldTopo = oldTopoVec.front();

        // new virtual topo node
        uint32_t size = sizeof(CoreFunctionTopo) + sizeof(uint64_t) * oldTopo->depNum;
        CoreFunctionTopo* virtualTopo = reinterpret_cast<CoreFunctionTopo*>(new uint8_t[size]);
        memcpy_s(virtualTopo, size, static_cast<uint8_t*>(static_cast<void*>(oldTopo)), size);
        virtualTopo->coreType = static_cast<uint64_t>(isPure ? MachineType::VIRTUAL_PURE : MachineType::VIRTUAL_MIX);
        virtualTopo->psgId = 0xFFFFFFFF; // invalid psgid
        virtualTopo->readyCount = (-1) * static_cast<int64_t>(oldTopoVec.size());
        newTopoIdToNewTopo_[virtualTopoId] = virtualTopo;
        MACHINE_LOGD("[TopoProcessor]new virtual topo %lu , readycount:%ld", virtualTopoId, virtualTopo->readyCount);
        for (uint64_t i = 0; i < oldTopo->depNum; i++) {
            MACHINE_LOGD(" [TopoProcessor]batch depend topo id %lu ", oldTopo->depIds[i]);
        }

        // connect old topo with virtual subgraph topo
        for (uint64_t i = 0; i < oldTopoVec.size(); i++) {
            oldTopo = oldTopoVec.at(i);
            oldTopo->depNum = 1;
            oldTopo->depIds[0] = virtualTopoId;
        }
        MACHINE_LOGD("[TopoProcessor]have %lu old topo connect virtual topo %lu:", oldTopoVec.size(), virtualTopoId);

        virtualTopoSize_ += (size + sizeof(uint64_t)); // add offset size
        virtualTopoNum_++;
        virtualTopoId++;
        return;
    };

    std::tuple<std::shared_ptr<CoreFunctionTopoCache>, uint64_t> GenFinalTopo()
    {
        if (virtualTopoSize_ == 0) {
            // use orgin topo
            return std::tuple<std::shared_ptr<CoreFunctionTopoCache>, uint64_t>(srcTopoData_, 0);
        }

        // add old topo
        uint64_t newTopoSize = srcTopoData_->dataSize + virtualTopoSize_;
        auto newTopoCache = CacheValue::CreateCache<CoreFunctionTopoCache>(newTopoSize + sizeof(uint64_t));
        uint8_t* topoCachePtr = reinterpret_cast<uint8_t*>(newTopoCache.get());

        *(reinterpret_cast<uint64_t*>(topoCachePtr)) = newTopoSize;
        uint64_t* offsetPtr = reinterpret_cast<uint64_t*>(topoCachePtr) + 1;
        uint8_t* topoPtr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t*>(topoCachePtr) + srcTopoNum_ +
                                                      virtualTopoNum_ + 1);
        uint64_t curCoreFuncOffset = sizeof(uint64_t) + (srcTopoNum_ + virtualTopoNum_) * sizeof(uint64_t);
        auto appendTopo = [&topoPtr, &offsetPtr, &curCoreFuncOffset](CoreFunctionTopo* srcTopo, uint64_t id) {
            offsetPtr[id] = curCoreFuncOffset;
            CoreFunctionTopo* tempPtr = reinterpret_cast<CoreFunctionTopo*>(topoPtr);
            tempPtr->coreType = srcTopo->coreType;
            tempPtr->extType = srcTopo->extType;
            tempPtr->psgId = srcTopo->psgId;
            tempPtr->readyCount = srcTopo->readyCount;
            tempPtr->depNum = srcTopo->depNum;
            tempPtr->extParamNum = srcTopo->extParamNum;
            const uint64_t depIdsLen = srcTopo->depNum + srcTopo->extParamNum;
            if (depIdsLen != 0) {
                (void)memcpy_s(static_cast<uint8_t*>(static_cast<void*>(tempPtr->depIds)), depIdsLen * sizeof(uint64_t),
                               static_cast<uint8_t*>(static_cast<void*>(srcTopo->depIds)),
                               depIdsLen * sizeof(uint64_t));
            }

            uint32_t tempLength = sizeof(CoreFunctionTopo) +
                                  sizeof(uint64_t) * (tempPtr->depNum + tempPtr->extParamNum);
            curCoreFuncOffset += tempLength;
            topoPtr += tempLength;
            MACHINE_LOGD("[TopoProcessor]gen final toppo, add topo, id = %lu, coretype = %lu", id, tempPtr->coreType);
        };

        for (uint32_t i = 0; i < srcTopoNum_; i++) {
            uint8_t* base = static_cast<uint8_t*>(static_cast<void*>(srcTopoData_.get()));
            CoreFunctionTopo* oldTopo = reinterpret_cast<CoreFunctionTopo*>(base + ((uint64_t*)base)[i + 1]);
            appendTopo(oldTopo, i);
        }
        MACHINE_LOGD("[TopoProcessor] finish add old topo, num = %lu", srcTopoNum_);

        // add virtual topo
        for (auto& elm : newTopoIdToNewTopo_) {
            appendTopo(elm.second, elm.first);
        }
        MACHINE_LOGD("[TopoProcessor] finish add virtual topo, num = %lu", newTopoIdToNewTopo_.size());

        return std::tuple<std::shared_ptr<CoreFunctionTopoCache>, uint64_t>(newTopoCache, virtualTopoNum_);
    }

    uint64_t ProcTopoBatchDepend(CoreFunctionTopo* topoNode)
    {
        if (!topoNode || topoNode->depNum == 0) {
            return 0;
        }

        uint64_t* idList = topoNode->depIds;
        uint32_t count = topoNode->depNum;
        uint64_t hash = IdListHash(idList, count * sizeof(uint64_t), topoNode->depNum);
        IdListKey key = {hash, count};
        uint32_t byteLen = count * sizeof(uint64_t);
        for (uint32_t i = 0; i < count; i++) {
            MACHINE_LOGD(" ##[TopoProcessor] batch depend id list %lu", idList[i]);
        }

        auto& candidates = forwardMap_[key];
        for (const auto& entry : candidates) {
            if (std::memcmp(entry->data.get(), idList, byteLen) == 0) {
                InsertNewOldKeyIdMap(entry->keyId, topoNode);
                return entry->keyId;
            }
        }

        const uint64_t newId = nextId_++;
        auto newEntry = std::make_unique<IdList>(idList, count, newId);
        candidates.push_back(std::move(newEntry));
        InsertNewOldKeyIdMap(newId, topoNode);
        return newId;
    }

    void InsertNewOldKeyIdMap(const uint64_t newTopoId, CoreFunctionTopo* oldTopo)
    {
        std::vector<CoreFunctionTopo*>& oldTopoVec = newTopoIdToOldTopo_[newTopoId];
        oldTopoVec.push_back(oldTopo);
    }

    void IdValueHash(const unsigned char* data2, int len, uint64_t& h, const uint64_t m)
    {
        const int index0 = 0;
        const int index1 = 1;
        const int index2 = 2;
        const int index3 = 3;
        const int index4 = 4;
        const int index5 = 5;
        const int index6 = 6;
        const int index7 = 7;
        const int shift = 8;

        switch (len & index7) {
            case index7: {
                h ^= (uint64_t(data2[index6]) << (shift * index6));
                break;
            }
            case index6: {
                h ^= (uint64_t(data2[index5]) << (shift * index5));
                break;
            }
            case index5: {
                h ^= (uint64_t(data2[index4]) << (shift * index4));
                break;
            }
            case index4: {
                h ^= (uint64_t(data2[index3]) << (shift * index3));
                break;
            }
            case index3: {
                h ^= (uint64_t(data2[index2]) << (shift * index2));
                break;
            }
            case index2: {
                h ^= (uint64_t(data2[index1]) << shift);
                break;
            }
            case index1: {
                h ^= uint64_t(data2[index0]);
                h *= m;
                break;
            }
            default: {
                break;
            }
        };
    }

    uint64_t IdListHash(const void* key, int len, unsigned int seed) NO_UBSAN
    {
        const uint64_t m = 0xc6a4a7935bd1e995;
        const int r = 47;
        uint64_t h = seed ^ (len * m);
        const uint64_t* data = static_cast<const uint64_t*>(key);
        const uint64_t* end = data + (len / 8);
        while (data != end) {
            uint64_t k = *data++;
            k *= m;
            k ^= k >> r;
            k *= m;
            h ^= k;
            h *= m;
        }

        IdValueHash(reinterpret_cast<const unsigned char*>(data), len, h, m);
        h ^= h >> r;
        h *= m;
        h ^= h >> r;
        return h;
    }

private:
    uint64_t nextId_{1};
    std::shared_ptr<CoreFunctionTopoCache> srcTopoData_;
    uint64_t srcTopoNum_;
    std::unordered_map<IdListKey, std::vector<std::unique_ptr<IdList>>, IdListKeyHash> forwardMap_;
    std::map<uint64_t, std::vector<CoreFunctionTopo*>> newTopoIdToOldTopo_;
    std::map<uint64_t, CoreFunctionTopo*> newTopoIdToNewTopo_;
    uint64_t virtualTopoSize_{0};
    uint64_t virtualTopoNum_{0};
};

} // namespace npu::tile_fwk
