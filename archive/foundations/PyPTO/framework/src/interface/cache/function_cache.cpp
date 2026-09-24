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
 * \file function_cache.cpp
 * \brief
 */

#include "function_cache.h"
#include <elf.h>
#include <cstdio>
#include <fstream>
#include "tilefwk/pypto_fwk_log.h"
#include "interface/utils/common.h"
#include "utils/file_utils.h"
#include "tilefwk/error_code.h"
#include "topo_processor.h"
#include "securec.h"

namespace npu::tile_fwk {
std::optional<CacheValue> FunctionCache::Get(HashKey key)
{
    std::lock_guard<std::mutex> cLockGuard(lock_);
    getCnt_++;
    if (auto it = cache_.find(key); it != cache_.end()) {
        hitCnt_++;
        return it->second;
    } else {
        return std::nullopt;
    }
}

static void AssertValidCoreType(uint64_t coreType)
{
    FE_ASSERT(FeError::INVALID_TYPE, (coreType == static_cast<uint64_t>(CoreType::AIV)) ||
                                         (coreType == static_cast<uint64_t>(CoreType::AIC)) ||
                                         (coreType == static_cast<uint64_t>(CoreType::HUB)) ||
                                         (coreType == static_cast<uint64_t>(CoreType::HUB_MIX)) ||
                                         (coreType == static_cast<uint64_t>(CoreType::AICPU)))
        << "Invalid core type: " << coreType;
}

void FunctionCache::UpdateTopoCache(const Function& func, CacheValue& value)
{
    uint64_t totalSize = 0;
    uint32_t topoNum = func.topoInfo_.topology_.size();
    totalSize += (topoNum * sizeof(uint64_t));
    for (uint32_t i = 0; i < topoNum; i++) {
        uint32_t depNum = func.topoInfo_.topology_[i].outGraph.size();
        uint32_t extParamNum = func.topoInfo_.topology_[i].extParamNum;
        totalSize += sizeof(CoreFunctionTopo) + sizeof(uint64_t) * depNum + sizeof(uint64_t) * extParamNum;
    }
    size_t topoSize = totalSize + sizeof(uint64_t);
    value.topoCache = CacheValue::CreateCache<CoreFunctionTopoCache>(topoSize);
    uint8_t* topoCachePtr = reinterpret_cast<uint8_t*>(value.topoCache.get());
    *reinterpret_cast<uint64_t*>(topoCachePtr) = totalSize;

    uint64_t* offsetPtr = reinterpret_cast<uint64_t*>(topoCachePtr + sizeof(uint64_t));
    uint8_t* topoPtr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t*>(topoCachePtr) + topoNum + 1);
    uint64_t curCoreFuncOffset = sizeof(uint64_t) + topoNum * sizeof(uint64_t);
    for (uint32_t i = 0; i < topoNum; i++) {
        offsetPtr[i] = curCoreFuncOffset;
        CoreFunctionTopo* tempPtr = reinterpret_cast<CoreFunctionTopo*>(topoPtr);
        tempPtr->coreType = static_cast<uint64_t>(func.GetSubFuncInvokeInfo(i).GetGraphType());
        AssertValidCoreType(tempPtr->coreType);
        tempPtr->psgId = func.GetSubFuncInvokeInfo(i).GetProgramId();
        tempPtr->readyCount = func.topoInfo_.topology_[i].readyState;
        tempPtr->depNum = func.topoInfo_.topology_[i].outGraph.size();
        tempPtr->extParamNum = func.topoInfo_.topology_[i].extParamNum;
        tempPtr->extType = func.topoInfo_.topology_[i].extType;
        MACHINE_LOGD("[function cache]topo %u, readycount:%ld, depnum:%lu, coreType:%lu, extType:%u", i,
                     tempPtr->readyCount, tempPtr->depNum, tempPtr->coreType, tempPtr->extType);
        uint32_t j = 0;
        for (auto& ele : func.topoInfo_.topology_[i].outGraph) {
            tempPtr->depIds[j] = ele;
            j++;
            MACHINE_LOGD("[function cache]depend %d", ele);
        }
        for (auto& ele : func.topoInfo_.topology_[i].extParams) {
            tempPtr->depIds[j++] = static_cast<uint64_t>(ele);
        }
        uint32_t tempLength = sizeof(CoreFunctionTopo) + sizeof(uint64_t) * (tempPtr->depNum + tempPtr->extParamNum);
        curCoreFuncOffset += tempLength;
        topoPtr += tempLength;
    }
    value.header.coreFunctionNum = topoNum;
    FE_ASSERT(FeError::INVALID_VAL, topoNum != 0) << "Invalid topoNum: " << topoNum;

    TopoProcessor processor(value.topoCache, topoNum);
    std::tuple<std::shared_ptr<CoreFunctionTopoCache>, uint64_t> newTopo = processor.MergeBatchDepend(10, 1);
    value.topoCache = std::get<0>(newTopo);
    value.header.virtualFunctionNum = std::get<1>(newTopo);
}

std::vector<uint8_t> LoadBinData(const std::string& binPath)
{
    std::vector<uint8_t> text;

    uint64_t fileSize = GetFileSize(binPath);
    std::vector<char> buf(fileSize);
    std::ifstream file(binPath);
    file.read(buf.data(), static_cast<std::streamsize>(fileSize));

    auto elfHeader = reinterpret_cast<Elf64_Ehdr*>(buf.data());
    if (elfHeader->e_ident[EI_MAG0] != ELFMAG0 || elfHeader->e_ident[EI_MAG1] != ELFMAG1 ||
        elfHeader->e_ident[EI_MAG2] != ELFMAG2 || elfHeader->e_ident[EI_MAG3] != ELFMAG3) {
        return text;
    }

    auto sectionHeaders = reinterpret_cast<Elf64_Shdr*>(reinterpret_cast<uint64_t>(elfHeader) + elfHeader->e_shoff);
    auto shstrHeader = &sectionHeaders[elfHeader->e_shstrndx];
    auto strtbl = buf.data() + shstrHeader->sh_offset;
    for (int i = 0; i < elfHeader->e_shnum; i++) {
        auto section = &sectionHeaders[i];
        auto sectionName = strtbl + section->sh_name;
        if (strcmp(sectionName, ".text") == 0) {
            text.resize(section->sh_size);
            memcpy_s(text.data(), section->sh_size, buf.data() + section->sh_offset, section->sh_size);
            break;
        }
    }

    return text;
}

void FunctionCache::UpdateBinCache(const Function& func, CacheValue& value)
{
    std::map<uint64_t, std::vector<uint8_t>> binMap;
    uint64_t totalSize = 0;
    for (auto& ele : func.programs_) {
        auto leafFuncAttr = ele.second->GetLeafFuncAttribute();
        FE_ASSERT(FeError::INVALID_VAL, leafFuncAttr != nullptr) << "Leaf function attr not found";
        auto binPath = leafFuncAttr->binPath;
        if (!RealPath(binPath).empty()) {
            auto binData = LoadBinData(binPath);
            FE_ASSERT(binData.size() != 0) << "binData is empty.";
            totalSize += binData.size() + sizeof(uint64_t);
            binMap[ele.first] = std::move(binData);
        } else if (leafFuncAttr->coreType == CoreType::AICPU) {
            std::vector<uint8_t> binData(0, 0);
            totalSize += binData.size() + sizeof(uint64_t);
            binMap[ele.first] = std::move(binData);
        } else {
            MACHINE_LOGE(InternalError::MACHINE_INNER_ERROR, "bin path %s is not existed", binPath.c_str());
            abort();
        }
    }

    uint64_t progNum = func.programs_.size();
    totalSize += progNum * sizeof(uint64_t);

    size_t binSize = totalSize + sizeof(uint64_t);
    value.binCache = CacheValue::CreateCache<CoreFunctionBinCache>(binSize);
    uint8_t* buf = reinterpret_cast<uint8_t*>(value.binCache.get());

    value.header.programFuncionNum = progNum;
    value.binCache->dataSize = totalSize;

    auto binOffsets = reinterpret_cast<uint64_t*>(buf + sizeof(CoreFunctionBinCache));
    uint64_t curOffset = sizeof(uint64_t) + progNum * sizeof(uint64_t);
    for (auto& ele : func.programs_) {
        *binOffsets++ = curOffset;
        auto* funcBin = reinterpret_cast<CoreFunctionBin*>(buf + curOffset);
        auto binData = binMap[ele.first];
        funcBin->size = binData.size();
        memcpy_s(funcBin->data, binData.size(), binData.data(), funcBin->size);
        curOffset += sizeof(CoreFunctionBin) + funcBin->size;
    }
}

void FunctionCache::UpdateReadyFunction(const Function& func, CacheValue& value)
{
    uint64_t readyNum = func.GetAllReadySubGraphCount();
    uint64_t totalSize = readyNum * sizeof(ReadyCoreFunction);
    size_t size = totalSize + sizeof(uint64_t);
    value.readyListCache = CacheValue::CreateCache<ReadyCoreFunctionCache>(size);
    uint8_t* readyFuncPtr = reinterpret_cast<uint8_t*>(value.readyListCache.get());
    ReadyCoreFunction* listPtr = reinterpret_cast<ReadyCoreFunction*>(reinterpret_cast<uint64_t*>(readyFuncPtr) + 1);
    // ReadyCoreFunctionCache
    size_t index = 0;
    for (size_t i = 0; i < func.GetReadySubGraphCount(CoreType::AIC); i++) {
        listPtr[index].id = func.GetReadySubGraphId(CoreType::AIC, i);
        listPtr[index].coreType = static_cast<uint64_t>(CoreType::AIC);
        index++;
    }

    for (size_t i = 0; i < func.GetReadySubGraphCount(CoreType::AIV); i++) {
        listPtr[index].id = func.GetReadySubGraphId(CoreType::AIV, i);
        listPtr[index].coreType = static_cast<uint64_t>(CoreType::AIV);
        index++;
    }

    for (size_t i = 0; i < func.GetReadySubGraphCount(CoreType::AICPU); i++) {
        listPtr[index].id = func.GetReadySubGraphId(CoreType::AICPU, i);
        listPtr[index].coreType = static_cast<uint64_t>(CoreType::AICPU);
        index++;
    }
    value.header.readyCoreFunctionNum = readyNum;
    FE_ASSERT(FeError::INVALID_VAL, value.header.readyCoreFunctionNum != 0)
        << "readyCoreFunctionNum is 0, value.header.readyCoreFunctionNum=" << value.header.readyCoreFunctionNum;
}

void FunctionCache::Insert(const HashKey& key, Function& func)
{
    CacheValue cacheVal;
    if (func.IsFunctionTypeAndGraphType(
            {FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH, FunctionType::STATIC},
            {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH})) {
        if (func.rootFunc_ && func.GetFunctionType() == FunctionType::STATIC) {
            UpdateTopoCache(*func.rootFunc_, cacheVal);
            UpdateBinCache(*func.rootFunc_, cacheVal);
            UpdateReadyFunction(*func.rootFunc_, cacheVal);
        }
        cacheVal.SetCacheFunction(&func);
    } else {
        if (func.GetGraphType() == GraphType::BLOCK_GRAPH) {
            cacheVal.SetCacheFunction(&func);
        } else {
            return;
        }
    }
    Insert(key, cacheVal);
}

void FunctionCache::Insert(const HashKey& key, CacheValue value)
{
    std::lock_guard<std::mutex> cLockGuard(lock_);
    cache_[key] = value;
}

size_t FunctionCache::Size()
{
    std::lock_guard<std::mutex> cLockGuard(lock_);
    return cache_.size();
}

std::string FunctionCache::GetHitRate()
{
    std::lock_guard<std::mutex> cLockGuard(lock_);
    std::string temp = std::to_string(hitCnt_) + "/" + std::to_string(getCnt_);
    return temp;
}

Function* FunctionCache::GetCacheFunction(const HashKey& key)
{
    std::lock_guard<std::mutex> cLockGuard(lock_);
    getCnt_++;
    if (auto it = cache_.find(key); it != cache_.end()) {
        hitCnt_++;
        return it->second.GetFunction();
    } else {
        return nullptr;
    }
}

void FunctionCache::Reset()
{
    std::lock_guard<std::mutex> cLockGuard(lock_);
    for (auto& ele : cache_) {
        ele.second.topoCache = nullptr;
        ele.second.binCache = nullptr;
        ele.second.readyListCache = nullptr;
    }
    cache_.clear();
}

FunctionCache::~FunctionCache() {}

} // namespace npu::tile_fwk
