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
 * \file codegen_symbol.cpp
 * \brief
 */

#include "codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
AllocKey SymbolManager::CreateAllocKey(const std::shared_ptr<LogicalTensor>& tensor) const
{
    auto memType = tensor->GetMemoryTypeOriginal();
    if (OPERAND_TYPE_TO_MEMORY_TYPE.count(memType) == 0) {
        ASSERT(OperErr::OPERAND_TYPE_UNSUPPORTED, false)
            << "invalid memory type: " << static_cast<size_t>(memType) << ", tensor is " << tensor->Dump();
        return {};
    }

    const TileRange& range = tensor->memoryrange;
    auto bufferType = OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);
    AllocKey key = AllocKey(bufferType, range.start, range.end);
    return key;
}

AllocKey SymbolManager::CreateAllocKey(int tensorMagicNum) const
{
    std::shared_ptr<LogicalTensor> tensor = SymbolManager::GetTensorByMagic(tensorMagicNum);
    if (!tensor) {
        CODEGEN_LOGE(GenCodeErr::TENSOR_NOT_FOUND, "can not query tensor object from tensor magicnum: %d",
                     tensorMagicNum);
        return {};
    }

    return CreateAllocKey(tensor);
}

bool SymbolManager::BindAddrWithVariableName(const AllocKey& key, const std::string& varName,
                                             const std::string& varNameT)
{
    auto iter = key2VariableName_.find(key);
    if (iter != key2VariableName_.end()) {
        return true;
    } else {
        key2VariableName_.insert(std::pair<AllocKey, std::string>(key, varName));
        key2VariableNameTileTensor_.insert(std::pair<AllocKey, std::string>(key, varNameT));
    }
    return false;
}

std::shared_ptr<LogicalTensor> SymbolManager::GetTensorByMagic(int magicNum) const
{
    auto iter = tensorMap_.find(magicNum);
    if (iter != tensorMap_.end()) {
        return iter->second;
    } else {
        ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false) << "can not find tensor by magicNum:" << magicNum;
        return nullptr;
    }
}

std::string SymbolManager::FormatAllocKey(const AllocKey& key)
{
    auto [bufType, start, end] = key;
    std::ostringstream os;
    os << "alloc identifier <buf_type=" << OperandTypeToStr(bufType) << ", ";
    os << "range_start=" << start << ", ";
    os << "range_end=" << end << ">";
    return os.str();
}

std::string SymbolManager::QueryVariableName(const AllocKey& key)
{
    CODEGEN_LOGI("query varname by identifier: %s", FormatAllocKey(key).c_str());
    auto iter = key2VariableName_.find(key);
    ASSERT(GenCodeErr::SYMBOL_NOT_FOUND, iter != key2VariableName_.end())
        << "QueryVariableName Failed: UNDEFINED_VAR !!! AllocKey: " << FormatAllocKey(key);
    return iter->second;
}

std::string SymbolManager::QueryVariableNameTileTensor(const AllocKey& key)
{
    CODEGEN_LOGI("query varname TileTensor mode by identifier: %s", FormatAllocKey(key).c_str());

    auto iter = key2VariableNameTileTensor_.find(key);
    if (iter != key2VariableNameTileTensor_.end()) {
        return iter->second;
    }

    CODEGEN_LOGE(GenCodeErr::SYMBOL_NOT_FOUND, "failed to query by identifier: %s", FormatAllocKey(key).c_str());
    ASSERT(GenCodeErr::SYMBOL_NOT_FOUND, false)
        << "QueryVariableNameTileTensor Failed: UNDEFINED_VAR !!! AllocKey: " << FormatAllocKey(key);
    return "UNDEFINED_VAR";
}

// NEXTNEXT: after TileTensor Mode is applied to all tensor, just retain TileTensor Mode
std::string SymbolManager::QueryVarNameByTensorMagic(int magic, bool isTileTensor)
{
    CODEGEN_LOGI("QueryVarNameByTensorMagic: magic is %d, isTileTensor is %d", magic, isTileTensor);
    AllocKey key = CreateAllocKey(magic);
    std::string varName = isTileTensor ? QueryVariableNameTileTensor(key) : QueryVariableName(key);
    return varName;
}

std::string SymbolManager::FindUsingName(const TileTensorUsing& tileTensorUsing) const
{
    for (const auto& usingPair : tileTensorUsing_) {
        if (usingPair.second == tileTensorUsing) {
            return usingPair.first;
        }
    }
    return "";
}

std::string SymbolManager::AddTileTensorUsing(const TileTensorUsing& tileTensorUsing)
{
    std::string tensorUsingType = FindUsingName(tileTensorUsing);
    if (!tensorUsingType.empty()) {
        CODEGEN_LOGI("found tensorUsingType %s", tensorUsingType.c_str());
        return tensorUsingType;
    }
    tensorUsingType = GenTensorUsingName(tileTensorUsing);
    CODEGEN_LOGI("Add tensorUsingType %s", tensorUsingType.c_str());
    tileTensorUsing_.insert({tensorUsingType, tileTensorUsing});
    CODEGEN_LOGI("insert tensorUsingType %s = %s", tensorUsingType.c_str(), tileTensorUsing.ToString().c_str());
    return tensorUsingType;
}

TileTensorKey SymbolManager::BuildTileTensorKey(const TileTensor& tileTensor) const
{
    return {tileTensor.dim,   tileTensor.dtype,    tileTensor.bufVar,
            tileTensor.shape, tileTensor.rawShape, tileTensor.localBufOffset};
}

std::string SymbolManager::AddTileTensor(int opMagic, const TileTensor& tileTensor)
{
    TileTensorKey tileTensorKey = BuildTileTensorKey(tileTensor);

    CODEGEN_LOGI("AddTileTensor --> tensorMagic=%d, opMagic=%d, key={%s}", tileTensor.magic, opMagic,
                 tileTensorKey.ToString().c_str());

    auto tileTensorByKeyIter = tileTensorByKey_.find(tileTensorKey);
    bool isNewTileTensor = false;
    if (tileTensorByKeyIter == tileTensorByKey_.end()) {
        tileTensorStorage_.push_back(tileTensor);
        const TileTensor* storedTileTensor = &tileTensorStorage_.back();
        tileTensorByKeyIter = tileTensorByKey_.emplace(std::move(tileTensorKey), std::cref(*storedTileTensor)).first;
        isNewTileTensor = true;
    }

    const TileTensor& storedTileTensor = tileTensorByKeyIter->second.get();
    TileTensorMagicKey key{tileTensor.magic, opMagic};
    auto& tileTensorByMagic = tileTensor.isInLoop ? tileTensorByMagicInLoop_ : tileTensorByMagic_;
    auto [iter, inserted] = tileTensorByMagic.emplace(key, std::cref(storedTileTensor));
    ASSERT(GenCodeErr::TENSOR_MAGIC_CONFLICT, inserted || &iter->second.get() == &storedTileTensor)
        << "TileTensor conflict for tensor magic " << tileTensor.magic << ", op magic " << opMagic
        << "\nnew tile tensor: " << storedTileTensor.ToString()
        << "\nexisting tile tensor: " << iter->second.get().ToString();

    std::string tensorName = inserted ? storedTileTensor.tensorName : iter->second.get().tensorName;

    CODEGEN_LOGI(
        "tileTensorStorage_.insert result is %d Add TileTensor --> tensor magic: %d, op magic: %d, tensor name: %s, "
        "tile tensor: %s",
        isNewTileTensor, tileTensor.magic, opMagic, tensorName.c_str(), storedTileTensor.ToString().c_str());
    return tensorName;
}

const TileTensor* SymbolManager::QueryTileTensorByMagic(int tensorMagic, int opMagic) const
{
    CODEGEN_LOGI("QueryTileTensorByMagic tensor magic is %d, op magic is %d", tensorMagic, opMagic);
    auto iter = tileTensorByMagic_.find({tensorMagic, opMagic});
    if (iter != tileTensorByMagic_.end()) {
        return &iter->second.get();
    }
    return nullptr;
}

const TileTensor* SymbolManager::QueryTileTensorInLoopByMagic(int tensorMagic, int opMagic) const
{
    CODEGEN_LOGI("QueryTileTensorInLoopByMagic tensor magic is %d, op magic is %d", tensorMagic, opMagic);
    auto iter = tileTensorByMagicInLoop_.find({tensorMagic, opMagic});
    if (iter != tileTensorByMagicInLoop_.end()) {
        return &iter->second.get();
    }
    return nullptr;
}

void SymbolManager::InsertTensorNameInLoopToFullDim(const std::string& tensorName, const std::string& fullDimTensorName)
{
    auto res = tensorNameInLoopToFullDim_.insert({tensorName, fullDimTensorName});
    CODEGEN_LOGI("res is %d, InsertTensorNameInLoopToFullDim %s -> %s", res.second, tensorName.c_str(),
                 fullDimTensorName.c_str());
}

std::string SymbolManager::QueryTileTensorFullDimByTensorInLoop(const std::string& tensorName)
{
    std::string fullDimTensorName;
    auto iter = tensorNameInLoopToFullDim_.find(tensorName);
    if (iter != tensorNameInLoopToFullDim_.end()) {
        CODEGEN_LOGI("QueryTileTensorFullDimByTensorInLoop found tensor in loop %s, full dim tensor is %s",
                     tensorName.c_str(), iter->second.c_str());
        fullDimTensorName = iter->second;
    }

    ASSERT(GenCodeErr::SYMBOL_NOT_FOUND, !fullDimTensorName.empty())
        << "tensor in loop: " << tensorName << " is not found in tensorNameInLoopToFullDim_!!!";
    return fullDimTensorName;
}

void SymbolManager::GenUsingList(std::ostringstream& oss)
{
    for (const auto& usingPair : tileTensorUsing_) {
        const std::string& usingName = usingPair.first;
        const TileTensorUsing& tileTensorUsing = usingPair.second;
        oss << "using " << usingName << " = " << tileTensorUsing.ToString();
    }
}

std::string SymbolManager::GenTileTensorDefList()
{
    std::ostringstream oss;
    for (const auto& tileTensor : tileTensorStorage_) {
        oss << tileTensor.ToString();
    }
    return oss.str();
}

std::string SymbolManager::GenNewTileTensorDefs()
{
    std::ostringstream oss;
    for (size_t i = tileTensorOutputIdx_; i < tileTensorStorage_.size(); ++i) {
        oss << tileTensorStorage_[i].ToString();
    }
    tileTensorOutputIdx_ = tileTensorStorage_.size();
    return oss.str();
}

void SymbolManager::PrintAddrAllocs(std::ostringstream& oss) const
{
    for (const auto& addrAlloc : addrAllocs_) {
        if (tensorUseNormalAlloc_.find(addrAlloc.allocKey) != tensorUseNormalAlloc_.end()) {
            oss << addrAlloc.allocNormal;
        }
        oss << addrAlloc.allocForTileTensor;
    }
}

} // namespace npu::tile_fwk
