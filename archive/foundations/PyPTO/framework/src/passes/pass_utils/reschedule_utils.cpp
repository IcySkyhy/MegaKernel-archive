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
 * \file reschedule_utils.cpp
 * \brief
 */

#include "reschedule_utils.h"
#include <algorithm>
#include <unordered_map>

#include "interface/utils/common.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RecheduleUtils"

namespace npu::tile_fwk {
namespace {
constexpr uint64_t TOPO_HASH_BASE = 37;
constexpr uint64_t TOPO_HASH_MOD = 0xFFFFFFFFFFFFF;
constexpr uint64_t TOPO_HASH_XOR_SEED = 0x12345678;
} // namespace

bool RescheduleUtils::isAllocOp(Operation* op)
{
    static std::unordered_set<Opcode> allocOpcodes = {
        Opcode::OP_L1_ALLOC,  Opcode::OP_UB_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC,
        Opcode::OP_L0C_ALLOC, Opcode::OP_BT_ALLOC, Opcode::OP_FIX_ALLOC, Opcode::OP_REG_ALLOC,
    };
    return allocOpcodes.find(op->GetOpcode()) != allocOpcodes.end();
}

void RescheduleUtils::SortUnique(std::vector<int>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

RescheduleUtils::TensorDependencyMap RescheduleUtils::BuildTensorDependencyInfo(const OperationsViewer& opList,
                                                                                bool collectConsumerColors)
{
    TensorDependencyMap tensorDeps;
    tensorDeps.reserve(opList.size() * 2);
    for (size_t i = 0; i < opList.size(); i++) {
        const auto& op = opList[i];
        int subgraphId = op.GetSubgraphID();
        for (const auto& tensor : op.GetOOperands()) {
            if (tensor == nullptr) {
                continue;
            }
            auto& depInfo = tensorDeps[tensor.get()];
            depInfo.producerOps.emplace_back(static_cast<int>(i));
            if (subgraphId >= 0) {
                depInfo.producerColors.emplace_back(subgraphId);
            }
        }
        if (!collectConsumerColors || subgraphId < 0) {
            continue;
        }
        for (const auto& tensor : op.GetIOperands()) {
            if (tensor == nullptr) {
                continue;
            }
            tensorDeps[tensor.get()].consumerColors.emplace_back(subgraphId);
        }
    }

    for (auto& [tensor, depInfo] : tensorDeps) {
        (void)tensor;
        std::sort(depInfo.producerOps.begin(), depInfo.producerOps.end(),
                  [&](int lhs, int rhs) { return opList[lhs].GetOpMagic() < opList[rhs].GetOpMagic(); });
        depInfo.producerOps.erase(
            std::unique(depInfo.producerOps.begin(), depInfo.producerOps.end(),
                        [&](int lhs, int rhs) { return opList[lhs].GetOpMagic() == opList[rhs].GetOpMagic(); }),
            depInfo.producerOps.end());
        SortUnique(depInfo.producerColors);
        SortUnique(depInfo.consumerColors);
    }
    return tensorDeps;
}

uint64_t RescheduleUtils::GetOpcodeHash(const std::string& op)
{
    uint64_t hash = 0;
    for (char c : op) {
        hash = (hash * TOPO_HASH_BASE + static_cast<uint64_t>(c)) % TOPO_HASH_MOD;
    }
    return hash;
}

RescheduleUtils::TopoHashContribution RescheduleUtils::BuildTopoHashContribution(const std::vector<int>& producerOps,
                                                                                 const std::vector<uint64_t>& hashList)
{
    TopoHashContribution contribution;
    for (int producerIdx : producerOps) {
        contribution.polynomial = (contribution.polynomial * TOPO_HASH_BASE +
                                   (hashList[producerIdx] ^ TOPO_HASH_XOR_SEED)) %
                                  TOPO_HASH_MOD;
        contribution.power = (contribution.power * TOPO_HASH_BASE) % TOPO_HASH_MOD;
    }
    return contribution;
}

uint64_t RescheduleUtils::ApplyTopoHashContribution(uint64_t hash, const TopoHashContribution& contribution)
{
    return (hash * contribution.power + contribution.polynomial) % TOPO_HASH_MOD;
}

void RescheduleUtils::BuildInputTopoHashList(const OperationsViewer& opList, const TensorDependencyMap& tensorDeps,
                                             std::vector<uint64_t>& hashList)
{
    hashList.assign(opList.size(), 0);
    // opList must be topologically sorted so every producer hash is finalized before the first consumer caches it.
    std::unordered_map<const LogicalTensor*, TopoHashContribution> tensorHashCache;
    tensorHashCache.reserve(tensorDeps.size());
    for (size_t i = 0; i < opList.size(); i++) {
        uint64_t hash = GetOpcodeHash(opList[i].GetOpcodeStr());
        for (const auto& tensor : opList[i].GetIOperands()) {
            if (tensor == nullptr) {
                continue;
            }
            auto depIter = tensorDeps.find(tensor.get());
            if (depIter == tensorDeps.end()) {
                continue;
            }
            const auto& depInfo = depIter->second;
            TopoHashContribution contribution;
            auto cacheIter = tensorHashCache.find(tensor.get());
            if (cacheIter != tensorHashCache.end()) {
                contribution = cacheIter->second;
            } else {
                contribution = BuildTopoHashContribution(depInfo.producerOps, hashList);
                tensorHashCache.emplace(tensor.get(), contribution);
            }
            hash = ApplyTopoHashContribution(hash, contribution);
        }
        hashList[i] = hash;
    }
}

void RescheduleUtils::BuildInputTopoHashList(const OperationsViewer& opList, std::vector<uint64_t>& hashList)
{
    auto tensorDeps = BuildTensorDependencyInfo(opList, false);
    BuildInputTopoHashList(opList, tensorDeps, hashList);
}

// vector size == 2
// 这里in_graph和out_graph只考虑了数据依赖，需要确认控制依赖是不是也要加入到in_graph和out_graph中
std::vector<std::vector<std::vector<int>>> RescheduleUtils::GetInOutGraphs(const std::vector<Operation*>& opList)
{
    std::vector<std::vector<int>> inGraph(opList.size());
    std::vector<std::vector<int>> outGraph(opList.size());
    std::map<int, size_t> magic2Index;
    for (size_t i = 0; i < opList.size(); i++) {
        magic2Index[opList[i]->GetOpMagic()] = i;
    }
    std::unordered_map<int, std::vector<size_t>> tensor2ProducerIdx;
    for (size_t i = 0; i < opList.size(); i++) {
        for (auto& inTensor : opList[i]->GetIOperands()) {
            int tensorMagic = inTensor->GetMagic();
            auto cacheIt = tensor2ProducerIdx.find(tensorMagic);
            if (cacheIt == tensor2ProducerIdx.end()) {
                std::vector<size_t> prodIdx;
                for (auto* producer : inTensor->GetProducers()) {
                    auto iter = magic2Index.find(producer->GetOpMagic());
                    if (iter != magic2Index.end()) {
                        prodIdx.push_back(iter->second);
                    }
                }
                cacheIt = tensor2ProducerIdx.emplace(tensorMagic, std::move(prodIdx)).first;
            }
            for (size_t prodIdx : cacheIt->second) {
                inGraph[i].push_back(prodIdx);
                outGraph[prodIdx].push_back(i);
            }
        }
    }
    std::vector<std::vector<std::vector<int>>> inOutGraph{inGraph, outGraph};
    return inOutGraph;
}

PipeType RescheduleUtils::GetOpPipeType(const Operation* op)
{
    auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
    if (op->GetOpcode() == Opcode::OP_RESHAPE) {
        if (op->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            op->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            return PipeType::PIPE_MTE3;
        }
    }
    if (op->GetOpcode() == Opcode::OP_COPY_IN) {
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        auto dstMemType = attr->GetCopyInAttr().second;
        if (dstMemType == MemoryType::MEM_L1) {
            return PipeType::PIPE_MTE2;
        }
        if (dstMemType == MemoryType::MEM_UB) {
            return PipeType::PIPE_MTE2;
        }
    }
    if (op->GetOpcode() == Opcode::OP_COPY_OUT) {
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        auto srcMemType = attr->GetCopyOutAttr().first;
        if (srcMemType == MemoryType::MEM_L0C) {
            return PipeType::PIPE_FIX;
        }
        if (srcMemType == MemoryType::MEM_UB) {
            return PipeType::PIPE_MTE3;
        }
        if (srcMemType == MemoryType::MEM_L1) {
            return PipeType::PIPE_MTE3;
        }
    }
    return opcfg.pipeIdStart_;
}

void GetConvOpAttrStr(std::stringstream& ss, const Operation* op)
{
    ss << "<" << op->GetIntAttribute(ConvOpAttributeKey::cin) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::cout) << ",",
        ss << op->GetIntAttribute(ConvOpAttributeKey::paddingLeft) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::paddingTop) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::paddingRight) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::paddingBottom) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::strideh) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::stridew) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hposX) << ",",
        ss << op->GetIntAttribute(ConvOpAttributeKey::hsteP) << ",",
        ss << op->GetIntAttribute(ConvOpAttributeKey::wposX) << ",",
        ss << op->GetIntAttribute(ConvOpAttributeKey::wstep) << ",",
        ss << op->GetIntAttribute(ConvOpAttributeKey::hoffsetY) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::woffsetY) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::reluType) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::reluAlpha) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::clearFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hasAccFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hasEltFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hasBiasFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::eltBrcbFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::eltMode) << ">";
}

unsigned long RescheduleUtils::ComputeOperationHash(const Operation* op)
{
    std::stringstream ss;
    ss << op->GetOpcodeStr();

    // 不需要cycle数校准的op。
    switch (op->GetOpcode()) {
        case Opcode::OP_L1_ALLOC:
        case Opcode::OP_UB_ALLOC:
        case Opcode::OP_L0A_ALLOC:
        case Opcode::OP_L0B_ALLOC:
        case Opcode::OP_L0C_ALLOC:
        case Opcode::OP_BT_ALLOC:
        case Opcode::OP_FIX_ALLOC:
        case Opcode::OP_SYNC_SRC:
        case Opcode::OP_SYNC_DST:
        case Opcode::OP_CV_SYNC_SRC:
        case Opcode::OP_CV_SYNC_DST:
        case Opcode::OP_PHASE1:
        case Opcode::OP_PHASE2:
        case Opcode::OP_BAR_V:
        case Opcode::OP_BAR_M:
            return 0;
        default:
            break;
    }

    switch (op->GetOpcode()) {
        case Opcode::OP_CONV2D:
        case Opcode::OP_CONV3D:
        case Opcode::OP_CONV_ADD:
            GetConvOpAttrStr(ss, op);
            break;
        default:
            break;
    }

    for (const auto& inp : op->GetIOperands()) {
        ss << "[i(";
        for (auto s : inp->GetShape()) {
            ss << s << ",";
        }
        ss << ")" << DataType2String(inp->Datatype()) << "]";
    }
    for (const auto& inp : op->GetOOperands()) {
        ss << "[o(";
        for (auto s : inp->GetShape()) {
            ss << s << ",";
        }
        ss << ")" << DataType2String(inp->Datatype()) << "]";
    }

    std::hash<std::string> hasher;
    auto result = hasher(ss.str());

    return result;
}

void RescheduleUtils::EraseOpsBelongToFunc(std::set<Operation*, LogicalTensor::CompareOp>& ops, Function* funcPtr)
{
    for (auto it = ops.begin(); it != ops.end();) {
        if ((*it)->BelongTo() == funcPtr) {
            it = ops.erase(it);
        } else {
            ++it;
        }
    }
}

void RescheduleUtils::ClearInputConsProd(Operation& op, Function* funcPtr,
                                         const std::unordered_set<LogicalTensorPtr>& incastSet)
{
    for (auto& inOperand : op.GetIOperands()) {
        if (incastSet.count(inOperand) == 0) {
            auto& prods = inOperand->GetProducers();
            EraseOpsBelongToFunc(prods, funcPtr);
        }
        auto& cons = inOperand->GetConsumers();
        EraseOpsBelongToFunc(cons, funcPtr);
    }
}

void RescheduleUtils::ClearOutputConsProd(Operation& op, Function* funcPtr,
                                          const std::unordered_set<LogicalTensorPtr>& outcastSet)
{
    for (auto& outOperand : op.GetOOperands()) {
        auto& prods = outOperand->GetProducers();
        EraseOpsBelongToFunc(prods, funcPtr);

        if (outcastSet.count(outOperand) == 0) {
            auto& cons = outOperand->GetConsumers();
            EraseOpsBelongToFunc(cons, funcPtr);
        }
    }
}

void RescheduleUtils::UpdateTensorConsProd(Function* funcPtr)
{
    std::unordered_map<int, std::set<std::shared_ptr<LogicalTensor>, TensorPtrComparator>> tensorMap;
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> inverseMap;
    std::unordered_set<LogicalTensorPtr> incastSet;
    std::unordered_set<LogicalTensorPtr> outcastSet;
    for (auto& incast : funcPtr->inCasts_) {
        incastSet.emplace(incast);
    }
    for (auto& outcast : funcPtr->outCasts_) {
        outcastSet.emplace(outcast);
    }
    for (auto& op : funcPtr->Operations()) {
        ClearInputConsProd(op, funcPtr, incastSet);
        ClearOutputConsProd(op, funcPtr, outcastSet);
    }
    for (auto& op : funcPtr->Operations()) {
        for (auto& inOperand : op.GetIOperands()) {
            inOperand->AddConsumer(op);
            tensorMap[inOperand->GetRawTensor()->GetRawMagic()].insert(inOperand);
            inverseMap[inOperand->GetMagic()] = inOperand;
        }
    }
    for (auto& op : funcPtr->Operations()) {
        if (op.GetOpcodeStr().find("ALLOC") != std::string::npos) {
            op.oOperand.clear();
            continue;
        }
        for (auto& outOperand : op.GetOOperands()) {
            outOperand->AddProducer(op);
            tensorMap[outOperand->GetRawTensor()->GetRawMagic()].insert(outOperand);
            inverseMap[outOperand->GetMagic()] = outOperand;
        }
    }
}

void RescheduleUtils::PrintColorNode(Function& func)
{
    std::map<int, std::vector<size_t>> colorNode;
    for (size_t i = 0; i < func.Operations().size(); i++) {
        auto& op = func.Operations()[i];
        auto color = op.GetSubgraphID();
        colorNode[color].push_back(i);
    }
    for (auto color : colorNode) {
        std::string colorInfo;
        colorInfo += "color " + std::to_string(color.first) + " : {";
        for (auto& opIdx : color.second) {
            colorInfo += "op " + std::to_string(func.Operations()[opIdx].GetOpMagic()) + "--" +
                         func.Operations()[opIdx].GetOpcodeStr() + ", ";
        }
        colorInfo += "}";
        APASS_LOG_DEBUG_F(Elements::Function, "%s", colorInfo.c_str());
    }
}
} // namespace npu::tile_fwk
