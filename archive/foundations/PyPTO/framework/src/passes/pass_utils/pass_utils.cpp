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
 * \file pass_utils.cpp
 * \brief
 */

#include "pass_utils.h"

#include <algorithm>
#include <climits>
#include <sstream>
#include <unordered_map>
#include "interface/operation/attribute.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/error.h"
#include "tilefwk/platform.h"

#define MODULE_NAME "FunctionUtils"

namespace npu::tile_fwk {

void FunctionUtils::RelinkOperationInput(Operation* op, const size_t inputIndex, const Operation* targetOp,
                                         const size_t outputIndex)
{
    if (op == nullptr || targetOp == nullptr) {
        return;
    }
    if (inputIndex >= op->GetIOperands().size() || outputIndex >= targetOp->GetOOperands().size()) {
        return;
    }
    LogicalTensorPtr inputTenosr = op->GetIOperands().at(inputIndex);
    LogicalTensorPtr targetOutputTenosr = targetOp->GetOOperands().at(outputIndex);
    // update consumer of operation
    inputTenosr->RemoveConsumer(*op);
    targetOutputTenosr->AddConsumer(*op);
    // replace input tensor of op with the output tensor of target op
    op->ReplaceInputOperand(inputTenosr, targetOutputTenosr);
}

bool IsOverlapping(const LogicalTensor& a, const LogicalTensor& b)
{
    for (size_t i = 0; i < a.shape.size(); ++i) {
        int aStart = a.offset[i];
        int aEnd = aStart + a.shape[i];
        int bStart = b.offset[i];
        int bEnd = bStart + b.shape[i];

        // 如果任意一维不重叠，则整体不重叠
        if (aEnd <= bStart || aStart >= bEnd) {
            return false;
        }
    }
    return true;
}

// 计算矩形的体积（面积、体积等）
int CalculateVolume(const LogicalTensor& tensor)
{
    int volume = 1;
    for (int dim : tensor.shape) {
        volume *= dim;
    }
    return volume;
}

namespace {
bool HasNonImmediateDynValidShape(const std::vector<SymbolicScalar>& dynValidShape)
{
    return std::any_of(dynValidShape.begin(), dynValidShape.end(),
                       [](const SymbolicScalar& dim) { return !dim.IsImmediate(); });
}

bool IsViewWithNonImmediateDynValidShape(Operation* op)
{
    if (op->GetOpcode() != Opcode::OP_VIEW || op->GetOOperands().empty()) {
        return false;
    }
    auto output = op->GetOOperands().front();
    if (output != nullptr && HasNonImmediateDynValidShape(output->GetDynValidShape())) {
        return true;
    }
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    return viewAttr != nullptr && HasNonImmediateDynValidShape(viewAttr->GetToDynValidShape());
}

bool HasOnlyDirectViewConsumersWithNonImmediateDynValidShape(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->GetConsumers().empty()) {
        return false;
    }
    for (auto* consumer : tensor->GetConsumers()) {
        if (!IsViewWithNonImmediateDynValidShape(consumer)) {
            return false;
        }
    }
    return true;
}

std::string JoinOpMagics(const std::vector<Operation*>& ops)
{
    std::ostringstream oss;
    for (size_t idx = 0; idx < ops.size(); ++idx) {
        if (idx > 0) {
            oss << ",";
        }
        oss << (ops[idx] == nullptr ? -1 : ops[idx]->GetOpMagic());
    }
    return oss.str();
}

std::string JoinTensorMagics(const std::vector<LogicalTensorPtr>& tensors)
{
    std::ostringstream oss;
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        if (idx > 0) {
            oss << ",";
        }
        oss << (tensors[idx] == nullptr ? -1 : tensors[idx]->GetMagic());
    }
    return oss.str();
}
} // namespace

// 判断一组 LogicalTensor 是否可以拼接成一个大矩形
bool FunctionUtils::IsContinuous(const std::vector<std::shared_ptr<LogicalTensor>>& tensors)
{
    if (tensors.empty()) {
        return false;
    }

    size_t numDims = tensors[0]->shape.size();

    // 计算整体边界
    std::vector<int64_t> minCoords(numDims, INT_MAX);
    std::vector<int64_t> maxCoords(numDims, INT_MIN);

    for (const auto& tensor : tensors) {
        for (size_t i = 0; i < numDims; ++i) {
            minCoords[i] = std::min(minCoords[i], tensor->offset[i]);
            maxCoords[i] = std::max(maxCoords[i], tensor->offset[i] + tensor->shape[i]);
        }
    }

    // 计算整体体积
    int totalVolume = 1;
    for (size_t i = 0; i < numDims; ++i) {
        totalVolume *= (maxCoords[i] - minCoords[i]);
    }

    // 计算所有矩形的总体积
    int sumVolume = 0;
    for (const auto& tensor : tensors) {
        sumVolume += CalculateVolume(*tensor);
    }

    // 如果总体积不等于整体体积，说明有缝隙
    if (sumVolume != totalVolume) {
        return false;
    }

    // 检查是否有重叠
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = i + 1; j < tensors.size(); ++j) {
            if (IsOverlapping(*tensors[i], *tensors[j])) {
                return false;
            }
        }
    }

    return true;
}

NodeType FunctionUtils::GetNodeType(const LogicalTensor& tensor, const Function& function)
{
    // 检查是否为 INCAST
    auto in_it = std::find_if(function.inCasts_.begin(), function.inCasts_.end(),
                              [&tensor](const auto& t) { return t != nullptr && t.get() == &tensor; });
    if (in_it != function.inCasts_.end()) {
        return NodeType::INCAST;
    }

    // 检查是否为 OUTCAST
    auto out_it = std::find_if(function.outCasts_.begin(), function.outCasts_.end(),
                               [&tensor](const auto& t) { return t != nullptr && t.get() == &tensor; });
    if (out_it != function.outCasts_.end()) {
        return NodeType::OUTCAST;
    }

    // 既不是输入也不是输出，则为局部变量
    return NodeType::LOCAL;
}

std::unordered_map<MemoryType, int64_t> CommonUtils::GetLocalMemorySize()
{
    std::unordered_map<MemoryType, int64_t> localMemorySize;
    auto& die = Platform::Instance().GetDie();

    localMemorySize[MemoryType::MEM_UB] = die.GetMemoryLimit(MemoryType::MEM_UB);
    localMemorySize[MemoryType::MEM_L1] = die.GetMemoryLimit(MemoryType::MEM_L1);
    localMemorySize[MemoryType::MEM_L0A] = die.GetMemoryLimit(MemoryType::MEM_L0A);
    localMemorySize[MemoryType::MEM_L0B] = die.GetMemoryLimit(MemoryType::MEM_L0B);
    localMemorySize[MemoryType::MEM_L0C] = die.GetMemoryLimit(MemoryType::MEM_L0C);
    localMemorySize[MemoryType::MEM_L0AMX] = die.GetMemoryLimit(MemoryType::MEM_L0AMX);
    localMemorySize[MemoryType::MEM_L0BMX] = die.GetMemoryLimit(MemoryType::MEM_L0BMX);
    localMemorySize[MemoryType::MEM_BT] = die.GetMemoryLimit(MemoryType::MEM_BT);
    localMemorySize[MemoryType::MEM_FIX] = die.GetMemoryLimit(MemoryType::MEM_FIX);
    localMemorySize[MemoryType::MEM_FIX_QUANT_PRE] = die.GetMemoryLimit(MemoryType::MEM_FIX_QUANT_PRE);

    return localMemorySize;
}

std::vector<SymbolicScalar> CommonUtils::CreateConstIntVector(const std::vector<int64_t>& values)
{
    IRBuilder builder;
    std::vector<SymbolicScalar> result;
    result.reserve(values.size());
    for (auto value : values) {
        result.emplace_back(builder.CreateConstInt(value));
    }
    return result;
}

int CommonUtils::GetTensorSubgraphID(const LogicalTensorPtr& tensor) { return GetTensorSubgraphID(tensor.get()); }

int CommonUtils::GetTensorSubgraphID(const LogicalTensor* tensor)
{
    if (tensor == nullptr) {
        return NOT_IN_SUBGRAPH;
    }
    if (tensor->GetProducers().size() > 0) {
        auto& producers = tensor->GetProducers();
        return (*producers.begin())->GetSubgraphID();
    }
    if (tensor->GetConsumers().size() > 0) {
        auto& consumers = tensor->GetConsumers();
        return (*consumers.begin())->GetSubgraphID();
    }
    return NOT_IN_SUBGRAPH;
}

namespace {
bool MayOverlap(const Operation* op0, const Operation* op1)
{
    auto attr0 = std::dynamic_pointer_cast<AssembleOpAttribute>(op0->GetOpAttribute());
    auto attr1 = std::dynamic_pointer_cast<AssembleOpAttribute>(op1->GetOpAttribute());
    ASSERT(attr0 != nullptr && attr1 != nullptr) << "missing assemble attribute";

    auto getOffset = [](auto attr) {
        auto dynOffset = attr->GetToDynOffset();
        if (dynOffset.empty()) {
            dynOffset = SymbolicScalar::FromConcrete(attr->GetToOffset());
        }
        return dynOffset;
    };

    auto check = [](SymbolicScalar cond) {
        cond = cond.Simplify();
        return cond.ConcreteValid() && cond.Concrete() == true;
    };

    auto offset0 = getOffset(attr0);
    auto offset1 = getOffset(attr1);
    auto& shape0 = op0->GetIOperands()[0]->GetShape();
    auto& shape1 = op1->GetIOperands()[0]->GetShape();
    ASSERT(shape0.size() == shape1.size()) << "shape0 and shape1 must have the same size";

    for (size_t i = 0; i < shape0.size(); i++) {
        if (check(offset0[i] + shape0[i] <= offset1[i]) || check(offset1[i] + shape1[i] <= offset0[i])) {
            return false;
        }
    }
    return true;
}

bool MayOverlap(const std::vector<Operation*>& prods)
{
    for (size_t i = 0; i < prods.size(); i++) {
        for (size_t j = i + 1; j < prods.size(); j++) {
            if (MayOverlap(prods[i], prods[j])) {
                return true;
            }
        }
    }
    return false;
}

bool IsSerialDassemble(const Operation* prod)
{
    if (prod == nullptr || prod->GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }
    if (prod->GetIOperands().size() != 1) {
        return false;
    }
    if (!prod->HasAttr(OpAttributeKey::parallel)) {
        return false;
    }
    return !prod->GetBoolAttribute(OpAttributeKey::parallel);
}
} // namespace

Status FunctionUtils::InferOutcastWriteConflict(Function& function)
{
    for (auto& outcast : function.GetOutcast()) {
        auto& prodSet = outcast->GetProducers();
        bool hasConflict = std::any_of(prodSet.begin(), prodSet.end(),
                                       [](auto& prod) { return prod->GetOpcode() == Opcode::OP_ATOMIC_RMW; });
        if (hasConflict) {
            outcast->SetAttr(OpAttributeKey::writeConflict, true);
            continue;
        }

        std::vector<Operation*> prods;
        for (auto& prod : prodSet) {
            if (prod->GetOpcode() == Opcode::OP_ASSEMBLE || prod->GetOpcode() == Opcode::OP_ASSEMBLE_SSA) {
                prods.push_back(prod);
            }
            if (IsSerialDassemble(prod)) {
                outcast->SetAttr("NORMAL", true);
            }
        }
        if (MayOverlap(prods)) {
            outcast->SetAttr(OpAttributeKey::writeConflict, true);
        }
    }
    return SUCCESS;
}

void FunctionUtils::WarnAssembleDynValidShapeRisk(Function& function)
{
    struct AssembleOutputInfo {
        LogicalTensorPtr output;
        std::vector<Operation*> producers;
        std::vector<LogicalTensorPtr> inputsWithDynValidShape;
    };

    std::unordered_map<int, AssembleOutputInfo> outputInfoMap;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE || op.GetIOperands().empty() || op.GetOOperands().empty()) {
            continue;
        }
        bool hasInputDynValidShape = false;
        std::vector<LogicalTensorPtr> inputsWithDynValidShape;
        for (const auto& input : op.GetIOperands()) {
            if (input != nullptr && HasNonImmediateDynValidShape(input->GetDynValidShape())) {
                hasInputDynValidShape = true;
                inputsWithDynValidShape.emplace_back(input);
            }
        }
        if (!hasInputDynValidShape) {
            continue;
        }
        for (const auto& output : op.GetOOperands()) {
            if (output == nullptr) {
                continue;
            }
            auto& info = outputInfoMap[output->GetMagic()];
            info.output = output;
            info.producers.emplace_back(&op);
            info.inputsWithDynValidShape.insert(info.inputsWithDynValidShape.end(), inputsWithDynValidShape.begin(),
                                                inputsWithDynValidShape.end());
        }
    }

    std::vector<int> outputMagics;
    outputMagics.reserve(outputInfoMap.size());
    for (const auto& outputInfo : outputInfoMap) {
        outputMagics.emplace_back(outputInfo.first);
    }
    std::sort(outputMagics.begin(), outputMagics.end());

    for (const auto tensorMagic : outputMagics) {
        const auto& info = outputInfoMap.at(tensorMagic);
        if ((info.output != nullptr && HasNonImmediateDynValidShape(info.output->GetDynValidShape())) ||
            HasOnlyDirectViewConsumersWithNonImmediateDynValidShape(info.output)) {
            continue;
        }
        APASS_LOG_WARN_F(Elements::Tensor,
                         "Assemble output tensor[%d] may miss dynValidShape handling and cause precision issues. "
                         "Assemble opmagic(s): [%s], input tensor(s) with non-immediate dynValidShape: [%s].",
                         tensorMagic, JoinOpMagics(info.producers).c_str(),
                         JoinTensorMagics(info.inputsWithDynValidShape).c_str());
    }
}

} // namespace npu::tile_fwk
