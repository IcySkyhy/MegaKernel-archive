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
 * \file infer_memory_conflict.cpp
 * \brief
 */

#include "infer_memory_conflict.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include <limits>
#include <unordered_set>

#define MODULE_NAME "InferMemoryConflict"

namespace npu {
namespace tile_fwk {
namespace {
// 常量定义
constexpr size_t MIN_DIMENSIONS = 2;
constexpr size_t MAX_DIMENSIONS = 5;
constexpr size_t DIMENSIONS_2D = 2;
constexpr size_t DIMENSIONS_3D = 3;
constexpr size_t DIMENSIONS_4D = 4;
constexpr size_t DIMENSIONS_5D = 5;

int64_t GetPowerOfTwo(int64_t cur)
{
    if (cur <= 1) {
        return 1;
    }
    int64_t ret = 1;
    while (ret < cur) {
        if (ret > (std::numeric_limits<int64_t>::max() >> 1)) {
            break;
        }
        ret <<= 1;
    }
    return ret;
}

bool CheckDynRawShape(const Shape& shape)
{
    return std::any_of(shape.begin(), shape.end(), [](int dim) { return dim < 0; });
}

bool HasNegativeDimAfterFirst(const LogicalTensorPtr& tensor)
{
    if (!tensor) {
        return false;
    }

    const auto& shape = tensor->GetShape();
    return shape.size() > 1 && std::any_of(shape.begin() + 1, shape.end(), [](int dim) { return dim < 0; });
}

// BMM场景只支持incast和outcast的最高轴为动轴，否则会插入registercopy
bool HasNegativeDimAfterFirstCast(const Function& function)
{
    for (const auto& incast : function.GetIncast()) {
        if (HasNegativeDimAfterFirst(incast)) {
            return true;
        }
    }
    for (const auto& outcast : function.GetOutcast()) {
        if (HasNegativeDimAfterFirst(outcast)) {
            return true;
        }
    }
    return false;
}

LogicalTensorPtr FindNegativeDimAfterFirstNzIncast(const Function& function)
{
    for (const auto& incast : function.GetIncast()) {
        if (HasNegativeDimAfterFirst(incast) && incast->Format() == TileOpFormat::TILEOP_NZ) {
            return incast;
        }
    }
    return nullptr;
}

bool HasOnlyViewConsumers(const LogicalTensorPtr& tensor)
{
    return std::all_of(tensor->GetConsumers().begin(), tensor->GetConsumers().end(), [](const auto& consumer) {
        return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_VIEW;
    });
}

bool HasMatmulConsumerWithSingleProducer(const LogicalTensorPtr& tensor)
{
    for (const auto& consumer : tensor->GetConsumers()) {
        if (consumer == nullptr) {
            continue;
        }
        if (OpcodeManager::Inst().GetOpCalcType(consumer->GetOpcode()) == OpCalcType::MATMUL) {
            auto matmulIn = consumer->GetIOperands().front();
            return matmulIn->GetProducers().size() == 1;
        }
    }
    return false;
}

// Assemble-like ops are traceable to support nested assemble/atomic merge chains.
bool IsPreMergeTraceableOpcode(Opcode opcode)
{
    return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_RESHAPE ||
           opcode == Opcode::OP_VIEW_TYPE || opcode == Opcode::OP_ATOMIC_RMW;
}

bool IsAssembleLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_ATOMIC_RMW; }

bool AddUniqueSource(std::vector<LogicalTensorPtr>& sourceTensors, const LogicalTensorPtr& sourceTensor)
{
    if (std::find(sourceTensors.begin(), sourceTensors.end(), sourceTensor) != sourceTensors.end()) {
        return false;
    }
    sourceTensors.emplace_back(sourceTensor);
    return true;
}

bool CollectPreMergeBranchSources(Function& function, Operation* op, std::unordered_set<Operation*>& visited,
                                  std::vector<LogicalTensorPtr>& sourceTensors)
{
    if (op == nullptr || !IsPreMergeTraceableOpcode(op->GetOpcode()) || !visited.insert(op).second) {
        return false;
    }
    bool found = false;
    for (const auto& input : op->GetIOperands()) {
        if (input == nullptr) {
            continue;
        }
        if (function.IsFromInCast(input)) {
            AddUniqueSource(sourceTensors, input);
            found = true;
            continue;
        }
        for (const auto& producer : input->GetProducers()) {
            found = CollectPreMergeBranchSources(function, producer, visited, sourceTensors) || found;
        }
    }
    return found;
}

bool CollectPreMergeCopyCandidates(Function& function, const LogicalTensorPtr& tensor,
                                   std::vector<Operation*>& assembleProducers,
                                   std::vector<std::pair<Operation*, LogicalTensorPtr>>& copyCandidates)
{
    if (tensor == nullptr) {
        return false;
    }
    std::unordered_set<Operation*> uniqueAssembleProducers;
    for (auto producer : tensor->GetProducers()) {
        if (producer != nullptr && IsAssembleLikeOpcode(producer->GetOpcode()) &&
            uniqueAssembleProducers.insert(producer).second) {
            assembleProducers.emplace_back(producer);
        }
    }
    if (assembleProducers.size() <= 1) {
        return false;
    }

    for (auto producer : assembleProducers) {
        std::unordered_set<Operation*> visited;
        std::vector<LogicalTensorPtr> sourceTensors;
        if (!CollectPreMergeBranchSources(function, producer, visited, sourceTensors)) {
            continue;
        }
        for (const auto& sourceTensor : sourceTensors) {
            copyCandidates.emplace_back(producer, sourceTensor);
        }
    }
    return !copyCandidates.empty();
}

bool AreReshapeShapesValid(const Shape& inputShape, const Shape& outputShape)
{
    const size_t inputDims = inputShape.size();
    const size_t outputDims = outputShape.size();
    if (inputDims < MIN_DIMENSIONS || outputDims < MIN_DIMENSIONS || inputDims > MAX_DIMENSIONS ||
        outputDims > MAX_DIMENSIONS) {
        return false;
    }

    return std::accumulate(inputShape.begin(), inputShape.end(), int64_t{1}, std::multiplies<int64_t>()) ==
           std::accumulate(outputShape.begin(), outputShape.end(), int64_t{1}, std::multiplies<int64_t>());
}

bool MatchReshapeDimensionPair(const Shape& inputShape, const Shape& outputShape)
{
    constexpr int kDimPairShiftBits = 4; // inputDims 占高 4 位, outputDims 占低 4 位
    const size_t inputDims = inputShape.size();
    const size_t outputDims = outputShape.size();
    const uint32_t dimensionPair = (inputDims << kDimPairShiftBits) | outputDims;

    switch (dimensionPair) {
        case (DIMENSIONS_4D << kDimPairShiftBits) | DIMENSIONS_2D:
            return inputShape[0] == 1 && inputShape[1] == 1 && inputShape[2] == outputShape[0] &&
                   inputShape[3] == outputShape[1];
        case (DIMENSIONS_2D << kDimPairShiftBits) | DIMENSIONS_4D:
            return outputShape[0] == 1 && outputShape[1] == 1 && inputShape[0] == outputShape[2] &&
                   inputShape[1] == outputShape[3];
        case (DIMENSIONS_3D << kDimPairShiftBits) | DIMENSIONS_2D:
            return inputShape[0] == 1 && inputShape[1] == outputShape[0] && inputShape[2] == outputShape[1];
        case (DIMENSIONS_2D << kDimPairShiftBits) | DIMENSIONS_3D:
            return outputShape[0] == 1 && inputShape[0] == outputShape[1] && inputShape[1] == outputShape[2];
        case (DIMENSIONS_4D << kDimPairShiftBits) | DIMENSIONS_3D:
            return inputShape[0] == 1 && inputShape[1] == outputShape[0] && inputShape[2] == outputShape[1] &&
                   inputShape[3] == outputShape[2];
        case (DIMENSIONS_3D << kDimPairShiftBits) | DIMENSIONS_4D:
            return outputShape[0] == 1 && inputShape[0] == outputShape[1] && inputShape[1] == outputShape[2] &&
                   inputShape[2] == outputShape[3];
        case (DIMENSIONS_5D << kDimPairShiftBits) | DIMENSIONS_3D:
            return inputShape[0] == 1 && inputShape[1] == 1 && inputShape[2] == outputShape[0] &&
                   inputShape[3] == outputShape[1] && inputShape[4] == outputShape[2];
        case (DIMENSIONS_3D << kDimPairShiftBits) | DIMENSIONS_5D:
            return outputShape[0] == 1 && outputShape[1] == 1 && inputShape[0] == outputShape[2] &&
                   inputShape[1] == outputShape[3] && inputShape[2] == outputShape[4];
        default:
            return false;
    }
}

bool AccumulateRawShapeSize(const Shape& shape, const char* shapeName, int64_t& rawSize)
{
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "%s[%zu] = %ld, dynamic shape should trigger conflict", shapeName, i,
                              static_cast<long>(shape[i]));
            return false;
        }
        rawSize *= shape[i];
    }
    return true;
}

} // namespace

Status InferMemoryConflict::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start InferMemoryConflict for function [%s].",
                     function.GetRawName().c_str());
    if (Init(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed.");
        return FAILED;
    }
    if (ForwardPropagation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ForwardPropagation failed.");
        return FAILED;
    }
    if (BackwardPropagation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BackwardPropagation failed.");
        return FAILED;
    }
    if (InsertCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertCopys failed.");
        return FAILED;
    }
    if (UpdateViewTypeTileShape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateViewTypeTileShape failed.");
        return FAILED;
    }
    // Mark outcast memory conflict for machine.
    if (FunctionUtils::InferOutcastWriteConflict(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InferOutcastWriteConflict failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "End InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status InferMemoryConflict::UpdateViewTypeTileShape(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW_TYPE) {
            continue;
        }
        auto output = op.GetOOperands()[0];
        auto outOp = *output->GetConsumers().begin();
        if (outOp == nullptr || outOp->GetOpcode() != Opcode::OP_REGISTER_COPY) {
            continue;
        }
        TileShape viewTypeTile;
        auto vecTypeTile = op.GetTileShape().GetVecTile();
        auto viewTypeIn = op.GetIOperands()[0];
        auto viewTypeOut = op.GetOOperands()[0];
        auto inType = viewTypeIn->tensor->datatype;
        auto outType = viewTypeOut->tensor->datatype;
        auto inEntry = viewTypeTable.find(inType);
        auto outEntry = viewTypeTable.find(outType);
        if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
            return FAILED;
        }
        if (inEntry->second < outEntry->second) {
            if (vecTypeTile.tile[vecTypeTile.tile.size() - 1] % (outEntry->second / inEntry->second) != 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "vecTypeTile tile dim n is not even.");
                return FAILED;
            }
            vecTypeTile.tile[vecTypeTile.tile.size() - 1] /= (outEntry->second / inEntry->second);
        } else {
            vecTypeTile.tile[vecTypeTile.tile.size() - 1] *= (inEntry->second / outEntry->second);
        }
        viewTypeTile.SetVecTile(vecTypeTile);
        outOp->UpdateTileShape(viewTypeTile);
    }
    return SUCCESS;
}

bool InferMemoryConflict::CheckConflict(const LogicalTensorPtr& inTensor, const LogicalTensorPtr& outTensor)
{
    if (inTensor->Symbol() == outTensor->Symbol()) {
        return false;
    }
    if (inTensor->GetRawTensor()->memoryId == outTensor->GetRawTensor()->memoryId) {
        return false;
    }
    return true;
}

bool InferMemoryConflict::CheckRawShapeConflict(const LogicalTensorPtr& inTensor, const LogicalTensorPtr& outTensor)
{
    int64_t inRawSize = 1;
    int64_t outRawSize = 1;
    Shape inShape = inTensor->GetRawTensor()->GetRawShape();
    Shape outShape = outTensor->GetRawTensor()->GetRawShape();
    auto inType = inTensor->tensor->datatype;
    auto outType = outTensor->tensor->datatype;
    auto inEntry = viewTypeTable.find(inType);
    auto outEntry = viewTypeTable.find(outType);
    auto consumerOp = *inTensor->GetConsumers().begin();
    if (consumerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
        if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
            return true;
        }
        if (inEntry->second > outEntry->second) {
            inRawSize *= (inEntry->second / outEntry->second);
        } else {
            outRawSize *= (outEntry->second / inEntry->second);
        }
    }
    if (!AccumulateRawShapeSize(inShape, "inShape", inRawSize) ||
        !AccumulateRawShapeSize(outShape, "outShape", outRawSize)) {
        return true;
    }
    if (inRawSize > 0 && outRawSize > 0 && inRawSize != outRawSize) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "The raw size of input is %ld bytes, the raw size of output is %ld bytes", inRawSize,
                          outRawSize);
        return true;
    }
    return false;
}

bool InferMemoryConflict::CheckTransmit(Operation& curOp)
{
    LogicalTensorPtr curTensor;
    std::set<Opcode> NonCalcNode = {Opcode::OP_VIEW,          Opcode::OP_ASSEMBLE,  Opcode::OP_RESHAPE,
                                    Opcode::OP_INDEX_OUTCAST, Opcode::OP_VIEW_TYPE, Opcode::OP_ATOMIC_RMW};
    bool transmit = (NonCalcNode.find(curOp.GetOpcode()) != NonCalcNode.end());
    if (curOp.GetOpcode() == Opcode::OP_ASSEMBLE || curOp.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        curTensor = *(curOp.GetIOperands().begin());
        for (const auto& producer : curTensor->GetProducers()) {
            if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                transmit = false;
            }
        }
    }
    return transmit;
}

bool InferMemoryConflict::IsValidTileShape(const Operation& op) const
{
    auto input = op.GetIOperands().front();
    VecTile tileSize = op.GetTileShape().GetVecTile();
    if (input->GetShape().size() != tileSize.size()) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "%s[%d] has unequal input shape dims size and tile shape dims, input shape: %s, tile size: %s. %s",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), input->DumpType().c_str(),
            op.GetTileShape().ToString(TileType::VEC).c_str(), GetFormatBacktrace(op).c_str());
        return false;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "The size info of %s[%d]: input shape: %s, tile size: %s",
                      op.GetOpcodeStr().c_str(), op.GetOpMagic(), input->DumpType().c_str(),
                      op.GetTileShape().ToString(TileType::VEC).c_str());
    return true;
}

/*
支持四种优化场景（不插入registercopy）
1. View -> Reshape -> MatMul (MatMul 的输入只有一个，且为当前处理的reshape)
2. MatMul -> Reshape -> Assemble (MatMul 的输出只有一个，且为当前处理的reshape)
3. View -> Reshape -> View(s) (reshape后面所有consumer都是View)
4. ADD/REDUCE_ACC -> Reshape -> Assemble (ADD/REDUCE_ACC的输出只有一个，且为当前处理的reshape)
*/
bool InferMemoryConflict::CheckReshapeContext(const LogicalTensorPtr& reshapeIn, const LogicalTensorPtr& reshapeOut)
{
    if (reshapeIn->GetProducers().empty() || reshapeOut->GetConsumers().empty()) {
        return false;
    }

    auto producer = *(reshapeIn->GetProducers().begin());
    if (producer == nullptr) {
        return false;
    }

    if (producer->GetOpcode() == Opcode::OP_VIEW) {
        return HasMatmulConsumerWithSingleProducer(reshapeOut) || HasOnlyViewConsumers(reshapeOut);
    }

    auto calcType = OpcodeManager::Inst().GetOpCalcType(producer->GetOpcode());
    if (calcType == OpCalcType::MATMUL || producer->GetOpcode() == Opcode::OP_ADD ||
        producer->GetOpcode() == Opcode::OP_REDUCE_ACC) {
        if (producer->GetOOperands().empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has no output operands.", producer->GetOpcodeStr().c_str(),
                              producer->GetOpMagic());
            return false;
        }
        auto producerOut = producer->GetOOperands().front();
        if (producerOut->GetConsumers().size() != 1) {
            return false;
        }
        return std::all_of(
            reshapeOut->GetConsumers().begin(), reshapeOut->GetConsumers().end(), [](const auto& consumer) {
                return consumer != nullptr &&
                       (consumer->GetOpcode() == Opcode::OP_ASSEMBLE || consumer->GetOpcode() == Opcode::OP_ATOMIC_RMW);
            });
    }

    return false;
}

// batch MatMul优化pattern，不插入register copy
Status InferMemoryConflict::MatchReshapePattern(Function& function, const LogicalTensorPtr& reshapeIn,
                                                const LogicalTensorPtr& reshapeOut, bool& matched)
{
    matched = false;
    auto nzIncast = FindNegativeDimAfterFirstNzIncast(function);
    if (nzIncast) {
        auto formatStr = std::to_string(nzIncast->Format());
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "The BMM input tensor does not support two or more dynamic axes when the tensor format is NZ. "
            "Please check the format of the BMM input tensor. tensor format: %s",
            formatStr.c_str());
        return FAILED;
    }
    if (!reshapeIn || !reshapeOut || HasNegativeDimAfterFirstCast(function)) {
        return SUCCESS;
    }
    Shape inRawShape = reshapeIn->GetRawTensor()->GetRawShape();
    Shape outRawShape = reshapeOut->GetRawTensor()->GetRawShape();
    if (CheckDynRawShape(inRawShape) || CheckDynRawShape(outRawShape) || !CheckReshapeContext(reshapeIn, reshapeOut)) {
        return SUCCESS;
    }

    const auto& inputShape = reshapeIn->GetShape();
    const auto& outputShape = reshapeOut->GetShape();
    if (!AreReshapeShapesValid(inputShape, outputShape)) {
        return SUCCESS;
    }

    matched = MatchReshapeDimensionPair(inputShape, outputShape);
    return SUCCESS;
}

Status InferMemoryConflict::UpdateForwardTensor(Function& function, const LogicalTensorPtr& curTensor,
                                                Operation* consumer, std::queue<LogicalTensorPtr>& curTensors)
{
    for (const auto& outputTensor : consumer->GetOOperands()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE) {
            auto reshapeInput = consumer->GetIOperands().front();
            bool isInplace = consumer->GetBoolAttribute(OP_ATTR_PREFIX + "isInplace");
            if (!isInplace && CheckRawShapeConflict(memoryInfo[curTensor], outputTensor)) {
                bool matched = false;
                if (MatchReshapePattern(function, reshapeInput, outputTensor, matched) != SUCCESS) {
                    return FAILED;
                }
                if (!matched) {
                    preregcopys.insert(consumer);
                    continue;
                }
            }
        }
        if (memoryInfo.find(outputTensor) != memoryInfo.end() && function.IsFromOutCast(memoryInfo[outputTensor])) {
            if (TryInsertPreMergeCopy(function, curTensor, memoryInfo[outputTensor], consumer)) {
                continue;
            }
            if (CheckConflict(memoryInfo[curTensor], memoryInfo[outputTensor])) {
                preregcopys.insert(consumer);
            }
        } else {
            if (consumer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                int index = 2;
                memoryInfo[outputTensor] = memoryInfo[consumer->GetInputOperand(index)];
            } else {
                memoryInfo[outputTensor] = memoryInfo[curTensor];
            }
            curTensors.push(outputTensor);
        }
    }
    return SUCCESS;
}

bool InferMemoryConflict::ShouldSkipOutcastInput(const LogicalTensorPtr& inputTensor, Function& function)
{
    if (!function.IsFromOutCast(inputTensor)) {
        return false;
    }
    return true;
}

Status InferMemoryConflict::HandleReshapeBackward(Function& function, const LogicalTensorPtr& curTensor,
                                                  const LogicalTensorPtr& inputTensor, Operation* producer,
                                                  const LogicalTensorPtr& reshapeOutput, bool& needSkip)
{
    needSkip = false;
    if (producer->GetOpcode() != Opcode::OP_RESHAPE) {
        return SUCCESS;
    }
    auto reshapeInput = producer->GetIOperands().front();
    bool isInplace = producer->GetBoolAttribute(OP_ATTR_PREFIX + "isInplace");
    if (isInplace || !CheckRawShapeConflict(inputTensor, memoryInfo[curTensor])) {
        return SUCCESS;
    }
    bool matched = false;
    if (MatchReshapePattern(function, reshapeInput, reshapeOutput, matched) != SUCCESS) {
        return FAILED;
    }
    if (!matched) {
        postregcopys.insert(producer);
        needSkip = true;
    }
    return SUCCESS;
}

Status InferMemoryConflict::HandleConflictBackward(Function& function, const LogicalTensorPtr& curTensor,
                                                   const LogicalTensorPtr& inputTensor, Operation* producer,
                                                   const LogicalTensorPtr& reshapeOutput)
{
    if (function.IsFromOutCast(inputTensor)) {
        return SUCCESS;
    }
    if (TryInsertPreMergeCopy(function, inputTensor, memoryInfo[curTensor], producer)) {
        return SUCCESS;
    }
    if (!CheckConflict(memoryInfo[curTensor], memoryInfo[inputTensor])) {
        return SUCCESS;
    }
    bool matched = false;
    if (producer->GetOpcode() == Opcode::OP_RESHAPE &&
        MatchReshapePattern(function, inputTensor, reshapeOutput, matched) != SUCCESS) {
        return FAILED;
    }
    if (producer->GetOpcode() == Opcode::OP_RESHAPE && !matched) {
        postregcopys.insert(producer);
    } else {
        preregcopys.insert(producer);
    }
    return SUCCESS;
}

bool InferMemoryConflict::TryInsertPreMergeCopy(Function& function, const LogicalTensorPtr& tensor,
                                                const LogicalTensorPtr& conflictTensor, Operation* finalOp)
{
    if (conflictTensor == nullptr || finalOp == nullptr || !IsAssembleLikeOpcode(finalOp->GetOpcode())) {
        return false;
    }
    std::vector<std::pair<Operation*, LogicalTensorPtr>> preMergeCopyCandidates;
    std::vector<Operation*> assembleProducers;
    if (!CollectPreMergeCopyCandidates(function, tensor, assembleProducers, preMergeCopyCandidates)) {
        return false;
    }
    std::vector<Operation*> copyTargets;
    for (const auto& candidate : preMergeCopyCandidates) {
        auto target = candidate.first;
        auto sourceTensor = candidate.second;
        if (!CheckConflict(sourceTensor, conflictTensor)) {
            continue;
        }
        if (std::find(copyTargets.begin(), copyTargets.end(), target) == copyTargets.end()) {
            copyTargets.emplace_back(target);
        }
    }
    // If every pre-merge branch conflicts, insert one copy before finalOp instead of copying every branch.
    if (copyTargets.empty() || copyTargets.size() == assembleProducers.size()) {
        return false;
    }
    for (auto target : copyTargets) {
        if (preregcopys.insert(target).second) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Insert register copy before pre-merge assemble-like op [%d] instead of final op [%d].",
                              target->GetOpMagic(), finalOp->GetOpMagic());
        }
    }
    return true;
}

Status InferMemoryConflict::UpdateBackwardTensor(Function& function, const LogicalTensorPtr& curTensor,
                                                 Operation* producer, std::queue<LogicalTensorPtr>& curTensors)
{
    for (auto& inputTensor : producer->GetIOperands()) {
        int index = 2;
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST && producer->GetIOperandIndex(inputTensor) != index) {
            continue;
        }
        if (ShouldSkipOutcastInput(inputTensor, function)) {
            continue;
        }
        auto reshapeOutput = producer->GetOOperands().front();
        bool needSkip = false;
        if (HandleReshapeBackward(function, curTensor, inputTensor, producer, reshapeOutput, needSkip) != SUCCESS) {
            return FAILED;
        }
        if (needSkip) {
            continue;
        }
        if (memoryInfo.find(inputTensor) != memoryInfo.end()) {
            if (HandleConflictBackward(function, curTensor, inputTensor, producer, reshapeOutput) != SUCCESS) {
                return FAILED;
            }
        } else {
            memoryInfo[inputTensor] = memoryInfo[curTensor];
            curTensors.push(inputTensor);
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::ForwardPropagation(Function& function)
{
    std::queue<LogicalTensorPtr> curTensors;
    for (const auto& incast : function.GetIncast()) {
        curTensors.push(incast);
    }
    while (!curTensors.empty()) {
        auto curTensor = curTensors.front();
        curTensors.pop();
        for (const auto& consumer : curTensor->GetConsumers()) {
            if (!CheckTransmit(*consumer)) {
                continue;
            }
            int index = 2;
            if (consumer->GetOpcode() == Opcode::OP_INDEX_OUTCAST && consumer->GetIOperandIndex(curTensor) != index) {
                continue;
            }
            if (UpdateForwardTensor(function, curTensor, consumer, curTensors) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "UpdateForwardTensor failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::BackwardPropagation(Function& function)
{
    std::queue<LogicalTensorPtr> curTensors;
    for (const auto& outcast : function.GetOutcast()) {
        curTensors.push(outcast);
    }
    while (!curTensors.empty()) {
        auto curTensor = curTensors.front();
        curTensors.pop();
        for (const auto& producer : curTensor->GetProducers()) {
            if (!CheckTransmit(*producer)) {
                continue;
            }
            if (UpdateBackwardTensor(function, curTensor, producer, curTensors) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "UpdateBackwardTensor failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::SetDefaultShape(const LogicalTensorPtr& tensor, std::vector<int64_t>& defaultTile)
{
    int64_t maximalTileSize = 16384;
    int64_t alignTailSize = 32;
    Shape shape = tensor->GetShape();
    size_t shapeDim = shape.size();
    int64_t curTile;
    defaultTile.clear();
    if (shapeDim == 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "SetDefaultShape failed because tensor shape is empty.");
        return FAILED;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        defaultTile.emplace_back(1);
    }
    int64_t lastDim = shape[shapeDim - 1];
    curTile = lastDim < alignTailSize ? alignTailSize : GetPowerOfTwo(lastDim);
    defaultTile[shapeDim - 1] = maximalTileSize < curTile ? maximalTileSize : curTile;
    for (size_t i = shapeDim - 1; i > 0; --i) {
        size_t dimIndex = i - 1;
        maximalTileSize /= defaultTile[dimIndex + 1];
        curTile = shape[dimIndex] <= 0 ? 1 : GetPowerOfTwo(shape[dimIndex]);
        defaultTile[dimIndex] = maximalTileSize < curTile ? maximalTileSize : curTile;
        defaultTile[dimIndex] = defaultTile[dimIndex] == 0 ? 1 : defaultTile[dimIndex];
    }
    return SUCCESS;
}

TileShape InferMemoryConflict::ObtainTileShape(const std::unordered_set<Operation*>& origOps)
{
    TileShape tile;
    if (origOps.empty()) {
        return tile;
    }
    TileShape base = (*origOps.begin())->GetTileShape();
    if (origOps.size() == 1) {
        return base;
    }
    for (const auto& origOp : origOps) {
        if (origOp->GetTileShape().GetVecTile().tile == base.GetVecTile().tile) {
            return tile;
        }
    }
    return base;
}

Status InferMemoryConflict::InferTileShape(Operation& op, const LogicalTensorPtr& tensor, TileShape parentTile,
                                           Shape& reshapeTile)
{
    auto tileShapeSize = parentTile.GetVecTile().size();
    auto tensorSize = tensor->GetShape().size();
    if (tileShapeSize == 0 || tileShapeSize != tensorSize) {
        APASS_LOG_WARN_F(Elements::Operation, "Inserted op[%d]'s producer/consumer op has no tile shape.",
                         op.GetOpMagic());
        TileShape tile;
        Shape defaultTile;
        if (!reshapeTile.empty() && reshapeTile.size() == tensorSize) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Derivate reshape tile shape.");
            defaultTile = reshapeTile;
        } else {
            if (SetDefaultShape(tensor, defaultTile) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SetDefaultShape failed.");
                return FAILED;
            }
        }
        tile.SetVecTile(defaultTile);
        op.UpdateTileShape(tile);
    } else {
        op.UpdateTileShape(parentTile);
    }
    if (!IsValidTileShape(op)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid tile size for %s[%d]. %s", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status InferMemoryConflict::ObtainReshapeTile(Operation& op, Shape& inTileShape, Shape& outTileShape)
{
    if (op.GetOpcode() == Opcode::OP_RESHAPE) {
        // 同时为空，证明对端不存在可用的tileshape
        if (inTileShape.empty() && outTileShape.empty()) {
            return SUCCESS;
        }
        if (!inTileShape.empty() && !op.GetOOperands()[0]->GetConsumers().empty()) {
            auto consumerOp = *op.GetOOperands()[0]->GetConsumers().begin();

            auto vec = consumerOp->GetTileShape().GetVecTile();
            outTileShape = vec.tile;
        }
    }
    return SUCCESS;
}

// 在OP_RESHAPE前面插入OP_REGISTER_COPY
Status InferMemoryConflict::InsertPrecededCopys(Function& function)
{
    for (const auto op : preregcopys) {
        LogicalTensorPtr inputTensor = op->GetIOperands().front();
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(inputTensor->Datatype(),
                                                                              inputTensor->GetShape());
        Offset newOffset(inputTensor->GetShape().size(), 0);
        LogicalTensorPtr newTensor = irBuilder_.CreateTensorVar(newRawTensor, newOffset, inputTensor->GetShape(),
                                                                inputTensor->GetDynValidShape());
        auto& copyOp = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_REGISTER_COPY, {inputTensor}, {newTensor});
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert copy op [%d]", copyOp.GetOpMagic());
        Shape reshapeTile;
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            TileShape vecTile = op->GetTileShape();
            if (vecTile.GetVecTile().size() > 0) {
                reshapeTile = vecTile.GetVecTile().tile;
            }
        }
        if (InferTileShape(copyOp, inputTensor, ObtainTileShape(copyOp.ProducerOps()), reshapeTile) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InferTileShape failed. %s", GetFormatBacktrace(copyOp).c_str());
            return FAILED;
        }
        inputTensor->RemoveConsumer(op);
        op->ReplaceInput(newTensor, inputTensor);
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertPostCopys(Function& function)
{
    for (const auto op : postregcopys) {
        LogicalTensorPtr outputTensor = op->GetOOperands().front();
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(outputTensor->Datatype(),
                                                                              outputTensor->GetShape());
        Offset newOffset(outputTensor->GetShape().size(), 0);
        LogicalTensorPtr newTensor = irBuilder_.CreateTensorVar(newRawTensor, newOffset, outputTensor->GetShape(),
                                                                outputTensor->GetDynValidShape());
        auto& copyOp = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_REGISTER_COPY, {newTensor}, {outputTensor});
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert copy op [%d]", copyOp.GetOpMagic());
        Shape reshapeTile;
        if (ObtainReshapeTile(*op, ObtainTileShape(op->ProducerOps()).GetVecTile().tile, reshapeTile) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ObtainReshapeTile failed. %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (InferTileShape(copyOp, outputTensor, ObtainTileShape(copyOp.ConsumerOps()), reshapeTile) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InferTileShape failed. %s", GetFormatBacktrace(copyOp).c_str());
            return FAILED;
        }
        outputTensor->RemoveConsumer(op);
        op->ReplaceOutput(newTensor, outputTensor);
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertCopys(Function& function)
{
    if (InsertPrecededCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertPrecededCopys failed.");
        return FAILED;
    }
    if (InsertPostCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertPostCopys failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status InferMemoryConflict::Init(Function& function)
{
    for (auto& incast : function.GetIncast()) {
        memoryInfo[incast] = incast;
    }
    for (auto& outcast : function.GetOutcast()) {
        memoryInfo[outcast] = outcast;
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
