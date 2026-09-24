/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_discontinuous_input_checker.cpp
 * \brief
 */

#include "infer_discontinuous_input_checker.h"
#include <queue>
#include <set>
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "InferDiscontinuousInputChecker"

namespace npu {
namespace tile_fwk {
std::unordered_set<Opcode> inplaceNodes{Opcode::OP_VIEW,     Opcode::OP_SLICE,   Opcode::OP_ASSEMBLE,
                                        Opcode::OP_CONTRACT, Opcode::OP_RESHAPE, Opcode::OP_INDEX_OUTCAST};

Status checkAssemble(const std::unordered_map<LogicalTensorPtr, int64_t>& tensorMap,
                     const std::unordered_map<LogicalTensorPtr, std::pair<Offset, Offset>>& offsetMap,
                     std::unordered_map<int64_t, int64_t>& rawTensorSize)
{
    std::unordered_map<int, Offset> rawMagicToRawOffset;
    for (auto [logicTensor, rawMagic] : tensorMap) {
        auto shape = logicTensor->GetShape();
        int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        rawTensorSize[rawMagic] -= shapeSize;
        size_t rawshapeSize = logicTensor->GetRawTensor()->GetRawShape().size();
        Offset rawOffset(rawshapeSize, 0);
        std::pair<Offset, Offset> p = offsetMap.at(logicTensor);
        for (size_t dim = 0; dim < rawshapeSize; dim++) {
            rawOffset[dim] = p.second[dim] - p.first[dim];
        }
        if (rawMagicToRawOffset.find(rawMagic) == rawMagicToRawOffset.end()) {
            rawMagicToRawOffset[rawMagic] = rawOffset;
        } else if (rawMagicToRawOffset[rawMagic] != rawOffset) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_SHAPE_MISMATCH, Elements::Operation,
                              "LogicTensor(%d) relative position to rawTensor(%ld) changed after the assemble op.",
                              logicTensor->GetMagic(), static_cast<long>(rawMagic));
            return FAILED;
        }
    }
    for (auto& [rawMagic, shape] : rawTensorSize) {
        if (shape != 0) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_SHAPE_MISMATCH, Elements::Tensor,
                              "RawTensor(%ld) is not fully covered.", static_cast<long>(rawMagic));
            return FAILED;
        }
    }
    return SUCCESS;
}

Status checkView(Operation* op)
{
    for (const auto& logicTensor : op->GetIOperands()) {
        auto producers = logicTensor->GetProducers();
        if (producers.size() != 1 || (*producers.begin())->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto shape = logicTensor->GetShape();
        if (std::any_of(shape.begin(), shape.end(), [](int64_t num) { return num < 0; })) {
            continue;
        }
        if (logicTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_ERROR_C(
                TensorErr::TENSOR_INVALID_MEMORY_TYPE, Elements::Tensor,
                "Tensor(%d) memory type is MEM_DEVICE_DDR, which is not supported for view-like->assemble-like case.",
                logicTensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status checkTensor(const LogicalTensorPtr& tensor)
{
    std::unordered_map<int64_t, int64_t> rawTensorSize;
    std::unordered_map<LogicalTensorPtr, int64_t> tensorMap;
    std::unordered_map<LogicalTensorPtr, std::pair<Offset, Offset>> offsetMap;
    bool allAssemble = true;
    for (auto producer : tensor->GetProducers()) {
        if (inplaceNodes.find(producer->GetOpcode()) == inplaceNodes.end()) {
            continue;
        }
        if (!IsAssembleLike(producer->GetOpcode())) {
            allAssemble = false;
            continue;
        }
        if (checkView(producer) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "CheckView Failed.");
            return FAILED;
        }
        std::shared_ptr<AssembleOpAttribute> attr = std::dynamic_pointer_cast<AssembleOpAttribute>(
            producer->GetOpAttribute());
        if (attr == nullptr) {
            APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                              "Assemble op %d do not have attribute. %s", producer->GetOpMagic(),
                              GetFormatBacktrace(producer).c_str());
            return FAILED;
        }
        LogicalTensorPtr inputTensor = *(producer->GetIOperands().begin());
        rawTensorSize[inputTensor->tensor->GetRawMagic()] = inputTensor->tensor->GetRawShapeSize();
        tensorMap[inputTensor] = inputTensor->GetRawTensor()->GetRawMagic();
        offsetMap[inputTensor] = std::make_pair(inputTensor->GetOffset(), attr->GetToOffset());
    }
    if (!allAssemble) {
        return SUCCESS;
    }
    if (checkAssemble(tensorMap, offsetMap, rawTensorSize) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckAssemble Failed.");
        return FAILED;
    }

    return SUCCESS;
}

Status InferDisContinuousInputChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for DisContinuousInput.");
    if (CheckGraphLoop(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Find loop");
        return FAILED;
    }

    auto tensorMap = GraphUtils::GetAllTensors(function);
    for (const auto& logicalTensor : tensorMap) {
        if (checkTensor(logicalTensor) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor(%d) CheckTensor Failed.", logicalTensor->GetMagic());
            return FAILED;
        }
    }

    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
