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
 * \file pre_graph_checker.cpp
 * \brief
 */

#include "pre_graph_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "tilefwk/error_code.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "PreGraphProcess"

namespace npu {
namespace tile_fwk {
Status PreGraphProcessChecker::DoPreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PreCheck for PreGraph.");
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "Loopcheck failed before PreGraph.");
        return FAILED;
    }
    for (auto& op : function.Operations()) {
        // 校验是否切分
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            APASS_LOG_ERROR_C(GraphErr::GRAPH_SUBGRAPH_ID_INVALID, Elements::Graph, "%s[%d] is not partitioned. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if ((op.GetOpcode() != Opcode::OP_ASSEMBLE) && (op.GetOpcode() != Opcode::OP_VIEW) &&
            (op.GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        if ((op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1)) {
            // 校验非空单输入单输出
            APASS_LOG_ERROR_C(OperationErr::OP_INVALID_OPERAND_COUNT, Elements::Operation,
                              "Invalid %s[%d], input num: %zu, output num: %zu .%s", op.GetOpcodeStr().c_str(),
                              op.opmagic, op.GetIOperands().size(), op.GetOOperands().size(),
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands().front();
        auto tensorOut = op.GetOOperands().front();
        if ((tensorIn == nullptr) || (tensorOut == nullptr)) {
            // 校验输入输出非空
            APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                              "Invalid %s[%d], has nullptr input/output. %s", op.GetOpcodeStr().c_str(), op.opmagic,
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (tensorIn->GetMemoryTypeOriginal() != tensorOut->GetMemoryTypeOriginal()) {
            // 校验输入输出mem类型相同
            APASS_LOG_ERROR_C(TensorErr::TENSOR_INVALID_MEMORY_TYPE, Elements::Tensor,
                              "Unmatched input output memory type for %s[%d], input mem type: "
                              "%s, output mem type: %s",
                              op.GetOpcodeStr().c_str(), op.opmagic,
                              MemoryTypeToString(tensorIn->GetMemoryTypeOriginal()).c_str(),
                              MemoryTypeToString(tensorOut->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PreGraphProcessChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for PreGraph.");
    // 检测是否成环
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "Loopcheck failed after PreGraph.");
        return FAILED;
    }
    std::unordered_set<std::shared_ptr<LogicalTensor>> checkedTensors;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE && PostCheckReshape(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheckReshape failed. %s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        for (const std::shared_ptr<LogicalTensor>& inputTensor : op.GetIOperands()) {
            if (checkedTensors.count(inputTensor) > 0) {
                continue;
            }
            checkedTensors.insert(inputTensor);
            if (PostCheckHelpFunc(*inputTensor) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "PostCheckHelpFunc inputTensor failed.");
                return FAILED;
            }
        }
        for (const std::shared_ptr<LogicalTensor>& outputTensor : op.GetOOperands()) {
            if (checkedTensors.count(outputTensor) > 0) {
                continue;
            }
            checkedTensors.insert(outputTensor);
            if (PostCheckHelpFunc(*outputTensor) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "PostCheckHelpFunc outputTensor failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PreGraphProcessChecker::PostCheckHelpFunc(const LogicalTensor& singleTensor)
{
    if (singleTensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        SubgraphUtils::IsBoundary(singleTensor) == false) {
        // gm tensor 是否被标记为boundary
        APASS_LOG_WARN_F(Elements::Tensor,
                         "Tensor magic: %d, when memory type is DDR, this tensor should be subgraph boundary.",
                         singleTensor.GetMagic());
    }
    if (singleTensor.GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
        (singleTensor.Datatype() != DataType::DT_FP32 && singleTensor.Datatype() != DataType::DT_INT32)) {
        // L0C tensor 数据类型是否为FP32或INT32
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Tensor magic: %d, when memory type is L0C, this tensor should be fp32 or int32.",
                          singleTensor.GetMagic());
        return FAILED;
    }
    if (singleTensor.MemorySize() < 1 && !singleTensor.IsDummy()) {
        // 是否存在 dummy tensor
        APASS_LOG_INFO_F(Elements::Tensor,
                         "Tensor magic: %d, its memory size %zu bytes should be over than 0, but not.",
                         singleTensor.GetMagic(), singleTensor.MemorySize());
    }
    return SUCCESS;
}

Status PreGraphProcessChecker::PostCheckReshape(const Operation& op)
{
    auto reshapeIn = op.GetIOperands().front();
    auto reshapeOut = op.GetOOperands().front();

    if (reshapeIn->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Reshape on local buffer, opmagic: %d.", op.opmagic);
        auto opSubgraphId = op.GetSubgraphID();
        auto inputSubgraphId = CommonUtils::GetTensorSubgraphID(reshapeIn);
        auto outSubgraphId = CommonUtils::GetTensorSubgraphID(reshapeOut);
        if (opSubgraphId != inputSubgraphId || opSubgraphId != outSubgraphId) {
            // local buffer 上的reshape，输入/输出/op的子图编号相同
            APASS_LOG_ERROR_F(Elements::Operation,
                              "OP_RESHAPE[%d], op subGraphId: %d, input subGraphId: %d, output subGraphId: %d, %s",
                              op.GetOpMagic(), opSubgraphId, inputSubgraphId, outSubgraphId,
                              GetFormatBacktrace(op).c_str());
            return FAILED;
        }

        // Debug Print
        APASS_LOG_DEBUG_F(Elements::Operation, "Check done, input magic %d (raw %d), output magic %d (raw %d)",
                          reshapeIn->magic, reshapeIn->GetRawMagic(), reshapeOut->magic, reshapeOut->GetRawMagic());
        auto childOp = *(reshapeOut->GetConsumers().begin());
        APASS_LOG_DEBUG_F(Elements::Operation, "Child op: %s, opmagic: %d", childOp->GetOpcodeStr().c_str(),
                          childOp->opmagic);
        APASS_LOG_DEBUG_F(Elements::Operation, "Child op output magic %d (raw %d)", childOp->GetOOperands()[0]->magic,
                          childOp->GetOOperands()[0]->GetRawMagic());
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
