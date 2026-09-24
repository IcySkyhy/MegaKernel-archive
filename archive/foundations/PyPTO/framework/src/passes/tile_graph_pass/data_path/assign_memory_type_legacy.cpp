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
 * \file assign_memory_type.cpp
 * \brief
 */

#include "assign_memory_type_legacy.h"

#include <algorithm>
#include <map>
#include <set>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/checker_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/tile_graph_pass/data_path/memory_path_utils.h"
#include "tilefwk/tilefwk.h"

#define MODULE_NAME "AssignMemoryType"

#define RETURN_IF_NOT_SUCCESS(expr)                \
    do {                                           \
        Status assignMemoryReturnStatus = (expr);  \
        if (assignMemoryReturnStatus != SUCCESS) { \
            return assignMemoryReturnStatus;       \
        }                                          \
    } while (0)

namespace npu::tile_fwk::legacy {
namespace {
// 穿透时视为透明的 op 集合（shape 兜底/端点收集跳过它们继续穿透）
const std::unordered_set<Opcode> TRANSPARENT_OPS = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_REGISTER_COPY};
} // namespace
Status AssignMemoryType::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start AssignMemoryType.");
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    RETURN_IF_NOT_SUCCESS(AssignConfirmedMemoryTypes(function));
    RETURN_IF_NOT_SUCCESS(InferUncertainMemoryTypes(function));
    RETURN_IF_NOT_SUCCESS(ResolveMemoryUnknowns(function));
    RETURN_IF_NOT_SUCCESS(SyncViewAssembleMemoryAttrs(function));
    RETURN_IF_NOT_SUCCESS(InsertConvertOpsAndInferShape(function));
    RETURN_IF_NOT_SUCCESS(SyncTensorToBe(function));
    APASS_LOG_INFO_F(Elements::Function, "===> End AssignMemoryType.");
    return SUCCESS;
}

Status AssignMemoryType::AssignConfirmedMemoryTypes(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            RETURN_IF_NOT_SUCCESS(AssignViewAttrMemoryType(op));
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            RETURN_IF_NOT_SUCCESS(AssignAssembleAttrMemoryType(op));
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            RETURN_IF_NOT_SUCCESS(AssignReduceAccInputRequirements(op));
        }
        if (OpChecker::check(op, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
            RETURN_IF_NOT_SUCCESS(AssignMatmulInputRequirements(op));
        }
        RETURN_IF_NOT_SUCCESS(AssignOpcodeDefinedMemoryTypes(op));
    }
    RETURN_IF_NOT_SUCCESS(AssignInOutCastMemoryTypes(function));
    return EnsureAllConsumerRequirementsExist(function);
}

Status AssignMemoryType::AssignOpcodeDefinedMemoryTypes(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    bool hasSpecialInputRule = opcode == Opcode::OP_REDUCE_ACC ||
                               OpChecker::check(operation, OpChecker::CalcTypeChecker(OpCalcType::MATMUL));
    const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
    if (!hasSpecialInputRule) {
        for (size_t i = 0; i < operation.iOperand.size(); ++i) {
            MemoryType inputMemType = (i < inputsMemType.size()) ? inputsMemType[i] : MemoryType::MEM_UNKNOWN;
            RETURN_IF_NOT_SUCCESS(
                SetRequirementChecked(operation.iOperand[i], operation, inputMemType, "AssignOpcodeDefinedInput"));
        }
    }
    const auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(opcode);
    for (size_t i = 0; i < operation.oOperand.size(); ++i) {
        if (i >= outputsMemType.size())
            continue;
        RETURN_IF_NOT_SUCCESS(
            SetOriginalChecked(operation.oOperand[i], outputsMemType[i], "AssignOpcodeDefinedOutput"));
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignReduceAccInputRequirements(Operation& operation)
{
    for (auto& tensor : operation.iOperand) {
        RETURN_IF_NOT_SUCCESS(
            SetRequirementChecked(tensor, operation, MemoryType::MEM_DEVICE_DDR, "AssignReduceAccInputRequirements"));
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignMatmulInputRequirements(Operation& operation)
{
    for (auto& tensor : operation.iOperand) {
        for (const auto& producerOp : tensor->GetProducers()) {
            auto producerOpcode = producerOp->GetOpcode();
            MemoryType requirement = MemoryType::MEM_DEVICE_DDR;
            if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
                requirement = MemoryType::MEM_L0C;
            } else if (producerOpcode == Opcode::OP_VIEW) {
                auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
                if (viewOpAttribute == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation,
                                      "View attribute is null for %s[%d] while assigning matmul input.",
                                      producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
                    return FAILED;
                }
                requirement = viewOpAttribute->GetTo();
                if (requirement == MemoryType::MEM_UNKNOWN) {
                    requirement = MemoryType::MEM_DEVICE_DDR;
                }
            } else if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                        OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                        OpChecker::OutputMemTypeChecker(MemoryType::MEM_L0A))) {
                requirement = MemoryType::MEM_L0A;
            } else if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                        OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                        OpChecker::OutputMemTypeChecker(MemoryType::MEM_L0B))) {
                requirement = MemoryType::MEM_L0B;
            }
            RETURN_IF_NOT_SUCCESS(
                SetRequirementChecked(tensor, operation, requirement, "AssignMatmulInputRequirements"));
            if (requirement != MemoryType::MEM_DEVICE_DDR && requirement != MemoryType::MEM_UNKNOWN) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Infer %s[%d] input tensor[%d] as %s.",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), tensor->GetMagic(),
                                  BriefMemoryTypeToString(requirement).c_str());
            }
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignViewAttrMemoryType(Operation& operation)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "View attribute is null for %s[%d] while assigning view attr memory type.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType attrToType = viewOpAttribute->GetTo();
    if (attrToType == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    return SetOriginalChecked(operation.oOperand.front(), attrToType, "AssignViewAttrMemoryType");
}

Status AssignMemoryType::AssignAssembleAttrMemoryType(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assemble attribute is null for %s[%d] while assigning assemble attr memory type.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType attrFromType = assembleOpAttribute->GetFrom();
    if (attrFromType == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    return SetRequirementChecked(operation.iOperand.front(), operation, attrFromType, "AssignAssembleAttrMemoryType");
}

Status AssignMemoryType::AssignInOutCastMemoryTypes(Function& function)
{
    for (auto& incast : function.inCasts_) {
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(incast, MemoryType::MEM_DEVICE_DDR, "AssignIncastMemoryType", true));
    }

    for (auto& outcast : function.outCasts_) {
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(outcast, MemoryType::MEM_DEVICE_DDR, "AssignOutcastMemoryType", true));
    }
    return SUCCESS;
}

Status AssignMemoryType::EnsureAllConsumerRequirementsExist(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visited;
    auto ensureTensor = [this, &visited](const LogicalTensorPtr& tensor) -> Status {
        if (tensor == nullptr || !visited.insert(tensor).second)
            return SUCCESS;
        for (const auto& consumerOp : tensor->GetConsumers()) {
            if (inserter.HasRequirement(tensor, *consumerOp))
                continue;
            RETURN_IF_NOT_SUCCESS(SetRequirementChecked(tensor, *consumerOp, MemoryType::MEM_UNKNOWN,
                                                        "EnsureAllConsumerRequirementsExist"));
        }
        return SUCCESS;
    };
    for (auto& op : function.Operations()) {
        for (auto& input : op.iOperand) {
            RETURN_IF_NOT_SUCCESS(ensureTensor(input));
        }
        for (auto& output : op.oOperand) {
            RETURN_IF_NOT_SUCCESS(ensureTensor(output));
        }
    }
    return SUCCESS;
}

// l0c2ub pattern: batchmatmul case: cube op -> assemble(s) -> reshape op -> view(s)/assemble(s) -> vector
bool AssignMemoryType::IsReshapeCubeToVecL0C2UBPattern(Operation& op)
{
    if (op.GetOpcode() != npu::tile_fwk::Opcode::OP_RESHAPE) {
        return false;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();

    bool isL0C2UBPattern = true;
    if (producers.empty() || consumers.empty()) {
        return false;
    }

    for (auto& producer : producers) {
        bool isProducerAssemble = producer->GetOpcode() == npu::tile_fwk::Opcode::OP_ASSEMBLE;
        bool isProducerProducerAllCube = false;

        std::vector<bool> isProducerProducerCube;
        for (auto& producerIOperand : producer->iOperand) {
            for (auto& producerProducer : producerIOperand->GetProducers()) {
                isProducerProducerCube.push_back(producerProducer->GetCoreType() == CoreType::AIC);
            }
        }

        isProducerProducerAllCube = !isProducerProducerCube.empty() &&
                                    std::all_of(isProducerProducerCube.begin(), isProducerProducerCube.end(),
                                                [](bool val) { return val == true; });
        if (!isProducerAssemble || !isProducerProducerAllCube) {
            isL0C2UBPattern = false;
        }
    }
    for (auto& consumer : consumers) {
        bool isConsumerViewAssemble = consumer->GetOpcode() == npu::tile_fwk::Opcode::OP_VIEW ||
                                      consumer->GetOpcode() == npu::tile_fwk::Opcode::OP_ASSEMBLE;
        bool isConsumerConsumerAllVector = false;

        std::vector<bool> isConsumerConsumerVector;
        for (auto& consumerOOperand : consumer->oOperand) {
            for (auto& consumerConsumer : consumerOOperand->GetConsumers()) {
                isConsumerConsumerVector.push_back(consumerConsumer->GetCoreType() == CoreType::AIV);
            }
        }
        isConsumerConsumerAllVector = !isConsumerConsumerVector.empty() &&
                                      std::all_of(isConsumerConsumerVector.begin(), isConsumerConsumerVector.end(),
                                                  [](bool val) { return val == true; });
        if (!isConsumerViewAssemble || !isConsumerConsumerAllVector) {
            isL0C2UBPattern = false;
        }
    }
    return isL0C2UBPattern;
}

// ub2l1 pattern: 2 patterns
// 1. vector op -> view/assemble -> reshape op -> view from l1 -> view from l0a -> cube
// 2. vector op -> view/assemble -> view/assemble -> reshape op -> view from l1 -> view from l0a -> cube
bool AssignMemoryType::IsReshapeVecToCubeUB2L1Pattern(Operation& op)
{
    if (op.GetOpcode() != npu::tile_fwk::Opcode::OP_RESHAPE) {
        return false;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    if ((input == nullptr) || (output == nullptr)) {
        return false;
    }

    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    if (producers.empty() || consumers.empty()) {
        return false;
    }

    if (!IsReshapeVecToCubeUB2L1ProducerPattern(producers)) {
        return false;
    }

    if (!IsReshapeVecToCubeUB2L1ConsumerPattern(consumers)) {
        return false;
    }

    return true;
}

void AssignMemoryType::CollectProducerAIVFlags(Operation* op, std::vector<bool>& isProducerVector)
{
    for (auto& opInput : op->iOperand) {
        for (auto& producer : opInput->GetProducers()) {
            isProducerVector.push_back(producer->GetCoreType() == CoreType::AIV);
        }
    }
}

void AssignMemoryType::CollectConsumerAICFlags(Operation* op, std::vector<bool>& isConsumerCube)
{
    for (auto& opOutput : op->oOperand) {
        for (auto& consumer : opOutput->GetConsumers()) {
            isConsumerCube.push_back(consumer->GetCoreType() == CoreType::AIC);
        }
    }
}

bool AssignMemoryType::IsReshapeVecToCubeUB2L1ProducerPattern(
    const std::set<Operation*, LogicalTensor::CompareOp>& producers)
{
    for (auto& producer : producers) {
        bool isProducerDepth1ViewAssemble = producer->GetOpcode() == npu::tile_fwk::Opcode::OP_VIEW ||
                                            producer->GetOpcode() == npu::tile_fwk::Opcode::OP_ASSEMBLE;
        bool isProducerDepth2AllVector = false;
        bool isProducerDepth2AllViewAssemble = false;
        bool isProducerDepth3AllVector = false;

        std::vector<bool> isProducerDepth2Vector;
        std::vector<bool> isProducerDepth2ViewAssemble;
        std::vector<bool> isProducerDepth3Vector;
        for (auto& producerIOperand : producer->iOperand) {
            for (auto& producerProducer : producerIOperand->GetProducers()) {
                isProducerDepth2Vector.push_back(producerProducer->GetCoreType() == CoreType::AIV);
                isProducerDepth2ViewAssemble.push_back(
                    producerProducer->GetOpcode() == npu::tile_fwk::Opcode::OP_VIEW ||
                    producerProducer->GetOpcode() == npu::tile_fwk::Opcode::OP_ASSEMBLE);
                CollectProducerAIVFlags(producerProducer, isProducerDepth3Vector);
            }
        }
        isProducerDepth2AllVector = !isProducerDepth2Vector.empty() &&
                                    std::all_of(isProducerDepth2Vector.begin(), isProducerDepth2Vector.end(),
                                                [](bool val) { return val; });
        isProducerDepth2AllViewAssemble = !isProducerDepth2ViewAssemble.empty() &&
                                          std::all_of(isProducerDepth2ViewAssemble.begin(),
                                                      isProducerDepth2ViewAssemble.end(), [](bool val) { return val; });
        isProducerDepth3AllVector = !isProducerDepth3Vector.empty() &&
                                    std::all_of(isProducerDepth3Vector.begin(), isProducerDepth3Vector.end(),
                                                [](bool val) { return val; });
        // currently only support the following patterns:
        // 1. vector op -> view(s)/assemble(s) -> reshape op
        // 2. vector op -> view(s)/assemble(s) -> view(s)/assemble(s) -> reshape op
        if (!((isProducerDepth1ViewAssemble && isProducerDepth2AllViewAssemble && isProducerDepth3AllVector) ||
              (isProducerDepth1ViewAssemble && isProducerDepth2AllVector))) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::IsReshapeVecToCubeUB2L1ConsumerPattern(
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    for (auto& consumer : consumers) {
        bool isConsumerDepth1View = consumer->GetOpcode() == npu::tile_fwk::Opcode::OP_VIEW;
        bool isConsumerDepth2AllView = false;
        bool isConsumerDepth3AllCube = false;

        std::vector<bool> isConsumerDepth2View;
        std::vector<bool> isConsumerDepth3Cube;
        for (auto& consumerOOperand : consumer->oOperand) {
            for (auto& consumerConsumer : consumerOOperand->GetConsumers()) {
                isConsumerDepth2View.push_back(consumerConsumer->GetOpcode() == npu::tile_fwk::Opcode::OP_VIEW);
                CollectConsumerAICFlags(consumerConsumer, isConsumerDepth3Cube);
            }
        }
        isConsumerDepth2AllView = !isConsumerDepth2View.empty() &&
                                  std::all_of(isConsumerDepth2View.begin(), isConsumerDepth2View.end(),
                                              [](bool val) { return val; });
        isConsumerDepth3AllCube = !isConsumerDepth3Cube.empty() &&
                                  std::all_of(isConsumerDepth3Cube.begin(), isConsumerDepth3Cube.end(),
                                              [](bool val) { return val; });
        if (!isConsumerDepth1View || !isConsumerDepth2AllView || !isConsumerDepth3AllCube) {
            return false;
        }
    }
    return true;
}

Status AssignMemoryType::InferReshapeL0C2UBAndUB2L1PatternLiteNPU(Operation& op)
{
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        return SUCCESS;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();

    // l0c2ub pattern: batchmatmul case: cube op -> assemble(s) -> reshape op -> view(s)/assemble(s) -> vector
    if (IsReshapeCubeToVecL0C2UBPattern(op) && FitsTensorInUb(input)) {
        for (auto& producer : producers) {
            auto& producerInput = producer->iOperand.front();
            auto& producerOutput = producer->oOperand.front();

            // set producer assemble input to L0C
            producerInput->SetMemoryTypeOriginal(MemoryType::MEM_L0C, true);
            inserter.UpdateTensorTobeMap(producerInput, *producer, MemoryType::MEM_L0C);

            // set producer output to be UB
            producerOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(producerOutput, op, MemoryType::MEM_UB);
        }
        // set reshape output to UB
        output->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

        // set all consumer view/assembles input to UB
        for (auto& consumer : consumers) {
            inserter.UpdateTensorTobeMap(output, *consumer, MemoryType::MEM_UB);
        }
        return SUCCESS;
    }

    // ub2l1 pattern:
    // 1. vector op -> view(s)/assemble(s) -> reshape op -> view(s) from l1 -> view(s) from l0a -> cube
    // 2. vector op -> assemble(s) -> view(s) -> reshape op -> view(s) from l1 -> view(s) from l0a -> cube
    if (IsReshapeVecToCubeUB2L1Pattern(op) && FitsTensorInUb(output)) {
        for (auto& producer : producers) {
            auto& producerInput = producer->iOperand.front();
            auto& producerOutput = producer->oOperand.front();

            // set producer view/assemble input to UB
            producerInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(producerInput, *producer, MemoryType::MEM_UB);

            // set reshape input to UB
            producerOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(producerOutput, op, MemoryType::MEM_UB);
        }

        for (auto& consumer : consumers) {
            auto& consumerInput = consumer->iOperand.front();
            auto& consumerOutput = consumer->oOperand.front();

            // set reshape output to UB, set its tobe mem type to L1
            consumerInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(consumerInput, *consumer, MemoryType::MEM_L1);

            // set consumer view output to be L1
            consumerOutput->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
            for (auto& consumerConsumer : consumerOutput->GetConsumers()) {
                inserter.UpdateTensorTobeMap(consumerOutput, *consumerConsumer, MemoryType::MEM_L1);
            }
        }
        return SUCCESS;
    }

    return SUCCESS;
}

Status AssignMemoryType::InferUncertainMemoryTypes(Function& function)
{
    std::unordered_set<LogicalTensorPtr> inferredAssembleOutputs;
    for (auto& op : function.Operations()) {
        switch (op.GetOpcode()) {
            case Opcode::OP_VIEW:
                RETURN_IF_NOT_SUCCESS(InferViewMemoryType(op));
                break;
            case Opcode::OP_VIEW_TYPE:
                RETURN_IF_NOT_SUCCESS(InferViewTypeMemoryType(op));
                break;
            case Opcode::OP_ASSEMBLE:
                RETURN_IF_NOT_SUCCESS(InferAssembleMemoryType(function, op, inferredAssembleOutputs));
                break;
            case Opcode::OP_RESHAPE:
                RETURN_IF_NOT_SUCCESS(InferReshapeMemoryType(op));
                RETURN_IF_NOT_SUCCESS(InferReshapeL0C2UBAndUB2L1PatternLiteNPU(op));
                break;
            default:
                break;
        }
    }

    RETURN_IF_NOT_SUCCESS(ApplyOtherSpecialOpcodeRules(function));
    RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(function));
    return ApplyPlatformPathFallbackRules(function);
}

Status AssignMemoryType::GetFirstInputOutputIfOpcode(Operation& operation, Opcode expectedOpcode,
                                                     const std::string& action, LogicalTensorPtr& input,
                                                     LogicalTensorPtr& output, bool& shouldHandle) const
{
    shouldHandle = operation.GetOpcode() == expectedOpcode;
    if (!shouldHandle)
        return SUCCESS;
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s for %s[%d] failed because operand is empty.", action.c_str(),
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    input = operation.iOperand.front();
    output = operation.oOperand.front();
    if (input == nullptr || output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s for %s[%d] failed because operand tensor is null.", action.c_str(),
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status AssignMemoryType::InferViewMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_VIEW, "Infer OP_VIEW memory type", input,
                                                      output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_VIEW[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    RETURN_IF_NOT_SUCCESS(InferViewOutputFromRequirement(output, outputOriginal));
    // HasPermuteProducerAndTransDataDownstream 为 conv 算子问题的临时规避，正式方案落地后删除
    bool forceInputDdr = HasDynOffsetViewAndReshape(operation, output) ||
                         HasPermuteProducerAndTransDataDownstream(input, output);
    bool handled = TryHandleUnalignedView(operation, input, inputOriginal, outputOriginal);
    if (!handled && inputOriginal != MemoryType::MEM_UNKNOWN && outputOriginal != MemoryType::MEM_UNKNOWN) {
        RETURN_IF_NOT_SUCCESS(InferViewKnownInputOutput(operation, input, inputOriginal, outputOriginal));
        handled = true;
    }
    if (!handled && inputOriginal != MemoryType::MEM_UNKNOWN && outputOriginal == MemoryType::MEM_UNKNOWN) {
        RETURN_IF_NOT_SUCCESS(InferViewKnownInputUnknownOutput(operation, input, output, inputOriginal));
    }
    if (forceInputDdr) {
        ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "InferDynamicOffsetViewInputDdr");
    }
    return SUCCESS;
}

Status AssignMemoryType::InferViewOutputFromRequirement(const LogicalTensorPtr& output, MemoryType& outputOriginal)
{
    MemoryType uniqueOutputRequirement = output == nullptr ? MemoryType::MEM_UNKNOWN :
                                                             inserter.TryGetUniqueKnownRequiredType(output);
    if (outputOriginal != MemoryType::MEM_UNKNOWN || uniqueOutputRequirement == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, uniqueOutputRequirement, "InferViewOutputRequirement"));
    outputOriginal = output->GetMemoryTypeOriginal();
    return SUCCESS;
}

Status AssignMemoryType::InferViewKnownInputOutput(Operation& operation, const LogicalTensorPtr& input,
                                                   MemoryType inputOriginal, MemoryType outputOriginal)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_VIEW[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    viewOpAttribute->SetToType(outputOriginal);
    if (CanUseDirectViewPath(operation, inputOriginal, outputOriginal)) {
        ForceSetRequirement(input, operation, inputOriginal, "InferViewDirectPath");
        APASS_LOG_DEBUG_F(Elements::Operation, "Infer OP_VIEW[%d] direct %s for input tensor[%d].",
                          operation.GetOpMagic(), BriefMemoryTypeToString(inputOriginal).c_str(), input->GetMagic());
        return SUCCESS;
    }
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewFallbackDdr");
    return SUCCESS;
}

Status AssignMemoryType::InferViewKnownInputUnknownOutput(Operation& operation, const LogicalTensorPtr& input,
                                                          const LogicalTensorPtr& output, MemoryType inputOriginal)
{
    if (inputOriginal == MemoryType::MEM_L0C)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, inputOriginal, "InferViewReuseInputOriginal"));
    ForceSetRequirement(input, operation, inputOriginal, "InferViewReuseInputOriginal");
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_VIEW[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    viewOpAttribute->SetToType(inputOriginal);
    APASS_LOG_DEBUG_F(Elements::Operation, "Infer OP_VIEW[%d] reuse %s for output tensor[%d].", operation.GetOpMagic(),
                      BriefMemoryTypeToString(inputOriginal).c_str(), output->GetMagic());
    return SUCCESS;
}

bool AssignMemoryType::TryHandleUnalignedView(Operation& operation, const LogicalTensorPtr& input,
                                              MemoryType inputOriginal, MemoryType outputOriginal)
{
    if (inputOriginal == MemoryType::MEM_UNKNOWN) {
        return false;
    }
    // cube 数据加载路径不受 32 字节对齐约束
    if (outputOriginal == MemoryType::MEM_L0A || outputOriginal == MemoryType::MEM_L0B ||
        outputOriginal == MemoryType::MEM_L0AMX || outputOriginal == MemoryType::MEM_L0BMX) {
        return false;
    }
    if (IsViewFromOffsetAligned(operation)) {
        return false;
    }
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
        if (viewOpAttribute != nullptr) {
            viewOpAttribute->SetToType(outputOriginal);
        }
        ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewUnalignedOffset");
        return true;
    }
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewUnknownOutputUnaligned");
    return true;
}

bool AssignMemoryType::TryHandleSpecialDirectMemoryPath(Operation& operation, MemoryType from, MemoryType to,
                                                        bool& directPath)
{
    LogicalTensorPtr input = operation.iOperand.empty() ? nullptr : operation.iOperand.front();
    if (MemoryPathUtils::IsSpecialDirectMemoryPath(from, to) && HasParallelDifferentConsumerRequirement(input, to)) {
        directPath = false;
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Disable direct %s -> %s path for %s[%d] because source tensor has parallel different requirements.",
            BriefMemoryTypeToString(from).c_str(), BriefMemoryTypeToString(to).c_str(),
            operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return true;
    }
    if (from == MemoryType::MEM_L0C && to == MemoryType::MEM_L1) {
        directPath = inserter.FitL0C2L1(operation);
        return true;
    }
    bool isA5 = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if (isA5 && from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) {
        directPath = true;
        return true;
    }
    if (isA5 && from == MemoryType::MEM_UB && to == MemoryType::MEM_L1) {
        directPath = inserter.FitUB2L1(operation.iOperand.front());
        return true;
    }
    return false;
}

// 特殊进阶数据通路，不满足特定条件时回退到通过DDR搬运：L0C2L1, L0C2UB, UB2L1
bool AssignMemoryType::IsAdvancedMemoryPath(MemoryType from, MemoryType to) const
{
    if (from == MemoryType::MEM_L0C && to == MemoryType::MEM_L1) {
        return true;
    }
    bool isA5 = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    return isA5 && ((from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) ||
                    (from == MemoryType::MEM_UB && to == MemoryType::MEM_L1));
}

bool AssignMemoryType::HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor,
                                                               MemoryType targetType) const
{
    if (tensor == nullptr || tensor->GetConsumers().size() <= 1) {
        return false;
    }
    auto requirements = inserter.GetConsumerRequirements(tensor);
    return std::any_of(requirements.begin(), requirements.end(), [this, targetType](const auto& item) {
        auto resolveOutputRequirement = [this](const LogicalTensorPtr& output) {
            return InferUniqueRequirementThroughViewConsumers(output);
        };
        MemoryType requirement = MemoryPathUtils::ResolveEffectiveConsumerRequirement(
            item.first, item.second, targetType, resolveOutputRequirement);
        return MemoryPathUtils::IsDifferentKnownRequirement(requirement, targetType);
    });
}

bool AssignMemoryType::CanUseDirectViewPath(Operation& operation, MemoryType from, MemoryType to)
{
    if (from == MemoryType::MEM_UNKNOWN || to == MemoryType::MEM_UNKNOWN)
        return false;
    if (from == to)
        return true;
    if (from != MemoryType::MEM_DEVICE_DDR && to == MemoryType::MEM_DEVICE_DDR)
        return false;
    bool directPath = false;
    if (TryHandleSpecialDirectMemoryPath(operation, from, to, directPath))
        return directPath;
    std::vector<MemoryType> paths;
    bool pathFound = Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (!pathFound || paths.empty())
        return false;
    static constexpr size_t DIRECT_MEMORY_PATH_LENGTH = 2;
    bool isDirectPath = paths.size() == DIRECT_MEMORY_PATH_LENGTH && paths.front() == from && paths.back() == to;
    return isDirectPath;
}

bool AssignMemoryType::IsViewFromOffsetAligned(Operation& operation) const
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr || operation.iOperand.empty() || operation.oOperand.empty()) {
        return false;
    }
    auto fromOffset = viewOpAttribute->GetFromOffset();
    if (fromOffset.empty()) {
        return true;
    }
    auto input = operation.iOperand.front();
    if (input == nullptr || input->GetRawTensor() == nullptr) {
        return false;
    }
    int64_t lineOffset = CalcLineOffset(input->GetRawTensor()->rawshape, fromOffset);
    if (lineOffset == -1) {
        return true;
    }
    static constexpr int VIEW_ALIGN_BYTES = 32;
    auto output = operation.oOperand.front();
    return (BytesOf(output->Datatype()) * lineOffset) % VIEW_ALIGN_BYTES == 0;
}

bool AssignMemoryType::HasDynOffsetViewAndReshape(Operation& operation, const LogicalTensorPtr& output) const
{
    if (operation.GetOpcode() != Opcode::OP_VIEW) {
        return false;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr || viewOpAttribute->GetFromDynOffset().empty()) {
        return false;
    }
    const auto& fromOffset = viewOpAttribute->GetFromOffset();
    const auto& fromDynOffset = viewOpAttribute->GetFromDynOffset();
    bool hasDynamicOffset = false;
    if (fromOffset.size() != fromDynOffset.size()) {
        hasDynamicOffset = true;
    } else {
        for (size_t i = 0; i < fromDynOffset.size(); ++i) {
            if (!fromDynOffset[i].ConcreteValid() || fromDynOffset[i].Concrete() != fromOffset[i]) {
                hasDynamicOffset = true;
                break;
            }
        }
    }
    if (!hasDynamicOffset || output == nullptr) {
        return false;
    }
    for (const auto& consumerOp : output->GetConsumers()) {
        if (consumerOp != nullptr && consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

bool AssignMemoryType::HasTransDataConsumer(const LogicalTensorPtr& tensor) const
{
    for (const auto& consumerOp : tensor->GetConsumers()) {
        if (consumerOp != nullptr && consumerOp->HasAttr(OpAttributeKey::transDataOffset)) {
            return true;
        }
    }
    return false;
}

bool AssignMemoryType::HasPermuteProducerAndTransDataDownstream(const LogicalTensorPtr& input,
                                                                const LogicalTensorPtr& output) const
{
    bool hasPermuteProducer = false;
    for (const auto& producerOp : input->GetProducers()) {
        if (producerOp != nullptr && producerOp->GetOpcode() == Opcode::OP_PERMUTE) {
            hasPermuteProducer = true;
            break;
        }
    }
    if (!hasPermuteProducer) {
        return false;
    }
    if (HasTransDataConsumer(output)) {
        return true;
    }
    for (const auto& consumerOp : output->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_REGISTER_COPY ||
            consumerOp->oOperand.empty()) {
            continue;
        }
        const auto& registerCopyOutput = consumerOp->oOperand.front();
        if (registerCopyOutput == nullptr) {
            continue;
        }
        for (const auto& assembleOp : registerCopyOutput->GetConsumers()) {
            if (assembleOp == nullptr || assembleOp->GetOpcode() != Opcode::OP_ASSEMBLE ||
                assembleOp->oOperand.empty()) {
                continue;
            }
            if (HasTransDataConsumer(assembleOp->oOperand.front())) {
                return true;
            }
        }
    }
    return false;
}

Status AssignMemoryType::AssignAssembleToOutCastRequirement(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Handle OP_ASSEMBLE[%d] to outcast failed because operand is empty.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.front();
    if (input == nullptr || output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Handle OP_ASSEMBLE[%d] to outcast failed because operand tensor is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    ForceSetRequirement(input, operation, inputOriginal, "AssignAssembleToOutCastRequirement");
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "AssignAssembleToOutCastRequirement");
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Handle OP_ASSEMBLE[%d] to outcast failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (inputOriginal != MemoryType::MEM_UNKNOWN) {
        assembleOpAttribute->SetFromType(inputOriginal);
    }
    return SUCCESS;
}

Status AssignMemoryType::InferAssembleMemoryType(Function& function, Operation& operation,
                                                 std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    if (!operation.oOperand.empty() && std::find(function.outCasts_.begin(), function.outCasts_.end(),
                                                 operation.oOperand.front()) != function.outCasts_.end()) {
        return AssignAssembleToOutCastRequirement(operation);
    }
    if (!operation.oOperand.empty() && operation.oOperand.front() != nullptr &&
        !inferredAssembleOutputs.insert(operation.oOperand.front()).second) {
        return SUCCESS;
    }
    return InferAssembleMemoryType(operation);
}

Status AssignMemoryType::InferAssembleMemoryType(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] memory type failed because operand is empty.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute()) == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_ASSEMBLE[%d] memory type failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    return InferAssembleOutputMemoryType(operation.oOperand.front());
}

Status AssignMemoryType::InferAssembleOutputMemoryType(const LogicalTensorPtr& output)
{
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Infer assemble output memory type failed because output tensor is null.");
        return FAILED;
    }
    if (HasAssembleInputOutputElementCountMismatch(output)) {
        RETURN_IF_NOT_SUCCESS(ApplyAssembleDdrOutputWithInputOriginals(output, "InferAssembleElementCountMismatch",
                                                                       "InferAssembleElementCountMismatchFillInput"));
        return SUCCESS;
    }
    MemoryType tempOriginal = InferAssembleTempOriginal(output);
    bool handled = false;
    RETURN_IF_NOT_SUCCESS(TryInferAssembleOutputByTempOriginal(output, tempOriginal, handled));
    if (handled)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(TryInferAssembleOutputByProducerCandidate(output, handled));
    if (handled)
        return SUCCESS;
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferAssembleUnknownFallbackDdr");
    return SUCCESS;
}

bool AssignMemoryType::HasAssembleInputOutputElementCountMismatch(const LogicalTensorPtr& output) const
{
    if (output == nullptr)
        return false;
    int64_t assembleInputElements = 0;
    bool hasAssembleProducer = false;
    for (const auto& producerOp : output->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_ASSEMBLE || producerOp->iOperand.empty() ||
            producerOp->iOperand.front() == nullptr) {
            continue;
        }
        hasAssembleProducer = true;
        assembleInputElements += CommonUtils::Numel(producerOp->iOperand.front()->GetShape());
    }
    if (!hasAssembleProducer)
        return false;
    int64_t assembleOutputElements = CommonUtils::Numel(output->GetShape());
    if (assembleInputElements == assembleOutputElements)
        return false;
    return true;
}

Status AssignMemoryType::TryInferAssembleOutputByTempOriginal(const LogicalTensorPtr& output, MemoryType tempOriginal,
                                                              bool& handled)
{
    handled = tempOriginal != MemoryType::MEM_UNKNOWN;
    if (!handled)
        return SUCCESS;
    if (tempOriginal == MemoryType::MEM_DEVICE_DDR) {
        RETURN_IF_NOT_SUCCESS(
            ApplyAssembleDdrOutputWithInputOriginals(output, "InferAssembleTempDdr", "InferAssembleDdrFillInput"));
        return SUCCESS;
    }
    if (AreAssembleDirectPathsSupported(output, tempOriginal)) {
        RETURN_IF_NOT_SUCCESS(ApplyAssembleDirectOutputOriginal(output, tempOriginal));
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(ApplyAssembleDdrOutputWithInputOriginals(output, "InferAssembleUnsupportedPath",
                                                                   "InferAssembleFallbackFillInput"));
    return SUCCESS;
}

bool AssignMemoryType::AreAssembleDirectPathsSupported(const LogicalTensorPtr& output, MemoryType targetOriginal)
{
    for (auto& producerOp : output->GetProducers()) {
        if (!IsAssembleProducer(producerOp))
            continue;
        auto input = producerOp->iOperand.front();
        if (input == nullptr)
            return false;
        MemoryType fromType = GetAssembleInputType(*producerOp);
        if (fromType == MemoryType::MEM_UNKNOWN)
            return false;
        bool checkOffsetAlignment = !IsAdvancedMemoryPath(fromType, targetOriginal);
        if ((checkOffsetAlignment && !IsAssembleToOffsetAligned(*producerOp, output)) ||
            !CanUseDirectAssemblePath(*producerOp, fromType, targetOriginal)) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::IsAssembleProducer(Operation* operation) const
{
    return operation != nullptr && operation->GetOpcode() == Opcode::OP_ASSEMBLE && !operation->iOperand.empty();
}

MemoryType AssignMemoryType::GetAssembleInputType(Operation& operation) const
{
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr)
        return MemoryType::MEM_UNKNOWN;
    auto input = operation.iOperand.front();
    MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
    return fromType != MemoryType::MEM_UNKNOWN ? fromType : input->GetMemoryTypeOriginal();
}

Status AssignMemoryType::ApplyAssembleDirectOutputOriginal(const LogicalTensorPtr& output, MemoryType targetOriginal)
{
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, targetOriginal, "InferAssembleTempOriginal"));
    for (auto& producerOp : output->GetProducers()) {
        if (!IsAssembleProducer(producerOp))
            continue;
        RETURN_IF_NOT_SUCCESS(SyncAssembleInputRequirementAndAttr(*producerOp, MemoryType::MEM_UNKNOWN,
                                                                  "InferAssembleFillInputRequirement"));
    }
    return SUCCESS;
}

Status AssignMemoryType::SyncAssembleInputRequirementAndAttr(Operation& operation, MemoryType fallbackType,
                                                             const std::string& reason)
{
    auto input = operation.iOperand.front();
    if (input == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because input tensor is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
    MemoryType resolvedFallback = fallbackType == MemoryType::MEM_UNKNOWN ? input->GetMemoryTypeOriginal() :
                                                                            fallbackType;
    if (fromType == MemoryType::MEM_UNKNOWN && resolvedFallback != MemoryType::MEM_UNKNOWN) {
        fromType = resolvedFallback;
        ForceSetRequirement(input, operation, fromType, reason);
    }
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (fromType != MemoryType::MEM_UNKNOWN) {
        assembleOpAttribute->SetFromType(fromType);
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyAssembleDdrOutputWithInputOriginals(const LogicalTensorPtr& output,
                                                                  const std::string& originalReason,
                                                                  const std::string& inputReason)
{
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, originalReason);
    return FillAssembleInputRequirementsFromOriginal(output, inputReason);
}

Status AssignMemoryType::FillAssembleInputRequirementsFromOriginal(const LogicalTensorPtr& output,
                                                                   const std::string& reason)
{
    for (auto& producerOp : output->GetProducers()) {
        if (!IsAssembleProducer(producerOp))
            continue;
        auto input = producerOp->iOperand.front();
        if (input != nullptr && inserter.GetRequirementOrUnknown(input, *producerOp) == MemoryType::MEM_UNKNOWN &&
            input->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
            ForceSetRequirement(input, *producerOp, input->GetMemoryTypeOriginal(), reason);
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::TryInferAssembleOutputByProducerCandidate(const LogicalTensorPtr& output, bool& handled)
{
    bool hasConflict = false;
    MemoryType producerCandidate = InferAssembleProducerCandidate(output, hasConflict);
    handled = !hasConflict && producerCandidate != MemoryType::MEM_UNKNOWN &&
              FitsAssembleOutputMemoryLimit(output, producerCandidate) &&
              AreAssembleDirectPathsSupported(output, producerCandidate);
    if (!handled)
        return SUCCESS;
    return ApplyAssembleProducerCandidate(output, producerCandidate);
}

MemoryType AssignMemoryType::InferAssembleProducerCandidate(const LogicalTensorPtr& output, bool& hasConflict) const
{
    MemoryType producerCandidate = MemoryType::MEM_UNKNOWN;
    hasConflict = false;
    for (auto& producerOp : output->GetProducers()) {
        if (!IsAssembleProducer(producerOp))
            continue;
        MemoryType fromType = GetAssembleInputType(*producerOp);
        if (fromType == MemoryType::MEM_UNKNOWN) {
            hasConflict = true;
            break;
        }
        if (producerCandidate == MemoryType::MEM_UNKNOWN) {
            producerCandidate = fromType;
        } else if (producerCandidate != fromType) {
            hasConflict = true;
            break;
        }
    }
    return hasConflict ? MemoryType::MEM_UNKNOWN : producerCandidate;
}

Status AssignMemoryType::ApplyAssembleProducerCandidate(const LogicalTensorPtr& output, MemoryType producerCandidate)
{
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, producerCandidate, "InferAssembleProducerCandidate"));
    for (auto& producerOp : output->GetProducers()) {
        if (!IsAssembleProducer(producerOp))
            continue;
        RETURN_IF_NOT_SUCCESS(
            SyncAssembleInputRequirementAndAttr(*producerOp, producerCandidate, "InferAssembleProducerCandidate"));
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Infer assemble output tensor[%d] original as %s by producer candidate.",
                      output->GetMagic(), BriefMemoryTypeToString(producerCandidate).c_str());
    return SUCCESS;
}

MemoryType AssignMemoryType::InferAssembleTempOriginal(const LogicalTensorPtr& output) const
{
    if (output == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    MemoryType tempOriginal = MemoryType::MEM_UNKNOWN;
    bool hasL1ViewTarget = false;
    bool hasUnknownViewTarget = false;
    auto requirements = inserter.GetConsumerRequirements(output);
    for (const auto& item : requirements) {
        auto consumerOp = item.first;
        MemoryType candidate = item.second;
        if (consumerOp != nullptr && consumerOp->GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
            if (viewOpAttribute == nullptr || viewOpAttribute->GetTo() == MemoryType::MEM_UNKNOWN) {
                hasUnknownViewTarget = true;
            } else {
                hasL1ViewTarget = hasL1ViewTarget || viewOpAttribute->GetTo() == MemoryType::MEM_L1;
                if (candidate == MemoryType::MEM_UNKNOWN) {
                    candidate = viewOpAttribute->GetTo();
                }
            }
        }
        if (candidate == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        if (tempOriginal == MemoryType::MEM_UNKNOWN) {
            tempOriginal = candidate;
        } else if (tempOriginal != candidate) {
            return MemoryType::MEM_DEVICE_DDR;
        }
    }
    if (tempOriginal == MemoryType::MEM_L1 && hasL1ViewTarget && hasUnknownViewTarget) {
        return MemoryType::MEM_UNKNOWN;
    }
    return tempOriginal;
}

bool AssignMemoryType::CanUseDirectAssemblePath(Operation& operation, MemoryType from, MemoryType to)
{
    if (from == MemoryType::MEM_UNKNOWN || to == MemoryType::MEM_UNKNOWN) {
        return false;
    }
    if (from == to) {
        return true;
    }
    bool directPath = false;
    if (TryHandleSpecialDirectMemoryPath(operation, from, to, directPath)) {
        return directPath;
    }
    std::vector<MemoryType> paths;
    Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (paths.empty()) {
        return false;
    }
    bool hasDdr = std::find(paths.begin(), paths.end(), MemoryType::MEM_DEVICE_DDR) != paths.end();
    return !hasDdr;
}

bool AssignMemoryType::IsAssembleToOffsetAligned(Operation& operation, const LogicalTensorPtr& output)
{
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr || output == nullptr) {
        return false;
    }
    int64_t lineOffset = CalcLineOffset(output->GetRawTensor()->rawshape, assembleOpAttribute->GetToOffset());
    if (lineOffset == -1) {
        return true;
    }
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(output->Datatype()));
    return (tensorBytes * lineOffset) % ASSEMBLE_ALIGN_BYTES == 0;
}

bool AssignMemoryType::FitsAssembleOutputMemoryLimit(const LogicalTensorPtr& output, MemoryType memoryType) const
{
    if (output == nullptr) {
        return false;
    }
    if (memoryType == MemoryType::MEM_UB) {
        const size_t ubThreshold = static_cast<size_t>(
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_ASSEMBLE);
        return static_cast<size_t>(output->GetDataSize()) <= ubThreshold;
    }
    if (memoryType == MemoryType::MEM_L1) {
        const size_t l1Threshold = static_cast<size_t>(
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) * L1_THRESHOLD);
        return static_cast<size_t>(output->GetDataSize()) <= l1Threshold;
    }
    return true;
}

Status AssignMemoryType::InferReshapeMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_RESHAPE, "Infer OP_RESHAPE memory type",
                                                      input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    MemoryType inputRequirement = GetReshapeInputRequirement(operation, input, inputOriginal);
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    RETURN_IF_NOT_SUCCESS(InferReshapeOutputFromRequirement(output, outputOriginal));
    if (KeepSplitReshapeUb(operation, input, output)) {
        return SUCCESS;
    }
    bool isDynamic = IsDynamicReshape(operation, output);
    bool canUseUb = CanUseUbForReshape(input, output, inputRequirement, outputOriginal);
    return ApplyReshapeMemoryType(operation, input, output, isDynamic, canUseUb);
}

MemoryType AssignMemoryType::GetReshapeInputRequirement(Operation& operation, const LogicalTensorPtr& input,
                                                        MemoryType inputOriginal)
{
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    if (inputRequirement != MemoryType::MEM_UNKNOWN || inputOriginal == MemoryType::MEM_UNKNOWN) {
        return inputRequirement;
    }
    ForceSetRequirement(input, operation, inputOriginal, "InferReshapeInputOriginal");
    return inputOriginal;
}

Status AssignMemoryType::InferReshapeOutputFromRequirement(const LogicalTensorPtr& output, MemoryType& outputOriginal)
{
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryType outputRequirement = InferUniqueRequirementThroughViewConsumers(output);
    if (outputRequirement == MemoryType::MEM_UNKNOWN) {
        std::unordered_set<const LogicalTensor*> visitedTensors;
        if (HasRequirementThroughViewConsumers(output, MemoryType::MEM_UB, visitedTensors)) {
            outputRequirement = MemoryType::MEM_UB;
        }
    }
    if (outputRequirement == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, outputRequirement, "InferReshapeOutputRequirement"));
    outputOriginal = output->GetMemoryTypeOriginal();
    return SUCCESS;
}

MemoryType AssignMemoryType::InferUniqueRequirementThroughViewConsumers(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    return InferUniqueRequirementThroughViewConsumers(tensor, visitedTensors);
}

MemoryType AssignMemoryType::InferUniqueRequirementThroughViewConsumers(
    const LogicalTensorPtr& tensor, std::unordered_set<const LogicalTensor*>& visitedTensors) const
{
    if (tensor == nullptr || !visitedTensors.insert(tensor.get()).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    auto addCandidate = [&candidates](MemoryType candidate) {
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    };
    auto consumerRequirements = inserter.GetConsumerRequirements(tensor);
    for (const auto& item : consumerRequirements) {
        Operation* consumerOp = item.first;
        addCandidate(item.second);
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewOpAttribute != nullptr) {
            addCandidate(viewOpAttribute->GetTo());
        }
        if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
            continue;
        }
        auto viewOutput = consumerOp->oOperand.front();
        addCandidate(viewOutput->GetMemoryTypeOriginal());
        addCandidate(InferUniqueRequirementThroughViewConsumers(viewOutput, visitedTensors));
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
}

bool AssignMemoryType::HasRequirementThroughViewConsumers(
    const LogicalTensorPtr& tensor, MemoryType targetRequirement,
    std::unordered_set<const LogicalTensor*>& visitedTensors) const
{
    if (tensor == nullptr || targetRequirement == MemoryType::MEM_UNKNOWN ||
        !visitedTensors.insert(tensor.get()).second) {
        return false;
    }
    auto consumerRequirements = inserter.GetConsumerRequirements(tensor);
    for (const auto& item : consumerRequirements) {
        Operation* consumerOp = item.first;
        if (item.second == targetRequirement) {
            return true;
        }
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewOpAttribute != nullptr && viewOpAttribute->GetTo() == targetRequirement) {
            return true;
        }
        if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
            continue;
        }
        auto viewOutput = consumerOp->oOperand.front();
        if (viewOutput->GetMemoryTypeOriginal() == targetRequirement ||
            HasRequirementThroughViewConsumers(viewOutput, targetRequirement, visitedTensors)) {
            return true;
        }
    }
    return false;
}

bool AssignMemoryType::CanUseUbForReshape(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                          MemoryType inputRequirement, MemoryType outputOriginal) const
{
    if (inputRequirement != outputOriginal) {
        return false;
    }
    return inputRequirement == MemoryType::MEM_UB && FitsTensorInUb(input) && FitsTensorInUb(output);
}

Status AssignMemoryType::ApplyReshapeMemoryType(Operation& operation, const LogicalTensorPtr& input,
                                                const LogicalTensorPtr& output, bool isDynamic, bool canUseUb)
{
    if (canUseUb) {
        const char* reason = isDynamic ? "InferDynamicReshapeUb" : "InferStaticReshapeUb";
        ForceSetRequirement(input, operation, MemoryType::MEM_UB, reason);
        ForceSetOriginal(output, MemoryType::MEM_UB, reason);
        return SUCCESS;
    }
    const char* reason = isDynamic ? "InferDynamicReshapeFallbackDdr" : "InferStaticReshapeFallbackDdr";
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, reason);
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, reason);
    return SUCCESS;
}

Status AssignMemoryType::InferViewTypeMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_VIEW_TYPE, "Infer OP_VIEW_TYPE memory type",
                                                      input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    MemoryType outputRequirement = output == nullptr ? MemoryType::MEM_UNKNOWN :
                                                       inserter.TryGetUniqueKnownRequiredType(output);
    // 输出 toBeMap 未知时，沿后续未推导的视图链向前查找有效内存类型
    if (outputRequirement == MemoryType::MEM_UNKNOWN) {
        MemoryType forwarded = InferTargetTypeThroughForwardViews(output);
        if (forwarded != MemoryType::MEM_UNKNOWN) {
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Infer OP_VIEW_TYPE[%d] memory type reused from forward view requirement %s for output tensor[%d].",
                operation.GetOpMagic(), BriefMemoryTypeToString(forwarded).c_str(), output->GetMagic());
            outputRequirement = forwarded;
        }
    }
    MemoryType targetType = outputRequirement != MemoryType::MEM_UNKNOWN ? outputRequirement : outputOriginal;
    bool handled = false;
    RETURN_IF_NOT_SUCCESS(TryInferViewTypeFromProducerView(operation, input, output, targetType, handled));
    if (handled) {
        return SUCCESS;
    }
    return InferViewTypeInput(operation, input, output, targetType);
}

Status AssignMemoryType::TryInferViewTypeFromProducerView(Operation& operation, const LogicalTensorPtr& input,
                                                          const LogicalTensorPtr& output, MemoryType targetType,
                                                          bool& handled)
{
    handled = false;
    auto& producers = input->GetProducers();
    if (producers.empty()) {
        return SUCCESS;
    }
    auto producer = *producers.begin();
    if (producer == nullptr || producer->GetOpcode() != Opcode::OP_VIEW) {
        return SUCCESS;
    }
    handled = true;
    if (producer->iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_VIEW_TYPE[%d] memory type failed because producer OP_VIEW[%d] input is empty.",
                          operation.GetOpMagic(), producer->GetOpMagic());
        return FAILED;
    }
    auto viewInput = producer->iOperand.front();
    MemoryType viewInputRequirement = inserter.GetRequirementOrUnknown(viewInput, *producer);
    if (viewInputRequirement == MemoryType::MEM_UNKNOWN) {
        viewInputRequirement = viewInput->GetMemoryTypeOriginal();
    }
    if (targetType != MemoryType::MEM_UNKNOWN && CanUseDirectViewPath(*producer, viewInputRequirement, targetType)) {
        ForceSetOriginal(input, targetType, "InferViewTypeProducerView");
        ForceSetRequirement(input, operation, targetType, "InferViewTypeProducerView");
        ForceSetOriginal(output, targetType, "InferViewTypeProducerView");
        return SUCCESS;
    }
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewTypeProducerViewFallback");
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferViewTypeProducerViewFallback");
    return SUCCESS;
}

Status AssignMemoryType::InferViewTypeInput(Operation& operation, const LogicalTensorPtr& input,
                                            const LogicalTensorPtr& output, MemoryType targetType)
{
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    if (targetType != MemoryType::MEM_UNKNOWN && inputOriginal == targetType) {
        ForceSetRequirement(input, operation, targetType, "InferViewTypeSameMemory");
        ForceSetOriginal(output, targetType, "InferViewTypeSameMemory");
        return SUCCESS;
    }
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewTypeFallbackDdr");
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferViewTypeFallbackDdr");
    return SUCCESS;
}

MemoryType AssignMemoryType::InferTargetTypeThroughForwardViews(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<LogicalTensorPtr> visitedTensors;
    return InferTargetTypeThroughForwardViews(tensor, visitedTensors);
}

MemoryType AssignMemoryType::InferTargetTypeThroughForwardViews(
    const LogicalTensorPtr& tensor, std::unordered_set<LogicalTensorPtr>& visitedTensors) const
{
    if (tensor == nullptr || !visitedTensors.insert(tensor).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    // 仅当唯一 consumer 为 OP_VIEW 时沿视图链前向推导，规避多分支分歧
    const auto& consumers = tensor->GetConsumers();
    if (consumers.size() != 1) {
        return MemoryType::MEM_UNKNOWN;
    }
    auto consumerOp = *consumers.begin();
    if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
        return MemoryType::MEM_UNKNOWN;
    }
    if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    auto viewOutput = consumerOp->oOperand.front();
    if (viewOutput->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return viewOutput->GetMemoryTypeOriginal();
    }
    MemoryType viewOutputRequirement = inserter.TryGetUniqueKnownRequiredType(viewOutput);
    if (viewOutputRequirement != MemoryType::MEM_UNKNOWN) {
        return viewOutputRequirement;
    }
    return InferTargetTypeThroughForwardViews(viewOutput, visitedTensors);
}

bool AssignMemoryType::KeepSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input,
                                          const LogicalTensorPtr& output)
{
    if (input == nullptr || output == nullptr) {
        return false;
    }
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    if (producers.empty() || consumers.empty()) {
        return false;
    }
    Operation* producer = *producers.begin();
    Operation* consumer = *consumers.begin();
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_ASSEMBLE);
    int64_t inputDataSize = input->GetDataSize();
    if (producer != nullptr && consumer != nullptr && producer->GetOpcode() == Opcode::OP_ASSEMBLE &&
        consumer->GetOpcode() == Opcode::OP_VIEW && input->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
        output->GetMemoryTypeOriginal() == MemoryType::MEM_UB && inputDataSize >= 0 &&
        static_cast<size_t>(inputDataSize) <= ubThreshold) {
        ForceSetRequirement(input, operation, MemoryType::MEM_UB, "InferSplitReshapeUb");
        for (const auto& consumerOp : output->GetConsumers()) {
            if (consumerOp != nullptr && !consumerOp->oOperand.empty() &&
                consumerOp->oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                ForceSetRequirement(output, *consumerOp, MemoryType::MEM_UB, "InferSplitReshapeUb");
            }
        }
        return true;
    }
    return false;
}

bool AssignMemoryType::IsDynamicReshape(Operation& operation, const LogicalTensorPtr& output) const
{
    static const std::string validShapeAttr = "op_attr_validShape";
    if (operation.HasAttr(validShapeAttr)) {
        return true;
    }
    if (output == nullptr) {
        return false;
    }
    for (const auto& dim : output->GetDynValidShape()) {
        if (!dim.IsImmediate()) {
            return true;
        }
    }
    return false;
}

bool AssignMemoryType::FitsTensorInUb(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr) {
        return false;
    }
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_NORMAL);
    int64_t dataSize = tensor->GetDataSize();
    return dataSize >= 0 && static_cast<size_t>(dataSize) <= ubThreshold;
}

Status AssignMemoryType::ApplyOtherSpecialOpcodeRules(Function& function)
{
    for (auto& op : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(HandleNopMemoryType(op));
    }
    return SUCCESS;
}

Status AssignMemoryType::HandleNopMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_NOP, "Handle OP_NOP memory type", input,
                                                      output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (inputRequirement == MemoryType::MEM_UNKNOWN || outputOriginal == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    if (inputRequirement != outputOriginal) {
        ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "HandleNopMismatchFallbackDdr");
        ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "HandleNopMismatchFallbackDdr");
        return SUCCESS;
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallback(Function& function)
{
    const size_t ubAssembleThreshold = static_cast<size_t>(
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_ASSEMBLE);
    const size_t ubNormalThreshold = static_cast<size_t>(
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_NORMAL);
    const size_t l1Threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) *
                                                   L1_THRESHOLD);
    APASS_LOG_INFO_F(Elements::Function, "Memory threshold: UB assemble %zu bytes, UB normal %zu bytes, L1 %zu bytes.",
                     ubAssembleThreshold, ubNormalThreshold, l1Threshold);
    for (auto& op : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(op));
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallback(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE && operation.GetOpcode() != Opcode::OP_VIEW) {
        return SUCCESS;
    }
    if (operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because output operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto output = operation.oOperand.front();
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because output tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    bool isAssemble = operation.GetOpcode() == Opcode::OP_ASSEMBLE;
    // op_view的输出不做L1内存类型回退，避免tile_shape设置异常场景下，回退到DDR导致出现非预期的view
    if (!IsOversizedLocalBuffer(output, output->GetMemoryTypeOriginal(), isAssemble, isAssemble)) {
        return SUCCESS;
    }
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedLocalBufferFallback");
    APASS_LOG_DEBUG_F(Elements::Operation, "Force %s[%d] output tensor[%d] to DDR by size limit.",
                      operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), output->GetMagic());
    if (operation.GetOpcode() == Opcode::OP_VIEW) {
        RETURN_IF_NOT_SUCCESS(DowngradeOversizedViewInputRequirement(operation));
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
        if (viewAttr != nullptr) {
            viewAttr->SetToType(MemoryType::MEM_DEVICE_DDR);
        }
    }
    return SUCCESS;
}

bool AssignMemoryType::IsOversizedLocalBuffer(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                              bool useAssembleUbLimit, bool allowL1Fallback) const
{
    if (memoryType == MemoryType::MEM_UB) {
        double ubLimitRatio = useAssembleUbLimit ? UB_THRESHOLD_ASSEMBLE : UB_THRESHOLD_NORMAL;
        size_t threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                               ubLimitRatio);
        return ExceedsMemoryLimit(tensor, threshold);
    }
    if (memoryType == MemoryType::MEM_L1 && allowL1Fallback) {
        size_t threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) *
                                               L1_THRESHOLD);
        return ExceedsMemoryLimit(tensor, threshold);
    }
    return false;
}

Status AssignMemoryType::DowngradeOversizedViewInputRequirement(Operation& operation)
{
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for OP_VIEW[%d] failed because of invalid input operand.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    MemoryType inputType = inserter.GetRequirementOrUnknown(input, operation);
    if (!IsOversizedLocalBuffer(input, inputType, false, true)) {
        return SUCCESS;
    }
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedViewInputFallback");
    APASS_LOG_DEBUG_F(Elements::Operation, "Force OP_VIEW[%d] input tensor[%d] requirement to DDR by size limit.",
                      operation.GetOpMagic(), input->GetMagic());
    return SUCCESS;
}

bool AssignMemoryType::ExceedsMemoryLimit(const LogicalTensorPtr& tensor, size_t threshold) const
{
    if (tensor == nullptr) {
        return false;
    }
    int64_t dataSize = tensor->GetDataSize();
    if (dataSize < 0) {
        return false;
    }
    return static_cast<size_t>(dataSize) > threshold;
}

Status AssignMemoryType::ApplyPlatformPathFallbackRules(Function& function)
{
    ProcessL0C2L1SmallToLarge(function);
    ProcessL0C2L1LargeToSmall(function);
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        ProcessL0C2UBSmallToLarge(function);
        ProcessL0C2UBLargeToSmall(function);
        ProcessUB2L1SmallToLarge(function);
        ProcessUB2L1LargeToSmall(function);
    }
    ProcessShapeTransportFallback(function);
    return SUCCESS;
}

Status AssignMemoryType::ResolveMemoryUnknowns(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visited;
    auto resolveTensor = [this, &visited](const LogicalTensorPtr& tensor) -> Status {
        if (tensor != nullptr && !visited.insert(tensor).second) {
            return SUCCESS;
        }
        return ResolveTensorMemoryUnknowns(tensor);
    };
    for (auto& op : function.Operations()) {
        for (auto& input : op.iOperand) {
            RETURN_IF_NOT_SUCCESS(resolveTensor(input));
        }
        for (auto& output : op.oOperand) {
            RETURN_IF_NOT_SUCCESS(resolveTensor(output));
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::ResolveTensorMemoryUnknowns(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Resolve tensor memory unknown failed because tensor is null.");
        return FAILED;
    }
    MemoryType original = tensor->GetMemoryTypeOriginal();
    if (original == MemoryType::MEM_UNKNOWN) {
        MemoryType inferredOriginal = InferOriginalFromRequirements(tensor);
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(tensor, inferredOriginal, "ResolveMemoryUnknowns"));
        original = tensor->GetMemoryTypeOriginal();
    }
    FillUnknownRequirementsWith(tensor, original, "ResolveMemoryUnknowns");
    return SUCCESS;
}

Status AssignMemoryType::SyncViewAssembleMemoryAttrs(Function& function)
{
    for (auto& operation : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(SyncViewMemoryAttr(operation));
        RETURN_IF_NOT_SUCCESS(SyncAssembleMemoryAttr(operation));
    }
    return SUCCESS;
}

Status AssignMemoryType::SyncViewMemoryAttr(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_VIEW) {
        return SUCCESS;
    }
    if (operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync OP_VIEW[%d] toAttr failed because output operand is empty.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync OP_VIEW[%d] toAttr failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto output = operation.oOperand.front();
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync OP_VIEW[%d] toAttr failed because output tensor is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType toType = output->GetMemoryTypeOriginal();
    if (toType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    viewOpAttribute->SetToType(toType);
    return SUCCESS;
}

Status AssignMemoryType::SyncAssembleMemoryAttr(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE) {
        return SUCCESS;
    }
    if (operation.iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync OP_ASSEMBLE[%d] fromAttr failed because input operand is empty.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync OP_ASSEMBLE[%d] fromAttr failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    if (input == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync OP_ASSEMBLE[%d] fromAttr failed because input tensor is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
    if (fromType == MemoryType::MEM_UNKNOWN) {
        fromType = input->GetMemoryTypeOriginal();
    }
    if (fromType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    assembleOpAttribute->SetFromType(fromType);
    return SUCCESS;
}

MemoryType AssignMemoryType::InferOriginalFromRequirements(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr) {
        return MemoryType::MEM_DEVICE_DDR;
    }
    auto knownRequirements = inserter.GetKnownRequiredTypes(tensor);
    if (knownRequirements.size() == 1) {
        return *knownRequirements.begin();
    }
    return MemoryType::MEM_DEVICE_DDR;
}

Status AssignMemoryType::SyncTensorToBe(Function& function)
{
    size_t syncCount = 0;
    std::unordered_set<LogicalTensorPtr> visited;
    auto syncTensor = [&syncCount, &visited](const LogicalTensorPtr& tensor) {
        if (tensor == nullptr) {
            return;
        }
        if (!visited.insert(tensor).second) {
            return;
        }
        tensor->SetMemoryTypeToBe(tensor->GetMemoryTypeOriginal());
        ++syncCount;
    };
    for (auto& op : function.Operations()) {
        for (auto& input : op.iOperand) {
            syncTensor(input);
        }
        for (auto& output : op.oOperand) {
            syncTensor(output);
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::SetOriginalChecked(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                            const std::string& reason, bool allowOverride)
{
    std::string context = reason.empty() ? "unknown" : reason;
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "SetOriginalChecked failed because tensor is null, reason: %s.",
                          context.c_str());
        return FAILED;
    }
    if (memoryType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryType currentType = tensor->GetMemoryTypeOriginal();
    if (currentType != MemoryType::MEM_UNKNOWN && currentType != memoryType && !allowOverride) {
        APASS_LOG_WARN_F(
            Elements::Tensor,
            "Skip tensor %d original memory type update because current %s conflicts with new %s, reason: %s.",
            tensor->GetMagic(), BriefMemoryTypeToString(currentType).c_str(),
            BriefMemoryTypeToString(memoryType).c_str(), context.c_str());
        return SUCCESS;
    }
    tensor->SetMemoryTypeOriginal(memoryType, allowOverride);
    return SUCCESS;
}

void AssignMemoryType::ForceSetOriginal(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                        const std::string& reason)
{
    if (tensor != nullptr && memoryType != MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Force tensor[%d] original as %s, reason %s.", tensor->GetMagic(),
                          BriefMemoryTypeToString(memoryType).c_str(), reason.c_str());
    }
    SetOriginalChecked(tensor, memoryType, reason, true);
}

Status AssignMemoryType::SetRequirementChecked(const LogicalTensorPtr& tensor, Operation& operation,
                                               MemoryType memoryType, const std::string& reason, bool allowOverride)
{
    std::string context = reason.empty() ? "unknown" : reason;
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "SetRequirementChecked failed because tensor is null for operation %s[%d], reason: %s.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), context.c_str());
        return FAILED;
    }
    if (!tensor->HasConsumer(operation)) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Operation %s[%d] is not a consumer of tensor %d, reason: %s.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), tensor->GetMagic(),
                          context.c_str());
        return FAILED;
    }
    bool hasRequirement = inserter.HasRequirement(tensor, operation);
    MemoryType currentType = inserter.GetRequirementOrUnknown(tensor, operation);
    if (hasRequirement && currentType != MemoryType::MEM_UNKNOWN && memoryType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    if (currentType != MemoryType::MEM_UNKNOWN && memoryType != MemoryType::MEM_UNKNOWN && currentType != memoryType &&
        !allowOverride) {
        APASS_LOG_WARN_F(
            Elements::Tensor,
            "Skip tensor %d requirement update for operation %s[%d] because current %s conflicts with new %s, "
            "reason: %s.",
            tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
            BriefMemoryTypeToString(currentType).c_str(), BriefMemoryTypeToString(memoryType).c_str(), context.c_str());
        return SUCCESS;
    }
    inserter.UpdateTensorTobeMap(tensor, operation, memoryType, context.c_str());
    return SUCCESS;
}

void AssignMemoryType::ForceSetRequirement(const LogicalTensorPtr& tensor, Operation& operation, MemoryType memoryType,
                                           const std::string& reason)
{
    if (tensor != nullptr && memoryType != MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Force tensor[%d] requirement for %s[%d] as %s, reason %s.",
                          tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                          BriefMemoryTypeToString(memoryType).c_str(), reason.c_str());
    }
    SetRequirementChecked(tensor, operation, memoryType, reason, true);
}

void AssignMemoryType::FillUnknownRequirementsWith(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                                   const char* reason)
{
    if (tensor == nullptr || memoryType == MemoryType::MEM_UNKNOWN) {
        return;
    }
    auto requirements = inserter.GetConsumerRequirements(tensor);
    for (const auto& item : requirements) {
        if (item.second == MemoryType::MEM_UNKNOWN) {
            inserter.UpdateTensorTobeMap(tensor, *item.first, memoryType, reason);
        }
    }
}

bool AssignMemoryType::AreAllConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType memoryType) const
{
    auto requirements = inserter.GetConsumerRequirements(tensor);
    return std::all_of(requirements.begin(), requirements.end(),
                       [memoryType](const auto& item) { return item.second == memoryType; });
}

void AssignMemoryType::DowngradeConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType fromType)
{
    for (const auto& [consumerOp, memoryType] : inserter.GetConsumerRequirements(tensor)) {
        if (memoryType == fromType) {
            inserter.UpdateTensorTobeMap(tensor, *consumerOp, MemoryType::MEM_DEVICE_DDR);
        }
    }
}

Status AssignMemoryType::InsertConvertOpsAndInferShape(Function& function)
{
    std::unordered_set<Operation*> existingOps;
    for (auto& op : function.Operations()) {
        existingOps.insert(&op);
    }
    RETURN_IF_NOT_SUCCESS(inserter.DoInsertion(function));
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    std::vector<Operation*> addedOps;
    for (auto& op : function.Operations(false)) {
        if (existingOps.find(&op) == existingOps.end()) {
            addedOps.push_back(&op);
        }
    }
    if (!addedOps.empty()) {
        if (InferShapeUtils::InferShape(function, addedOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape for added ops failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::PreCheck(Function& function) { return checker.DoPreCheck(function); }

Status AssignMemoryType::PostCheck(Function& function) { return checker.DoPostCheck(function); }

int64_t AssignMemoryType::CalcLineOffset(const Shape& shape, const Offset& offset) const
{
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }
    int64_t lineOffset = 0;
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        lineOffset += offset[i - 1] * stride;
        stride *= shape[i - 1];
    }
    return lineOffset;
}

void AssignMemoryType::ProcessL0C2L1SmallToLarge(Function& function)
{
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto oOperand = op.GetOOperands().front();
        auto iOperand = op.GetIOperands().front();
        if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            continue;
        }
        if (iOperand->GetMemoryTypeOriginal() != MEM_L0C) {
            continue;
        }
        bool isConsumerOutputMultiple = CheckConsumerViewShapeMultiple(oOperand, iOperand);
        if (HasParallelDifferentConsumerRequirement(iOperand, MemoryType::MEM_L1) ||
            !AreAllConsumerRequirements(oOperand, MemoryType::MEM_L1) ||
            !IsDimMultiple(oOperand->GetShape(), iOperand->GetShape()) || !isConsumerOutputMultiple) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, MemoryType::MEM_L0C);
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Set tensor %d original memory type "
                              "to DDR since not towards L1 or not multipule dimensions.",
                              oOperand->magic);
        }
    }
}

void AssignMemoryType::ProcessL0C2L1LargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewOpAttribute->GetTo() != MEM_L1) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            HasParallelDifferentConsumerRequirement(iOperand, MemoryType::MEM_L1)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            !IsDimMultiple(iOperand->GetShape(), oOperand->GetShape())) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 &&
            iOperand->GetMemoryTypeOriginal() == MEM_UB && oOperand->shape != iOperand->shape) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
    }
}

bool AssignMemoryType::CheckUBTileShape(const LogicalTensorPtr& output)
{
    if (output->GetShape()[0] % L0C_TILE_SIZE == 0 && output->GetShape()[1] % L0C_TILE_SIZE == 0) {
        return true;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor,
                      "Set tensor %d original memory type to DDR since vector tile shape "
                      "is not 16-element aligned.",
                      output->magic);
    return false;
}

bool AssignMemoryType::CheckConsumerViewShapeMultiple(const LogicalTensorPtr& output, const LogicalTensorPtr& input)
{
    for (auto& consumerOp : output->GetConsumers()) {
        if (consumerOp->GetOpcode() == Opcode::OP_VIEW &&
            !IsDimMultiple(consumerOp->GetOOperands().front()->GetShape(), input->GetShape())) {
            return false;
        }
    }
    return true;
}

static bool IsViewConsumerToUb(Operation* consumerOp)
{
    if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW || consumerOp->oOperand.empty()) {
        return false;
    }
    auto output = consumerOp->oOperand.front();
    if (output != nullptr && output->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        return true;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
    return viewOpAttribute != nullptr && viewOpAttribute->GetTo() == MemoryType::MEM_UB;
}

static bool IsConsumerRequirementTowardsUb(const LogicalTensorPtr& tensor, Operation* consumerOp,
                                           MemoryType requirement)
{
    if (requirement == MemoryType::MEM_UB) {
        return true;
    }
    if (requirement != tensor->GetMemoryTypeOriginal()) {
        return false;
    }
    return IsViewConsumerToUb(consumerOp);
}

static bool AreAllConsumerRequirementsTowardsUb(ConvertInserter& inserter, const LogicalTensorPtr& tensor)
{
    auto requirements = inserter.GetConsumerRequirements(tensor);
    if (requirements.empty()) {
        return false;
    }
    return std::all_of(requirements.begin(), requirements.end(), [&tensor](const auto& item) {
        return IsConsumerRequirementTowardsUb(tensor, item.first, item.second);
    });
}

void AssignMemoryType::ProcessL0C2UBSmallToLarge(Function& function)
{
    constexpr size_t kMatrixShapeDimCount = 2; // 矩阵形状维度数 (M, N)
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto oOperand = op.GetOOperands().front();
        auto iOperand = op.GetIOperands().front();
        if (iOperand->GetMemoryTypeOriginal() != MEM_L0C) {
            continue;
        }
        if (iOperand->GetShape().size() != kMatrixShapeDimCount ||
            oOperand->GetShape().size() != kMatrixShapeDimCount) {
            continue;
        }
        bool isConsumerOutputMultiple = CheckConsumerViewShapeMultiple(oOperand, iOperand);
        bool isVecTileShapeValid = CheckUBTileShape(oOperand);
        bool canUseUb = !HasParallelDifferentConsumerRequirement(iOperand, MemoryType::MEM_UB) &&
                        AreAllConsumerRequirementsTowardsUb(inserter, oOperand) &&
                        IsDimMultiple(oOperand->GetShape(), iOperand->GetShape()) && isConsumerOutputMultiple &&
                        isVecTileShapeValid && FitsAssembleOutputMemoryLimit(oOperand, MemoryType::MEM_UB);
        if (!canUseUb) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, MemoryType::MEM_L0C);
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Set tensor %d original memory type to DDR since "
                              "not towards UB or not multiple dimensions.",
                              oOperand->magic);
            continue;
        }
        ForceSetOriginal(oOperand, MemoryType::MEM_UB, "ProcessL0C2UBSmallToLarge");
        for (const auto& [consumerOp, memoryType] : inserter.GetConsumerRequirements(oOperand)) {
            if (memoryType != MemoryType::MEM_UB && IsViewConsumerToUb(consumerOp)) {
                inserter.UpdateTensorTobeMap(oOperand, *consumerOp, MemoryType::MEM_UB, "ProcessL0C2UBSmallToLarge");
            }
        }
    }
}

void AssignMemoryType::ProcessL0C2UBLargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewOpAttribute->GetTo() != MEM_UB) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        bool isVecTileShapeValid = CheckUBTileShape(oOperand);
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            HasParallelDifferentConsumerRequirement(iOperand, MemoryType::MEM_UB)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            (!IsDimMultiple(iOperand->GetShape(), oOperand->GetShape()) || !isVecTileShapeValid)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
    }
}

void AssignMemoryType::ProcessUB2L1SmallToLarge(Function& function)
{
    constexpr size_t kMatrixShapeDimCount = 2; // 矩阵形状维度数 (M, N)
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto oOperand = op.GetOOperands().front();
        auto iOperand = op.GetIOperands().front();
        if (iOperand->GetMemoryTypeOriginal() != MEM_UB || oOperand->GetMemoryTypeOriginal() != MEM_L1) {
            continue;
        }
        if (iOperand->GetShape().size() != kMatrixShapeDimCount ||
            oOperand->GetShape().size() != kMatrixShapeDimCount) {
            continue;
        }
        if (ShouldSkipUB2L1SmallToLarge(iOperand, oOperand)) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            continue;
        }
        if (HasParallelDifferentConsumerRequirement(iOperand, MemoryType::MEM_L1) ||
            !AreAllConsumerRequirements(oOperand, MemoryType::MEM_L1) ||
            !IsDimMultiple(oOperand->GetShape(), iOperand->GetShape()) ||
            !CheckConsumerViewShapeMultiple(oOperand, iOperand)) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, MemoryType::MEM_UB);
        }
    }
}

bool AssignMemoryType::ShouldSkipUB2L1SmallToLarge(const LogicalTensorPtr& iOperand,
                                                   const LogicalTensorPtr& oOperand) const
{
    const size_t UB_LIMIT = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                UB_THRESHOLD_NORMAL);
    if (CalcNZTensorSize(iOperand) > UB_LIMIT) {
        return true;
    }
    // 检查 consumer view 是否有 copy_in_mode=0 属性
    for (auto& consumerOp : oOperand->GetConsumers()) {
        if (consumerOp->GetOpcode() == Opcode::OP_VIEW) {
            int64_t copyInModeValue = 0;
            if (consumerOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0) {
                return true;
            }
        }
    }
    return !CheckInnerAxisC0Size(iOperand, oOperand);
}

void AssignMemoryType::ProcessUB2L1LargeToSmall(Function& function)
{
    constexpr size_t kMatrixShapeDimCount = 2; // 矩阵形状维度数 (M, N)
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        MemoryType attrToType = viewOpAttribute->GetTo();
        if (attrToType != MEM_L1) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        if (iOperand->GetMemoryTypeOriginal() != MEM_UB) {
            continue;
        }
        if (HasParallelDifferentConsumerRequirement(iOperand, MemoryType::MEM_L1)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (iOperand->GetShape().size() != kMatrixShapeDimCount ||
            oOperand->GetShape().size() != kMatrixShapeDimCount) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        const size_t UB_LIMIT = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                    UB_THRESHOLD_NORMAL);
        size_t totalSize = CalcNZTensorSize(iOperand);
        if (totalSize > UB_LIMIT) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "UB2L1 large to small: totalSize %zu exceeds UB_LIMIT %zu, downgrade to DDR", totalSize,
                              UB_LIMIT);
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        int64_t copyInModeValue = 0;
        if (op.GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "UB2L1 large to small skip: bias/scale tensor (copy_in_mode=%ld), View Op[%d]",
                              static_cast<long>(copyInModeValue), op.GetOpMagic());
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (!IsDimMultiple(iOperand->GetShape(), oOperand->GetShape())) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
        }
    }
}

// 判定两 tensor 是否满足任意方向的逐维整除关系（含等大放行）
bool AssignMemoryType::IsAllowedTransport(const LogicalTensorPtr& prodOut, const LogicalTensorPtr& consIn) const
{
    if (prodOut == nullptr || consIn == nullptr) {
        return false;
    }
    return IsDimMultiple(prodOut->GetShape(), consIn->GetShape()) ||
           IsDimMultiple(consIn->GetShape(), prodOut->GetShape());
}

namespace {
// 端点收集方向：沿 producer 方向找真实生产者输出 / 沿 consumer 方向找真实消费者输入
enum class TraverseDirection { PRODUCER, CONSUMER };

// 沿 direction 方向穿透 TRANSPARENT_OPS，限定在 boundType 层级内；
// 若透明 op 的对侧 tensor 离开 boundType，把当前 t 标记为端点（层级边界，不再继续）
std::vector<LogicalTensorPtr> CollectRealEndpoints(const LogicalTensorPtr& tensor, MemoryType boundType,
                                                   TraverseDirection direction)
{
    if (tensor == nullptr) {
        return {};
    }
    std::vector<LogicalTensorPtr> endpoints;
    std::unordered_set<LogicalTensorPtr> visited;
    std::vector<LogicalTensorPtr> stack = {tensor};
    while (!stack.empty()) {
        auto t = stack.back();
        stack.pop_back();
        if (t == nullptr || !visited.insert(t).second) {
            continue;
        }
        bool isEndpoint = false;
        auto& ops = (direction == TraverseDirection::PRODUCER) ? t->GetProducers() : t->GetConsumers();
        for (auto* op : ops) {
            if (TRANSPARENT_OPS.count(op->GetOpcode()) == 0) {
                isEndpoint = true;
                continue;
            }
            auto& nextTensors = (direction == TraverseDirection::PRODUCER) ? op->GetIOperands() : op->GetOOperands();
            for (auto& nextT : nextTensors) {
                if (nextT->GetMemoryTypeOriginal() == boundType) {
                    stack.push_back(nextT);
                } else {
                    isEndpoint = true;
                }
            }
        }
        if (isEndpoint) {
            endpoints.push_back(t);
        }
    }
    return endpoints;
}
} // namespace

// 穿透式 shape 兜底：对仍走 L0C↔L1/L0C↔UB/UB↔L1 片上直连的 view/assemble，
// 用真实生产者输出×真实消费者输入做双向逐维整除判断，不满足则拦截回退 DDR
void AssignMemoryType::ProcessShapeTransportFallback(Function& function)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    bool is3510 = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        MemoryType fromType = iOperand->GetMemoryTypeOriginal();
        MemoryType toType = MEM_DEVICE_DDR;
        bool isTransport = false;
        if (opcode == Opcode::OP_VIEW) {
            auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            if (viewAttr == nullptr) {
                continue;
            }
            toType = viewAttr->GetTo();
            // L0C→L1 全平台; L0C→UB / UB→L1 仅 DAV_3510
            bool pathL0C2L1 = (fromType == MEM_L0C && toType == MEM_L1);
            bool path3510 = is3510 &&
                            ((fromType == MEM_L0C && toType == MEM_UB) || (fromType == MEM_UB && toType == MEM_L1));
            isTransport = (pathL0C2L1 || path3510) && inserter.GetRequirementOrUnknown(iOperand, op) != MEM_DEVICE_DDR;
        } else if (opcode == Opcode::OP_ASSEMBLE) {
            toType = oOperand->GetMemoryTypeOriginal();
            if (toType == MEM_DEVICE_DDR) {
                continue;
            }
            // L0C→L1(iOperand=L0C, oOperand≠UB) 全平台; L0C→UB / UB→L1 仅 DAV_3510
            bool pathL0C2L1 = (fromType == MEM_L0C && toType != MEM_UB);
            bool path3510 = is3510 &&
                            ((fromType == MEM_L0C && toType == MEM_UB) || (fromType == MEM_UB && toType == MEM_L1));
            isTransport = (pathL0C2L1 || path3510);
        }
        if (!isTransport) {
            continue;
        }
        // 收集穿透端点：iOperand 向前找真实生产者输出，oOperand 向后找真实消费者输入；
        // 穿透限定在各自层级内(fromType/toType)，跨层级边界即停
        auto prodOutputs = CollectRealEndpoints(iOperand, fromType, TraverseDirection::PRODUCER);
        auto consInputs = CollectRealEndpoints(oOperand, toType, TraverseDirection::CONSUMER);
        // 所有端点对 IsAllowedTransport 全成立才放行；端点集为空或任一不成立则拦截。
        // 空端点集（如函数出口/无 producer 的 tensor）无法证明兼容，保守拦截回退 DDR
        Shape firstBadProdShape;
        Shape firstBadConsShape;
        bool allowed = !prodOutputs.empty() && !consInputs.empty();
        for (const auto& prodOut : prodOutputs) {
            for (const auto& consIn : consInputs) {
                if (!IsAllowedTransport(prodOut, consIn)) {
                    allowed = false;
                    firstBadProdShape = prodOut->GetShape();
                    firstBadConsShape = consIn->GetShape();
                    break;
                }
            }
            if (!allowed) {
                break;
            }
        }
        if (allowed) {
            continue;
        }
        // 拦截：view→tobe=DDR; assemble→origin=DDR+DowngradeConsumerRequirements
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "ShapeTransportFallback intercepts %s[%d] on-chip path %s->%s to DDR: prodOut=%s consIn=%s.",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), BriefMemoryTypeToString(fromType).c_str(),
                          BriefMemoryTypeToString(toType).c_str(), IntVecToStr(firstBadProdShape).c_str(),
                          IntVecToStr(firstBadConsShape).c_str());
        if (opcode == Opcode::OP_VIEW) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
        } else {
            oOperand->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, fromType);
        }
    }
}

bool AssignMemoryType::CheckInnerAxisC0Size(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const
{
    constexpr int64_t kC0AlignBytes = 32;
    size_t inputInnerAxis = input->GetShape().back();
    size_t outputInnerAxis = output->GetShape().back();
    // 如果输入内轴大小等于输出内轴大小，说明内轴未被切分
    // 这种情况不需要检查对齐，直接返回 true
    if (inputInnerAxis == outputInnerAxis) {
        return true;
    }
    int64_t inputDtypeBytes = BytesOf(input->Datatype());
    int64_t outputDtypeBytes = BytesOf(output->Datatype());
    // 检查数据类型字节数是否有效（避免除零）
    int64_t inputC0Size = (inputDtypeBytes > 0) ? (kC0AlignBytes / inputDtypeBytes) : 0;
    int64_t outputC0Size = (outputDtypeBytes > 0) ? (kC0AlignBytes / outputDtypeBytes) : 0;
    if (inputC0Size <= 0 || outputC0Size <= 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "CheckInnerAxisC0Size: invalid C0 size, inputC0Size=%ld, outputC0Size=%ld",
                          static_cast<long>(inputC0Size), static_cast<long>(outputC0Size));
        return false;
    }
    // 分别检查 input 和 output 的内轴是否满足各自的 C0 size 切分
    if (inputInnerAxis % static_cast<size_t>(inputC0Size) != 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "CheckInnerAxisC0Size: input inner=%zu, dtypeBytes=%ld, c0Size=%ld, not aligned",
                          inputInnerAxis, static_cast<long>(inputDtypeBytes), static_cast<long>(inputC0Size));
        return false;
    }
    if (outputInnerAxis % static_cast<size_t>(outputC0Size) != 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "CheckInnerAxisC0Size: output inner=%zu, dtypeBytes=%ld, c0Size=%ld, not aligned",
                          outputInnerAxis, static_cast<long>(outputDtypeBytes), static_cast<long>(outputC0Size));
        return false;
    }
    return true;
}

bool AssignMemoryType::IsDimMultiple(const Shape& shape1, const Shape& shape2) const
{
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] <= 0 || shape2[i] <= 0 || shape1[i] % shape2[i] != 0) {
            return false;
        }
    }
    return true;
}

size_t AssignMemoryType::CalcNZTensorSize(const LogicalTensorPtr& tensor) const
{
    constexpr int64_t kAlignBytes = 4;
    constexpr int64_t kC0AlignBytes = 32;
    DataType dtype = tensor->Datatype();
    int64_t bytes = BytesOf(dtype);
    size_t outer = tensor->GetShape()[0];
    size_t inner = tensor->GetShape()[1];
    // 外轴对齐：INT8/FP8 对齐到 32，其他对齐到 16
    size_t outerAlign = (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8) ? 32 : 16;
    // 内轴对齐：C0 size = 32 / 元素字节数
    size_t c0 = 0;
    if (bytes > 0) {
        c0 = static_cast<size_t>(kC0AlignBytes / bytes);
    }
    if (c0 <= 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "CalcNZTensorSize: invalid C0 size, c0=%zu", c0);
        // 返回原始 ND 格式大小作为 fallback
        return outer * inner * static_cast<size_t>(bytes > 0 ? bytes : kAlignBytes);
    }
    size_t alignedOuter = (outer + outerAlign - 1) / outerAlign * outerAlign + 1;
    size_t alignedInner = (inner + c0 - 1) / c0 * c0;
    // NZ 格式大小
    size_t nzSize = alignedOuter * alignedInner * static_cast<size_t>(bytes);
    // ND 格式原始大小
    size_t ndSize = outer * inner * static_cast<size_t>(bytes);
    // ND + NZ 同时存在，需要两者之和
    return ndSize + nzSize;
}
} // namespace npu::tile_fwk::legacy
