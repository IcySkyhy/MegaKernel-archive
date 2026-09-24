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

#include "assign_memory_type.h"
#include "assign_memory_type_legacy.h"

#include <algorithm>
#include <set>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
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

namespace npu::tile_fwk {
Status AssignMemoryType::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start AssignMemoryType.");
    if (!config::EnableSlice()) {
        return RunOnFunctionLegacy(function);
    }
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    RETURN_IF_NOT_SUCCESS(AssignConfirmedMemoryTypes(function));
    RETURN_IF_NOT_SUCCESS(InferUncertainMemoryTypes(function));
    RETURN_IF_NOT_SUCCESS(ResolveMemoryUnknowns(function));
    RETURN_IF_NOT_SUCCESS(SyncViewAssembleMemoryAttrs(function));
    RETURN_IF_NOT_SUCCESS(FixViewAssembleSemanticMismatch(function));
    RETURN_IF_NOT_SUCCESS(InsertConvertOpsAndInferShape(function));
    RETURN_IF_NOT_SUCCESS(FallbackSameMemoryMoveOps(function));
    RETURN_IF_NOT_SUCCESS(SyncTensorToBe(function));
    APASS_LOG_INFO_F(Elements::Function, "===> End AssignMemoryType.");
    return SUCCESS;
}

Status AssignMemoryType::AssignConfirmedMemoryTypes(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_SLICE) {
            RETURN_IF_NOT_SUCCESS(AssignViewAttrMemoryType(op));
            RETURN_IF_NOT_SUCCESS(AssignSliceInputRequirement(op));
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_CONTRACT) {
            RETURN_IF_NOT_SUCCESS(AssignAssembleAttrMemoryType(op));
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
            } else if (producerOpcode == Opcode::OP_SLICE) {
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
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(operation.oOperand.front(), attrToType, "AssignViewAttrMemoryType"));
    if (operation.GetOpcode() == Opcode::OP_VIEW) {
        RETURN_IF_NOT_SUCCESS(
            SetRequirementChecked(operation.iOperand.front(), operation, attrToType, "AssignViewAttrMemoryType"));
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignSliceInputRequirement(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_SLICE) {
        return SUCCESS;
    }
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assign OP_SLICE[%d] input requirement failed because input operand is empty or null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.empty() ? nullptr : operation.oOperand.front();
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assign OP_SLICE[%d] input requirement failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType targetType = viewOpAttribute->GetTo();
    if (output != nullptr && output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        targetType = output->GetMemoryTypeOriginal();
    }
    if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
        (targetType == MemoryType::MEM_L0A || targetType == MemoryType::MEM_L0B)) {
        return SetRequirementChecked(input, operation, MemoryType::MEM_L1, "AssignSliceInputRequirementLocalCopyIn");
    }
    MemoryType requirement = MemoryType::MEM_DEVICE_DDR;
    for (const auto& producerOp : input->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto producerViewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
        if (producerViewOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Assign OP_SLICE[%d] input requirement failed because producer OP_SLICE[%d] view attr is null.",
                operation.GetOpMagic(), producerOp->GetOpMagic());
            return FAILED;
        }
        if (producerViewOpAttribute->GetTo() == MemoryType::MEM_L1) {
            requirement = MemoryType::MEM_L1;
            break;
        }
    }
    return SetRequirementChecked(input, operation, requirement, "AssignSliceInputRequirement");
}

Status AssignMemoryType::AssignAssembleAttrMemoryType(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_ASSEMBLE && opcode != Opcode::OP_CONTRACT)
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
    RETURN_IF_NOT_SUCCESS(
        SetRequirementChecked(operation.iOperand.front(), operation, attrFromType, "AssignAssembleAttrMemoryType"));
    if (opcode != Opcode::OP_ASSEMBLE) {
        return SUCCESS;
    }
    auto output = operation.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal == MemoryType::MEM_UNKNOWN) {
        return SetOriginalChecked(output, attrFromType, "AssignAssembleAttrMemoryType");
    }
    if (outputOriginal == attrFromType) {
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation,
                      "Assign assemble attr memory type failed because view operation memory type conflict exists. "
                      "%s[%d] from type is %s, output tensor[%d] original type is %s.",
                      operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                      BriefMemoryTypeToString(attrFromType).c_str(), output->GetMagic(),
                      BriefMemoryTypeToString(outputOriginal).c_str());
    return FAILED;
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

// l0c2ub pattern: batchmatmul case: cube op -> contract(s) -> reshape op -> slice(s)/contract(s) -> vector
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
        bool isProducerContract = producer->GetOpcode() == npu::tile_fwk::Opcode::OP_CONTRACT;
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
        if (!isProducerContract || !isProducerProducerAllCube) {
            isL0C2UBPattern = false;
        }
    }
    for (auto& consumer : consumers) {
        bool isConsumerSliceContract = consumer->GetOpcode() == npu::tile_fwk::Opcode::OP_SLICE ||
                                       consumer->GetOpcode() == npu::tile_fwk::Opcode::OP_CONTRACT;
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
        if (!isConsumerSliceContract || !isConsumerConsumerAllVector) {
            isL0C2UBPattern = false;
        }
    }
    return isL0C2UBPattern;
}

// ub2l1 pattern: 2 patterns
// 1. vector op -> slice/contract -> reshape op -> slice from l1 -> slice from l0a -> cube
// 2. vector op -> slice/contract -> slice/contract -> reshape op -> slice from l1 -> slice from l0a -> cube
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
        bool isProducerDepth1SliceContract = producer->GetOpcode() == npu::tile_fwk::Opcode::OP_SLICE ||
                                             producer->GetOpcode() == npu::tile_fwk::Opcode::OP_CONTRACT;
        bool isProducerDepth2AllVector = false;
        bool isProducerDepth2AllSliceContract = false;
        bool isProducerDepth3AllVector = false;

        std::vector<bool> isProducerDepth2Vector;
        std::vector<bool> isProducerDepth2SliceContract;
        std::vector<bool> isProducerDepth3Vector;
        for (auto& producerIOperand : producer->iOperand) {
            for (auto& producerProducer : producerIOperand->GetProducers()) {
                isProducerDepth2Vector.push_back(producerProducer->GetCoreType() == CoreType::AIV);
                isProducerDepth2SliceContract.push_back(
                    producerProducer->GetOpcode() == npu::tile_fwk::Opcode::OP_SLICE ||
                    producerProducer->GetOpcode() == npu::tile_fwk::Opcode::OP_CONTRACT);
                CollectProducerAIVFlags(producerProducer, isProducerDepth3Vector);
            }
        }
        isProducerDepth2AllVector = !isProducerDepth2Vector.empty() &&
                                    std::all_of(isProducerDepth2Vector.begin(), isProducerDepth2Vector.end(),
                                                [](bool val) { return val; });
        isProducerDepth2AllSliceContract = !isProducerDepth2SliceContract.empty() &&
                                           std::all_of(isProducerDepth2SliceContract.begin(),
                                                       isProducerDepth2SliceContract.end(),
                                                       [](bool val) { return val; });
        isProducerDepth3AllVector = !isProducerDepth3Vector.empty() &&
                                    std::all_of(isProducerDepth3Vector.begin(), isProducerDepth3Vector.end(),
                                                [](bool val) { return val; });
        // currently only support the following patterns:
        // 1. vector op -> slice(s)/contract(s) -> reshape op
        // 2. vector op -> slice(s)/contract(s) -> slice(s)/contract(s) -> reshape op
        if (!((isProducerDepth1SliceContract && isProducerDepth2AllSliceContract && isProducerDepth3AllVector) ||
              (isProducerDepth1SliceContract && isProducerDepth2AllVector))) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::IsReshapeVecToCubeUB2L1ConsumerPattern(
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    for (auto& consumer : consumers) {
        bool isConsumerDepth1Slice = consumer->GetOpcode() == npu::tile_fwk::Opcode::OP_SLICE;
        bool isConsumerDepth2AllSlice = false;
        bool isConsumerDepth3AllCube = false;

        std::vector<bool> isConsumerDepth2Slice;
        std::vector<bool> isConsumerDepth3Cube;
        for (auto& consumerOOperand : consumer->oOperand) {
            for (auto& consumerConsumer : consumerOOperand->GetConsumers()) {
                isConsumerDepth2Slice.push_back(consumerConsumer->GetOpcode() == npu::tile_fwk::Opcode::OP_SLICE);
                CollectConsumerAICFlags(consumerConsumer, isConsumerDepth3Cube);
            }
        }
        isConsumerDepth2AllSlice = !isConsumerDepth2Slice.empty() &&
                                   std::all_of(isConsumerDepth2Slice.begin(), isConsumerDepth2Slice.end(),
                                               [](bool val) { return val; });
        isConsumerDepth3AllCube = !isConsumerDepth3Cube.empty() &&
                                  std::all_of(isConsumerDepth3Cube.begin(), isConsumerDepth3Cube.end(),
                                              [](bool val) { return val; });
        if (!isConsumerDepth1Slice || !isConsumerDepth2AllSlice || !isConsumerDepth3AllCube) {
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

    // l0c2ub pattern: batchmatmul case: cube op -> contract(s) -> reshape op -> slice(s)/contract(s) -> vector
    if (IsReshapeCubeToVecL0C2UBPattern(op) && FitsTensorInUb(input)) {
        for (auto& producer : producers) {
            auto& producerInput = producer->iOperand.front();
            auto& producerOutput = producer->oOperand.front();

            // set producer contract input to L0C
            producerInput->SetMemoryTypeOriginal(MemoryType::MEM_L0C, true);
            inserter.UpdateTensorTobeMap(producerInput, *producer, MemoryType::MEM_L0C);

            // set producer output to be UB
            producerOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(producerOutput, op, MemoryType::MEM_UB);
        }
        // set reshape output to UB
        output->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

        // set all consumer slice/contracts input to UB
        for (auto& consumer : consumers) {
            inserter.UpdateTensorTobeMap(output, *consumer, MemoryType::MEM_UB);
        }
        return SUCCESS;
    }

    // ub2l1 pattern:
    // 1. vector op -> slice(s)/contract(s) -> reshape op -> slice(s) from l1 -> slice(s) from l0a -> cube
    // 2. vector op -> contract(s) -> slice(s) -> reshape op -> slice(s) from l1 -> slice(s) from l0a -> cube
    if (IsReshapeVecToCubeUB2L1Pattern(op) && FitsTensorInUb(output)) {
        for (auto& producer : producers) {
            auto& producerInput = producer->iOperand.front();
            auto& producerOutput = producer->oOperand.front();

            // set producer slice/contract input to UB
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

            // set consumer slice output to be L1
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
            case Opcode::OP_SLICE:
                RETURN_IF_NOT_SUCCESS(InferSliceMemoryType(op));
                break;
            case Opcode::OP_VIEW_TYPE:
                RETURN_IF_NOT_SUCCESS(InferViewTypeMemoryType(op));
                break;
            case Opcode::OP_ASSEMBLE:
                RETURN_IF_NOT_SUCCESS(InferAssembleMemoryType(op, inferredAssembleOutputs));
                break;
            case Opcode::OP_CONTRACT:
                RETURN_IF_NOT_SUCCESS(InferContractMemoryType(op));
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
    return ApplyPlatformPathUpgradeRules(function);
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

static bool IsViewSemanticOpcode(Opcode opcode)
{
    return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_RESHAPE;
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
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryType inferredType = InferRequirementFromInputOriginals(input);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        inferredType = InferOriginalFromOutputRequirements(output);
    }
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    ForceSetRequirement(input, operation, inferredType, "InferViewSemanticType");
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, inferredType, "InferViewSemanticType"));
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute != nullptr) {
        viewOpAttribute->SetToType(inferredType);
    }
    return SUCCESS;
}

Status AssignMemoryType::InferSliceMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_SLICE, "Infer OP_SLICE memory type", input,
                                                      output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_SLICE[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        viewOpAttribute->SetToType(output->GetMemoryTypeOriginal());
        return SUCCESS;
    }
    MemoryType inferredType = InferOriginalFromOutputRequirements(output);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        inferredType = MemoryType::MEM_UB;
    }
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, inferredType, "InferSliceOutputOriginal"));
    viewOpAttribute->SetToType(inferredType);
    return SUCCESS;
}

Status AssignMemoryType::InferContractMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_CONTRACT, "Infer OP_CONTRACT memory type",
                                                      input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_CONTRACT[%d] memory type failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    if (inputRequirement != MemoryType::MEM_UNKNOWN) {
        assembleOpAttribute->SetFromType(inputRequirement);
        return SUCCESS;
    }
    MemoryType inferredType = InferRequirementFromInputOriginals(input);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        inferredType = MemoryType::MEM_UB;
    }
    ForceSetRequirement(input, operation, inferredType, "InferContractInputRequirement");
    assembleOpAttribute->SetFromType(inferredType);
    return SUCCESS;
}

MemoryType AssignMemoryType::InferOriginalFromOutputRequirements(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    return InferOriginalFromOutputRequirements(tensor, visitedTensors);
}

MemoryType AssignMemoryType::InferOriginalFromOutputRequirements(
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
        MemoryType requirement = item.second;
        if (requirement != MemoryType::MEM_UNKNOWN) {
            addCandidate(requirement);
            continue;
        }
        if (consumerOp == nullptr || !IsViewSemanticOpcode(consumerOp->GetOpcode())) {
            continue;
        }
        for (const auto& output : consumerOp->oOperand) {
            addCandidate(InferOriginalFromOutputRequirements(output, visitedTensors));
        }
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
}

MemoryType AssignMemoryType::InferRequirementFromInputOriginals(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    return InferRequirementFromInputOriginals(tensor, visitedTensors);
}

MemoryType AssignMemoryType::InferRequirementFromInputOriginals(
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
    addCandidate(tensor->GetMemoryTypeOriginal());
    for (const auto& producerOp : tensor->GetProducers()) {
        if (producerOp == nullptr || !IsViewSemanticOpcode(producerOp->GetOpcode())) {
            continue;
        }
        for (const auto& input : producerOp->iOperand) {
            addCandidate(InferRequirementFromInputOriginals(input, visitedTensors));
        }
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
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
    return HasDifferentConsumerRequirement(tensor, targetType);
}

bool AssignMemoryType::HasDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const
{
    if (tensor == nullptr) {
        return false;
    }
    auto hasTerminalDifferentMoveConsumer = [targetType](const LogicalTensorPtr& branchTensor) {
        if (targetType != MemoryType::MEM_UB || branchTensor == nullptr) {
            return false;
        }
        return std::any_of(
            branchTensor->GetConsumers().begin(), branchTensor->GetConsumers().end(),
            [targetType](Operation* consumerOp) {
                if (consumerOp == nullptr ||
                    (consumerOp->GetOpcode() != Opcode::OP_CONTRACT &&
                     consumerOp->GetOpcode() != Opcode::OP_ASSEMBLE) ||
                    consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                    return false;
                }
                auto output = consumerOp->oOperand.front();
                bool needCopy = false;
                return output->GetConsumers().empty() &&
                       ((consumerOp->GetAttr<bool>("NeedCopy", needCopy) && needCopy) ||
                        MemoryPathUtils::IsDifferentKnownRequirement(output->GetMemoryTypeOriginal(), targetType));
            });
    };
    auto requirements = inserter.GetConsumerRequirements(tensor);
    bool hasDifferentRequirement = std::any_of(
        requirements.begin(), requirements.end(),
        [this, targetType, &hasTerminalDifferentMoveConsumer](const auto& item) {
            auto resolveOutputRequirement = [this](const LogicalTensorPtr& output) {
                return InferUniqueRequirementThroughViewConsumers(output);
            };
            Operation* consumerOp = item.first;
            if (consumerOp != nullptr &&
                (consumerOp->GetOpcode() == Opcode::OP_VIEW || consumerOp->GetOpcode() == Opcode::OP_SLICE) &&
                !consumerOp->oOperand.empty() && hasTerminalDifferentMoveConsumer(consumerOp->oOperand.front())) {
                return true;
            }
            MemoryType requirement = MemoryPathUtils::ResolveEffectiveConsumerRequirement(
                consumerOp, item.second, targetType, resolveOutputRequirement);
            if (MemoryPathUtils::IsDifferentKnownRequirement(requirement, targetType)) {
                return true;
            }
            if (consumerOp == nullptr ||
                (consumerOp->GetOpcode() != Opcode::OP_CONTRACT && consumerOp->GetOpcode() != Opcode::OP_ASSEMBLE) ||
                consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                return false;
            }
            auto output = consumerOp->oOperand.front();
            auto outputRequirement = output->GetMemoryTypeOriginal();
            return output->GetConsumers().empty() &&
                   MemoryPathUtils::IsDifferentKnownRequirement(outputRequirement, targetType);
        });
    if (hasDifferentRequirement) {
        return true;
    }
    return hasTerminalDifferentMoveConsumer(tensor);
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

Status AssignMemoryType::InferAssembleMemoryType(Operation& operation,
                                                 std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_ASSEMBLE, "Infer OP_ASSEMBLE memory type",
                                                      input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    if (inserter.GetRequirementOrUnknown(input, operation) != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    if (!inferredAssembleOutputs.insert(output).second) {
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
    auto output = operation.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal == MemoryType::MEM_UNKNOWN) {
        outputOriginal = InferOriginalFromOutputRequirements(output);
    }
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, outputOriginal, "InferAssembleOutputRequirement"));
        return SetParallelAssembleInputRequirements(output, outputOriginal, "InferAssembleOutputRequirement");
    }
    MemoryType inputRequirement = InferParallelAssembleInputRequirement(output);
    if (inputRequirement == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, inputRequirement, "InferAssembleInputRequirement"));
    return SetParallelAssembleInputRequirements(output, inputRequirement, "InferAssembleInputRequirement");
}

MemoryType AssignMemoryType::InferParallelAssembleInputRequirement(const LogicalTensorPtr& output) const
{
    if (output == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    for (auto& producerOp : output->GetProducers()) {
        if (!IsAssembleProducer(producerOp)) {
            continue;
        }
        auto input = producerOp->iOperand.front();
        MemoryType candidate = inserter.GetRequirementOrUnknown(input, *producerOp);
        if (candidate == MemoryType::MEM_UNKNOWN) {
            candidate = InferRequirementFromInputOriginals(input);
        }
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    }
    if (candidates.empty()) {
        return MemoryType::MEM_UNKNOWN;
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_DEVICE_DDR;
}

Status AssignMemoryType::SetParallelAssembleInputRequirements(const LogicalTensorPtr& output, MemoryType memoryType,
                                                              const std::string& reason)
{
    if (output == nullptr || memoryType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    for (auto& producerOp : output->GetProducers()) {
        if (!IsAssembleProducer(producerOp)) {
            continue;
        }
        auto input = producerOp->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because input tensor is null.",
                              producerOp->GetOpMagic());
            return FAILED;
        }
        ForceSetRequirement(input, *producerOp, memoryType, reason);
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (assembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because assemble attr is null.",
                              producerOp->GetOpMagic());
            return FAILED;
        }
        assembleOpAttribute->SetFromType(memoryType);
    }
    return SUCCESS;
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

Status AssignMemoryType::IsAssembleToOffsetAligned(Operation& operation, const LogicalTensorPtr& output, bool& aligned)
{
    aligned = false;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr || output == nullptr || output->GetRawTensor() == nullptr) {
        return FAILED;
    }
    const auto& rawShape = output->GetRawTensor()->rawshape;
    const auto& toOffset = assembleOpAttribute->GetToOffset();
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(output->Datatype()));
    if (tensorBytes <= 0 || ASSEMBLE_ALIGN_BYTES % tensorBytes != 0 || rawShape.empty()) {
        aligned = true;
        return SUCCESS;
    }
    // Dynamic shape (-1) cannot be padded or aligned at compile time.
    if (std::find(rawShape.begin(), rawShape.end(), -1) != rawShape.end()) {
        return SUCCESS;
    }
    int64_t alignElements = ASSEMBLE_ALIGN_BYTES / tensorBytes;
    size_t lastIdx = rawShape.size() - 1;
    auto padUp = [](int64_t dim, int64_t base) { return (dim + base - 1) / base * base; };
    auto isAlignedAfterPad = [&](size_t padIdx, bool& padAligned) -> Status {
        padAligned = false;
        if (rawShape[padIdx] <= 0) {
            return SUCCESS;
        }
        Shape paddedShape = rawShape;
        paddedShape[padIdx] = padUp(rawShape[padIdx], alignElements);
        int64_t paddedOffset = 0;
        RETURN_IF_NOT_SUCCESS(CalcLineOffset(paddedShape, toOffset, paddedOffset));
        padAligned = (tensorBytes * paddedOffset) % ASSEMBLE_ALIGN_BYTES == 0;
        return SUCCESS;
    };
    // PadLocalBuffer always pads the tail axis to 32B (non axis-combine mode, or axis-combine
    // mode where tensor is not eligible). This check is mandatory.
    bool tailPadAligned = false;
    RETURN_IF_NOT_SUCCESS(isAlignedAfterPad(lastIdx, tailPadAligned));
    // When tail axis == 1 and there is a second-to-last axis, AxisCombine may pad the
    // second-to-last axis instead (ASSEMBLE is a shapeTransformOp). Since AssignMemoryType
    // cannot determine which padding will apply, both must be aligned to safely avoid DDR
    // fallback.
    if (rawShape[lastIdx] == 1 && rawShape.size() >= 2) {
        bool secondLastPadAligned = false;
        RETURN_IF_NOT_SUCCESS(isAlignedAfterPad(lastIdx - 1, secondLastPadAligned));
        aligned = tailPadAligned && secondLastPadAligned;
        return SUCCESS;
    }
    aligned = tailPadAligned;
    return SUCCESS;
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
    bool kept = false;
    RETURN_IF_NOT_SUCCESS(KeepSplitReshapeUb(operation, input, output, kept));
    if (kept) {
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
    RETURN_IF_NOT_SUCCESS(TryInferViewTypeFromProducerSlice(operation, input, output, targetType, handled));
    if (handled) {
        return SUCCESS;
    }
    return InferViewTypeInput(operation, input, output, targetType);
}

Status AssignMemoryType::TryInferViewTypeFromProducerSlice(Operation& operation, const LogicalTensorPtr& input,
                                                           const LogicalTensorPtr& output, MemoryType targetType,
                                                           bool& handled)
{
    handled = false;
    auto& producers = input->GetProducers();
    if (producers.empty()) {
        return SUCCESS;
    }
    auto producer = *producers.begin();
    if (producer == nullptr || producer->GetOpcode() != Opcode::OP_SLICE) {
        return SUCCESS;
    }
    handled = true;
    if (producer->iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_VIEW_TYPE[%d] memory type failed because producer OP_SLICE[%d] input is empty.",
                          operation.GetOpMagic(), producer->GetOpMagic());
        return FAILED;
    }
    auto sliceInput = producer->iOperand.front();
    MemoryType sliceInputRequirement = inserter.GetRequirementOrUnknown(sliceInput, *producer);
    if (sliceInputRequirement == MemoryType::MEM_UNKNOWN) {
        sliceInputRequirement = sliceInput->GetMemoryTypeOriginal();
    }
    if (targetType != MemoryType::MEM_UNKNOWN && CanUseDirectViewPath(*producer, sliceInputRequirement, targetType)) {
        ForceSetOriginal(input, targetType, "InferViewTypeProducerSlice");
        ForceSetRequirement(input, operation, targetType, "InferViewTypeProducerSlice");
        ForceSetOriginal(output, targetType, "InferViewTypeProducerSlice");
        return SUCCESS;
    }
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewTypeProducerSliceFallback");
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferViewTypeProducerSliceFallback");
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

Status AssignMemoryType::KeepSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input,
                                            const LogicalTensorPtr& output, bool& kept)
{
    kept = false;
    if (input == nullptr || output == nullptr) {
        return SUCCESS;
    }
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    if (producers.empty() || consumers.empty()) {
        return SUCCESS;
    }
    bool allProducersContract = std::all_of(producers.begin(), producers.end(), [](const auto& producer) {
        return producer != nullptr && producer->GetOpcode() == Opcode::OP_CONTRACT;
    });
    bool allConsumersSlice = std::all_of(consumers.begin(), consumers.end(), [](const auto& consumer) {
        return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE;
    });
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_ASSEMBLE);
    int64_t inputDataSize = input->GetDataSize();
    if (allProducersContract && allConsumersSlice && inputDataSize >= 0 &&
        static_cast<size_t>(inputDataSize) <= ubThreshold) {
        bool canKeepProducers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepContractProducersInUb(input, canKeepProducers));
        if (canKeepProducers) {
            ForceSetOriginal(input, MemoryType::MEM_UB, "InferSplitReshapeUb");
            ForceSetRequirement(input, operation, MemoryType::MEM_UB, "InferSplitReshapeUb");
            ForceSetOriginal(output, MemoryType::MEM_UB, "InferSplitReshapeUb");
            for (const auto& consumerOp : output->GetConsumers()) {
                if (consumerOp != nullptr) {
                    ForceSetRequirement(output, *consumerOp, MemoryType::MEM_UB, "InferSplitReshapeUb");
                }
            }
            kept = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::CanKeepContractProducersInUb(const LogicalTensorPtr& tensor, bool& canKeep)
{
    canKeep = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* producerOp : tensor->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_CONTRACT || producerOp->iOperand.empty()) {
            return SUCCESS;
        }
        MemoryType fromType = GetAssembleInputType(*producerOp);
        constexpr MemoryType targetType = MemoryType::MEM_UB;
        bool checkOffsetAlignment = !IsAdvancedMemoryPath(fromType, targetType);
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(*producerOp, tensor, aligned));
        if ((checkOffsetAlignment && !aligned) || !CanUseDirectAssemblePath(*producerOp, fromType, targetType)) {
            return SUCCESS;
        }
    }
    canKeep = true;
    return SUCCESS;
}

Status AssignMemoryType::IsSliceFromOffsetAligned(Operation& sliceOp, const LogicalTensorPtr& input, bool& aligned)
{
    aligned = false;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    if (viewOpAttribute == nullptr || input == nullptr) {
        return SUCCESS;
    }
    int64_t lineOffset = 0;
    RETURN_IF_NOT_SUCCESS(
        CalcLineOffset(input->GetRawTensor()->rawshape, viewOpAttribute->GetFromOffset(), lineOffset));
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(input->Datatype()));
    aligned = (tensorBytes * lineOffset) % ASSEMBLE_ALIGN_BYTES == 0;
    return SUCCESS;
}

Status AssignMemoryType::CanKeepSliceConsumersInUb(const LogicalTensorPtr& tensor, bool& canKeep)
{
    canKeep = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* consumerOp : tensor->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_SLICE || consumerOp->oOperand.empty() ||
            consumerOp->oOperand.front() == nullptr) {
            return SUCCESS;
        }
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsSliceFromOffsetAligned(*consumerOp, tensor, aligned));
        if (!aligned) {
            return SUCCESS;
        }
    }
    canKeep = true;
    return SUCCESS;
}

Status AssignMemoryType::HasNonZeroSliceFromOffset(const LogicalTensorPtr& tensor, bool& hasNonZero)
{
    hasNonZero = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* consumerOp : tensor->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewAttr == nullptr) {
            continue;
        }
        int64_t lineOffset = 0;
        RETURN_IF_NOT_SUCCESS(CalcLineOffset(tensor->GetRawTensor()->rawshape, viewAttr->GetFromOffset(), lineOffset));
        if (lineOffset > 0) {
            hasNonZero = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
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
    const size_t ubStrictThreshold = static_cast<size_t>(
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_ASSEMBLE);
    const size_t ubNormalThreshold = static_cast<size_t>(
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_NORMAL);
    const size_t l1Threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) *
                                                   L1_THRESHOLD);
    APASS_LOG_INFO_F(Elements::Function, "Memory threshold: UB strict %zu, UB normal %zu, L1 %zu.", ubStrictThreshold,
                     ubNormalThreshold, l1Threshold);
    for (auto& op : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(op));
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallback(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_SLICE && opcode != Opcode::OP_CONTRACT) {
        return SUCCESS;
    }
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Apply oversized fallback for %s[%d] failed because operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.front();
    if (input == nullptr || output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because operand tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }

    if (opcode == Opcode::OP_CONTRACT) {
        MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
        if (!IsOversizedLocalBuffer(input, inputRequirement, true, true)) {
            return SUCCESS;
        }
        ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedContractInputFallback");
        auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
        if (assembleAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Apply oversized fallback for OP_CONTRACT[%d] failed because assemble attr is null.",
                              operation.GetOpMagic());
            return FAILED;
        }
        assembleAttr->SetFromType(MemoryType::MEM_DEVICE_DDR);
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Force OP_CONTRACT[%d] input tensor[%d] requirement to DDR by size limit.",
                          operation.GetOpMagic(), input->GetMagic());
        return SUCCESS;
    }

    if (IsOversizedLocalBuffer(output, output->GetMemoryTypeOriginal(), false, true)) {
        ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedSliceOutputFallback");
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
        if (viewAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Apply oversized fallback for OP_SLICE[%d] failed because view attr is null.",
                              operation.GetOpMagic());
            return FAILED;
        }
        viewAttr->SetToType(MemoryType::MEM_DEVICE_DDR);
        APASS_LOG_DEBUG_F(Elements::Operation, "Force OP_SLICE[%d] output tensor[%d] to DDR by size limit.",
                          operation.GetOpMagic(), output->GetMagic());
    }
    return DowngradeOversizedSliceInputRequirement(operation);
}

bool AssignMemoryType::IsOversizedLocalBuffer(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                              bool useStrictUbLimit, bool allowL1Fallback) const
{
    if (memoryType == MemoryType::MEM_UB) {
        double ubLimitRatio = useStrictUbLimit ? UB_THRESHOLD_ASSEMBLE : UB_THRESHOLD_NORMAL;
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

Status AssignMemoryType::DowngradeOversizedSliceInputRequirement(Operation& operation)
{
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for OP_SLICE[%d] failed because of invalid input operand.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    MemoryType inputType = inserter.GetRequirementOrUnknown(input, operation);
    if (!IsOversizedLocalBuffer(input, inputType, false, true)) {
        return SUCCESS;
    }
    ForceSetRequirement(input, operation, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedSliceInputFallback");
    APASS_LOG_DEBUG_F(Elements::Operation, "Force OP_SLICE[%d] input tensor[%d] requirement to DDR by size limit.",
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

Status AssignMemoryType::ApplyPlatformPathUpgradeRules(Function& function)
{
    RETURN_IF_NOT_SUCCESS(ProcessDdrMultiReshape(function));
    RETURN_IF_NOT_SUCCESS(ProcessL1DdrL1(function));
    RETURN_IF_NOT_SUCCESS(ProcessL0C2L1SmallToLarge(function));
    RETURN_IF_NOT_SUCCESS(ProcessL0C2L1LargeToSmall(function));
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        RETURN_IF_NOT_SUCCESS(ProcessL0C2UBSmallToLarge(function));
        RETURN_IF_NOT_SUCCESS(ProcessL0C2UBLargeToSmall(function));
        RETURN_IF_NOT_SUCCESS(ProcessUB2L1SmallToLarge(function));
        RETURN_IF_NOT_SUCCESS(ProcessUB2L1LargeToSmall(function));
    }
    RETURN_IF_NOT_SUCCESS(ProcessUB2UBContractSlice(function));
    return SUCCESS;
}

Status AssignMemoryType::ProcessDdrMultiReshape(Function& function)
{
    constexpr MemoryType kUbMemoryType = MemoryType::MEM_UB;
    constexpr const char* kReason = "ProcessDdrMultiReshape";
    for (auto& contract : function.Operations()) {
        if (contract.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        auto contractInput = contract.iOperand.front();
        auto contractOutput = contract.oOperand.front();
        if (contractOutput->GetProducers().size() != 1 || contractInput->GetMemoryTypeOriginal() != kUbMemoryType ||
            contractOutput->GetConsumers().size() != 1) {
            continue;
        }
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(contract, contractOutput, aligned));
        if (!aligned) {
            continue;
        }
        auto* firstReshape = *contractOutput->GetConsumers().begin();
        if (firstReshape->GetOpcode() != Opcode::OP_RESHAPE || firstReshape->iOperand.size() != 1 ||
            firstReshape->oOperand.size() != 1 || firstReshape->iOperand.front() != contractOutput) {
            continue;
        }
        auto firstReshapeOutput = firstReshape->oOperand.front();
        if (firstReshapeOutput->GetConsumers().empty()) {
            continue;
        }

        std::vector<Operation*> branchViews;
        std::vector<Operation*> branchReshapes;
        std::vector<Operation*> branchSlices;
        std::vector<Operation*> directSlices;
        bool matches = true;
        for (auto* branch : firstReshapeOutput->GetConsumers()) {
            if (branch->GetOpcode() == Opcode::OP_SLICE) {
                if (branch->iOperand.size() != 1 || branch->oOperand.size() != 1 ||
                    branch->iOperand.front() != firstReshapeOutput ||
                    branch->oOperand.front()->GetMemoryTypeOriginal() != kUbMemoryType ||
                    std::dynamic_pointer_cast<ViewOpAttribute>(branch->GetOpAttribute()) == nullptr) {
                    matches = false;
                    break;
                }
                directSlices.push_back(branch);
                continue;
            }
            if (branch->GetOpcode() != Opcode::OP_VIEW || branch->iOperand.size() != 1 ||
                branch->oOperand.size() != 1 || branch->iOperand.front() != firstReshapeOutput ||
                std::dynamic_pointer_cast<ViewOpAttribute>(branch->GetOpAttribute()) == nullptr) {
                matches = false;
                break;
            }
            auto viewOutput = branch->oOperand.front();
            if (viewOutput->GetConsumers().empty()) {
                matches = false;
                break;
            }
            branchViews.push_back(branch);
            for (auto* reshape : viewOutput->GetConsumers()) {
                if (reshape->GetOpcode() != Opcode::OP_RESHAPE || reshape->iOperand.size() != 1 ||
                    reshape->oOperand.size() != 1 || reshape->iOperand.front() != viewOutput) {
                    matches = false;
                    break;
                }
                auto reshapeOutput = reshape->oOperand.front();
                if (reshapeOutput->GetConsumers().empty()) {
                    matches = false;
                    break;
                }
                branchReshapes.push_back(reshape);
                for (auto* slice : reshapeOutput->GetConsumers()) {
                    if (slice->GetOpcode() != Opcode::OP_SLICE || slice->iOperand.size() != 1 ||
                        slice->oOperand.size() != 1 || slice->iOperand.front() != reshapeOutput ||
                        slice->oOperand.front()->GetMemoryTypeOriginal() != kUbMemoryType ||
                        std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute()) == nullptr) {
                        matches = false;
                        break;
                    }
                    branchSlices.push_back(slice);
                }
                if (!matches) {
                    break;
                }
            }
            if (!matches) {
                break;
            }
        }
        if (!matches) {
            continue;
        }

        auto contractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(contract.GetOpAttribute());
        if (contractAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessDdrMultiReshape failed because contract attr is null.");
            return FAILED;
        }
        ForceSetRequirement(contractInput, contract, kUbMemoryType, kReason);
        contractAttr->SetFromType(kUbMemoryType);
        ForceSetOriginal(contractOutput, kUbMemoryType, kReason);
        ForceSetRequirement(contractOutput, *firstReshape, kUbMemoryType, kReason);
        ForceSetOriginal(firstReshapeOutput, kUbMemoryType, kReason);

        for (auto* view : branchViews) {
            auto viewOutput = view->oOperand.front();
            ForceSetRequirement(firstReshapeOutput, *view, kUbMemoryType, kReason);
            ForceSetOriginal(viewOutput, kUbMemoryType, kReason);
            std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute())->SetToType(kUbMemoryType);
        }
        for (auto* reshape : branchReshapes) {
            auto reshapeInput = reshape->iOperand.front();
            ForceSetRequirement(reshapeInput, *reshape, kUbMemoryType, kReason);
            ForceSetOriginal(reshape->oOperand.front(), kUbMemoryType, kReason);
        }
        for (auto* slice : branchSlices) {
            auto sliceInput = slice->iOperand.front();
            ForceSetRequirement(sliceInput, *slice, kUbMemoryType, kReason);
            ForceSetOriginal(slice->oOperand.front(), kUbMemoryType, kReason);
            std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute())->SetToType(kUbMemoryType);
        }
        for (auto* slice : directSlices) {
            ForceSetRequirement(firstReshapeOutput, *slice, kUbMemoryType, kReason);
            ForceSetOriginal(slice->oOperand.front(), kUbMemoryType, kReason);
            std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute())->SetToType(kUbMemoryType);
        }
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Upgrade contract[%d] reshape branches from DDR to UB for UB slice outputs.",
                          contract.GetOpMagic());
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL1DdrL1(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visitedOutputs;
    for (auto& contract : function.Operations()) {
        if (contract.GetOpcode() != Opcode::OP_CONTRACT || contract.oOperand.empty()) {
            continue;
        }
        auto middle = contract.oOperand.front();
        if (middle == nullptr || !visitedOutputs.insert(middle).second ||
            (middle->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN &&
             middle->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) ||
            IsOversizedLocalBuffer(middle, MemoryType::MEM_L1, false, true) || !HasOnlyContractProducers(middle) ||
            !HasOnlySliceConsumers(middle)) {
            continue;
        }

        bool allContractInputsFromL1 = true;
        for (auto* producer : middle->GetProducers()) {
            auto input = producer->iOperand.front();
            if (input == nullptr) {
                allContractInputsFromL1 = false;
                break;
            }
            MemoryType requirement = inserter.GetRequirementOrUnknown(input, *producer);
            if (requirement != MemoryType::MEM_L1 &&
                (requirement != MemoryType::MEM_UNKNOWN || input->GetMemoryTypeOriginal() != MemoryType::MEM_L1)) {
                allContractInputsFromL1 = false;
                break;
            }
        }
        if (!allContractInputsFromL1) {
            continue;
        }

        bool allSliceConsumersToL1 = true;
        for (auto* consumer : middle->GetConsumers()) {
            auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
            if (viewAttr == nullptr || viewAttr->GetTo() != MemoryType::MEM_L1 ||
                inserter.GetRequirementOrUnknown(middle, *consumer) != MemoryType::MEM_DEVICE_DDR) {
                allSliceConsumersToL1 = false;
                break;
            }
        }
        if (!allSliceConsumersToL1) {
            continue;
        }

        ForceSetOriginal(middle, MemoryType::MEM_L1, "ProcessL1DdrL1");
        for (auto* consumer : middle->GetConsumers()) {
            ForceSetRequirement(middle, *consumer, MemoryType::MEM_L1, "ProcessL1DdrL1");
        }
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Upgrade tensor[%d] from L1 -> DDR -> L1 path to L1 for contract/slice layout ops.",
                          middle->GetMagic());
    }
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
    if (operation.GetOpcode() != Opcode::OP_VIEW && operation.GetOpcode() != Opcode::OP_SLICE) {
        return SUCCESS;
    }
    if (operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because output operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because view attr is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto output = operation.oOperand.front();
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because output tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
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
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE && operation.GetOpcode() != Opcode::OP_CONTRACT) {
        return SUCCESS;
    }
    if (operation.iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because input operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because assemble attr is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    if (input == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because input tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
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

Status AssignMemoryType::FallbackSameMemoryMoveOps(Function& function)
{
    for (auto& operation : function.Operations()) {
        auto opcode = operation.GetOpcode();
        if (opcode != Opcode::OP_SLICE && opcode != Opcode::OP_CONTRACT) {
            continue;
        }
        if (operation.iOperand.empty() || operation.oOperand.empty() || operation.iOperand.front() == nullptr ||
            operation.oOperand.front() == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Fallback same-memory move op %s[%d] failed because operand is invalid.",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
            return FAILED;
        }
        MemoryType inputType = operation.iOperand.front()->GetMemoryTypeOriginal();
        MemoryType outputType = operation.oOperand.front()->GetMemoryTypeOriginal();
        if (inputType == MemoryType::MEM_UNKNOWN || inputType != outputType) {
            continue;
        }
        if (opcode == Opcode::OP_SLICE) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
            if (viewOpAttribute != nullptr) {
                viewOpAttribute->SetToType(outputType);
            }
            operation.SetOpCode(Opcode::OP_VIEW);
        } else {
            auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
            if (assembleOpAttribute != nullptr) {
                assembleOpAttribute->SetFromType(inputType);
            }
            operation.SetOpCode(Opcode::OP_ASSEMBLE);
        }
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Fallback same-memory move op %s[%d] to %s because input and output are both %s.",
                          opcode == Opcode::OP_SLICE ? "OP_SLICE" : "OP_CONTRACT", operation.GetOpMagic(),
                          operation.GetOpcodeStr().c_str(), BriefMemoryTypeToString(inputType).c_str());
    }
    return SUCCESS;
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

Status AssignMemoryType::CalcLineOffset(const Shape& shape, const Offset& offset, int64_t& lineOffset) const
{
    if (shape.size() != offset.size() || shape.empty()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "CalcLineOffset failed because shape size %zu != offset size %zu or shape is empty.",
                          shape.size(), offset.size());
        return FAILED;
    }
    lineOffset = 0;
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        lineOffset += offset[i - 1] * stride;
        stride *= shape[i - 1];
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2L1SmallToLarge(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleSliceContractPath(op, MemoryType::MEM_L0C, MemoryType::MEM_L1,
                                                                "ProcessL0C2L1SmallToLarge", false, false, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessUB2UBContractSlice(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT || op.oOperand.empty() || op.oOperand.front() == nullptr) {
            continue;
        }
        auto middle = op.oOperand.front();
        if (!FitsAssembleOutputMemoryLimit(middle, MemoryType::MEM_UB)) {
            continue;
        }
        bool canKeepProducers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepContractProducersInUb(middle, canKeepProducers));
        if (!canKeepProducers) {
            continue;
        }
        bool canKeepConsumers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepSliceConsumersInUb(middle, canKeepConsumers));
        if (!canKeepConsumers) {
            continue;
        }
        bool hasNonZero = false;
        RETURN_IF_NOT_SUCCESS(HasNonZeroSliceFromOffset(middle, hasNonZero));
        if (hasNonZero) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                                                "ProcessUB2UBContractSlice", false, false, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2L1LargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_L0C, MemoryType::MEM_L1,
                                                                "ProcessL0C2L1LargeToSmall", false, false, false));
    }
    return SUCCESS;
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

bool AssignMemoryType::CheckConsumerSliceShapeMultiple(const LogicalTensorPtr& output, const LogicalTensorPtr& input)
{
    for (auto& consumerOp : output->GetConsumers()) {
        if (consumerOp->GetOpcode() == Opcode::OP_SLICE &&
            !IsDimMultiple(consumerOp->GetOOperands().front()->GetShape(), input->GetShape())) {
            return false;
        }
    }
    return true;
}

Status AssignMemoryType::ProcessL0C2UBSmallToLarge(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleSliceContractPath(op, MemoryType::MEM_L0C, MemoryType::MEM_UB,
                                                                "ProcessL0C2UBSmallToLarge", true, true, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2UBLargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_L0C, MemoryType::MEM_UB,
                                                                "ProcessL0C2UBLargeToSmall", true, true, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessUB2L1SmallToLarge(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleSliceContractPath(op, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                                                "ProcessUB2L1SmallToLarge", true, false, true));
    }
    return SUCCESS;
}

bool AssignMemoryType::ShouldSkipUB2L1SmallToLarge(const LogicalTensorPtr& iOperand,
                                                   const LogicalTensorPtr& oOperand) const
{
    const size_t UB_LIMIT = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                UB_THRESHOLD_NORMAL);
    if (CalcNZTensorSize(iOperand) > UB_LIMIT) {
        return true;
    }
    // 检查 consumer slice 是否有 copy_in_mode=0 属性
    for (auto& consumerOp : oOperand->GetConsumers()) {
        if (consumerOp->GetOpcode() == Opcode::OP_SLICE) {
            int64_t copyInModeValue = 0;
            if (consumerOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0) {
                return true;
            }
        }
    }
    return !CheckInnerAxisC0Size(iOperand, oOperand);
}

Status AssignMemoryType::ProcessUB2L1LargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                                                "ProcessUB2L1LargeToSmall", true, false, true));
    }
    return SUCCESS;
}

bool AssignMemoryType::CanUseMiddleTensorForUpgrade(const LogicalTensorPtr& middle, MemoryType targetType) const
{
    if (middle == nullptr || targetType == MemoryType::MEM_UNKNOWN) {
        return false;
    }
    MemoryType currentType = middle->GetMemoryTypeOriginal();
    return currentType == MemoryType::MEM_UNKNOWN || currentType == MemoryType::MEM_DEVICE_DDR ||
           currentType == targetType;
}

bool AssignMemoryType::HasOnlyContractProducers(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetProducers().empty()) {
        return false;
    }
    return std::all_of(tensor->GetProducers().begin(), tensor->GetProducers().end(), [](Operation* producer) {
        return producer != nullptr && producer->GetOpcode() == Opcode::OP_CONTRACT && !producer->iOperand.empty() &&
               !producer->oOperand.empty() &&
               std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute()) != nullptr;
    });
}

bool AssignMemoryType::HasOnlySliceConsumers(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetConsumers().empty()) {
        return false;
    }
    return std::all_of(tensor->GetConsumers().begin(), tensor->GetConsumers().end(), [](Operation* consumer) {
        return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE && !consumer->iOperand.empty() &&
               !consumer->oOperand.empty() &&
               std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute()) != nullptr;
    });
}

bool AssignMemoryType::IsSliceOutputTarget(Operation& sliceOp, MemoryType targetType) const
{
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.oOperand.empty() || sliceOp.oOperand.front() == nullptr) {
        return false;
    }
    MemoryType outputOriginal = sliceOp.oOperand.front()->GetMemoryTypeOriginal();
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        return outputOriginal == targetType;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    return viewOpAttribute != nullptr && viewOpAttribute->GetTo() == targetType;
}

Status AssignMemoryType::EnsureSliceOutputTarget(Operation& sliceOp, MemoryType targetType, const std::string& reason)
{
    if (sliceOp.oOperand.empty() || sliceOp.oOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because slice output is invalid.", reason.c_str(),
                          sliceOp.GetOpMagic());
        return FAILED;
    }
    auto output = sliceOp.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal != MemoryType::MEM_UNKNOWN && outputOriginal != targetType) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Upgrade %s[%d] failed because slice output tensor[%d] original %s conflicts with target %s.",
                          reason.c_str(), sliceOp.GetOpMagic(), output->GetMagic(),
                          BriefMemoryTypeToString(outputOriginal).c_str(), BriefMemoryTypeToString(targetType).c_str());
        return FAILED;
    }
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, targetType, reason));
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because slice attr is null.", reason.c_str(),
                          sliceOp.GetOpMagic());
        return FAILED;
    }
    viewOpAttribute->SetToType(targetType);
    return SUCCESS;
}

Status AssignMemoryType::ApplySingleSliceContractUpgrade(Operation& sliceOp, MemoryType sourceType,
                                                         MemoryType targetType, const std::string& reason)
{
    auto middle = sliceOp.iOperand.front();
    ForceSetOriginal(middle, targetType, reason);
    for (auto* consumer : middle->GetConsumers()) {
        RETURN_IF_NOT_SUCCESS(EnsureSliceOutputTarget(*consumer, targetType, reason));
        ForceSetRequirement(middle, *consumer, targetType, reason);
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract input is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        ForceSetRequirement(input, *producer, sourceType, reason);
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (assembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        assembleOpAttribute->SetFromType(sourceType);
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Upgrade middle tensor[%d] to %s for %s contract-side special path.",
                      middle->GetMagic(), BriefMemoryTypeToString(targetType).c_str(), reason.c_str());
    return SUCCESS;
}

Status AssignMemoryType::ApplySingleContractSliceUpgrade(Operation& contractOp, MemoryType sourceType,
                                                         MemoryType targetType, const std::string& reason)
{
    auto middle = contractOp.oOperand.front();
    ForceSetOriginal(middle, sourceType, reason);
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(contractOp.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.", reason.c_str(),
                          contractOp.GetOpMagic());
        return FAILED;
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract input is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        ForceSetRequirement(input, *producer, sourceType, reason);
        auto producerAssembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (producerAssembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        producerAssembleOpAttribute->SetFromType(sourceType);
    }
    for (auto* consumer : middle->GetConsumers()) {
        RETURN_IF_NOT_SUCCESS(EnsureSliceOutputTarget(*consumer, targetType, reason));
        ForceSetRequirement(middle, *consumer, sourceType, reason);
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Upgrade middle tensor[%d] to %s for %s slice-side special path.",
                      middle->GetMagic(), BriefMemoryTypeToString(sourceType).c_str(), reason.c_str());
    return SUCCESS;
}

bool AssignMemoryType::CanUseL0C2L1UpgradePath(Operation& operation)
{
    if (operation.iOperand.empty() || operation.oOperand.empty() || operation.iOperand.front() == nullptr ||
        operation.oOperand.front() == nullptr) {
        return false;
    }
    return inserter.FitL0C2L1(operation);
}

Status AssignMemoryType::TryUpgradeSingleSliceContractPath(Operation& sliceOp, MemoryType sourceType,
                                                           MemoryType targetType, const std::string& reason,
                                                           bool requireMatrixShape, bool checkUbTileShape,
                                                           bool checkUb2L1Constraints)
{
    constexpr size_t kMatrixShapeDimCount = 2;
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.iOperand.empty() || sliceOp.oOperand.empty()) {
        return SUCCESS;
    }
    auto middle = sliceOp.iOperand.front();
    auto target = sliceOp.oOperand.front();
    if (middle == nullptr || target == nullptr || !CanUseMiddleTensorForUpgrade(middle, targetType) ||
        !HasOnlyContractProducers(middle) || !HasOnlySliceConsumers(middle) ||
        !IsSliceOutputTarget(sliceOp, targetType)) {
        return SUCCESS;
    }
    for (auto* consumer : middle->GetConsumers()) {
        if (consumer != nullptr && !IsSliceOutputTarget(*consumer, targetType)) {
            return SUCCESS;
        }
    }
    if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 && middle->GetConsumers().size() != 1) {
        return SUCCESS;
    }
    if (requireMatrixShape &&
        (middle->GetShape().size() != kMatrixShapeDimCount || target->GetShape().size() != kMatrixShapeDimCount)) {
        return SUCCESS;
    }
    if (checkUbTileShape && (!CheckUBTileShape(middle) || !FitsAssembleOutputMemoryLimit(middle, targetType))) {
        return SUCCESS;
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr || input->GetMemoryTypeOriginal() != sourceType) {
            return SUCCESS;
        }
        if (requireMatrixShape && input->GetShape().size() != kMatrixShapeDimCount) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*producer)) {
            return SUCCESS;
        }
        for (auto* consumer : middle->GetConsumers()) {
            if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
                !CanUseL0C2L1UpgradePath(*consumer)) {
                return SUCCESS;
            }
            if (targetType != MemoryType::MEM_UB) {
                continue;
            }
            auto output = consumer->oOperand.front();
            if (HasDifferentConsumerRequirement(output, targetType)) {
                return SUCCESS;
            }
        }
        if (HasParallelDifferentConsumerRequirement(input, targetType) ||
            !IsDimMultiple(middle->GetShape(), input->GetShape()) || !CheckConsumerSliceShapeMultiple(middle, input) ||
            !IsDimMultiple(target->GetShape(), input->GetShape())) {
            return SUCCESS;
        }
        if (checkUb2L1Constraints && ShouldSkipUB2L1SmallToLarge(input, middle)) {
            return SUCCESS;
        }
    }
    return ApplySingleSliceContractUpgrade(sliceOp, sourceType, targetType, reason);
}

Status AssignMemoryType::TryUpgradeSingleContractSlicePath(Operation& contractOp, MemoryType sourceType,
                                                           MemoryType targetType, const std::string& reason,
                                                           bool requireMatrixShape, bool checkUbTileShape,
                                                           bool checkUb2L1Constraints)
{
    constexpr size_t kMatrixShapeDimCount = 2;
    if (contractOp.GetOpcode() != Opcode::OP_CONTRACT || contractOp.iOperand.empty() || contractOp.oOperand.empty()) {
        return SUCCESS;
    }
    auto input = contractOp.iOperand.front();
    auto middle = contractOp.oOperand.front();
    if (input == nullptr || middle == nullptr || input->GetMemoryTypeOriginal() != sourceType ||
        !CanUseMiddleTensorForUpgrade(middle, sourceType) || !HasOnlyContractProducers(middle) ||
        !HasOnlySliceConsumers(middle)) {
        return SUCCESS;
    }
    if (requireMatrixShape && input->GetShape().size() != kMatrixShapeDimCount) {
        return SUCCESS;
    }
    if (HasParallelDifferentConsumerRequirement(input, targetType)) {
        return SUCCESS;
    }
    if (HasParallelDifferentConsumerRequirement(middle, targetType)) {
        return SUCCESS;
    }
    for (auto* producer : middle->GetProducers()) {
        auto producerInput = producer->iOperand.front();
        if (producerInput == nullptr || producerInput->GetMemoryTypeOriginal() != sourceType) {
            return SUCCESS;
        }
        if (requireMatrixShape && producerInput->GetShape().size() != kMatrixShapeDimCount) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*producer)) {
            return SUCCESS;
        }
        if (HasParallelDifferentConsumerRequirement(producerInput, targetType)) {
            return SUCCESS;
        }
    }
    for (auto* consumer : middle->GetConsumers()) {
        if (!IsSliceOutputTarget(*consumer, targetType)) {
            return SUCCESS;
        }
        auto output = consumer->oOperand.front();
        if (requireMatrixShape && output->GetShape().size() != kMatrixShapeDimCount) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*consumer)) {
            return SUCCESS;
        }
        if (!IsDimMultiple(input->GetShape(), output->GetShape()) &&
            !IsDimMultiple(output->GetShape(), input->GetShape())) {
            return SUCCESS;
        }
        if (checkUbTileShape && !CheckUBTileShape(output)) {
            return SUCCESS;
        }
        if (checkUb2L1Constraints) {
            const size_t ubLimit = static_cast<size_t>(
                Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_NORMAL);
            int64_t copyInModeValue = 0;
            if (CalcNZTensorSize(input) > ubLimit ||
                (consumer->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0)) {
                return SUCCESS;
            }
        }
    }
    return ApplySingleContractSliceUpgrade(contractOp, sourceType, targetType, reason);
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

bool AssignMemoryType::IsDimMultiple(const Shape& shape1, const Shape& shape2)
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

Status AssignMemoryType::RunOnFunctionLegacy(Function& function)
{
    legacy::AssignMemoryType legacyAssignMemoryType;
    return legacyAssignMemoryType.RunLegacy(function);
}

Status AssignMemoryType::FixViewAssembleSemanticMismatch(Function& function)
{
    for (auto& op : function.Operations(false)) {
        if (op.iOperand.empty() || op.oOperand.empty()) {
            continue;
        }
        auto input = op.iOperand.front();
        auto output = op.oOperand.front();
        if (input == nullptr || output == nullptr) {
            continue;
        }
        MemoryType inputOriginal = input->GetMemoryTypeOriginal();
        MemoryType outputOriginal = output->GetMemoryTypeOriginal();
        if (inputOriginal == MemoryType::MEM_UNKNOWN || outputOriginal == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            // VIEW 表达搬入语义；当 output==DDR 且 input!=DDR 时语义违反，
            // 需要在 VIEW 之前插入 ASSEMBLE(local→DDR)。
            // 给 input 增加 DDR requirement（指向 VIEW 自身），制造 local vs DDR 冲突。
            if (outputOriginal == MemoryType::MEM_DEVICE_DDR && inputOriginal != MemoryType::MEM_DEVICE_DDR) {
                inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR, "FixViewSemanticMismatch");
                APASS_LOG_INFO_F(
                    Elements::Operation,
                    "VIEW[%d] output=DDR input=%s, add DDR requirement on input[%d] to trigger ASSEMBLE insertion.",
                    op.GetOpMagic(), BriefMemoryTypeToString(inputOriginal).c_str(), input->GetMagic());
            }
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            // ASSEMBLE 表达搬出语义；当 input==DDR 且 output!=DDR 时语义违反，
            // 需要在 ASSEMBLE 之前插入 VIEW(DDR→local)。
            // 给 input 增加 output memType requirement（指向 ASSEMBLE 自身），制造 DDR vs local 冲突。
            if (inputOriginal == MemoryType::MEM_DEVICE_DDR && outputOriginal != MemoryType::MEM_DEVICE_DDR) {
                inserter.UpdateTensorTobeMap(input, op, outputOriginal, "FixAssembleSemanticMismatch");
                APASS_LOG_INFO_F(
                    Elements::Operation,
                    "ASSEMBLE[%d] input=DDR output=%s, add %s requirement on input[%d] to trigger VIEW insertion.",
                    op.GetOpMagic(), BriefMemoryTypeToString(outputOriginal).c_str(),
                    BriefMemoryTypeToString(outputOriginal).c_str(), input->GetMagic());
            }
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
